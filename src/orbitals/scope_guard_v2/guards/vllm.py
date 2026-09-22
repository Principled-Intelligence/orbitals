from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Literal

import aiohttp
import pydantic

if TYPE_CHECKING:
    import transformers  # noqa: F401
    import vllm  # noqa: F401

from ...types import AIServiceDescriptionV2, LLMUsage
from ..modeling import (
    ScopeClass,
    ScopeGuardV2Classification,
    ScopeGuardV2Input,
    ScopeGuardV2Output,
)
from ..prompting import (
    CLASS_PREFIX,
    PREDEFINED_PREFIX,
    SYSTEM_PROMPT,
    build_prompt,
    check_shipped_system_prompt,
    class_first_tokens,
    class_probabilities,
    predefined_candidates,
    response_model_for,
)
from .base import AsyncScopeGuardV2, ScopeGuardV2


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=8)
def _class_tokens(tokenizer_name: str) -> dict[str, str]:
    return class_first_tokens(_get_tokenizer(tokenizer_name))


def _classification(
    top_logprobs: dict[str, float],
    first_tokens: dict[str, str],
    *,
    temperature: float,
    model: str,
    usage: LLMUsage | None,
) -> ScopeGuardV2Classification:
    probabilities = class_probabilities(top_logprobs, first_tokens, temperature)
    best = max(probabilities, key=probabilities.__getitem__)
    return ScopeGuardV2Classification(
        scope_class=ScopeClass(best),
        probabilities=probabilities,
        confidence=probabilities[best],
        temperature=temperature,
        model=model,
        usage=usage,
    )


def _grammar_choices(candidates: list[str]) -> list[str]:
    """The texts the constrained-decoding grammar is built from.

    vLLM's grammar backend cannot compile a `choice` over texts containing line breaks
    (a multi-line address, for instance), so whitespace runs are collapsed to one space
    for the grammar only. The entry returned to the caller is the original, by position.
    """
    return [" ".join(c.split()) for c in candidates]


def _selected(text: str, candidates: list[str]) -> str:
    """Return the description's own entry the constrained completion corresponds to."""
    choices = _grammar_choices(candidates)
    if text in choices:
        return candidates[choices.index(text)]
    stripped = text.strip()
    if stripped in choices:
        return candidates[choices.index(stripped)]
    raise ValueError(f"constrained decoding returned a non-candidate: {text!r}")


# The generated reply is a JSON string value; the model closes it with `"}`.
_RESPONSE_STOP = '"}'


def _generated(text: str) -> str:
    """Unescape a generated JSON string value, tolerating a missing closing quote."""
    body = text.split(_RESPONSE_STOP, 1)[0]
    try:
        value = json.loads(f'"{body}"')
    except json.JSONDecodeError:
        return body.strip()
    return str(value).strip()


def _add_usage(a: LLMUsage | None, b: LLMUsage | None) -> LLMUsage | None:
    if a is None or b is None:
        return a or b
    return LLMUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


def _to_output(
    validated: pydantic.BaseModel, model: str, usage: LLMUsage | None
) -> ScopeGuardV2Output:
    """Lift a per-selection response onto the public output; absent fields are None."""
    data = validated.model_dump()
    return ScopeGuardV2Output(
        evidences=data.get("evidences"),
        reasoning=data.get("reasoning"),
        scope_class=data["scope_class"],
        suggested_response=data.get("suggested_response"),
        model=model,
        usage=usage,
    )


@ScopeGuardV2.register_guard("vllm")
class VLLMScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 30_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        include_default_safety_principles: bool = False,
        count_system_prompt_in_usage: bool = False,
        decision_temperature: float = 1.0,
        decision_logprobs: int = 20,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import vllm

        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        check_shipped_system_prompt(self.model)
        self.llm = vllm.LLM(
            model=self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = _get_tokenizer(self.model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.count_system_prompt_in_usage = count_system_prompt_in_usage
        self.decision_temperature = decision_temperature
        self.decision_logprobs = decision_logprobs

    def _offline_usage(self, output) -> LLMUsage:
        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(self.tokenizer.encode(SYSTEM_PROMPT))
        )
        prompt_tokens = len(output.prompt_token_ids) - system_prompt_tokens
        completion_tokens = len(output.outputs[0].token_ids)
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _classify(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        resolve_predefined: bool = True,
        **kwargs,
    ) -> ScopeGuardV2Classification:
        import vllm

        first_tokens = _class_tokens(self.model)
        prompt = (
            build_prompt(
                self.tokenizer, conversation, ai_service_description, output_fields=["scope_class"]
            )
            + CLASS_PREFIX
        )
        params = vllm.SamplingParams(max_tokens=1, temperature=0.0, logprobs=self.decision_logprobs)
        output = self.llm.generate([prompt], params, use_tqdm=False)[0]
        step = output.outputs[0].logprobs[0]
        top = {lp.decoded_token: lp.logprob for lp in step.values()}
        result = _classification(
            top,
            first_tokens,
            temperature=self.decision_temperature,
            model=self.model,
            usage=self._offline_usage(output),
        )
        if result.scope_class is not ScopeClass.PREDEFINED_ANSWER or not resolve_predefined:
            return result
        candidates = [r for _, r in predefined_candidates(ai_service_description)]
        if len(candidates) == 1:
            result.predefined_response = candidates[0]
            return result
        prompt = (
            build_prompt(
                self.tokenizer,
                conversation,
                ai_service_description,
                output_fields=["scope_class", "suggested_response"],
            )
            + PREDEFINED_PREFIX
        )
        if candidates:
            params = vllm.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=0.0,
                structured_outputs=vllm.sampling_params.StructuredOutputsParams(
                    choice=_grammar_choices(candidates)
                ),
            )
        else:
            params = vllm.SamplingParams(
                max_tokens=self.max_tokens, temperature=self.temperature, stop=[_RESPONSE_STOP]
            )
        output = self.llm.generate([prompt], params, use_tqdm=False)[0]
        text = output.outputs[0].text
        result.predefined_response = _selected(text, candidates) if candidates else _generated(text)
        result.usage = _add_usage(result.usage, self._offline_usage(output))
        return result

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        return self._batch_validate(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )[0]

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        import vllm

        selection = self._resolve_output_fields(output_fields, skip_evidences)
        response_model = response_model_for(selection)

        if ai_service_descriptions is not None:
            pairs = list(zip(conversations, ai_service_descriptions))
        elif ai_service_description is not None:
            pairs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError("an AI service description is required")

        prompts = [
            build_prompt(self.tokenizer, c, ad, output_fields=selection) for c, ad in pairs
        ]
        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            structured_outputs=vllm.sampling_params.StructuredOutputsParams(
                json=response_model.model_json_schema()
            ),
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(self.tokenizer.encode(SYSTEM_PROMPT))
        )

        results = []
        for output in outputs:
            text = output.outputs[0].text
            validated = response_model.model_validate(json.loads(text))
            prompt_tokens = len(output.prompt_token_ids) - system_prompt_tokens
            completion_tokens = len(output.outputs[0].token_ids)
            usage = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            results.append(_to_output(validated, self.model, usage))
        return results


@AsyncScopeGuardV2.register_guard("vllm-api")
class AsyncVLLMApiScopeGuardV2(AsyncScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm-api", "vllm-async-api"] = "vllm-api",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        chat_templating_tokenizer: str | None = None,
        count_system_prompt_in_usage: bool = False,
        include_default_safety_principles: bool = False,
        decision_temperature: float = 1.0,
        decision_logprobs: int = 20,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for AsyncScopeGuardV2.")
        self.decision_temperature = decision_temperature
        self.decision_logprobs = decision_logprobs
        self.default_model_name = model
        self.default_tokenizer_name = (
            chat_templating_tokenizer
            if chat_templating_tokenizer is not None
            else self.default_model_name
        )
        # The model, not the tokenizer: when a separate chat-templating tokenizer
        # is configured they are different repos, and the training prompt that
        # matters belongs to the model.
        check_shipped_system_prompt(self.default_model_name)
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens
        self.count_system_prompt_in_usage = count_system_prompt_in_usage

    async def _handle_request(
        self,
        model_name: str | None,
        conversation: ScopeGuardV2Input,
        ai_service_description: str | AIServiceDescriptionV2,
        selection: tuple[str, ...],
        prefill: bool,
        chat_templating_tokenizer: str | None = None,
    ) -> ScopeGuardV2Output:
        if chat_templating_tokenizer is not None:
            tokenizer = _get_tokenizer(chat_templating_tokenizer)
        elif model_name is not None:
            tokenizer = _get_tokenizer(model_name)
        else:
            tokenizer = _get_tokenizer(self.default_tokenizer_name)

        model_name = model_name if model_name is not None else self.default_model_name
        response_model = response_model_for(selection)

        prompt = build_prompt(
            tokenizer=tokenizer,
            conversation=conversation,
            ai_service_description=ai_service_description,
            prefill=prefill,
            output_fields=selection,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": self.vllm_temperature,
                    "max_tokens": self.vllm_max_tokens,
                    "structured_outputs": {"json": response_model.model_json_schema()},
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json["choices"][0]["text"]

        if prefill:
            response_text = prompt[prompt.rindex('{"') :] + response_text

        try:
            parsed_obj = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse generated text: {response_json}")

        try:
            validated = response_model.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(tokenizer.encode(SYSTEM_PROMPT))
        )
        usage = LLMUsage(
            prompt_tokens=response_json["usage"]["prompt_tokens"] - system_prompt_tokens,
            completion_tokens=response_json["usage"]["completion_tokens"],
            total_tokens=response_json["usage"]["total_tokens"] - system_prompt_tokens,
        )
        return _to_output(validated, model_name, usage)

    async def _completion(self, body: dict) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                return await response.json()

    def _api_usage(self, response_json: dict, tokenizer) -> LLMUsage:
        system_prompt_tokens = (
            0 if self.count_system_prompt_in_usage else len(tokenizer.encode(SYSTEM_PROMPT))
        )
        u = response_json["usage"]
        return LLMUsage(
            prompt_tokens=u["prompt_tokens"] - system_prompt_tokens,
            completion_tokens=u["completion_tokens"],
            total_tokens=u["total_tokens"] - system_prompt_tokens,
        )

    async def _classify(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        resolve_predefined: bool = True,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Classification:
        tokenizer_name = chat_templating_tokenizer or model or self.default_tokenizer_name
        tokenizer = _get_tokenizer(tokenizer_name)
        first_tokens = _class_tokens(tokenizer_name)
        model_name = model if model is not None else self.default_model_name

        prompt = (
            build_prompt(
                tokenizer, conversation, ai_service_description, output_fields=["scope_class"]
            )
            + CLASS_PREFIX
        )
        response_json = await self._completion(
            {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": self.decision_logprobs,
            }
        )
        top = response_json["choices"][0]["logprobs"]["top_logprobs"][0]
        result = _classification(
            top,
            first_tokens,
            temperature=self.decision_temperature,
            model=model_name,
            usage=self._api_usage(response_json, tokenizer),
        )
        if result.scope_class is not ScopeClass.PREDEFINED_ANSWER or not resolve_predefined:
            return result
        candidates = [r for _, r in predefined_candidates(ai_service_description)]
        if len(candidates) == 1:
            result.predefined_response = candidates[0]
            return result

        prompt = (
            build_prompt(
                tokenizer,
                conversation,
                ai_service_description,
                output_fields=["scope_class", "suggested_response"],
            )
            + PREDEFINED_PREFIX
        )
        body: dict = {"model": model_name, "prompt": prompt, "max_tokens": self.vllm_max_tokens}
        if candidates:
            # Constrained to the listed texts: the model picks, the customer's wording is kept.
            body |= {
                "temperature": 0.0,
                "structured_outputs": {"choice": _grammar_choices(candidates)},
            }
        else:
            # No list to choose from: generate the reply, as `validate` would.
            body |= {"temperature": self.vllm_temperature, "stop": [_RESPONSE_STOP]}
        response_json = await self._completion(body)
        text = response_json["choices"][0]["text"]
        result.predefined_response = _selected(text, candidates) if candidates else _generated(text)
        result.usage = _add_usage(result.usage, self._api_usage(response_json, tokenizer))
        return result

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        results = await self._batch_validate(
            conversations=[conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
            model=model,
            chat_templating_tokenizer=chat_templating_tokenizer,
        )
        return results[0]

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        if ai_service_description is not None:
            ai_service_descriptions = [ai_service_description] * len(conversations)  # type: ignore[invalid-assignment]

        tasks = [
            self._handle_request(
                model_name=model,
                conversation=c,
                ai_service_description=aisd,
                selection=selection,
                # Grammar-constrained decoding is unaware of a prefilled prefix, so
                # prefill stays off on this backend.
                prefill=False,
                chat_templating_tokenizer=chat_templating_tokenizer,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        return await asyncio.gather(*tasks)
