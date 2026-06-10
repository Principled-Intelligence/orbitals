from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import aiohttp
import pydantic

if TYPE_CHECKING:
    import transformers  # noqa: F401
    import vllm  # noqa: F401

from ...types import AIServiceDescriptionV2, LLMUsage
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import SYSTEM_PROMPT, ScopeGuardV2ResponseModel, build_prompt
from .base import AsyncScopeGuardV2, ScopeGuardV2


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


@ScopeGuardV2.register_guard("vllm")
class VLLMScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: str | None = None,
        skip_evidences: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 30_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        include_default_safety_principles: bool = False,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import vllm

        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        self.skip_evidences = skip_evidences
        self.llm = vllm.LLM(
            model=self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = _get_tokenizer(self.model)
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        return self._batch_validate(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
        )[0]

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        resolved_skip_evidences = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )

        if ai_service_descriptions is not None:
            prompts = [
                build_prompt(
                    self.tokenizer,
                    c,
                    ad,
                    skip_evidences=resolved_skip_evidences,
                )
                for c, ad in zip(conversations, ai_service_descriptions)
            ]
        elif ai_service_description is not None:
            prompts = [
                build_prompt(
                    self.tokenizer,
                    c,
                    ai_service_description,
                    skip_evidences=resolved_skip_evidences,
                )
                for c in conversations
            ]
        else:
            raise ValueError

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            parsed_obj = json.loads(text)
            validated_obj = ScopeGuardV2ResponseModel.model_validate(parsed_obj)
            results.append(
                ScopeGuardV2Output(
                    evidences=validated_obj.evidences,
                    reasoning=validated_obj.reasoning,
                    scope_class=validated_obj.scope_class,
                    suggested_response=validated_obj.suggested_response,
                    model=self.model,
                    usage=None,
                )
            )

        return results


@AsyncScopeGuardV2.register_guard("vllm-api")
class AsyncVLLMApiScopeGuardV2(AsyncScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm-api", "vllm-async-api"] = "vllm-api",
        model: str | None = None,
        skip_evidences: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        chat_templating_tokenizer: str | None = None,
        count_system_prompt_in_usage: bool = False,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
        )
        if model is None:
            raise ValueError("A model name must be provided for AsyncScopeGuardV2.")
        self.default_model_name = model
        self.default_tokenizer_name = (
            chat_templating_tokenizer
            if chat_templating_tokenizer is not None
            else self.default_model_name
        )
        self.skip_evidences = skip_evidences
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens
        self.count_system_prompt_in_usage = count_system_prompt_in_usage

    async def _handle_request(
        self,
        model_name: str | None,
        conversation: ScopeGuardV2Input,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None,
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
        skip_evidences = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )

        prompt = build_prompt(
            tokenizer=tokenizer,
            conversation=conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            prefill=prefill,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": self.vllm_temperature,
                    "max_tokens": self.vllm_max_tokens,
                    "structured_outputs": {
                        "json": ScopeGuardV2ResponseModel.model_json_schema()
                    },
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json["choices"][0]["text"]

        if prefill:
            response_text = prompt[prompt.rindex('{"evidences"') :] + response_text

        try:
            parsed_obj = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse generated text: {response_json}")

        try:
            validated_obj = ScopeGuardV2ResponseModel.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(tokenizer.encode(SYSTEM_PROMPT))
        )

        return ScopeGuardV2Output(
            scope_class=validated_obj.scope_class,
            evidences=validated_obj.evidences,
            reasoning=validated_obj.reasoning,
            suggested_response=validated_obj.suggested_response,
            model=model_name,
            usage=LLMUsage(
                prompt_tokens=response_json["usage"]["prompt_tokens"]
                - system_prompt_tokens,
                completion_tokens=response_json["usage"]["completion_tokens"],
                total_tokens=response_json["usage"]["total_tokens"]
                - system_prompt_tokens,
            ),
        )

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        results = await self._batch_validate(
            conversations=[conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
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
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        if ai_service_description is not None:
            ai_service_descriptions = [ai_service_description] * len(conversations)  # type: ignore[invalid-assignment]

        tasks = [
            self._handle_request(
                model_name=model,
                conversation=c,
                ai_service_description=aisd,
                skip_evidences=skip_evidences,
                # TODO supporting True on this parameter is a bit tricky
                # problem is the grammar-based decoding, which is unaware of the prefilling
                prefill=False,
                chat_templating_tokenizer=chat_templating_tokenizer,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        results = await asyncio.gather(*tasks)

        return results
