from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import aiohttp
import pydantic

if TYPE_CHECKING:
    import transformers  # noqa: F401
    import vllm  # noqa: F401

from ...types import AIServiceDescriptionV2, LLMUsage
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import SYSTEM_PROMPT, build_prompt, response_model_for
from .base import AsyncScopeGuardV2, ScopeGuardV2

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


def check_shipped_system_prompt(model_path: str) -> None:
    """Warn if the model directory ships a system prompt that is not ours.

    The 2608 releases include `system_prompt.txt`. The library never reads it to
    build prompts -- the built-in `SYSTEM_PROMPT` is the source of truth -- but a
    mismatch means the model was trained on a different prompt generation and will
    degrade quietly. Trailing newlines are ignored: the shipped copies are known to
    lack the one the model trained with, and that alone is not worth a warning.

    Args:
        model_path: A local model directory. Hub ids and missing files are ignored.
    """
    path = Path(model_path) / "system_prompt.txt"
    if not path.is_file():
        return
    shipped = path.read_text(encoding="utf-8").rstrip("\n")
    ours = SYSTEM_PROMPT.rstrip("\n")
    if shipped == ours:
        return
    logger.warning(
        "system_prompt.txt in %s (sha256 %s) differs from the prompt this library "
        "was built for (sha256 %s). The model was likely trained on a different "
        "prompt generation; classifications may be degraded.",
        model_path,
        hashlib.sha256(shipped.encode("utf-8")).hexdigest()[:8],
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8],
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

        results = []
        for output in outputs:
            text = output.outputs[0].text
            validated = response_model.model_validate(json.loads(text))
            results.append(_to_output(validated, self.model, usage=None))
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
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for AsyncScopeGuardV2.")
        self.default_model_name = model
        self.default_tokenizer_name = (
            chat_templating_tokenizer
            if chat_templating_tokenizer is not None
            else self.default_model_name
        )
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
