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

from ...types import AIServiceDescription, LLMUsage
from ..modeling import (
    ClaimExtractorInput,
    ClaimExtractorOutput,
)
from ..prompting import (
    SYSTEM_PROMPT_IC_EXTRACTION,
    SYSTEM_PROMPT_ICE_EXTRACTION,
    SYSTEM_PROMPT_ICE_EXTRACTION_GUIDED,
    ExtractionsResponseModel,
    build_prompt,
)
from .base import AsyncClaimExtractor, ClaimExtractor, DefaultModel


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


def _select_system_prompt(skip_evidences: bool, use_guided_prompt: bool) -> str:
    if skip_evidences:
        return SYSTEM_PROMPT_IC_EXTRACTION
    return (
        SYSTEM_PROMPT_ICE_EXTRACTION_GUIDED
        if use_guided_prompt
        else SYSTEM_PROMPT_ICE_EXTRACTION
    )


@ClaimExtractor.register_extractor("vllm")
class VLLMClaimExtractor(ClaimExtractor):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = False,
        use_guided_prompt: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 20_000,
        max_model_len: int = 40_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import vllm

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self.skip_evidences = skip_evidences
        self.use_guided_prompt = use_guided_prompt
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

    def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        use_guided_prompt: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        return self._batch_extract(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            use_guided_prompt=use_guided_prompt,
        )[0]

    def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        use_guided_prompt: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        resolved_skip_evidences = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )
        resolved_use_guided_prompt = (
            use_guided_prompt
            if use_guided_prompt is not None
            else self.use_guided_prompt
        )

        if ai_service_descriptions is not None:
            descriptions: list[str | AIServiceDescription | None] = list(
                ai_service_descriptions
            )
        else:
            descriptions = [ai_service_description] * len(conversations)

        prompts = [
            build_prompt(
                self.tokenizer,
                c,
                ad,
                skip_evidences=resolved_skip_evidences,
                use_guided_prompt=resolved_use_guided_prompt,
            )
            for c, ad in zip(conversations, descriptions)
        ]

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        results: list[ClaimExtractorOutput] = []

        for output in outputs:
            text = output.outputs[0].text

            # TODO generation errors: handle potentially invalid JSON (retry?)
            parsed_obj = json.loads(text)

            # TODO generation errors: handle model validation failure (retry?)
            validated_obj = ExtractionsResponseModel.model_validate(parsed_obj)

            results.append(
                ClaimExtractorOutput(
                    extractions=validated_obj.extractions,
                    model=self.model,
                    usage=None,
                )
            )

        return results


@AsyncClaimExtractor.register_extractor("vllm-api")
class AsyncVLLMApiClaimExtractor(AsyncClaimExtractor):
    def __init__(
        self,
        backend: Literal["vllm-api", "vllm-async-api"] = "vllm-api",
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = False,
        use_guided_prompt: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 20_000,
        chat_templating_tokenizer: str | None = None,
        count_system_prompt_in_usage: bool = False,
    ):
        super().__init__(backend)
        self.default_model_name = self.maybe_map_model(model)
        self.default_tokenizer_name = (
            self.maybe_map_model(chat_templating_tokenizer)
            if chat_templating_tokenizer is not None
            else self.default_model_name
        )
        self.skip_evidences = skip_evidences
        self.use_guided_prompt = use_guided_prompt
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens
        self.count_system_prompt_in_usage = count_system_prompt_in_usage

    async def _handle_request(
        self,
        model_name: str | None,
        conversation: ClaimExtractorInput,
        ai_service_description: str | AIServiceDescription | None,
        skip_evidences: bool | None,
        use_guided_prompt: bool | None,
        prefill: bool,
        chat_templating_tokenizer: str | None = None,
    ) -> ClaimExtractorOutput:
        model_name = (
            self.maybe_map_model(model_name) if model_name is not None else None
        )

        if chat_templating_tokenizer is not None:
            tokenizer = _get_tokenizer(self.maybe_map_model(chat_templating_tokenizer))
        elif model_name is not None:
            tokenizer = _get_tokenizer(model_name)
        else:
            tokenizer = _get_tokenizer(self.default_tokenizer_name)

        model_name = model_name if model_name is not None else self.default_model_name
        resolved_skip_evidences = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )
        resolved_use_guided_prompt = (
            use_guided_prompt
            if use_guided_prompt is not None
            else self.use_guided_prompt
        )

        prompt = build_prompt(
            tokenizer=tokenizer,
            conversation=conversation,
            ai_service_description=ai_service_description,
            skip_evidences=resolved_skip_evidences,
            use_guided_prompt=resolved_use_guided_prompt,
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
                        "json": ExtractionsResponseModel.model_json_schema()
                    },
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json["choices"][0]["text"]

        try:
            parsed_obj = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse generated text: {response_json}")

        try:
            validated_obj = ExtractionsResponseModel.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(
                tokenizer.encode(
                    _select_system_prompt(
                        resolved_skip_evidences, resolved_use_guided_prompt
                    )
                )
            )
        )

        return ClaimExtractorOutput(
            extractions=validated_obj.extractions,
            model=model_name,
            # TODO usage implementation is mocked
            usage=LLMUsage(
                prompt_tokens=response_json["usage"]["prompt_tokens"]
                - system_prompt_tokens,
                completion_tokens=response_json["usage"]["completion_tokens"],
                total_tokens=response_json["usage"]["total_tokens"]
                - system_prompt_tokens,
            ),
        )

    async def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        use_guided_prompt: bool | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        results = await self._batch_extract(
            conversations=[conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            use_guided_prompt=use_guided_prompt,
            model=model,
            chat_templating_tokenizer=chat_templating_tokenizer,
        )
        return results[0]

    async def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        use_guided_prompt: bool | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        if ai_service_descriptions is None:
            ai_service_descriptions = [ai_service_description] * len(conversations)  # type: ignore[invalid-assignment]

        # Currently, we are not actually batching ON-PURPOSE
        # assuming a production-ready scenario, where we scale vllm serving
        # to multiple instances, it's actually better to dispatch requests one-by-one
        # to better distribute the load.
        # Sending all requests in a single batched request would mean a single vLLM server
        # would have to process all of them.
        # We are not even re-using the session to avoid ALBs not distributing requests properly.
        # TODO further optimizations and discussions

        tasks = [
            self._handle_request(
                model_name=model,
                conversation=c,
                ai_service_description=aisd,
                skip_evidences=skip_evidences,
                use_guided_prompt=use_guided_prompt,
                # TODO supporting True on this parameter is a bit tricky
                # problem is the grammar-based decoding, which is unaware of the prefilling
                prefill=False,
                chat_templating_tokenizer=chat_templating_tokenizer,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        results = await asyncio.gather(*tasks)

        return results
