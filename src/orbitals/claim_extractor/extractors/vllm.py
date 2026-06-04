from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

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
    CLAIMS_STOP_STRING,
    build_prompt,
    get_system_prompt,
    parse_intents_only_output,
    validate_extractions_response,
)
from .base import AsyncClaimExtractor, ClaimExtractor, DefaultModel


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


def _strip_lone_surrogates(obj):
    # Structured decoding occasionally emits unpaired \uXXXX escapes (typically
    # half of an emoji pair). json.loads accepts them into str, but pydantic_core
    # / serde_json refuse to encode lone surrogates as UTF-8 — replace them with
    # U+FFFD so downstream serialization works.
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _strip_lone_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_lone_surrogates(x) for x in obj]
    return obj


_DEFAULT_SPECULATIVE_CONFIG = {"num_speculative_tokens": 4, "method": "mtp"}
# Sentinel for `speculative_config`: distinguishes "user did not specify, apply
# our default" from "user explicitly passed None to disable speculative decoding".
class _UseDefaultSpeculativeConfig:
    pass


_USE_DEFAULT_SPECULATIVE_CONFIG: Final = _UseDefaultSpeculativeConfig()


@ClaimExtractor.register_extractor("vllm")
class VLLMClaimExtractor(ClaimExtractor):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = True,
        intents_only: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 20_000,
        max_model_len: int = 40_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = True,
        language_model_only: bool = True,
        speculative_config: dict
        | None
        | _UseDefaultSpeculativeConfig = _USE_DEFAULT_SPECULATIVE_CONFIG,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import vllm

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self.skip_evidences = skip_evidences
        self.intents_only = intents_only
        if speculative_config is _USE_DEFAULT_SPECULATIVE_CONFIG:
            speculative_config = dict(_DEFAULT_SPECULATIVE_CONFIG)
        self.llm = vllm.LLM(
            model=self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            language_model_only=language_model_only,
            speculative_config=speculative_config,
        )
        self.tokenizer = _get_tokenizer(self.model)
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )

    def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        intents_only: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        return self._batch_extract(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            intents_only=intents_only,
        )[0]

    def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        intents_only: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        resolved_skip_evidences = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )
        resolved_intents_only = (
            intents_only if intents_only is not None else self.intents_only
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
            )
            for c, ad in zip(conversations, descriptions)
        ]

        if resolved_intents_only:
            sampling_params = self.sampling_params.clone()
            sampling_params.stop = [CLAIMS_STOP_STRING]
        else:
            sampling_params = self.sampling_params

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results: list[ClaimExtractorOutput] = []

        for output in outputs:
            text = output.outputs[0].text

            if resolved_intents_only:
                extractions = parse_intents_only_output(text)
            else:
                # TODO generation errors: handle potentially invalid JSON (retry?)
                parsed_obj = _strip_lone_surrogates(json.loads(text))

                # TODO generation errors: handle model validation failure (retry?)
                validated_obj = validate_extractions_response(
                    parsed_obj,
                    skip_evidences=resolved_skip_evidences,
                )
                extractions = validated_obj.extractions

            results.append(
                ClaimExtractorOutput(
                    extractions=extractions,
                    model=self.model,
                    usage=None,
                )
            )

        return results


@AsyncClaimExtractor.register_extractor("vllm-api")
class AsyncVLLMApiClaimExtractor(AsyncClaimExtractor):
    def __init__(
        self,
        backend: Literal["vllm-api"] = "vllm-api",
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = True,
        intents_only: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.7,
        max_tokens: int = 20_000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
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
        self.intents_only = intents_only
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens
        self.vllm_frequency_penalty = frequency_penalty
        self.vllm_presence_penalty = presence_penalty
        self.vllm_repetition_penalty = repetition_penalty
        self.vllm_top_p = top_p
        self.vllm_top_k = top_k
        self.vllm_min_p = min_p
        self.count_system_prompt_in_usage = count_system_prompt_in_usage

    async def _handle_request(
        self,
        model_name: str | None,
        conversation: ClaimExtractorInput,
        ai_service_description: str | AIServiceDescription | None,
        skip_evidences: bool | None,
        intents_only: bool | None,
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
        resolved_intents_only = (
            intents_only if intents_only is not None else self.intents_only
        )

        prompt = build_prompt(
            tokenizer=tokenizer,
            conversation=conversation,
            ai_service_description=ai_service_description,
            skip_evidences=resolved_skip_evidences,
            prefill=prefill,
        )

        request_body = {
            "model": model_name,
            "prompt": prompt,
            "temperature": self.vllm_temperature,
            "max_tokens": self.vllm_max_tokens,
            "frequency_penalty": self.vllm_frequency_penalty,
            "presence_penalty": self.vllm_presence_penalty,
            "repetition_penalty": self.vllm_repetition_penalty,
            "top_p": self.vllm_top_p,
            "top_k": self.vllm_top_k,
            "min_p": self.vllm_min_p,
        }
        if resolved_intents_only:
            request_body["stop"] = [CLAIMS_STOP_STRING]

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json["choices"][0]["text"]

        if resolved_intents_only:
            extractions = parse_intents_only_output(response_text)
        else:
            try:
                parsed_obj = _strip_lone_surrogates(json.loads(response_text))
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse generated text: {response_json}")

            try:
                validated_obj = validate_extractions_response(
                    parsed_obj,
                    skip_evidences=resolved_skip_evidences,
                )
            except pydantic.ValidationError as e:
                raise ValueError(f"Failed to validate generated text: {e}")
            extractions = validated_obj.extractions

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(tokenizer.encode(get_system_prompt(resolved_skip_evidences)))
        )

        return ClaimExtractorOutput(
            extractions=extractions,
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
        intents_only: bool | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        results = await self._batch_extract(
            conversations=[conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            intents_only=intents_only,
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
        intents_only: bool | None = None,
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
                intents_only=intents_only,
                # TODO supporting True on this parameter is a bit tricky
                # problem is the grammar-based decoding, which is unaware of the prefilling
                prefill=False,
                chat_templating_tokenizer=chat_templating_tokenizer,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        results = await asyncio.gather(*tasks)

        return results
