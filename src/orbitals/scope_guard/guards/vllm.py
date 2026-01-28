import asyncio
import json
from typing import TYPE_CHECKING, Literal

import aiohttp
import pydantic

if TYPE_CHECKING:
    import transformers  # noqa: F401
    import vllm  # noqa: F401

from ...types import AIServiceDescription, LLMUsage
from ..modeling import (
    ScopeGuardInput,
    ScopeGuardOutput,
)
from ..prompting import ScopeGuardResponseModel, build_prompt
from .base import AsyncScopeGuard, DefaultModel, ScopeGuard


@ScopeGuard.register_guard("vllm")
class VLLMScopeGuard(ScopeGuard):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 15_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import transformers
        import vllm

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self.skip_evidences = skip_evidences
        self.llm = vllm.LLM(
            model=self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        self.sampling_params = vllm.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
    ) -> ScopeGuardOutput:
        return self._batch_validate(
            [conversation], ai_service_description, skip_evidences=skip_evidences
        )[0]

    def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
    ) -> list[ScopeGuardOutput]:
        if ai_service_descriptions is not None:
            prompts = [
                build_prompt(
                    self.tokenizer,
                    c,
                    ad,
                    skip_evidences=skip_evidences
                    if skip_evidences is not None
                    else self.skip_evidences,
                )
                for c, ad in zip(conversations, ai_service_descriptions)
            ]
        elif ai_service_description is not None:
            prompts = [
                build_prompt(
                    self.tokenizer,
                    c,
                    ai_service_description,
                    skip_evidences=skip_evidences
                    if skip_evidences is not None
                    else self.skip_evidences,
                )
                for c in conversations
            ]
        else:
            raise ValueError

        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

        results = []

        for output in outputs:
            text = output.outputs[0].text

            # TODO generation errors: handle potentially invalid JSON (retry?)
            parsed_obj = json.loads(text)

            # TODO generation errors: handle model validation failure (retry?)
            validated_obj = ScopeGuardResponseModel.model_validate(parsed_obj)

            results.append(
                ScopeGuardOutput(
                    evidences=validated_obj.evidences,
                    scope_class=validated_obj.scope_class,
                    model=self.model,
                    usage=None,
                )
            )

        return results


@AsyncScopeGuard.register_guard("vllm-api")
class AsyncVLLMApiScopeGuard(AsyncScopeGuard):
    def __init__(
        self,
        backend: Literal["vllm-api", "vllm-async-api"] = "vllm-api",
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ):
        import transformers

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        self.skip_evidences = skip_evidences
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens

    async def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
    ) -> ScopeGuardOutput:
        results = await self._batch_validate(
            [conversation], ai_service_description, skip_evidences=skip_evidences
        )
        return results[0]

    async def _handle_request(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None,
        prefill: bool,
    ) -> ScopeGuardOutput:
        prompt = build_prompt(
            self.tokenizer,
            conversation,
            ai_service_description,
            skip_evidences=skip_evidences
            if skip_evidences is not None
            else self.skip_evidences,
            prefill=prefill,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.vllm_temperature,
                    "max_tokens": self.vllm_max_tokens,
                    "structured_outputs": {
                        "json": ScopeGuardResponseModel.model_json_schema()
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
            validated_obj = ScopeGuardResponseModel.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        return ScopeGuardOutput(
            scope_class=validated_obj.scope_class,
            evidences=validated_obj.evidences,
            model=self.model,
            # TODO usage implementation is mocked
            usage=LLMUsage(
                prompt_tokens=response_json["usage"]["prompt_tokens"],
                completion_tokens=response_json["usage"]["completion_tokens"],
                total_tokens=response_json["usage"]["total_tokens"],
            ),
        )

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
    ) -> list[ScopeGuardOutput]:
        if ai_service_description is not None:
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
                c,
                aisd,
                skip_evidences=skip_evidences,
                # TODO supporting True on this parameter is a bit tricky
                # problem is the grammar-based decoding, which is unaware of the prefilling
                # one possible solution is replacing the json decoding with
                prefill=False,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        results = await asyncio.gather(*tasks)

        return results
