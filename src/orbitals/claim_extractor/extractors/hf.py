from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescription
from ..modeling import (
    ClaimExtractorInput,
    ClaimExtractorOutput,
    _parse_raw_output,
)
from .base import ClaimExtractor, DefaultModel


@ClaimExtractor.register_extractor("hf")
class HuggingFaceClaimExtractor(ClaimExtractor):
    def __init__(
        self,
        backend: Literal["hf"] = "hf",
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = True,
        max_new_tokens: int = 20_000,
        do_sample: bool = False,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        **kwargs,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        from transformers import pipeline  # noqa: F401

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self.skip_evidences = skip_evidences
        self._pipeline = pipeline(
            task="claim-extractor",
            model=self.model,
            trust_remote_code=True,
            skip_evidences=skip_evidences,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            **kwargs,
        )  # type: ignore # ty: ignore[no-matching-overload]

    def _resolve_flags(
        self,
        skip_evidences: bool | None,
    ) -> dict[str, bool]:
        resolved: dict[str, bool] = {}
        if skip_evidences is not None:
            resolved["skip_evidences"] = skip_evidences
        return resolved

    def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        pipeline_kwargs = self._resolve_flags(skip_evidences)
        generated_text = self._pipeline(
            inputs=(conversation, ai_service_description),
            **pipeline_kwargs,
        )[0]["generated_text"]

        extractions = _parse_raw_output(generated_text)

        return ClaimExtractorOutput(
            extractions=extractions,
            model=self.model,
            usage=None,
        )

    def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        if ai_service_descriptions is not None:
            pipeline_inputs = [
                (c, ad) for c, ad in zip(conversations, ai_service_descriptions)
            ]
        else:
            pipeline_inputs = [(c, ai_service_description) for c in conversations]

        pipeline_kwargs = self._resolve_flags(skip_evidences)
        pipeline_outputs = self._pipeline(
            pipeline_inputs,
            **pipeline_kwargs,
        )

        results: list[ClaimExtractorOutput] = []

        for pipeline_output in pipeline_outputs:
            generated_text = pipeline_output[0]["generated_text"]
            extractions = _parse_raw_output(generated_text)
            results.append(
                ClaimExtractorOutput(
                    extractions=extractions,
                    model=self.model,
                    usage=None,
                )
            )

        return results
