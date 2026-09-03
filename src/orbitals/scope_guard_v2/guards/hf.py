import json
from typing import TYPE_CHECKING, Iterable, Literal

import pydantic

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescriptionV2
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import response_model_for
from .base import ScopeGuardV2


def _parse(generated_text: str, selection: tuple[str, ...], model: str) -> ScopeGuardV2Output:
    try:
        parsed_obj = json.loads(generated_text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse generated text: {generated_text}")
    try:
        validated = response_model_for(selection).model_validate(parsed_obj)
    except pydantic.ValidationError as e:
        raise ValueError(f"Failed to validate generated text: {e}")
    data = validated.model_dump()
    return ScopeGuardV2Output(
        evidences=data.get("evidences"),
        reasoning=data.get("reasoning"),
        scope_class=data["scope_class"],
        suggested_response=data.get("suggested_response"),
        model=model,
        usage=None,
    )


@ScopeGuardV2.register_guard("hf")
class HuggingFaceScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["hf"] = "hf",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        max_new_tokens: int = 3000,
        do_sample: bool = False,
        include_default_safety_principles: bool = False,
        **kwargs,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        from transformers import pipeline  # noqa: F401

        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        self._pipeline = pipeline(
            task="scope-guard-v2",
            model=self.model,
            trust_remote_code=True,
            output_fields=self._resolve_output_fields(None, None),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )  # type: ignore # ty: ignore[no-matching-overload]

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        generated_text = self._pipeline(
            (conversation, ai_service_description), output_fields=selection
        )[0]["generated_text"]
        return _parse(generated_text, selection, self.model)

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
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        if ai_service_descriptions is not None:
            pipeline_inputs = list(zip(conversations, ai_service_descriptions))
        elif ai_service_description is not None:
            pipeline_inputs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError("an AI service description is required")

        pipeline_outputs = self._pipeline(pipeline_inputs, output_fields=selection)
        return [
            _parse(out[0]["generated_text"], selection, self.model) for out in pipeline_outputs
        ]
