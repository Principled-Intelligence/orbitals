import json
from typing import TYPE_CHECKING, Literal

import pydantic

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescriptionV2
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import ScopeGuardV2ResponseModel
from .base import ScopeGuardV2


@ScopeGuardV2.register_guard("hf")
class HuggingFaceScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["hf"] = "hf",
        model: str | None = None,
        skip_evidences: bool = False,
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
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        self._pipeline = pipeline(
            task="scope-guard-v2",
            model=self.model,
            trust_remote_code=True,
            skip_evidences=skip_evidences,
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
        **kwargs,
    ) -> ScopeGuardV2Output:
        generated_text = self._pipeline(
            inputs=(conversation, ai_service_description),
            **(
                {"skip_evidences": skip_evidences} if skip_evidences is not None else {}
            ),
        )[0]["generated_text"]

        try:
            parsed_obj = json.loads(generated_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse generated text: {generated_text}")

        try:
            validated_obj = ScopeGuardV2ResponseModel.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        return ScopeGuardV2Output(
            scope_class=validated_obj.scope_class,
            evidences=validated_obj.evidences,
            reasoning=validated_obj.reasoning,
            suggested_response=validated_obj.suggested_response,
            model=self.model,
            usage=None,
        )

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        if ai_service_descriptions is not None:
            pipeline_inputs = [
                (c, ad) for c, ad in zip(conversations, ai_service_descriptions)
            ]
        elif ai_service_description is not None:
            pipeline_inputs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError

        pipeline_outputs = self._pipeline(
            pipeline_inputs,
            **(
                {"skip_evidences": skip_evidences} if skip_evidences is not None else {}
            ),
        )

        results = []
        for pipeline_output in pipeline_outputs:
            parsed_obj = json.loads(pipeline_output[0]["generated_text"])
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
