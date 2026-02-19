import json
from typing import TYPE_CHECKING, Literal

import pydantic

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescription
from ..modeling import (
    ScopeClass,
    ScopeGuardInput,
    ScopeGuardOutput,
)
from ..prompting import ScopeGuardResponseModel
from .base import DefaultModel, ScopeGuard


@ScopeGuard.register_guard("hf")
class HuggingFaceScopeGuard(ScopeGuard):
    def __init__(
        self,
        backend: Literal["hf"] = "hf",
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        max_new_tokens: int = 3000,
        do_sample: bool = False,
        **kwargs,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        from transformers import pipeline  # noqa: F401

        super().__init__(backend)
        self.model = self.maybe_map_model(model)
        self._pipeline = pipeline(
            task="scope-guard",
            model=self.model,
            trust_remote_code=True,
            skip_evidences=skip_evidences,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )  # type: ignore # ty: ignore[no-matching-overload]

    def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        *,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ScopeGuardOutput:
        generated_text = self._pipeline(
            inputs=(conversation, ai_service_description),
            **(
                {"skip_evidences": skip_evidences} if skip_evidences is not None else {}
            ),
        )[0]["generated_text"]

        try:
            parsed_obj = json.loads(generated_text)
        except json.JSONDecodeError:
            # TODO generation errors: handle potentially invalid JSON (retry?)
            raise ValueError(f"Failed to parse generated text: {generated_text}")

        try:
            validated_obj = ScopeGuardResponseModel.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            # TODO generation errors: handle model validation failure (retry?)
            raise ValueError(f"Failed to validate generated text: {e}")

        return ScopeGuardOutput(
            scope_class=validated_obj.scope_class,
            evidences=validated_obj.evidences,
            model=self.model,
            usage=None,
        )

    def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        *,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardOutput]:
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
            # TODO generation errors: handle potentially invalid JSON (retry?)
            parsed_obj = json.loads(pipeline_output[0]["generated_text"])

            # TODO generation errors: handle model validation failure (retry?)
            results.append(
                ScopeGuardOutput(
                    evidences=parsed_obj.get("evidences"),
                    scope_class=ScopeClass(parsed_obj["scope_class"]),
                    model=self.model,
                    usage=None,
                )
            )

        return results
