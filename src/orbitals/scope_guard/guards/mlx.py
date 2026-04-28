from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

from ...types import AIServiceDescription
from ..modeling import (
    ScopeGuardInput,
    ScopeGuardOutput,
)
from ..prompting import ScopeGuardResponseModel, build_prompt
from .base import DefaultModel, ScopeGuard

MLX_MODEL_MAPPING: dict[str, str] = {
    "scope-guard": "ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16",
    "scope-guard-q": "ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16",
    "scope-guard-g": "ivanfioravanti/scope-guard-4B-g-2601-mlx-bf16",
}


def _build_mlx_prompt(
    tokenizer,
    conversation: ScopeGuardInput,
    ai_service_description: str | AIServiceDescription,
    skip_evidences: bool,
) -> str:
    return build_prompt(
        tokenizer,
        conversation,
        ai_service_description,
        skip_evidences=skip_evidences,
    )


@ScopeGuard.register_guard("mlx")
class MLXScopeGuard(ScopeGuard):
    def __init__(
        self,
        backend: Literal["mlx"] = "mlx",
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        max_tokens: int = 3000,
        temp: float = 0.0,
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = None,
        top_p: float | None = None,
        **kwargs,
    ):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        super().__init__(backend)

        if model in MLX_MODEL_MAPPING:
            logging.warning(
                f"Detected simplified model name, using {MLX_MODEL_MAPPING[model]}"
            )
            self.model = MLX_MODEL_MAPPING[model]
        else:
            self.model = self.maybe_map_model(model)

        self.skip_evidences = skip_evidences
        self.max_tokens = max_tokens
        self._sampler = make_sampler(
            temp=temp,
            top_p=top_p if top_p is not None else 0.0,
        )
        self._repetition_penalty = repetition_penalty
        self._repetition_context_size = repetition_context_size

        self._model, self.tokenizer = load(self.model)

    def _generate_kwargs(self) -> dict:
        kwargs: dict = {}
        if self._repetition_penalty is not None:
            kwargs["repetition_penalty"] = self._repetition_penalty
        if self._repetition_context_size is not None:
            kwargs["repetition_context_size"] = self._repetition_context_size
        return kwargs

    def _validate(
        self,
        conversation: ScopeGuardInput,
        *,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ScopeGuardOutput:
        return self._batch_validate(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
        )[0]

    def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardOutput]:
        from mlx_lm import generate

        skip = (
            skip_evidences if skip_evidences is not None else self.skip_evidences
        )

        if ai_service_descriptions is not None:
            prompts = [
                _build_mlx_prompt(self.tokenizer, c, ad, skip)
                for c, ad in zip(conversations, ai_service_descriptions)
            ]
        elif ai_service_description is not None:
            prompts = [
                _build_mlx_prompt(self.tokenizer, c, ai_service_description, skip)
                for c in conversations
            ]
        else:
            raise ValueError

        gen_kwargs = self._generate_kwargs()

        results = []
        for prompt in prompts:
            text = generate(
                self._model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=self._sampler,
                verbose=False,
                **gen_kwargs,
            )

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
