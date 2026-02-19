from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from pydantic import ValidationError

if TYPE_CHECKING:
    from .api import APIScopeGuard, AsyncAPIScopeGuard
    from .hf import HuggingFaceScopeGuard
    from .vllm import AsyncVLLMApiScopeGuard, VLLMScopeGuard

from ...types import AIServiceDescription
from ..modeling import (
    ScopeGuardInput,
    ScopeGuardInputTypeAdapter,
    ScopeGuardOutput,
)

DefaultModel = Literal["scope-guard"]

MODEL_MAPPING = {
    "scope-guard": "principled-intelligence/scope-guard-4B-q-2601",
    "scope-guard-q": "principled-intelligence/scope-guard-4B-q-2601",
    "scope-guard-g": "principled-intelligence/scope-guard-4B-g-2601",
}


class BaseScopeGuard:
    _registry: dict[str, dict[str, type[ScopeGuard | AsyncScopeGuard]]] = {}

    @classmethod
    def _get_outer_registry_key(cls) -> str:
        return "async" if issubclass(cls, AsyncScopeGuard) else "sync"

    @classmethod
    def register_guard(cls, backend: str):
        """Class decorator for registering a backend."""
        outer_registry_key = cls._get_outer_registry_key()
        if outer_registry_key not in cls._registry:
            cls._registry[outer_registry_key] = {}

        def decorator(subclass):
            if backend in cls._registry[outer_registry_key]:
                raise ValueError(f"Backend '{backend}' already registered")

            cls._registry[outer_registry_key][backend] = subclass
            return subclass

        return decorator

    @classmethod
    def maybe_map_model(cls, model: DefaultModel | str) -> str:
        if model in MODEL_MAPPING:
            logging.warning(
                f"Detected simplified model name, using {MODEL_MAPPING[model]}"
            )
            return MODEL_MAPPING[model]
        return model

    def __new__(
        cls,
        backend: str = "hf",
        *args,
        **kwargs,
    ):
        if cls is not ScopeGuard and cls is not AsyncScopeGuard:
            # if called on subclass, behave normally
            return super().__new__(cls)

        try:
            subclass = cls._registry[cls._get_outer_registry_key()][backend]
        except KeyError:
            raise ValueError(
                f"Unknown backend '{backend}'. Available: {list(cls._registry[cls._get_outer_registry_key()].keys())}"
            )

        return super().__new__(subclass)

    def __init__(
        self,
        backend: str,
        *args,
        **kwargs,
    ):
        self.backend = backend

    def _validate_conversation(
        self, conversation: str | dict | list[dict]
    ) -> ScopeGuardInput:
        try:
            return ScopeGuardInputTypeAdapter.validate_python(conversation)
        except ValidationError as e:
            logging.error("Invalid input format for conversation")
            raise e from None

    def _validate_conversations(
        self, conversations: list[str] | list[dict] | list[list[dict]]
    ) -> list[ScopeGuardInput]:
        validated_conversations = []
        for c in conversations:
            try:
                validated_conversations.append(self._validate_conversation(c))
            except ValidationError as e:
                logging.error("Invalid input format for conversation")
                raise e from None
        return validated_conversations

    def _validate_ai_service_description_input(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription,
        ai_service_descriptions: list[str] | list[AIServiceDescription],
    ):
        if bool(ai_service_description is not None) == bool(
            ai_service_descriptions is not None
        ):
            if ai_service_description is not None:
                raise ValueError(
                    "Only one between [ai_service_description, ai_service_descriptions] must be provided"
                )
            else:
                raise ValueError(
                    "Either ai_service_description or ai_service_descriptions must be provided"
                )

        if ai_service_descriptions is not None and len(conversations) != len(
            ai_service_descriptions
        ):
            raise ValueError(
                "The number of conversations and ai_service_descriptions must be the same"
            )


class ScopeGuard(BaseScopeGuard):
    @overload
    def __new__(
        cls,
        backend: Literal["hf"] = "hf",
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        max_new_tokens: int = 3000,
        do_sample: bool = False,
        **kwargs,
    ) -> HuggingFaceScopeGuard: ...

    @overload
    def __new__(
        cls,
        backend: Literal["vllm"],
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 30_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
    ) -> VLLMScopeGuard: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: DefaultModel | str = "scope-guard",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] = {},
    ) -> APIScopeGuard: ...

    def __new__(cls, backend: str = "hf", *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    def validate(
        self,
        conversation: str | dict | list[dict],
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> ScopeGuardOutput:
        conversation = self._validate_conversation(conversation)
        return self._validate(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            model=model,
        )

    def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> ScopeGuardOutput:
        raise NotImplementedError

    def batch_validate(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> list[ScopeGuardOutput]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        return self._batch_validate(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            model=model,
        )

    def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> list[ScopeGuardOutput]:
        raise NotImplementedError


class AsyncScopeGuard(BaseScopeGuard):
    @overload
    def __new__(
        cls,
        backend: Literal["vllm-api"],
        model: DefaultModel | str = "scope-guard",
        skip_evidences: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        chat_templating_tokenizer: str | None = None,
    ) -> AsyncVLLMApiScopeGuard: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: DefaultModel | str = "scope-guard",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] = {},
    ) -> AsyncAPIScopeGuard: ...

    def __new__(cls, backend: str, *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    async def validate(
        self,
        conversation: str | dict | list[dict],
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> ScopeGuardOutput:
        conversation = self._validate_conversation(conversation)
        return await self._validate(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            model=model,
        )

    async def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> ScopeGuardOutput:
        raise NotImplementedError

    async def batch_validate(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> list[ScopeGuardOutput]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        return await self._batch_validate(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            model=model,
        )

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
    ) -> list[ScopeGuardOutput]:
        raise NotImplementedError
