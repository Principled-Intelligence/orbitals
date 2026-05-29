from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from pydantic import ValidationError

if TYPE_CHECKING:
    from .api import APIScopeGuardV2, AsyncAPIScopeGuardV2
    from .hf import HuggingFaceScopeGuardV2
    from .vllm import AsyncVLLMApiScopeGuardV2, VLLMScopeGuardV2

from ...types import AIServiceDescriptionV2
from ..modeling import (
    ScopeGuardV2Input,
    ScopeGuardV2InputTypeAdapter,
    ScopeGuardV2Output,
)
from ..safety_principles import augment_with_default_safety_principles_v2


class BaseScopeGuardV2:
    _registry: dict[str, dict[str, type[ScopeGuardV2 | AsyncScopeGuardV2]]] = {}

    @classmethod
    def _get_outer_registry_key(cls) -> str:
        return "async" if issubclass(cls, AsyncScopeGuardV2) else "sync"

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
    def maybe_map_model(cls, model: str) -> str:
        return model

    def __new__(
        cls,
        backend: str = "vllm",
        *args,
        **kwargs,
    ):
        if cls is not ScopeGuardV2 and cls is not AsyncScopeGuardV2:
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
        include_default_safety_principles: bool = False,
        **kwargs,
    ):
        self.backend = backend
        self.include_default_safety_principles = include_default_safety_principles

    def _resolve_include_default_safety_principles(
        self, per_call_value: bool | None
    ) -> bool:
        if per_call_value is not None:
            return per_call_value
        return getattr(self, "include_default_safety_principles", False)

    def _maybe_augment(
        self,
        ai_service_description: str | AIServiceDescriptionV2 | None,
        include: bool,
    ) -> str | AIServiceDescriptionV2 | None:
        if not include or ai_service_description is None:
            return ai_service_description
        return augment_with_default_safety_principles_v2(ai_service_description)

    def _maybe_augment_list(
        self,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None,
        include: bool,
    ) -> list[str] | list[AIServiceDescriptionV2] | None:
        if not include or ai_service_descriptions is None:
            return ai_service_descriptions
        return [
            augment_with_default_safety_principles_v2(ad)
            for ad in ai_service_descriptions
        ]  # type: ignore[return-value]

    def _validate_conversation(
        self, conversation: str | dict | list[dict]
    ) -> ScopeGuardV2Input:
        try:
            return ScopeGuardV2InputTypeAdapter.validate_python(conversation)
        except ValidationError as e:
            logging.error("Invalid input format for conversation")
            raise e from None

    def _validate_conversations(
        self, conversations: list[str] | list[dict] | list[list[dict]]
    ) -> list[ScopeGuardV2Input]:
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
        conversations: list[ScopeGuardV2Input],
        ai_service_description: str | AIServiceDescriptionV2 | None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None,
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


class ScopeGuardV2(BaseScopeGuardV2):
    @overload
    def __new__(
        cls,
        backend: Literal["vllm"],
        model: str,
        skip_evidences: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 30_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        include_default_safety_principles: bool = False,
    ) -> VLLMScopeGuardV2: ...

    @overload
    def __new__(
        cls,
        backend: Literal["hf"],
        model: str,
        skip_evidences: bool = False,
        max_new_tokens: int = 3000,
        do_sample: bool = False,
        include_default_safety_principles: bool = False,
        **kwargs,
    ) -> HuggingFaceScopeGuardV2: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: str | None = None,
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ) -> APIScopeGuardV2: ...

    def __new__(cls, backend: str = "vllm", *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    def validate(
        self,
        conversation: str | dict | list[dict],
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        include_default_safety_principles: bool | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        conversation = self._validate_conversation(conversation)
        include = self._resolve_include_default_safety_principles(
            include_default_safety_principles
        )
        ai_service_description = self._maybe_augment(ai_service_description, include)
        return self._validate(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        raise NotImplementedError

    def batch_validate(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        include_default_safety_principles: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        include = self._resolve_include_default_safety_principles(
            include_default_safety_principles
        )
        ai_service_description = self._maybe_augment(ai_service_description, include)
        ai_service_descriptions = self._maybe_augment_list(
            ai_service_descriptions, include
        )

        return self._batch_validate(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            **kwargs,
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
        raise NotImplementedError


class AsyncScopeGuardV2(BaseScopeGuardV2):
    @overload
    def __new__(
        cls,
        backend: Literal["vllm-api"],
        model: str,
        skip_evidences: bool = False,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        chat_templating_tokenizer: str | None = None,
        count_system_prompt_in_usage: bool = False,
        include_default_safety_principles: bool = False,
    ) -> AsyncVLLMApiScopeGuardV2: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: str | None = None,
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ) -> AsyncAPIScopeGuardV2: ...

    def __new__(cls, backend: str, *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    async def validate(
        self,
        conversation: str | dict | list[dict],
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        include_default_safety_principles: bool | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        conversation = self._validate_conversation(conversation)
        include = self._resolve_include_default_safety_principles(
            include_default_safety_principles
        )
        ai_service_description = self._maybe_augment(ai_service_description, include)
        return await self._validate(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        raise NotImplementedError

    async def batch_validate(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        include_default_safety_principles: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        include = self._resolve_include_default_safety_principles(
            include_default_safety_principles
        )
        ai_service_description = self._maybe_augment(ai_service_description, include)
        ai_service_descriptions = self._maybe_augment_list(
            ai_service_descriptions, include
        )

        return await self._batch_validate(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        raise NotImplementedError
