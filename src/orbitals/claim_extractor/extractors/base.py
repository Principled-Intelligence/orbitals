from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

from pydantic import ValidationError

if TYPE_CHECKING:
    from .api import APIClaimExtractor, AsyncAPIClaimExtractor
    from .hf import HuggingFaceClaimExtractor
    from .vllm import AsyncVLLMApiClaimExtractor, VLLMClaimExtractor

from ...types import AIServiceDescription
from ..modeling import (
    ClaimExtractorInput,
    ClaimExtractorInputTypeAdapter,
    ClaimExtractorOutput,
)

DefaultModel = Literal["claim-extractor"]

# TODO: the HF repos for claim-extractor are not finalized yet. These placeholders
# preserve the short-name -> repo indirection used by ScopeGuard; swap them with
# the real repos once they are published.
MODEL_MAPPING = {
    "claim-extractor": "principled-intelligence/claim-extractor-TBD",
    "claim-extractor-q": "principled-intelligence/claim-extractor-q-TBD",
    "claim-extractor-g": "principled-intelligence/claim-extractor-g-TBD",
}


class BaseClaimExtractor:
    _registry: dict[str, dict[str, type[ClaimExtractor | AsyncClaimExtractor]]] = {}

    @classmethod
    def _get_outer_registry_key(cls) -> str:
        return "async" if issubclass(cls, AsyncClaimExtractor) else "sync"

    @classmethod
    def register_extractor(cls, backend: str):
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
        if cls is not ClaimExtractor and cls is not AsyncClaimExtractor:
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
    ) -> ClaimExtractorInput:
        try:
            return ClaimExtractorInputTypeAdapter.validate_python(conversation)
        except ValidationError as e:
            logging.error("Invalid input format for conversation")
            raise e from None

    def _validate_conversations(
        self, conversations: list[str] | list[dict] | list[list[dict]]
    ) -> list[ClaimExtractorInput]:
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
        conversations: list[ClaimExtractorInput],
        ai_service_description: str | AIServiceDescription | None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None,
    ):
        if bool(ai_service_description is not None) == bool(
            ai_service_descriptions is not None
        ):
            if ai_service_description is not None:
                raise ValueError(
                    "Only one between [ai_service_description, ai_service_descriptions] must be provided"
                )
            # Claim extraction supports running without an AI Service Description;
            # downstream code treats missing descriptions as a neutral default.

        if ai_service_descriptions is not None and len(conversations) != len(
            ai_service_descriptions
        ):
            raise ValueError(
                "The number of conversations and ai_service_descriptions must be the same"
            )


class ClaimExtractor(BaseClaimExtractor):
    @overload
    def __new__(
        cls,
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
    ) -> HuggingFaceClaimExtractor: ...

    @overload
    def __new__(
        cls,
        backend: Literal["vllm"],
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 20_000,
        max_model_len: int = 40_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
    ) -> VLLMClaimExtractor: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: DefaultModel | str = "claim-extractor",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = True,
        custom_headers: dict[str, str] | None = None,
    ) -> APIClaimExtractor: ...

    def __new__(cls, backend: str = "hf", *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    def extract(
        self,
        conversation: str | dict | list[dict],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        conversation = self._validate_conversation(conversation)
        return self._extract(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        raise NotImplementedError

    def batch_extract(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        return self._batch_extract(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            **kwargs,
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
        raise NotImplementedError


class AsyncClaimExtractor(BaseClaimExtractor):
    @overload
    def __new__(
        cls,
        backend: Literal["vllm-api"],
        model: DefaultModel | str = "claim-extractor",
        skip_evidences: bool = True,
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
    ) -> AsyncVLLMApiClaimExtractor: ...

    @overload
    def __new__(
        cls,
        backend: Literal["api"],
        model: DefaultModel | str = "claim-extractor",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = True,
        custom_headers: dict[str, str] | None = None,
    ) -> AsyncAPIClaimExtractor: ...

    def __new__(cls, backend: str, *args, **kwargs):
        return super().__new__(cls, backend, *args, **kwargs)

    async def extract(
        self,
        conversation: str | dict | list[dict],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        conversation = self._validate_conversation(conversation)
        return await self._extract(
            conversation,
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    async def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        raise NotImplementedError

    async def batch_extract(
        self,
        conversations: list[str] | list[dict] | list[list[dict]],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        if len(conversations) == 0:
            return []

        validated_conversations = self._validate_conversations(conversations)
        self._validate_ai_service_description_input(
            validated_conversations, ai_service_description, ai_service_descriptions
        )

        return await self._batch_extract(
            validated_conversations,
            ai_service_description=ai_service_description,
            ai_service_descriptions=ai_service_descriptions,
            skip_evidences=skip_evidences,
            **kwargs,
        )

    async def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        raise NotImplementedError
