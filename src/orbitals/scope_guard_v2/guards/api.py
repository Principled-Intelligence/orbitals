import logging
import os
from typing import Literal

import aiohttp
import requests

from ...types import AIServiceDescriptionV2
from ..modeling import (
    ScopeGuardV2Input,
    ScopeGuardV2InputTypeAdapter,
    ScopeGuardV2Output,
)
from .base import AsyncScopeGuardV2, ScopeGuardV2


def _build_request_data(
    model: str | None,
    conversation: ScopeGuardV2Input,
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescriptionV2,
) -> dict:
    return {
        **({"model": model} if model is not None else {}),
        "conversation": ScopeGuardV2InputTypeAdapter.dump_python(conversation),
        "ai_service_description": ai_service_description.model_dump()
        if isinstance(ai_service_description, AIServiceDescriptionV2)
        else ai_service_description,
        "skip_evidences": skip_evidences,
    }


def _build_batch_request_data(
    model: str | None,
    conversations: list[ScopeGuardV2Input],
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescriptionV2 | None = None,
    ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
) -> dict:
    return {
        **({"model": model} if model is not None else {}),
        "conversations": [
            ScopeGuardV2InputTypeAdapter.dump_python(conversation)
            for conversation in conversations
        ],
        **(
            (
                {"ai_service_description": ai_service_description.model_dump()}
                if isinstance(ai_service_description, AIServiceDescriptionV2)
                else {"ai_service_description": ai_service_description}
            )
            if ai_service_description is not None
            else {}
        ),
        **(
            {
                "ai_service_descriptions": [
                    (
                        ad.model_dump()
                        if isinstance(ad, AIServiceDescriptionV2)
                        else ad
                    )
                    for ad in ai_service_descriptions
                ]
            }
            if ai_service_descriptions is not None
            else {}
        ),
        "skip_evidences": skip_evidences,
    }


def _maybe_get_api_key(
    args_api_key: str | None,
    custom_headers: dict[str, str] | None,
) -> str | None:
    if args_api_key is not None:
        logging.debug("Using API key from argument")
        return args_api_key

    if custom_headers is not None and "X-API-Key" in custom_headers:
        logging.debug("Using API key from custom headers")
        return custom_headers.pop("X-API-Key")

    api_key = os.environ.get("PRINCIPLED_API_KEY")
    if api_key is not None:
        logging.debug("Using API key from environment variable")

    return api_key


def _parse_output(result: dict) -> ScopeGuardV2Output:
    return ScopeGuardV2Output(
        scope_class=result["scope_class"],
        evidences=result.get("evidences"),
        reasoning=result["reasoning"],
        suggested_response=result.get("suggested_response"),
        model=result["model"],
        usage=result.get("usage"),
    )


@ScopeGuardV2.register_guard("api")
class APIScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["api"] = "api",
        model: str | None = None,
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
        )
        self.default_model = model
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = custom_headers if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard-v2/validate",
            json=_build_request_data(
                model=model if model is not None else self.default_model,
                conversation=conversation,
                skip_evidences=skip_evidences
                if skip_evidences is not None
                else self.skip_evidences,
                ai_service_description=ai_service_description,
            ),
            headers={**self.custom_headers, "Content-Type": "application/json"},
        )
        response.raise_for_status()
        return _parse_output(response.json())

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard-v2/batch-validate",
            json=_build_batch_request_data(
                model=model if model is not None else self.default_model,
                conversations=conversations,
                skip_evidences=skip_evidences
                if skip_evidences is not None
                else self.skip_evidences,
                ai_service_description=ai_service_description,
                ai_service_descriptions=ai_service_descriptions,
            ),
            headers={**self.custom_headers, "Content-Type": "application/json"},
        )
        response.raise_for_status()
        return [_parse_output(result) for result in response.json()]


@AsyncScopeGuardV2.register_guard("api")
class AsyncAPIScopeGuardV2(AsyncScopeGuardV2):
    def __init__(
        self,
        backend: Literal["api", "async-api"] = "api",
        model: str | None = None,
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
        )
        self.default_model = model
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = custom_headers if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/scope-guard-v2/validate",
                json=_build_request_data(
                    model=model if model is not None else self.default_model,
                    conversation=conversation,
                    skip_evidences=skip_evidences
                    if skip_evidences is not None
                    else self.skip_evidences,
                    ai_service_description=ai_service_description,
                ),
                headers={**self.custom_headers, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_data = await response.json()

        return _parse_output(response_data)

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/scope-guard-v2/batch-validate",
                json=_build_batch_request_data(
                    model=model if model is not None else self.default_model,
                    conversations=conversations,
                    skip_evidences=skip_evidences
                    if skip_evidences is not None
                    else self.skip_evidences,
                    ai_service_description=ai_service_description,
                    ai_service_descriptions=ai_service_descriptions,
                ),
                headers={**self.custom_headers, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_data = await response.json()

        return [_parse_output(result) for result in response_data]
