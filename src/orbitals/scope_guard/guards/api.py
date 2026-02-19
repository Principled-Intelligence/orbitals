import logging
import os
from typing import Literal

import aiohttp
import requests

from ...types import AIServiceDescription
from ..modeling import (
    ScopeGuardInput,
    ScopeGuardInputTypeAdapter,
    ScopeGuardOutput,
)
from .base import AsyncScopeGuard, DefaultModel, ScopeGuard


def _build_request_data(
    model: str,
    conversation: ScopeGuardInput,
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescription,
) -> dict:
    return {
        "model": model,
        "conversation": ScopeGuardInputTypeAdapter.dump_python(conversation),
        "ai_service_description": ai_service_description.model_dump()
        if isinstance(ai_service_description, AIServiceDescription)
        else ai_service_description,
        "skip_evidences": skip_evidences,
    }


def _build_batch_request_data(
    model: str,
    conversations: list[ScopeGuardInput],
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescription | None = None,
    ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
) -> dict:
    return {
        "model": model,
        "conversations": [
            ScopeGuardInputTypeAdapter.dump_python(conversation)
            for conversation in conversations
        ],
        **(
            (
                {"ai_service_description": ai_service_description.model_dump()}
                if isinstance(ai_service_description, AIServiceDescription)
                else {"ai_service_description": ai_service_description}
            )
            if ai_service_description is not None
            else {}
        ),
        **(
            {
                "ai_service_descriptions": [
                    (ad.model_dump() if isinstance(ad, AIServiceDescription) else ad)
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
    custom_headers: dict[str, str],
) -> str | None:
    if args_api_key is not None:
        logging.warning("Using API key from argument")
        return args_api_key

    if "X-API-Key" in custom_headers:
        logging.warning("Using API key from custom headers")
        return custom_headers.pop("X-API-Key")

    api_key = os.environ.get("PRINCIPLED_API_KEY")
    if api_key is not None:
        logging.warning("Using API key from environment variable")

    return api_key


@ScopeGuard.register_guard("api")
class APIScopeGuard(ScopeGuard):
    def __init__(
        self,
        backend: Literal["api"] = "api",
        model: DefaultModel | str = "scope-guard",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] = {},
    ):
        super().__init__(backend)
        self.default_model = self.maybe_map_model(model)
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = custom_headers
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        *,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardOutput:
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard/validate",
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

        response_data = response.json()
        return ScopeGuardOutput(
            scope_class=response_data["scope_class"],
            evidences=response_data["evidences"],
            model=response_data["model"],
            usage=response_data["usage"],
        )

    def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        *,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardOutput]:
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard/batch-validate",
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

        response_data = response.json()
        return [
            ScopeGuardOutput(
                scope_class=result["scope_class"],
                evidences=result["evidences"],
                model=result["model"],
                usage=result["usage"],
            )
            for result in response_data
        ]


@AsyncScopeGuard.register_guard("api")
class AsyncAPIScopeGuard(AsyncScopeGuard):
    def __init__(
        self,
        backend: Literal["api", "async-api"] = "api",
        model: DefaultModel | str = "scope-guard",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = False,
        custom_headers: dict[str, str] = {},
    ):
        super().__init__(backend)
        self.default_model = self.maybe_map_model(model)
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = custom_headers
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    async def _validate(
        self,
        conversation: ScopeGuardInput,
        ai_service_description: str | AIServiceDescription,
        *,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardOutput:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/in/scope-guard/validate",
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

        return ScopeGuardOutput(
            scope_class=response_data["scope_class"],
            evidences=response_data["evidences"],
            model=response_data["model"],
            usage=response_data["usage"],
        )

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardInput],
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        *,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardOutput]:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/in/scope-guard/batch-validate",
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

        return [
            ScopeGuardOutput(
                scope_class=result["scope_class"],
                evidences=result["evidences"],
                model=result["model"],
                usage=result["usage"],
            )
            for result in response_data
        ]
