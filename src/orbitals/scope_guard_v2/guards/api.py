import logging
import os
from typing import Iterable, Literal

import aiohttp
import requests

from ...types import AIServiceDescriptionV2
from ..modeling import (
    ScopeGuardV2Input,
    ScopeGuardV2InputTypeAdapter,
    ScopeGuardV2Output,
)
from ..prompting import resolve_selection
from .base import AsyncScopeGuardV2, ScopeGuardV2


def _selection_fields(
    output_fields: Iterable[str] | None, skip_evidences: bool | None
) -> dict:
    """The output-field part of a request body.

    `skip_evidences` is always sent, derived from the selection when the caller
    gave `output_fields` -- so a server that predates `output_fields` still narrows
    the output as far as it can. `output_fields` itself is sent only when the
    caller asked for it explicitly, which keeps the body byte-identical to
    pre-0.5 clients for every caller who did not.
    """
    if output_fields is not None:
        selection = resolve_selection(output_fields, skip_evidences) or ()
        return {
            "skip_evidences": "evidences" not in selection,
            "output_fields": list(selection),
        }
    return {"skip_evidences": bool(skip_evidences)}


def _effective_args(
    guard, output_fields: Iterable[str] | None, skip_evidences: bool | None
) -> tuple[Iterable[str] | None, bool | None]:
    """Per-call values, falling back to constructor values, level by level.

    A constructor selection is sent as `output_fields` only if the caller gave it
    as `output_fields`; one that came from `skip_evidences` keeps the legacy shape.
    """
    if output_fields is not None or skip_evidences is not None:
        return output_fields, skip_evidences
    if guard.output_fields is not None and guard._ctor_output_fields_explicit:
        return guard.output_fields, None
    return None, guard.skip_evidences


def _build_request_data(
    model: str | None,
    conversation: ScopeGuardV2Input,
    skip_evidences: bool | None,
    ai_service_description: str | AIServiceDescriptionV2,
    output_fields: Iterable[str] | None = None,
) -> dict:
    return {
        **({"model": model} if model is not None else {}),
        "conversation": ScopeGuardV2InputTypeAdapter.dump_python(conversation),
        "ai_service_description": ai_service_description.model_dump()
        if isinstance(ai_service_description, AIServiceDescriptionV2)
        else ai_service_description,
        **_selection_fields(output_fields, skip_evidences),
    }


def _build_batch_request_data(
    model: str | None,
    conversations: list[ScopeGuardV2Input],
    skip_evidences: bool | None,
    ai_service_description: str | AIServiceDescriptionV2 | None = None,
    ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
    output_fields: Iterable[str] | None = None,
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
                    (ad.model_dump() if isinstance(ad, AIServiceDescriptionV2) else ad)
                    for ad in ai_service_descriptions
                ]
            }
            if ai_service_descriptions is not None
            else {}
        ),
        **_selection_fields(output_fields, skip_evidences),
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
        reasoning=result.get("reasoning"),
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
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        self._ctor_output_fields_explicit = output_fields is not None
        self.default_model = model
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.custom_headers = custom_headers if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        eff_fields, eff_skip = _effective_args(self, output_fields, skip_evidences)
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard-v2/validate",
            json=_build_request_data(
                model=model if model is not None else self.default_model,
                conversation=conversation,
                output_fields=eff_fields,
                skip_evidences=eff_skip,
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
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        eff_fields, eff_skip = _effective_args(self, output_fields, skip_evidences)
        response = requests.post(
            f"{self.api_url}/orbitals/scope-guard-v2/batch-validate",
            json=_build_batch_request_data(
                model=model if model is not None else self.default_model,
                conversations=conversations,
                output_fields=eff_fields,
                skip_evidences=eff_skip,
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
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        custom_headers: dict[str, str] | None = None,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        self._ctor_output_fields_explicit = output_fields is not None
        self.default_model = model
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.custom_headers = custom_headers if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        eff_fields, eff_skip = _effective_args(self, output_fields, skip_evidences)
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/scope-guard-v2/validate",
                json=_build_request_data(
                    model=model if model is not None else self.default_model,
                    conversation=conversation,
                    output_fields=eff_fields,
                    skip_evidences=eff_skip,
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
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        eff_fields, eff_skip = _effective_args(self, output_fields, skip_evidences)
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/scope-guard-v2/batch-validate",
                json=_build_batch_request_data(
                    model=model if model is not None else self.default_model,
                    conversations=conversations,
                    output_fields=eff_fields,
                    skip_evidences=eff_skip,
                    ai_service_description=ai_service_description,
                    ai_service_descriptions=ai_service_descriptions,
                ),
                headers={**self.custom_headers, "Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_data = await response.json()

        return [_parse_output(result) for result in response_data]
