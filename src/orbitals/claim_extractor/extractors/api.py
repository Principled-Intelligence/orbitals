import logging
import os
from typing import Literal

import aiohttp
import requests

from ...types import AIServiceDescription
from ..modeling import (
    ClaimExtractorInput,
    ClaimExtractorInputTypeAdapter,
    ClaimExtractorOutput,
)
from .base import AsyncClaimExtractor, ClaimExtractor, DefaultModel


def _build_request_data(
    model: str,
    conversation: ClaimExtractorInput,
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescription | None,
) -> dict:
    return {
        "model": model,
        "conversation": ClaimExtractorInputTypeAdapter.dump_python(conversation),
        **(
            {
                "ai_service_description": ai_service_description.model_dump()
                if isinstance(ai_service_description, AIServiceDescription)
                else ai_service_description
            }
            if ai_service_description is not None
            else {}
        ),
        "skip_evidences": skip_evidences,
    }


def _build_batch_request_data(
    model: str,
    conversations: list[ClaimExtractorInput],
    skip_evidences: bool,
    ai_service_description: str | AIServiceDescription | None = None,
    ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
) -> dict:
    return {
        "model": model,
        "conversations": [
            ClaimExtractorInputTypeAdapter.dump_python(conversation)
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
    custom_headers: dict[str, str] | None,
) -> str | None:
    if args_api_key is not None:
        logging.debug("Using API key from argument")
        return args_api_key

    if custom_headers is not None and "X-API-Key" in custom_headers:
        logging.debug("Using API key from custom headers")
        return custom_headers["X-API-Key"]

    api_key = os.environ.get("PRINCIPLED_API_KEY")
    if api_key is not None:
        logging.debug("Using API key from environment variable")

    return api_key


@ClaimExtractor.register_extractor("api")
class APIClaimExtractor(ClaimExtractor):
    def __init__(
        self,
        backend: Literal["api"] = "api",
        model: DefaultModel | str = "claim-extractor",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = True,
        custom_headers: dict[str, str] | None = None,
    ):
        super().__init__(backend)
        self.default_model = self.maybe_map_model(model)
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = dict(custom_headers) if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        response = requests.post(
            f"{self.api_url}/orbitals/claim-extractor/extract",
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
        return ClaimExtractorOutput(
            extractions=response_data["extractions"],
            model=response_data["model"],
            usage=response_data["usage"],
        )

    def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        response = requests.post(
            f"{self.api_url}/orbitals/claim-extractor/batch-extract",
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
            ClaimExtractorOutput(
                extractions=result["extractions"],
                model=result["model"],
                usage=result["usage"],
            )
            for result in response_data
        ]


@AsyncClaimExtractor.register_extractor("api")
class AsyncAPIClaimExtractor(AsyncClaimExtractor):
    def __init__(
        self,
        backend: Literal["api"] = "api",
        model: DefaultModel | str = "claim-extractor",
        api_url: str = "http://localhost:8000",
        api_key: str | None = None,
        skip_evidences: bool = True,
        custom_headers: dict[str, str] | None = None,
    ):
        super().__init__(backend)
        self.default_model = self.maybe_map_model(model)
        self.api_url = api_url
        self.api_key = _maybe_get_api_key(api_key, custom_headers)
        self.skip_evidences = skip_evidences
        self.custom_headers = dict(custom_headers) if custom_headers is not None else {}
        if self.api_key is not None:
            self.custom_headers["X-API-Key"] = self.api_key

    async def _extract(
        self,
        conversation: ClaimExtractorInput,
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> ClaimExtractorOutput:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/claim-extractor/extract",
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

        return ClaimExtractorOutput(
            extractions=response_data["extractions"],
            model=response_data["model"],
            usage=response_data["usage"],
        )

    async def _batch_extract(
        self,
        conversations: list[ClaimExtractorInput],
        *,
        ai_service_description: str | AIServiceDescription | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
        skip_evidences: bool | None = None,
        model: str | None = None,
        **kwargs,
    ) -> list[ClaimExtractorOutput]:
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.api_url}/orbitals/claim-extractor/batch-extract",
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
            ClaimExtractorOutput(
                extractions=result["extractions"],
                model=result["model"],
                usage=result["usage"],
            )
            for result in response_data
        ]
