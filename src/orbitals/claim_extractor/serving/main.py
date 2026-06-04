import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orbitals.claim_extractor import AsyncClaimExtractor
from orbitals.claim_extractor.extractors import AsyncVLLMApiClaimExtractor
from orbitals.claim_extractor.modeling import (
    ClaimExtractorInput,
    Extractions,
)
from orbitals.types import AIServiceDescription, ConversationMessage, LLMUsage

claim_extractor: AsyncVLLMApiClaimExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global claim_extractor

    claim_extractor = AsyncClaimExtractor(  # type: ignore[invalid-assignment]
        backend="vllm-api",
        model=os.environ["CLAIM_EXTRACTOR_VLLM_MODEL"],
        skip_evidences=os.environ.get("CLAIM_EXTRACTOR_SKIP_EVIDENCES", "1") == "1",
        intents_only=os.environ.get("CLAIM_EXTRACTOR_INTENTS_ONLY", "0") == "1",
        vllm_serving_url=os.environ["CLAIM_EXTRACTOR_VLLM_SERVING_URL"],
        temperature=float(os.environ.get("CLAIM_EXTRACTOR_TEMPERATURE", "0.7")),
        frequency_penalty=float(
            os.environ.get("CLAIM_EXTRACTOR_FREQUENCY_PENALTY", "0.0")
        ),
        presence_penalty=float(
            os.environ.get("CLAIM_EXTRACTOR_PRESENCE_PENALTY", "1.5")
        ),
        repetition_penalty=float(
            os.environ.get("CLAIM_EXTRACTOR_REPETITION_PENALTY", "1.0")
        ),
        top_p=float(os.environ.get("CLAIM_EXTRACTOR_TOP_P", "0.8")),
        top_k=int(os.environ.get("CLAIM_EXTRACTOR_TOP_K", "20")),
        min_p=float(os.environ.get("CLAIM_EXTRACTOR_MIN_P", "0.0")),
    )

    yield


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimExtractorResponse(BaseModel):
    extractions: Extractions
    model: str
    usage: LLMUsage
    time_taken: float


class ConversationClaimExtractorResponse(BaseModel):
    extractions: list[Extractions]
    model: str
    usage: LLMUsage
    time_taken: float


def _require_usage(usage: LLMUsage | None) -> LLMUsage:
    if usage is None:
        raise HTTPException(
            status_code=500,
            detail="ClaimExtractor serving expected usage information from the backend",
        )
    return usage


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orbitals/claim-extractor/extract", response_model=ClaimExtractorResponse)
async def extract(
    conversation: ClaimExtractorInput,
    ai_service_description: Annotated[
        str | AIServiceDescription | None, Body()
    ] = None,
    skip_evidences: Annotated[bool | None, Body()] = None,
    intents_only: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
) -> ClaimExtractorResponse:
    global claim_extractor

    start_time = time.time()
    result = await claim_extractor.extract(
        conversation,
        ai_service_description=ai_service_description,
        skip_evidences=skip_evidences,
        intents_only=intents_only,
        model=model,
    )
    end_time = time.time()

    return ClaimExtractorResponse(
        extractions=result.extractions,
        model=result.model,
        usage=_require_usage(result.usage),
        time_taken=end_time - start_time,
    )


@app.post(
    "/orbitals/claim-extractor/batch-extract",
    response_model=list[ClaimExtractorResponse],
)
async def batch_extract(
    conversations: list[ClaimExtractorInput],
    ai_service_description: str | AIServiceDescription | None = Body(None),
    ai_service_descriptions: list[str] | list[AIServiceDescription] | None = Body(None),
    skip_evidences: Annotated[bool | None, Body()] = None,
    intents_only: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
) -> list[ClaimExtractorResponse]:
    global claim_extractor

    if ai_service_description is not None and ai_service_descriptions is not None:
        raise HTTPException(
            status_code=400,
            detail="Only one between ai_service_description and ai_service_descriptions must be provided",
        )

    if ai_service_descriptions is not None and len(conversations) != len(
        ai_service_descriptions
    ):
        raise HTTPException(
            status_code=400,
            detail="The number of conversations and ai_service_descriptions must be the same",
        )

    descriptions: list[str | AIServiceDescription | None]
    if ai_service_descriptions is not None:
        descriptions = list(ai_service_descriptions)
    else:
        descriptions = [ai_service_description] * len(conversations)

    async def _timed_extract(
        conversation: ClaimExtractorInput,
        description: str | AIServiceDescription | None,
    ) -> ClaimExtractorResponse:
        start_time = time.time()
        result = await claim_extractor.extract(
            conversation,
            ai_service_description=description,
            skip_evidences=skip_evidences,
            intents_only=intents_only,
            model=model,
        )
        end_time = time.time()
        return ClaimExtractorResponse(
            extractions=result.extractions,
            model=result.model,
            usage=_require_usage(result.usage),
            time_taken=end_time - start_time,
        )

    return await asyncio.gather(
        *[
            _timed_extract(conversation, description)
            for conversation, description in zip(conversations, descriptions)
        ]
    )


@app.post(
    "/orbitals/claim-extractor/extract-conversation",
    response_model=ConversationClaimExtractorResponse,
)
async def extract_conversation(
    conversation: ClaimExtractorInput,
    ai_service_description: Annotated[
        str | AIServiceDescription | None, Body()
    ] = None,
    skip_evidences: Annotated[bool | None, Body()] = None,
    intents_only: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
) -> ConversationClaimExtractorResponse:
    global claim_extractor

    if isinstance(conversation, str):
        messages = [ConversationMessage(role="assistant", content=conversation)]
    elif isinstance(conversation, ConversationMessage):
        messages = [conversation]
    else:
        messages = list(conversation)

    if len(messages) == 0:
        raise HTTPException(
            status_code=400, detail="conversation must contain at least one message"
        )

    prefixes = [messages[: i + 1] for i in range(len(messages))]

    start_time = time.time()
    results = await claim_extractor.batch_extract(
        prefixes,
        ai_service_description=ai_service_description,
        skip_evidences=skip_evidences,
        intents_only=intents_only,
        model=model,
    )
    end_time = time.time()

    usages = [_require_usage(r.usage) for r in results]
    total_usage = LLMUsage(
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )

    return ConversationClaimExtractorResponse(
        extractions=[r.extractions for r in results],
        model=results[0].model,
        usage=total_usage,
        time_taken=end_time - start_time,
    )
