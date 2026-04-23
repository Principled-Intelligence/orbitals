import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orbitals.claim_extractor import AsyncClaimExtractor
from orbitals.claim_extractor.extractors import AsyncVLLMApiClaimExtractor
from orbitals.claim_extractor.modeling import (
    ClaimExtractorInput,
    Extractions,
)
from orbitals.types import AIServiceDescription, LLMUsage

claim_extractor: AsyncVLLMApiClaimExtractor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global claim_extractor

    claim_extractor = AsyncClaimExtractor(  # type: ignore[invalid-assignment]
        backend="vllm-api",
        model=os.environ["CLAIM_EXTRACTOR_VLLM_MODEL"],
        skip_evidences=os.environ.get("CLAIM_EXTRACTOR_SKIP_EVIDENCES", "0") == "1",
        use_guided_prompt=os.environ.get("CLAIM_EXTRACTOR_USE_GUIDED_PROMPT", "0")
        == "1",
        vllm_serving_url=os.environ["CLAIM_EXTRACTOR_VLLM_SERVING_URL"],
    )

    yield


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClaimExtractorResponse(BaseModel):
    extractions: Extractions
    model: str
    usage: LLMUsage
    time_taken: float


@app.post("/orbitals/claim-extractor/extract", response_model=ClaimExtractorResponse)
async def extract(
    conversation: ClaimExtractorInput,
    ai_service_description: Annotated[
        str | AIServiceDescription | None, Body()
    ] = None,
    skip_evidences: Annotated[bool | None, Body()] = None,
    use_guided_prompt: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
) -> ClaimExtractorResponse:
    global claim_extractor

    start_time = time.time()
    result = await claim_extractor.extract(
        conversation,
        ai_service_description=ai_service_description,
        skip_evidences=skip_evidences,
        use_guided_prompt=use_guided_prompt,
        model=model,
    )
    end_time = time.time()

    return ClaimExtractorResponse(
        extractions=result.extractions,
        model=result.model,
        usage=result.usage,  # type: ignore[invalid-argument-type]
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
    use_guided_prompt: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
) -> list[ClaimExtractorResponse]:
    global claim_extractor

    start_time = time.time()
    results = await claim_extractor.batch_extract(
        conversations,
        ai_service_description=ai_service_description,
        ai_service_descriptions=ai_service_descriptions,
        skip_evidences=skip_evidences,
        use_guided_prompt=use_guided_prompt,
        model=model,
    )
    end_time = time.time()

    return [
        ClaimExtractorResponse(
            extractions=result.extractions,
            model=result.model,
            usage=result.usage,  # type: ignore[invalid-argument-type]
            time_taken=end_time - start_time,
        )
        for result in results
    ]
