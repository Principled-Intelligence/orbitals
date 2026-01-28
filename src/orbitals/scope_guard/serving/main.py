import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orbitals.scope_guard import AsyncScopeGuard
from orbitals.scope_guard.guards import AsyncVLLMApiScopeGuard
from orbitals.scope_guard.modeling import (
    ScopeClass,
    ScopeGuardInput,
)
from orbitals.types import AIServiceDescription, LLMUsage

scope_guard: AsyncVLLMApiScopeGuard


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scope_guard

    scope_guard = AsyncScopeGuard(  # type: ignore[invalid-assignment]
        backend="vllm-api",
        model=os.environ["SCOPE_GUARD_VLLM_MODEL"],
        skip_evidences=os.environ["SCOPE_GUARD_SKIP_EVIDENCES"] == "1",
        vllm_serving_url=os.environ["SCOPE_GUARD_VLLM_SERVING_URL"],
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


class ScopeGuardResponse(BaseModel):
    scope_class: ScopeClass
    evidences: list[str] | None
    model: str
    usage: LLMUsage
    time_taken: float


@app.post("/orbitals/scope-guard/validate", response_model=ScopeGuardResponse)
async def validate(
    request: Request,
    conversation: ScopeGuardInput,
    ai_service_description: Annotated[str | AIServiceDescription, Body()],
    skip_evidences: bool | None = Body(None),
) -> ScopeGuardResponse:
    global scope_guard

    start_time = time.time()
    result = await scope_guard.validate(
        conversation, ai_service_description, skip_evidences
    )
    end_time = time.time()

    return ScopeGuardResponse(
        scope_class=result.scope_class,
        evidences=result.evidences,
        model=scope_guard.model,
        usage=result.usage,  # type: ignore[invalid-argument-type]
        time_taken=end_time - start_time,
    )


@app.post(
    "/orbitals/scope-guard/batch-validate",
    response_model=list[ScopeGuardResponse],
)
async def batch_validate(
    request: Request,
    conversations: list[ScopeGuardInput],
    ai_service_description: str | AIServiceDescription | None = Body(None),
    ai_service_descriptions: list[str] | list[AIServiceDescription] | None = None,
    skip_evidences: bool | None = Body(None),
) -> list[ScopeGuardResponse]:
    global scope_guard

    start_time = time.time()
    results = await scope_guard.batch_validate(
        conversations,
        ai_service_description=ai_service_description,
        ai_service_descriptions=ai_service_descriptions,
        skip_evidences=skip_evidences,
    )
    end_time = time.time()

    return [
        ScopeGuardResponse(
            scope_class=result.scope_class,
            evidences=result.evidences,
            time_taken=end_time
            - start_time,  # TODO time in this way doesn't make much sense
            model=scope_guard.model,
            usage=result.usage,  # type: ignore[invalid-argument-type]
        )
        for result in results
    ]
