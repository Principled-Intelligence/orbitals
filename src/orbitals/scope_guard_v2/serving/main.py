import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from orbitals.scope_guard_v2 import AsyncScopeGuardV2
from orbitals.scope_guard_v2.guards import AsyncVLLMApiScopeGuardV2
from orbitals.scope_guard_v2.modeling import ScopeClass, ScopeGuardV2Input
from orbitals.types import AIServiceDescriptionV2, LLMUsage

scope_guard: AsyncVLLMApiScopeGuardV2


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scope_guard

    scope_guard = AsyncScopeGuardV2(  # type: ignore[invalid-assignment]
        backend="vllm-api",
        model=os.environ["SCOPE_GUARD_V2_VLLM_MODEL"],
        skip_evidences=os.environ["SCOPE_GUARD_V2_SKIP_EVIDENCES"] == "1",
        vllm_serving_url=os.environ["SCOPE_GUARD_V2_VLLM_SERVING_URL"],
    )

    yield


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScopeGuardV2Response(BaseModel):
    scope_class: ScopeClass
    evidences: list[str] | None
    reasoning: str
    suggested_response: str | None
    model: str
    usage: LLMUsage
    time_taken: float


@app.post("/orbitals/scope-guard-v2/validate", response_model=ScopeGuardV2Response)
async def validate(
    conversation: ScopeGuardV2Input,
    ai_service_description: Annotated[str | AIServiceDescriptionV2, Body()],
    skip_evidences: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
    include_default_safety_principles: Annotated[bool | None, Body()] = None,
) -> ScopeGuardV2Response:
    global scope_guard

    start_time = time.time()
    result = await scope_guard.validate(
        conversation,
        ai_service_description=ai_service_description,
        skip_evidences=skip_evidences,
        include_default_safety_principles=include_default_safety_principles,
        model=model,
    )
    end_time = time.time()

    return ScopeGuardV2Response(
        scope_class=result.scope_class,
        evidences=result.evidences,
        reasoning=result.reasoning,
        suggested_response=result.suggested_response,
        model=result.model,
        usage=result.usage,  # type: ignore[invalid-argument-type]
        time_taken=end_time - start_time,
    )


@app.post(
    "/orbitals/scope-guard-v2/batch-validate",
    response_model=list[ScopeGuardV2Response],
)
async def batch_validate(
    conversations: list[ScopeGuardV2Input],
    ai_service_description: str | AIServiceDescriptionV2 | None = Body(None),
    ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = Body(
        None
    ),
    skip_evidences: Annotated[bool | None, Body()] = None,
    model: Annotated[str | None, Body()] = None,
    include_default_safety_principles: Annotated[bool | None, Body()] = None,
) -> list[ScopeGuardV2Response]:
    global scope_guard

    start_time = time.time()
    results = await scope_guard.batch_validate(
        conversations,
        ai_service_description=ai_service_description,
        ai_service_descriptions=ai_service_descriptions,
        skip_evidences=skip_evidences,
        include_default_safety_principles=include_default_safety_principles,
        model=model,
    )
    end_time = time.time()

    return [
        ScopeGuardV2Response(
            scope_class=result.scope_class,
            evidences=result.evidences,
            reasoning=result.reasoning,
            suggested_response=result.suggested_response,
            time_taken=end_time - start_time,
            model=result.model,
            usage=result.usage,  # type: ignore[invalid-argument-type]
        )
        for result in results
    ]
