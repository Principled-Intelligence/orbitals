import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from ..types import ConversationMessage, LLMUsage

ExtractionSubType = Literal["Factoid", "Capability", "User Assertion", "Unverifiable"]


class Claim(BaseModel):
    subtype: ExtractionSubType
    content: str = Field(
        ...,
        description="A self-contained, decontextualized formulation of the extracted claim",
    )
    evidences: list[str] = Field(
        default_factory=list,
        description="Verbatim excerpts from the conversation messages that support the extracted claim",
    )


class Intent(BaseModel):
    content: str = Field(
        ...,
        description="A self-contained, decontextualized formulation of the extracted intent",
    )
    evidences: list[str] = Field(
        default_factory=list,
        description="Verbatim excerpts from the conversation messages that support the extracted intent",
    )


class Extractions(BaseModel):
    intents: list[Intent] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)


def _parse_raw_output(raw_output: str) -> Extractions:
    stripped = raw_output.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(stripped)
        inner = data["extractions"] if isinstance(data, dict) else data
        return Extractions(
            intents=[Intent.model_validate(i) for i in inner.get("intents", [])],
            claims=[Claim.model_validate(c) for c in inner.get("claims", [])],
        )
    except Exception:
        return Extractions()


class ClaimExtractorOutput(BaseModel):
    extractions: Extractions
    model: str
    usage: LLMUsage | None


def _select_model_based_on_fields(
    v: Any,
) -> str | ConversationMessage | list[ConversationMessage]:
    if isinstance(v, str):
        return TypeAdapter(str).validate_python(v)
    elif isinstance(v, dict):
        return TypeAdapter(ConversationMessage).validate_python(v)
    elif isinstance(v, list):
        return TypeAdapter(list[ConversationMessage]).validate_python(v)

    # no matching model found, let's fall back to standard pydantic behavior
    return v


ClaimExtractorInput = Annotated[
    str | ConversationMessage | list[ConversationMessage],
    BeforeValidator(_select_model_based_on_fields),
]


ClaimExtractorInputTypeAdapter = TypeAdapter(ClaimExtractorInput)
