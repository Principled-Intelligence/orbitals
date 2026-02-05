from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, TypeAdapter

from ..types import ConversationMessage, LLMUsage


class ScopeClass(str, Enum):
    DIRECTLY_SUPPORTED = "Directly Supported"
    POTENTIALLY_SUPPORTED = "Potentially Supported"
    OUT_OF_SCOPE = "Out of Scope"
    RESTRICTED = "Restricted"
    CHIT_CHAT = "Chit Chat"

    @property
    def description(self) -> str:
        return _SCOPE_DESCRIPTIONS[self]

    @property
    def examples(self) -> list[str] | None:
        return _SCOPE_EXAMPLES.get(self)

    @classmethod
    def get_classes_manifest(cls) -> str:
        parts = []
        for cc in cls:
            parts.append(f"**Name**: {cc.value}\n**Description**: {cc.description}")
            if cc.examples:
                parts[-1] += (
                    "\n"
                    + "**Examples**:\n"
                    + "\n".join(map(lambda x: f"* {x}", cc.examples))
                )

        return "\n\n".join(parts)


_SCOPE_DESCRIPTIONS = {
    ScopeClass.DIRECTLY_SUPPORTED: "The user's query can be definitively handled by the AI Service given the functionalities, knowledge scope, and capabilities described in the AI Service Description.",
    ScopeClass.POTENTIALLY_SUPPORTED: "The user's query can be plausibly handled by the AI Service given the functionalities, knowledge scope, and capabilities described in the AI Service Description. The request is a reasonable extension or interpretation of the service's capabilities that might be within scope.",
    ScopeClass.OUT_OF_SCOPE: "The user's query is outside the AI Service's role and functionalities. It's incompatible with the service's documented purpose.",
    ScopeClass.RESTRICTED: "The user's query cannot be handled by the AI Service due to either behavioral restrictions (content/advice the service must refuse to provide) or service limitations (technical constraints, access restrictions, or knowledge scope limitations).",
    ScopeClass.CHIT_CHAT: "The user's query is a casual or social interaction that does not pertain to the AI Service's functionalities, knowledge scope, or operational capabilities.",
}


_SCOPE_EXAMPLES = {
    ScopeClass.RESTRICTED: [
        "Do not provide personalized financial advice. → User: Should I invest my savings in Bitcoin?",
        "The AI service does not have access to real-time data. → User: What's the current stock price of Apple?",
        "AI service cannot access external databases. → User: Look up my account balance in the system.",
    ],
}


class ScopeGuardOutput(BaseModel):
    evidences: list[str] | None
    scope_class: ScopeClass
    model: str
    usage: LLMUsage | None


class ConversationUserMessage(BaseModel):
    role: Literal["user"]
    content: str


def _select_model_based_on_fields(
    v: Any,
) -> str | ConversationUserMessage | list[ConversationMessage]:
    if isinstance(v, str):
        return TypeAdapter(str).validate_python(v)
    elif isinstance(v, dict):
        return TypeAdapter(ConversationUserMessage).validate_python(v)
    elif isinstance(v, list):
        return TypeAdapter(list[ConversationMessage]).validate_python(v)

    # no matching model found, let's fall back to standard pydantic behavior
    return v


ScopeGuardInput = Annotated[
    str | ConversationUserMessage | list[ConversationMessage],
    BeforeValidator(_select_model_based_on_fields),
]

ScopeGuardInputTypeAdapter = TypeAdapter(ScopeGuardInput)
