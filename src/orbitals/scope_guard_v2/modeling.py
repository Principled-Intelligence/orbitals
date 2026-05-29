from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter

from ..types import ConversationMessage, LLMUsage


class ScopeClassDefinition(BaseModel):
    """Definition of a scope classification class"""

    name: str = Field(description="Name of the scope class")
    description: str = Field(
        description="Description of when this classification applies"
    )

    def dumps(self) -> str:
        return f"{self.name}: {self.description}"


class ScopeClass(str, Enum):
    """Constants for scope class names"""

    DIRECTLY_SUPPORTED = "Directly Supported"
    POTENTIALLY_SUPPORTED = "Potentially Supported"
    PREDEFINED_ANSWER = "Predefined Answer"
    HUMAN_OVERSIGHT = "Human Oversight"
    HUMAN_REVIEW = "Human Oversight"
    OUT_OF_SCOPE = "Out of Scope"
    RESTRICTED = "Restricted"
    CHIT_CHAT = "Chit Chat"

    @property
    def description(self) -> str:
        return _SCOPE_DESCRIPTIONS[self]

    @classmethod
    def get_all_definitions(cls) -> list[ScopeClassDefinition]:
        """Get all scope class definitions"""
        return SCOPE_CLASSES

    @classmethod
    def get_definition(cls, name: str) -> ScopeClassDefinition | None:
        """Get definition for a specific scope class"""
        for definition in SCOPE_CLASSES:
            if definition.name == name:
                return definition
        return None

    @classmethod
    def get_classes_manifest(cls) -> str:
        """Get a prompt-friendly description of all scope classes."""
        return "\n".join(f"- {definition.dumps()}" for definition in SCOPE_CLASSES)


_SCOPE_DESCRIPTIONS = {
    ScopeClass.DIRECTLY_SUPPORTED: "The query is clearly within the scope of the AI service. The AI service can and should handle this type of request.",
    ScopeClass.POTENTIALLY_SUPPORTED: "The query is plausibly within the scope of the AI service. In most situations, it's generally safe for the AI service to handle this, but may require careful consideration.",
    ScopeClass.PREDEFINED_ANSWER: "The query matches a predefined scenario or FAQ that has a specific, pre-written response. Use the exact predefined response.",
    ScopeClass.HUMAN_OVERSIGHT: "The query requires human oversight, judgment, or expertise. This could be due to complexity, sensitivity, legal implications, or escalation criteria being met.",
    ScopeClass.OUT_OF_SCOPE: "The query is outside the scope of the AI service and its responsibilities. The AI service should not process this type of request.",
    ScopeClass.RESTRICTED: "The query requests something that is explicitly forbidden or restricted. The AI service must never process this type of request.",
    ScopeClass.CHIT_CHAT: "Social pleasantries, small talk, or casual conversation not related to the service (e.g., 'thank you', 'goodbye', 'how are you'). Does not require substantial service assistance.",
}


SCOPE_CLASSES = [
    ScopeClassDefinition(name=scope_class.value, description=scope_class.description)
    for scope_class in ScopeClass
]


class ScopeGuardV2Output(BaseModel):
    evidences: list[str] | None = Field(
        default=None,
        description="Evidences from the AI Service Description supporting this classification.",
    )
    reasoning: str = Field(
        description="A short explanation of why this classification was chosen."
    )
    scope_class: ScopeClass = Field(
        description="The scope classification (must be one of the defined scope classes)."
    )
    suggested_response: str | None = Field(
        default=None, description="A short suggested answer or message for the user."
    )
    model: str = Field(description="The model used to produce this classification.")
    usage: LLMUsage | None = Field(
        default=None, description="Token usage reported by the model backend."
    )


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


ScopeGuardV2Input = Annotated[
    str | ConversationUserMessage | list[ConversationMessage],
    BeforeValidator(_select_model_based_on_fields),
]

ScopeGuardV2InputTypeAdapter = TypeAdapter(ScopeGuardV2Input)
