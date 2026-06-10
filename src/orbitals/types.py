from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMUsage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ConversationMessage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    messages: list[ConversationMessage]


class SupportingMaterial(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    pass


class Principle(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    title: str = Field(description="Title of the principle")
    description: str = Field(description="Description of the principle")
    supporting_materials: list[SupportingMaterial] | None


class AIServiceDescription(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    identity_role: str = Field(
        description="Identity, role and objectives of the AI Service. Gives a general idea of what the service is about."
    )
    context: str = Field(
        description="Context in which the AI Service operates. The company, the sector, the users, the location, etc."
    )
    knowledge_scope: str | None = Field(
        default=None, description="Scope of knowledge and expertise of the AI Service"
    )
    functionalities: str | list[str] | None = Field(
        default=None, description="Functionalities provided by the AI Service"
    )
    principles: str | list[str | Principle] | None = Field(
        default=None,
        description="Principles, rules and guidelines that the AI Service follows when interacting with users",
    )
    website_url: str | None = Field(
        default=None,
        description="The URL of the AI Service website",
    )


class PredefinedResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    trigger: str = Field(
        description="When this scenario occurs or topic is mentioned (e.g., 'user asks for support contact', 'mentions billing issues')"
    )
    response: str = Field(
        description="The specific response to provide in this scenario"
    )


class AIServiceDescriptionV2(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    identity_role: str = Field(
        description="Identity, role and objectives of the AI Service. Gives a general idea of what the service is about."
    )
    context: str = Field(
        description="Context in which the AI Service operates. The company, the sector, the users, the location, etc."
    )
    knowledge_scope: str | None = Field(
        default=None, description="Scope of knowledge and expertise of the AI Service"
    )
    functionalities: str | list[str] | None = Field(
        default=None, description="Functionalities provided by the AI Service"
    )
    constraints: str | list[str] | None = Field(
        default=None,
        description="Things the AI Service should not do, restricted topics, or forbidden actions",
    )
    predefined_responses: str | list[str | PredefinedResponse] | None = Field(
        default=None,
        description="Specific responses for certain scenarios, topics, or triggers. Can be simple strings describing what to respond with, or structured PredefinedResponse objects for more control",
    )
    escalation_criteria: str | list[str] | None = Field(
        default=None,
        description="Situations or conditions when queries should be escalated to human oversight",
    )
    response_guidelines: str | None = Field(
        default=None,
        description="Guidelines for how the AI Service should respond to users (tone, style, format)",
    )
