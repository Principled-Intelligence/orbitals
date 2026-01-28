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
