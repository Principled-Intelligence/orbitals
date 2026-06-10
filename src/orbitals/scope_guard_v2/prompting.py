import json

from pydantic import BaseModel, Field

from ..types import AIServiceDescriptionV2, Conversation, ConversationMessage
from .modeling import (
    ConversationUserMessage,
    ScopeClass,
    ScopeGuardV2Input,
    ScopeGuardV2InputTypeAdapter,
)


class ScopeGuardV2ResponseModel(BaseModel):
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


_SCOPE_CLASSES_BLOCK = ScopeClass.get_classes_manifest()
_RESPONSE_SCHEMA = json.dumps(ScopeGuardV2ResponseModel.model_json_schema())

SYSTEM_PROMPT = f"""You are an expert AI classifier specialized in classifying user queries given the description of an AI service.

Your task is to analyse a conversation and classify the last user message against the AI service description provided in the AI Service Description section below.

## AI Service Description

The AI Service Description may be a structured document with labelled fields, or a free-form text. Either way, when reading it you should identify the following conceptual categories — they may be explicitly labelled or simply implied by the prose:

- **Identity & Role**: What the AI service is and what it is fundamentally meant to do.
- **Context**: The company, sector, user base, or operating environment the service operates in.
- **Knowledge Scope**: The subject-matter domains the service has expertise in. Anything clearly outside this scope is likely Out of Scope.
- **Functionalities**: Specific capabilities or tasks the service can perform. Direct matches are strong evidence for "Directly Supported".
- **Constraints**: Topics, actions, or behaviours the service must never engage in, regardless of how the request is phrased. Matches are strong evidence for "Restricted".
- **Predefined Responses**: Specific triggers or questions that require a fixed, pre-written answer. Matches are strong evidence for "Predefined Answer".
- **Escalation Criteria**: Situations (e.g. legal threats, safety concerns, fraud, but even specific use cases) that must be routed to a human. Matches are strong evidence for "Human Oversight".
- **Response Guidelines**: Tone or style guidance — does not affect scope classification.

## Available scope classes

{_SCOPE_CLASSES_BLOCK}

## Instructions

1. **Extract Evidences**: Identify and quote specific excerpts from the AI Service Description that are relevant to understanding whether and how the AI Service can handle the message tagged LAST MESSAGE. Look for:
   - Functionalities that might address the user's query
   - Constraints that would forbid or restrict the request
   - Knowledge Scope boundaries that the request may fall outside of
   - Predefined Responses whose trigger matches the request
   - Escalation Criteria that the request may meet
2. **Contextualise**: Read the conversation for context. Prior messages provide context only; your classification must reflect the intent of the message tagged LAST MESSAGE alone.
3. **Classify**: Based on the evidence extracted, assign exactly one of the scope classes above. If the message could fall under multiple classes, prefer the more specific or more restrictive one (e.g. "Restricted" over "Out of Scope", "Directly Supported" over "Potentially Supported").
4. **Respond**: Provide:
   - `evidences`: verbatim quotes from the AI service description that support your choice (or null if not applicable)
   - `reasoning`: a short, useful explanation of why you chose that class. Prefer few sentences that mentions only the decisive evidence or rule. It must be in English, regardless of the language of the user message or service description, to ensure consistency in evaluation.
   - `scope_class`: one of the exact scope class names listed above
   - `suggested_response`: the shortest useful response for the user when the class is "Predefined Answer", "Human Oversight", "Out of Scope", "Restricted", or "Chit Chat"; it must still be meaningful, polite, and convey the required message clearly; otherwise null

## Important Guidelines
- Base your classification EXCLUSIVELY on the AI Service Description provided.
- Give priority to **Predefined Responses** > **Escalation Criteria** > **Constraints** fields — they are hard rules that override other considerations.
- When a **Predefined Response** trigger is matched, always classify as "Predefined Answer" and use the exact pre-written response.
- Extract evidence first, then use it to inform your classification decision.
- If at least one of the user requests / intents matches a constraint, predefined response, or escalation criterion, classify the request accordingly (respecting classes priorities) and **the suggested_response must reflect that, even if other aspects of the query could be considered "Directly Supported" or "Potentially Supported" completely ignore them**.
- Keep `reasoning` and `suggested_response` as short as possible while preserving the meaning. Do not add filler, repeated explanations, or unnecessary detail.

## Language
- The suggested response must be in the same language as the user's message. If the user's message is in a language other than English, translate the predefined response or escalation instructions into that language while preserving the meaning as closely as possible.

## Output Format
You MUST respond with a single JSON object and nothing else — no markdown fences, no preamble, no explanation outside the JSON. The JSON must conform to this schema:
{_RESPONSE_SCHEMA}
"""


LAST_MESSAGE_TAG = "LAST MESSAGE"


def dumps_conversation(
    conversation_or_message: Conversation | ConversationMessage | str,
) -> str:
    if isinstance(conversation_or_message, str):
        conversation_or_message = ConversationMessage(
            role="user", content=conversation_or_message
        )
    if isinstance(conversation_or_message, ConversationMessage):
        conversation_or_message = Conversation(messages=[conversation_or_message])

    if len(conversation_or_message.messages) == 0:
        raise ValueError("Conversation must contain at least one message.")

    if conversation_or_message.messages[-1].role != "user":
        raise ValueError(
            "The last message in the conversation must be from the user, representing the LAST USER MESSAGE to classify."
        )

    conversation_dump = ""

    for message in conversation_or_message.messages[:-1]:
        conversation_dump += f"{message.role.upper()}:\n{message.content}\n\n"

    last_message = conversation_or_message.messages[-1]
    conversation_dump += (
        f"{LAST_MESSAGE_TAG} ({last_message.role.upper()}):\n{last_message.content}\n"
    )

    return conversation_dump


def convert_to_conversation(messages: ScopeGuardV2Input) -> Conversation:
    if isinstance(messages, str):
        messages = [ConversationMessage(role="user", content=messages)]
    elif isinstance(messages, list):
        messages = [
            ConversationMessage(role=message.role, content=message.content)
            for message in messages
        ]
    elif isinstance(messages, ConversationUserMessage):
        messages = [ConversationMessage(role=messages.role, content=messages.content)]
    else:
        messages = ScopeGuardV2InputTypeAdapter.validate_python(messages)

    conversation = Conversation(messages=messages)
    return conversation


def prepare_input_messages(
    conversation: ScopeGuardV2Input,
    ai_service_description: AIServiceDescriptionV2 | str,
    skip_evidences: bool = False,
):
    if isinstance(ai_service_description, AIServiceDescriptionV2):
        ai_service_description = ai_service_description.model_dump_json()

    _conv = convert_to_conversation(conversation)
    conversation_dump = dumps_conversation(_conv)

    user_input = f"**START OF THE AI SERVICE DESCRIPTION**\n\n{ai_service_description}\n\n**END OF THE AI SERVICE DESCRIPTION**\n\n\n"
    user_input += f"**START OF THE CONVERSATION DUMP**\n\n{conversation_dump}\n\n**END OF THE CONVERSATION DUMP**"
    if skip_evidences:
        user_input += "\n\n**SKIP EVIDENCES**: do not report evidences, report only reasoning, scope_class, and suggested_response."

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_input},
    ]
    return messages


def build_prompt(
    tokenizer,
    conversation: ScopeGuardV2Input,
    ai_service_description: AIServiceDescriptionV2 | str,
    skip_evidences: bool = False,
    prefill: bool = False,
) -> str:
    messages = prepare_input_messages(
        conversation,
        ai_service_description,
        skip_evidences,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if prefill:
        if skip_evidences:
            prompt += '{"evidences": null, "reasoning": "'
        else:
            prompt += '{"evidences":'

    return prompt
