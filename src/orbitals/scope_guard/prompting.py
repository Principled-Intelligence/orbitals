import json

from pydantic import BaseModel, Field

from ..types import AIServiceDescription
from .modeling import (
    ConversationUserMessage,
    ScopeClass,
    ScopeGuardInput,
)


class ScopeGuardResponseModel(BaseModel):
    evidences: list[str] | None = Field(
        description="Excerpts quoted directly from the AI Service Description that are relevant to understanding whether and how the AI Service can handle the LAST USER MESSAGE.",
        default=None,
    )
    scope_class: ScopeClass = Field(
        description="The classification assigned to the LAST USER MESSAGE based on the evidence extracted from the AI Service Description."
    )


SYSTEM_PROMPT = f"""
You are an expert AI classifier specialized in classifying user queries given the description of an AI service.

Your task is to classify the LAST USER MESSAGE in a conversation between a user and an AI Service. Follow these steps:

1. **Extract Evidence**: Identify and quote specific excerpts from the AI Service Description that are relevant to understanding whether and how the AI Service can handle the LAST USER MESSAGE. Look for:
   - Functionalities that might address the user's query
   - Explicit restrictions or limitations mentioned
   - Knowledge scope boundaries
   - Any relevant service capabilities or constraints

2. **Classify**: Based on the evidence extracted, assign one of the classification classes below to the LAST USER MESSAGE.

## Classification Classes

{ScopeClass.get_classes_manifest()}

## Important Guidelines
* Base your classification EXCLUSIVELY on the AI Service Description provided.
* Pay special attention to RESTRICTIONS, CONSTRAINTS, and LIMITATIONS sections in the AI Service Description.
* Extract evidence first, then use it to inform your classification decision.
* Quote excerpts verbatim from the AI Service Description as evidence.

Output the classification in the following JSON schema:
{json.dumps(ScopeGuardResponseModel.model_json_schema())}
""".strip()


def dump_conversation(conversation: ScopeGuardInput) -> str:
    if isinstance(conversation, str):
        dict_conv = [f"User\n{conversation}"]
    elif isinstance(conversation, ConversationUserMessage):
        dict_conv = [f"User\n{conversation.content}"]
    elif isinstance(conversation, list):
        dict_conv = [
            f"{'User' if message.role == 'user' else 'AI Service'}\n {message.content}"
            for message in conversation
            if message.role == "user" or message.role == "assistant"
        ]
    else:
        raise NotImplementedError

    assert dict_conv[-1].startswith("User")
    dict_conv[-1] = dict_conv[-1].replace("User\n", "LAST USER MESSAGE\n")
    return "\n\n".join(dict_conv)


def prepare_messages(
    conversation: ScopeGuardInput,
    ai_service_description: str | AIServiceDescription,
    skip_evidences: bool,
):
    conversation_dump = dump_conversation(conversation)
    user_input = f"**START OF THE AI SERVICE DESCRIPTION**\n\n{ai_service_description if isinstance(ai_service_description, str) else ai_service_description.model_dump_json()}\n\n**END OF THE AI SERVICE DESCRIPTION**\n\n\n"
    user_input += f"**START OF THE CONVERSATION DUMP**\n\n{conversation_dump}\n\n**END OF THE CONVERSATION DUMP**"
    if skip_evidences:
        user_input += "\n\n**SKIP EVIDENCES**: do not report evidences, report only the scope_class."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    return messages


def build_prompt(
    tokenizer,
    conversation: ScopeGuardInput,
    ai_service_description: str | AIServiceDescription,
    skip_evidences: bool,
    prefill: bool = False,
) -> str:
    messages = prepare_messages(
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
            prompt += '{"evidences": null, "scope_class": "'
        else:
            prompt += '{"evidences":'

    return prompt
