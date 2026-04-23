import json

from loguru import logger

from transformers import AutoTokenizer

from ..types import AIServiceDescription, Conversation, ConversationMessage
from .modeling import ClaimExtractorInput, Extractions, _parse_raw_output
from .prompting import (
    SYSTEM_PROMPT_IC_EXTRACTION,
    SYSTEM_PROMPT_ICE_EXTRACTION,
    SYSTEM_PROMPT_ICE_EXTRACTION_GUIDED,
)


LAST_MESSAGE_TAG = "LAST MESSAGE"


def dumps_ai_service_description(
    ai_service_description: AIServiceDescription | str,
) -> str:
    if isinstance(ai_service_description, str):
        return ai_service_description
    return ai_service_description.model_dump_json()


def dumps_conversation(
    conversation_or_message: ClaimExtractorInput,
) -> str:
    if isinstance(conversation_or_message, str):
        conversation_or_message = ConversationMessage(
            role="assistant", content=conversation_or_message
        )
    if isinstance(conversation_or_message, ConversationMessage):
        conversation_or_message = Conversation(messages=[conversation_or_message])

    conversation_dump = ""

    for message in conversation_or_message.messages[:-1]:
        conversation_dump += f"{message.role.upper()}:\n{message.content}\n\n"

    last_message = conversation_or_message.messages[-1]
    conversation_dump += (
        f"{LAST_MESSAGE_TAG} ({last_message.role.upper()}):\n{last_message.content}\n"
    )

    return conversation_dump


def convert_to_conversation(messages: str | list[dict[str, str]]) -> Conversation:
    if isinstance(messages, str):
        messages = json.loads(messages)

    conversation = Conversation(
        messages=[
            ConversationMessage(role=message["role"], content=message["content"])
            for message in messages
        ]
    )
    return conversation


def prepare_messages(
    conversation: list[dict] | str,
    ai_service_description: str | AIServiceDescription | None,
    include_evidences: bool = True,
    use_guided_prompt: bool = False,
):
    _conv = convert_to_conversation(conversation)
    conversation_dump = dumps_conversation(_conv)

    if ai_service_description is None:
        ai_service_description = "No AI service description provided."
    user_input = f"**START OF THE AI SERVICE DESCRIPTION**\n\n{dumps_ai_service_description(ai_service_description)}\n\n**END OF THE AI SERVICE DESCRIPTION**\n\n\n"
    user_input += f"**START OF THE CONVERSATION DUMP**\n\n{conversation_dump}\n\n**END OF THE CONVERSATION DUMP**"

    if not include_evidences:
        if use_guided_prompt:
            raise ValueError(
                "use_guided_prompt=True is incompatible with include_evidences=False; "
                "no guided IC prompt is defined."
            )
        system_prompt = SYSTEM_PROMPT_IC_EXTRACTION
    else:
        system_prompt = (
            SYSTEM_PROMPT_ICE_EXTRACTION_GUIDED
            if use_guided_prompt
            else SYSTEM_PROMPT_ICE_EXTRACTION
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_input},
    ]
    return messages


class VLLMFinetunedExtractor:
    def __init__(
        self,
        model_name_or_path: str,
    ) -> None:
        import os

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM, SamplingParams

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            max_model_len=40_000,
            max_num_seqs=2,
        )
        self._sampling_params = SamplingParams(max_tokens=20_000, temperature=0)

    def extract(
        self,
        conversation_or_message: Conversation
        | ConversationMessage
        | list[dict[str, str]]
        | str,
        ai_service_description: str | AIServiceDescription | None = None,
        include_evidences: bool = True,
        use_guided_prompt: bool = False,
    ) -> Extractions:
        if isinstance(conversation_or_message, Conversation):
            messages_list = [
                {"role": m.role, "content": m.content}
                for m in conversation_or_message.messages
            ]
        elif isinstance(conversation_or_message, ConversationMessage):
            messages_list = [
                {
                    "role": conversation_or_message.role,
                    "content": conversation_or_message.content,
                }
            ]
        else:
            messages_list = conversation_or_message

        messages = prepare_messages(
            conversation=messages_list,
            ai_service_description=ai_service_description,
            include_evidences=include_evidences,
            use_guided_prompt=use_guided_prompt,
        )

        logger.debug(f"Prepared messages for claim extraction: {messages}")

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        outputs = self._llm.generate([prompt], self._sampling_params)
        raw_output = outputs[0].outputs[0].text
        extractions = _parse_raw_output(raw_output)
        logger.debug(f"Parsed output for claim extraction: {extractions}")
        return extractions
