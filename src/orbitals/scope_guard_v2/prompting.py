"""Prompt construction for ScopeGuard V2.

The system prompt here is the one the 2608 "promptfix" models were trained on and
is pinned by sha256 in the tests. Do not edit it in place: a prompt that differs
from training degrades accuracy without any parse error to signal it. A change to
the prompt is a retrain, not a patch.

The selector helpers (`normalize_selection`, `render_selector_block`) and the
per-selection response schema (`response_model_for`) are ported from the trainer's
`src/output_fields.py` and `src/schema.py`. If the field set or a field's type ever
changes there, it must change here in the same release; the selector test with
eight literal strings is what catches the rendering half of that.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..types import AIServiceDescriptionV2, Conversation, ConversationMessage
from .modeling import (
    ConversationUserMessage,
    ScopeClass,
    ScopeGuardV2Input,
    ScopeGuardV2InputTypeAdapter,
)

SCOPE_CLASS = "scope_class"

# `scope_class` is always emitted; only these three are selectable.
OPTIONAL_FIELDS: tuple[str, ...] = ("evidences", "reasoning", "suggested_response")

# Emission order for the selector block and for the JSON object the model returns.
# Matches the key order of the pre-existing four-field output, so a caller who
# asks for everything sees exactly what they saw before.
CANONICAL_ORDER: tuple[str, ...] = (
    "evidences",
    "reasoning",
    SCOPE_CLASS,
    "suggested_response",
)

ALL_FIELDS: tuple[str, ...] = CANONICAL_ORDER

SELECTOR_HEADER = "**REQUESTED OUTPUT FIELDS**"


def normalize_selection(fields: Iterable[str]) -> tuple[str, ...]:
    """Canonicalise an output-field selection.

    Dedupes, forces `scope_class` in, and sorts to canonical order, so two
    selections that differ only in iteration order render byte-identical prompts.

    Args:
        fields: Any iterable of field names. May be empty.

    Returns:
        The selection as a tuple in canonical order, always containing `scope_class`.

    Raises:
        ValueError: If a name is not one of the four known fields.
    """
    requested = set(fields) | {SCOPE_CLASS}
    unknown = sorted(requested - set(CANONICAL_ORDER))
    if unknown:
        raise ValueError(
            f"unknown output field(s) {unknown}; expected a subset of {list(CANONICAL_ORDER)}"
        )
    return tuple(f for f in CANONICAL_ORDER if f in requested)


def render_selector_block(selection: Iterable[str]) -> str:
    """The block appended to the user turn naming the keys the model must emit.

    A JSON array in canonical order, so the selector reads as the literal key list
    of the object the model is being asked to produce.
    """
    keys = json.dumps(list(normalize_selection(selection)))
    return f"{SELECTOR_HEADER}\n\n{keys}"


# Per-field type and description for the wire model of one selection. Kept next to
# the prompt's "Field types" list so the two cannot drift apart unnoticed.
_FIELD_SPECS: dict[str, tuple[type, str]] = {
    "evidences": (
        list[str] | None,
        "Evidences from the AI Service Description supporting this classification.",
    ),
    "reasoning": (str, "A short explanation of why this classification was chosen."),
    SCOPE_CLASS: (
        ScopeClass,
        "The scope classification (must be one of the defined scope classes).",
    ),
    "suggested_response": (
        str | None,
        "A short suggested answer or message for the user.",
    ),
}


def response_model_for(selection: Iterable[str]) -> type[BaseModel]:
    """The wire schema for one output-field selection.

    Every requested field is required (no defaults) and extra keys are forbidden, so
    the generated JSON schema is a precise contract: structured decoding cannot emit
    an unrequested key, and validating a completion against the model is itself the
    check that the model obeyed the selector.

    Args:
        selection: Field names; normalised before lookup, so logically equal
            selections share one cached model.

    Returns:
        A pydantic model class with exactly the selected fields.
    """
    return _response_model_for_normalized(normalize_selection(selection))


@lru_cache(maxsize=None)
def _response_model_for_normalized(selection: tuple[str, ...]) -> type[BaseModel]:
    fields = {
        name: (_FIELD_SPECS[name][0], Field(description=_FIELD_SPECS[name][1]))
        for name in selection
    }
    name = "ScopeGuardV2Response_" + "_".join(selection)
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


_SCOPE_CLASSES_BLOCK = ScopeClass.get_classes_manifest()

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

1. **Consider the Evidence**: Identify specific excerpts from the AI Service Description that are relevant to understanding whether and how the AI Service can handle the LAST MESSAGE. Look for:
   - Functionalities that might address the user's query
   - Constraints that would forbid or restrict the request
   - Knowledge Scope boundaries that the request may fall outside of
   - Predefined Responses whose trigger matches the request
   - Escalation Criteria that the request may meet
   Do this whether or not you are asked to report the evidence — it is how you reach the right class.
2. **Contextualise**: Read the conversation for context. Prior messages provide context only; your classification must reflect the intent of the message tagged LAST MESSAGE alone.
3. **Classify**: Based on the evidence considered, assign exactly one of the scope classes above. If the message could fall under multiple classes, prefer the more specific or more restrictive one (e.g. "Restricted" over "Out of Scope", "Directly Supported" over "Potentially Supported").
   **"Potentially Supported" is unavailable whenever a constraint, escalation criterion, or predefined response matches the request.** When one matches, the class is determined by that rule -- "Restricted", "Human Oversight", or "Predefined Answer" respectively. Difficulty in deciding is not itself a reason to choose "Potentially Supported": if you can name the rule that applies, apply it.
4. **Respond**: The last block of the user turn is `{SELECTOR_HEADER}`, a JSON array naming the keys you must emit. Emit each requested key and no others:
   - `evidences`: verbatim quotes from the AI service description that support your choice (or null if not applicable)
   - `reasoning`: a short, useful explanation of why you chose that class. Prefer few sentences that mention only the decisive evidence or rule. It must be in English, regardless of the language of the user message or service description, to ensure consistency in evaluation.
   - `scope_class`: one of the exact scope class names listed above — always requested, always emitted
   - `suggested_response`: the shortest useful response for the user when the class is "Predefined Answer", "Human Oversight", "Out of Scope", "Restricted", or "Chit Chat"; it must still be meaningful, polite, and convey the required message clearly; otherwise null

## Important Guidelines
- Base your classification EXCLUSIVELY on the AI Service Description provided.
- Give priority to **Predefined Responses** > **Escalation Criteria** > **Constraints** fields — they are hard rules that override other considerations.
- When a **Predefined Response** trigger is matched, always classify as "Predefined Answer" and use the exact pre-written response.
- Consider the evidence first, then use it to inform your classification decision.
- If at least one of the user requests / intents matches a constraint, predefined response, or escalation criterion, classify the request accordingly (respecting classes priorities) and **the suggested_response, when requested, must reflect that, even if other aspects of the query could be considered "Directly Supported" or "Potentially Supported" completely ignore them**.
- Keep `reasoning` and `suggested_response` as short as possible while preserving the meaning. Do not add filler, repeated explanations, or unnecessary detail.
- Requesting fewer fields never changes the class you would have chosen with all of them.

## Language
- The suggested response must be in the same language as the user's message. If the user's message is in a language other than English, translate the predefined response or escalation instructions into that language while preserving the meaning as closely as possible.

## Output Format
You MUST respond with a single JSON object and nothing else — no markdown fences, no preamble, no explanation outside the JSON.

The `{SELECTOR_HEADER}` block at the end of the user turn lists the requested keys. Your JSON object must contain exactly those keys, in the order they are listed there, and no others. A requested key is always present even when its value is null; a key that was not requested must be absent entirely.

Field types:
- "evidences": array of strings (verbatim quotes from the service description), or null if not applicable
- "reasoning": string, concise English explanation
- "scope_class": string, exactly one of the scope class names listed above
- "suggested_response": string with a brief response for the user (when scope_class is "Predefined Answer", "Human Oversight", "Out of Scope", "Restricted", or "Chit Chat"), or null otherwise
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


def _selection_from_skip_evidences(skip_evidences: bool) -> tuple[str, ...]:
    if skip_evidences:
        return tuple(f for f in ALL_FIELDS if f != "evidences")
    return ALL_FIELDS


def resolve_selection(
    output_fields: Iterable[str] | None,
    skip_evidences: bool | None,
) -> tuple[str, ...] | None:
    """Combine the two ways of asking for output fields at one precedence level.

    `output_fields` is the general mechanism; `skip_evidences` is the pre-existing
    convenience meaning "everything except evidences". When both are given and
    disagree, `output_fields` wins and a DeprecationWarning names both values, so
    a caller migrating from one to the other cannot silently get the wrong shape.

    Args:
        output_fields: Field names, or None if not specified at this level.
        skip_evidences: The legacy flag, or None if not specified at this level.

    Returns:
        A normalised selection, or None when neither argument was given -- the
        caller then falls through to the next precedence level (constructor
        defaults, then all fields).
    """
    if output_fields is None and skip_evidences is None:
        return None
    if output_fields is None:
        return _selection_from_skip_evidences(bool(skip_evidences))
    selection = normalize_selection(output_fields)
    if skip_evidences is not None:
        implied = _selection_from_skip_evidences(skip_evidences)
        if implied != selection:
            warnings.warn(
                f"output_fields={list(selection)} and skip_evidences={skip_evidences} "
                f"disagree (skip_evidences implies {list(implied)}); using output_fields. "
                "Pass only one of them.",
                DeprecationWarning,
                stacklevel=3,
            )
    return selection


def prepare_input_messages(
    conversation: ScopeGuardV2Input,
    ai_service_description: AIServiceDescriptionV2 | str,
    skip_evidences: bool | None = None,
    output_fields: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Build the system and user turns for one classification request.

    The system prompt is constant; the per-request field selection rides in the last
    block of the user turn, so the long prefix stays cacheable across requests.

    Args:
        conversation: The conversation or single user message to classify.
        ai_service_description: The service description, structured or free text.
        skip_evidences: Legacy convenience for "all fields except evidences".
        output_fields: The fields the model must emit. `scope_class` is always
            included. Takes precedence over `skip_evidences` if both are given.

    Returns:
        A two-message list suitable for a chat template.
    """
    if isinstance(ai_service_description, AIServiceDescriptionV2):
        ai_service_description = ai_service_description.model_dump_json()

    selection = resolve_selection(output_fields, skip_evidences) or ALL_FIELDS

    _conv = convert_to_conversation(conversation)
    conversation_dump = dumps_conversation(_conv)

    user_input = f"**START OF THE AI SERVICE DESCRIPTION**\n\n{ai_service_description}\n\n**END OF THE AI SERVICE DESCRIPTION**\n\n\n"
    user_input += f"**START OF THE CONVERSATION DUMP**\n\n{conversation_dump}\n\n**END OF THE CONVERSATION DUMP**"
    user_input += f"\n\n\n{render_selector_block(selection)}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]


def build_prompt(
    tokenizer,
    conversation: ScopeGuardV2Input,
    ai_service_description: AIServiceDescriptionV2 | str,
    skip_evidences: bool | None = None,
    prefill: bool = False,
    output_fields: Iterable[str] | None = None,
) -> str:
    """Render the full prompt string for a completion-style backend.

    Args:
        tokenizer: A tokenizer exposing `apply_chat_template`.
        conversation: The conversation or single user message to classify.
        ai_service_description: The service description, structured or free text.
        skip_evidences: Legacy convenience for "all fields except evidences".
        prefill: If True, append the opening of the JSON object up to and including
            the first requested key, so generation starts at its value.
        output_fields: The fields the model must emit; see `prepare_input_messages`.

    Returns:
        The prompt string.
    """
    messages = prepare_input_messages(
        conversation,
        ai_service_description,
        skip_evidences,
        output_fields=output_fields,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    if prefill:
        selection = resolve_selection(output_fields, skip_evidences) or ALL_FIELDS
        prompt += f'{{"{selection[0]}":'

    return prompt
