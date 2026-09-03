# ScopeGuard V2 Selectable Output Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `orbitals.scope_guard_v2` to the 2608 promptfix prompt and let callers choose which output fields the model emits, without breaking any existing interface.

**Architecture:** The system prompt is replaced by the training prompt and pinned by sha256. A small pure-Python selector module (`normalize_selection`, `render_selector_block`, `response_model_for`) is ported from the trainer into `prompting.py`. One resolver on the base guard turns `output_fields` / `skip_evidences` into a normalized selection; every backend renders the selector into the user turn, requests a per-selection JSON schema, and validates against it.

**Tech Stack:** Python 3.13, pydantic v2, `uv`, pytest (+ pytest-asyncio), FastAPI/TestClient, Typer. No GPU or model download in any test.

**Spec:** `docs/superpowers/specs/2026-09-03-scope-guard-v2-selectable-output-fields-design.md`

## Global Constraints

- `sha256(SYSTEM_PROMPT) == "f0f68e48096938b93c2c497386295ff7db2929b128e9dddea944ebb6578960b5"`, `len == 7665`, trailing `"\n"`. Do not "fix" it to `1f37419f…`; that is the stripped copy.
- Default selection when nothing is passed is all four fields in order `("evidences", "reasoning", "scope_class", "suggested_response")`.
- `scope_class` is always in the selection.
- Every existing public signature keeps every existing parameter, in place, with its current default.
- Run tests with `uv run pytest` from `/home/edobobo/orbitals`. Baseline: `tests/test_scope_guard_v2.py` has 17 passing tests; that number only goes up.
- Google-style docstrings on public functions. Use `uv`, never `pip`.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
  ```
- Work on branch `feat/scope-guard-v2-output-fields`.

---

## File map

| file | responsibility after this plan |
| --- | --- |
| `src/orbitals/scope_guard_v2/prompting.py` | the system prompt; selector constants and helpers; `response_model_for`; user-turn builder; `build_prompt` |
| `src/orbitals/scope_guard_v2/modeling.py` | `ScopeClass` + descriptions (one description changes); `ScopeGuardV2Output` (`reasoning` optional) |
| `src/orbitals/scope_guard_v2/guards/base.py` | public signatures gain `output_fields`; `_resolve_output_fields` |
| `src/orbitals/scope_guard_v2/guards/vllm.py` | offline + HTTP vLLM backends use the selection; `system_prompt.txt` drift warning |
| `src/orbitals/scope_guard_v2/guards/hf.py` | HF pipeline backend passes the selection |
| `src/orbitals/scope_guard_v2/guards/api.py` | wire body carries `output_fields` only when explicit; `reasoning` read with `.get` |
| `src/orbitals/scope_guard_v2/serving/main.py` | body + response model |
| `src/orbitals/scope_guard_v2/cli/serve.py` | `--output-fields` |
| `src/hf_pipeline/scope_guard_v2.py` | remote-code pipeline passes the selection |
| `tests/test_scope_guard_v2.py` | all new tests appended here, matching the file's existing style |
| `README.scope-guard-v2.md`, `pyproject.toml` | docs and version |

---

### Task 1: Selector helpers and the pinned system prompt

**Files:**
- Modify: `src/orbitals/scope_guard_v2/prompting.py` (replace `SYSTEM_PROMPT`, remove `_RESPONSE_SCHEMA`, add helpers)
- Modify: `src/orbitals/scope_guard_v2/modeling.py:57-65` (`_SCOPE_DESCRIPTIONS[POTENTIALLY_SUPPORTED]`)
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Produces: `SCOPE_CLASS: str`, `OPTIONAL_FIELDS: tuple[str, ...]`, `CANONICAL_ORDER: tuple[str, ...]`, `ALL_FIELDS: tuple[str, ...]`, `SELECTOR_HEADER: str`, `normalize_selection(fields: Iterable[str]) -> tuple[str, ...]`, `render_selector_block(selection: Iterable[str]) -> str`, `SYSTEM_PROMPT: str` — all in `orbitals.scope_guard_v2.prompting`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- output fields: selector helpers and the pinned prompt -----------------


def test_scope_guard_v2_system_prompt_is_byte_identical_to_training():
    """The client prompt must be the bytes the 2608 models trained on.

    A diverged system prompt degrades accuracy with no parse error to show for it,
    so drift has to be a red test. The hash is of the trainer's
    src/prompting.SYSTEM_PROMPT, trailing newline included (the chat template puts
    <|im_end|> right after the content, so the newline is part of what the model
    saw). Do NOT change this to 1f37419f...: that is the newline-stripped copy
    unsafe-eval and the Hub system_prompt.txt carry.
    """
    import hashlib

    from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT

    assert len(SYSTEM_PROMPT) == 7665
    assert SYSTEM_PROMPT.endswith("\n")
    assert (
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        == "f0f68e48096938b93c2c497386295ff7db2929b128e9dddea944ebb6578960b5"
    )


def test_scope_guard_v2_system_prompt_has_no_embedded_schema_or_skip_marker():
    from orbitals.scope_guard_v2.prompting import SELECTOR_HEADER, SYSTEM_PROMPT

    assert '"$defs"' not in SYSTEM_PROMPT
    assert "SKIP EVIDENCES" not in SYSTEM_PROMPT
    assert SELECTOR_HEADER in SYSTEM_PROMPT
    assert "exactly those keys" in SYSTEM_PROMPT


def test_normalize_selection_forces_scope_class_dedupes_and_reorders():
    from orbitals.scope_guard_v2.prompting import ALL_FIELDS, normalize_selection

    assert normalize_selection([]) == ("scope_class",)
    assert normalize_selection(["scope_class", "reasoning"]) == (
        "reasoning",
        "scope_class",
    )
    assert normalize_selection(["reasoning", "reasoning", "evidences"]) == (
        "evidences",
        "reasoning",
        "scope_class",
    )
    assert normalize_selection(reversed(ALL_FIELDS)) == ALL_FIELDS


def test_normalize_selection_rejects_unknown_fields():
    from orbitals.scope_guard_v2.prompting import normalize_selection

    with pytest.raises(ValueError, match="unknown output field"):
        normalize_selection(["scope_class", "confidence"])


def test_render_selector_block_matches_the_trainer_for_all_eight_selections():
    """Eight literal strings copied from the trainer's render_selector_block.

    The user turn must be byte-identical to training, and this block is the part
    of it the client composes itself.
    """
    from orbitals.scope_guard_v2.prompting import render_selector_block

    expected = {
        ("scope_class",): '**REQUESTED OUTPUT FIELDS**\n\n["scope_class"]',
        ("evidences", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "scope_class"]',
        ("reasoning", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["reasoning", "scope_class"]',
        ("evidences", "reasoning", "scope_class"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "reasoning", "scope_class"]',
        ("scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["scope_class", "suggested_response"]',
        ("evidences", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "scope_class", "suggested_response"]',
        ("reasoning", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["reasoning", "scope_class", "suggested_response"]',
        ("evidences", "reasoning", "scope_class", "suggested_response"): '**REQUESTED OUTPUT FIELDS**\n\n["evidences", "reasoning", "scope_class", "suggested_response"]',
    }
    for selection, block in expected.items():
        assert render_selector_block(selection) == block, selection
    # order-insensitive on input
    assert render_selector_block(["suggested_response", "scope_class", "evidences"]) == (
        expected[("evidences", "scope_class", "suggested_response")]
    )


def test_potentially_supported_description_is_the_promptfix_text():
    from orbitals.scope_guard_v2 import ScopeClass

    desc = ScopeClass.POTENTIALLY_SUPPORTED.description
    assert desc.startswith("The query is adjacent to the service's stated functionalities")
    assert "never as a way to avoid committing to a clearer class" in desc
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "system_prompt or normalize_selection or render_selector or potentially_supported"`
Expected: FAIL — `ImportError: cannot import name 'normalize_selection'` and the hash assertion.

- [ ] **Step 3: Update the `Potentially Supported` description**

In `src/orbitals/scope_guard_v2/modeling.py`, replace the `ScopeClass.POTENTIALLY_SUPPORTED` entry of `_SCOPE_DESCRIPTIONS` with exactly:

```python
    ScopeClass.POTENTIALLY_SUPPORTED: "The query is adjacent to the service's stated functionalities or knowledge scope without being named by any of them, and no constraint, escalation criterion, or predefined response applies to it. Use it only when the service could handle the request as a reasonable extension of what it already does -- never as a way to avoid committing to a clearer class.",
```

(Two ASCII hyphens before "never". The other six descriptions are already identical to the trainer's and stay as they are.)

- [ ] **Step 4: Replace the prompt and add the helpers in `prompting.py`**

Replace everything from the top of `src/orbitals/scope_guard_v2/prompting.py` down to (and including) the `SYSTEM_PROMPT = f"""..."""` assignment with the following. Keep `LAST_MESSAGE_TAG`, `dumps_conversation`, `convert_to_conversation`, `prepare_input_messages`, `build_prompt` below it for now (Task 3 rewrites the last two).

```python
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
```

Delete the old `class ScopeGuardV2ResponseModel(BaseModel): ...` and `_RESPONSE_SCHEMA = ...` lines. (Task 2 adds `response_model_for`; the `BaseModel`/`ConfigDict`/`Field`/`create_model`/`lru_cache` imports above are for it. Leave them in now so Task 2 is a pure addition.)

- [ ] **Step 5: Run the tests**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "system_prompt or normalize_selection or render_selector or potentially_supported"`
Expected: 6 PASS. If the hash test fails, diff against the trainer: `cd /home/edobobo/sg2-trainer && uv run python -c "import sys; sys.path.insert(0,'.'); from src.prompting import SYSTEM_PROMPT; open('/tmp/train_prompt.txt','w').write(SYSTEM_PROMPT)"` then `cd /home/edobobo/orbitals && uv run python -c "from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT; open('/tmp/orb_prompt.txt','w').write(SYSTEM_PROMPT)" && diff /tmp/train_prompt.txt /tmp/orb_prompt.txt`. Fix the prompt, never the hash.

Other tests will now fail with `ImportError: cannot import name 'ScopeGuardV2ResponseModel'` from `hf.py`/`vllm.py`. That is expected until Task 2.

- [ ] **Step 6: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/prompting.py src/orbitals/scope_guard_v2/modeling.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): pin the 2608 promptfix system prompt and add selector helpers

Replaces the 2606 prompt with the one the 2608 models trained on, byte-identical
and pinned by sha256 (trailing newline included -- the chat template puts
<|im_end|> right after the content). Updates the Potentially Supported description
the prompt interpolates. Adds normalize_selection and render_selector_block, ported
from the trainer, with the eight selector renderings pinned as literals.

Temporarily breaks hf.py/vllm.py imports of ScopeGuardV2ResponseModel; the next
commit replaces it with a per-selection model.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 2: Per-selection response model; `reasoning` optional on the output

**Files:**
- Modify: `src/orbitals/scope_guard_v2/prompting.py` (add `response_model_for`)
- Modify: `src/orbitals/scope_guard_v2/modeling.py:88-104` (`ScopeGuardV2Output.reasoning`)
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `normalize_selection` (Task 1).
- Produces: `response_model_for(selection: Iterable[str]) -> type[pydantic.BaseModel]` in `prompting`; `ScopeGuardV2Output.reasoning: str | None = None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- per-selection response model -----------------------------------------


def test_response_model_for_requires_exactly_the_selected_keys():
    from orbitals.scope_guard_v2.prompting import response_model_for

    model = response_model_for(["scope_class"])
    parsed = model.model_validate({"scope_class": "Restricted"})
    assert parsed.scope_class == "Restricted"

    with pytest.raises(ValidationError):  # unrequested key is forbidden
        model.model_validate({"scope_class": "Restricted", "reasoning": "because"})

    full = response_model_for(["evidences", "reasoning", "scope_class", "suggested_response"])
    with pytest.raises(ValidationError):  # requested key is required, even if nullable
        full.model_validate({"reasoning": "r", "scope_class": "Restricted", "suggested_response": None})
    ok = full.model_validate(
        {"evidences": None, "reasoning": "r", "scope_class": "Restricted", "suggested_response": None}
    )
    assert ok.evidences is None


def test_response_model_for_is_cached_per_normalized_selection():
    from orbitals.scope_guard_v2.prompting import response_model_for

    a = response_model_for(["reasoning", "scope_class"])
    b = response_model_for(["scope_class", "reasoning", "reasoning"])
    assert a is b


def test_response_model_for_schema_lists_only_selected_properties():
    from orbitals.scope_guard_v2.prompting import response_model_for

    schema = response_model_for(["reasoning", "scope_class"]).model_json_schema()
    assert list(schema["properties"]) == ["reasoning", "scope_class"]
    assert set(schema["required"]) == {"reasoning", "scope_class"}
    assert schema.get("additionalProperties") is False


def test_scope_guard_v2_output_reasoning_is_optional():
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2Output

    out = ScopeGuardV2Output(scope_class=ScopeClass.CHIT_CHAT, model="m")
    assert out.reasoning is None
    assert out.evidences is None
    assert out.suggested_response is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "response_model_for or reasoning_is_optional"`
Expected: FAIL — `ImportError: cannot import name 'response_model_for'`; `ValidationError` on the `reasoning`-less output.

- [ ] **Step 3: Add `response_model_for` to `prompting.py`**

Insert directly after `render_selector_block`:

```python
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
```

- [ ] **Step 4: Make `reasoning` optional on `ScopeGuardV2Output`**

In `src/orbitals/scope_guard_v2/modeling.py`, change the `reasoning` field of `ScopeGuardV2Output` to:

```python
    reasoning: str | None = Field(
        default=None,
        description=(
            "A short explanation of why this classification was chosen. None only "
            "when the caller requested an output-field selection without `reasoning`."
        ),
    )
```

- [ ] **Step 5: Point `hf.py` and `vllm.py` at the new model so the package imports again**

This is a temporary shim so the suite can run; Tasks 5 and 6 rewrite these backends properly. In `src/orbitals/scope_guard_v2/guards/hf.py` change the import line

```python
from ..prompting import ScopeGuardV2ResponseModel
```
to
```python
from ..prompting import ALL_FIELDS, response_model_for
```
and replace both occurrences of `ScopeGuardV2ResponseModel.model_validate(` with `response_model_for(ALL_FIELDS).model_validate(`.

In `src/orbitals/scope_guard_v2/guards/vllm.py` change

```python
from ..prompting import SYSTEM_PROMPT, ScopeGuardV2ResponseModel, build_prompt
```
to
```python
from ..prompting import ALL_FIELDS, SYSTEM_PROMPT, build_prompt, response_model_for
```
and replace `ScopeGuardV2ResponseModel.model_validate(` (two places) with `response_model_for(ALL_FIELDS).model_validate(`, and `ScopeGuardV2ResponseModel.model_json_schema()` (one place) with `response_model_for(ALL_FIELDS).model_json_schema()`.

- [ ] **Step 6: Run the whole V2 suite**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q`
Expected: all PASS (17 original + 6 from Task 1 + 4 from this task = 27).

- [ ] **Step 7: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/prompting.py src/orbitals/scope_guard_v2/modeling.py src/orbitals/scope_guard_v2/guards/hf.py src/orbitals/scope_guard_v2/guards/vllm.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): per-selection response model; reasoning becomes optional

response_model_for(selection) builds and caches a pydantic model with exactly the
requested keys, all required, extra forbidden. Its JSON schema is what structured
decoding will be given, so an unrequested key cannot be emitted, and validating
against it is the check that the model obeyed the selector.

ScopeGuardV2Output.reasoning is now str | None, None only when a caller asked for
a selection without it. Every existing read of .reasoning keeps working.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 3: User turn carries the selector; `build_prompt` prefill follows the selection

**Files:**
- Modify: `src/orbitals/scope_guard_v2/prompting.py` (`prepare_input_messages`, `build_prompt`)
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `render_selector_block`, `normalize_selection`, `ALL_FIELDS`, `OPTIONAL_FIELDS`.
- Produces:
  - `resolve_selection(output_fields: Iterable[str] | None, skip_evidences: bool | None) -> tuple[str, ...] | None` — returns None when both are None (so callers can fall through to a lower-precedence level), else a normalized selection; warns on conflict.
  - `prepare_input_messages(conversation, ai_service_description, skip_evidences: bool | None = None, output_fields: Iterable[str] | None = None) -> list[dict]`
  - `build_prompt(tokenizer, conversation, ai_service_description, skip_evidences: bool | None = None, prefill: bool = False, output_fields: Iterable[str] | None = None) -> str`

Note the parameter order: `skip_evidences` keeps its existing third position in both functions (callers pass it positionally in `hf_pipeline`), and `output_fields` is appended last.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- user turn and selection resolution -------------------------------------


def _user_turn(**kwargs) -> str:
    from orbitals.scope_guard_v2.prompting import prepare_input_messages

    messages = prepare_input_messages("hello", "desc", **kwargs)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    return messages[1]["content"]


def test_prepare_input_messages_default_requests_all_four_fields():
    turn = _user_turn()
    assert turn.endswith(
        '**END OF THE CONVERSATION DUMP**\n\n\n**REQUESTED OUTPUT FIELDS**\n\n'
        '["evidences", "reasoning", "scope_class", "suggested_response"]'
    )
    assert "SKIP EVIDENCES" not in turn


def test_prepare_input_messages_skip_evidences_drops_only_evidences():
    turn = _user_turn(skip_evidences=True)
    assert turn.endswith('["reasoning", "scope_class", "suggested_response"]')
    assert _user_turn(skip_evidences=False) == _user_turn()


def test_prepare_input_messages_output_fields_selects_the_keys():
    turn = _user_turn(output_fields=["scope_class"])
    assert turn.endswith('**REQUESTED OUTPUT FIELDS**\n\n["scope_class"]')


def test_prepare_input_messages_conflicting_flags_warn_and_output_fields_wins():
    with pytest.warns(DeprecationWarning, match="output_fields"):
        turn = _user_turn(output_fields=["evidences", "scope_class"], skip_evidences=True)
    assert turn.endswith('["evidences", "scope_class"]')


def test_prepare_input_messages_agreeing_flags_do_not_warn(recwarn):
    turn = _user_turn(output_fields=["reasoning", "scope_class"], skip_evidences=True)
    assert turn.endswith('["reasoning", "scope_class"]')
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_resolve_selection_returns_none_when_nothing_was_passed():
    from orbitals.scope_guard_v2.prompting import resolve_selection

    assert resolve_selection(None, None) is None
    assert resolve_selection(None, True) == ("reasoning", "scope_class", "suggested_response")
    assert resolve_selection(None, False) == (
        "evidences", "reasoning", "scope_class", "suggested_response",
    )
    assert resolve_selection(["scope_class", "evidences"], None) == ("evidences", "scope_class")


def test_build_prompt_prefill_uses_the_selections_first_key():
    from orbitals.scope_guard_v2.prompting import build_prompt

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            return "<prompt>"

    assert build_prompt(_Tok(), "hello", "desc", prefill=True).endswith('{"evidences":')
    assert build_prompt(_Tok(), "hello", "desc", skip_evidences=True, prefill=True).endswith(
        '{"reasoning":'
    )
    assert build_prompt(
        _Tok(), "hello", "desc", prefill=True, output_fields=["scope_class"]
    ).endswith('{"scope_class":')
    assert build_prompt(_Tok(), "hello", "desc") == "<prompt>"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "prepare_input_messages or resolve_selection or prefill"`
Expected: FAIL — `TypeError: unexpected keyword argument 'output_fields'`, `ImportError` for `resolve_selection`, and the default turn still ends with `**END OF THE CONVERSATION DUMP**`.

- [ ] **Step 3: Implement `resolve_selection`, rewrite `prepare_input_messages` and `build_prompt`**

In `src/orbitals/scope_guard_v2/prompting.py`, add `import warnings` to the imports, then replace the existing `prepare_input_messages` and `build_prompt` with:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q`
Expected: all PASS (27 + 8 = 35).

- [ ] **Step 5: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/prompting.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): user turn carries the requested-fields selector

prepare_input_messages and build_prompt accept output_fields alongside the
existing skip_evidences. resolve_selection combines them at one precedence level:
output_fields wins, and a disagreeing pair raises a DeprecationWarning naming both
values. Nothing passed means all four fields, so existing callers render the same
key set they did before -- now announced in the selector block the 2608 models
expect instead of the SKIP EVIDENCES marker they never saw.

Prefill follows the selection's first key rather than hardcoding "evidences".

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 4: Base guard — `output_fields` on the public signatures and one resolver

**Files:**
- Modify: `src/orbitals/scope_guard_v2/guards/base.py`
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `resolve_selection`, `ALL_FIELDS` from `prompting`.
- Produces on `BaseScopeGuardV2`:
  - `__init__(self, backend, *args, include_default_safety_principles=False, skip_evidences: bool | None = None, output_fields: Iterable[str] | None = None, **kwargs)` storing `self.skip_evidences`, `self.output_fields`.
  - `_resolve_output_fields(self, output_fields, skip_evidences) -> tuple[str, ...]` — per-call level first, then constructor level, then `ALL_FIELDS`. Never returns None.
- `validate` / `batch_validate` (sync and async) gain keyword `output_fields: Iterable[str] | None = None` and forward both it and `skip_evidences` unchanged to `_validate` / `_batch_validate`. Backends call `self._resolve_output_fields(...)`.
- Every `__new__` overload gains `output_fields: Sequence[str] | None = None` after `skip_evidences`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- base guard resolution --------------------------------------------------


def _stub_guard(**ctor):
    """A ScopeGuardV2 with no backend, exposing what _batch_validate received."""
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2, ScopeGuardV2Output
    from orbitals.scope_guard_v2.guards.base import BaseScopeGuardV2

    class _Stub(ScopeGuardV2):
        def __new__(cls, *args, **kwargs):
            return BaseScopeGuardV2.__new__(cls)

        def __init__(self, **kwargs):
            super().__init__("stub", **kwargs)
            self.seen: list[tuple[str, ...]] = []

        def _batch_validate(self, conversations, *, skip_evidences=None, output_fields=None, **kwargs):
            selection = self._resolve_output_fields(output_fields, skip_evidences)
            self.seen.append(selection)
            return [
                ScopeGuardV2Output(scope_class=ScopeClass.CHIT_CHAT, model="stub")
                for _ in conversations
            ]

    return _Stub(**ctor)


def test_base_guard_resolution_order():
    all_four = ("evidences", "reasoning", "scope_class", "suggested_response")
    no_evid = ("reasoning", "scope_class", "suggested_response")

    g = _stub_guard()
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == all_four  # nothing anywhere -> all fields

    g = _stub_guard(skip_evidences=True)
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == no_evid  # constructor skip_evidences
    g.validate("q", ai_service_description="d", skip_evidences=False)
    assert g.seen[-1] == all_four  # per-call skip_evidences overrides constructor
    g.validate("q", ai_service_description="d", output_fields=["scope_class"])
    assert g.seen[-1] == ("scope_class",)  # per-call output_fields overrides everything

    g = _stub_guard(output_fields=["reasoning", "scope_class"])
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == ("reasoning", "scope_class")  # constructor output_fields
    g.validate("q", ai_service_description="d", skip_evidences=True)
    assert g.seen[-1] == no_evid  # per-call skip_evidences beats constructor output_fields
    g.batch_validate(["a", "b"], ai_service_description="d", output_fields=["scope_class"])
    assert g.seen[-1] == ("scope_class",)


def test_base_guard_constructor_conflict_warns_once_at_construction():
    with pytest.warns(DeprecationWarning, match="output_fields"):
        g = _stub_guard(output_fields=["evidences", "scope_class"], skip_evidences=True)
    g.validate("q", ai_service_description="d")
    assert g.seen[-1] == ("evidences", "scope_class")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "base_guard"`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'skip_evidences'` / `'output_fields'`.

- [ ] **Step 3: Update `base.py`**

At the top of `src/orbitals/scope_guard_v2/guards/base.py`, extend the typing import to `from typing import TYPE_CHECKING, Iterable, Literal, Sequence, overload` and add:

```python
from ..prompting import ALL_FIELDS, resolve_selection
```

Replace `BaseScopeGuardV2.__init__` with:

```python
    def __init__(
        self,
        backend: str,
        *args,
        include_default_safety_principles: bool = False,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ):
        self.backend = backend
        self.include_default_safety_principles = include_default_safety_principles
        # Resolved once here so a conflicting constructor pair warns at
        # construction, not on every call.
        self.skip_evidences = skip_evidences
        self.output_fields: tuple[str, ...] | None = resolve_selection(
            output_fields, skip_evidences
        )

    def _resolve_output_fields(
        self,
        output_fields: Iterable[str] | None,
        skip_evidences: bool | None,
    ) -> tuple[str, ...]:
        """The effective field selection for one call.

        Per-call arguments win over constructor arguments; within a level
        `output_fields` wins over `skip_evidences` (see `resolve_selection`).
        Falls back to all four fields, which is what every caller got before
        selections existed.
        """
        per_call = resolve_selection(output_fields, skip_evidences)
        if per_call is not None:
            return per_call
        if self.output_fields is not None:
            return self.output_fields
        return ALL_FIELDS
```

Then, in **each** of the five `__new__` overloads (`ScopeGuardV2`: vllm, hf, api; `AsyncScopeGuardV2`: vllm-api, api), add `output_fields: Sequence[str] | None = None,` on the line directly after `skip_evidences: bool = False,`.

In `ScopeGuardV2.validate`, `ScopeGuardV2.batch_validate`, `AsyncScopeGuardV2.validate`, `AsyncScopeGuardV2.batch_validate`: add the keyword parameter `output_fields: Iterable[str] | None = None,` directly after `skip_evidences: bool | None = None,`, and add `output_fields=output_fields,` to the `self._validate(...)` / `self._batch_validate(...)` call directly after `skip_evidences=skip_evidences,`.

In the four abstract `_validate` / `_batch_validate` signatures add `output_fields: Iterable[str] | None = None,` after `skip_evidences`.

- [ ] **Step 4: Run the tests**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q`
Expected: all PASS (35 + 2 = 37). The pre-existing `test_scope_guard_v2_batch_validate_invariants` stub sets `self.backend` without calling `super().__init__`; it must still pass — `_resolve_output_fields` is not called there.

- [ ] **Step 5: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/guards/base.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): output_fields on the guard interface, one resolver for all backends

validate/batch_validate (sync and async) and every constructor overload accept
output_fields next to the existing skip_evidences. BaseScopeGuardV2 resolves the
effective selection in one place -- per-call over constructor, output_fields over
skip_evidences, all four fields as the floor -- so five backends cannot disagree
about what a given pair of arguments means.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 5: vLLM backends — selection-aware prompts, per-selection structured output, prompt-drift warning

**Files:**
- Modify: `src/orbitals/scope_guard_v2/guards/vllm.py`
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `build_prompt(..., output_fields=)`, `response_model_for`, `self._resolve_output_fields`, `SYSTEM_PROMPT`.
- Produces: `check_shipped_system_prompt(model_path: str) -> None` (module-level, in `vllm.py`) — warns via `logging.warning` if `<model_path>/system_prompt.txt` exists and differs from `SYSTEM_PROMPT` after `rstrip("\n")` on both sides (the shipped copies are known to lack the trailing newline; that alone is not drift worth a warning).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- vllm backends -----------------------------------------------------------


def test_check_shipped_system_prompt_warns_only_on_real_drift(tmp_path, caplog):
    import logging

    from orbitals.scope_guard_v2.guards.vllm import check_shipped_system_prompt
    from orbitals.scope_guard_v2.prompting import SYSTEM_PROMPT

    # no file: silent
    check_shipped_system_prompt(str(tmp_path))
    # same bytes minus trailing newline (what the Hub ships): silent
    (tmp_path / "system_prompt.txt").write_text(SYSTEM_PROMPT.rstrip("\n"), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        check_shipped_system_prompt(str(tmp_path))
    assert not caplog.records
    # a different prompt generation: warns and names both hashes
    (tmp_path / "system_prompt.txt").write_text("You are a 2606 classifier.", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        check_shipped_system_prompt(str(tmp_path))
    assert any("system_prompt.txt" in r.message and "f0f68e48" in r.message for r in caplog.records)


async def test_async_vllm_api_backend_sends_selection_schema_and_parses_partial_output(monkeypatch):
    """The HTTP vLLM backend must ask for exactly the selected keys and accept a
    completion that contains only them."""
    from orbitals.scope_guard_v2 import AsyncScopeGuardV2, ScopeClass

    captured: dict[str, Any] = {}

    class _Tok:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            return "PROMPT"

        def encode(self, text):
            return [0] * 10

    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.vllm._get_tokenizer", lambda name: _Tok()
    )

    payload = {
        "choices": [{"text": '{"scope_class": "Out of Scope"}'}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
    }

    # vllm.py uses `async with session.post(...) as response`, unlike api.py which
    # awaits it, so the fake must return an async context manager, not a coroutine.
    class _PostCtx:
        async def __aenter__(self):
            return _FakeAiohttpResponse(payload)

        async def __aexit__(self, *exc_info):
            return None

    class _FakeVllmSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

        def post(self, url, *, json, headers):
            captured["url"] = url
            captured["json"] = json
            return _PostCtx()

    def _session_factory():
        return _FakeVllmSession()

    monkeypatch.setattr(
        "orbitals.scope_guard_v2.guards.vllm.aiohttp.ClientSession", _session_factory
    )

    sg = AsyncScopeGuardV2(backend="vllm-api", model="m", vllm_serving_url="http://x")
    result = await sg.validate(
        "hello", ai_service_description="desc", output_fields=["scope_class"]
    )

    body = captured["json"]
    assert body["prompt"] == "PROMPT"
    assert list(body["structured_outputs"]["json"]["properties"]) == ["scope_class"]
    assert captured["messages"][1]["content"].endswith('["scope_class"]')
    assert result.scope_class == ScopeClass.OUT_OF_SCOPE
    assert result.reasoning is None
    assert result.evidences is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "shipped_system_prompt or async_vllm_api"`
Expected: FAIL — `ImportError: cannot import name 'check_shipped_system_prompt'`; the second test fails on `structured_outputs` listing four properties and on `ValidationError` for missing `reasoning`.

- [ ] **Step 3: Rewrite `vllm.py`**

Replace the file's imports and both classes with the following (keep `_get_tokenizer` as it is):

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import aiohttp
import pydantic

if TYPE_CHECKING:
    import transformers  # noqa: F401
    import vllm  # noqa: F401

from ...types import AIServiceDescriptionV2, LLMUsage
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import SYSTEM_PROMPT, build_prompt, response_model_for
from .base import AsyncScopeGuardV2, ScopeGuardV2

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_tokenizer(model_name: str) -> transformers.PreTrainedTokenizer:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(model_name)


def check_shipped_system_prompt(model_path: str) -> None:
    """Warn if the model directory ships a system prompt that is not ours.

    The 2608 releases include `system_prompt.txt`. The library never reads it to
    build prompts -- the built-in `SYSTEM_PROMPT` is the source of truth -- but a
    mismatch means the model was trained on a different prompt generation and will
    degrade quietly. Trailing newlines are ignored: the shipped copies are known to
    lack the one the model trained with, and that alone is not worth a warning.

    Args:
        model_path: A local model directory. Hub ids and missing files are ignored.
    """
    path = Path(model_path) / "system_prompt.txt"
    if not path.is_file():
        return
    shipped = path.read_text(encoding="utf-8").rstrip("\n")
    ours = SYSTEM_PROMPT.rstrip("\n")
    if shipped == ours:
        return
    logger.warning(
        "system_prompt.txt in %s (sha256 %s) differs from the prompt this library "
        "was built for (sha256 %s). The model was likely trained on a different "
        "prompt generation; classifications may be degraded.",
        model_path,
        hashlib.sha256(shipped.encode("utf-8")).hexdigest()[:8],
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:8],
    )


def _to_output(
    validated: pydantic.BaseModel, model: str, usage: LLMUsage | None
) -> ScopeGuardV2Output:
    """Lift a per-selection response onto the public output; absent fields are None."""
    data = validated.model_dump()
    return ScopeGuardV2Output(
        evidences=data.get("evidences"),
        reasoning=data.get("reasoning"),
        scope_class=data["scope_class"],
        suggested_response=data.get("suggested_response"),
        model=model,
        usage=usage,
    )


@ScopeGuardV2.register_guard("vllm")
class VLLMScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm"] = "vllm",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_model_len: int = 30_000,
        max_num_seqs: int = 2,
        gpu_memory_utilization: float = 0.9,
        include_default_safety_principles: bool = False,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        import vllm

        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        check_shipped_system_prompt(self.model)
        self.llm = vllm.LLM(
            model=self.model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = _get_tokenizer(self.model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        return self._batch_validate(
            [conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )[0]

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        import vllm

        selection = self._resolve_output_fields(output_fields, skip_evidences)
        response_model = response_model_for(selection)

        if ai_service_descriptions is not None:
            pairs = list(zip(conversations, ai_service_descriptions))
        elif ai_service_description is not None:
            pairs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError("an AI service description is required")

        prompts = [
            build_prompt(self.tokenizer, c, ad, output_fields=selection) for c, ad in pairs
        ]
        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            structured_outputs=vllm.sampling_params.StructuredOutputsParams(
                json=response_model.model_json_schema()
            ),
        )
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            validated = response_model.model_validate(json.loads(text))
            results.append(_to_output(validated, self.model, usage=None))
        return results


@AsyncScopeGuardV2.register_guard("vllm-api")
class AsyncVLLMApiScopeGuardV2(AsyncScopeGuardV2):
    def __init__(
        self,
        backend: Literal["vllm-api", "vllm-async-api"] = "vllm-api",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        vllm_serving_url: str = "http://localhost:8000",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        chat_templating_tokenizer: str | None = None,
        count_system_prompt_in_usage: bool = False,
        include_default_safety_principles: bool = False,
    ):
        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for AsyncScopeGuardV2.")
        self.default_model_name = model
        self.default_tokenizer_name = (
            chat_templating_tokenizer
            if chat_templating_tokenizer is not None
            else self.default_model_name
        )
        self.vllm_serving_url = vllm_serving_url
        self.vllm_temperature = temperature
        self.vllm_max_tokens = max_tokens
        self.count_system_prompt_in_usage = count_system_prompt_in_usage

    async def _handle_request(
        self,
        model_name: str | None,
        conversation: ScopeGuardV2Input,
        ai_service_description: str | AIServiceDescriptionV2,
        selection: tuple[str, ...],
        prefill: bool,
        chat_templating_tokenizer: str | None = None,
    ) -> ScopeGuardV2Output:
        if chat_templating_tokenizer is not None:
            tokenizer = _get_tokenizer(chat_templating_tokenizer)
        elif model_name is not None:
            tokenizer = _get_tokenizer(model_name)
        else:
            tokenizer = _get_tokenizer(self.default_tokenizer_name)

        model_name = model_name if model_name is not None else self.default_model_name
        response_model = response_model_for(selection)

        prompt = build_prompt(
            tokenizer=tokenizer,
            conversation=conversation,
            ai_service_description=ai_service_description,
            prefill=prefill,
            output_fields=selection,
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_serving_url}/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": self.vllm_temperature,
                    "max_tokens": self.vllm_max_tokens,
                    "structured_outputs": {"json": response_model.model_json_schema()},
                },
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                response_json = await response.json()
                response_text = response_json["choices"][0]["text"]

        if prefill:
            response_text = prompt[prompt.rindex('{"') :] + response_text

        try:
            parsed_obj = json.loads(response_text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse generated text: {response_json}")

        try:
            validated = response_model.model_validate(parsed_obj)
        except pydantic.ValidationError as e:
            raise ValueError(f"Failed to validate generated text: {e}")

        system_prompt_tokens = (
            0
            if self.count_system_prompt_in_usage
            else len(tokenizer.encode(SYSTEM_PROMPT))
        )
        usage = LLMUsage(
            prompt_tokens=response_json["usage"]["prompt_tokens"] - system_prompt_tokens,
            completion_tokens=response_json["usage"]["completion_tokens"],
            total_tokens=response_json["usage"]["total_tokens"] - system_prompt_tokens,
        )
        return _to_output(validated, model_name, usage)

    async def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        results = await self._batch_validate(
            conversations=[conversation],
            ai_service_description=ai_service_description,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
            model=model,
            chat_templating_tokenizer=chat_templating_tokenizer,
        )
        return results[0]

    async def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        model: str | None = None,
        chat_templating_tokenizer: str | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        if ai_service_description is not None:
            ai_service_descriptions = [ai_service_description] * len(conversations)  # type: ignore[invalid-assignment]

        tasks = [
            self._handle_request(
                model_name=model,
                conversation=c,
                ai_service_description=aisd,
                selection=selection,
                # Grammar-constrained decoding is unaware of a prefilled prefix, so
                # prefill stays off on this backend.
                prefill=False,
                chat_templating_tokenizer=chat_templating_tokenizer,
            )
            for c, aisd in zip(conversations, ai_service_descriptions)  # type: ignore[invalid-argument-type]
        ]
        return await asyncio.gather(*tasks)
```

Note: the offline `vllm` backend now passes `structured_outputs` too. `vllm.sampling_params.StructuredOutputsParams` exists from vLLM 0.19 (the `pyproject` floor is `>=0.19.1`); it is imported lazily inside the method so the module still imports without vLLM installed.

- [ ] **Step 4: Run the whole suite**

Run: `cd /home/edobobo/orbitals && uv run pytest -q`
Expected: all PASS (37 + 2 = 39 in the V2 file; other files unchanged).

- [ ] **Step 5: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/guards/vllm.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): vLLM backends render the selector and constrain decoding to it

Both vLLM backends resolve the selection through the base guard, render it into
the user turn, hand the per-selection JSON schema to structured decoding, and
validate the completion against the same model, so the key set is right by
construction and checked anyway. The offline backend gains structured outputs
it never had.

The offline backend also warns at construction if the model directory ships a
system_prompt.txt from a different prompt generation. Trailing newlines are
ignored in that comparison: the Hub copies lack the one the model trained with,
which is a known artefact and not drift.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 6: HF backend and the remote-code pipeline

**Files:**
- Modify: `src/orbitals/scope_guard_v2/guards/hf.py`
- Modify: `src/hf_pipeline/scope_guard_v2.py`
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `response_model_for`, `_resolve_output_fields`, `prepare_input_messages(..., output_fields=)`, `_to_output` pattern (re-implemented locally; `hf.py` must not import from `vllm.py`).
- Produces: `HuggingFaceScopeGuardV2.__init__(..., skip_evidences: bool | None = None, output_fields: Iterable[str] | None = None, ...)`; the pipeline accepts `output_fields` in `__init__` and per call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- hf backend -------------------------------------------------------------


def test_hf_backend_passes_selection_to_the_pipeline_and_accepts_partial_output(monkeypatch):
    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2

    seen: dict[str, Any] = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def __call__(self, inputs, **kwargs):
            seen["call"] = kwargs
            single = isinstance(inputs, tuple)
            out = [{"generated_text": '{"reasoning": "r", "scope_class": "Chit Chat"}'}]
            return out if single else [out for _ in inputs]

    monkeypatch.setattr("orbitals.utils.maybe_configure_gpu_usage", lambda: None)
    import transformers

    monkeypatch.setattr(transformers, "pipeline", lambda **kwargs: _FakePipeline(**kwargs))

    sg = ScopeGuardV2(backend="hf", model="m", output_fields=["reasoning", "scope_class"])
    assert seen["init"]["output_fields"] == ("reasoning", "scope_class")

    result = sg.validate("hello", ai_service_description="desc")
    assert seen["call"]["output_fields"] == ("reasoning", "scope_class")
    assert result.scope_class == ScopeClass.CHIT_CHAT
    assert result.reasoning == "r"
    assert result.evidences is None

    results = sg.batch_validate(["a", "b"], ai_service_description="desc", output_fields=["scope_class"])
    assert seen["call"]["output_fields"] == ("scope_class",)
    assert len(results) == 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k hf_backend`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'output_fields'`.

- [ ] **Step 3: Rewrite `hf.py`**

Replace the file with:

```python
import json
from typing import TYPE_CHECKING, Iterable, Literal

import pydantic

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescriptionV2
from ..modeling import ScopeGuardV2Input, ScopeGuardV2Output
from ..prompting import response_model_for
from .base import ScopeGuardV2


def _parse(generated_text: str, selection: tuple[str, ...], model: str) -> ScopeGuardV2Output:
    try:
        parsed_obj = json.loads(generated_text)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse generated text: {generated_text}")
    try:
        validated = response_model_for(selection).model_validate(parsed_obj)
    except pydantic.ValidationError as e:
        raise ValueError(f"Failed to validate generated text: {e}")
    data = validated.model_dump()
    return ScopeGuardV2Output(
        evidences=data.get("evidences"),
        reasoning=data.get("reasoning"),
        scope_class=data["scope_class"],
        suggested_response=data.get("suggested_response"),
        model=model,
        usage=None,
    )


@ScopeGuardV2.register_guard("hf")
class HuggingFaceScopeGuardV2(ScopeGuardV2):
    def __init__(
        self,
        backend: Literal["hf"] = "hf",
        model: str | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        max_new_tokens: int = 3000,
        do_sample: bool = False,
        include_default_safety_principles: bool = False,
        **kwargs,
    ):
        from ...utils import maybe_configure_gpu_usage

        maybe_configure_gpu_usage()

        from transformers import pipeline  # noqa: F401

        super().__init__(
            backend,
            include_default_safety_principles=include_default_safety_principles,
            skip_evidences=skip_evidences,
            output_fields=output_fields,
        )
        if model is None:
            raise ValueError("A model name must be provided for ScopeGuardV2.")
        self.model = model
        self._pipeline = pipeline(
            task="scope-guard-v2",
            model=self.model,
            trust_remote_code=True,
            output_fields=self._resolve_output_fields(None, None),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )  # type: ignore # ty: ignore[no-matching-overload]

    def _validate(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> ScopeGuardV2Output:
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        generated_text = self._pipeline(
            (conversation, ai_service_description), output_fields=selection
        )[0]["generated_text"]
        return _parse(generated_text, selection, self.model)

    def _batch_validate(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        skip_evidences: bool | None = None,
        output_fields: Iterable[str] | None = None,
        **kwargs,
    ) -> list[ScopeGuardV2Output]:
        selection = self._resolve_output_fields(output_fields, skip_evidences)
        if ai_service_descriptions is not None:
            pipeline_inputs = list(zip(conversations, ai_service_descriptions))
        elif ai_service_description is not None:
            pipeline_inputs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError("an AI service description is required")

        pipeline_outputs = self._pipeline(pipeline_inputs, output_fields=selection)
        return [
            _parse(out[0]["generated_text"], selection, self.model) for out in pipeline_outputs
        ]
```

- [ ] **Step 4: Update the remote-code pipeline**

In `src/hf_pipeline/scope_guard_v2.py`:

`__init__` signature: change `skip_evidences: bool = False,` to `skip_evidences: bool | None = None,` and add `output_fields=None,` on the next line. Replace `self.skip_evidences = skip_evidences` with:

```python
        from orbitals.scope_guard_v2.prompting import resolve_selection

        self.output_fields = resolve_selection(output_fields, skip_evidences)
```

Replace `_sanitize_parameters` with:

```python
    def _sanitize_parameters(self, **kwargs):
        from orbitals.scope_guard_v2.prompting import ALL_FIELDS, resolve_selection

        per_call = resolve_selection(
            kwargs.get("output_fields"), kwargs.get("skip_evidences")
        )
        selection = per_call if per_call is not None else (self.output_fields or ALL_FIELDS)
        return ({"output_fields": selection}, {}, {})
```

Replace `preprocess` with:

```python
    def preprocess(
        self,
        inputs: tuple[
            orbitals.scope_guard_v2.modeling.ScopeGuardV2Input,
            str | orbitals.types.AIServiceDescriptionV2,
        ],
        output_fields=None,
    ):
        conversation, ai_service_description = inputs

        model_messages = orbitals.scope_guard_v2.prompting.prepare_input_messages(
            conversation,
            ai_service_description,
            output_fields=output_fields,
        )

        text = self.tokenizer.apply_chat_template(
            model_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        return {"text": text}
```

- [ ] **Step 5: Run the whole suite**

Run: `cd /home/edobobo/orbitals && uv run pytest -q`
Expected: all PASS (40 in the V2 file).

- [ ] **Step 6: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/guards/hf.py src/hf_pipeline/scope_guard_v2.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): HF backend and remote pipeline take output_fields

The transformers backend resolves the selection through the base guard and hands
it to the pipeline per call; the remote-code pipeline renders it into the user
turn. skip_evidences keeps working on both.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 7: API backend — wire format stays identical unless `output_fields` is explicit

**Files:**
- Modify: `src/orbitals/scope_guard_v2/guards/api.py`
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `resolve_selection`, base `__init__(skip_evidences=, output_fields=)`.
- Produces: request bodies carry `"skip_evidences": <bool>` exactly as today, plus `"output_fields": [...]` **only** when an explicit `output_fields` was given at the call or constructor level. `_parse_output` tolerates a missing `reasoning`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- api backend wire format ---------------------------------------------------


def test_api_backend_body_is_unchanged_when_output_fields_not_used(mocked_v2_post):
    from orbitals.scope_guard_v2 import ScopeGuardV2

    sg = ScopeGuardV2(backend="api", api_url="http://example.com")
    sg.validate("hello", ai_service_description="desc")
    body = mocked_v2_post.call_args.kwargs["json"]
    assert body["skip_evidences"] is False
    assert "output_fields" not in body

    sg = ScopeGuardV2(backend="api", api_url="http://example.com", skip_evidences=True)
    sg.validate("hello", ai_service_description="desc", skip_evidences=False)
    body = mocked_v2_post.call_args.kwargs["json"]
    assert body["skip_evidences"] is False
    assert "output_fields" not in body


def test_api_backend_body_carries_explicit_output_fields(mocked_v2_post):
    from orbitals.scope_guard_v2 import ScopeGuardV2

    sg = ScopeGuardV2(backend="api", api_url="http://example.com", output_fields=["scope_class"])
    sg.validate("hello", ai_service_description="desc")
    body = mocked_v2_post.call_args.kwargs["json"]
    assert body["output_fields"] == ["scope_class"]
    assert body["skip_evidences"] is True  # derived, so an older server still narrows

    sg.batch_validate(["a"], ai_service_description="desc", output_fields=["reasoning", "scope_class"])
    body = mocked_v2_post.call_args.kwargs["json"]
    assert body["output_fields"] == ["reasoning", "scope_class"]


def test_api_parse_output_tolerates_missing_reasoning():
    from orbitals.scope_guard_v2.guards.api import _parse_output

    out = _parse_output({"scope_class": "Chit Chat", "model": "m"})
    assert out.reasoning is None
    assert out.scope_class == "Chit Chat"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "api_backend_body or parse_output_tolerates"`
Expected: FAIL — `TypeError` on `output_fields` kwarg; `KeyError: 'reasoning'`.

- [ ] **Step 3: Update `api.py`**

Add to the imports: `from typing import Iterable, Literal` (replacing the existing `from typing import Literal`) and `from ..prompting import resolve_selection`.

Replace `_build_request_data` and `_build_batch_request_data` with:

```python
def _selection_fields(
    output_fields: Iterable[str] | None, skip_evidences: bool | None
) -> dict:
    """The output-field part of a request body.

    `skip_evidences` is always sent, derived from the selection when the caller
    gave `output_fields` -- so a server that predates `output_fields` still narrows
    the output as far as it can. `output_fields` itself is sent only when the
    caller asked for it explicitly, which keeps the body byte-identical to
    pre-0.5 clients for every caller who did not.
    """
    selection = resolve_selection(output_fields, skip_evidences)
    if output_fields is not None:
        return {
            "skip_evidences": "evidences" not in (selection or ()),
            "output_fields": list(selection or ()),
        }
    return {"skip_evidences": bool(skip_evidences)}


def _build_request_data(
    model: str | None,
    conversation: ScopeGuardV2Input,
    skip_evidences: bool | None,
    ai_service_description: str | AIServiceDescriptionV2,
    output_fields: Iterable[str] | None = None,
) -> dict:
    return {
        **({"model": model} if model is not None else {}),
        "conversation": ScopeGuardV2InputTypeAdapter.dump_python(conversation),
        "ai_service_description": ai_service_description.model_dump()
        if isinstance(ai_service_description, AIServiceDescriptionV2)
        else ai_service_description,
        **_selection_fields(output_fields, skip_evidences),
    }


def _build_batch_request_data(
    model: str | None,
    conversations: list[ScopeGuardV2Input],
    skip_evidences: bool | None,
    ai_service_description: str | AIServiceDescriptionV2 | None = None,
    ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
    output_fields: Iterable[str] | None = None,
) -> dict:
    return {
        **({"model": model} if model is not None else {}),
        "conversations": [
            ScopeGuardV2InputTypeAdapter.dump_python(conversation)
            for conversation in conversations
        ],
        **(
            (
                {"ai_service_description": ai_service_description.model_dump()}
                if isinstance(ai_service_description, AIServiceDescriptionV2)
                else {"ai_service_description": ai_service_description}
            )
            if ai_service_description is not None
            else {}
        ),
        **(
            {
                "ai_service_descriptions": [
                    (ad.model_dump() if isinstance(ad, AIServiceDescriptionV2) else ad)
                    for ad in ai_service_descriptions
                ]
            }
            if ai_service_descriptions is not None
            else {}
        ),
        **_selection_fields(output_fields, skip_evidences),
    }
```

Replace `_parse_output` with:

```python
def _parse_output(result: dict) -> ScopeGuardV2Output:
    return ScopeGuardV2Output(
        scope_class=result["scope_class"],
        evidences=result.get("evidences"),
        reasoning=result.get("reasoning"),
        suggested_response=result.get("suggested_response"),
        model=result["model"],
        usage=result.get("usage"),
    )
```

In both `APIScopeGuardV2.__init__` and `AsyncAPIScopeGuardV2.__init__`: change `skip_evidences: bool = False,` to `skip_evidences: bool | None = None,`, add `output_fields: Iterable[str] | None = None,` after it, pass `skip_evidences=skip_evidences, output_fields=output_fields,` to `super().__init__(...)`, and delete the line `self.skip_evidences = skip_evidences` (the base class now stores it).

Add a helper method to **both** classes (identical body):

```python
    def _effective_args(
        self, output_fields: Iterable[str] | None, skip_evidences: bool | None
    ) -> tuple[Iterable[str] | None, bool | None]:
        """Per-call values, falling back to constructor values, level by level."""
        if output_fields is not None or skip_evidences is not None:
            return output_fields, skip_evidences
        if self.output_fields is not None:
            # constructor output_fields was given explicitly (possibly via
            # skip_evidences); send it as output_fields only if it came from
            # output_fields, otherwise keep the legacy shape.
            if self._ctor_output_fields_explicit:
                return self.output_fields, None
            return None, self.skip_evidences
        return None, self.skip_evidences
```

and in both `__init__`, after `super().__init__(...)`, add `self._ctor_output_fields_explicit = output_fields is not None`.

Then in all four `_validate` / `_batch_validate` methods (sync and async): add `output_fields: Iterable[str] | None = None,` after `skip_evidences`, and replace the `skip_evidences=skip_evidences if skip_evidences is not None else self.skip_evidences,` argument with:

```python
                output_fields=eff_fields,
                skip_evidences=eff_skip,
```

preceded, at the top of the method body, by:

```python
        eff_fields, eff_skip = self._effective_args(output_fields, skip_evidences)
```

- [ ] **Step 4: Run the whole suite**

Run: `cd /home/edobobo/orbitals && uv run pytest -q`
Expected: all PASS (43 in the V2 file). `test_scope_guard_v2_api_backend_hits_v2_endpoint` and friends still pass with an unchanged body.

- [ ] **Step 5: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/guards/api.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): API backend sends output_fields only when asked

A 0.5 client talking to a pre-0.5 server sends exactly the body it sent before
unless the caller used output_fields explicitly. When it did, skip_evidences is
still sent, derived from the selection, so an older server narrows the output as
far as it can. reasoning is read with .get, since a selection without it is now
a legitimate response.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 8: Serving endpoints and the `serve` CLI

**Files:**
- Modify: `src/orbitals/scope_guard_v2/serving/main.py`
- Modify: `src/orbitals/scope_guard_v2/cli/serve.py`
- Test: `tests/test_scope_guard_v2.py`

**Interfaces:**
- Consumes: `AsyncVLLMApiScopeGuardV2(output_fields=)`, `ScopeGuardV2Output.reasoning: str | None`.
- Produces: body field `output_fields: list[str] | None`; env var `SCOPE_GUARD_V2_OUTPUT_FIELDS` (comma-separated, empty means unset); CLI flag `--output-fields`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scope_guard_v2.py`:

```python
# --- serving ------------------------------------------------------------------


def test_scope_guard_v2_serving_forwards_output_fields_and_allows_null_reasoning(monkeypatch):
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_MODEL", "v2-model")
    monkeypatch.setenv("SCOPE_GUARD_V2_VLLM_SERVING_URL", "http://localhost:8001")
    monkeypatch.setenv("SCOPE_GUARD_V2_SKIP_EVIDENCES", "0")
    monkeypatch.setenv("SCOPE_GUARD_V2_OUTPUT_FIELDS", "")

    from fastapi.testclient import TestClient

    from orbitals.scope_guard_v2 import ScopeClass, ScopeGuardV2Output
    from orbitals.scope_guard_v2.serving import main as serving_main

    seen: dict[str, Any] = {}

    class _StubAsyncGuard:
        async def validate(self, conversation, *, ai_service_description, **kwargs):
            seen.update(kwargs)
            return ScopeGuardV2Output(
                scope_class=ScopeClass.OUT_OF_SCOPE,
                model="stub-model",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

    with TestClient(serving_main.app) as client:
        monkeypatch.setattr(serving_main, "scope_guard", _StubAsyncGuard())
        response = client.post(
            "/orbitals/scope-guard-v2/validate",
            json={
                "conversation": "hello",
                "ai_service_description": "desc",
                "output_fields": ["scope_class"],
            },
        )

    assert response.status_code == 200
    assert seen["output_fields"] == ["scope_class"]
    body = response.json()
    assert body["scope_class"] == "Out of Scope"
    assert body["reasoning"] is None


def test_scope_guard_v2_serve_cli_exposes_output_fields():
    from orbitals.cli.main import app

    result = CliRunner().invoke(app, ["scope-guard-v2", "serve", "--help"])
    assert result.exit_code == 0
    assert "--output-fields" in result.stdout
    assert "--skip-evidences" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/edobobo/orbitals && uv run pytest tests/test_scope_guard_v2.py -q -k "serving_forwards or serve_cli_exposes"`
Expected: FAIL — `output_fields` not forwarded (`KeyError`) and `ScopeGuardV2Response` rejects `reasoning=None`; `--output-fields` absent from help.

- [ ] **Step 3: Update `serving/main.py`**

In `lifespan`, replace the `AsyncScopeGuardV2(...)` construction with:

```python
    raw_fields = os.environ.get("SCOPE_GUARD_V2_OUTPUT_FIELDS", "")
    output_fields = [f.strip() for f in raw_fields.split(",") if f.strip()] or None

    scope_guard = AsyncScopeGuardV2(  # type: ignore[invalid-assignment]
        backend="vllm-api",
        model=os.environ["SCOPE_GUARD_V2_VLLM_MODEL"],
        skip_evidences=(os.environ.get("SCOPE_GUARD_V2_SKIP_EVIDENCES") == "1") or None,
        output_fields=output_fields,
        vllm_serving_url=os.environ["SCOPE_GUARD_V2_VLLM_SERVING_URL"],
    )
```

(`skip_evidences` becomes `True` or `None` — never `False` — so it does not shadow an `output_fields` given via the environment.)

Change `ScopeGuardV2Response.reasoning: str` to `reasoning: str | None`.

In both endpoint signatures add, after the `skip_evidences` parameter:

```python
    output_fields: Annotated[list[str] | None, Body()] = None,
```

and pass `output_fields=output_fields,` to the `scope_guard.validate(...)` / `scope_guard.batch_validate(...)` calls after `skip_evidences=skip_evidences,`.

- [ ] **Step 4: Update `cli/serve.py`**

After the `skip_evidences` option add:

```python
    output_fields: str | None = typer.Option(
        None,
        help=(
            "Comma-separated output fields the model should emit, e.g. "
            "'reasoning,scope_class'. scope_class is always included. "
            "Overrides --skip-evidences."
        ),
    ),
```

and after the `SCOPE_GUARD_V2_SKIP_EVIDENCES` assignment add:

```python
    os.environ["SCOPE_GUARD_V2_OUTPUT_FIELDS"] = output_fields or ""
```

- [ ] **Step 5: Run the whole suite**

Run: `cd /home/edobobo/orbitals && uv run pytest -q`
Expected: all PASS (45 in the V2 file).

- [ ] **Step 6: Commit**

```bash
cd /home/edobobo/orbitals
git add src/orbitals/scope_guard_v2/serving/main.py src/orbitals/scope_guard_v2/cli/serve.py tests/test_scope_guard_v2.py
git commit -m "$(cat <<'EOF'
feat(scope-guard-v2): output_fields on the serving endpoints and serve CLI

Both endpoints accept output_fields next to skip_evidences and forward it; the
response model allows a null reasoning. `orbitals scope-guard-v2 serve` gains
--output-fields, exported as SCOPE_GUARD_V2_OUTPUT_FIELDS.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 9: README, version bump

**Files:**
- Modify: `README.scope-guard-v2.md`
- Modify: `pyproject.toml:7` (`version = "0.4.0"` → `"0.5.0"`)

**Interfaces:** none (docs).

- [ ] **Step 1: Bump the version**

In `pyproject.toml` change `version = "0.4.0"` to `version = "0.5.0"`. Then run `cd /home/edobobo/orbitals && uv lock` so `uv.lock` records the new version.

- [ ] **Step 2: Update the README**

In `README.scope-guard-v2.md`:

(a) In the bullet list under "On top of the classification, ScopeGuard V2 returns:", make the three bullets read:

```markdown
* **`evidences`**: verbatim spans from the AI service description supporting the decision
* **`reasoning`**: a short explanation of why the class was chosen
* **`suggested_response`**: a ready-to-use response for the user when the query should not be processed (e.g., for Predefined Answer, Human Oversight, Out of Scope, Restricted, or Chit Chat)

All three are optional: you choose which ones the model emits per guard or per call (see [Output Fields](#output-fields)). `scope_class` is always returned.
```

(b) Replace the whole "### Skipping Evidences" section with:

```markdown
### Output Fields

You can choose which fields the model emits, either per guard or per call. `scope_class` is always included; `evidences`, `reasoning`, and `suggested_response` are optional. Fewer fields mean fewer generated tokens and lower latency, and never change the class the model would have chosen.

```python
# per guard
sg = ScopeGuardV2(backend="api", api_key="principled_1234", output_fields=["scope_class"])

# per call
result = sg.validate(
    user_query,
    ai_service_description=ai_service_description,
    output_fields=["reasoning", "scope_class"],
)

print(result.scope_class)      # always present
print(result.reasoning)        # present when requested, otherwise None
print(result.evidences)        # None -- not requested
```

Fields you did not request come back as `None` on the result object.

> [!TIP]
> On our benchmarks the `scope_class`-only mode is both the cheapest and the most accurate. Request `reasoning`, `evidences`, or `suggested_response` when you need them for your product, not to improve the classification.

The pre-existing `skip_evidences` flag keeps working and means "every field except `evidences`":

```python
sg = ScopeGuardV2(backend="api", api_key="principled_1234", skip_evidences=True)
```

If you pass both `output_fields` and `skip_evidences` and they disagree, `output_fields` wins and a `DeprecationWarning` is emitted.
```

(c) In the "## Self-hosting" section, insert after the first paragraph:

```markdown
> [!IMPORTANT]
> Since `orbitals` 0.5.0 the `vllm`, `vllm-api`, and `hf` backends use the prompt the **2608 "promptfix"** models were trained on. Self-hosting an older ScopeGuard V2 checkpoint with this version of the library will degrade its classifications. The `api` backend is unaffected. When the model directory ships a `system_prompt.txt`, the `vllm` backend logs a warning if it does not match.
```

- [ ] **Step 3: Run the whole suite one more time**

Run: `cd /home/edobobo/orbitals && uv run pytest -q`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/edobobo/orbitals
git add README.scope-guard-v2.md pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
docs(scope-guard-v2): output fields section, 2608 model requirement; bump to 0.5.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

### Task 10: Cross-repo drift test in the trainer

**Files:**
- Modify: `/home/edobobo/sg2-trainer/tests/test_prompting.py`

**Interfaces:**
- Consumes: `orbitals.scope_guard_v2.prompting.SYSTEM_PROMPT`, `render_selector_block` (when orbitals is importable in the trainer's environment); trainer's `src.prompting.SYSTEM_PROMPT`, `src.output_fields.render_selector_block`, `enumerate_selections`.

- [ ] **Step 1: Add the tests**

Append to `/home/edobobo/sg2-trainer/tests/test_prompting.py`:

```python
import hashlib

import pytest


def test_prompt_hash_is_pinned():
    """The 2608 promptfix prompt, trailing newline included.

    orbitals pins the same hash. Changing this value is a retrain, and it has to be
    changed in both repositories in the same release.
    """
    assert len(SYSTEM_PROMPT) == 7665
    assert (
        hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        == "f0f68e48096938b93c2c497386295ff7db2929b128e9dddea944ebb6578960b5"
    )


def test_prompt_matches_orbitals_when_installed():
    """Byte-equality with the client library that serves this model.

    Skipped, not failed, where orbitals is not installed: the pin above still
    holds each side to the same bytes independently.
    """
    orbitals_prompting = pytest.importorskip("orbitals.scope_guard_v2.prompting")
    from src.output_fields import enumerate_selections, render_selector_block

    assert orbitals_prompting.SYSTEM_PROMPT == SYSTEM_PROMPT
    for selection in enumerate_selections():
        assert orbitals_prompting.render_selector_block(selection) == render_selector_block(
            selection
        ), selection
```

- [ ] **Step 2: Run the trainer's prompt tests**

Run: `cd /home/edobobo/sg2-trainer && PYTHONPATH=. uv run pytest tests/test_prompting.py -q`
Expected: `test_prompt_hash_is_pinned` PASS; `test_prompt_matches_orbitals_when_installed` SKIPPED (orbitals is not in the trainer venv) — or PASS if it is. To exercise it once for real: `cd /home/edobobo/sg2-trainer && PYTHONPATH=.:/home/edobobo/orbitals/src uv run pytest tests/test_prompting.py -q -k orbitals` — expected PASS. (Adding the path this way imports orbitals' source directly; it needs pydantic, which the trainer venv has.)

- [ ] **Step 3: Commit**

```bash
cd /home/edobobo/sg2-trainer
git add tests/test_prompting.py
git commit -m "$(cat <<'EOF'
test: pin the promptfix prompt hash and check byte-equality with orbitals

The client library now ships this exact prompt. Both repositories pin sha256
f0f68e48...; this test also compares the two directly when orbitals is
importable, and skips otherwise.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LbS1LRQcDDcXAhDz6zkF5F
EOF
)"
```

---

## Out of scope, recorded for follow-up

- `unsafe-eval` strips the trailing newline from `profile/system.txt` (`_resources.py:14`), so every published 2608 evaluation ran with a one-character train/eval mismatch. Not fixed here.
- The `system_prompt.txt` files in the 2608 Hub releases lack the trailing newline the model trained with. Not fixed here; the vllm backend's drift check deliberately ignores that difference.
- Sampling defaults differ between orbitals (`temperature=0.0`) and the trainer's reference guard (`0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5`). Unchanged here.
