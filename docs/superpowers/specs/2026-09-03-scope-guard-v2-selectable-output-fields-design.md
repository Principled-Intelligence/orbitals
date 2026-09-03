# ScopeGuard V2: promptfix system prompt and selectable output fields

Status: approved design, not yet implemented.
Date: 2026-09-03
Scope: `src/orbitals/scope_guard_v2/`, `src/hf_pipeline/scope_guard_v2.py`, tests, README.

## Problem

`orbitals.scope_guard_v2` implements the **2606** prompt generation: a fixed
four-field output (`evidences`, `reasoning`, `scope_class`, `suggested_response`),
a `skip_evidences` boolean expressed as a `**SKIP EVIDENCES**` marker in the user
turn, and the response JSON schema embedded in the system prompt.

The models now being trained and released (`sg2-Qwen3.5-4B-2608-promptfix`,
`sg2-Qwen3.5-9B-2608-promptfix`, `...-merge4`) were trained on the **2608
promptfix** generation, which differs in three ways:

1. **The prompt text.** The `Potentially Supported` class description and
   Instruction 3 were rewritten so that class is unavailable whenever a constraint,
   escalation criterion, or predefined response matches. This is the "promptfix",
   and it is the single largest measured improvement in the model's history
   (+1.15pp balanced accuracy on the binary axis at equal training).
2. **Selectable output fields.** The user turn ends with a
   `**REQUESTED OUTPUT FIELDS**` block naming the keys to emit, as a JSON array in
   canonical order. `scope_class` is always requested. The model emits exactly those
   keys. `skip_evidences` is one point (of eight) in this space.
3. **No schema in the system prompt.** The field types are listed in prose; the
   per-request selector carries the key set.

A 2608 model served through the 2606 prompt degrades quietly: the system prompt it
sees differs from training in dozens of places and the selector block it expects is
absent. We have measured that a diverged system prompt costs accuracy without any
parse error to signal it. So the library prompt must be **byte-identical** to the
training prompt, and the user turn must carry the selector.

## Decisions taken

- **Clean break on the prompt.** One system prompt, the promptfix text. Library
  `>= 0.5.0` requires a 2608-trained model for the `vllm`, `vllm-api` and `hf`
  backends. The `api` backend sends no prompt (the hosted server owns it) and is
  unaffected. The V2 weights are documented as private, so the population of
  self-hosted 2606 deployments this breaks is close to empty. No `prompt_version`
  switch: a second path could never be validated against a model we still ship.
- **Interfaces stay backward compatible.** Every existing constructor argument,
  method signature, body field and response field keeps working. New capability is
  added alongside; nothing is removed. Behaviour for a caller who passes nothing new
  is unchanged: all four fields, same key order.
- **Default modality stays all four fields.** `scope_class` alone measures best
  (0.8812 balanced accuracy on the 9B merge vs 0.8663 for all four) and is the
  cheapest to serve, but changing the default would silently remove fields callers
  read today. The README recommends it instead.
- **The precedence rule `Restricted > Out of Scope` does not go in.** It is absent
  from the training prompt. Adding it here would diverge the client from training,
  which is the failure this work exists to prevent. It belongs in a retrain.
- **Prompt text is ported, not imported (approach A).** Orbitals copies the prompt
  and the small pure-Python selector/schema helpers. Drift is prevented by a hash
  test in both repositories, not by a dependency in either direction. Making the
  trainer depend on the client library would tie training reproducibility to a
  release cadence; loading the prompt from the model directory at runtime would
  make behaviour depend on folder contents and still leave the selector format in
  code. The `vllm` backend does *verify* against `system_prompt.txt` when the model
  directory ships one, and warns on mismatch.

## The contract

### System prompt

Replaced verbatim with the training prompt. Pinned:

    sha256(SYSTEM_PROMPT) == 1f37419f2f236b695c73ec814a9682905bdd0ce8ae98bc5fa52ab2ef2dcd0fc4

This is the hash `unsafe-eval` records as `profile/system.txt` in every 2608
promptfix evaluation, so the eval harness, the trainer and the client all pin the
same bytes.

The prompt interpolates two things that therefore also change:

- `ScopeClass.get_classes_manifest()` — the `Potentially Supported` description in
  `_SCOPE_DESCRIPTIONS` becomes: *"The query is adjacent to the service's stated
  functionalities or knowledge scope without being named by any of them, and no
  constraint, escalation criterion, or predefined response applies to it. Use it
  only when the service could handle the request as a reasonable extension of what
  it already does -- never as a way to avoid committing to a clearer class."*
- `SELECTOR_HEADER = "**REQUESTED OUTPUT FIELDS**"`.

The embedded `_RESPONSE_SCHEMA` is removed from the prompt.

### Output fields

```python
SCOPE_CLASS = "scope_class"
OPTIONAL_FIELDS = ("evidences", "reasoning", "suggested_response")
CANONICAL_ORDER = ("evidences", "reasoning", "scope_class", "suggested_response")
ALL_FIELDS = CANONICAL_ORDER

def normalize_selection(fields: Iterable[str]) -> tuple[str, ...]
    # dedupe, force scope_class in, reorder to canonical, reject unknown names

def render_selector_block(selection) -> str
    # "**REQUESTED OUTPUT FIELDS**\n\n" + json.dumps(list(normalized))
```

The user turn becomes:

    **START OF THE AI SERVICE DESCRIPTION**

    {asd}

    **END OF THE AI SERVICE DESCRIPTION**


    **START OF THE CONVERSATION DUMP**

    {conversation}

    **END OF THE CONVERSATION DUMP**


    **REQUESTED OUTPUT FIELDS**

    ["reasoning", "scope_class"]

Three newlines before the selector block, matching
`src/data_processing.prepare_input_messages` in the trainer exactly. The
`**SKIP EVIDENCES**` marker is gone.

### Resolving `skip_evidences` and `output_fields`

Both are accepted everywhere `skip_evidences` is accepted today: constructors,
`validate`, `batch_validate`, the serving body, the CLI. One resolver on
`BaseScopeGuardV2` computes the effective selection, so five backends cannot
disagree:

1. per-call `output_fields`, if not None
2. else per-call `skip_evidences`, if not None → `ALL_FIELDS` minus `evidences`
   when True, `ALL_FIELDS` when False
3. else constructor `output_fields`, if not None
4. else constructor `skip_evidences` → as in 2
5. else `ALL_FIELDS`

If a caller passes **both** `output_fields` and `skip_evidences` at the same level
and they disagree (e.g. `output_fields=["evidences","scope_class"]` with
`skip_evidences=True`), `output_fields` wins and a `DeprecationWarning` is emitted
naming both values. `skip_evidences` itself is not deprecated in 0.5 — it stays a
documented convenience — only the conflicting combination warns.

### Response model

`ScopeGuardV2ResponseModel` (fixed shape) is replaced by

```python
def response_model_for(selection) -> type[BaseModel]
```

which builds, and caches per normalized selection, a pydantic model with **exactly
the requested keys, all required, `extra="forbid"`**. Its `model_json_schema()` is
what the `vllm-api` backend sends as `structured_outputs`, so guided decoding
cannot emit an unrequested key; validating the completion against the same model is
itself the check that the model obeyed the selector. Field types are unchanged:
`evidences: list[str] | None`, `reasoning: str`, `scope_class: ScopeClass`,
`suggested_response: str | None`.

`ScopeGuardV2Output.reasoning` becomes `str | None = None`. This is the only type
change on the public output and is a loosening: every existing read of
`.reasoning` still works, and it is `None` only when the caller asked for a
selection without it. The serving `ScopeGuardV2Response.reasoning` changes the
same way. `_parse_output` in the `api` backend reads `reasoning` with `.get`.

### Prefill

`build_prompt(prefill=True)` currently appends `'{"evidences":'` or
`'{"evidences": null, "reasoning": "'`. Both hardcode a key order that the
selector now controls. It becomes `'{"' + selection[0] + '":'`. The parameter is
kept (it is part of `build_prompt`'s signature) and remains unused by the
`vllm-api` backend for the reason already noted there: guided decoding is unaware
of the prefilled prefix.

### Wire format

The `api` backend adds `"output_fields": [...]` to the request body **only when the
resolved selection came from an explicit `output_fields`** (levels 1 or 3 above).
When the caller used `skip_evidences` or nothing, the body is byte-identical to
today's, so a 0.5 client against a pre-0.5 server behaves exactly as before.

The serving endpoints accept `output_fields: list[str] | None = Body(None)` next to
`skip_evidences`. The CLI `serve` gains `--output-fields` (comma-separated),
exported as `SCOPE_GUARD_V2_OUTPUT_FIELDS`; `--skip-evidences` is kept.

### HF remote pipeline

`src/hf_pipeline/scope_guard_v2.py` gains `output_fields` in `__init__`,
`_sanitize_parameters` and `preprocess`, passing it through to
`prepare_input_messages`. Same resolution rule as the guards.

### Verification against the shipped prompt

`VLLMScopeGuardV2.__init__` looks for `system_prompt.txt` in the model directory
(every 2608 Hub release ships one). If present and its bytes differ from
`SYSTEM_PROMPT`, it logs a warning naming both hashes. It never *uses* the file —
the built-in prompt is the source of truth — this is a drift detector for the case
where someone points the library at a model from a different prompt generation.

## Files touched

| file | change |
| --- | --- |
| `scope_guard_v2/prompting.py` | new `SYSTEM_PROMPT`; selector helpers; `response_model_for`; `prepare_input_messages(output_fields=, skip_evidences=)`; prefill fix |
| `scope_guard_v2/modeling.py` | `Potentially Supported` description; `ScopeGuardV2Output.reasoning` optional |
| `scope_guard_v2/guards/base.py` | `output_fields` on overloads, `validate`, `batch_validate`; `_resolve_output_fields` |
| `scope_guard_v2/guards/vllm.py` | plumb selection; per-selection `structured_outputs`; `system_prompt.txt` check |
| `scope_guard_v2/guards/hf.py` | plumb selection |
| `scope_guard_v2/guards/api.py` | body includes `output_fields` when explicit; `.get("reasoning")` |
| `scope_guard_v2/serving/main.py` | body + response model |
| `scope_guard_v2/cli/serve.py` | `--output-fields` |
| `src/hf_pipeline/scope_guard_v2.py` | plumb selection |
| `scope_guard_v2/__init__.py` | unchanged. `__all__` stays as is so `test_scope_guard_v2_package_all_is_exhaustive` holds; `normalize_selection` and `ALL_FIELDS` are reachable via `orbitals.scope_guard_v2.prompting`, which is where callers already import `build_prompt` from. |
| `tests/test_scope_guard_v2.py` | see Testing |
| `README.scope-guard-v2.md` | "Output fields" section; model requirement; recommendation |
| `pyproject.toml` | 0.4.0 → 0.5.0 |

Not touched: `scope_guard/` (V1), `claim_extractor/`, `types.py`,
`safety_principles.py`.

## Testing

All pure-Python, no model download, following the existing mock-`requests.post`
pattern in `tests/test_scope_guard_v2.py`.

- `sha256(SYSTEM_PROMPT)` equals the pinned hash. This is the test that makes drift
  a red build.
- `normalize_selection`: forces `scope_class`, dedupes, reorders, rejects unknown.
- `render_selector_block` for all 8 combinations equals the trainer's rendering
  (eight literal expected strings).
- `prepare_input_messages`: default → all four; `skip_evidences=True` → three;
  `output_fields` overrides; conflicting pair warns and `output_fields` wins; user
  turn ends with the selector block and contains no `**SKIP EVIDENCES**`.
- `response_model_for`: forbids extra keys; requires every requested key; cached
  per normalized selection; `("scope_class",)` model validates `{"scope_class":
  "Restricted"}` and rejects the same object with `reasoning` added.
- `ScopeGuardV2Output(reasoning=None, ...)` constructs; API `_parse_output` tolerates
  a payload without `reasoning`.
- `api` backend: body omits `output_fields` when unset or when only
  `skip_evidences` was used; includes it when explicit. Sync and async.
- `build_prompt(prefill=True)` prefixes the selection's first key.
- Existing 17 tests keep passing unchanged.

In the **trainer** (`sg2-trainer/tests/test_prompting.py`): pin the same hash, and
add a test that imports `orbitals.scope_guard_v2.prompting` when available and
asserts `SYSTEM_PROMPT` byte-equality plus identical `render_selector_block` output
for all 8 selections. Skipped, not failed, when orbitals is not installed.

## Risks

- **The 2608 requirement is a hard break for self-hosters.** Mitigated by the
  changelog, the README banner, and the `system_prompt.txt` warning in the vllm
  backend. Accepted because the weights are private.
- **`reasoning` becoming optional could surprise typed callers.** It is `None` only
  when the caller opted into a selection without it. Documented next to the field.
- **Duplicated helper code can drift.** The hash test guards the prompt; the
  8-string selector test guards the user turn. The schema builder has no
  cross-repo guard beyond its own unit tests, so a change to field types on either
  side must be made in both — noted in a comment at the top of the copied block.
- **Sampling defaults differ between repos.** Orbitals decodes greedily
  (`temperature=0.0`); the trainer's reference guard uses Qwen's recommended
  `0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5`. This spec does not change
  orbitals' default. Whether the published benchmark numbers were produced greedily
  or sampled is not settled here and should be before any number is quoted in the
  README as reproducible.
