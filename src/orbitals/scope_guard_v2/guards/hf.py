import json
from typing import TYPE_CHECKING, Iterable, Literal

import pydantic

if TYPE_CHECKING:
    from transformers import pipeline  # noqa: F401

from ...types import AIServiceDescriptionV2, LLMUsage
from ..modeling import (
    ScopeClass,
    ScopeGuardV2Classification,
    ScopeGuardV2Input,
    ScopeGuardV2Output,
)
from ..prompting import (
    CLASS_PREFIX,
    PREDEFINED_PREFIX,
    SYSTEM_PROMPT,
    build_prompt,
    check_shipped_system_prompt,
    class_first_tokens,
    predefined_candidates,
    response_model_for,
)
from .base import ScopeGuardV2
from .vllm import (
    _DECISION_LOGPROBS,
    _RESPONSE_STOP,
    _add_usage,
    _classification,
    _generated,
    _grammar_choices,
    _selected,
)


def _parse(
    record: dict, selection: tuple[str, ...], model: str, usage: LLMUsage | None
) -> ScopeGuardV2Output:
    generated_text = record["generated_text"]
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
        usage=usage,
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
        count_system_prompt_in_usage: bool = False,
        decision_temperature: float = 1.0,
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
        check_shipped_system_prompt(self.model)
        self._pipeline = pipeline(
            task="scope-guard-v2",
            model=self.model,
            trust_remote_code=True,
            output_fields=self._resolve_output_fields(None, None),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )  # type: ignore # ty: ignore[no-matching-overload]
        self.count_system_prompt_in_usage = count_system_prompt_in_usage
        self._cached_system_prompt_tokens: int | None = None
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.decision_temperature = decision_temperature

    @property
    def _system_prompt_tokens(self) -> int:
        """Overhead to exclude from a caller's prompt count.

        Resolved on first use rather than at construction: it needs the pipeline's
        tokenizer, and a checkpoint whose pipeline reports no counts never asks.
        """
        if self.count_system_prompt_in_usage:
            return 0
        if self._cached_system_prompt_tokens is None:
            self._cached_system_prompt_tokens = len(
                self._pipeline.tokenizer.encode(SYSTEM_PROMPT)
            )
        return self._cached_system_prompt_tokens

    def _usage(self, record: dict) -> LLMUsage | None:
        """Token counts, when the checkpoint's pipeline reports them.

        That pipeline ships inside the model repo and versions independently of this
        library, so a checkpoint published before the counts existed simply omits the
        keys. Reading them defensively is what keeps those checkpoints working.
        """
        prompt_tokens = record.get("prompt_tokens")
        completion_tokens = record.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            return None
        prompt_tokens -= self._system_prompt_tokens
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _classify(
        self,
        conversation: ScopeGuardV2Input,
        *,
        ai_service_description: str | AIServiceDescriptionV2,
        resolve_predefined: bool = True,
        **kwargs,
    ) -> ScopeGuardV2Classification:
        import torch

        tokenizer = self._pipeline.tokenizer
        model = self._pipeline.model
        prompt = (
            build_prompt(
                tokenizer,
                conversation,
                ai_service_description,
                output_fields=["scope_class"],
            )
            + CLASS_PREFIX
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(self._pipeline.device)
        with torch.inference_mode():
            logprobs = model(**inputs).logits[0, -1].float().log_softmax(-1)
        # The same top-k vLLM returns, so both backends floor missing classes alike.
        top = logprobs.topk(_DECISION_LOGPROBS)
        result = _classification(
            {
                tokenizer.decode([i]): lp
                for lp, i in zip(top.values.tolist(), top.indices.tolist())
            },
            class_first_tokens(tokenizer),
            temperature=self.decision_temperature,
            model=self.model,
            usage=self._usage(
                {"prompt_tokens": inputs.input_ids.shape[1], "completion_tokens": 1}
            ),
        )
        if (
            result.scope_class is not ScopeClass.PREDEFINED_ANSWER
            or not resolve_predefined
        ):
            return result
        candidates = predefined_candidates(ai_service_description)
        if len(candidates) == 1:
            result.predefined_response = candidates[0]
            return result

        prompt = (
            build_prompt(
                tokenizer,
                conversation,
                ai_service_description,
                output_fields=["scope_class", "suggested_response"],
            )
            + PREDEFINED_PREFIX
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(self._pipeline.device)
        start = inputs.input_ids.shape[1]
        generate_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if candidates:
            choices = [
                tokenizer.encode(c, add_special_tokens=False)
                for c in _grammar_choices(candidates)
            ]

            # Greedy decoding over the listed texts' tokens, as vLLM's `choice` does.
            def allowed(batch_id: int, ids) -> list[int]:
                done = ids[start:].tolist()
                n = len(done)
                nxt = {c[n] for c in choices if len(c) > n and c[:n] == done}
                if done in choices:
                    nxt.add(tokenizer.eos_token_id)
                return list(nxt)

            generate_kwargs |= {
                "do_sample": False,
                "prefix_allowed_tokens_fn": allowed,
            }
        else:
            generate_kwargs |= {
                "do_sample": self.do_sample,
                "stop_strings": [_RESPONSE_STOP],
                "tokenizer": tokenizer,
            }
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)[0, start:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        result.predefined_response = (
            _selected(text, candidates) if candidates else _generated(text)
        )
        second = self._usage(
            {"prompt_tokens": start, "completion_tokens": generated.shape[0]}
        )
        if result.usage is not None and second is not None:
            result.usage = _add_usage(result.usage, second)
        return result

    def _batch_classify(
        self,
        conversations: list[ScopeGuardV2Input],
        *,
        ai_service_description: str | AIServiceDescriptionV2 | None = None,
        ai_service_descriptions: list[str] | list[AIServiceDescriptionV2] | None = None,
        resolve_predefined: bool = True,
        **kwargs,
    ) -> list[ScopeGuardV2Classification]:
        if ai_service_descriptions is not None:
            pairs = list(zip(conversations, ai_service_descriptions))
        elif ai_service_description is not None:
            pairs = [(c, ai_service_description) for c in conversations]
        else:
            raise ValueError("an AI service description is required")

        return [
            self._classify(
                c, ai_service_description=ad, resolve_predefined=resolve_predefined
            )
            for c, ad in pairs
        ]

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
        record = self._pipeline(
            (conversation, ai_service_description), output_fields=selection
        )[0]
        return _parse(record, selection, self.model, self._usage(record))

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
            _parse(out[0], selection, self.model, self._usage(out[0]))
            for out in pipeline_outputs
        ]
