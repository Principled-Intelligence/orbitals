import torch
import transformers
from transformers import Pipeline

try:
    import orbitals.scope_guard_v2
    import orbitals.scope_guard_v2.modeling
    import orbitals.scope_guard_v2.prompting
    import orbitals.types
except ModuleNotFoundError:
    raise ImportError(
        "orbitals.scope_guard_v2 module not found. Please install it: `pip install orbitals`"
    )


class ScopeGuardV2Pipeline(Pipeline):
    def __init__(
        self,
        model,
        tokenizer=None,
        skip_evidences: bool | None = None,
        output_fields=None,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        **kwargs,
    ):
        if tokenizer is None and isinstance(model, str):
            tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        elif isinstance(tokenizer, str):
            tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

        if isinstance(model, str):
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model, dtype="auto", device_map="auto"
            )

        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        from orbitals.scope_guard_v2.prompting import resolve_selection

        self.output_fields = resolve_selection(output_fields, skip_evidences)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        from orbitals.scope_guard_v2.prompting import ALL_FIELDS, resolve_selection

        per_call = resolve_selection(
            kwargs.get("output_fields"), kwargs.get("skip_evidences")
        )
        selection = per_call if per_call is not None else (self.output_fields or ALL_FIELDS)
        return ({"output_fields": selection}, {}, {})

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

    def _forward(self, model_inputs):
        tokenized = self.tokenizer(
            model_inputs["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **tokenized,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return {
            "output_ids": outputs,
            "input_ids": tokenized["input_ids"],
        }

    def postprocess(self, model_outputs):
        output_ids = model_outputs["output_ids"]
        input_ids = model_outputs["input_ids"]

        results = []
        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i][input_ids.shape[1] :]
            generated_output = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            results.append({"generated_text": generated_output})

        return results
