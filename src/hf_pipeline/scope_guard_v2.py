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
        skip_evidences: bool = False,
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

        self.skip_evidences = skip_evidences
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "skip_evidences" in kwargs or self.skip_evidences:
            preprocess_kwargs["skip_evidences"] = kwargs.get(
                "skip_evidences", self.skip_evidences
            )

        return (
            preprocess_kwargs,
            {},
            {},
        )

    def preprocess(
        self,
        inputs: tuple[
            orbitals.scope_guard_v2.modeling.ScopeGuardV2Input,
            str | orbitals.types.AIServiceDescriptionV2,
        ],
        skip_evidences: bool = False,
    ):
        conversation, ai_service_description = inputs

        model_messages = orbitals.scope_guard_v2.prompting.prepare_input_messages(
            conversation,
            ai_service_description,
            skip_evidences,
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
