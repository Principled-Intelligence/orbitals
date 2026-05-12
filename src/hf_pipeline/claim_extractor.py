import torch
import transformers
from transformers import Pipeline

try:
    import orbitals.claim_extractor
    import orbitals.claim_extractor.modeling
    import orbitals.claim_extractor.prompting
    import orbitals.types
except ModuleNotFoundError:
    raise ImportError(
        "orbitals.claim_extractor module not found. Please install it: `pip install orbitals`"
    )


class ClaimExtractorPipeline(Pipeline):
    def __init__(
        self,
        model,
        tokenizer=None,
        skip_evidences: bool = True,
        max_new_tokens: int = 20_000,
        do_sample: bool = False,
        temperature: float = 0.7,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 1.5,
        repetition_penalty: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
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

        # Set left padding for decoder-only models (required for batched generation)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            # Ensure pad token is set (use eos_token if pad_token doesn't exist)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        self.skip_evidences = skip_evidences
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p

        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(
        self,
        **kwargs,
    ):
        preprocess_kwargs = {
            "skip_evidences": kwargs.get("skip_evidences", self.skip_evidences)
        }

        return (
            preprocess_kwargs,
            {},
            {},
        )

    def preprocess(
        self,
        inputs: tuple[
            orbitals.claim_extractor.modeling.ClaimExtractorInput,
            str | orbitals.types.AIServiceDescription | None,
        ],
        skip_evidences: bool = True,
    ):
        conversation, ai_service_description = inputs

        model_messages = orbitals.claim_extractor.prompting.prepare_messages(
            conversation,
            ai_service_description,
            skip_evidences=skip_evidences,
        )

        text = self.tokenizer.apply_chat_template(
            model_messages,
            tokenize=False,  # we are not tokenizing so as to enable batching
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
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
            )
        return {
            "output_ids": outputs,
            "input_ids": tokenized["input_ids"],
        }

    def postprocess(self, model_outputs):
        output_ids = model_outputs["output_ids"]
        input_ids = model_outputs["input_ids"]

        # Decode each output in the batch
        results = []
        for i in range(output_ids.shape[0]):
            # Skip the input tokens to get only the generated text
            generated_ids = output_ids[i][input_ids.shape[1] :]
            generated_output = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            results.append({"generated_text": generated_output})

        return results
