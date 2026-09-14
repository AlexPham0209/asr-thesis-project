import asyncio
import logging
import os
import warnings
from typing import Any

from google import genai
from google.genai import types
import nest_asyncio
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asr_thesis_project.generator.generator import (
    BaseGenerator,
    clean_prediction,
    examples_to_text,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")


def build_user_message(input_text: str, examples: list[Any]) -> str:
    """Few-shot block + the sentence to correct, as one user turn.

    Examples go in the user message rather than the system instruction so the
    system prompt stays static (better for caching) and the model sees a
    clear boundary between references and the actual query.
    """
    block = examples_to_text(examples)
    if block:
        return (
            "Reference examples (ASR output -> corrected sentence with inline LaTeX):\n\n"
            f"{block}\n\n"
            "Now correct the following ASR output in exactly the same format. "
            "Output only the corrected sentence.\n\n"
            f"ASR output: {input_text}"
        )
    return (
        "Correct the following ASR output, writing mathematics as inline LaTeX "
        "inside $...$. Output only the corrected sentence.\n\n"
        f"ASR output: {input_text}"
    )


class GeminiGenerator(BaseGenerator):
    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model_name: str = "gemini-2.5-flash",
        max_concurrent: int = 4,
        use_async: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        thinking_budget: int | None = 0,
        retry_attempts: int = 5,
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.use_async = use_async

        # SDK-level back-off on 429 / transient 5xx.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=retry_attempts,
                    initial_delay=2.0,
                    max_delay=60.0,
                    http_status_codes=[429, 500, 502, 503, 504],
                )
            ),
        )

        # Thinking tokens share max_output_tokens; leave it on and a small cap
        # regularly yields response.text == None.
        thinking = (
            types.ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        )
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
        )

    @staticmethod
    def _extract_text(response) -> str:
        text = response.text
        if text is None:
            reason = None
            if response.candidates:
                reason = getattr(response.candidates[0], "finish_reason", None)
            logger.warning(f"Gemini returned no text (finish_reason={reason}); using ''.")
            return ""
        return clean_prediction(text)

    def _generate(self, inputs: list[str], batched_examples: list[list[Any]]) -> list[str]:
        responses = []
        for input_text, examples in zip(inputs, batched_examples):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=build_user_message(input_text, examples),
                config=self.config,
            )
            responses.append(self._extract_text(response))
        return responses

    async def _generate_one_async(
        self, input_text: str, examples: list[Any], semaphore: asyncio.Semaphore
    ) -> str:
        contents = build_user_message(input_text, examples)
        async with semaphore:
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=contents, config=self.config
            )
        return self._extract_text(response)

    async def _generate_async(
        self, inputs: list[str], batched_examples: list[list[Any]]
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._generate_one_async(text, examples, semaphore)
            for text, examples in zip(inputs, batched_examples)
        ]
        return await asyncio.gather(*tasks)  # order preserved

    def generate(
        self, inputs: list[str], batched_examples: list[list[Any]], **kwargs
    ) -> list[str]:
        if not self.use_async:
            return self._generate(inputs, batched_examples)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():  # notebooks
            nest_asyncio.apply()
            return loop.run_until_complete(self._generate_async(inputs, batched_examples))
        return asyncio.run(self._generate_async(inputs, batched_examples))


class HuggingFaceGenerator(BaseGenerator):
    """Local causal-LM generator (optionally with a LoRA adapter from train_post_correction).

    Point `tokenizer_path` at the *saved training run* (models/<run>/result) so the
    chat template matches what the adapter was trained on; the stock Llama-3.2
    template injects a date preamble that create_llama3_tokenizer does not.
    """

    def __init__(
        self,
        system_prompt: str,
        model_path: str,
        tokenizer_path: str | None = None,
        peft_path: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        do_sample: bool = True,
        max_input_length: int = 2048,
    ):
        self.system_prompt = system_prompt
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.peft_path = peft_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_input_length = max_input_length

        logger.info(f"Loading tokenizer: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "left"  # required for batched generate
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading LLM model: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if self.peft_path:
            if not os.path.exists(self.peft_path):
                # Silently evaluating the base model when the adapter path is
                # wrong is exactly the kind of error that's invisible in metrics.
                raise FileNotFoundError(f"peft_path does not exist: {self.peft_path}")
            logger.info(f"Applying LoRA weights from: {self.peft_path}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_path)

        self.model.eval()

    def generate(
        self, inputs: list[str], batched_examples: list[list[Any]], **kwargs
    ) -> list[str]:
        prompts = []
        for input_text, examples in zip(inputs, batched_examples):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_message(input_text, examples)},
            ]
            prompts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            max_length=self.max_input_length,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=self.do_sample,
        )
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**encoded, **gen_kwargs)

        # Left padding => every prompt occupies the same number of positions.
        prompt_length = encoded.input_ids.shape[-1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, prompt_length:], skip_special_tokens=True
        )
        return [clean_prediction(t) for t in decoded]
