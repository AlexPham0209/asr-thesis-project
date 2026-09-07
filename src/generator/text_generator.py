from abc import ABC, abstractmethod
import asyncio
import logging
import os
import warnings
from typing import Union

import chromadb
from google import genai
from google.genai import types
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest_asyncio

from generator.generator import BaseGenerator

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")


class GeminiGenerator(BaseGenerator):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gemini-2.5-flash",
        max_concurrent: int = 10,
        use_async: bool = False,
    ):
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.use_async = use_async
        self.client = genai.Client()
        self.model_name = model_name

    async def generate_prompt(
        self, dynamic_system_prompt: str, prompt_text: str, semaphore: asyncio.Semaphore
    ) -> str:
        async with semaphore:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_system_prompt,
                    temperature=0.2,
                    max_output_tokens=256,
                ),
            )
            return response.text or ""

    async def _generate_async(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:
        # Initialize semaphore inside the running event loop
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = []
        for input_text, examples in zip(inputs, batched_examples):
            examples_str = "\n\n".join(examples)
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Use the following pairs of text and LaTeX as examples:\n{examples_str}"
            )
            tasks.append(
                self.generate_prompt(dynamic_system_prompt, input_text, semaphore)
            )

        return await asyncio.gather(*tasks)

    def _generate(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:
        responses = []
        for input_text, examples in zip(inputs, batched_examples):
            examples_str = "\n\n".join(examples)
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Use the following pairs of text and LaTeX as examples:\n{examples_str}"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=input_text,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_system_prompt,
                    temperature=0.2,
                    max_output_tokens=256,
                ),
            )
            # Fixed: Append text, not the response object
            responses.append(response.text or "")

        return responses

    def generate(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:

        if self.use_async:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            # Safe execution for Jupyter Notebooks or FastAPI
            if loop and loop.is_running():
                nest_asyncio.apply()
                return loop.run_until_complete(
                    self._generate_async(inputs, batched_examples)
                )
            else:
                return asyncio.run(self._generate_async(inputs, batched_examples))

        return self._generate(inputs, batched_examples)


class HuggingFaceGenerator(BaseGenerator):
    def __init__(
        self,
        system_prompt: str,
        model_path: str,
        tokenizer_path: str,
        peft_path: str | None = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.peft_path = peft_path
        self.system_prompt = system_prompt

        logger.info(f"Loading tokenizer: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading LLM model: {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        if self.peft_path and os.path.exists(self.peft_path):
            logger.info(f"Applying LoRA weights from: {self.peft_path}")
            self.model = PeftModel.from_pretrained(self.model, self.peft_path)

        self.model.eval()

    def generate(
        self, inputs: list[str], batched_examples: list[list[str]]
    ) -> list[str]:
        prompts = []
        for i, input_text in enumerate(inputs):
            examples_str = "\n\n".join(batched_examples[i])
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Use the following pairs of text and LaTeX as examples:\n{examples_str}"
            )

            message = [
                {"role": "system", "content": dynamic_system_prompt},
                {"role": "user", "content": input_text},
            ]

            prompt = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        encoded_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded_inputs,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.2,
                do_sample=True,
            )

        prompt_length = encoded_inputs.input_ids.shape[-1]
        generated_ids = outputs[:, prompt_length:]

        # Decode the whole batch
        res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return res
