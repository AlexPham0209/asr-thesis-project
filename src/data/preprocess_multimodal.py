from abc import ABC, abstractmethod
import asyncio
import io
import logging
import os
import warnings
from typing import Union, List

import chromadb
from google import genai
from google.genai import types
from peft import PeftModel
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
import nest_asyncio

from generator.generator import BaseGenerator

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")


class GeminiMultimodalGenerator(BaseGenerator):
    def __init__(
        self,
        system_prompt: str,
        model_name: str = "gemini-2.5-flash",
        max_concurrent: int = 10,
        use_async: bool = False,
        sample_rate: int = 16000, # Added sample rate for WAV conversion
    ):
        self.system_prompt = system_prompt
        self.max_concurrent = max_concurrent
        self.use_async = use_async
        self.client = genai.Client()
        self.model_name = model_name
        self.sample_rate = sample_rate

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Helper to convert a torchcodec tensor to WAV bytes in memory."""
        # torchaudio expects shape [channels, frames]. 
        # If it's a 1D tensor [frames], add a channel dimension.
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, self.sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()

    async def generate_prompt(
        self, dynamic_system_prompt: str, audio_tensor: torch.Tensor, semaphore: asyncio.Semaphore
    ) -> str:
        
        audio_bytes = self._tensor_to_wav_bytes(audio_tensor)
        content = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
            
        async with semaphore:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[content],
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_system_prompt,
                    temperature=0.2,
                    max_output_tokens=256,
                ),
            )
            return response.text or ""

    async def _generate_async(
        self, inputs: List[torch.Tensor], batched_examples: List[List[str]]
    ) -> List[str]:
        # Initialize semaphore inside the running event loop
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = []
        for audio_tensor, examples in zip(inputs, batched_examples):
            examples_str = "\n\n".join(examples)
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Use the following pairs of text and LaTeX as examples:\n{examples_str}"
            )
            tasks.append(
                self.generate_prompt(dynamic_system_prompt, audio_tensor, semaphore)
            )

        return await asyncio.gather(*tasks)

    def _generate(
        self, inputs: List[torch.Tensor], batched_examples: List[List[str]]
    ) -> List[str]:
        responses = []
        for audio_tensor, examples in zip(inputs, batched_examples):
            examples_str = "\n\n".join(examples)
            dynamic_system_prompt = (
                f"{self.system_prompt}\n\n"
                f"Use the following pairs of text and LaTeX as examples:\n{examples_str}"
            )
            
            audio_bytes = self._tensor_to_wav_bytes(audio_tensor)
            content = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[content],
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_system_prompt,
                    temperature=0.2,
                    max_output_tokens=256,
                )
            )
            
            responses.append(response.text or "")

        return responses

    def generate(
        self, inputs: List[torch.Tensor], batched_examples: List[List[str]]
    ) -> List[str]:

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