import asyncio
import functools
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
import nest_asyncio
import torch
from torchcodec.encoders import AudioEncoder

from asr_thesis_project.generator.generator import (
    BaseGenerator,
    RetrievedExample,
    clean_prediction,
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("inference")

def tensor_to_bytes(audio: torch.Tensor, sample_rate: int, format="flac") -> bytes:
    """Encode a float waveform to in-memory WAV bytes.

    torchaudio.save() in >=2.9 delegates to torchcodec's AudioEncoder.to_file,
    which needs a real path and ignores `format`, so we call torchcodec directly.
    """
    audio = audio.detach().cpu().to(torch.float32)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    elif audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # downmix to mono

    encoded = AudioEncoder(audio, sample_rate=sample_rate).to_tensor(format=format)
    return encoded.numpy().tobytes()


@functools.lru_cache(maxsize=4096)
def load_audio_bytes(path: str) -> bytes:
    """Cached read of a stored example clip; the same neighbours recur often."""
    return Path(path).read_bytes()


class GeminiMultimodalGenerator(BaseGenerator):
    """Few-shot audio -> LaTeX with Gemini.

    Each request is: [example_1 audio, example_1 LaTeX, ..., example_k audio,
    example_k LaTeX, instruction, query audio]. Examples come from
    MultiModalRAG.retrieve as RetrievedExample objects whose `audio_path` is
    relative to `audio_root` (the vector-db directory).
    """

    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        model_name: str = "gemini-2.5-flash",
        sample_rate: int = 16000,
        audio_root: str | None = None,
        max_concurrent: int = 4,
        use_async: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        thinking_budget: int | None = 0,
        retry_attempts: int = 5,
        include_text_fallback: bool = True,
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.audio_root = audio_root
        self.max_concurrent = max_concurrent
        self.use_async = use_async
        self.include_text_fallback = include_text_fallback

        # Let the SDK back off on rate limits / transient 5xx instead of
        # killing the whole datasets.map run on the first 429.
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

        # Thinking tokens share max_output_tokens; with the default budget a
        # 256-512 cap regularly yields response.text == None.
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


    def _resolve_audio_path(self, example: RetrievedExample) -> str | None:
        if not example.audio_path:
            return None
        if os.path.isabs(example.audio_path) or self.audio_root is None:
            return example.audio_path
        return os.path.join(self.audio_root, example.audio_path)

    def _example_audio(self, example: RetrievedExample) -> bytes | None:
        path = self._resolve_audio_path(example)
        if path is None:
            return None
        try:
            return load_audio_bytes(path)
        except FileNotFoundError:
            logger.warning(f"Example audio missing: {path} (id={example.id})")
            return None

    def build_contents(
        self,
        query_audio: torch.Tensor,
        examples: list[RetrievedExample],
        hint: str | None = None,
    ) -> list[Any]:
        """[examples...] [ASR hint] [instruction] [query audio]"""
        parts: list[Any] = []

        if examples:
            parts.append(
                "Here are reference examples of spoken mathematics. Each example is an "
                "audio clip followed by its exact transcription with inline LaTeX."
            )
            for k, ex in enumerate(examples, start=1):
                audio = self._example_audio(ex)
                if audio is not None:
                    parts.append(f"Example {k} audio:")
                    parts.append(types.Part.from_bytes(data=audio, mime_type="audio/flac"))
                elif self.include_text_fallback:
                    parts.append(f"Example {k} ASR transcript: {ex.document}")
                parts.append(f"Example {k} transcription: {ex.target}")

        if hint:
            parts.append(
                f"A speech recogniser transcribed the following audio as: {hint}\n"
                "Use the audio as the source of truth and this transcript only as a hint."
            )

        parts.append(
            "Now transcribe the following audio in exactly the same format as the "
            "examples: plain text with mathematics written as inline LaTeX inside "
            "$...$. Output only the transcription, with no explanation or markdown."
        )
        parts.append(
            types.Part.from_bytes(
                data=tensor_to_bytes(query_audio, self.sample_rate),
                mime_type="audio/flac",
            )
        )
        return parts

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


    def _generate(
        self,
        inputs: list[torch.Tensor],
        batched_examples: list[list[RetrievedExample]],
        hints: list[str | None],
    ) -> list[str]:
        responses = []
        for query_audio, examples, hint in zip(inputs, batched_examples, hints):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=self.build_contents(query_audio, examples, hint),
                config=self.config,
            )
            responses.append(self._extract_text(response))
        return responses

    async def _generate_one_async(
        self,
        query_audio: torch.Tensor,
        examples: list[RetrievedExample],
        hint: str | None,
        semaphore: asyncio.Semaphore,
    ) -> str:
        contents = self.build_contents(query_audio, examples, hint)
        async with semaphore:
            response = await self.client.aio.models.generate_content(
                model=self.model_name, contents=contents, config=self.config
            )
        return self._extract_text(response)

    async def _generate_async(
        self,
        inputs: list[torch.Tensor],
        batched_examples: list[list[RetrievedExample]],
        hints: list[str | None],
    ) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._generate_one_async(audio, examples, hint, semaphore)
            for audio, examples, hint in zip(inputs, batched_examples, hints)
        ]
        return await asyncio.gather(*tasks)


    def generate(
        self,
        inputs: list[torch.Tensor],
        batched_examples: list[list[RetrievedExample]],
        hints: list[str] | None = None,
        **kwargs,
    ) -> list[str]:
        hints_list: list[str | None] = list(hints) if hints is not None else [None] * len(inputs)
        if len(hints_list) != len(inputs):
            raise ValueError(f"hints ({len(hints_list)}) and inputs ({len(inputs)}) differ in length")

        if not self.use_async:
            return self._generate(inputs, batched_examples, hints_list)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():  # notebooks
            nest_asyncio.apply()
            return loop.run_until_complete(
                self._generate_async(inputs, batched_examples, hints_list)
            )
        return asyncio.run(self._generate_async(inputs, batched_examples, hints_list))
