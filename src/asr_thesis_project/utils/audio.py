"""Small audio helpers shared by the RAG index builders and inference scripts."""

from pathlib import Path

import torch
import torchaudio
from torchcodec.encoders import AudioEncoder


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """(channels, T) or (T,) -> (T,)"""
    if wav.ndim == 2:
        wav = wav.mean(dim=0) if wav.shape[0] > 1 else wav.squeeze(0)
    return wav


def resample(wav: torch.Tensor, original_sampling_rte: int, target_sampling_rate: int) -> torch.Tensor:
    if original_sampling_rte == target_sampling_rate:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=original_sampling_rte, new_freq=target_sampling_rate)


def decode_audio(audio, target_sr: int) -> tuple[torch.Tensor, float]:
    """Decode one `datasets` Audio cell (a torchcodec AudioDecoder) to a mono
    waveform at `target_sr`. Returns (waveform, duration)."""
    samples = audio.get_all_samples()
    wav = to_mono(samples.data)
    duration = wav.shape[-1] / samples.sample_rate
    return resample(wav, samples.sample_rate, target_sr), duration


def decode_batch_audio(audio_column, target_sr: int) -> list[torch.Tensor]:
    """The per-batch decode loop used by every RAG/ASR map function."""
    return [decode_audio(audio, target_sr)[0] for audio in audio_column]

def write(path: str | Path, audio: torch.Tensor, sample_rate: int, overwrite: bool = False) -> Path:
    """Write a mono waveform to `path` as WAV via torchcodec. Skips existing
    files unless `overwrite`, so index rebuilds are resumable."""
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = to_mono(audio).to(torch.float32).unsqueeze(0)
    AudioEncoder(audio, sample_rate=sample_rate).to_file(path)
    return path
