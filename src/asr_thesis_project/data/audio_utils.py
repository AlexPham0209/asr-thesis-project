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


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)


def decode_audio(audio, target_sr: int) -> tuple[torch.Tensor, float]:
    """Decode one `datasets` Audio cell (a torchcodec AudioDecoder) to a mono
    waveform at `target_sr`. Returns (waveform, duration_seconds)."""
    samples = audio.get_all_samples()
    wav = to_mono(samples.data)
    duration_s = wav.shape[-1] / samples.sample_rate
    return resample(wav, samples.sample_rate, target_sr), duration_s


def decode_batch_audio(audio_column, target_sr: int) -> list[torch.Tensor]:
    """The per-batch decode loop used by every RAG/ASR map function."""
    return [decode_audio(audio, target_sr)[0] for audio in audio_column]


def write_wav(path: str | Path, wav: torch.Tensor, sample_rate: int, overwrite: bool = False) -> Path:
    """Write a mono waveform to `path` as WAV via torchcodec. Skips existing
    files unless `overwrite`, so index rebuilds are resumable."""
    path = Path(path)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = to_mono(wav).to(torch.float32).unsqueeze(0)
    AudioEncoder(wav, sample_rate=sample_rate).to_file(path)
    return path
