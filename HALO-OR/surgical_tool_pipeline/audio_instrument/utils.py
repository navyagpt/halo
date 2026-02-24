"""Audio IO helpers used by the MedASR extraction workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def configure_logging(level: int = logging.INFO) -> None:
    """Configure default logging format for audio utilities."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def is_audio_file(path: Path) -> bool:
    """Return True when path extension is supported audio media."""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def collect_audio_paths(input_path: str) -> List[Path]:
    """Collect one or more supported audio files from a path."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if not is_audio_file(path):
            raise ValueError(f"Unsupported audio extension: {path.suffix}")
        return [path]

    files = [p for p in path.rglob("*") if p.is_file() and is_audio_file(p)]
    files.sort()
    return files


def _load_with_torchaudio(path: str) -> Tuple[torch.Tensor, int]:
    """Load audio with torchaudio and normalize tensor rank."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sr)


def _load_with_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    """Fallback loader when torchaudio backend support is unavailable."""
    try:
        import soundfile as sf
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("soundfile is not available as fallback audio reader") from exc

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.asarray(data, dtype=np.float32).T)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, int(sr)


def load_audio(path: str, target_sampling_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio, convert to mono, resample to target_sampling_rate.

    Returns:
        (audio_np_float32, target_sampling_rate)
    """
    torchaudio_error = None
    torchaudio_module = None
    try:
        import torchaudio

        torchaudio_module = torchaudio
        waveform, sr = _load_with_torchaudio(path)
    except Exception as exc:  # noqa: BLE001
        torchaudio_error = exc
        try:
            waveform, sr = _load_with_soundfile(path)
        except Exception as sf_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to read audio file '{path}'. torchaudio error: {torchaudio_error}; "
                f"soundfile error: {sf_exc}"
            ) from sf_exc

    waveform = waveform.float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sampling_rate:
        if torchaudio_module is not None:
            waveform = torchaudio_module.functional.resample(waveform, sr, target_sampling_rate)
        else:
            # soundfile fallback path without torchaudio: linear interpolation resampling.
            old_len = waveform.shape[-1]
            new_len = int(round(old_len * float(target_sampling_rate) / float(sr)))
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=new_len,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
        sr = target_sampling_rate

    audio_np = waveform.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    return audio_np, sr


def write_json(path: str, payload: dict) -> None:
    """Write a JSON payload, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    """Write iterable rows as JSONL, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
