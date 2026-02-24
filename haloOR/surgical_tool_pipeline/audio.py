"""Audio-to-instrument orchestration for the main pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .audio_instrument.asr import MedASRTranscriber, REQUIRED_MODEL_ID
from .audio_instrument.extract import CANONICAL_LABELS, extract_instrument


def normalize_instrument_label(label: str) -> str:
    """Normalize ASR-extracted label text to a canonical instrument alias."""
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "forcep": "forceps",
        "forceps": "forceps",
        "hemostat": "hemostat",
        "hemostats": "hemostat",
        "scissor": "scissors",
        "scissors": "scissors",
        "scalpel": "scalpel",
    }
    return aliases.get(value, value)


def run_audio_instrument_inference(
    audio_path: str,
    device: str = "auto",
    model_id: str = REQUIRED_MODEL_ID,
    chunk_length_s: float = 0.0,
    stride_length_s: float | None = None,
) -> Dict[str, Any]:
    """Run MedASR transcription and rule-based instrument extraction for one audio file."""
    resolved_audio_path = Path(audio_path).expanduser().resolve()
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio path not found: {resolved_audio_path}")

    transcriber = MedASRTranscriber(
        model_id=model_id,
        device=device,
        chunk_length_s=chunk_length_s,
        stride_length_s=stride_length_s,
    )
    transcript = transcriber.transcribe_file(str(resolved_audio_path))
    extraction = extract_instrument(transcript)

    instrument = normalize_instrument_label(str(extraction.get("instrument", "unknown")))
    if instrument not in CANONICAL_LABELS:
        instrument = "unknown"

    return {
        "audio_path": str(resolved_audio_path),
        "transcript": transcript,
        "instrument": instrument,
        "matched_pattern": extraction.get("matched_pattern", ""),
        "all_matches": extraction.get("all_matches", []),
        "model_id": model_id,
        "device": device,
    }
