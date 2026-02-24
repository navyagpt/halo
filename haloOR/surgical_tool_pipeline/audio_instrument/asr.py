"""MedASR loading and transcription utilities with compatibility patches."""

from __future__ import annotations

import inspect
import logging
import os
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import pipeline

from .utils import load_audio

REQUIRED_MODEL_ID = "google/medasr"


def _hf_auth_kwargs() -> dict:
    """Collect optional Hugging Face token kwargs from supported env var names."""
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        return {}
    return {"token": token}


def _patch_torch_extract_fbank_features(target: object) -> None:
    """Patch LASR feature extractor signatures that miss the `center` argument."""
    if getattr(target, "_medasr_center_compat_patched", False):
        return

    method = getattr(target, "_torch_extract_fbank_features", None)
    if method is None:
        return

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return

    if "center" in signature.parameters:
        setattr(target, "_medasr_center_compat_patched", True)
        return

    if inspect.ismethod(method):
        original_method = method

        def _patched_torch_extract_fbank_features(waveform, device="cpu", center=True):
            del center
            return original_method(waveform, device)

    else:
        original_method = method

        def _patched_torch_extract_fbank_features(self, waveform, device="cpu", center=True):
            del center
            return original_method(self, waveform, device)

    setattr(target, "_torch_extract_fbank_features", _patched_torch_extract_fbank_features)
    setattr(target, "_medasr_center_compat_patched", True)


def _apply_lasr_feature_extractor_compat_patch() -> None:
    """
    Compatibility patch for transformers LASR feature extractor signatures.

    Some transformers builds call:
      _torch_extract_fbank_features(input_features, device, center)
    while the LASR implementation only accepts:
      _torch_extract_fbank_features(waveform, device="cpu")
    """
    try:
        from transformers.models.lasr.feature_extraction_lasr import LasrFeatureExtractor
    except Exception:  # noqa: BLE001
        return

    _patch_torch_extract_fbank_features(LasrFeatureExtractor)


def _patch_pipeline_feature_extractor_compat(asr_pipe: object) -> None:
    """Apply feature extractor signature patch to a constructed ASR pipeline."""
    feature_extractor = getattr(asr_pipe, "feature_extractor", None)
    if feature_extractor is None:
        return
    _patch_torch_extract_fbank_features(feature_extractor.__class__)
    _patch_torch_extract_fbank_features(feature_extractor)


def _resolve_device(device: str = "auto") -> Tuple[torch.device, int]:
    """Resolve a torch device plus transformers pipeline device integer."""
    if device == "cpu":
        return torch.device("cpu"), -1
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda"), 0
    if torch.cuda.is_available():
        return torch.device("cuda"), 0
    return torch.device("cpu"), -1


@lru_cache(maxsize=8)
def get_asr_pipeline(
    model_id: str = REQUIRED_MODEL_ID,
    device: str = "auto",
    chunk_length_s: float = 0.0,
    stride_length_s: Optional[float] = None,
):
    """Create and cache the ASR pipeline. Loads ONLY google/medasr."""
    if model_id != REQUIRED_MODEL_ID:
        raise ValueError(f"Only '{REQUIRED_MODEL_ID}' is allowed. Received: {model_id}")
    _apply_lasr_feature_extractor_compat_patch()

    torch_device, hf_device = _resolve_device(device)
    logging.info("Loading ASR model '%s' on device=%s", model_id, torch_device)

    kwargs = {
        "task": "automatic-speech-recognition",
        "model": model_id,
        "device": hf_device,
    }
    kwargs.update(_hf_auth_kwargs())

    if chunk_length_s and chunk_length_s > 0:
        kwargs["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            kwargs["stride_length_s"] = stride_length_s

    try:
        asr_pipe = pipeline(**kwargs)
    except TypeError:
        # Backward compatibility for older transformers versions.
        if "token" in kwargs:
            legacy_kwargs = dict(kwargs)
            legacy_kwargs["use_auth_token"] = legacy_kwargs.pop("token")
            asr_pipe = pipeline(**legacy_kwargs)
        else:
            raise
    _patch_pipeline_feature_extractor_compat(asr_pipe)
    return asr_pipe


class MedASRTranscriber:
    """Transcriber wrapper for google/medasr with cached model loading."""

    def __init__(
        self,
        model_id: str = REQUIRED_MODEL_ID,
        device: str = "auto",
        chunk_length_s: float = 0.0,
        stride_length_s: Optional[float] = None,
        target_sampling_rate: int = 16000,
    ) -> None:
        if model_id != REQUIRED_MODEL_ID:
            raise ValueError(f"Only '{REQUIRED_MODEL_ID}' is allowed. Received: {model_id}")
        self.model_id = model_id
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.target_sampling_rate = target_sampling_rate

    @property
    def pipe(self):
        """Lazily loaded/cached Hugging Face ASR pipeline."""
        return get_asr_pipeline(
            model_id=self.model_id,
            device=self.device,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
        )

    def transcribe_array(self, audio: np.ndarray, sampling_rate: int = 16000) -> str:
        """Transcribe an in-memory audio array."""
        asr_pipe = self.pipe
        payload = {"array": audio, "sampling_rate": sampling_rate}
        try:
            output = asr_pipe(payload)
        except TypeError as exc:
            if "_torch_extract_fbank_features" not in str(exc):
                raise
            _patch_pipeline_feature_extractor_compat(asr_pipe)
            output = asr_pipe(payload)
        if isinstance(output, dict):
            return str(output.get("text", "")).strip()
        return str(output).strip()

    def transcribe_file(self, audio_path: str) -> str:
        """Load and transcribe one audio file path."""
        audio, sr = load_audio(audio_path, target_sampling_rate=self.target_sampling_rate)
        return self.transcribe_array(audio, sampling_rate=sr)
