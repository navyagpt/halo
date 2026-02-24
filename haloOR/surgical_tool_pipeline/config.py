"""Typed configuration objects shared across CLI and API entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
DEFAULT_OUTPUT_DIR = "pipeline_outputs"
DEFAULT_MODEL_DIR = str(Path(__file__).resolve().parent / "models" / "best_model")


@dataclass
class DetectorConfig:
    """Classical CV detector thresholds and post-processing settings."""

    v_dark: int = 110
    roi_close_k: int = 31
    roi_open_k: int = 17
    delta_v: int = 35
    s_min: int = 25
    obj_close_k: int = 23
    obj_open_k: int = 7
    min_area: int | None = None
    max_area_frac: float = 0.60
    max_aspect: float = 20.0
    split_merged: bool = True
    split_erode_max_iter: int = 8
    auto_thresholds: bool = True
    manual_v_dark_override: bool = False
    crop_pad: int = 0


@dataclass
class ClassifierConfig:
    """Embedding/classifier inference configuration."""

    model_dir: str = DEFAULT_MODEL_DIR
    model_id_override: str | None = None
    device: str = "auto"
    batch_size: int = 128
    num_workers: int = 4
    use_amp: bool = True
    candidate_labels: List[str] = field(default_factory=list)


@dataclass
class RobotConfig:
    """Robot API options used to fetch an input image path dynamically."""

    enabled: bool = False
    api_url: str | None = None
    api_method: str = "POST"
    timeout_sec: float = 10.0
    payload: Dict[str, Any] = field(default_factory=dict)
    response_image_key: str = "image_path"
    response_image_list_key: str = "image_paths"


@dataclass
class AudioConfig:
    """Audio transcription and instrument extraction configuration."""

    enabled: bool = False
    input_path: str | None = None
    model_id: str = "google/medasr"
    device: str = "auto"
    chunk_length_s: float = 0.0
    stride_length_s: float | None = None


@dataclass
class PipelineConfig:
    """Top-level runtime configuration for one pipeline run."""

    input_path: str | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    recursive: bool = False
    extensions: List[str] = field(default_factory=lambda: list(DEFAULT_EXTENSIONS))
    save_masks: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_MODEL_DIR",
    "DetectorConfig",
    "ClassifierConfig",
    "RobotConfig",
    "AudioConfig",
    "PipelineConfig",
]
