"""Programmatic entrypoints for running the full pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from .config import AudioConfig, ClassifierConfig, DetectorConfig, PipelineConfig, RobotConfig, DEFAULT_MODEL_DIR
from .helpers import parse_candidate_labels
from .pipeline import run_pipeline


def run_pipeline_in_memory(
    *,
    input_path: str | None = None,
    output_dir: str = "pipeline_outputs",
    model_dir: str = DEFAULT_MODEL_DIR,
    robot: bool = False,
    robot_api_url: str | None = None,
    robot_api_method: str = "POST",
    robot_timeout_sec: float = 10.0,
    robot_payload: Dict[str, Any] | None = None,
    robot_response_image_key: str = "image_path",
    robot_response_image_list_key: str = "image_paths",
    audio_input_path: str | None = None,
    audio_device: str = "auto",
    audio_model_id: str = "google/medasr",
    audio_chunk_length_s: float = 0.0,
    audio_stride_length_s: float | None = None,
    candidate_labels: List[str] | None = None,
    recursive: bool = False,
    device: str = "auto",
    batch_size: int = 128,
    num_workers: int = 4,
    save_masks: bool = False,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Build a pipeline config from Python arguments and run one inference pass."""
    config = PipelineConfig(
        input_path=input_path,
        output_dir=output_dir,
        recursive=bool(recursive),
        save_masks=bool(save_masks),
        detector=DetectorConfig(),
        classifier=ClassifierConfig(
            model_dir=model_dir,
            device=device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            use_amp=bool(use_amp),
            candidate_labels=parse_candidate_labels(candidate_labels),
        ),
        robot=RobotConfig(
            enabled=bool(robot),
            api_url=robot_api_url,
            api_method=robot_api_method,
            timeout_sec=float(robot_timeout_sec),
            payload=robot_payload or {},
            response_image_key=robot_response_image_key,
            response_image_list_key=robot_response_image_list_key,
        ),
        audio=AudioConfig(
            enabled=bool(audio_input_path),
            input_path=audio_input_path,
            device=audio_device,
            model_id=audio_model_id,
            chunk_length_s=float(audio_chunk_length_s),
            stride_length_s=(float(audio_stride_length_s) if audio_stride_length_s is not None else None),
        ),
    )
    return run_pipeline(config)


def run_pipeline_api_entrypoint(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    """API-friendly wrapper that normalizes request payload values before execution."""
    raw_labels = request_obj.get("candidate_labels")
    candidate_labels: List[str] | None
    if raw_labels is None:
        candidate_labels = None
    elif isinstance(raw_labels, list):
        candidate_labels = [str(x) for x in raw_labels]
    else:
        candidate_labels = [str(raw_labels)]

    robot_payload = request_obj.get("robot_payload")
    if robot_payload is not None and not isinstance(robot_payload, dict):
        raise ValueError("robot_payload must be a dictionary when provided.")

    num_workers_raw = request_obj.get("num_workers")
    num_workers_val = int(num_workers_raw) if num_workers_raw is not None else 4

    return run_pipeline_in_memory(
        input_path=request_obj.get("input_path"),
        output_dir=str(request_obj.get("output_dir") or "pipeline_outputs"),
        model_dir=str(request_obj.get("model_dir") or DEFAULT_MODEL_DIR),
        robot=bool(request_obj.get("robot", False)),
        robot_api_url=request_obj.get("robot_api_url"),
        robot_api_method=str(request_obj.get("robot_api_method", "POST")),
        robot_timeout_sec=float(request_obj.get("robot_timeout_sec", 10.0)),
        robot_payload=robot_payload,
        robot_response_image_key=str(request_obj.get("robot_response_image_key", "image_path")),
        robot_response_image_list_key=str(request_obj.get("robot_response_image_list_key", "image_paths")),
        audio_input_path=request_obj.get("audio_input_path"),
        audio_device=str(request_obj.get("audio_device", "auto")),
        audio_model_id=str(request_obj.get("audio_model_id", "google/medasr")),
        audio_chunk_length_s=float(request_obj.get("audio_chunk_length_s", 0.0)),
        audio_stride_length_s=(
            float(request_obj["audio_stride_length_s"])
            if request_obj.get("audio_stride_length_s") is not None
            else None
        ),
        candidate_labels=candidate_labels,
        recursive=bool(request_obj.get("recursive", False)),
        device=str(request_obj.get("device", "auto")),
        batch_size=int(request_obj.get("batch_size", 128)),
        num_workers=num_workers_val,
        save_masks=bool(request_obj.get("save_masks", False)),
        use_amp=bool(request_obj.get("use_amp", True)),
    )
