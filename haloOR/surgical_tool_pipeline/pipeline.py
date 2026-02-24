"""End-to-end orchestration across robot, audio, detector, and classifier stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .helpers import (
    collect_images,
    ensure_output_dirs,
    image_id_for_path,
    normalize_extensions,
    write_flat_csv,
)
from .robot import resolve_input_path


def _normalize_label_for_match(label: str) -> str:
    """Normalize labels so audio/classifier names can be matched reliably."""
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


def _find_first_matching_bbox(image_records: List[Dict[str, Any]], target_label: str) -> Optional[Dict[str, Any]]:
    """Return the first bbox prediction that matches the requested target label."""
    normalized_target = _normalize_label_for_match(target_label)
    if not normalized_target or normalized_target == "unknown":
        return None

    for image in image_records:
        for det in image.get("detections", []):
            pred = det.get("prediction")
            if not pred:
                continue
            pred_label = _normalize_label_for_match(str(pred.get("label", "")))
            if pred_label != normalized_target:
                continue
            return {
                "target_instrument": normalized_target,
                "image_id": image.get("image_id"),
                "image_path": image.get("image_path"),
                "bbox_id": det.get("bbox_id"),
                "bbox_xyxy": det.get("bbox_xyxy"),
                "bbox_centroid_xy": det.get("bbox_centroid_xy"),
                "crop_path": det.get("crop_path"),
                "prediction": pred,
            }
    return None


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run one full pipeline pass and write merged reports."""
    from .audio import run_audio_instrument_inference
    from .classifier import load_classifier_metadata, predict_crops
    from .detector import detect_and_crop_one_image

    input_path, robot_meta = resolve_input_path(config.robot, config.input_path)
    output_dir = Path(config.output_dir).expanduser().resolve()
    model_dir = Path(config.classifier.model_dir).expanduser().resolve()
    audio_result: Optional[Dict[str, Any]] = None
    if config.audio.enabled:
        if not config.audio.input_path:
            raise ValueError("Audio mode is enabled but no audio input path was provided.")
        audio_result = run_audio_instrument_inference(
            audio_path=str(config.audio.input_path),
            device=str(config.audio.device),
            model_id=str(config.audio.model_id),
            chunk_length_s=float(config.audio.chunk_length_s),
            stride_length_s=(
                float(config.audio.stride_length_s) if config.audio.stride_length_s is not None else None
            ),
        )
        print(
            "[audio] instrument="
            f"{audio_result.get('instrument', 'unknown')} file={audio_result.get('audio_path', '')}"
        )

    extensions = normalize_extensions(config.extensions)
    image_paths = collect_images(input_path, recursive=config.recursive, extensions=extensions)
    if not image_paths:
        raise RuntimeError(f"No matching images found under: {input_path}")

    dirs = ensure_output_dirs(output_dir, save_masks=bool(config.save_masks))

    image_records: List[Dict[str, Any]] = []
    crop_paths: List[str] = []
    crop_to_detection: Dict[str, Dict[str, Any]] = {}

    print(f"[detect] scanning {len(image_paths)} image(s)")
    for idx, image_path in enumerate(image_paths, start=1):
        image_id = image_id_for_path(image_path, input_path, idx)
        image_record, per_image_crop_paths = detect_and_crop_one_image(
            image_path=image_path,
            image_id=image_id,
            dirs=dirs,
            cfg=config.detector,
            save_masks=bool(config.save_masks),
        )
        image_records.append(image_record)

        for det in image_record["detections"]:
            if det["crop_exists"]:
                crop_to_detection[det["crop_path"]] = det

        crop_paths.extend(per_image_crop_paths)
        print(
            f"[detect {idx}/{len(image_paths)}] {image_path.name} "
            f"bboxes={image_record['bbox_count']} crops={len(per_image_crop_paths)}"
        )

    classifier, id2label, model_id = load_classifier_metadata(
        model_dir=model_dir,
        model_id_override=config.classifier.model_id_override,
    )

    predictions_by_path, class_names, skipped = predict_crops(
        crop_paths=crop_paths,
        classifier=classifier,
        id2label=id2label,
        model_id=model_id,
        device_name=config.classifier.device,
        batch_size=int(config.classifier.batch_size),
        num_workers=int(config.classifier.num_workers),
        use_amp=bool(config.classifier.use_amp),
        candidate_labels=list(config.classifier.candidate_labels),
    )

    for path, det in crop_to_detection.items():
        if path in predictions_by_path:
            det["prediction"] = predictions_by_path[path]

    audio_target = audio_result.get("instrument", "unknown") if audio_result else "unknown"
    first_audio_matched_bbox = _find_first_matching_bbox(image_records=image_records, target_label=audio_target)

    skipped_log_path = dirs["reports"] / "classification_skipped.log"
    if skipped:
        with skipped_log_path.open("w", encoding="utf-8") as f:
            for row in skipped:
                f.write(f"{row.get('path', '')}\t{row.get('error', '')}\n")

    total_bboxes = sum(len(item["detections"]) for item in image_records)
    total_predicted = sum(
        1 for image in image_records for det in image["detections"] if det.get("prediction") is not None
    )

    merged_output = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "model_id": model_id,
        "robot_mode": bool(config.robot.enabled),
        "robot_api_meta": robot_meta,
        "audio_mode": bool(config.audio.enabled),
        "audio_result": audio_result,
        "audio_target_instrument": audio_target,
        "first_audio_matched_bbox": first_audio_matched_bbox,
        "candidate_labels_requested": list(config.classifier.candidate_labels),
        "summary": {
            "images": len(image_records),
            "bboxes": int(total_bboxes),
            "classified_bboxes": int(total_predicted),
            "unclassified_bboxes": int(total_bboxes - total_predicted),
            "skipped_crops": int(len(skipped)),
            "audio_match_found": bool(first_audio_matched_bbox),
        },
        "class_labels": class_names,
        "images": image_records,
    }

    merged_json_path = dirs["reports"] / "merged_bbox_predictions.json"
    with merged_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_output, f, indent=2)
    if first_audio_matched_bbox is not None:
        first_match_path = dirs["reports"] / "first_audio_matched_bbox.json"
        with first_match_path.open("w", encoding="utf-8") as f:
            json.dump(first_audio_matched_bbox, f, indent=2)

    merged_csv_path = dirs["reports"] / "merged_bbox_predictions.csv"
    write_flat_csv(merged_csv_path, image_records=image_records, class_names=class_names)

    print("[done]")
    print(f"  merged json: {merged_json_path}")
    print(f"  merged csv : {merged_csv_path}")
    if first_audio_matched_bbox is not None:
        print(f"  first bbox : {dirs['reports'] / 'first_audio_matched_bbox.json'}")
    if skipped:
        print(f"  skipped log: {skipped_log_path}")

    return merged_output
