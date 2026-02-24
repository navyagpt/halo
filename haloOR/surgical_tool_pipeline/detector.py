"""Image-level detection wrapper around the classical CV detector module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import bboxsi_main
from .config import DetectorConfig


def detect_and_crop_one_image(
    image_path: Path,
    image_id: str,
    dirs: Dict[str, Path],
    cfg: DetectorConfig,
    save_masks: bool,
) -> Tuple[Dict[str, Any], List[str]]:
    """Detect bboxes for one image, write artifacts, and return metadata plus crop paths."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]

    roi_mask, roi_meta = bboxsi_main.extract_roi_mask(
        image,
        v_dark=cfg.v_dark,
        roi_close_k=cfg.roi_close_k,
        roi_open_k=cfg.roi_open_k,
        auto_thresholds=cfg.auto_thresholds,
        manual_v_dark_override=cfg.manual_v_dark_override,
    )

    obj_mask, obj_meta = bboxsi_main.extract_obj_mask_instruments(
        image,
        roi_mask,
        delta_v=cfg.delta_v,
        s_min=cfg.s_min,
        obj_close_k=cfg.obj_close_k,
        obj_open_k=cfg.obj_open_k,
        auto_thresholds=cfg.auto_thresholds,
    )

    split_meta = {"split_components": 0}
    if cfg.split_merged:
        obj_mask, split_meta = bboxsi_main.split_merged_components(
            obj_mask,
            image.shape,
            min_area=cfg.min_area,
            erode_max_iter=cfg.split_erode_max_iter,
        )

    bboxes, bbox_meta = bboxsi_main.extract_bboxes_from_mask(
        obj_mask,
        image.shape,
        min_area=cfg.min_area,
        max_area_frac=cfg.max_area_frac,
        max_aspect=cfg.max_aspect,
    )

    vis = bboxsi_main.draw_debug(
        image,
        bboxes,
        roi_found=bool(roi_meta["roi_found"]),
        used_fallback=bool(obj_meta["used_fallback"]),
    )
    cv2.imwrite(str(dirs["vis"] / f"{image_id}_boxed.png"), vis)

    if save_masks:
        cv2.imwrite(str(dirs["masks"] / f"{image_id}_roi.png"), roi_mask)
        cv2.imwrite(str(dirs["masks"] / f"{image_id}_obj.png"), obj_mask)

    bboxsi_main.save_bbox_crops(
        image,
        bboxes,
        dirs["crops"],
        image_id,
        crop_pad=cfg.crop_pad,
    )

    detections: List[Dict[str, Any]] = []
    crop_paths: List[str] = []
    for bbox_name, box in bboxes.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        centroid_x = (x1 + x2) / 2.0
        centroid_y = (y1 + y2) / 2.0

        crop_path = dirs["crops"] / image_id / f"{bbox_name}.png"
        crop_exists = crop_path.exists()

        detection = {
            "bbox_id": bbox_name,
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_centroid_xy": [centroid_x, centroid_y],
            "crop_path": str(crop_path.resolve()),
            "crop_exists": bool(crop_exists),
            "prediction": None,
        }
        detections.append(detection)
        if crop_exists:
            crop_paths.append(str(crop_path.resolve()))

    image_record: Dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(image_path.resolve()),
        "image_name": image_path.name,
        "width": int(w),
        "height": int(h),
        "roi_area_frac": float(roi_meta["roi_area_frac"]),
        "bbox_count": int(len(bboxes)),
        "detections": detections,
        "detector_meta": {
            "roi": roi_meta,
            "objects": obj_meta,
            "bboxes": bbox_meta,
            "split": split_meta,
        },
    }

    per_image_json_path = dirs["json"] / f"{image_id}.json"
    with per_image_json_path.open("w", encoding="utf-8") as f:
        json.dump(image_record, f, indent=2)

    return image_record, crop_paths
