#!/usr/bin/env python3
"""
OpenCV medicine bounding-box extraction for black-cloth ROI.

Run:
    python main.py --input_dir . --output_dir outputs

Optional:
    python main.py --input_dir . --output_dir outputs --recursive --save_masks --auto_thresholds
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Thresholds:
    """Resolved thresholds for one image."""

    v_dark: int
    v_obj: int
    s_obj: int
    v_col: int


def clamp(value: float, lo: int, hi: int) -> int:
    """Clamp numeric value to integer bounds."""
    return int(max(lo, min(hi, round(value))))


def odd_kernel_size(size: int) -> int:
    """Return odd kernel size >= 3."""
    if size < 3:
        size = 3
    if size % 2 == 0:
        size += 1
    return size


def list_images(input_dir: Path, recursive: bool, extensions: Sequence[str]) -> List[Path]:
    """List image paths in input directory."""
    ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [p for p in candidates if p.is_file() and p.suffix.lower() in ext_set]
    images.sort()
    return images


def otsu_threshold(values: np.ndarray, fallback: int) -> int:
    """Compute Otsu threshold from 1D uint8 values."""
    if values.size < 32:
        return fallback
    t, _ = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def resolve_thresholds(hsv: np.ndarray, roi_mask: Optional[np.ndarray], args: argparse.Namespace) -> Thresholds:
    """Resolve dark/obj thresholds with optional auto mode."""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    if not args.auto_thresholds:
        return Thresholds(
            v_dark=clamp(args.v_dark, 25, 120),
            v_obj=clamp(args.v_obj, 70, 250),
            s_obj=clamp(args.s_obj, 20, 250),
            v_col=clamp(args.v_col, 40, 240),
        )

    v_all = v.reshape(-1)
    v_dark_auto = clamp(np.percentile(v_all, 25), 40, 95)

    if roi_mask is not None and np.count_nonzero(roi_mask) > 0:
        v_roi = v[roi_mask > 0]
        s_roi = s[roi_mask > 0]
    else:
        v_roi = v_all
        s_roi = s.reshape(-1)

    v_obj_auto = clamp(max(np.percentile(v_roi, 85), otsu_threshold(v_roi.astype(np.uint8), 140)), 100, 240)
    s_obj_auto = clamp(np.percentile(s_roi, 70), 40, 200)
    v_col_auto = clamp(np.percentile(v_roi, 60), 65, 210)

    return Thresholds(v_dark=v_dark_auto, v_obj=v_obj_auto, s_obj=s_obj_auto, v_col=v_col_auto)


def extract_black_roi(hsv: np.ndarray, th: Thresholds, s_dark_max: int) -> Tuple[np.ndarray, bool, float]:
    """
    Stage A: extract largest black-cloth ROI.
    Returns (roi_mask, roi_found, roi_ratio).
    """
    h, w = hsv.shape[:2]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    dark_mask = ((v <= th.v_dark) & (s <= s_dark_max)).astype(np.uint8) * 255

    k_base = odd_kernel_size(int(round(min(h, w) * 0.01)))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel_size(k_base + 4), odd_kernel_size(k_base + 4)))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_base, k_base))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k_close)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if num_labels <= 1:
        full = np.full((h, w), 255, dtype=np.uint8)
        return full, False, 1.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas) + 1)
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    roi_ratio = largest_area / float(h * w)

    # Confidence threshold for selecting black cloth.
    if roi_ratio < 0.06:
        full = np.full((h, w), 255, dtype=np.uint8)
        return full, False, 1.0

    roi_mask = np.where(labels == largest_idx, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(roi_mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    roi_ratio_final = np.count_nonzero(filled) / float(h * w)
    return filled, True, roi_ratio_final


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill internal mask holes via flood fill."""
    h, w = binary_mask.shape
    flood = binary_mask.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    inv_flood = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary_mask, inv_flood)


def extract_objects(hsv: np.ndarray, roi_mask: np.ndarray, th: Thresholds) -> np.ndarray:
    """Stage B: object mask inside ROI from bright + colored criteria."""
    h, w = hsv.shape[:2]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    bright = v >= th.v_obj
    colored = (s >= th.s_obj) & (v >= th.v_col)
    obj = ((bright | colored) & (roi_mask > 0)).astype(np.uint8) * 255

    k_open = odd_kernel_size(int(round(min(h, w) * 0.004)))
    k_close = odd_kernel_size(int(round(min(h, w) * 0.007)))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, open_kernel)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, close_kernel)
    obj = fill_holes(obj)

    return cv2.bitwise_and(obj, roi_mask)


def extract_boxes(
    obj_mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    roi_mask: np.ndarray,
    min_area_cli: Optional[int],
    max_aspect: float,
) -> Tuple[List[Tuple[int, int, int, int]], int, int]:
    """Stage C: connected components to filtered/sorted boxes."""
    h, w = image_shape[:2]
    roi_area = int(np.count_nonzero(roi_mask))
    min_area_auto = clamp(0.0005 * h * w, 20, 5000)
    min_area = int(min_area_cli) if min_area_cli is not None else min_area_auto
    max_area = int(max(roi_area * 0.35, min_area * 4))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    total_components = max(0, num_labels - 1)
    boxes: List[Tuple[int, int, int, int]] = []

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        ww = int(stats[idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue

        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        if aspect > max_aspect:
            continue

        x1, y1 = x, y
        x2, y2 = x + ww - 1, y + hh - 1
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, total_components, len(boxes)


def draw_boxes(image_bgr: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    """Create debug image with boxes and labels."""
    vis = image_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = y1 - 8 if y1 > 18 else y1 + 18
        cv2.putText(vis, f"bbox{i}", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    return vis


def make_payload(image_name: str, w: int, h: int, boxes: Sequence[Tuple[int, int, int, int]]) -> Dict[str, object]:
    """Build JSON payload for one image."""
    bboxes: Dict[str, List[int]] = {}
    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        bboxes[f"bbox{i}"] = [int(x1), int(y1), int(x2), int(y2)]
    return {
        "image": image_name,
        "width": int(w),
        "height": int(h),
        "bboxes": bboxes,
    }


def ensure_dirs(output_dir: Path) -> Tuple[Path, Path, Path]:
    """Create output directory tree."""
    json_dir = output_dir / "json"
    vis_dir = output_dir / "vis"
    mask_dir = output_dir / "masks"
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return json_dir, vis_dir, mask_dir


def process_image(
    image_path: Path,
    input_root: Path,
    args: argparse.Namespace,
    json_root: Path,
    vis_root: Path,
    mask_root: Path,
) -> None:
    """Process one image end-to-end and save outputs."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    th_pre = resolve_thresholds(hsv, None, args)
    roi_mask, roi_found, roi_ratio = extract_black_roi(hsv, th_pre, clamp(args.s_dark_max, 20, 200))
    th = resolve_thresholds(hsv, roi_mask, args)
    if th.v_dark != th_pre.v_dark:
        roi_mask, roi_found, roi_ratio = extract_black_roi(hsv, th, clamp(args.s_dark_max, 20, 200))

    obj_mask = extract_objects(hsv, roi_mask, th)
    boxes, n_before, n_after = extract_boxes(
        obj_mask=obj_mask,
        image_shape=image.shape,
        roi_mask=roi_mask,
        min_area_cli=args.min_area,
        max_aspect=args.max_aspect,
    )

    print(
        f"[INFO] {image_path.name} | ROI found: {roi_found} | ROI area: {100.0 * roi_ratio:.2f}% | "
        f"components: {n_before} -> {n_after} | final bboxes: {len(boxes)} | "
        f"v_dark={th.v_dark}, v_obj={th.v_obj}, s_obj={th.s_obj}, v_col={th.v_col}"
    )
    if not roi_found:
        print(f"[WARN] {image_path.name} ROI not confident; used full-image fallback.")

    rel = image_path.relative_to(input_root)
    stem_rel = rel.with_suffix("")

    payload = make_payload(image_path.name, w, h, boxes)
    json_path = json_root / rel.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    vis = draw_boxes(image, boxes)
    vis_path = vis_root / stem_rel.parent / f"{stem_rel.name}_boxed.png"
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), vis)

    if args.save_masks or len(boxes) == 0:
        roi_path = mask_root / stem_rel.parent / f"{stem_rel.name}_roi.png"
        obj_path = mask_root / stem_rel.parent / f"{stem_rel.name}_objmask.png"
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(roi_path), roi_mask)
        cv2.imwrite(str(obj_path), obj_mask)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Detect medicines inside black cloth ROI and export bounding boxes."
    )
    parser.add_argument("--input_dir", type=str, default=".", help="Input image directory.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search input directory.")
    parser.add_argument("--save_masks", action="store_true", help="Save ROI/object masks for debugging.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        help="Image extensions to process.",
    )

    parser.add_argument("--v_dark", type=int, default=70, help="Dark threshold on V channel for black ROI.")
    parser.add_argument("--s_dark_max", type=int, default=120, help="Max S for dark ROI pixels.")
    parser.add_argument("--v_obj", type=int, default=135, help="V threshold for bright tablets.")
    parser.add_argument("--s_obj", type=int, default=60, help="S threshold for colored capsules.")
    parser.add_argument("--v_col", type=int, default=80, help="V threshold for colored capsules.")
    parser.add_argument(
        "--min_area",
        type=int,
        default=None,
        help="Minimum area in pixels per component (auto if omitted).",
    )
    parser.add_argument(
        "--max_aspect",
        type=float,
        default=4.5,
        help="Maximum allowed aspect ratio (long side / short side).",
    )
    parser.add_argument(
        "--auto_thresholds",
        action="store_true",
        help="Enable percentile/Otsu threshold adaptation per image.",
    )
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    json_dir, vis_dir, mask_dir = ensure_dirs(output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = list_images(input_dir, args.recursive, args.extensions)
    if not image_paths:
        print(f"[WARN] No images found in {input_dir} with extensions={args.extensions}")
        return

    print(f"[INFO] Processing {len(image_paths)} image(s) from {input_dir}")
    for img_path in image_paths:
        process_image(
            image_path=img_path,
            input_root=input_dir,
            args=args,
            json_root=json_dir,
            vis_root=vis_dir,
            mask_root=mask_dir,
        )
    print(f"[INFO] Done. JSON: {json_dir} | VIS: {vis_dir} | MASKS: {mask_dir}")


if __name__ == "__main__":
    main()
