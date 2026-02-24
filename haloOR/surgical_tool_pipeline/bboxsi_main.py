#!/usr/bin/env python3
"""
Detect surgical instruments on a dark mat using classical CV (OpenCV + NumPy).

Pipeline:
1) Extract dark ROI (mat).
2) Extract instrument foreground inside ROI.
3) Apply fallback edge-based mask when needed.
4) Extract connected components and produce filtered bounding boxes.
5) Save JSON + visualization (+ optional masks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


DEFAULT_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def clamp(value: int, low: int, high: int) -> int:
    """Clamp integer to [low, high]."""
    return max(low, min(high, value))


def normalize_kernel_size(k: int) -> int:
    """Ensure morphology kernel size is odd and >= 1."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return k


def ellipse_kernel(k: int) -> np.ndarray:
    """Create an elliptical structuring element."""
    kk = normalize_kernel_size(k)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))


def find_images(input_dir: str, extensions: List[str] | None, recursive: bool) -> List[Path]:
    """Find image files by extension, optionally recursively."""
    root = Path(input_dir)
    exts = extensions if extensions else DEFAULT_EXTENSIONS
    ext_set = {
        (e.lower() if e.startswith(".") else f".{e.lower()}")
        for e in exts
    }

    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in ext_set]

    return sorted(files)


def ensure_dirs(output_dir: str, save_masks: bool, save_crops: bool) -> Dict[str, Path]:
    """Create output directories and return paths."""
    base = Path(output_dir)
    json_dir = base / "json"
    vis_dir = base / "vis"
    json_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    out = {"base": base, "json": json_dir, "vis": vis_dir}
    if save_masks:
        masks_dir = base / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        out["masks"] = masks_dir
    if save_crops:
        crops_dir = base / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        out["crops"] = crops_dir
    return out


def fill_external_contours(mask: np.ndarray) -> np.ndarray:
    """Fill external contours to create solid blobs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def count_holes(component_mask: np.ndarray) -> int:
    """Count holes in a single connected binary component."""
    contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return 0
    hier = hierarchy[0]
    return int(np.sum(hier[:, 3] >= 0))


def split_merged_components(
    obj_mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    min_area: int | None = None,
    erode_max_iter: int = 8,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Split touching blobs conservatively using erosion seeds + nearest-seed assignment.

    This targets merged, hole-free components and avoids splitting ringed tools.
    """
    h, w = image_shape[:2]
    total_area = h * w
    if min_area is None:
        min_area = clamp(int(0.0010 * total_area), 2000, 50000)

    out = obj_mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    split_count = 0

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        ww = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if ww <= 0 or hh <= 0:
            continue

        aspect = max(ww / float(hh), hh / float(ww))
        extent = area / float(ww * hh)
        if area < int(1.5 * min_area):
            continue
        if aspect < 1.5:
            continue
        if extent < 0.08 or extent > 0.42:
            continue

        patch = (labels[y : y + hh, x : x + ww] == label_idx).astype(np.uint8) * 255
        if count_holes(patch) > 0:
            continue

        seed_labels = None
        keep_labels: List[int] = []
        seed_min_area = max(80, int(0.010 * area))
        for it in range(1, max(2, int(erode_max_iter)) + 1):
            eroded = cv2.erode(patch, ellipse_kernel(3), iterations=it)
            n_seed, lbl_seed, st_seed, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            if n_seed <= 1:
                continue
            areas = st_seed[1:, cv2.CC_STAT_AREA]
            keep = [int(i + 1) for i, a in enumerate(areas) if int(a) >= seed_min_area]
            if 2 <= len(keep) <= 3:
                seed_labels = lbl_seed
                keep_labels = keep
                break

        if seed_labels is None:
            continue

        fg_bool = patch > 0
        dmaps: List[np.ndarray] = []
        for seed_id in keep_labels:
            seed = (seed_labels == seed_id).astype(np.uint8)
            dmap = cv2.distanceTransform((1 - seed).astype(np.uint8), cv2.DIST_L2, 3)
            dmaps.append(dmap)
        if not dmaps:
            continue

        assign = (np.argmin(np.stack(dmaps, axis=0), axis=0) + 1).astype(np.int32)
        assign[~fg_bool] = 0

        boundary = np.zeros_like(assign, dtype=bool)
        boundary[:, 1:] |= (assign[:, 1:] != assign[:, :-1]) & (assign[:, 1:] > 0) & (assign[:, :-1] > 0)
        boundary[1:, :] |= (assign[1:, :] != assign[:-1, :]) & (assign[1:, :] > 0) & (assign[:-1, :] > 0)

        split_patch = patch.copy()
        split_patch[boundary] = 0
        split_patch = cv2.morphologyEx(split_patch, cv2.MORPH_OPEN, ellipse_kernel(3), iterations=1)

        n_new, _, st_new, _ = cv2.connectedComponentsWithStats(split_patch, connectivity=8)
        if n_new <= 2:
            continue
        min_part_area = max(300, int(0.35 * min_area))
        kept_parts = []
        for i_part in range(1, n_new):
            pa = int(st_new[i_part, cv2.CC_STAT_AREA])
            if pa < min_part_area:
                continue
            px = int(st_new[i_part, cv2.CC_STAT_LEFT])
            py = int(st_new[i_part, cv2.CC_STAT_TOP])
            pw = int(st_new[i_part, cv2.CC_STAT_WIDTH])
            ph = int(st_new[i_part, cv2.CC_STAT_HEIGHT])
            kept_parts.append((pa, px, py, pw, ph))

        if len(kept_parts) != 2:
            continue
        kept_parts.sort(key=lambda t: t[0], reverse=True)
        (a1, x1, y1, w1, h1), (a2, x2, y2, w2, h2) = kept_parts
        area_ratio = min(a1, a2) / float(max(a1, a2))
        if area_ratio < 0.22:
            continue

        ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        x_overlap = ox / float(max(1, min(w1, w2)))
        y_overlap = oy / float(max(1, min(h1, h2)))
        if max(x_overlap, y_overlap) < 0.45:
            continue

        region = out[y : y + hh, x : x + ww]
        region[patch > 0] = 0
        region[split_patch > 0] = 255
        out[y : y + hh, x : x + ww] = region
        split_count += 1

    return out, {"split_components": int(split_count)}


def extract_roi_mask(
    image_bgr: np.ndarray,
    v_dark: int = 110,
    roi_close_k: int = 31,
    roi_open_k: int = 17,
    auto_thresholds: bool = True,
    manual_v_dark_override: bool = False,
) -> Tuple[np.ndarray, Dict[str, float | bool | int | str]]:
    """
    Extract the black/dark mat ROI as the largest dark region.

    Returns:
        roi_mask: uint8 binary mask (0/255)
        meta: diagnostics and fallback state
    """
    h, w = image_bgr.shape[:2]
    total_pixels = float(h * w)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    used_v_dark = int(v_dark)
    bg_v_ref = float(np.percentile(v, 40))

    if auto_thresholds and not manual_v_dark_override:
        darker_half_cut = np.percentile(v, 75)
        dark_candidates = v[v < darker_half_cut]
        if dark_candidates.size >= 100:
            bg_v_ref = float(np.percentile(dark_candidates, 40))
        used_v_dark = clamp(int(bg_v_ref + 10), 70, 140)

    # Base dark mask plus relaxed low-saturation expansion for dark-gray mat edges.
    dark_base = v < used_v_dark
    dark_sat_relaxed = (v < clamp(used_v_dark + 12, 0, 255)) & (s < 160)
    dark_mask = np.where(dark_base | dark_sat_relaxed, 255, 0).astype(np.uint8)

    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        ellipse_kernel(roi_close_k),
        iterations=2,
    )
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        ellipse_kernel(roi_open_k),
        iterations=1,
    )

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_found = False
    warning = ""

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi_mask, [largest], -1, 255, thickness=cv2.FILLED)
        roi_found = True

    roi_area_frac = float(cv2.countNonZero(roi_mask) / total_pixels)
    if (not roi_found) or roi_area_frac < 0.10:
        roi_mask[:] = 255
        roi_found = False
        warning = "ROI fallback to full image (missing or too small)."
        roi_area_frac = 1.0

    meta: Dict[str, float | bool | int | str] = {
        "roi_area_frac": roi_area_frac,
        "roi_found": roi_found,
        "warning": warning,
        "bg_v_ref": bg_v_ref,
        "used_v_dark": used_v_dark,
    }
    return roi_mask, meta


def edge_based_obj_fallback(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    obj_close_k: int = 23,
) -> np.ndarray:
    """
    Edge-based fallback to recover fragmented/missed instruments.

    Steps: Canny -> dilate -> close -> fill contours.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    roi_bool = roi_mask > 0
    if not np.any(roi_bool):
        return np.zeros_like(roi_mask)

    roi_vals = gray_blur[roi_bool]
    v_med = float(np.median(roi_vals))
    lower = clamp(int(0.66 * v_med), 10, 120)
    upper = clamp(int(1.33 * v_med + 40), 80, 255)
    if upper <= lower:
        upper = clamp(lower + 40, 80, 255)

    edges = cv2.Canny(gray_blur, threshold1=lower, threshold2=upper)
    edges = cv2.bitwise_and(edges, roi_mask)

    edges = cv2.dilate(edges, ellipse_kernel(3), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ellipse_kernel(max(9, obj_close_k // 2)), iterations=2)

    filled = fill_external_contours(edges)
    filled = cv2.bitwise_and(filled, roi_mask)
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, ellipse_kernel(5), iterations=1)
    return filled


def extract_obj_mask_instruments(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    delta_v: int = 35,
    s_min: int = 25,
    obj_close_k: int = 23,
    obj_open_k: int = 7,
    auto_thresholds: bool = True,
) -> Tuple[np.ndarray, Dict[str, float | bool | int]]:
    """
    Extract instrument mask inside ROI using foreground-vs-background thresholding.

    Foreground rule:
        ((V > bg_v + delta_v) OR (S > s_min)) AND ROI

    Adds an edge-based fallback if mask looks empty/fragmented.
    """
    h, w = image_bgr.shape[:2]
    total_pixels = h * w

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    roi_bool = roi_mask > 0
    if not np.any(roi_bool):
        return np.zeros((h, w), dtype=np.uint8), {
            "bg_v": 0.0,
            "v_thr": 0,
            "effective_s_min": int(s_min),
            "local_thr": 0,
            "used_fallback": False,
            "num_components_before": 0,
            "num_components_after": 0,
        }

    v_roi = v[roi_bool]
    s_roi = s[roi_bool]
    bg_v = float(np.percentile(v_roi, 35))

    effective_delta = int(delta_v)
    effective_s_min = int(s_min)
    if auto_thresholds and v_roi.size > 100:
        spread_hint = float(np.percentile(v_roi, 65) - bg_v)
        effective_delta = clamp(int(0.7 * delta_v + 0.3 * spread_hint), 20, 60)
        # Raise saturation threshold when the mat itself has moderate saturation.
        sat_ref = int(np.percentile(s_roi, 80)) if s_roi.size > 100 else int(s_min)
        effective_s_min = clamp(max(int(s_min), sat_ref + 5), 10, 180)

    v_thr = clamp(int(bg_v + effective_delta), 0, 255)
    # Local contrast gate suppresses bright mat gradients while keeping tool edges/highlights.
    sigma = max(3.0, 0.015 * min(h, w))
    v_smooth = cv2.GaussianBlur(v, (0, 0), sigmaX=sigma, sigmaY=sigma)
    v_diff = v.astype(np.int16) - v_smooth.astype(np.int16)
    v_diff_roi = v_diff[roi_bool]
    local_thr = clamp(int(np.percentile(v_diff_roi, 93)) if v_diff_roi.size > 100 else 6, 4, 24)
    very_bright_thr = clamp(int(np.percentile(v_roi, 92)), 140, 255)

    bright_fg = (v > v_thr) & ((v_diff > local_thr) | (v > very_bright_thr))
    sat_fg = (
        (s > effective_s_min)
        & (v_diff > max(4, local_thr // 2))
        & (v > int(bg_v + max(8, effective_delta // 2)))
    )
    fg = np.where((bright_fg | sat_fg) & roi_bool, 255, 0).astype(np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ellipse_kernel(obj_close_k), iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ellipse_kernel(obj_open_k), iterations=1)
    fg = fill_external_contours(fg)
    fg = cv2.bitwise_and(fg, roi_mask)

    num_labels_before, _, stats_before, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas_before = stats_before[1:, cv2.CC_STAT_AREA] if num_labels_before > 1 else np.array([], dtype=np.int32)
    num_components_before = int(areas_before.size)

    min_small = max(64, int(0.00005 * total_pixels))
    nontrivial = areas_before[areas_before >= min_small]
    largest_before = int(nontrivial.max()) if nontrivial.size else 0

    fragmented = (
        num_components_before == 0
        or (num_components_before > 120)
        or (nontrivial.size > 20 and largest_before < int(0.002 * total_pixels))
    )

    used_fallback = False
    if fragmented:
        edge_mask = edge_based_obj_fallback(image_bgr, roi_mask, obj_close_k=obj_close_k)
        fg = cv2.bitwise_or(fg, edge_mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, ellipse_kernel(obj_close_k), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ellipse_kernel(obj_open_k), iterations=1)
        fg = fill_external_contours(fg)
        fg = cv2.bitwise_and(fg, roi_mask)
        used_fallback = True

    num_labels_after, _, stats_after, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas_after = stats_after[1:, cv2.CC_STAT_AREA] if num_labels_after > 1 else np.array([], dtype=np.int32)
    num_components_after = int(areas_after.size)

    meta: Dict[str, float | bool | int] = {
        "bg_v": bg_v,
        "v_thr": v_thr,
        "effective_s_min": int(effective_s_min),
        "local_thr": int(local_thr),
        "used_fallback": used_fallback,
        "num_components_before": num_components_before,
        "num_components_after": num_components_after,
    }
    return fg, meta


def extract_bboxes_from_mask(
    obj_mask: np.ndarray,
    image_shape: Tuple[int, int, int],
    min_area: int | None = None,
    max_area_frac: float = 0.60,
    max_aspect: float = 20.0,
) -> Tuple[Dict[str, List[int]], Dict[str, int | float]]:
    """
    Extract and filter bounding boxes from binary object mask.

    Output bboxes format:
        {"bbox1": [x1, y1, x2, y2], ...}
    sorted by (y1, x1).
    """
    h, w = image_shape[:2]
    total_area = h * w

    if min_area is None:
        min_area = clamp(int(0.0010 * total_area), 2000, 50000)

    max_area = int(max_area_frac * total_area)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)

    boxes: List[Tuple[int, int, int, int]] = []
    total_components = max(0, num_labels - 1)

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        ww = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if ww <= 0 or hh <= 0:
            continue
        if area < min_area or area > max_area:
            continue

        aspect = max(ww / float(hh), hh / float(ww))
        if aspect > max_aspect:
            continue

        extent = area / float(ww * hh)
        touches_border = (x <= 1) or (y <= 1) or ((x + ww) >= (w - 1)) or ((y + hh) >= (h - 1))
        # Border-attached regions are usually ROI spill/noise; keep only unlikely-to-be-artifact ones.
        if touches_border:
            if extent > 0.35:
                continue
            if area < max(int(min_area), int(0.003 * total_area)):
                continue

        x1 = x
        y1 = y
        x2 = x + ww - 1
        y2 = y + hh - 1
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[1], b[0]))

    out: Dict[str, List[int]] = {}
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        out[f"bbox{idx}"] = [int(x1), int(y1), int(x2), int(y2)]

    meta: Dict[str, int | float] = {
        "total_components": total_components,
        "final_bbox_count": len(out),
        "min_area": int(min_area),
        "max_area": int(max_area),
    }
    return out, meta


def draw_debug(
    image_bgr: np.ndarray,
    bboxes: Dict[str, List[int]],
    roi_found: bool,
    used_fallback: bool,
) -> np.ndarray:
    """Draw labeled bounding boxes and summary header."""
    vis = image_bgr.copy()

    for name, box in bboxes.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        tx = x1
        ty = max(16, y1 - 6)
        cv2.putText(vis, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    roi_txt = "largest-dark" if roi_found else "fallback-full"
    fallback_txt = "yes" if used_fallback else "no"
    header = f"boxes={len(bboxes)}  roi={roi_txt}  edge_fallback={fallback_txt}"

    cv2.putText(vis, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def save_bbox_crops(
    image_bgr: np.ndarray,
    bboxes: Dict[str, List[int]],
    crops_root: Path,
    image_stem: str,
    crop_pad: int = 0,
) -> int:
    """Save one crop per bbox under outputs/crops/<image_stem>/bboxN.png."""
    h, w = image_bgr.shape[:2]
    pad = max(0, int(crop_pad))
    image_crop_dir = crops_root / image_stem
    image_crop_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for name, box in bboxes.items():
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = clamp(x1 - pad, 0, w - 1)
        y1 = clamp(y1 - pad, 0, h - 1)
        x2 = clamp(x2 + pad, 0, w - 1)
        y2 = clamp(y2 + pad, 0, h - 1)
        if x2 < x1 or y2 < y1:
            continue

        crop = image_bgr[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            continue

        out_path = image_crop_dir / f"{name}.png"
        if cv2.imwrite(str(out_path), crop):
            saved += 1

    return saved


def flag_was_provided(flag_name: str) -> bool:
    """Check whether a long-form argparse flag was explicitly passed."""
    return any(arg == flag_name or arg.startswith(flag_name + "=") for arg in sys.argv[1:])


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Detect individual surgical instruments on a dark mat and export bounding boxes."
    )

    parser.add_argument("--input_dir", type=str, default=".", help="Input image directory (default: .)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders")
    parser.add_argument("--save_masks", action="store_true", help="Save ROI/object masks")
    parser.add_argument(
        "--save_crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable per-bbox crop export (default: enabled)",
    )
    parser.add_argument(
        "--crop_pad",
        type=int,
        default=0,
        help="Pixels of padding to add around each crop (default: 0)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Optional extension list override, e.g. --extensions .jpg .png .tif",
    )

    # ROI
    parser.add_argument("--v_dark", type=int, default=110, help="Dark threshold on V channel")
    parser.add_argument("--roi_close_k", type=int, default=31, help="ROI close kernel size")
    parser.add_argument("--roi_open_k", type=int, default=17, help="ROI open kernel size")

    # Foreground
    parser.add_argument("--delta_v", type=int, default=35, help="Foreground threshold offset from bg V")
    parser.add_argument("--s_min", type=int, default=25, help="Saturation minimum for colored/plastic parts")
    parser.add_argument("--obj_close_k", type=int, default=23, help="Object close kernel size")
    parser.add_argument("--obj_open_k", type=int, default=7, help="Object open kernel size")

    # Component filtering
    parser.add_argument(
        "--min_area",
        type=int,
        default=None,
        help="Minimum component area in pixels (default: auto)",
    )
    parser.add_argument("--max_area_frac", type=float, default=0.60, help="Maximum component area fraction")
    parser.add_argument("--max_aspect", type=float, default=20.0, help="Maximum aspect ratio")
    parser.add_argument(
        "--split_merged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable conservative split of touching components before bbox extraction",
    )
    parser.add_argument(
        "--split_erode_max_iter",
        type=int,
        default=8,
        help="Max erosion iterations for merged-component split seed search",
    )

    # Auto-thresholding (default ON)
    parser.add_argument(
        "--auto_thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable auto-threshold estimation from image/ROI stats",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manual_v_dark_override = flag_was_provided("--v_dark")

    image_paths = find_images(args.input_dir, args.extensions, args.recursive)
    if not image_paths:
        print(f"[INFO] No images found in '{args.input_dir}'.")
        return

    dirs = ensure_dirs(args.output_dir, args.save_masks, args.save_crops)

    for idx, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] {img_path.name}: failed to read image, skipping.")
            continue

        h, w = image.shape[:2]

        roi_mask, roi_meta = extract_roi_mask(
            image,
            v_dark=args.v_dark,
            roi_close_k=args.roi_close_k,
            roi_open_k=args.roi_open_k,
            auto_thresholds=args.auto_thresholds,
            manual_v_dark_override=manual_v_dark_override,
        )

        obj_mask, obj_meta = extract_obj_mask_instruments(
            image,
            roi_mask,
            delta_v=args.delta_v,
            s_min=args.s_min,
            obj_close_k=args.obj_close_k,
            obj_open_k=args.obj_open_k,
            auto_thresholds=args.auto_thresholds,
        )

        split_meta = {"split_components": 0}
        if args.split_merged:
            obj_mask, split_meta = split_merged_components(
                obj_mask,
                image.shape,
                min_area=args.min_area,
                erode_max_iter=args.split_erode_max_iter,
            )

        bboxes, bbox_meta = extract_bboxes_from_mask(
            obj_mask,
            image.shape,
            min_area=args.min_area,
            max_area_frac=args.max_area_frac,
            max_aspect=args.max_aspect,
        )

        json_obj = {
            "image": img_path.name,
            "width": int(w),
            "height": int(h),
            "roi_area_frac": float(round(float(roi_meta["roi_area_frac"]), 6)),
            "bboxes": bboxes,
        }

        json_path = dirs["json"] / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2)

        vis = draw_debug(
            image,
            bboxes,
            roi_found=bool(roi_meta["roi_found"]),
            used_fallback=bool(obj_meta["used_fallback"]),
        )
        vis_path = dirs["vis"] / f"{img_path.stem}_boxed.png"
        cv2.imwrite(str(vis_path), vis)

        if args.save_masks:
            roi_path = dirs["masks"] / f"{img_path.stem}_roi.png"
            obj_path = dirs["masks"] / f"{img_path.stem}_obj.png"
            cv2.imwrite(str(roi_path), roi_mask)
            cv2.imwrite(str(obj_path), obj_mask)

        crop_count = 0
        if args.save_crops:
            crop_count = save_bbox_crops(
                image,
                bboxes,
                dirs["crops"],
                img_path.stem,
                crop_pad=args.crop_pad,
            )

        warn = str(roi_meta.get("warning", ""))
        warn_txt = f" | {warn}" if warn else ""
        print(
            f"[{idx}/{len(image_paths)}] {img_path.name} | "
            f"ROI={100.0 * float(roi_meta['roi_area_frac']):.2f}% | "
            f"components={int(bbox_meta['total_components'])} | "
            f"final_boxes={int(bbox_meta['final_bbox_count'])} | "
            f"crops={int(crop_count)} | "
            f"edge_fallback={'yes' if bool(obj_meta['used_fallback']) else 'no'} | "
            f"split={int(split_meta['split_components'])}"
            f"{warn_txt}"
        )


if __name__ == "__main__":
    main()
