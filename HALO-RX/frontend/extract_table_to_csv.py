#!/usr/bin/env python3
"""Extract table data from PNG images and save to CSV."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image

_PYTESSERACT_IMPORT_ERROR: Exception | None = None
try:
    import pytesseract  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment-specific import failure
    pytesseract = None  # type: ignore[assignment]
    _PYTESSERACT_IMPORT_ERROR = exc

EXPECTED_COLUMNS = ["Prescription", "Instruction", "QTY", "Datefilled", "Refill"]

_CV2_IMPORT_ERROR: Exception | None = None
try:
    import cv2  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - environment-specific import failure
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc


def _opencv_import_message(error: Exception | None) -> str:
    """Build a helpful architecture-mismatch message for cv2 import failures."""
    base = (
        "OpenCV (cv2) could not be imported. This usually means the Python interpreter and "
        "opencv wheel use different CPU architectures (for example x86_64 vs arm64 on macOS). "
        "Recreate the virtual environment and reinstall dependencies with `uv venv && uv sync`, "
        "then launch with `uv run python -m streamlit run app.py`."
    )
    if error is None:
        return base
    return f"{base} Import error: {error}"


def require_cv2() -> Any:
    """Return imported cv2 module or raise actionable runtime error."""
    if cv2 is None:
        raise RuntimeError(_opencv_import_message(_CV2_IMPORT_ERROR))
    return cv2


def _pytesseract_import_message(error: Exception | None) -> str:
    """Build a consistent troubleshooting message for pytesseract import issues."""
    base = (
        "pytesseract is not available in the current interpreter. Recreate the virtual environment "
        "and reinstall dependencies with `uv venv && uv sync`, then launch with "
        "`uv run python -m streamlit run app.py`."
    )
    if error is None:
        return base
    return f"{base} Import error: {error}"


def require_pytesseract() -> Any:
    """Return imported pytesseract module or raise actionable runtime error."""
    if pytesseract is None:
        raise RuntimeError(_pytesseract_import_message(_PYTESSERACT_IMPORT_ERROR))
    return pytesseract


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments for batch table extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract table values from PNG images and write CSV files named "
            "<image_name>.csv."
        )
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Image paths to process. If omitted, all .png files in current directory are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV output files (default: current directory).",
    )
    parser.add_argument(
        "--lang",
        default="eng",
        help="Tesseract OCR language code (default: eng).",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=50,
        help="Minimum OCR confidence threshold 0-99 (default: 50).",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default="",
        help="Optional full path to tesseract executable.",
    )
    return parser.parse_args(argv)


def resolve_images(image_args: Sequence[str]) -> list[Path]:
    """Resolve explicit image args or default to local ``*.png`` files."""
    if image_args:
        return [Path(p) for p in image_args]
    return sorted(Path(".").glob("*.png"))


def check_tesseract_available(tesseract_cmd: str) -> bool:
    """Verify tesseract executable availability and configure custom path if provided."""
    pytesseract_mod = require_pytesseract()
    if tesseract_cmd:
        pytesseract_mod.pytesseract.tesseract_cmd = tesseract_cmd
        return Path(tesseract_cmd).exists()
    return shutil.which("tesseract") is not None


def normalize_lang(lang: str) -> str:
    """Normalize common language aliases into Tesseract language codes."""
    mapping = {
        "en": "eng",
        "es": "spa",
        "fr": "fra",
        "de": "deu",
        "it": "ita",
        "pt": "por",
        "nl": "nld",
        "ja": "jpn",
        "ko": "kor",
        "zh": "chi_sim",
        "zh-cn": "chi_sim",
        "zh-tw": "chi_tra",
    }
    key = lang.strip().lower()
    return mapping.get(key, key)


def configure_tessdata(lang: str) -> None:
    """Best-effort discovery of tessdata directory containing the requested language."""
    pytesseract_mod = require_pytesseract()
    candidates: list[Path] = []

    env_path = os.environ.get("TESSDATA_PREFIX")
    if env_path:
        candidates.append(Path(env_path))

    for path in (
        "/usr/local/share/tessdata",
        "/opt/homebrew/share/tessdata",
        "/usr/share/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
    ):
        candidates.append(Path(path))

    tesseract_bin = pytesseract_mod.pytesseract.tesseract_cmd
    if not tesseract_bin or tesseract_bin == "tesseract":
        tesseract_bin = shutil.which("tesseract") or ""
    if tesseract_bin:
        tbin = Path(tesseract_bin).resolve()
        prefix = tbin.parent.parent
        candidates.append(prefix / "share" / "tessdata")
        cellar = prefix / "Cellar" / "tesseract"
        if cellar.exists():
            for version_dir in sorted(cellar.glob("*")):
                candidates.append(version_dir / "share" / "tessdata")

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if not candidate.exists():
            continue
        if (candidate / f"{lang}.traineddata").exists():
            os.environ["TESSDATA_PREFIX"] = str(candidate)
            return

    for candidate in unique_candidates:
        if candidate.exists() and any(candidate.glob("*.traineddata")):
            os.environ["TESSDATA_PREFIX"] = str(candidate)
            return


def preprocess_for_lines(image: np.ndarray) -> np.ndarray:
    """Generate binarized image optimized for table-line morphology operations."""
    cv2_mod = require_cv2()
    gray = cv2_mod.cvtColor(image, cv2_mod.COLOR_BGR2GRAY)
    return cv2_mod.adaptiveThreshold(
        ~gray,
        255,
        cv2_mod.ADAPTIVE_THRESH_MEAN_C,
        cv2_mod.THRESH_BINARY,
        15,
        -2,
    )


def detect_cells(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect probable table cells by combining horizontal and vertical line masks."""
    cv2_mod = require_cv2()
    h, w = image.shape[:2]
    bw = preprocess_for_lines(image)

    horizontal_size = max(20, w // 30)
    vertical_size = max(20, h // 30)

    horizontal_kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (horizontal_size, 1))
    vertical_kernel = cv2_mod.getStructuringElement(cv2_mod.MORPH_RECT, (1, vertical_size))

    horizontal = cv2_mod.morphologyEx(bw, cv2_mod.MORPH_OPEN, horizontal_kernel)
    vertical = cv2_mod.morphologyEx(bw, cv2_mod.MORPH_OPEN, vertical_kernel)
    table_mask = cv2_mod.add(horizontal, vertical)

    contours, hierarchy = cv2_mod.findContours(table_mask, cv2_mod.RETR_TREE, cv2_mod.CHAIN_APPROX_SIMPLE)
    cells: list[tuple[int, int, int, int]] = []

    for idx, cnt in enumerate(contours):
        if hierarchy is not None:
            # Prefer leaf contours that are inside a parent table/grid contour.
            child = hierarchy[0][idx][2]
            parent = hierarchy[0][idx][3]
            if child != -1 or parent == -1:
                continue
        x, y, cw, ch = cv2_mod.boundingRect(cnt)
        area = cw * ch
        if area < 300:
            continue
        if cw < 20 or ch < 12:
            continue
        if cw > w * 0.98 or ch > h * 0.98:
            continue
        cells.append((x, y, cw, ch))

    if not cells:
        return []

    deduped: list[tuple[int, int, int, int]] = []
    for cell in sorted(cells, key=lambda b: (b[1], b[0], b[2], b[3])):
        x, y, cw, ch = cell
        replaced = False
        for idx, (dx, dy, dw, dh) in enumerate(deduped):
            if abs(x - dx) <= 4 and abs(y - dy) <= 4 and abs(cw - dw) <= 6 and abs(ch - dh) <= 6:
                if cw * ch > dw * dh:
                    deduped[idx] = cell
                replaced = True
                break
        if not replaced:
            deduped.append(cell)

    # Drop container boxes that still surround multiple smaller cells.
    filtered: list[tuple[int, int, int, int]] = []
    for cell in deduped:
        x, y, w, h = cell
        contained = 0
        for other in deduped:
            if other == cell:
                continue
            ox, oy, ow, oh = other
            if ox >= x - 2 and oy >= y - 2 and (ox + ow) <= (x + w + 2) and (oy + oh) <= (y + h + 2):
                contained += 1
        if contained >= 2:
            continue
        filtered.append(cell)

    return filtered


def group_rows(cells: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Cluster detected cells into row groups based on vertical proximity."""
    if not cells:
        return []
    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    median_h = int(np.median([h for _, _, _, h in cells]))
    row_tol = max(8, median_h // 2)

    rows: list[list[tuple[int, int, int, int]]] = []
    for cell in cells:
        x, y, cw, ch = cell
        placed = False
        for row in rows:
            ref_y = int(np.median([r[1] for r in row]))
            if abs(y - ref_y) <= row_tol:
                row.append(cell)
                placed = True
                break
        if not placed:
            rows.append([cell])

    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda r: min(c[1] for c in r))
    return rows


def ocr_cell(
    image: np.ndarray,
    cell: tuple[int, int, int, int],
    lang: str,
    min_confidence: int,
) -> str:
    """OCR one detected cell region and return confidence-filtered text."""
    cv2_mod = require_cv2()
    pytesseract_mod = require_pytesseract()
    x, y, w, h = cell
    pad = 2
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(image.shape[1], x + w - pad)
    y1 = min(image.shape[0], y + h - pad)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return ""

    gray = cv2_mod.cvtColor(crop, cv2_mod.COLOR_BGR2GRAY)
    proc = cv2_mod.threshold(gray, 0, 255, cv2_mod.THRESH_BINARY + cv2_mod.THRESH_OTSU)[1]

    try:
        data = pytesseract_mod.image_to_data(
            proc,
            lang=lang,
            output_type=pytesseract_mod.Output.DICT,
            config="--psm 6",
        )
    except pytesseract_mod.TesseractError as exc:
        raise RuntimeError(str(exc)) from exc
    words: list[str] = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        token = str(txt).strip()
        if not token:
            continue
        try:
            score = float(conf)
        except (TypeError, ValueError):
            score = -1.0
        if score >= min_confidence:
            words.append(token)
    return " ".join(words).strip()


def _kmeans_1d(values: list[float], k: int, iterations: int = 20) -> list[float]:
    """Lightweight 1D k-means for estimating column centers from token x-positions."""
    if not values:
        return []
    arr = np.array(values, dtype=float)
    if arr.size <= k:
        return sorted(float(v) for v in arr)
    percentiles = np.linspace(0, 100, k + 2)[1:-1]
    centers = np.percentile(arr, percentiles)
    for _ in range(iterations):
        distances = np.abs(arr[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(k):
            group = arr[labels == idx]
            if group.size:
                new_centers[idx] = float(group.mean())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return sorted(float(v) for v in centers)


def _collect_tokens_without_cv2(
    image_path: Path,
    lang: str,
    min_confidence: int,
) -> list[dict[str, float | str]]:
    """Collect OCR tokens directly from PIL image for cv2-less fallback flow."""
    pytesseract_mod = require_pytesseract()
    try:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
            data = pytesseract_mod.image_to_data(
                image_rgb,
                lang=lang,
                output_type=pytesseract_mod.Output.DICT,
                config="--psm 6",
            )
    except pytesseract_mod.TesseractError as exc:
        raise RuntimeError(f"OCR failed for {image_path}: {exc}") from exc

    tokens: list[dict[str, float | str]] = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])
    count = min(len(texts), len(confs), len(lefts), len(tops), len(widths), len(heights))
    for idx in range(count):
        text = str(texts[idx]).strip()
        if not text:
            continue
        try:
            conf = float(confs[idx])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_confidence:
            continue
        x = int(lefts[idx])
        y = int(tops[idx])
        w = int(widths[idx])
        h = int(heights[idx])
        if w <= 0 or h <= 0:
            continue
        tokens.append(
            {
                "text": text,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "cx": float(x + (w / 2.0)),
                "cy": float(y + (h / 2.0)),
            }
        )
    return tokens


def _group_tokens_into_rows(tokens: list[dict[str, float | str]]) -> list[list[dict[str, float | str]]]:
    """Group OCR tokens into row-like lines by y-center proximity."""
    if not tokens:
        return []
    heights = [float(tok["h"]) for tok in tokens if float(tok["h"]) > 0]
    median_h = float(np.median(heights)) if heights else 14.0
    row_tol = max(8.0, median_h * 0.7)

    rows: list[list[dict[str, float | str]]] = []
    for token in sorted(tokens, key=lambda t: (float(t["cy"]), float(t["x"]))):
        cy = float(token["cy"])
        placed = False
        for row in rows:
            ref = float(np.median([float(r["cy"]) for r in row]))
            if abs(cy - ref) <= row_tol:
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])

    cleaned_rows: list[list[dict[str, float | str]]] = []
    for row in rows:
        sorted_row = sorted(row, key=lambda t: float(t["x"]))
        if len(sorted_row) >= 2:
            cleaned_rows.append(sorted_row)
    cleaned_rows.sort(key=lambda r: float(np.median([float(t["y"]) for t in r])))
    return cleaned_rows


def _build_table_without_cv2(tokens: list[dict[str, float | str]]) -> list[list[str]]:
    """Approximate table cells from token clusters when geometric cell detection is unavailable."""
    rows = _group_tokens_into_rows(tokens)
    if not rows:
        return []

    cx_values = [float(tok["cx"]) for tok in tokens]
    if len(cx_values) < 2:
        return []
    centers = _kmeans_1d(cx_values, k=5)
    if not centers:
        return []

    table_data: list[list[str]] = []
    for row in rows:
        cell_text: list[list[str]] = [[] for _ in centers]
        for token in row:
            cx = float(token["cx"])
            idx = int(np.argmin([abs(cx - c) for c in centers]))
            cell_text[idx].append(str(token["text"]))
        row_values = [" ".join(parts).strip() for parts in cell_text]
        if sum(bool(value) for value in row_values) >= 2:
            table_data.append(row_values)
    return table_data


def _parse_image_without_cv2(
    image_path: Path,
    lang: str,
    min_confidence: int,
) -> tuple[pd.DataFrame | None, bool]:
    """Fallback parser used when cv2 is unavailable or image decoding fails."""
    tokens = _collect_tokens_without_cv2(
        image_path=image_path,
        lang=lang,
        min_confidence=min_confidence,
    )
    table_data = _build_table_without_cv2(tokens)
    if not table_data:
        return None, False
    df, has_headers = normalize_table_data(table_data)
    return standardize_prescription_dataframe(df), has_headers


def normalize_table_data(table_data: list[list[str]]) -> tuple[pd.DataFrame, bool]:
    """Normalize raw extracted rows and infer header-style first-row layouts."""
    if not table_data:
        return pd.DataFrame(), False

    clean = [[(v or "").strip() for v in row] for row in table_data]
    first = clean[0]
    if len(first) >= 6 and len(first) % 2 == 0:
        headers = [h.strip() for h in first[::2]]
        first_values = [v.strip() for v in first[1::2]]
        header_like = sum(bool(h) and len(h.split()) <= 3 for h in headers)
        values_present = sum(bool(v) for v in first_values)
        if header_like >= max(3, len(headers) - 1) and values_present >= max(3, len(first_values) - 1):
            normalized: list[list[str]] = [first_values]
            col_count = len(headers)
            for row in clean[1:]:
                normalized.append((row + [""] * col_count)[:col_count])
            return pd.DataFrame(normalized, columns=headers), True

    return pd.DataFrame(clean), False


def standardize_prescription_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map extracted columns into canonical prescription schema used by the UI."""
    clean = df.copy()
    clean = clean.replace({np.nan: ""})
    clean.columns = [str(c).strip() for c in clean.columns]

    name_map = {
        "prescription": "Prescription",
        "medicine": "Prescription",
        "drug": "Prescription",
        "instruction": "Instruction",
        "instructions": "Instruction",
        "sig": "Instruction",
        "qty": "QTY",
        "quantity": "QTY",
        "datefilled": "Datefilled",
        "date filled": "Datefilled",
        "refill": "Refill",
        "refills": "Refill",
    }

    renamed = {}
    for col in clean.columns:
        key = col.strip().lower()
        if key in name_map:
            renamed[col] = name_map[key]
    clean = clean.rename(columns=renamed)

    if set(EXPECTED_COLUMNS).issubset(set(clean.columns)):
        clean = clean[EXPECTED_COLUMNS]
    elif clean.shape[1] >= 5:
        clean = clean.iloc[:, :5]
        clean.columns = EXPECTED_COLUMNS
    else:
        for col in EXPECTED_COLUMNS:
            if col not in clean.columns:
                clean[col] = ""
        clean = clean[EXPECTED_COLUMNS]

    clean = clean.map(lambda v: str(v).replace("\n", " ").strip() if v is not None else "")
    clean = clean[~(clean == "").all(axis=1)].reset_index(drop=True)
    return clean


def parse_image_to_dataframe(
    image_path: Path,
    lang: str,
    min_confidence: int,
) -> tuple[pd.DataFrame | None, bool]:
    """Parse one prescription image into normalized DataFrame plus header-detected flag."""
    if cv2 is None:
        return _parse_image_without_cv2(
            image_path=image_path,
            lang=lang,
            min_confidence=min_confidence,
        )

    cv2_mod = require_cv2()
    image = cv2_mod.imread(str(image_path))
    if image is None:
        return _parse_image_without_cv2(
            image_path=image_path,
            lang=lang,
            min_confidence=min_confidence,
        )

    cells = detect_cells(image)
    if not cells:
        return None, False

    rows = group_rows(cells)
    if not rows:
        return None, False

    max_cols = max(len(r) for r in rows)
    table_data: list[list[str]] = []
    for row in rows:
        try:
            row_values = [
                ocr_cell(image, cell, lang=lang, min_confidence=min_confidence)
                for cell in row
            ]
        except RuntimeError as exc:
            raise RuntimeError(f"OCR failed for {image_path}: {exc}") from exc
        if len(row_values) < max_cols:
            row_values.extend([""] * (max_cols - len(row_values)))
        table_data.append(row_values)

    df, has_headers = normalize_table_data(table_data)
    return standardize_prescription_dataframe(df), has_headers


def ensure_ocr_ready(lang: str, tesseract_cmd: str = "") -> tuple[bool, str]:
    """Validate OCR runtime dependencies before attempting image extraction."""
    try:
        pytesseract_mod = require_pytesseract()
    except RuntimeError as exc:
        return False, str(exc)

    if not check_tesseract_available(tesseract_cmd):
        return (
            False,
            "Tesseract executable not found. Install it first (macOS: `brew install tesseract`) "
            "or pass --tesseract-cmd /full/path/to/tesseract.",
        )

    configure_tessdata(lang)
    try:
        languages = pytesseract_mod.get_languages(config="")
    except pytesseract_mod.TesseractError as exc:
        return (
            False,
            "Tesseract is installed but language data failed to load: "
            f"{exc}. Try setting TESSDATA_PREFIX to your tessdata directory.",
        )

    if lang not in languages:
        return False, f"Language '{lang}' not available in Tesseract. Installed: {', '.join(languages)}"
    return True, ""


def extract_one_image(image_path: Path, output_dir: Path, lang: str, min_confidence: int) -> bool:
    """Extract one image and write CSV output; return success status."""
    try:
        parsed, has_headers = parse_image_to_dataframe(
            image_path=image_path,
            lang=lang,
            min_confidence=min_confidence,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return False

    if parsed is None:
        print(f"[WARN] Could not extract a table from {image_path}")
        return False

    df = parsed
    output_path = output_dir / f"{image_path.stem}.csv"
    # Always write headers for consistency in downstream tooling.
    df.to_csv(output_path, index=False, header=True)
    print(f"[OK] {image_path} -> {output_path} ({df.shape[0]} rows x {df.shape[1]} cols)")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for batch extraction utility."""
    args = parse_args(argv or sys.argv[1:])

    lang = normalize_lang(args.lang)
    ready, message = ensure_ocr_ready(lang=lang, tesseract_cmd=args.tesseract_cmd)
    if not ready:
        print(message, file=sys.stderr)
        return 2

    images = resolve_images(args.images)
    if not images:
        print("No PNG files found. Provide image paths or place PNG files in this folder.")
        return 1

    missing = [img for img in images if not img.exists()]
    if missing:
        for img in missing:
            print(f"[ERROR] File not found: {img}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for image_path in images:
        if extract_one_image(
            image_path=image_path,
            output_dir=args.output_dir,
            lang=lang,
            min_confidence=args.min_confidence,
        ):
            success_count += 1

    print(f"Completed: {success_count}/{len(images)} images extracted.")
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
