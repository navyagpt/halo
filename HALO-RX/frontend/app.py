#!/usr/bin/env python3
"""HALO RX web UI for prescription parsing and slash-command workflows."""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import streamlit.components.v1 as st_components

try:
    from extract_table_to_csv import (
        ensure_ocr_ready,
        normalize_lang,
        parse_image_to_dataframe,
        standardize_prescription_dataframe,
    )
except ModuleNotFoundError:
    from frontend.extract_table_to_csv import (
        ensure_ocr_ready,
        normalize_lang,
        parse_image_to_dataframe,
        standardize_prescription_dataframe,
    )

WORKDAY_START = 7 * 60
WORKDAY_END = 19 * 60
ADMIN_WINDOW_MINUTES = 45
COMMAND_CATALOG = [
    {
        "usage": "/medicine-timetable",
        "example": "/medicine-timetable",
        "description": "Generate medicine timetable from the parsed prescription (07:00-19:00 only).",
        "needs": "Parsed prescription",
    },
    {
        "usage": "/override-medicine-timetable",
        "example": "/override-medicine-timetable",
        "description": "Enable editable timetable mode so staff can update timings manually.",
        "needs": "Existing timetable",
    },
    {
        "usage": "/medicine-administer <medicine-name>",
        "example": "/medicine-administer Ibuprofen 800mg",
        "description": "Start administer flow for one medicine with safety checks and warning prompts.",
        "needs": "Existing timetable",
    },
    {
        "usage": "/medicine-info <medicine-name>",
        "example": "/medicine-info Ibuprofen",
        "description": "Show parsed prescription details for a medicine.",
        "needs": "Parsed prescription",
    },
    {
        "usage": "/auto-administer",
        "example": "/auto-administer",
        "description": "Pick the nearest scheduled medicine and run administer flow with warnings if needed.",
        "needs": "Existing timetable",
    },
    {
        "usage": "/wrong-medicine",
        "example": "/wrong-medicine",
        "description": "Trigger wrong-medicine flow and show med.png.",
        "needs": "After /medicine-administer or /auto-administer",
    },
    {
        "usage": "/stop",
        "example": "/stop",
        "description": "Stop the autonomous pipeline and cancel any running administration flow.",
        "needs": "Active auto-administer session",
    },
]
POSTOP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = POSTOP_ROOT.parent
BACKEND_ROOT = REPO_ROOT / "backend"
PIPELINE_BACKEND_ROOT = BACKEND_ROOT
FRONTEND_ASSETS_DIR = POSTOP_ROOT / "assets"
FRONTEND_RUNTIME_DIR = POSTOP_ROOT / "runtime"
UPLOADS_DIR = FRONTEND_RUNTIME_DIR / "uploads"
MED_IMAGE_PATH = FRONTEND_RUNTIME_DIR / "med.png"
PIPELINE_OUTPUT_ROOT = FRONTEND_RUNTIME_DIR / "pipeline_runs"
PIPELINE_IMAGE_INPUT_DIR = REPO_ROOT / "image-input"
IMAGE_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
# Modules that must exist for model-backed crop generation; otherwise UI falls back to synthetic med.png.
PIPELINE_REQUIRED_MODULES = ("torch", "transformers", "accelerate", "peft", "sentencepiece", "safetensors")


def init_state() -> None:
    """Initialize all Streamlit session keys used by the nurse workflow."""
    defaults: dict[str, Any] = {
        "prescription_df": None,
        "prescription_image_path": "",
        "last_csv_path": "",
        "timetable_df": None,
        "override_mode": False,
        "pending_action": None,
        "last_feedback": None,
        "command_log": [],
        "show_med_image": False,
        "last_administered": None,
        "command_input": "",
        "_pending_execute_cmd": None,
        "administer_confirm": None,
        "_override_version": 0,
        "auto_administer_warn": False,
        "_auto_administer_medicine": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_styles() -> None:
    """Inject custom CSS theme for the HALO RX console."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --ink: #111111;
  --paper: #fff9f0;
  --lemon: #fff36a;
  --mint: #7cffb7;
  --sky: #78d6ff;
  --rose: #ff8fab;
}

html, body, [class*="st-"] {
  font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    linear-gradient(145deg, #fff9f0 0%, #fff4d9 45%, #ffe9c4 100%);
}

h1, h2, h3 {
  font-family: "Archivo Black", "Impact", sans-serif;
  letter-spacing: 0.02em;
}

[data-testid="stMetricValue"] {
  font-family: "Archivo Black", sans-serif;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--sky) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 6px 6px 0 var(--ink);
}

[data-testid="stFileUploaderDropzone"] button {
  background: var(--lemon) !important;
  color: var(--ink) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  font-weight: 700 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
  background: var(--mint) !important;
}

[data-testid="stBaseButton-primary"] button,
button[data-testid="stBaseButton-primary"],
button[kind="primary"],
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] > button {
  background: var(--lemon) !important;
  color: var(--ink) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  font-weight: 700 !important;
  box-shadow: 4px 4px 0 var(--ink);
}
[data-testid="stBaseButton-primary"] button:hover,
button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
  background: var(--mint) !important;
}

[data-testid="stBaseButton-secondary"] button,
button[data-testid="stBaseButton-secondary"],
[data-testid="baseButton-secondary"] {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  font-weight: 700 !important;
  box-shadow: 4px 4px 0 var(--ink);
}
[data-testid="stBaseButton-secondary"] button:hover,
button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="baseButton-secondary"]:hover {
  background: var(--sky) !important;
}

[data-testid="stForm"] {
  background: #ffffff;
  border: 3px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);
  padding: 12px;
}

[data-testid="stTextInputRootElement"] input {
  background: #ffffff !important;
  color: var(--ink) !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
}

code {
  background: var(--mint) !important;
  color: var(--ink) !important;
  border: 2px solid var(--ink) !important;
  border-radius: 0 !important;
  padding: 2px 6px !important;
  font-weight: 700 !important;
}
pre code {
  background: #ffffff !important;
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
}

[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 3px solid var(--ink) !important;
  box-shadow: 6px 6px 0 var(--ink);
}

[data-testid="stVerticalBlockBorderWrapper"] > div[style*="overflow"] {
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  background: var(--paper) !important;
  box-shadow: 6px 6px 0 var(--ink);
  scroll-behavior: smooth;
}

[data-testid="stAlert"] {
  border: 3px solid var(--ink) !important;
  border-radius: 0 !important;
  box-shadow: 4px 4px 0 var(--ink);
}

.cmd-log-entry {
  background: #ffffff;
  border: 3px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);
  padding: 12px 14px;
  margin-bottom: 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.cmd-log-cmd {
  font-family: "Archivo Black", sans-serif;
  font-size: 0.95rem;
  color: var(--ink);
}
.cmd-log-outcome {
  font-size: 0.88rem;
  color: var(--ink);
  line-height: 1.35;
}
.cmd-log-time {
  font-size: 0.74rem;
  color: #333333;
  margin-top: 2px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def append_log(command: str, outcome: str) -> None:
    """Append one command event to the in-session command feed."""
    st.session_state.command_log.append(
        {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Command": command,
            "Outcome": outcome,
        }
    )


def set_feedback(level: str, text: str) -> None:
    """Store a user-facing status message to render in the right panel."""
    st.session_state.last_feedback = {"level": level, "text": text}


def show_feedback() -> None:
    """Render the latest status message using Streamlit alert components."""
    feedback = st.session_state.last_feedback
    if not feedback:
        return
    level = feedback.get("level", "info")
    text = feedback.get("text", "")
    if level == "success":
        st.success(text)
    elif level == "warning":
        st.warning(text)
    elif level == "error":
        st.error(text)
    else:
        st.info(text)


def save_uploaded_file(uploaded_file) -> Path:
    """Persist uploaded image with a sanitized, timestamped filename."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_file.name or "upload.png")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = UPLOADS_DIR / f"{stamp}_{safe_name}"
    image_path.write_bytes(uploaded_file.getvalue())
    return image_path


def minutes_to_hhmm(minutes: int) -> str:
    """Convert minute-of-day integer to HH:MM 24-hour string."""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def hhmm_to_minutes(value: str) -> int | None:
    """Parse HH:MM string into minute-of-day; return ``None`` on invalid input."""
    text = value.strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour * 60 + minute


def human_time(hhmm: str) -> str:
    """Render HH:MM as user-friendly 12-hour time for warning messages."""
    minutes = hhmm_to_minutes(hhmm)
    if minutes is None:
        return hhmm
    hour24 = minutes // 60
    minute = minutes % 60
    suffix = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def parse_interval_hours(instruction: str) -> int | None:
    """Infer a dosing interval in hours from free-text instructions."""
    lower = instruction.lower()

    range_match = re.search(r"every\s+(\d+)\s*(?:-|to)\s*(\d+)\s*hours?", lower)
    if range_match:
        return int(range_match.group(2))

    single_match = re.search(r"every\s+(\d+)\s*hours?", lower)
    if single_match:
        return int(single_match.group(1))

    word_to_num = {
        "once": 1,
        "one": 1,
        "twice": 2,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }

    times_match = re.search(r"(once|twice|one|two|three|four|five|\d+)\s+times?\s+(?:a|per)?\s*day", lower)
    if times_match:
        token = times_match.group(1)
        count = word_to_num.get(token, None)
        if count is None and token.isdigit():
            count = int(token)
        if count is not None and count > 0:
            if count == 1:
                return 24
            span = WORKDAY_END - WORKDAY_START
            interval = max(1, round((span / (count - 1)) / 60))
            return interval

    if "daily" in lower or "once a day" in lower:
        return 24
    return None


def build_times_for_instruction(instruction: str) -> tuple[list[str], str]:
    """Generate schedule times constrained to workday hours plus explanatory notes."""
    lower = instruction.lower()
    interval_hours = parse_interval_hours(instruction)
    notes: list[str] = []

    if interval_hours is None:
        times = [540]  # 09:00 default
        notes.append("Default daytime schedule")
    else:
        step = max(1, interval_hours) * 60
        times = [WORKDAY_START]
        while times[-1] + step <= WORKDAY_END:
            times.append(times[-1] + step)
        notes.append(f"Derived from every {interval_hours} hour(s)")

    if "as needed" in lower:
        notes.append("PRN (as needed)")

    unique = sorted(set(times))
    return [minutes_to_hhmm(t) for t in unique], "; ".join(notes)


def build_timetable_from_prescription(prescription_df: pd.DataFrame) -> pd.DataFrame:
    """Convert parsed prescription rows into timetable rows for command workflows."""
    rows: list[dict[str, str]] = []
    for _, row in prescription_df.iterrows():
        med = str(row.get("Prescription", "")).strip()
        instruction = str(row.get("Instruction", "")).strip()
        if not med:
            continue
        times, notes = build_times_for_instruction(instruction)
        rows.append(
            {
                "Medicine": med,
                "Times": ", ".join(times),
                "Instruction": instruction,
                "Notes": notes,
            }
        )
    return pd.DataFrame(rows)


def parse_times_text(times_text: str) -> tuple[list[str], list[str]]:
    """Parse comma-separated times into normalized HH:MM list and invalid tokens."""
    items = [token.strip() for token in times_text.split(",") if token.strip()]
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for token in items:
        minute_value = hhmm_to_minutes(token)
        if minute_value is None:
            invalid.append(token)
            continue
        hhmm = minutes_to_hhmm(minute_value)
        if hhmm not in seen:
            seen.add(hhmm)
            valid.append(hhmm)

    valid.sort(key=lambda text: hhmm_to_minutes(text) or 0)
    return valid, invalid


def resolve_medicine_name(query: str, timetable_df: pd.DataFrame) -> str | None:
    """Resolve user medicine query by exact then substring matching."""
    if timetable_df is None or timetable_df.empty:
        return None
    q = query.strip().lower()
    if not q:
        return None

    meds = [str(m).strip() for m in timetable_df["Medicine"].tolist()]
    exact = [m for m in meds if m.lower() == q]
    if exact:
        return exact[0]

    contains = [m for m in meds if q in m.lower()]
    if contains:
        return contains[0]
    return None


def nearest_schedule_delta(times_text: str, now_minutes: int) -> tuple[bool, str, int]:
    """Return whether now is within admin window and the nearest scheduled slot."""
    valid, _ = parse_times_text(times_text)
    if not valid:
        return False, "", 10_000
    minute_values = [hhmm_to_minutes(v) for v in valid]
    minute_values = [v for v in minute_values if v is not None]
    deltas = [abs(now_minutes - v) for v in minute_values]
    idx = int(np.argmin(deltas))
    nearest = minutes_to_hhmm(minute_values[idx])
    delta = deltas[idx]
    return delta <= ADMIN_WINDOW_MINUTES, nearest, delta


def _short_label(text: str, max_chars: int = 36) -> str:
    """Compact long medicine labels to fit fallback image canvas."""
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def ensure_med_image(medicine: str | None = None, overwrite: bool = False) -> Path:
    """Create or reuse fallback ``med.png`` when model crop is unavailable."""
    out = MED_IMAGE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        return out

    canvas = Image.new("RGB", (620, 360), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((20, 20, 600, 340), outline=(40, 130, 40), width=3)
    draw.text((160, 95), "Medication Event", fill=(40, 60, 40))
    if medicine:
        draw.text((85, 160), f"Medicine: {_short_label(medicine)}", fill=(30, 30, 30))
    else:
        draw.text((120, 170), "Check patient ID and dosage", fill=(30, 30, 30))
    draw.text((270, 245), "med.png", fill=(30, 30, 30))
    canvas.save(out)
    return out


def _medicine_tokens(text: str) -> set[str]:
    """Tokenize medicine text while dropping dosage-only and generic tokens."""
    stopwords = {"and", "tablet", "tablets", "capsule", "capsules", "tab", "cap"}
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    cleaned: set[str] = set()
    for token in tokens:
        if token in stopwords:
            continue
        if token.isdigit():
            continue
        if re.fullmatch(r"\d+(mg|mcg|g|ml)", token):
            continue
        cleaned.add(token)
    return cleaned


def _medicine_label_matches(label: str, target: str) -> bool:
    """Match medicine labels using normalized strings plus subset token matching."""
    label_norm = " ".join(re.findall(r"[a-z0-9]+", (label or "").lower()))
    target_norm = " ".join(re.findall(r"[a-z0-9]+", (target or "").lower()))
    if not label_norm or not target_norm:
        return False
    if label_norm == target_norm:
        return True
    if target_norm in label_norm or label_norm in target_norm:
        return True
    target_tokens = _medicine_tokens(target)
    label_tokens = _medicine_tokens(label)
    return bool(target_tokens) and target_tokens.issubset(label_tokens)


def _first_matching_crop_from_predictions(predictions_jsonl: Path, target_medicine: str) -> Path | None:
    """Return first crop path whose predicted labels match target medicine."""
    with predictions_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels = [str(row.get("predicted_label", ""))]
            top_k = row.get("top_k_labels", [])
            if isinstance(top_k, list):
                labels.extend(str(v) for v in top_k)
            if any(_medicine_label_matches(label, target_medicine) for label in labels):
                image_path = Path(str(row.get("image_path", "")))
                if image_path.exists():
                    return image_path
    return None


def _candidate_pipeline_pythons() -> list[Path]:
    """List viable Python executables in priority order for backend pipeline runs."""
    candidates = [
        REPO_ROOT / ".venv" / "bin" / "python",
        POSTOP_ROOT / ".venv" / "bin" / "python",
        BACKEND_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _missing_modules_for_python(python_exec: Path, modules: tuple[str, ...]) -> list[str]:
    """Return missing module names for a given interpreter via lightweight probe."""
    check_code = (
        "import importlib.util\n"
        f"mods = {modules!r}\n"
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
        "print('\\n'.join(missing))\n"
    )
    result = subprocess.run(
        [str(python_exec), "-c", check_code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0:
        return list(modules)
    return missing


def resolve_pipeline_python() -> tuple[str, list[str]]:
    """Choose interpreter with the fewest missing pipeline dependencies."""
    candidates = _candidate_pipeline_pythons()
    if not candidates:
        return str(Path(sys.executable)), list(PIPELINE_REQUIRED_MODULES)

    best = candidates[0]
    best_missing = _missing_modules_for_python(best, PIPELINE_REQUIRED_MODULES)
    for candidate in candidates:
        missing = _missing_modules_for_python(candidate, PIPELINE_REQUIRED_MODULES)
        if not missing:
            return str(candidate), []
        if len(missing) < len(best_missing):
            best = candidate
            best_missing = missing
    return str(best), best_missing


def resolve_pipeline_image_path() -> Path:
    """Resolve source image for pipeline, preferring newest file in root ``image-input``."""
    image_input_candidates: list[Path] = []
    if PIPELINE_IMAGE_INPUT_DIR.exists():
        image_input_candidates = sorted(
            [
                p
                for p in PIPELINE_IMAGE_INPUT_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if image_input_candidates:
        return image_input_candidates[0]

    raw_uploaded = str(st.session_state.get("prescription_image_path", "")).strip()
    candidates: list[Path] = []
    if raw_uploaded:
        candidates.append(Path(raw_uploaded))

    uploads_dir = UPLOADS_DIR
    if uploads_dir.exists():
        uploaded_images = sorted(
            [
                p
                for p in uploads_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(uploaded_images)
    candidates.extend(
        [
            POSTOP_ROOT / "presImg.png",
            POSTOP_ROOT / "presImg.jpg",
            POSTOP_ROOT / "presImg.jpeg",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise RuntimeError("No source prescription image found for pipeline execution.")


def build_pipeline_prescription_file(target_medicine: str) -> Path:
    """Build prescription candidate file consumed by modular pipeline CLI."""
    PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = PIPELINE_OUTPUT_ROOT / "prescription_candidates.txt"

    candidates: list[str] = []
    pres_df = st.session_state.get("prescription_df")
    if isinstance(pres_df, pd.DataFrame) and not pres_df.empty and "Prescription" in pres_df.columns:
        seen: set[str] = set()
        for raw in pres_df["Prescription"].tolist():
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(text)

    if not candidates:
        default_file = PIPELINE_BACKEND_ROOT / "meds.txt"
        if default_file.exists():
            return default_file
        candidates = [target_medicine]

    out.write_text("\n".join(candidates) + "\n", encoding="utf-8")
    return out


def generate_med_image_for_target(target_medicine: str) -> Path:
    """Run backend pipeline for target medicine and copy matched crop to ``med.png``."""
    pipeline_pkg = PIPELINE_BACKEND_ROOT
    if not pipeline_pkg.exists():
        raise RuntimeError(f"Pipeline package not found at {pipeline_pkg}")

    PIPELINE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    source_image = resolve_pipeline_image_path()
    prescription_file = build_pipeline_prescription_file(target_medicine)

    uv_exec = shutil.which("uv")
    cmd: list[str]
    if uv_exec:
        cmd = [
            uv_exec,
            "run",
            "--project",
            str(REPO_ROOT),
            "--no-sync",
            "python",
            "-m",
            "backend.pipeline",
        ]
    else:
        python_exec, missing_modules = resolve_pipeline_python()
        if missing_modules:
            joined = ", ".join(missing_modules)
            raise RuntimeError(
                "Pipeline dependencies are missing "
                f"({joined}) in interpreter `{python_exec}`. "
                "Install with: `uv sync` from repository root."
            )
        cmd = [python_exec, "-m", "backend.pipeline"]

    started_at = datetime.now().timestamp()
    cmd.extend(
        [
            "--image_path",
            str(source_image),
            "--prescription_file",
            str(prescription_file),
            "--target_medicine",
            target_medicine,
            "--top_k",
            "4",
            "--threshold",
            "0.0",
            "--output_root",
            str(PIPELINE_OUTPUT_ROOT),
        ]
    )
    env = os.environ.copy()
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)

    # Force backend package visibility and offline model resolution for reproducible local runs.
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_lines = [line.strip() for line in (result.stderr or result.stdout).splitlines() if line.strip()]
        message = error_lines[-1] if error_lines else f"Pipeline exited with code {result.returncode}"
        raise RuntimeError(message)

    run_dirs = [
        path
        for path in PIPELINE_OUTPUT_ROOT.iterdir()
        if path.is_dir() and path.stat().st_mtime >= started_at - 1.0
    ]
    if not run_dirs:
        run_dirs = [path for path in PIPELINE_OUTPUT_ROOT.iterdir() if path.is_dir()]
    if not run_dirs:
        raise RuntimeError("Pipeline did not produce any run directory.")
    latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
    predictions_jsonl = latest_run / "predictions.jsonl"
    if not predictions_jsonl.exists():
        raise RuntimeError(f"Missing predictions file: {predictions_jsonl}")

    matched_crop = _first_matching_crop_from_predictions(predictions_jsonl, target_medicine)
    if matched_crop is None:
        raise RuntimeError(f"No matching crop found for '{target_medicine}'.")

    shutil.copy2(matched_crop, MED_IMAGE_PATH)
    return MED_IMAGE_PATH


def execute_administer(medicine: str, command_name: str) -> None:
    """Execute medication flow, falling back to synthetic image if pipeline fails."""
    pipeline_error = ""
    try:
        generate_med_image_for_target(medicine)
    except Exception as exc:
        pipeline_error = str(exc).strip() or exc.__class__.__name__
        try:
            ensure_med_image(medicine=medicine, overwrite=True)
        except Exception as fallback_exc:
            append_log(command_name, f"Failed: could not generate fallback med.png for '{medicine}'.")
            set_feedback("error", f"Could not generate med.png for '{medicine}': {fallback_exc}")
            return
        append_log(command_name, f"Pipeline image unavailable for '{medicine}'. Using fallback med.png.")

    st.session_state.last_administered = {
        "medicine": medicine,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": command_name,
    }
    st.session_state.show_med_image = True
    st.session_state.administer_confirm = {
        "medicine": medicine,
        "command_name": command_name,
    }
    append_log(command_name, f"Administered flow started for '{medicine}'.")
    if pipeline_error:
        append_log(command_name, f"Pipeline detail: {pipeline_error}")
        set_feedback(
            "info",
            f"Showing fallback med.png for '{medicine}' (model crop not available in this run).",
        )
    else:
        set_feedback("info", f"Review medication details for '{medicine}' and confirm administration below.")


def queue_pending_administer(medicine: str, command_name: str, reason: str) -> None:
    """Queue a warning gate that requires explicit nurse confirmation."""
    st.session_state.pending_action = {
        "type": "administer",
        "medicine": medicine,
        "command_name": command_name,
        "reason": reason,
    }
    set_feedback("warning", reason)


def render_pending_action() -> None:
    """Render pending warning gate with continue/cancel actions."""
    pending = st.session_state.pending_action
    if not pending:
        return

    st.warning(f"Warning: {pending['reason']}")
    col1, col2 = st.columns(2)
    if col1.button("Continue anyway", key="pending_continue", type="primary"):
        execute_administer(pending["medicine"], pending["command_name"])
        st.session_state.pending_action = None
        st.rerun()
    if col2.button("Cancel", key="pending_cancel"):
        append_log(pending["command_name"], "Cancelled after warning.")
        st.session_state.pending_action = None
        set_feedback("info", "Action cancelled.")
        st.rerun()


def handle_command(raw_command: str) -> None:
    """Parse and execute supported slash commands against current session state."""
    command = raw_command.strip()
    # Accept accidental double-slash input from users.
    if command.startswith("//"):
        command = command[1:]
    if not command:
        set_feedback("warning", "Enter a slash command.")
        return
    if not command.startswith("/"):
        set_feedback("error", "Commands must start with '/'.")
        return

    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    prescription_df: pd.DataFrame | None = st.session_state.prescription_df
    timetable_df: pd.DataFrame | None = st.session_state.timetable_df
    now = datetime.now()
    now_minutes = now.hour * 60 + now.minute

    if cmd == "/medicine-timetable":
        if prescription_df is None or prescription_df.empty:
            set_feedback("error", "Upload and parse a prescription image first.")
            append_log(cmd, "Failed: no parsed prescription.")
            return
        generated = build_timetable_from_prescription(prescription_df)
        st.session_state.timetable_df = generated
        st.session_state.override_mode = False
        set_feedback(
            "success",
            f"Timetable generated for {len(generated)} medicine(s), constrained to 07:00-19:00.",
        )
        append_log(cmd, f"Generated timetable with {len(generated)} rows.")
        return

    if cmd == "/override-medicine-timetable":
        if timetable_df is None or timetable_df.empty:
            set_feedback("error", "Generate timetable first using /medicine-timetable.")
            append_log(cmd, "Failed: timetable missing.")
            return
        st.session_state.override_mode = True
        st.session_state._override_version += 1
        old_key = f"override_editor_{st.session_state._override_version - 1}"
        if old_key in st.session_state:
            del st.session_state[old_key]
        set_feedback("info", "Override mode enabled. Edit timetable and click save.")
        append_log(cmd, "Opened timetable override editor.")
        return

    if cmd == "/medicine-administer":
        if not arg:
            set_feedback("error", "Usage: /medicine-administer <medicine-name>")
            append_log(cmd, "Failed: medicine name missing.")
            return
        if timetable_df is None or timetable_df.empty:
            set_feedback("error", "Generate timetable first using /medicine-timetable.")
            append_log(cmd, "Failed: timetable missing.")
            return

        matched = resolve_medicine_name(arg, timetable_df)
        if matched is None:
            queue_pending_administer(
                medicine=arg,
                command_name=cmd,
                reason=f"'{arg}' is not in the prescription list. Continue anyway?",
            )
            append_log(cmd, f"Warning queued: medicine '{arg}' not found.")
            return

        row = timetable_df[timetable_df["Medicine"] == matched].iloc[0]
        on_time, nearest, delta = nearest_schedule_delta(str(row["Times"]), now_minutes)
        if not on_time:
            queue_pending_administer(
                medicine=matched,
                command_name=cmd,
                reason=(
                    f"'{matched}' is not scheduled right now. "
                    f"Nearest slot is {human_time(nearest)} ({delta} min away). Continue?"
                ),
            )
            append_log(cmd, f"Warning queued: off-schedule for '{matched}'.")
            return

        execute_administer(matched, cmd)
        return

    if cmd == "/auto-administer":
        if timetable_df is None or timetable_df.empty:
            set_feedback("error", "Generate timetable first using /medicine-timetable.")
            append_log(cmd, "Failed: timetable missing.")
            return

        best_medicine = ""
        best_nearest = ""
        best_delta = 10_000
        for _, row in timetable_df.iterrows():
            medicine = str(row.get("Medicine", "")).strip()
            times_text = str(row.get("Times", ""))
            _, nearest, delta = nearest_schedule_delta(times_text, now_minutes)
            if delta < best_delta:
                best_delta = delta
                best_medicine = medicine
                best_nearest = nearest

        if not best_medicine:
            set_feedback("error", "No medicine schedule found for auto-administer.")
            append_log(cmd, "Failed: no timetable rows.")
            return

        if best_delta > ADMIN_WINDOW_MINUTES:
            queue_pending_administer(
                medicine=best_medicine,
                command_name=cmd,
                reason=(
                    f"Auto-administer selected '{best_medicine}', but it is off-schedule now. "
                    f"Nearest slot is {human_time(best_nearest)} ({best_delta} min away). Continue?"
                ),
            )
            append_log(cmd, f"Warning queued: auto-administer off-schedule for '{best_medicine}'.")
            return

        st.session_state.auto_administer_warn = True
        st.session_state._auto_administer_medicine = best_medicine
        set_feedback(
            "warning",
            f"Auto-administer selected '{best_medicine}'. "
            f"The robot will operate autonomously till it is stopped. Review the warning below.",
        )
        append_log(cmd, f"Awaiting autonomous pipeline confirmation for '{best_medicine}'.")
        return

    if cmd == "/wrong-medicine":
        if st.session_state.last_administered is None:
            set_feedback("error", "This command is available only after /medicine-administer or /auto-administer.")
            append_log(cmd, "Failed: no prior administer action.")
            return
        st.session_state.show_med_image = True
        append_log(cmd, "Wrong-medicine flow triggered. Showing med.png.")
        set_feedback("warning", "Wrong-medicine flow triggered. Showing med.png.")
        return

    if cmd == "/medicine-info":
        if not arg:
            set_feedback("error", "Usage: /medicine-info <medicine-name>")
            append_log(cmd, "Failed: medicine name missing.")
            return
        if prescription_df is None or prescription_df.empty:
            set_feedback("error", "Upload and parse a prescription image first.")
            append_log(cmd, "Failed: no parsed prescription.")
            return
        target = arg.lower()
        matches = prescription_df[
            prescription_df["Prescription"].astype(str).str.lower().str.contains(target, na=False)
        ]
        if matches.empty:
            set_feedback("warning", f"No medicine info found for '{arg}'.")
            append_log(cmd, f"No match for '{arg}'.")
            return
        first = matches.iloc[0]
        msg = (
            f"Medicine: {first['Prescription']} | "
            f"Instruction: {first['Instruction']} | "
            f"QTY: {first['QTY']} | Date: {first['Datefilled']} | Refill: {first['Refill']}"
        )
        set_feedback("info", msg)
        append_log(cmd, f"Returned medicine info for '{first['Prescription']}'.")
        return

    if cmd == "/stop":
        stopped_something = False
        if st.session_state.auto_administer_warn:
            st.session_state.auto_administer_warn = False
            st.session_state._auto_administer_medicine = ""
            stopped_something = True
        if st.session_state.administer_confirm is not None:
            stopped_something = True
            st.session_state.administer_confirm = None
        if st.session_state.show_med_image:
            stopped_something = True
            st.session_state.show_med_image = False
        if st.session_state.pending_action is not None:
            stopped_something = True
            st.session_state.pending_action = None

        if stopped_something:
            set_feedback("success", "Autonomous pipeline stopped.")
            append_log(cmd, "Stopped active pipeline.")
        else:
            set_feedback("info", "Nothing to stop — no active pipeline running.")
            append_log(cmd, "No active pipeline to stop.")
        return

    set_feedback("error", f"Unknown command '{cmd}'.")
    append_log(cmd, "Failed: unknown command.")


def render_timetable_display() -> None:
    """Read-only timetable view (safe inside scrollable container)."""
    timetable_df: pd.DataFrame | None = st.session_state.timetable_df
    if timetable_df is None or timetable_df.empty:
        return

    st.markdown("### Timetable Output")
    display_df = timetable_df.copy()
    display_df["Human Times"] = display_df["Times"].apply(
        lambda text: ", ".join(human_time(t) for t in [p.strip() for p in str(text).split(",") if p.strip()])
    )
    st.dataframe(display_df[["Medicine", "Times", "Human Times", "Notes"]], use_container_width=True)


def render_override_editor() -> None:
    """Form-based override editor using text_input per medicine row."""
    timetable_df: pd.DataFrame | None = st.session_state.timetable_df
    if timetable_df is None or timetable_df.empty:
        return
    if not st.session_state.override_mode:
        return

    st.markdown("### Override Timetable")
    st.caption("Edit times as comma-separated 24-hour values (example: `07:00, 13:00, 19:00`).")

    ver = st.session_state._override_version
    with st.form(f"override_form_{ver}", clear_on_submit=False):
        time_inputs: list[tuple[str, str, str]] = []
        for idx, row in timetable_df.iterrows():
            medicine = str(row.get("Medicine", "")).strip()
            current_times = str(row.get("Times", "")).strip()
            instruction = str(row.get("Instruction", "")).strip()
            new_times = st.text_input(
                medicine,
                value=current_times,
                key=f"override_times_{ver}_{idx}",
            )
            time_inputs.append((medicine, new_times, instruction))

        save_clicked = st.form_submit_button("Save timetable override")

    if save_clicked:
        normalized_rows: list[dict[str, str]] = []
        errors: list[str] = []
        warnings: list[str] = []

        for row_idx, (medicine, times_text, instruction) in enumerate(time_inputs):
            valid_times, invalid_times = parse_times_text(times_text)
            if invalid_times:
                errors.append(
                    f"{medicine}: invalid time(s): {', '.join(invalid_times)}"
                )
                continue
            if not valid_times:
                errors.append(f"{medicine}: at least one valid time is required.")
                continue

            out_of_hours = [
                t for t in valid_times if (hhmm_to_minutes(t) or 0) < WORKDAY_START or (hhmm_to_minutes(t) or 0) > WORKDAY_END
            ]
            if out_of_hours:
                warnings.append(
                    f"{medicine}: outside 07:00-19:00 -> {', '.join(out_of_hours)}"
                )

            normalized_rows.append(
                {
                    "Medicine": medicine,
                    "Times": ", ".join(valid_times),
                    "Instruction": instruction,
                    "Notes": "",
                }
            )

        if errors:
            set_feedback("error", "Could not save override. " + " | ".join(errors))
            return

        st.session_state.timetable_df = pd.DataFrame(normalized_rows)
        st.session_state.override_mode = False
        if warnings:
            set_feedback("warning", "Override saved with warnings: " + " | ".join(warnings))
        else:
            set_feedback("success", "Timetable override saved.")
        append_log("/override-medicine-timetable", "Timetable updated by user override.")
        st.rerun()


def render_command_catalog() -> None:
    """Display clickable command catalog that auto-fills command input."""
    st.markdown("### Available Commands")
    st.caption("Nurses can use any of these commands directly in the command box.")
    for idx, item in enumerate(COMMAND_CATALOG):
        c1, c2 = st.columns([4.5, 1])
        with c1:
            st.markdown(f"`{item['usage']}`")
            st.caption(f"{item['description']} | Needs: {item['needs']}")
        with c2:
            if st.button("Use", key=f"use_cmd_{idx}"):
                st.session_state._pending_execute_cmd = item["example"]
                st.rerun()


def render_command_chat() -> None:
    """Render recent command history with timestamped entries."""
    st.markdown("### Command Feed")
    logs = st.session_state.command_log[-10:]
    if not logs:
        st.info("No commands run yet.")
        return
    for entry in logs[::-1]:
        command = str(entry.get("Command", ""))
        outcome = str(entry.get("Outcome", ""))
        time_text = str(entry.get("Time", ""))
        st.markdown(
            f"""<div class="cmd-log-entry">
<span class="cmd-log-cmd">{command}</span>
<span class="cmd-log-outcome">{outcome}</span>
<span class="cmd-log-time">{time_text}</span>
</div>""",
            unsafe_allow_html=True,
        )


def main() -> None:
    """Streamlit entry point for upload, parsing, and command-driven workflows."""
    st.set_page_config(page_title="HALO RX Nurse UI", layout="wide")
    apply_styles()
    init_state()

    if st.session_state._pending_execute_cmd is not None:
        handle_command(st.session_state._pending_execute_cmd)
        st.session_state._pending_execute_cmd = None
        st.rerun()

    logo_candidates = [
        FRONTEND_ASSETS_DIR / "halorx.png",
        FRONTEND_ASSETS_DIR / "halorx.jpg",
        FRONTEND_ASSETS_DIR / "halorx.jpeg",
        POSTOP_ROOT / "halorx.png",
        POSTOP_ROOT / "halorx.jpg",
        POSTOP_ROOT / "halorx.jpeg",
    ]
    logo_path = next((path for path in logo_candidates if path.exists()), None)
    header_left, header_right = st.columns([1, 4], gap="small")
    with header_left:
        if logo_path is not None:
            st.image(str(logo_path), width=120)
    with header_right:
        st.title("HALO RX Nurse Console")
        st.caption("Upload prescription images, parse them, and run medication workflow slash commands.")

    lang = normalize_lang("eng")
    ready, message = ensure_ocr_ready(lang=lang)
    if not ready:
        # OCR is a hard dependency for the UI flow; fail early with actionable message.
        st.error(message)
        st.stop()

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.subheader("1) Upload and Parse Prescription")
        if st.session_state.prescription_df is not None:
            pres_df = st.session_state.prescription_df
            st.markdown("### Parsed Prescription")
            st.dataframe(pres_df, use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Medicines", len(pres_df))
            c2.metric("CSV", Path(st.session_state.last_csv_path).name if st.session_state.last_csv_path else "-")
            st.markdown("---")

        st.markdown("### Upload New Prescription Image")
        uploaded = st.file_uploader(
            "Upload prescription image",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG",
        )
        if st.button("Parse Image", type="primary"):
            if uploaded is None:
                set_feedback("warning", "Choose an image file first.")
                st.rerun()
            else:
                image_path = save_uploaded_file(uploaded)
                try:
                    parsed_df, _ = parse_image_to_dataframe(
                        image_path=image_path,
                        lang=lang,
                        min_confidence=50,
                    )
                except RuntimeError as exc:
                    set_feedback("error", str(exc))
                    parsed_df = None

                if parsed_df is None or parsed_df.empty:
                    set_feedback("error", "No prescription table could be extracted from this image.")
                else:
                    clean = standardize_prescription_dataframe(parsed_df)
                    csv_path = Path(f"{Path(uploaded.name).stem}.csv")
                    clean.to_csv(csv_path, index=False)
                    st.session_state.prescription_df = clean
                    st.session_state.prescription_image_path = str(image_path)
                    st.session_state.last_csv_path = str(csv_path)
                    st.session_state.timetable_df = None
                    st.session_state.override_mode = False
                    append_log("parse-image", f"Parsed {uploaded.name} and wrote {csv_path.name}.")
                    set_feedback("success", f"Prescription parsed. CSV saved as `{csv_path.name}`.")
                st.rerun()

        render_command_chat()

    with right:
        st.subheader("2) Nurse Command Console")

        render_pending_action()
        show_feedback()

        has_output = (
            st.session_state.timetable_df is not None
            or st.session_state.show_med_image
            or st.session_state.auto_administer_warn
        )
        if has_output:
            with st.container(height=420):
                render_timetable_display()
                if st.session_state.show_med_image:
                    med_path = ensure_med_image()
                    st.markdown("### med.png")
                    try:
                        with Image.open(med_path) as med_img:
                            med_img_rgb = med_img.convert("RGB")
                            disp_w = max(1, med_img_rgb.width // 2)
                            disp_h = max(1, med_img_rgb.height // 2)
                            med_img_small = med_img_rgb.resize((disp_w, disp_h), IMAGE_RESAMPLE)
                        st.image(med_img_small, use_container_width=False)
                    except Exception:
                        st.image(str(med_path), use_container_width=True)

                if st.session_state.administer_confirm is not None:
                    confirm = st.session_state.administer_confirm
                    st.markdown(f"**Administer {confirm['medicine']}?**")
                    ac1, ac2 = st.columns(2)
                    if ac1.button("Yes, administer", key="administer_yes", type="primary"):
                        append_log(
                            confirm["command_name"],
                            f"Confirmed: '{confirm['medicine']}' administered.",
                        )
                        set_feedback("success", f"'{confirm['medicine']}' has been administered successfully.")
                        st.session_state.administer_confirm = None
                        st.session_state.show_med_image = False
                        st.rerun()
                    if ac2.button("No, cancel", key="administer_no"):
                        append_log(
                            confirm["command_name"],
                            f"Cancelled administration of '{confirm['medicine']}'.",
                        )
                        set_feedback("info", f"Administration of '{confirm['medicine']}' cancelled.")
                        st.session_state.administer_confirm = None
                        st.session_state.show_med_image = False
                        st.rerun()

                if st.session_state.auto_administer_warn:
                    med = st.session_state._auto_administer_medicine
                    st.warning(
                        "**Autonomous Pipeline Warning**\n\n"
                        "The robot will operate autonomously till it is stopped. "
                        "Do you want to continue?"
                    )
                    aw1, aw2 = st.columns(2)
                    if aw1.button("Yes, continue", key="auto_yes", type="primary"):
                        st.session_state.auto_administer_warn = False
                        execute_administer(med, "/auto-administer")
                        st.rerun()
                    if aw2.button("No, cancel", key="auto_no"):
                        st.session_state.auto_administer_warn = False
                        st.session_state._auto_administer_medicine = ""
                        append_log("/auto-administer", "Cancelled autonomous pipeline.")
                        set_feedback("info", "Auto-administer cancelled.")
                        st.rerun()

                st_components.html(
                    """<script>
                    const frame = window.frameElement;
                    if (frame) {
                        let el = frame;
                        while (el) {
                            const style = el.ownerDocument.defaultView.getComputedStyle(el);
                            if (style.overflowY === 'scroll' || style.overflowY === 'auto') {
                                el.scrollTop = el.scrollHeight;
                                break;
                            }
                            el = el.parentElement;
                        }
                    }
                    </script>""",
                    height=0,
                )

        render_override_editor()

        st.markdown("### Enter Command")
        with st.form("slash_form", clear_on_submit=False):
            command_text = st.text_input(
                "Command",
                placeholder="/medicine-timetable",
                help="Type one slash command and submit.",
                key="command_input",
            )
            submitted = st.form_submit_button("Run command")

        render_command_catalog()

    if submitted:
        handle_command(command_text)
        st.rerun()

    


if __name__ == "__main__":
    main()
