# HALO-RX


## Table of Contents

1. [What HALO-RX Does](#what-halo-rx-does)
2. [System Architecture](#system-architecture)
3. [Repository Layout](#repository-layout)
4. [Runtime Data Flow](#runtime-data-flow)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Model Download (Required)](#model-download-required)
8. [Run Commands](#run-commands)
9. [Frontend Workflow](#frontend-workflow)
10. [Backend Pipeline](#backend-pipeline)
11. [Complete CLI Reference](#complete-cli-reference)
12. [Artifact Contracts](#artifact-contracts)
13. [Configuration & Tunables](#configuration--tunables)
14. [Operational Runbooks](#operational-runbooks)
15. [Troubleshooting](#troubleshooting)
16. [Developer Validation Commands](#developer-validation-commands)
17. [Notes for Collaborators](#notes-for-collaborators)

## What HALO-RX Does

HALO-RX provides two connected layers:

- **Frontend (`frontend/`)**: Streamlit nurse console.
  - Uploads prescription image.
  - Extracts table with OCR.
  - Builds medication timetable.
  - Accepts nurse slash commands (`/medicine-administer`, `/auto-administer`, etc.).
  - Calls backend pipeline to generate `med.png` from detected target medicine crop.
- **Backend (`backend/`)**: modular image pipeline.
  - Detects medicine bounding boxes.
  - Crops candidate pills.
  - Runs local MedSigLIP inference.
  - Matches predictions against target medicine.
  - Computes center coordinates for target matches.

---

## System Architecture

### High-Level Component Graph

```text
+------------------------+
|      Nurse Console     |
|   frontend/app.py      |
+-----------+------------+
            |
            | OCR parse (pytesseract + cv2 fallback logic)
            v
+------------------------+
| Parsed prescription DF |
| + timetable generation |
+-----------+------------+
            |
            | slash command: administer / auto-administer
            v
+------------------------+
| backend.pipeline       |
| (modular orchestrator) |
+-----------+------------+
            |
            +--> bounder  -> bbox JSON + boxed visualization
            +--> cropper  -> crop images
            +--> labeler  -> predictions.jsonl
            +--> center   -> centers.json
            v
+------------------------+
| matching crop copied   |
| to frontend/runtime/   |
| med.png                |
+------------------------+
```

### Backend Stage Pipeline

```text
source image
   |
   v
bounder.py
  - black-cloth ROI extraction
  - connected components
  - bbox filtering
   |
   v
cropper.py
  - crops bbox1.png, bbox2.png, ...
   |
   v
labeler.py
  - candidate meds from prescription
  - contrastive or linear_probe inference
  - top-k per crop
   |
   v
center.py
  - bbox center points
  - target-medicine match rows
```

---

## Repository Layout

```text
HALO-RX/
├── pyproject.toml
├── README.md
├── scripts/
│   └── run.sh
├── image-input/
├── frontend/
└── backend/
    ├── __init__.py
    ├── pipeline.py
    ├── prescriber.py
    ├── bounder.py
    ├── cropper.py
    ├── labeler.py
    ├── center.py
    ├── infer_core.py
    ├── models.py
    ├── pipeline_helpers.py
    ├── common.py
    ├── utils.py
    ├── labels.json
    ├── meds.txt
    ├── boundingbox/
    │   ├── main.py
    │   └── crop_bboxes.py
    └── inference/
        ├── models/
        └── processor/
```

## Prerequisites

- Python `>=3.10,<3.13`
- `uv` package manager
- Tesseract OCR binary on system path

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Tesseract

macOS (Homebrew):

```bash
brew install tesseract
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

Verify:

```bash
tesseract --version
```

---

## Installation

From repo root:

```bash
cd HALO-RX
uv sync
```

This installs all project dependencies from `pyproject.toml`, including platform-specific Torch/Torchvision pins.

---

## Model Download (Required)

Download the backend model assets from:

- `https://drive.google.com/drive/folders/1kkkcYwWtJQnKJdtWwpgQccnoZkVDLx7W?usp=sharing`

After download/extract, place files into the repo as:

- `backend/inference/models/`
- `backend/inference/processor/`

Expected layout:

```text
backend/inference/
├── models/
│   ├── config.json
│   └── model.safetensors
└── processor/
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── spiece.model
```

If your downloaded folder contains `models/` and `processor/`, you can copy with:

```bash
cp -R /path/to/downloaded/models/* backend/inference/models/
cp -R /path/to/downloaded/processor/* backend/inference/processor/
```

Quick verification:

```bash
ls -lah backend/inference/models
ls -lah backend/inference/processor
```

Note:

- Large model/checkpoint artifacts are intentionally ignored by `.gitignore`.

---

## Run Commands

### Frontend Launch

```bash
uv run python -m streamlit run frontend/app.py
```

### Equivalent helper script

```bash
bash scripts/run.sh
```

### Run backend pipeline directly

```bash
uv run --project . --no-sync python -m backend.pipeline \
  --image_path image-input/photo.jpg \
  --prescription_file backend/meds.txt \
  --target_medicine "Ibuprofen 800mg" \
  --top_k 4 \
  --threshold 0.0
```

---

## Frontend Workflow

### 1) Parse Prescription

1. Upload image in left panel.
2. Click **Parse Image**.
3. Extracted table is displayed and exported as `<uploaded_name>.csv` in current working directory.

### OCR extraction behavior

- Preferred path: OpenCV table-cell detection + Tesseract per cell.
- Fallback path: token-level OCR reconstruction when cv2 unavailable/failed.
- Language code defaults to `eng`.

### 2) Build and Adjust Timetable

- `/medicine-timetable` generates schedule constrained to **07:00 to 19:00**.
- `/override-medicine-timetable` enables manual edit form for times.

Timetable generation rules from instruction text:

- `every X hours` -> uses interval `X`.
- `every X-Y hours` -> uses upper bound `Y`.
- `N times a day` -> distributes slots across 07:00-19:00.
- `daily` / `once a day` -> once per day.
- no recognized pattern -> defaults to `09:00`.

### 3) Administer commands

- `/medicine-administer <name>`
  - resolves medicine name in timetable
  - checks timing window (45 min)
  - warns for off-schedule or unknown medicine
- `/auto-administer`
  - chooses nearest scheduled medicine
  - warns if off-schedule
- `/wrong-medicine`
  - displays `med.png` warning flow
- `/stop`
  - cancels active pending/admin/autonomous states

### Slash command reference

| Command | Purpose | Preconditions |
|---|---|---|
| `/medicine-timetable` | Generate schedule from parsed prescription | Parsed prescription exists |
| `/override-medicine-timetable` | Edit schedule manually | Timetable exists |
| `/medicine-administer <medicine>` | Start administer flow | Timetable exists |
| `/auto-administer` | Pick nearest medicine and start flow | Timetable exists |
| `/medicine-info <medicine>` | Show parsed metadata for medicine | Parsed prescription exists |
| `/wrong-medicine` | Trigger wrong-medicine flow | Administer flow already started |
| `/stop` | Stop/cancel active flow | None |

---

## Backend Pipeline

Backend package root is `backend/` and module namespace is `backend.*`.

### Full pipeline entrypoint

```bash
uv run --project . --no-sync python -m backend.pipeline --help
```

### Key flags

- `--image_path` (required): input image file
- `--run_dir`: where model/config/labels are loaded from (default: `backend/`)
- `--output_root`: run output base (default: `backend/runs`)
- `--prescription_file`: text file with one medicine per line
- `--prescription` (repeatable): inline candidate medicine
- `--target_medicine`: medicine to locate in predictions
- `--mode`: `contrastive|linear_probe|partial_unfreeze|lora_optional`
- `--batch_size`, `--top_k`, `--threshold`
- bbox controls: `--save_masks`, `--no_auto_thresholds`, `--v_dark`, `--s_dark_max`, `--v_obj`, `--s_obj`, `--v_col`, `--min_area`, `--max_aspect`

### Stage-by-stage commands

#### Prescriber

```bash
uv run --project . --no-sync python -m backend.prescriber \
  --prescription_file backend/meds.txt \
  --target_medicine "Ibuprofen 800mg" \
  --output_json backend/runs/manual/prescriber_output.json
```

Interactive mode:

```bash
uv run --project . --no-sync python -m backend.prescriber \
  --interactive \
  --output_json backend/runs/manual/prescriber_output.json
```

#### Bounder

```bash
uv run --project . --no-sync python -m backend.bounder \
  --image_path image-input/photo.jpg \
  --inputs_dir backend/runs/manual/inputs \
  --bbox_output_dir backend/runs/manual/bbox_outputs \
  --output_json backend/runs/manual/bounder_result.json
```

#### Cropper

```bash
uv run --project . --no-sync python -m backend.cropper \
  --input_dir backend/runs/manual/inputs \
  --json_dir backend/runs/manual/bbox_outputs/json \
  --output_dir backend/runs/manual/crops \
  --output_json backend/runs/manual/cropper_result.json
```

#### Labeler

```bash
uv run --project . --no-sync python -m backend.labeler \
  --run_dir backend \
  --input_path backend/runs/manual/crops \
  --prescriber_json backend/runs/manual/prescriber_output.json \
  --bbox_json backend/runs/manual/bbox_outputs/json/photo.json \
  --source_image photo.jpg \
  --top_k 4 \
  --threshold 0.0 \
  --output_jsonl backend/runs/manual/predictions.jsonl \
  --output_summary_json backend/runs/manual/labeler_result.json
```

#### Center

```bash
uv run --project . --no-sync python -m backend.center \
  --bbox_json backend/runs/manual/bbox_outputs/json/photo.json \
  --predictions_jsonl backend/runs/manual/predictions.jsonl \
  --target_medicine "Ibuprofen 800mg" \
  --output_json backend/runs/manual/centers.json
```

---

## Complete CLI Reference

This section lists the command surfaces with practical defaults and required arguments.

### Frontend App

```bash
uv run python -m streamlit run frontend/app.py
```

Primary behavior:

- Starts nurse UI.
- Requires Tesseract to be installed.
- On administer commands, calls `backend.pipeline`.
- Writes runtime outputs under `frontend/runtime/`.

### OCR Utility (`frontend/extract_table_to_csv.py`)

```bash
uv run python frontend/extract_table_to_csv.py [images ...] [flags]
```

Flags:

- `images` (optional positional): paths to images; if omitted, all local `*.png` are processed.
- `--output-dir` (default `.`): output directory for CSV files.
- `--lang` (default `eng`): Tesseract language code.
- `--min-confidence` (default `50`): OCR token confidence threshold.
- `--tesseract-cmd` (default empty): full path to Tesseract binary.

### `backend.pipeline`

```bash
uv run --project . --no-sync python -m backend.pipeline [flags]
```

Flags:

- `--image_path` (required)
- `--run_dir` (default `backend`)
- `--output_root` (default `backend/runs`)
- `--prescription_file` (optional)
- `--prescription` (repeatable, optional)
- `--target_medicine` (optional)
- `--prescriber_interactive` (optional flag)
- `--mode` in `contrastive|linear_probe|partial_unfreeze|lora_optional`
- `--batch_size` (default `8`)
- `--top_k` (default `5`)
- `--threshold` (default `0.0`)
- `--save_masks` (optional flag)
- `--no_auto_thresholds` (optional flag)
- `--v_dark` (default `70`)
- `--s_dark_max` (default `120`)
- `--v_obj` (default `135`)
- `--s_obj` (default `60`)
- `--v_col` (default `80`)
- `--min_area` (default auto)
- `--max_aspect` (default `4.5`)

### `backend.prescriber`

```bash
uv run --project . --no-sync python -m backend.prescriber [flags]
```

Flags:

- `--prescription_file` (optional)
- `--prescription` (repeatable, optional)
- `--target_medicine` (optional)
- `--interactive` (optional flag)
- `--output_json` (required)

### `backend.bounder`

```bash
uv run --project . --no-sync python -m backend.bounder [flags]
```

Flags:

- `--image_path` (required)
- `--inputs_dir` (required)
- `--bbox_output_dir` (required)
- `--save_masks` (optional flag)
- `--no_auto_thresholds` (optional flag)
- `--v_dark` (default `70`)
- `--s_dark_max` (default `120`)
- `--v_obj` (default `135`)
- `--s_obj` (default `60`)
- `--v_col` (default `80`)
- `--min_area` (optional)
- `--max_aspect` (default `4.5`)
- `--output_json` (optional)

### `backend.cropper`

```bash
uv run --project . --no-sync python -m backend.cropper [flags]
```

Flags:

- `--input_dir` (required)
- `--json_dir` (required)
- `--output_dir` (required)
- `--output_json` (optional)

### `backend.labeler`

```bash
uv run --project . --no-sync python -m backend.labeler [flags]
```

Flags:

- `--run_dir` (default `backend`)
- `--input_path` (required)
- `--prescription_file` (optional)
- `--prescriber_json` (optional)
- `--prescription` (repeatable, optional)
- `--mode` in `contrastive|linear_probe|partial_unfreeze|lora_optional`
- `--batch_size` (default `8`)
- `--top_k` (default `5`)
- `--threshold` (default `0.0`)
- `--output_jsonl` (required)
- `--bbox_json` (optional)
- `--source_image` (optional)
- `--output_summary_json` (optional)

### `backend.center`

```bash
uv run --project . --no-sync python -m backend.center [flags]
```

Flags:

- `--bbox_json` (required)
- `--predictions_jsonl` (optional)
- `--target_medicine` (optional)
- `--output_json` (required)

### `backend.infer_core` (standalone inference utility)

```bash
uv run --project . --no-sync python -m backend.infer_core [flags]
```

Flags:

- `--run_dir` (default `backend`)
- `--input_path` (required image or directory)
- `--mode` in `contrastive|linear_probe|partial_unfreeze|lora_optional`
- `--batch_size` (default `8`)
- `--top_k` (default `5`)
- `--threshold` (default `0.5`)
- `--candidate_meds_file` (optional)
- `--output_jsonl` (optional, otherwise prints JSON lines to stdout)

---

## Artifact Contracts

### Pipeline run folder naming

Each run is created under `<output_root>/<image_stem>_<YYYYMMDD_HHMMSS>/`.

### Standard files generated

- `prescriber_output.json`
- `prescription_candidates.txt`
- `bbox_outputs/json/<image_stem>.json`
- `bbox_outputs/vis/<image_stem>_boxed.png`
- `bbox_outputs/masks/*` (when `--save_masks` or no boxes)
- `crops/.../bbox1.png`, `bbox2.png`, ...
- `predictions.jsonl`
- `centers.json`
- `pipeline_result.json`
- stage summaries:
  - `bounder_result.json`
  - `cropper_result.json`
  - `labeler_result.json`

### Key JSON/JSONL schemas

#### `bbox_outputs/json/<image>.json`

```json
{
  "image": "photo.jpg",
  "width": 1920,
  "height": 1080,
  "bboxes": {
    "bbox1": [x1, y1, x2, y2],
    "bbox2": [x1, y1, x2, y2]
  }
}
```

#### `predictions.jsonl` (one row per crop)

```json
{
  "image_path": ".../crops/photo/bbox1.png",
  "top_k_labels": ["Ibuprofen 800mg", "..."],
  "top_k_scores": [0.92, 0.04, 0.02, 0.01],
  "predicted_label": "Ibuprofen 800mg",
  "confidence": 0.92,
  "abstain_flag": false,
  "bbox_name": "bbox1",
  "bbox_coords": [x1, y1, x2, y2],
  "source_image": "photo.jpg"
}
```

#### `centers.json`

```json
{
  "bbox_json": ".../photo.json",
  "predictions_jsonl": ".../predictions.jsonl",
  "target_medicine": "Ibuprofen 800mg",
  "all_bbox_centers": [
    {
      "bbox_name": "bbox1",
      "bbox_coords": [x1, y1, x2, y2],
      "center": [cx, cy],
      "predicted_label": "Ibuprofen 800mg",
      "confidence": 0.92
    }
  ],
  "target_bbox_centers": [
    {
      "bbox_name": "bbox1",
      "bbox_coords": [x1, y1, x2, y2],
      "center": [cx, cy],
      "predicted_label": "Ibuprofen 800mg",
      "confidence": 0.92
    }
  ]
}
```

---

## Configuration & Tunables

### OCR settings (`frontend/extract_table_to_csv.py`)

- `--lang`: Tesseract language code (default `eng`)
- `--min-confidence`: token confidence cutoff (default `50`)
- `--tesseract-cmd`: explicit binary path

### Scheduling settings (`frontend/app.py`)

- Workday window: `07:00` to `19:00`
- Administer warning window: `45` minutes

### Bounding box settings (`backend.pipeline` pass-through)

- `v_dark`, `s_dark_max`, `v_obj`, `s_obj`, `v_col`
- `min_area`, `max_aspect`
- `--no_auto_thresholds` to disable auto adaptation

### Inference settings

- `--mode` controls inference family
- `--top_k` controls ranked outputs
- `--threshold` controls abstain threshold (`UNKNOWN` when below threshold)

### Frontend-to-backend execution environment

When `frontend/app.py` spawns `backend.pipeline`, it sets:

- `PYTHONPATH=<repo_root>[:existing]` so `backend.*` modules resolve reliably
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

This means model assets are expected to be present locally under `backend/inference/`.

---

## Operational Runbooks (some issues that the authors faced while developing the project, and how to address them)

### Standard Nurse Session (UI)

1. Place source image in `image-input/` (recommended for deterministic backend selection).
2. Start UI:
   ```bash
   uv run python -m streamlit run frontend/app.py
   ```
3. Upload prescription image and click **Parse Image**.
4. Run `/medicine-timetable`.
5. Run `/medicine-administer <medicine>` or `/auto-administer`.
6. Confirm/cancel administration in right panel.
7. Review outputs in:
   - `frontend/runtime/med.png`
   - `frontend/runtime/pipeline_runs/`

### Backend-only Batch Run

1. Prepare source image path.
2. Prepare candidate file (`backend/meds.txt` or custom text file, one medicine per line).
3. Execute:
   ```bash
   uv run --project . --no-sync python -m backend.pipeline \
     --image_path image-input/photo.jpg \
     --prescription_file backend/meds.txt \
     --target_medicine "Ibuprofen 800mg" \
     --output_root backend/runs \
     --top_k 4 \
     --threshold 0.0
   ```
4. Inspect latest run directory under `backend/runs/`.

### Bounding-box Debug Session

Use this when crops are missing or wrong.

```bash
uv run --project . --no-sync python -m backend.bounder \
  --image_path image-input/photo.jpg \
  --inputs_dir backend/runs/debug/inputs \
  --bbox_output_dir backend/runs/debug/bbox_outputs \
  --save_masks \
  --output_json backend/runs/debug/bounder_result.json
```

Then inspect:

- `backend/runs/debug/bbox_outputs/vis/*_boxed.png`
- `backend/runs/debug/bbox_outputs/masks/*`

### OCR-only Extraction

```bash
uv run python frontend/extract_table_to_csv.py \
  image-input/photo.jpg \
  --output-dir . \
  --lang eng \
  --min-confidence 50
```

## Developer Validation Commands

### Syntax check

```bash
python3 -m compileall frontend backend
```

### Run OCR extractor directly

```bash
uv run python frontend/extract_table_to_csv.py \
  image-input/photo.jpg \
  --output-dir . \
  --lang eng \
  --min-confidence 50
```

### Quick backend import sanity check

```bash
python3 -c "import backend.pipeline; print('ok')"
```
