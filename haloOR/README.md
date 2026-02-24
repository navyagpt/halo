# HALO-OR

## Table of Contents

1. [What HALO-OR Does](#what-halo-or-does)
2. [System Architecture](#system-architecture)
3. [Repository Layout](#repository-layout)
4. [Runtime Data Flow](#runtime-data-flow)
5. [Prerequisites](#prerequisites)
6. [Installation](#installation)
7. [Model Download (Required)](#model-download-required)
8. [Run Commands](#run-commands)
9. [Audio Workflow](#audio-workflow)
10. [Core Pipeline](#core-pipeline)
11. [Complete CLI Reference](#complete-cli-reference)
12. [Artifact Contracts](#artifact-contracts)
13. [Configuration & Tunables](#configuration--tunables)
14. [Operational Runbooks](#operational-runbooks)
15. [Troubleshooting](#troubleshooting)
16. [Developer Validation Commands](#developer-validation-commands)
17. [Notes for Collaborators](#notes-for-collaborators)

## What HALO-OR Does

HALO-OR is an audio-guided surgical instrument localization pipeline.

It combines four capabilities:

- **Input resolution**: accepts image file/folder from CLI or Robot API response.
- **Instrument detection**: classical CV detector finds bbox candidates on dark surgical mats.
- **Instrument classification**: MedSigLIP embeddings + saved scikit-learn classifier head.
- **Audio-guided targeting (optional)**: transcribes audio command with MedASR and selects first bbox whose predicted label matches the extracted instrument.

Primary outputs include:

- Per-image JSON records (`json/`)
- Box visualizations (`vis/`)
- Crops (`crops/`)
- Merged machine-readable reports (`reports/merged_bbox_predictions.json` and `.csv`)
- Optional first audio-matched bbox report (`reports/first_audio_matched_bbox.json`)

---

## System Architecture

### High-Level Component Graph

```text
+------------------------------+
| CLI / API Request            |
| surgical_tool_pipeline.cli   |
| surgical_tool_pipeline.api   |
+---------------+--------------+
                |
                | input path resolution
                v
+------------------------------+
| robot.py (optional)          |
| resolve_input_path()         |
+---------------+--------------+
                |
                | image file(s)
                v
+------------------------------+
| detector.py                  |
| bboxsi_main primitives       |
| - ROI mask                   |
| - object mask                |
| - split merged blobs         |
| - bbox extraction + crops    |
+---------------+--------------+
                |
                | crop paths
                v
+------------------------------+
| classifier.py                |
| medsiglip_infer helpers      |
| - embeddings                 |
| - scikit classifier head     |
+---------------+--------------+
                |
                | attach predictions to detections
                v
+------------------------------+
| pipeline.py                  |
| merged reports + CSV         |
| first audio-matched bbox     |
+------------------------------+

Optional parallel audio branch:

audio file --> audio.py --> audio_instrument/asr.py --> audio_instrument/extract.py
                      MedASR transcript        canonical target instrument label
```

### Runtime Stage Pipeline

```text
input path (manual or robot)
   |
   v
collect_images()
   |
   v
detect_and_crop_one_image()
  - ROI extraction
  - foreground extraction
  - connected-components + bbox filtering
  - crop export
   |
   v
predict_crops()
  - load classifier artifacts
  - load MedSigLIP backbone
  - embed crops
  - predict class probabilities
   |
   v
merge + export
  - merged_bbox_predictions.json
  - merged_bbox_predictions.csv
  - first_audio_matched_bbox.json (if match found)
```

---

## Repository Layout

```text
haloOR/
├── README.md
├── requirements-combined.txt
├── assets/
│   └── samples/
│       ├── audio/
│       └── images/
├── docs/
│   ├── RUN_INSTRUCTIONS.md
│   ├── model_build.md
│   └── images/
├── scripts/
│   ├── RUN_COMMANDS.sh
│   └── RUN_COMMANDS_WITH_HF_TOKEN.sh
└── surgical_tool_pipeline/
    ├── __init__.py
    ├── api.py
    ├── audio.py
    ├── bboxsi_main.py
    ├── classifier.py
    ├── cli.py
    ├── config.py
    ├── detector.py
    ├── helpers.py
    ├── medsiglip_infer.py
    ├── pipeline.py
    ├── robot.py
    ├── audio_instrument/
    │   ├── asr.py
    │   ├── extract.py
    │   └── utils.py
    └── models/
        └── best_model/
            ├── model.joblib
            ├── config.json
            ├── label_mapping.json
            └── candidate_results.json
```

## Runtime Data Flow

For one run of `python -m surgical_tool_pipeline.cli`:

1. Parse CLI arguments into `PipelineConfig` (`cli.py` + `config.py`).
2. Resolve input source:
   - manual path (`--input`) when robot mode is off
   - robot API response when `--robot` is enabled
3. Optionally run audio stage (`audio.py`) if `--audio_input` is provided.
4. Collect image files from input path (single file or directory, optional recursive scan).
5. For each image:
   - detect bboxes
   - save per-image artifacts
   - generate crop images
6. Load classifier artifacts + MedSigLIP backbone.
7. Embed crops and classify each valid crop.
8. Attach predictions back to bbox records.
9. Match the first predicted bbox against audio target instrument (if audio enabled).
10. Write merged JSON/CSV reports and optional skipped-crop log.

---

## Prerequisites

- Python `>=3.10`
- `pip`/virtualenv workflow (or equivalent)
- Network access on first run to download Hugging Face backbones:
  - `google/medsiglip-448`
  - `google/medasr` (when audio is used)
- Optional GPU/CUDA for acceleration
- Optional Hugging Face token for gated model access

Install toolchain (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt
```

---

## Installation

From repo root:

```bash
cd haloOR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt
```

Equivalent helper scripts:

```bash
bash scripts/RUN_COMMANDS.sh
```

or, when auth is needed:

```bash
export HF_TOKEN="hf_xxx_your_token_here"
bash scripts/RUN_COMMANDS_WITH_HF_TOKEN.sh
```

---

## Model Download (Required)

HALO-OR model artifacts are distributed from the **same Google Drive as HALO-RX**:

- `https://drive.google.com/drive/folders/1kkkcYwWtJQnKJdtWwpgQccnoZkVDLx7W?usp=sharing`

Place HALO-OR classifier artifacts under:

- `surgical_tool_pipeline/models/best_model/`

Expected HALO-OR model directory layout:

```text
surgical_tool_pipeline/models/best_model/
├── model.joblib
├── config.json
├── label_mapping.json
└── candidate_results.json
```

If your download has a `best_model/` folder:

```bash
cp -R /path/to/downloaded/best_model/* surgical_tool_pipeline/models/best_model/
```

Quick verification:

```bash
ls -lah surgical_tool_pipeline/models/best_model
```

Notes:

- `model.joblib` + `label_mapping.json` are required for classification.
- `config.json` controls backbone resolution fallback (`model_id`) for classifier.

---

## Run Commands

### Image-only run

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs \
  --device auto
```

### Audio-guided run

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --audio_input assets/samples/audio/new.wav \
  --audio_device auto \
  --device auto \
  --output_dir pipeline_outputs
```

### Folder run (recursive)

```bash
python -m surgical_tool_pipeline.cli \
  --input /path/to/image_folder \
  --recursive \
  --output_dir pipeline_outputs
```

### Restrict predictions to candidate labels

```bash
python -m surgical_tool_pipeline.cli \
  --input /path/to/image.jpg \
  --candidate_labels forceps scissors
```

### Robot API mode

```bash
python -m surgical_tool_pipeline.cli \
  --robot \
  --robot_api_url http://localhost:8000/get_image_path \
  --robot_api_method POST \
  --robot_payload_json '{"case_id": 101}' \
  --input /optional/fallback/path.jpg \
  --output_dir pipeline_outputs
```

---

## Audio Workflow

Audio extraction is optional and only runs when `--audio_input` is provided.

### Stage sequence

1. `audio.py` validates and resolves audio path.
2. `audio_instrument/asr.py` transcribes with MedASR (`google/medasr` only).
3. `audio_instrument/extract.py` normalizes transcript text and extracts canonical instrument label.
4. `pipeline.py` finds the first predicted bbox matching that label.

### Canonical labels

- `forceps`
- `hemostat`
- `scissors`
- `scalpel`
- `unknown` (fallback when no robust match)

### Tie-break policy

When transcript contains multiple instruments, chosen priority is:

- `forceps > hemostat > scissors > scalpel`

### Audio output payload fields

`audio_result` in merged JSON includes:

- `audio_path`
- `transcript`
- `instrument`
- `matched_pattern`
- `all_matches`
- `model_id`
- `device`

---

## Core Pipeline

Pipeline package root is `surgical_tool_pipeline/` and namespace is `surgical_tool_pipeline.*`.

### Full pipeline entrypoint

```bash
python -m surgical_tool_pipeline.cli --help
```

### Key top-level flags

- `--input`: image file/folder path (required unless `--robot` with valid API response)
- `--output_dir`: output root (default: `pipeline_outputs`)
- `--recursive`: recurse folders
- `--extensions`: accepted image extensions
- `--save_masks`: export ROI/object masks
- `--crop_pad`: bbox crop padding in pixels
- `--model_dir`: classifier artifacts directory
- `--model_id_override`: override MedSigLIP backbone ID
- `--batch_size`, `--num_workers`, `--device`, `--no_amp`
- `--candidate_labels`: prediction label subset
- `--audio_input`, `--audio_device`, `--audio_model_id`, `--audio_chunk_length_s`, `--audio_stride_length_s`
- `--robot`, `--robot_api_url`, `--robot_api_method`, `--robot_timeout_sec`, `--robot_payload_json`, `--robot_response_image_key`, `--robot_response_image_list_key`
- Detector tuning: `--v_dark`, `--roi_close_k`, `--roi_open_k`, `--delta_v`, `--s_min`, `--obj_close_k`, `--obj_open_k`, `--min_area`, `--max_area_frac`, `--max_aspect`, `--split_merged`, `--split_erode_max_iter`, `--auto_thresholds`

### Stage-by-stage commands

#### Full CLI orchestration

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs
```

#### Standalone detector

```bash
python -m surgical_tool_pipeline.bboxsi_main \
  --input_dir assets/samples/images \
  --output_dir bbox_outputs \
  --recursive
```

#### Standalone classifier inference

```bash
python -m surgical_tool_pipeline.medsiglip_infer \
  --model_dir surgical_tool_pipeline/models/best_model \
  --input_folder assets/samples/images \
  --output_csv pipeline_outputs/reports/preds_from_input.csv \
  --device auto
```

#### Programmatic API

```python
from surgical_tool_pipeline import run_pipeline_in_memory

result = run_pipeline_in_memory(
    input_path="assets/samples/images/image1.jpg",
    audio_input_path="assets/samples/audio/new.wav",
    output_dir="pipeline_outputs",
    device="auto",
    audio_device="auto",
)

print(result["audio_target_instrument"])
print(result["first_audio_matched_bbox"])
```

---

## Complete CLI Reference

This section lists command surfaces with practical defaults and required arguments.

### `surgical_tool_pipeline.cli`

```bash
python -m surgical_tool_pipeline.cli [flags]
```

Flags:

- Input/output:
  - `--input` (required when `--robot` is false)
  - `--output_dir` (default `pipeline_outputs`)
  - `--recursive` (optional)
  - `--extensions` (default `.jpg .jpeg .png .bmp .tif .tiff .webp`)
  - `--save_masks` (optional)
  - `--crop_pad` (default `0`)
- Classifier:
  - `--model_dir` (default `surgical_tool_pipeline/models/best_model`)
  - `--model_id_override` (optional)
  - `--batch_size` (default `128`)
  - `--num_workers` (default `min(8, cpu_count)`)
  - `--device` (`auto|cuda|cpu`, default `auto`)
  - `--no_amp` (optional)
  - `--candidate_labels` (optional, supports space/comma separated values)
- Audio:
  - `--audio_input` (optional)
  - `--audio_device` (`auto|cuda|cpu`, default `auto`)
  - `--audio_model_id` (default `google/medasr`, only this model is accepted)
  - `--audio_chunk_length_s` (default `0.0`)
  - `--audio_stride_length_s` (optional)
- Robot:
  - `--robot / --no-robot` (default `--no-robot`)
  - `--robot_api_url` (required when `--robot` is enabled)
  - `--robot_api_method` (`GET|POST`, default `POST`)
  - `--robot_timeout_sec` (default `10.0`)
  - `--robot_payload_json` (optional JSON object string)
  - `--robot_response_image_key` (default `image_path`)
  - `--robot_response_image_list_key` (default `image_paths`)
- Detector tuning:
  - `--v_dark` (manual override; default derived as `110` when unset)
  - `--roi_close_k` (default `31`)
  - `--roi_open_k` (default `17`)
  - `--delta_v` (default `35`)
  - `--s_min` (default `25`)
  - `--obj_close_k` (default `23`)
  - `--obj_open_k` (default `7`)
  - `--min_area` (optional auto when unset)
  - `--max_area_frac` (default `0.60`)
  - `--max_aspect` (default `20.0`)
  - `--split_merged / --no-split_merged` (default enabled)
  - `--split_erode_max_iter` (default `8`)
  - `--auto_thresholds / --no-auto_thresholds` (default enabled)

### `surgical_tool_pipeline.bboxsi_main`

```bash
python -m surgical_tool_pipeline.bboxsi_main [flags]
```

Flags:

- `--input_dir` (default `.`)
- `--output_dir` (default `outputs`)
- `--recursive` (optional)
- `--save_masks` (optional)
- `--save_crops / --no-save_crops` (default enabled)
- `--crop_pad` (default `0`)
- `--extensions` (optional extension list override)
- ROI/object/component tuning:
  - `--v_dark`, `--roi_close_k`, `--roi_open_k`
  - `--delta_v`, `--s_min`, `--obj_close_k`, `--obj_open_k`
  - `--min_area`, `--max_area_frac`, `--max_aspect`
  - `--split_merged`, `--split_erode_max_iter`
  - `--auto_thresholds`

### `surgical_tool_pipeline.medsiglip_infer`

```bash
python -m surgical_tool_pipeline.medsiglip_infer [flags]
```

Flags:

- `--model_dir` (default `outputs/models/best_model`; for this repo usually pass `surgical_tool_pipeline/models/best_model`)
- `--input_folder` (required)
- `--output_csv` (default `outputs/reports/preds_from_input.csv`)
- `--batch_size` (default `64`)
- `--num_workers` (default `4`)
- `--device` (`auto|cuda|cpu`, default `auto`)
- `--no_amp` (optional)
- `--model_id_override` (optional)
- `--candidate_labels` (optional subset filter)

### `surgical_tool_pipeline.api`

Programmatic entrypoints:

- `run_pipeline_in_memory(...)`
- `run_pipeline_api_entrypoint(request_obj)`

The API wrapper normalizes request payloads and forwards to the same internal pipeline used by CLI.

---

## Artifact Contracts

### Output directory structure

For `--output_dir pipeline_outputs`, this pipeline writes:

- `pipeline_outputs/json/`
- `pipeline_outputs/vis/`
- `pipeline_outputs/crops/`
- `pipeline_outputs/reports/`
- `pipeline_outputs/masks/` (only when `--save_masks`)

### Standard files generated

- Per-image detection JSON:
  - `json/<image_id>.json`
- Per-image visualization:
  - `vis/<image_id>_boxed.png`
- Per-image crops:
  - `crops/<image_id>/bbox1.png`, `bbox2.png`, ...
- Merged reports:
  - `reports/merged_bbox_predictions.json`
  - `reports/merged_bbox_predictions.csv`
- Optional reports:
  - `reports/first_audio_matched_bbox.json` (only when audio label matches one predicted bbox)
  - `reports/classification_skipped.log` (only when some crop files fail to load)

### Key JSON schemas

#### Per-image JSON (`json/<image_id>.json`)

```json
{
  "image_id": "img00001_image1",
  "image_path": "/abs/path/image1.jpg",
  "image_name": "image1.jpg",
  "width": 1920,
  "height": 1080,
  "roi_area_frac": 0.84,
  "bbox_count": 3,
  "detections": [
    {
      "bbox_id": "bbox1",
      "bbox_xyxy": [x1, y1, x2, y2],
      "bbox_centroid_xy": [cx, cy],
      "crop_path": "/abs/path/pipeline_outputs/crops/img00001_image1/bbox1.png",
      "crop_exists": true,
      "prediction": {
        "label": "forceps",
        "class_id": 0,
        "confidence": 0.97,
        "probabilities": {
          "forceps": 0.97,
          "hemostat": 0.02,
          "scalpel": 0.01,
          "scissors": 0.00
        }
      }
    }
  ],
  "detector_meta": {
    "roi": {},
    "objects": {},
    "bboxes": {},
    "split": {}
  }
}
```

#### Merged report (`reports/merged_bbox_predictions.json`)

```json
{
  "input": "/abs/path/input",
  "output_dir": "/abs/path/pipeline_outputs",
  "model_dir": "/abs/path/surgical_tool_pipeline/models/best_model",
  "model_id": "google/medsiglip-448",
  "robot_mode": false,
  "robot_api_meta": null,
  "audio_mode": true,
  "audio_result": {},
  "audio_target_instrument": "forceps",
  "first_audio_matched_bbox": {},
  "candidate_labels_requested": [],
  "summary": {
    "images": 1,
    "bboxes": 3,
    "classified_bboxes": 3,
    "unclassified_bboxes": 0,
    "skipped_crops": 0,
    "audio_match_found": true
  },
  "class_labels": ["forceps", "hemostat", "scalpel", "scissors"],
  "images": []
}
```

#### Flattened CSV (`reports/merged_bbox_predictions.csv`)

Columns include:

- `image_id`, `image_path`, `bbox_id`
- bbox coordinates and centroid
- `crop_path`
- `predicted_label`, `predicted_class_id`, `confidence`
- one probability column per class (`prob_<label>`)

---

## Configuration & Tunables

### Detector controls

Main detector parameters exposed via CLI:

- ROI tuning: `v_dark`, `roi_close_k`, `roi_open_k`
- Foreground tuning: `delta_v`, `s_min`, `obj_close_k`, `obj_open_k`
- Component filtering: `min_area`, `max_area_frac`, `max_aspect`
- Split logic: `split_merged`, `split_erode_max_iter`
- Auto mode: `auto_thresholds`
- Crop expansion: `crop_pad`

### Classifier controls

- `model_dir`: where classifier artifacts are loaded from
- `model_id_override`: override backbone model id
- `batch_size`, `num_workers`
- `device`: `auto|cuda|cpu`
- `use_amp` (inverse of `--no_amp`)
- `candidate_labels`: optional subset restriction with probability renormalization

### Audio controls

- `audio_input` toggles audio stage
- `audio_model_id` (must be `google/medasr`)
- `audio_device`
- `audio_chunk_length_s`
- `audio_stride_length_s`

### Robot controls

- API URL and method
- timeout
- optional JSON payload for POST
- response field keys for single path or list path
- fallback to `--input` when robot response lacks image path

### Hugging Face authentication environment variables

Accepted token variables in code:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`
- `HUGGING_FACE_HUB_TOKEN`

---

## Operational Runbooks

### Standard image-only session

1. Activate environment and install dependencies.
2. Confirm model artifacts under `surgical_tool_pipeline/models/best_model/`.
3. Run:
   ```bash
   python -m surgical_tool_pipeline.cli \
     --input assets/samples/images/image1.jpg \
     --output_dir pipeline_outputs
   ```
4. Inspect:
   - `pipeline_outputs/reports/merged_bbox_predictions.json`
   - `pipeline_outputs/vis/`
   - `pipeline_outputs/crops/`

### Audio-guided session

1. Ensure audio file exists (for example `assets/samples/audio/new.wav`).
2. Run:
   ```bash
   python -m surgical_tool_pipeline.cli \
     --input assets/samples/images/image1.jpg \
     --audio_input assets/samples/audio/new.wav \
     --audio_device auto \
     --output_dir pipeline_outputs
   ```
3. Check:
   - `audio_target_instrument` in merged JSON
   - `first_audio_matched_bbox.json` (if match found)

### Robot API session

1. Ensure robot endpoint returns JSON with one of:
   - `image_path`, or
   - `image_paths` list
2. Run:
   ```bash
   python -m surgical_tool_pipeline.cli \
     --robot \
     --robot_api_url http://localhost:8000/get_image_path \
     --robot_api_method POST \
     --robot_payload_json '{"case_id": 101}' \
     --input /optional/fallback/image.jpg \
     --output_dir pipeline_outputs
   ```
3. Verify `robot_api_meta` in merged JSON.

### Detector-debug session (classical CV only)

```bash
python -m surgical_tool_pipeline.bboxsi_main \
  --input_dir assets/samples/images \
  --output_dir bbox_outputs \
  --save_masks \
  --recursive
```

Then inspect:

- `bbox_outputs/vis/*_boxed.png`
- `bbox_outputs/masks/*`
- `bbox_outputs/json/*.json`

### Standalone classifier session

```bash
python -m surgical_tool_pipeline.medsiglip_infer \
  --model_dir surgical_tool_pipeline/models/best_model \
  --input_folder assets/samples/images \
  --output_csv pipeline_outputs/reports/preds_from_input.csv
```

---

## Troubleshooting

### `FileNotFoundError: Missing model artifact ... model.joblib`

- Confirm files are present under `surgical_tool_pipeline/models/best_model/`.
- Re-download HALO-OR model artifacts from the shared Google Drive.

### `Failed to load MedSigLIP backbone`

- Ensure internet access to `huggingface.co` on first model download.
- Export one supported token variable when model access requires auth.

### `401 Unauthorized` from Hugging Face

- Verify token in current shell:
  ```bash
  python -c "import os; print(bool(os.environ.get('HF_TOKEN')))"
  ```
- Confirm account access to required model repositories.

### `No matching images found under: ...`

- Check `--input` path.
- Check extension filters and `--recursive` usage.

### `first_audio_matched_bbox` is `null`

- Audio extracted instrument may be `unknown`.
- No predicted bbox label matched the extracted instrument.

### `CUDA requested but not available`

- Use `--device cpu` and/or `--audio_device cpu`.
- Or use default `--device auto`.

---

## Developer Validation Commands

### Syntax check

```bash
python3 -m compileall surgical_tool_pipeline
```

### CLI surface sanity check

```bash
python -m surgical_tool_pipeline.cli --help
python -m surgical_tool_pipeline.bboxsi_main --help
python -m surgical_tool_pipeline.medsiglip_infer --help
```

### Sample end-to-end run

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs \
  --device auto
```

### Python import check

```bash
python -c "import surgical_tool_pipeline as s; print(hasattr(s, 'run_pipeline'))"
```

## Notes for Collaborators

- Keep canonical label vocabulary stable:
  - `forceps`, `hemostat`, `scissors`, `scalpel`
- Preserve report keys consumed by downstream integrations:
  - `audio_target_instrument`
  - `first_audio_matched_bbox`
  - `summary`
  - `images[*].detections[*].prediction`
- If you retrain the classifier head, keep artifact schema consistent with:
  - `model.joblib`
  - `config.json`
  - `label_mapping.json`
  - `candidate_results.json`
- `surgical_tool_pipeline.medsiglip_infer` uses a legacy default model dir (`outputs/models/best_model`), so pass `--model_dir surgical_tool_pipeline/models/best_model` unless you intentionally store artifacts elsewhere.
