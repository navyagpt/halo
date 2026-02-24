# Run Instructions

Quick runbook for the reorganized repository.

## 1) Environment Setup

From the parent directory that contains `haloOR/`, enter the repo first:

```bash
cd haloOR
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt
```

If you are already inside `haloOR/`, skip `cd haloOR`.

## 2) Optional Hugging Face Authentication

Needed for default backbone IDs (`google/medsiglip-448`, `google/medasr`) when your account requires auth:

```bash
export HF_TOKEN="hf_xxx_your_token_here"
python -c "import os; from huggingface_hub import whoami; print(whoami(token=os.environ['HF_TOKEN']))"
```

Note: first-time runs need network access to download model files from Hugging Face.

## 3) Image-Only Run

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs \
  --device auto
```

## 4) Audio-Guided Run

```bash
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --audio_input assets/samples/audio/new.wav \
  --audio_device auto \
  --device auto \
  --output_dir pipeline_outputs
```

## 5) Run on a Folder

```bash
python -m surgical_tool_pipeline.cli \
  --input /path/to/image_folder \
  --recursive \
  --output_dir pipeline_outputs
```

## 6) Key Outputs

- `pipeline_outputs/reports/merged_bbox_predictions.json`
- `pipeline_outputs/reports/merged_bbox_predictions.csv`
- `pipeline_outputs/reports/first_audio_matched_bbox.json` (created only when audio match exists)
- `pipeline_outputs/crops/`
- `pipeline_outputs/vis/`

## 7) Helper Scripts

- `bash scripts/RUN_COMMANDS.sh`
- `bash scripts/RUN_COMMANDS_WITH_HF_TOKEN.sh`
