#!/usr/bin/env bash
set -euo pipefail

# This script can be launched from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1) Create and activate virtual environment.
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies.
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt

# 3) Optional auth for gated Hugging Face backbones.
# export HF_TOKEN="hf_xxx_your_token_here"
# huggingface-cli login --token "$HF_TOKEN"

# 4) Run pipeline on included sample image.
python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs \
  --device auto

# 5) Optional: run on a folder recursively.
# python -m surgical_tool_pipeline.cli --input /path/to/images --recursive --output_dir pipeline_outputs --device auto

# 6) Optional: audio-guided first-bbox selection.
# python -m surgical_tool_pipeline.cli \
#   --input /path/to/uploaded_image.jpg \
#   --audio_input /path/to/instrument_command.wav \
#   --audio_device auto \
#   --output_dir pipeline_outputs
