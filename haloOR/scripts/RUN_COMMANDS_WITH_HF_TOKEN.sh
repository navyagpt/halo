#!/usr/bin/env bash
set -euo pipefail

# This script can be launched from anywhere.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[error] HF_TOKEN is not set."
  echo "Set it first: export HF_TOKEN='hf_xxx_your_token_here'"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-combined.txt

# Auth check using Python (does not require huggingface-cli binary).
python - <<'PY'
import os
from huggingface_hub import whoami
token = os.environ["HF_TOKEN"]
print(whoami(token=token))
PY

python -m surgical_tool_pipeline.cli \
  --input assets/samples/images/image1.jpg \
  --output_dir pipeline_outputs \
  --device auto

# Optional audio-guided run:
# python -m surgical_tool_pipeline.cli \
#   --input /path/to/uploaded_image.jpg \
#   --audio_input /path/to/instrument_command.wav \
#   --audio_device auto \
#   --output_dir pipeline_outputs \
#   --device auto
