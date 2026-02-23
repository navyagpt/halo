#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
uv sync
uv run python -m streamlit run frontend/app.py
