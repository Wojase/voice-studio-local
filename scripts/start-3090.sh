#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export STUDIO_GPUS=2
export STUDIO_BATCH_SIZE="${STUDIO_BATCH_SIZE:-4}"
export STUDIO_STT_MODEL="${STUDIO_STT_MODEL:-openai/whisper-large-v3}"
export STUDIO_HOST="${STUDIO_HOST:-0.0.0.0}"
export STUDIO_PUBLIC=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
# Keep checkpointing until real memory measurements justify disabling it.
exec .venv/bin/python -X utf8 app.py
