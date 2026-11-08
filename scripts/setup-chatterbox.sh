#!/usr/bin/env bash
# Separate environment for the Chatterbox backend (pins torch 2.6.0; never touches .venv).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${CHATTERBOX_PYTHON:-python3.11}"
[ -x .venv-chatterbox/bin/python ] || "$PY" -m venv .venv-chatterbox
.venv-chatterbox/bin/python -m pip install --upgrade pip
.venv-chatterbox/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-chatterbox/bin/python -m pip install chatterbox-tts==0.1.7 soundfile
.venv-chatterbox/bin/python -c "import torch, chatterbox; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
