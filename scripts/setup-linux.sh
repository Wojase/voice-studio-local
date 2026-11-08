#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
APT=(apt-get)
if [ "$(id -u)" -ne 0 ]; then APT=(sudo apt-get); fi
"${APT[@]}" update
"${APT[@]}" install -y python3 python3-venv python3-dev ffmpeg libsndfile1 git build-essential
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; print(torch.__version__); print("CUDA:",torch.cuda.is_available(),"GPUs:",torch.cuda.device_count()); assert torch.cuda.is_available()'
