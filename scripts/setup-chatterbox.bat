@echo off
rem Separate environment for the Chatterbox backend (pins torch 2.6.0; never touches .venv).
cd /d "%~dp0\.."
if not exist .venv-chatterbox\Scripts\python.exe py -3.11 -m venv .venv-chatterbox || exit /b 1
.venv-chatterbox\Scripts\python -m pip install --upgrade pip
.venv-chatterbox\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
.venv-chatterbox\Scripts\python -m pip install chatterbox-tts==0.1.7 soundfile || exit /b 1
.venv-chatterbox\Scripts\python -c "import torch, chatterbox; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())"
