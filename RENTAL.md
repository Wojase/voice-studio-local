# Vast.ai / Linux: two RTX 3090 GPUs

Use Python 3.11+ (recommended 3.12), FFmpeg, CUDA-enabled PyTorch and enough disk space for models, source audio and Trash. Run these inside your existing project directory after installation:

```bash
bash scripts/server.sh start
bash scripts/server.sh logs
bash scripts/server.sh status
bash scripts/server.sh stop
bash scripts/server.sh restart
```

The controller runs in the background, listens on `0.0.0.0:7860`, enables external Host headers for your Vast.ai tunnel, selects English Whisper large-v3 and configures two-GPU training with batch 4 per GPU. Stopping waits for an active job to stop and save. It does not forcibly kill a training process that fails to respond. Forward port **7860** using your Vast.ai tunnel panel.

## Update an existing installation

Stop or finish any running job first. Previous copy/paste setup commands changed tracked `app.py` and `web/index.html`; back those changes up in Git's local stash before pulling. No recordings or checkpoints are touched by these Git commands.

```bash
cd "${STUDIO_WORKSPACE:-$HOME/voice-work}/voice-studio-local"
git fetch origin
if ! git diff --quiet; then
  git stash push -m backup-before-english-srt-update
fi
git pull --ff-only
bash scripts/server.sh restart
```

Untracked custom launcher files are left in place. Use the new server controller to avoid old launcher settings overriding the defaults. If your environment already defines different settings, set these explicitly before restarting:

```bash
export STUDIO_HOST=0.0.0.0
export STUDIO_PUBLIC=1
export STUDIO_GPUS=2
export STUDIO_BATCH_SIZE=4
export STUDIO_STT_MODEL=openai/whisper-large-v3
bash scripts/server.sh restart
```

## Fresh Ubuntu 24.04 setup

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv ffmpeg libsndfile1 build-essential
git clone https://github.com/Wojase/voice-studio-local.git
cd voice-studio-local
bash scripts/setup-linux.sh
bash scripts/server.sh start
```

If logged in as root, omit `sudo`. On older Ubuntu, use Python 3.12 instead of the system's Python 3.10. A working NVIDIA driver supporting the CUDA wheel is required; check `nvidia-smi` before installing. Python dependencies are installed in `.venv`.

## Headless workflow

Do not run CLI and panel jobs simultaneously against the same dataset. An OS lock prevents concurrent model jobs.

```bash
source .venv/bin/activate
export STUDIO_STT_MODEL=openai/whisper-large-v3
python -m studio.importer /path/to/original/audio-or-video
python -m studio.cli prepare
# Review transcripts in the panel before training.
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  --module studio.cli train --epochs 3 --batch-size 4 --accumulation 2 --lr 0.00005
CUDA_VISIBLE_DEVICES=0 python -m studio.cli generate --text 'Hello. This is my saved voice.'
```

Start with batch 4 per GPU and accumulation 2 (effective batch up to 16). Increase batch to 6 or 8 only after measuring memory. With sufficient VRAM, `STUDIO_CHECKPOINTING=0` avoids activation recomputation but increases memory usage. Larger batches or more epochs do not automatically improve voice quality. Both GPUs train; STT, subtitle checks and individual synthesis currently use one GPU.

## Migration and backups

At rest, copy the entire `data` folder into this checkout. Keep all files in `models`, including the reference WAV and split metadata. Do not copy a Windows `.venv` to Linux. Source code is public; audio, subtitle contents, voice checkpoints and Trash are ignored by Git.

```bash
watch -n 1 nvidia-smi
python -m unittest discover -s tests -v
```

The trainer is experimental. Two-process gradient synchronization was checked locally using CPU Gloo, and single-GPU training/resume/generation was tested on RTX 2070. Full two-3090 and large-v3 inference need a rental-machine smoke test. See [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) for NCCL behavior.
