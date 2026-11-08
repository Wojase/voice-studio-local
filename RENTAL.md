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

## Update an existing installation (one command, Jupyter terminal)

Stops the panel gracefully (a running job saves first), writes a verified backup of models and metadata to
`~/voice-work/backups`, saves any local modifications from earlier setup snippets as a patch file **and** a Git
stash (never a hard reset), pulls, updates dependencies and restarts. Untracked custom launchers are left in place.

```bash
cd ~/voice-work/voice-studio-local && bash scripts/server.sh stop && T=$(date +%Y%m%d-%H%M%S) && mkdir -p ~/voice-work/backups && STUDIO_BACKUPS=~/voice-work/backups .venv/bin/python -m studio.backup models 2>/dev/null; git diff > ~/voice-work/backups/local-changes-$T.patch; git diff --quiet || git stash push -m "backup-before-update-$T"; git pull --ff-only && .venv/bin/python -m pip install -r requirements.txt && bash scripts/server.sh start
```

The old version has no `studio.backup`, so the first update skips that step (`2>/dev/null`). Back up first with
the `tar` command in the next section. Restore stashed edits with `git stash list` / `git stash apply`.

## Backups to download through Jupyter

```bash
cd ~/voice-work/voice-studio-local && STUDIO_BACKUPS=~/voice-work/backups .venv/bin/python -m studio.backup all && ls -lh ~/voice-work/backups
```

Then open the Jupyter file browser at `voice-work/backups/`, tick the `.zip` and `.sha256` files and click
**Download**. Verify locally with `sha256sum -c *.sha256`. Files left on the rented instance disappear when it is
destroyed. On a version without `studio.backup`:

```bash
cd ~/voice-work/voice-studio-local && D="${STUDIO_DATA:-$PWD/data}" && T=$(date +%Y%m%d-%H%M%S) && B=~/voice-work/backups && mkdir -p $B && (cd "$D" && tar -czf $B/voice-models-$T.tar.gz models subtitles jobs $(ls samples/*.json clips/*.json 2>/dev/null)) && (cd "$D" && tar -czf $B/voice-dataset-$T.tar.gz --exclude=trash samples clips features subtitles) && cd $B && sha256sum voice-*-$T.tar.gz > SHA256SUMS-$T.txt && ls -lh
```

Optional Chatterbox backend: `sudo apt-get install -y python3.11 python3.11-venv` (or use deadsnakes), then
`bash scripts/setup-chatterbox.sh`. It downloads ~3 GB of packages and, on first generation, 3.2 GB of weights.
Check `df -h` first.

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
python -m studio.importer /path/to/original/audio-or-video
python -m studio.cli prepare
# Review transcripts in the panel before training.
python -m studio.cli create --name "Speaker 1 rank 8"            # prints the experiment ID
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2   --module studio.cli train --experiment ID --epochs 1 --max-updates 20 --batch-size 4 --accumulation 2 --lr 0.00005
CUDA_VISIBLE_DEVICES=0 python -m studio.cli generate --experiment ID --checkpoint best --text 'Hello. This is my saved voice.'
python -m studio.cli export --experiment ID --checkpoint best
```

Start with batch 4 per GPU and accumulation 2 (effective global batch 16 on two GPUs). Raise the batch only after
measuring memory. Each card holds its own full model copy. Turning activation checkpointing off is faster but
uses more memory. Larger batches or more epochs do not automatically improve voice quality. Training uses all
selected GPUs; STT, subtitle checks and generation use one.

## Migration and backups

At rest, copy the entire `data` folder into this checkout (or restore the backup zips into it). Keep `experiments`, `datasets.json`, `migrations.json` and the legacy `models` folder. Do not copy a Windows `.venv` to Linux. Source code is public; audio, subtitle contents, voice checkpoints and Trash are ignored by Git.

```bash
watch -n 1 nvidia-smi
python -m unittest discover -s tests -v
```

The trainer is experimental. Two-process gradient synchronization was checked locally using CPU Gloo, and single-GPU training/resume/generation was tested on RTX 2070. Full two-3090 and large-v3 inference need a rental-machine smoke test. See [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html) for NCCL behavior.
