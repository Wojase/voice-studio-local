# Voice Studio

English voice-training studio with a browser panel: named datasets and speakers, an experiment registry with
per-experiment checkpoints, real LoRA fine-tuning of KittenTTS2 (single GPU, or multi-GPU on Linux), a pretrained
Chatterbox backend for zero-shot reference cloning, A/B comparison, and portable model downloads.

## Quick start

Windows: run `START.bat` (keeps the existing CUDA-enabled `.venv`). Open http://127.0.0.1:7860.

Linux / Vast.ai: see [RENTAL.md](RENTAL.md).

```bash
bash scripts/server.sh start      # status | logs | stop | restart
```

`stop` waits for a running job to finish its current update and save, and refuses to terminate an unrelated
program on port 7860. The application has no user authentication; on a rented server expose it only through your
controlled tunnel.

Optional Chatterbox backend (separate environment, about 3 GB of Python packages + 3.2 GB of model weights):

```bash
scripts\setup-chatterbox.bat          # Windows
bash scripts/setup-chatterbox.sh      # Linux (needs python3.11)
```

## Backup first

```bash
python -m studio.backup models     # checkpoints, experiments, reference clips, metadata, splits, configs, logs
python -m studio.backup dataset    # recordings, clips, subtitles, feature caches, converted MP3
python -m studio.backup all
```

Archives are written to `STUDIO_BACKUPS` (default `~/voice-studio-backups`, never inside the data folder), are
re-read and verified member by member after writing, and get a `.sha256` file (`sha256sum -c *.sha256`).

## Panel sections

1. **Datasets** – create named datasets and speakers; each recording belongs to one dataset + speaker. Upload
   staging with an optional SRT per recording (**Use SRT** or **SRT + check**), English STT preparation, Audio + SRT
   import, and clip review. Shows original recording count, enabled clips, usable duration, sample rates and
   subtitle-check status. Warnings flag very short or long clips, clipping, mostly-silent clips, implausible
   characters-per-second and STT disagreement. Bulk select/enable/disable/delete. **Deleting recordings does not
   remove their influence from an already trained checkpoint.**
2. **Models** – every experiment has an ID, name, backend, exact base model + revision, dataset + speaker,
   configuration, seed, timestamps, software versions, checkpoints, update count, completed epochs, validation
   metrics, status and a readable log. Create, duplicate configuration, rename, archive, delete (to Trash, by
   typing its name), import. Every checkpoint has a **Download** button.
3. **Train** – fixed settings (LoRA rank, alpha, dropout, target layers, seed, validation fraction) are chosen
   when the experiment is created and are never changed silently; duplicate to change them. Per-run settings:
   learning rate (number + presets), additional epochs, optional max updates, per-GPU batch, gradient
   accumulation, weight decay, gradient clipping, warmup, scheduler (constant/cosine/linear), activation
   checkpointing, checkpoint frequency + retention, validation frequency, early stopping and GPU selection.
   Shows the effective global batch and estimated updates per epoch.
4. **Monitor** – training and validation loss chart, BEST and LATEST checkpoints, updates, epochs, elapsed time,
   throughput, a remaining-time estimate once at least 3 updates are measured, per-GPU utilization and memory,
   the experiment log, and **Stop and save**.
5. **Generate & compare** – choose backend, fine-tuned experiment or pretrained base, checkpoint (BEST, LATEST or a
   numbered snapshot), reference recording, and only the controls the backend implements. A/B comparison speaks
   identical text; a fixed set of 10 English evaluation sentences is built in. Every output is stored with its
   backend, experiment, checkpoint, text and settings, and labelled **FINE-TUNED** or **PRETRAINED BASE**.
6. **Manage** – converted MP3s, Trash with restore (single and bulk) and permanent deletion (type `DELETE`).

## Backends

| Backend | Inference | Training |
|---|---|---|
| KittenTTS2 + LoRA (`KittenML/kitten-tts-2` @ `f2eee74f…`, `kittenml==0.9.4`) | Fine-tuned adapters, built-in base voices, zero-shot clone from a clip | **Yes**: cross-entropy on target audio tokens, AdamW updates of LoRA weights, checkpoint save/resume/reload |
| Chatterbox (`ResembleAI/chatterbox` @ `5bb1f6ee…`, `chatterbox-tts==0.1.7`, original English 0.5B model) | Built-in voice or zero-shot reference cloning (reference ≥ 5 s) | **Unavailable** (see below) |

**Chatterbox.** Resemble AI archived Chatterbox as retired research in August 2026 (MIT license, no further
updates). The original English model was chosen over Turbo/Nano because it exposes exaggeration and CFG
controls for English speech and fits an 8 GB card (3.4 GB peak measured on an RTX 2070). Turbo would also
download an unused 1 GB decoder. It runs in its own environment (`.venv-chatterbox`, torch 2.6.0), separate from
KittenTTS2's environment, through a worker process. Every Chatterbox output carries Resemble AI's Perth watermark.
Reference cloning changes no weights and is **not** training.

Chatterbox fine-tuning is shown as unavailable: the official code ships no training path and none is
implemented or verified here. Finishing it requires a T3 training loop (cross-entropy on S3Tokenizer speech
tokens, conditioned on voice-encoder embeddings and text), LoRA or 8-bit optimizer states to fit one card,
checkpoint save/resume, and a test that reloaded T3 weights generate correctly.

No backend or preset guarantees a particular quality level, such as that of commercial services. Lowest validation loss is only
a selection heuristic. Listen to BEST, LATEST and the base voices on the same sentences.

## Experiments, checkpoints and validation

```
data/experiments/<id>/experiment.json   identity, base, architecture, config, status, checkpoint index, runs
data/experiments/<id>/checkpoints/      latest.pt, best.pt, update-000025.pt …
data/experiments/<id>/metrics.jsonl     loss per update, validation loss
data/experiments/<id>/train.log         readable log
data/experiments/<id>/split.json        persisted validation membership
data/experiments/<id>/reference.wav     decoder reference clip
```

`latest.pt` is written every *save_every* updates, after each epoch and on stop. A numbered snapshot is kept at
the same times (newest *keep_checkpoints* retained); `best.pt` is the lowest validation loss. Resume loads
`latest.pt` (weights + optimizer). An interrupted epoch restarts. A checkpoint is loaded only if its backend,
base revision, adapter architecture, tensor names and shapes all match the experiment.

New experiments split validation **by original recording**, so adjacent clips of one recording never land in both
partitions. The smallest recordings are held out first until the validation fraction is reached; one recording
always stays in training. Membership is persisted, and recordings added later are assigned deterministically
without moving existing ones. With one recording there is no validation set (and no BEST). Validation sets under
5 clips or 60 seconds show a warning. The imported legacy experiment keeps its original clip-level split, marked
as such.

Feature caches (codec tokens + speaker embedding per clip) depend only on the base model and are shared by
experiments on that base.

## Downloads (export) and import

Each checkpoint downloads as a zip: `adapter.safetensors` (LoRA weights + speaker embedding), `model.json`
(backend, base ID + revision, exact architecture, sample rate 24 kHz, language, preprocessing, training config,
software versions), `reference.wav` (**a short clip of the speaker's voice**; a second button omits it),
`infer.py`, `README.txt` and `SHA256SUMS.txt`. It is an **adapter that requires separately obtained base
weights**; KittenML base weights are not redistributed. Optimizer state, recordings and other experiments are
never included. Archives are built on disk and streamed.

Import accepts only that flat allow-list of file names (no paths, `..`, drive letters, symlinks or extra files),
verifies every checksum, the base model and the tensor shapes, and loads weights from safetensors, never pickle.
Imported models are inference-only.

## GPUs

Training with two or more GPUs (Linux) runs one process per card via `torchrun`; adapter gradients are averaged
with one NCCL all-reduce after each update and validation is split across cards. Each card holds a full model
copy in **its own** memory: two 24 GB cards are not one 48 GB pool. STT, subtitle checks, feature encoding and
generation use one GPU. Windows trains on one GPU. GPU utilization depends on batch size and data loading;
constant 100% is not expected.

## Tools and tests

```bash
python -m studio.cli experiments
python -m studio.cli train --experiment ID --epochs 1 --max-updates 10
python -m studio.cli generate --experiment ID --checkpoint best --text "Hello."
python -m studio.cli export --experiment ID --checkpoint best
python -m unittest discover -s tests -v
STUDIO_DATA=/path/to/a/copy python -X utf8 tests/check_experiments_gpu.py   # real-GPU end-to-end check
```

See [MIGRATION.md](MIGRATION.md) for how existing data is upgraded.

Powered by KittenML and Resemble AI Chatterbox. Application code is MIT. The KittenTTS2 base model and derived
adapters are subject to the [KittenML model license](https://kittenml.com/license/). Chatterbox weights are MIT.
