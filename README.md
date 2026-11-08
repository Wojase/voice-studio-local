# Voice Studio

English voice-training studio with a browser panel, saved LoRA checkpoints, multi-GPU training, video conversion, timed subtitles and data management.

## Quick start

Windows: run `START.bat`. Keep the existing CUDA-enabled `.venv`.

Linux / Vast.ai: see [RENTAL.md](RENTAL.md). Server controls:

```bash
bash scripts/server.sh start
bash scripts/server.sh status
bash scripts/server.sh logs
bash scripts/server.sh stop
bash scripts/server.sh restart
```

Stop waits for the current job to finish its update and save before shutting down the panel. It refuses to terminate unrelated programs using port 7860. The rental profile binds `0.0.0.0:7860`, uses two GPUs for training and Whisper large-v3 for English STT. Expose it through your controlled Vast.ai tunnel; the application itself has no user authentication.

## Workflow

1. Upload clean recordings of one speaker. MP4/MOV/MKV creates MP3 for download and WAV for training.
2. Either prepare clips with English STT, or select a recording and import its matching UTF-8 SRT.
3. Review text and audio, enable good clips, then train.
4. Generate new sentences from the saved model. Later training resumes weights and optimizer.

### Audio + SRT and SRT + check

SRT timestamps must be ordered, non-overlapping, inside the selected audio, and individual cues must not exceed 12 seconds. Subtitle text is paired with precisely those audio ranges. Nearby short cues are grouped into roughly 6-12 second training clips. Long silent gaps are not combined.

Importing SRT replaces the selected recording's old clips and feature caches; they are moved to Trash. New subtitle clips start disabled, so unchecked text is not silently used for training. Review and enable them manually, or run **SRT + check with STT**. It transcribes each grouped fragment independently and compares normalized words with the supplied text. Word error above 25% disables the fragment and marks it for review. This threshold is a heuristic, not a confidence guarantee. Check results show what STT heard; your SRT text is never overwritten. Manual review can override automatic exclusion. Checks do not perform forced alignment or prove timestamps or speaker identity. Incorrect transcription, clip boundaries, silence and proper names may produce false flags or false matches.

Preparation skips recordings with attached SRT. To start over with automatic STT, use **Delete clips / redo STT** on that recording, then Prepare. Existing old transcripts are not automatically retranscribed when switching STT models.

### Manage and delete

Delete individual recordings, clips, outputs and converted MP3s. Deleting a recording also moves its clips, SRT metadata and feature caches to Trash. Reset just the trained model, just the dataset, or both. Existing trained weights retain information from deleted recordings until you reset and retrain the model.

Trash is recoverable and still consumes disk space. Restore refuses to overwrite newer files. Permanently removing a Trash entry requires typing `DELETE` in the panel and cannot be undone. Management operations are blocked while a job uses the dataset.

## Models and quality

The TTS base remains pinned KittenTTS2 0.9.4 with custom experimental Q/V LoRA training. A stronger STT model improves transcription, not the capacity or naturalness of the TTS base. ElevenLabs-level quality is not guaranteed. Evaluate with clean held-out sentences and listening tests.

Local default: `openai/whisper-small.en`. Rental default: `openai/whisper-large-v3`. Override with `STUDIO_STT_MODEL`; recognition is explicitly English. Models download on first use and run on the server. Uploaded audio and subtitles are not sent to a transcription service.

Training uses BF16 because FP16 overflowed on a real sample. Per-GPU batch and gradient accumulation are separate settings. Linux multi-GPU training synchronizes adapter gradients through NCCL. Preparation/checking and a single generated recording run on one GPU. Two 24 GB GPUs are separate memory pools. Constant 100% utilization is not guaranteed.

## Data and checkpoints

Everything private is under ignored `data/` (or `STUDIO_DATA`). Keep `samples`, `clips`, `features`, `subtitles` and the complete `models` folder when migrating machines. `models/latest.pt` resumes training; `best.pt` retains the lowest observed validation loss. Generation currently uses `latest.pt`. The base-model cache is also required, or will download again.

Checkpoints save every 25 updates, after epochs, and on graceful stop. Resume preserves weights and optimizer, but starts an interrupted epoch again. Validation membership persists in `models/split.json`; it is a clip-level split, so clips from the same recording may appear in both partitions. Validation scores are not independent recording-level quality measurements.

## Tools and tests

```bash
python -m studio.media /path/to/videos --output /path/to/mp3
python -m studio.importer /path/to/recordings
python -m studio.cli prepare
python -m unittest discover -s tests -v
```

Windows conversion: drag a video or folder onto `MP4_NA_MP3.bat`. FFmpeg must be in PATH. Conversion never overwrites originals.

Tests cover real FFmpeg conversion, upload, gradient learning, data sharding, SRT validation and audio cuts, STT-check logic using a mocked recognizer, and delete/restore/purge behavior. The real user-provided SRT was parsed locally without publishing it. Actual English large-v3 inference and full two-RTX-3090 throughput still require verification on the rental machine.

Powered by KittenML. Application code is MIT; base model and derived adapters retain the [KittenML model license](https://kittenml.com/license/).
