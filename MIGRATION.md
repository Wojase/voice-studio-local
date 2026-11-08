# Migration to datasets + experiment registry

The migration runs automatically when the panel or CLI starts. It never deletes or moves existing files.

1. `data/datasets.json` is created with a **Default dataset** and speaker **Speaker 1**.
2. Every recording in `data/samples/*.json` without a dataset gets `dataset_id: "default"`, `speaker: "Speaker 1"`
   and its sample rate. Nothing else in the file changes.
3. If `data/models/latest.pt` exists, it is **copied** into a new experiment **Legacy model (imported)**, along with
   `best.pt` (if present), `reference.wav` and `split.json`. Its architecture is recorded as the legacy adapter:
   rank 8, alpha 16 (scale 2.0), dropout 0, targets `q_proj` + `v_proj`. Base `KittenML/kitten-tts-2` @ `f2eee74f…`.
4. The legacy validation split is preserved as `kind: clip-legacy`: clips of one recording may be in both
   partitions, so its validation loss is optimistic. New experiments use a recording-level split.
5. The SHA-256 of the imported `latest.pt` is stored in `data/migrations.json`, so deleting the imported experiment
   later does not re-import it.

`data/models/` is left untouched as a fallback. You may archive it once you have confirmed the legacy experiment
generates correctly. Before upgrading, back up:

```bash
python -m studio.backup all
```

Rollback: stop the server, check out the previous commit, and delete `data/experiments`, `data/datasets.json` and
`data/migrations.json`. Your original `data/models` and recordings remain as they were.
