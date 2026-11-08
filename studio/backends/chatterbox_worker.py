"""Chatterbox inference worker. Runs ONLY inside .venv-chatterbox (torch 2.6.0, chatterbox-tts 0.1.7).

It must not import the `studio` package: the two environments have conflicting dependencies.
Usage: python chatterbox_worker.py request.json   (writes request.json's "result" path)

Request: {"mode": "check" | "generate", "device": "cuda", "result": "...json",
          "items": [{"text", "out", "reference" (wav path or null), "exaggeration", "cfg_weight",
                     "temperature", "repetition_penalty", "min_p", "top_p", "seed"}]}
This is zero-shot reference cloning with the pretrained weights. Nothing here trains or changes weights.
"""
import json
import sys
import time
import traceback

REPO_ID = 'ResembleAI/chatterbox'
REVISION = '5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18'  # Pinned main as of 2026-06-10 (repo archived).
FILES = ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']


def emit(result_path, value):
    with open(result_path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream)


def main():
    with open(sys.argv[1], encoding='utf-8') as stream:
        request = json.load(stream)
    result = {'ok': False, 'items': []}
    try:
        import importlib.metadata as metadata
        import torch
        result['versions'] = {'torch': torch.__version__, 'chatterbox-tts': metadata.version('chatterbox-tts'),
                              'model': REPO_ID, 'revision': REVISION}
        result['cuda'] = torch.cuda.is_available()
        if request['mode'] == 'check':
            import chatterbox.tts  # noqa: F401  Import check only; no download.
            result['ok'] = True
            return
        from huggingface_hub import snapshot_download
        from chatterbox.tts import ChatterboxTTS
        import torchaudio
        device = request.get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in .venv-chatterbox.')
        folder = snapshot_download(REPO_ID, revision=REVISION, allow_patterns=FILES)
        started = time.time()
        model = ChatterboxTTS.from_local(folder, device)
        result['load_seconds'] = round(time.time() - started, 1)
        result['sample_rate'] = model.sr
        for item in request['items']:
            if item.get('seed') is not None:
                torch.manual_seed(int(item['seed']))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(item['seed']))
            started = time.time()
            with torch.inference_mode():
                wav = model.generate(item['text'], audio_prompt_path=item.get('reference') or None,
                                     exaggeration=float(item.get('exaggeration', .5)),
                                     cfg_weight=float(item.get('cfg_weight', .5)),
                                     temperature=float(item.get('temperature', .8)),
                                     repetition_penalty=float(item.get('repetition_penalty', 1.2)),
                                     min_p=float(item.get('min_p', .05)), top_p=float(item.get('top_p', 1.0)))
            torchaudio.save(item['out'], wav.cpu(), model.sr)
            result['items'].append({'out': item['out'], 'seconds': round(wav.shape[-1] / model.sr, 2),
                                    'elapsed': round(time.time() - started, 1)})
            print(f"generated {item['out']}", flush=True)
        if torch.cuda.is_available():
            result['peak_vram_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        result['ok'] = True
    except Exception as exc:  # Reported to the studio, which shows it to the user.
        traceback.print_exc()
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        emit(request['result'], result)


if __name__ == '__main__':
    main()
