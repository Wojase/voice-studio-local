"""Manual regression: real cached voice fragment must have finite loss and gradients."""
import torch
from studio.engine import load_model, adapters, gpu
from studio.storage import DATA, clips, read
from kittenml.kittentts2.prompt import build_generation_prompt
from kittenml.kittentts2.tokens import TokenMap

gpu()
tts = load_model(training=True)
model = tts.model
params = adapters(model)
dtype = next(model.parameters()).dtype
assert dtype == torch.bfloat16
row = read(clips()[0])
feature = torch.load(DATA / 'features' / (row['id'] + '.pt'), weights_only=True)
tm = TokenMap.from_config(tts.config)
prompt = build_generation_prompt(tm, tts.tokenizer.encode(row['text'], add_special_tokens=False))
target = [tm.audio_id_base+t for t in feature['tokens']] + [tm.speech_end_id]
ids = torch.tensor([prompt + target[:-1]], device='cuda')
with torch.autocast('cuda', dtype=dtype):
    inputs = model.get_input_embeddings()(ids)
    projected = model.spk_proj(feature['embedding'].to('cuda', dtype=dtype)[None]).unsqueeze(1)
    inputs = torch.cat([projected, inputs[:, 1:]], dim=1)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    hidden = model.model(inputs_embeds=inputs, use_cache=False).last_hidden_state
    logits = model.lm_head(hidden[:, len(prompt)-1:])
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),torch.tensor(target,device='cuda'))
print('loss',loss.item(),flush=True)
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
print('PASS: finite loss and gradients',flush=True)
