const $ = id => document.getElementById(id);
let state = {}, meta = null, detail = null, limit = 30, uploading = false, uploadQueue = [];
let sigs = {};
const selectedClips = new Set(), selectedRecs = new Set(), selectedTrash = new Set();
const variants = {
  A: {enabled: true, backend: 'kittentts2-lora', kind: 'finetuned', experiment_id: '', checkpoint: '', voice: '', reference_clip: '', controls: {}},
  B: {enabled: false, backend: 'kittentts2-lora', kind: 'base', experiment_id: '', checkpoint: '', voice: '', reference_clip: '', controls: {}},
};

function notice(s) { $('toast').textContent = s; $('toast').hidden = false; clearTimeout(notice.t); notice.t = setTimeout(() => $('toast').hidden = true, 9000) }
function el(tag, text, cls) { const e = document.createElement(tag); if (text !== undefined && text !== null) e.textContent = text; if (cls) e.className = cls; return e }
async function api(path, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'X-Studio': 'local', ...(body instanceof FormData ? {} : {'Content-Type': 'application/json'})}, body: body instanceof FormData ? body : JSON.stringify(body)};
  const r = await fetch('api/' + path, options); const d = await r.json(); if (!r.ok) throw Error(d.error || 'Request failed'); return d;
}
function button(text, fn, cls = '') { const b = el('button', text, cls); b.onclick = async () => { try { b.disabled = true; await fn() } catch (e) { notice(e.message) } finally { b.disabled = false } }; return b }
function audio(kind, name) { const a = el('audio'); a.controls = true; a.preload = 'none'; a.src = 'audio/' + kind + '/' + name; return a }
function link(href, text, download) { const a = el('a', text, 'button'); a.href = href; if (download) a.download = download; return a }
function option(value, text) { const o = el('option', text); o.value = value; return o }
function fill(select, items, keep = true) { const v = select.value; select.replaceChildren(...items.map(([value, text]) => option(value, text))); if (keep && items.some(i => String(i[0]) === v)) select.value = v }
function fmtTime(s) { if (s == null) return '—'; s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60); return h ? `${h} h ${m} min` : m ? `${m} min ${s % 60} s` : `${s} s` }
function stat(label, value) { const d = el('div', undefined, 'stat'); d.append(el('span', label), el('strong', value)); return d }
function invalidate() { sigs = {} }
async function job(action, args = {}) { await api('jobs', {action, ...args}); await refresh() }
async function remove(kind, key = '', keys = null) {
  const what = keys ? `${keys.length} selected item(s)` : (key || kind);
  if (!confirm(`Move ${what} to Trash? Trained checkpoints keep what they learned; delete experiments separately.`)) return;
  await api('manage', keys ? {kind, keys} : {kind, key}); invalidate(); await refresh(); notice('Moved to Trash. Restore it in Manage.');
}
const dataset = () => (state.datasets || []).find(d => d.id === $('datasetSelect').value) || (state.datasets || [])[0];
const experiments = () => state.experiments || [];
const experiment = id => experiments().find(e => e.id === id);
const sampleOf = c => c.sample_id || c.id.split('_')[0];

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button,section').forEach(e => e.classList.remove('active'));
  b.classList.add('active'); $(b.dataset.tab).classList.add('active'); if (b.dataset.tab === 'monitor') loadDetail();
});
document.querySelectorAll('[data-delete]').forEach(b => b.onclick = () => remove(b.dataset.delete).catch(e => notice(e.message)));
$('stop').onclick = () => api('stop', {}).then(d => notice(d.message)).catch(e => notice(e.message));
$('prepare').onclick = () => job('prepare').catch(e => notice(e.message));

// ---------- datasets ----------
$('datasetSelect').onchange = () => { invalidate(); renderDatasets(); refreshViews() };
$('createDataset').onclick = () => api('datasets', {action: 'create', name: $('newDataset').value, speaker: $('newDatasetSpeaker').value})
  .then(async d => { await refresh(); $('datasetSelect').value = d.id; invalidate(); renderDatasets(); refreshViews(); notice('Dataset created.') }).catch(e => notice(e.message));
$('saveDataset').onclick = () => api('datasets', {action: 'update', id: dataset().id, name: $('renameDataset').value || null, add_speaker: $('addSpeaker').value || null})
  .then(async () => { $('addSpeaker').value = ''; invalidate(); await refresh(); notice('Dataset saved.') }).catch(e => notice(e.message));
$('selectAllRec').onclick = () => { recordingsShown().forEach(s => selectedRecs.add(s.id)); sigs.samples = ''; refreshViews() };
$('assignRec').onclick = () => { if (!selectedRecs.size) return notice('Select recordings first.');
  api('datasets', {action: 'assign', sample_ids: [...selectedRecs], dataset_id: dataset().id, speaker: $('speakerSelect').value})
    .then(async () => { selectedRecs.clear(); invalidate(); await refresh(); notice('Recordings moved.') }).catch(e => notice(e.message)) };
$('deleteRec').onclick = () => { if (!selectedRecs.size) return notice('Select recordings first.'); remove('samples', '', [...selectedRecs]).then(() => selectedRecs.clear()).catch(e => notice(e.message)) };
function recordingsShown() { const d = dataset(); return (state.samples || []).filter(s => !d || (s.dataset_id || 'default') === d.id) }
function clipsInDataset() { const ids = new Set(recordingsShown().map(s => s.id)); return (state.clips || []).filter(c => ids.has(sampleOf(c))) }

function renderDatasets() {
  const list = state.datasets || [];
  fill($('datasetSelect'), list.map(d => [d.id, d.name]));
  const d = dataset(); if (!d) return;
  fill($('speakerSelect'), d.speakers.map(s => [s, s]));
  const checks = Object.entries(d.subtitle_checks || {}).map(([k, v]) => `${k}: ${v}`).join(' · ') || 'no SRT clips';
  $('datasetStats').replaceChildren(stat('Original recordings', d.recordings), stat('Enabled clips', `${d.enabled_clips} / ${d.clips}`),
    stat('Usable duration', fmtTime(d.usable_seconds)), stat('Sample rates', d.sample_rates.map(r => r + ' Hz').join(', ') || '—'),
    stat('Subtitle checks', checks), stat('Enabled clips with warnings', d.warnings));
  $('renameDataset').placeholder = d.name;
}

// ---------- upload staging ----------
function stageFiles(files) {
  if (uploading) return notice('Wait for the current upload to finish.');
  for (const file of files) {
    if (!/\.(wav|mp3|mp4|m4v|mov|mkv|ogg|flac)$/i.test(file.name)) { notice('Choose audio/video first. Attach SRT in the staged recording row.'); continue }
    uploadQueue.push({file, srt: null, mode: 'use'});
  }
  $('files').value = ''; renderUploadQueue();
}
function renderUploadQueue() {
  $('uploadStage').hidden = !uploadQueue.length; $('uploadQueue').replaceChildren();
  for (const item of uploadQueue) {
    const row = el('div', undefined, 'clip'); row.append(el('strong', item.file.name));
    const label = el('label', 'Optional subtitle file (.srt)'), input = el('input'); input.type = 'file'; input.accept = '.srt';
    input.onchange = () => { item.srt = input.files[0] || null; renderUploadQueue() }; label.append(input);
    row.append(label, el('p', item.srt ? item.srt.name : 'No subtitles attached.', 'small'));
    if (item.srt) {
      const modeLabel = el('label', 'Subtitle processing'), select = el('select');
      select.append(option('use', 'Use SRT — trust text, no STT'), option('check', 'SRT + check — verify using English STT'));
      select.value = item.mode; select.onchange = () => item.mode = select.value; modeLabel.append(select);
      row.append(modeLabel, button('Remove SRT', () => { item.srt = null; renderUploadQueue() }));
    }
    row.append(button('Remove recording', () => { uploadQueue = uploadQueue.filter(q => q !== item); renderUploadQueue() }, 'danger'));
    $('uploadQueue').append(row);
  }
  $('confirmUpload').textContent = `Confirm upload (${uploadQueue.length}) to ${dataset() ? dataset().name : 'dataset'} · ${$('speakerSelect').value}`;
  document.querySelectorAll('#uploadStage button,#uploadStage input,#uploadStage select').forEach(e => e.disabled = uploading);
  $('confirmUpload').disabled = uploading || state.busy || !uploadQueue.length;
}
async function submitUploads() {
  if (uploading || state.busy) return notice('Wait for the current operation.');
  uploading = true; renderUploadQueue(); const errors = [], checkIds = [];
  for (const [i, item] of [...uploadQueue].entries()) {
    try {
      notice(`Uploading ${i + 1}: ${item.file.name}`); const form = new FormData(); form.append('file', item.file);
      form.append('dataset_id', dataset().id); form.append('speaker', $('speakerSelect').value);
      if (item.srt) { form.append('subtitles', item.srt); form.append('subtitle_mode', item.mode) }
      const result = await api('upload', form); if (result.check_requested) checkIds.push(result.id);
      uploadQueue = uploadQueue.filter(q => q !== item);
    } catch (e) { errors.push(`${item.file.name}: ${e.message}`) }
  }
  uploading = false; renderUploadQueue(); invalidate(); await refresh();
  if (checkIds.length) { try { await job('check_subtitles', {sample_ids: [...new Set(checkIds)]}) } catch (e) { errors.push('Files uploaded, but the STT check could not start: ' + e.message + '. Run it again from Audio + SRT.') } }
  notice(errors.length ? errors.join('\n') : checkIds.length ? 'Uploaded. SRT checking started.' : 'Upload complete. Recordings without SRT still need preparation.');
}
$('confirmUpload').onclick = () => submitUploads().catch(e => notice(e.message));
$('cancelUpload').onclick = () => { if (!uploading) { uploadQueue = []; renderUploadQueue() } };
$('files').onchange = e => stageFiles([...e.target.files]);
$('drop').ondragover = e => { e.preventDefault(); $('drop').classList.add('drag') };
$('drop').ondragleave = () => $('drop').classList.remove('drag');
$('drop').ondrop = e => { e.preventDefault(); $('drop').classList.remove('drag'); stageFiles([...e.dataTransfer.files]) };
$('importSrt').onclick = async () => {
  try {
    const f = $('srtFile').files[0], s = $('srtAudio').value; if (!f || !s) return notice('Choose a recording and SRT first.');
    if (!confirm('Replace this recording’s clips? Existing clips go to Trash.')) return;
    const form = new FormData(); form.append('file', f); const d = await api('subtitles/' + s, form); invalidate(); await refresh();
    notice(`Imported ${d.cues} cues as ${d.clips} clips. Review them or run SRT + check.`);
  } catch (e) { notice(e.message) }
};
$('checkSrt').onclick = () => job('check_subtitles', {sample_id: $('srtAudio').value}).catch(e => notice(e.message));

// ---------- recordings + clips ----------
function renderSamples() {
  const shown = recordingsShown(); $('sampleList').replaceChildren(); const selected = $('srtAudio').value; $('srtAudio').replaceChildren();
  if (!shown.length) $('sampleList').append(el('p', 'No recordings in this dataset yet.', 'empty'));
  for (const s of shown) {
    const n = (state.clips || []).filter(c => sampleOf(c) === s.id), on = n.filter(c => c.enabled !== false).length;
    const row = el('div', undefined, 'clip'), head = el('label', undefined, 'inline'), box = el('input'); box.type = 'checkbox'; box.checked = selectedRecs.has(s.id);
    box.onchange = () => box.checked ? selectedRecs.add(s.id) : selectedRecs.delete(s.id); head.append(box, el('strong', ' ' + s.name));
    const warn = s.sample_rate && s.sample_rate < 16000 ? ' · low sample rate' : '';
    row.append(head, el('p', `${(s.seconds / 60).toFixed(1)} min · ${s.sample_rate ? s.sample_rate + ' Hz' : 'rate unknown'} · speaker: ${s.speaker || '—'} · clips ${on}/${n.length} enabled${warn}`, 'small'),
      audio('samples', s.id + '.wav'), button('Delete clips / redo STT', () => remove('sample_clips', s.id), 'danger'), button('Delete recording + its clips', () => remove('samples', s.id), 'danger'));
    $('sampleList').append(row); $('srtAudio').append(option(s.id, s.name));
  }
  if (shown.some(s => s.id === selected)) $('srtAudio').value = selected;
  $('subtitleList').replaceChildren(...(state.subtitles || []).map(s => el('p', `${s.filename}: ${s.cues} cues → ${s.clips} clips`, 'small')));
}
function filteredClips() {
  const f = $('clipFilter').value;
  return clipsInDataset().filter(c => f === 'all' || (f === 'warnings' && c.warnings.length) || (f === 'disabled' && c.enabled === false) || (f === 'review' && c.review === 'needs_review'));
}
$('clipFilter').onchange = () => { limit = 30; renderClips() };
$('selectAllClips').onclick = () => { filteredClips().forEach(c => selectedClips.add(c.id)); renderClips() };
$('clearClips').onclick = () => { selectedClips.clear(); renderClips() };
async function bulk(enabled) { if (!selectedClips.size) return notice('Select clips first.'); const d = await api('clips/bulk', {ids: [...selectedClips], enabled}); invalidate(); await refresh(); notice(`${d.changed} clips ${enabled ? 'enabled' : 'disabled'}.`) }
$('enableClips').onclick = () => bulk(true).catch(e => notice(e.message));
$('disableClips').onclick = () => bulk(false).catch(e => notice(e.message));
$('deleteClips').onclick = () => { if (!selectedClips.size) return notice('Select clips first.'); remove('clips', '', [...selectedClips]).then(() => selectedClips.clear()).catch(e => notice(e.message)) };
$('more').onclick = () => { limit += 30; renderClips() };
function renderClips() {
  const list = filteredClips(); $('clips').replaceChildren();
  if (!list.length) $('clips').append(el('p', 'No clips match.', 'empty'));
  for (const c of list.slice(0, limit)) {
    const box = el('div', undefined, 'clip'), row = el('div', undefined, 'row'), pick = el('input'); pick.type = 'checkbox'; pick.checked = selectedClips.has(c.id);
    pick.setAttribute('aria-label', 'Select clip'); pick.onchange = () => pick.checked ? selectedClips.add(c.id) : selectedClips.delete(c.id);
    const label = el('span', ` ${(c.seconds || 0).toFixed(1)} s · ${c.source || 'STT'} · ${c.review || 'manual review'} · ${c.enabled === false ? 'disabled' : 'enabled'}`, c.review === 'needs_review' ? 'review' : 'small');
    const left = el('div'); left.append(pick, label); row.append(left, audio('clips', c.id + '.wav'));
    const ta = el('textarea'); ta.value = c.text; ta.rows = 2; ta.setAttribute('aria-label', 'Clip transcript');
    const check = el('input'); check.type = 'checkbox'; check.checked = c.enabled !== false; const inc = el('label', ' Include in training', 'inline'); inc.prepend(check);
    box.append(row, ta, inc);
    if (c.warnings.length) box.append(el('p', '⚠ ' + c.warnings.join(' · '), 'review'));
    if (c.check) box.append(el('p', `STT heard: ${c.check.recognized} · word error ${(c.check.word_error * 100).toFixed(0)}%`, 'small'));
    const tools = el('div', undefined, 'tools');
    tools.append(button('Save transcript / review', async () => { await api('clips/' + c.id, {text: ta.value, enabled: check.checked}); invalidate(); await refresh(); notice('Clip saved.') }),
      button('Delete clip', () => remove('clips', c.id), 'danger'));
    box.append(tools); $('clips').append(box);
  }
  $('more').hidden = list.length <= limit;
}

// ---------- experiments ----------
function fieldInput(name, spec, value) {
  const wrap = el('div'), label = el('label', name.replace(/_/g, ' ')); label.title = spec.help; let input;
  if (spec.type === 'bool') { input = el('input'); input.type = 'checkbox'; input.checked = value ?? spec.default }
  else if (name === 'scheduler') { input = el('select'); meta.training.schedulers.forEach(s => input.append(option(s, s))); input.value = value ?? spec.default }
  else { input = el('input'); input.type = 'number'; if (spec.min != null) input.min = spec.min; if (spec.max != null) input.max = spec.max; input.step = spec.type === 'int' ? 1 : 'any'; input.value = value ?? spec.default }
  input.dataset.name = name; input.dataset.kind = spec.type; wrap.append(label, input, el('p', spec.help, 'help')); return wrap;
}
function readFields(container) {
  const out = {};
  container.querySelectorAll('[data-name]').forEach(i => out[i.dataset.name] = i.type === 'checkbox' ? i.checked : i.dataset.kind === 'int' ? parseInt(i.value) : i.dataset.kind === 'float' ? parseFloat(i.value) : i.value);
  return out;
}
function renderFixedFields() {
  const fixed = meta.training.fixed, box = $('fixedFields'); box.replaceChildren();
  for (const [name, spec] of Object.entries(fixed)) {
    if (name === 'targets') {
      const wrap = el('div', undefined, 'wide'); wrap.append(el('label', 'LoRA target layers'));
      for (const t of meta.training.targets) { const l = el('label', ' ' + t, 'inline'), c = el('input'); c.type = 'checkbox'; c.value = t; c.className = 'target'; c.checked = spec.default.includes(t); l.prepend(c); wrap.append(l) }
      wrap.append(el('p', spec.help + ' More targets = more trainable weights and memory.', 'help')); box.append(wrap);
    } else box.append(fieldInput(name, spec));
  }
}
$('createExperiment').onclick = () => {
  const cfg = readFields($('fixedFields')); cfg.targets = [...document.querySelectorAll('#fixedFields .target:checked')].map(c => c.value);
  api('experiments', {name: $('expName').value, dataset_id: $('expDataset').value, speaker: $('expSpeaker').value, config: cfg})
    .then(async e => { await refresh(); $('trainExperiment').value = e.id; renderTraining(); notice('Experiment created. Configure the run in Train.') }).catch(e => notice(e.message));
};
$('expDataset').onchange = () => { const d = (state.datasets || []).find(x => x.id === $('expDataset').value); fill($('expSpeaker'), (d ? d.speakers : []).map(s => [s, s])) };
$('showArchived').onchange = () => { sigs.exp = ''; refreshViews() };
$('importModel').onclick = async () => {
  try { const f = $('importFile').files[0]; if (!f) return notice('Choose a .zip export first.'); const form = new FormData(); form.append('file', f);
    const e = await api('import', form); await refresh(); notice(`Imported as “${e.name}” (inference only).`) } catch (e) { notice(e.message) }
};
function checkpointRows(e) {
  return Object.entries(e.checkpoints || {}).sort((a, b) => b[1].update - a[1].update || a[0].localeCompare(b[0]));
}
function renderExperiments() {
  const box = $('experimentList'); box.replaceChildren();
  const list = experiments().filter(e => $('showArchived').checked || !e.archived);
  if (!list.length) box.append(el('p', 'No experiments yet. Create one below.', 'empty'));
  for (const e of list) {
    const card = el('div', undefined, 'card'), title = el('div', undefined, 'row');
    const badges = [e.status, e.legacy && 'legacy', e.imported && 'imported · inference only', e.archived && 'archived'].filter(Boolean);
    title.append(el('h3', e.name), el('span', badges.join(' · '), 'badge'));
    const d = (state.datasets || []).find(x => x.id === e.dataset_id);
    card.append(title, el('p', `ID ${e.id} · ${e.backend} · base ${e.base.id}@${e.base.revision.slice(0, 10)} · dataset ${d ? d.name : e.dataset_id} · speaker ${e.speaker}`, 'small'),
      el('p', `LoRA rank ${e.architecture.rank}, alpha ${e.architecture.alpha}, dropout ${e.architecture.dropout}, targets ${e.architecture.targets.join(', ')} · seed ${e.config.seed} · ${e.updates} updates · ${e.epochs} completed epochs · best validation ${e.best_validation_loss != null ? e.best_validation_loss.toFixed(4) : '—'}`, 'small'));
    const table = el('div', undefined, 'checkpoints');
    for (const [name, info] of checkpointRows(e)) {
      const row = el('div', undefined, 'tools');
      const tag = name === 'best.pt' ? 'BEST' : name === 'latest.pt' ? 'LATEST' : 'snapshot';
      row.append(el('span', `${tag} · ${name} · update ${info.update} · epochs ${info.epochs}${info.validation_loss != null ? ' · val ' + info.validation_loss.toFixed(4) : ''}`, 'small grow'),
        link(`api/experiments/${e.id}/export?checkpoint=${encodeURIComponent(name)}`, 'Download', ''),
        link(`api/experiments/${e.id}/export?checkpoint=${encodeURIComponent(name)}&reference=0`, 'Download without reference clip', ''));
      table.append(row);
    }
    if (!checkpointRows(e).length) table.append(el('p', 'No checkpoints yet.', 'small'));
    card.append(table, el('p', 'Downloads are LoRA adapters: they need the KittenML base weights (not included). The default download includes this experiment’s short reference clip of the speaker.', 'help'));
    const tools = el('div', undefined, 'tools');
    tools.append(button('Train / resume', async () => { document.querySelector('[data-tab=training]').click(); $('trainExperiment').value = e.id; renderTraining() }),
      button('Monitor', async () => { document.querySelector('[data-tab=monitor]').click(); $('monitorExperiment').value = e.id; loadDetail() }),
      button('Rename', async () => { const n = prompt('New name', e.name); if (n) { await api('experiments/' + e.id, {action: 'rename', name: n}); await refresh() } }),
      button('Duplicate configuration', async () => { const x = await api('experiments', {action: 'duplicate', id: e.id}); await refresh(); notice(`Created “${x.name}” with fresh weights.`) }),
      button(e.archived ? 'Unarchive' : 'Archive', async () => { await api('experiments/' + e.id, {action: e.archived ? 'unarchive' : 'archive'}); await refresh() }),
      button('Delete', async () => { const n = prompt(`Moves this experiment (all checkpoints) to Trash. Type its exact name to confirm:\n${e.name}`); if (n === null) return; await api('experiments/' + e.id, {action: 'delete', confirm: n}); await refresh(); notice('Experiment moved to Trash.') }, 'danger'));
    card.append(tools); box.append(card);
  }
}

// ---------- training ----------
function renderTraining() {
  const e = experiment($('trainExperiment').value); if (!e || !meta) return;
  const fixed = meta.training.fixed;
  $('fixedSummary').textContent = Object.keys(fixed).map(k => `${k.replace(/_/g, ' ')}: ${Array.isArray(e.config[k]) ? e.config[k].join(', ') : e.config[k]}`).join(' · ');
  $('trainInfo').textContent = e.imported ? 'Imported experiments are inference-only (no optimizer state). Duplicate the configuration to train.'
    : e.latest ? `Resumes from latest.pt at update ${e.updates} (${e.epochs} completed epochs). Settings below apply to this run.` : 'New experiment: training starts from freshly initialized adapter weights.';
  const box = $('runFields'); box.replaceChildren();
  for (const [name, spec] of Object.entries(meta.training.resumable)) { if (name !== 'gpus') box.append(fieldInput(name, spec, (e.config || {})[name])) }
  const gpus = $('gpuChoice'); gpus.replaceChildren(el('label', 'GPUs for this run'));
  for (const g of meta.gpus) {
    const l = el('label', ` GPU ${g.index}: ${g.name} (${(g.total / 1024).toFixed(0)} GB)`, 'inline'), c = el('input'); c.type = 'checkbox'; c.value = g.index; c.className = 'gpu-pick';
    c.checked = (e.config.gpus || [0]).includes(g.index); c.onchange = estimate; l.prepend(c); gpus.append(l);
  }
  gpus.append(el('p', meta.multi_gpu ? 'Two or more GPUs: one training process per card, gradients averaged after each update (NCCL). Batch size is per card; each card must fit the model on its own.' : 'This system (Windows) trains on one GPU. Multi-GPU training requires Linux with NCCL.', 'help'));
  box.querySelectorAll('input,select').forEach(i => i.oninput = estimate); estimate();
  $('train').disabled = state.busy || e.imported;
}
function estimate() {
  const e = experiment($('trainExperiment').value); if (!e) return;
  const run = readFields($('runFields')), n = Math.max(1, document.querySelectorAll('.gpu-pick:checked').length);
  const d = (state.datasets || []).find(x => x.id === e.dataset_id);
  const ids = new Set((state.samples || []).filter(s => s.dataset_id === e.dataset_id && s.speaker === e.speaker).map(s => s.id));
  const enabled = (state.clips || []).filter(c => c.enabled !== false && ids.has(sampleOf(c))).length;
  const train = Math.max(1, Math.round(enabled * (1 - e.config.validation_fraction)));
  const effective = run.batch_size * run.accumulation * n, perEpoch = Math.ceil(train / effective);
  let total = perEpoch * run.epochs; if (run.max_updates) total = Math.min(total, run.max_updates);
  $('estimate').replaceChildren(stat('Effective global batch', `${effective} clips (${run.batch_size} × ${run.accumulation} × ${n} GPU)`),
    stat('≈ training clips', `${train} of ${enabled} enabled${d ? '' : ''}`), stat('≈ updates per epoch', perEpoch), stat('≈ updates this run', total));
}
$('lrPreset').onchange = () => { const v = $('lrPreset').value; const i = $('runFields').querySelector('[data-name=lr]'); if (v && i) i.value = v };
$('trainExperiment').onchange = renderTraining;
$('train').onclick = () => {
  const run = readFields($('runFields')); run.gpus = [...document.querySelectorAll('.gpu-pick:checked')].map(c => parseInt(c.value));
  job('train', {experiment_id: $('trainExperiment').value, run}).then(() => { document.querySelector('[data-tab=monitor]').click(); $('monitorExperiment').value = $('trainExperiment').value; loadDetail() }).catch(e => notice(e.message));
};

// ---------- monitoring ----------
$('monitorExperiment').onchange = loadDetail;
async function loadDetail() {
  const id = $('monitorExperiment').value; if (!id) return;
  try { detail = await api('experiments/' + id); renderMonitor() } catch (e) { notice(e.message) }
}
function renderMonitor() {
  if (!detail) return; const e = detail.experiment, j = state.job, live = j && state.busy && (j.args || {}).experiment_id === e.id;
  const run = (e.runs || []).slice(-1)[0];
  $('monitorStats').replaceChildren(stat('Status', live ? 'training' : e.status), stat('Updates', e.updates), stat('Completed epochs', e.epochs),
    stat('LATEST', e.latest ? `update ${e.latest.update}` : '—'), stat('BEST', e.best ? `update ${e.best.update} · val ${e.best.validation_loss.toFixed(4)}` : '— (no validation yet)'),
    stat('Elapsed', live ? fmtTime(j.elapsed) : '—'), stat('Throughput', live && j.updates_per_second ? `${j.updates_per_second} updates/s · ${j.clips_per_second} clips/s` : '—'),
    stat('Remaining (estimate)', live && j.eta != null ? fmtTime(j.eta) : '—'),
    stat('Split', detail.split ? (detail.split.kind === 'recording' ? 'by original recording' : 'legacy clip-level') : 'not created yet'),
    stat('Last run', run ? `${run.status} · ${run.train_clips} train / ${run.validation_clips} validation clips · ${run.gpus} GPU` : '—'));
  drawChart(detail.metrics); $('trainLog').textContent = detail.log || 'No log yet.'; $('trainLog').scrollTop = 1e9;
}
function drawChart(rows) {
  const c = $('lossChart'), g = c.getContext('2d'), w = c.width, h = c.height, pad = 46; g.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.body), ink = css.getPropertyValue('--muted'), line = css.getPropertyValue('--line');
  const tr = rows.filter(r => r.loss != null), va = rows.filter(r => r.validation_loss != null);
  g.fillStyle = ink; g.font = '13px sans-serif';
  if (!tr.length && !va.length) { g.fillText('No metrics yet. Train this experiment to see loss curves.', pad, h / 2); return }
  const all = [...tr.map(r => r.loss), ...va.map(r => r.validation_loss)], lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const u = [...tr, ...va].map(r => r.update), u0 = Math.min(...u), u1 = Math.max(...u), uspan = u1 - u0 || 1;
  const x = v => pad + (v - u0) / uspan * (w - pad * 1.5), y = v => h - pad + 10 - (v - lo) / span * (h - pad * 1.5);
  g.strokeStyle = line; g.lineWidth = 1; for (let i = 0; i <= 4; i++) { const v = lo + span * i / 4; g.beginPath(); g.moveTo(pad, y(v)); g.lineTo(w - pad / 2, y(v)); g.stroke(); g.fillText(v.toFixed(3), 2, y(v) + 4) }
  g.fillText(`update ${u0}`, pad, h - 8); g.fillText(`update ${u1}`, w - pad * 2.2, h - 8);
  const series = (data, key, color, dots) => { g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2; g.beginPath(); data.forEach((r, i) => i ? g.lineTo(x(r.update), y(r[key])) : g.moveTo(x(r.update), y(r[key]))); g.stroke(); if (dots) data.forEach(r => { g.beginPath(); g.arc(x(r.update), y(r[key]), 4, 0, 7); g.fill() }) };
  series(tr, 'loss', css.getPropertyValue('--green'), tr.length < 3); series(va, 'validation_loss', css.getPropertyValue('--orange'), true);
}
async function gpu() {
  try {
    const g = await api('gpu'); const text = c => `${c.use}% · ${(c.used / 1024).toFixed(1)}/${(c.total / 1024).toFixed(0)} GB · ${c.temperature}°C`;
    $('gpu').replaceChildren(...g.cards.map(c => el('span', `GPU ${c.index}: ${c.name} · ${text(c)}`)));
    $('gpuDetail').replaceChildren(...g.cards.map(c => stat(`GPU ${c.index} · ${c.name}`, text(c))));
  } catch (e) { $('gpu').textContent = 'GPU monitoring unavailable.' }
}

// ---------- generation ----------
function backendMeta(id) { return meta.backends.find(b => b.id === id) }
function controlInput(v, name, spec) {
  const wrap = el('div'), label = el('label', name.replace(/_/g, ' ')); let input;
  if (spec.type === 'choice') { input = el('select'); spec.choices.forEach(c => input.append(option(c, c))); input.value = v.controls[name] ?? spec.default }
  else { input = el('input'); input.type = 'number'; input.min = spec.min; input.max = spec.max; input.step = spec.type === 'int' ? 1 : 'any'; input.placeholder = spec.default == null ? 'default' : spec.default; input.value = v.controls[name] ?? '' }
  input.onchange = () => v.controls[name] = input.value === '' ? null : input.value; wrap.append(label, input, el('p', spec.help, 'help')); return wrap;
}
function renderVariant(key) {
  const v = variants[key], box = $('variant' + key); box.replaceChildren(); if (!meta) return;
  const head = el('div', undefined, 'row'); head.append(el('h3', 'Variant ' + key));
  if (key === 'B') { const l = el('label', ' Compare (A/B)', 'inline'), c = el('input'); c.type = 'checkbox'; c.checked = v.enabled; c.onchange = () => { v.enabled = c.checked; renderVariant('B') }; l.prepend(c); head.append(l) }
  box.append(head); if (!v.enabled) return;
  const select = (label, items, value, onchange) => { const w = el('div'), s = el('select'); w.append(el('label', label), s); fill(s, items, false); if (items.some(i => String(i[0]) === String(value))) s.value = value; else if (items.length) s.value = items[0][0]; onchange(s.value, true); s.onchange = () => { onchange(s.value); renderVariant(key) }; box.append(w); return s };
  select('Backend', meta.backends.map(b => [b.id, b.label + (b.inference.available ? '' : ' (unavailable)')]), v.backend, x => v.backend = x);
  const b = backendMeta(v.backend);
  box.append(el('p', b.description, 'help'));
  if (!b.inference.available) { box.append(el('p', 'Unavailable: ' + b.inference.reason, 'review')); return }
  box.append(el('p', 'Training: ' + (b.training.available ? 'available (real fine-tuning).' : b.training.reason), 'help'));
  const clips = (state.clips || []).filter(c => c.enabled !== false && (v.backend !== 'chatterbox' || c.seconds >= 5));
  const clipItems = clips.map(c => [c.id, `${c.seconds.toFixed(1)} s · ${c.text.slice(0, 50)}`]);
  if (v.backend === 'kittentts2-lora') {
    select('Model type', [['finetuned', 'Fine-tuned experiment (trained adapter weights)'], ['base', 'Pretrained base (no fine-tuning)']], v.kind, x => v.kind = x);
    if (v.kind === 'finetuned') {
      const list = experiments().filter(e => e.latest && !e.archived);
      if (!list.length) { box.append(el('p', 'No trained experiment yet.', 'review')); return }
      select('Experiment', list.map(e => [e.id, `${e.name} · ${e.updates} updates`]), v.experiment_id, x => v.experiment_id = x);
      const e = experiment(v.experiment_id), items = [];
      if (e.best) items.push(['best', `BEST · update ${e.best.update} · val ${e.best.validation_loss.toFixed(4)}`]);
      items.push(['latest', `LATEST · update ${e.latest.update}`]);
      checkpointRows(e).filter(([n]) => n.startsWith('update-')).forEach(([n, i]) => items.push([n, `${n} · update ${i.update}`]));
      select('Checkpoint', items, v.checkpoint, x => v.checkpoint = x);
      select('Reference recording (decoder conditioning)', [['', 'Experiment reference clip'], ...clipItems], v.reference_clip, x => v.reference_clip = x);
    } else {
      select('Voice', [['voice:', '— built-in voice —'], ...meta.voices.map(n => ['voice:' + n, 'Built-in: ' + n]), ...clipItems.map(([id, t]) => ['clip:' + id, 'Clone zero-shot from clip: ' + t])],
        v.reference_clip ? 'clip:' + v.reference_clip : 'voice:' + (v.voice || 'Bruno'), x => { if (x.startsWith('clip:')) { v.reference_clip = x.slice(5); v.voice = '' } else { v.voice = x.slice(6) || 'Bruno'; v.reference_clip = '' } });
    }
  } else {
    select('Reference recording (zero-shot cloning, ≥ 5 s)', [['', 'Built-in default voice'], ...clipItems], v.reference_clip, x => v.reference_clip = x);
  }
  const controls = el('div', undefined, 'fields compact'); Object.entries(b.controls).forEach(([n, s]) => controls.append(controlInput(v, n, s))); box.append(controls);
}
function variantPayload(v, key) {
  const p = {backend: v.backend, label: key};
  if (v.backend === 'kittentts2-lora') { p.kind = v.kind; if (v.kind === 'finetuned') { p.experiment_id = v.experiment_id; p.checkpoint = v.checkpoint } else p.voice = v.voice || null }
  if (v.reference_clip) p.reference_clip = v.reference_clip;
  for (const [n, x] of Object.entries(v.controls)) if (x !== null && x !== '' && n in backendMeta(v.backend).controls) p[n] = x;
  return p;
}
$('evalPick').onchange = () => { if ($('evalPick').value) $('text').value = $('evalPick').value };
$('generate').onclick = () => {
  const list = ['A', 'B'].filter(k => variants[k].enabled).map(k => variantPayload(variants[k], k));
  const texts = $('useEvalSet').checked ? meta.evaluation : [$('text').value];
  job('generate', {variants: list, texts}).catch(e => notice(e.message));
};
async function renderHistory() {
  const {history} = await api('history'); const box = $('outputs'); box.replaceChildren();
  if (!history.length) return box.append(el('p', 'No generated audio yet.', 'empty'));
  const groups = []; for (const h of history) { let g = groups.find(x => x.id === h.group && x.text === h.text); if (!g) groups.push(g = {id: h.group, text: h.text, rows: []}); g.rows.push(h) }
  for (const g of groups.slice(0, 40)) {
    const card = el('div', undefined, 'card'); card.append(el('p', '“' + g.text + '”'));
    const grid = el('div', undefined, 'compare');
    for (const h of g.rows.sort((a, b) => a.index - b.index)) {
      const cell = el('div', undefined, 'cell'), tag = h.trained ? 'FINE-TUNED' : 'PRETRAINED BASE';
      const what = h.backend === 'chatterbox' ? `Chatterbox · ${h.reference_clip ? 'zero-shot clone of clip' : 'built-in voice'} · watermarked`
        : h.trained ? `${h.experiment_name} · ${h.checkpoint} (update ${h.update})` : `KittenTTS2 · ${h.voice ? 'built-in ' + h.voice : 'zero-shot clone of clip'}`;
      const settings = Object.entries(h.settings).filter(([k, v]) => v != null && !['experiment_id', 'checkpoint', 'voice', 'reference_clip'].includes(k)).map(([k, v]) => `${k} ${v}`).join(' · ');
      cell.append(el('span', `${h.label || ''} · ${tag}`, h.trained ? 'badge' : 'badge base'), el('p', what, 'small'), el('p', `${settings || 'default settings'} · ${h.seconds} s · ${h.created}`, 'help'),
        audio('outputs', h.file), link('audio/outputs/' + h.file + '?download=1', 'Download WAV', h.file), button('Delete', () => remove('outputs', h.file), 'danger'));
      grid.append(cell);
    }
    card.append(grid); box.append(card);
  }
}

// ---------- manage ----------
$('restoreAll').onclick = () => { if (!selectedTrash.size) return notice('Select Trash entries first.');
  api('manage', {action: 'restore_many', keys: [...selectedTrash]}).then(async () => { selectedTrash.clear(); invalidate(); await refresh(); notice('Restored.') }).catch(e => notice(e.message)) };
function renderManage() {
  $('trash').replaceChildren(); $('converted').replaceChildren();
  if (!state.trash.length) $('trash').append(el('p', 'Trash is empty.', 'empty'));
  for (const t of state.trash) {
    const row = el('div', undefined, 'clip'), l = el('label', undefined, 'inline'), c = el('input'); c.type = 'checkbox'; c.checked = selectedTrash.has(t.id);
    c.onchange = () => c.checked ? selectedTrash.add(t.id) : selectedTrash.delete(t.id); l.append(c, el('span', ` ${t.label} · ${t.paths.length} files`));
    row.append(l, button('Restore', async () => { await api('manage', {action: 'restore', key: t.id}); invalidate(); await refresh() }),
      button('Permanently delete', async () => { const confirmText = prompt('This cannot be undone. Type DELETE to permanently remove this Trash entry.'); if (confirmText !== 'DELETE') return; await api('manage', {action: 'purge', key: t.id, confirm: confirmText}); invalidate(); await refresh() }, 'danger'));
    $('trash').append(row);
  }
  for (const name of state.converted) { const row = el('div', undefined, 'tools'); row.append(link('audio/converted/' + name + '?download=1', 'Download MP3 ' + name, name), button('Delete MP3', () => remove('converted', name), 'danger')); $('converted').append(row) }
}

// ---------- refresh loop ----------
function refreshViews() {
  const sig = (k, v) => { const s = JSON.stringify(v); if (sigs[k] === s) return false; sigs[k] = s; return true };
  if (sig('datasets', [state.datasets, $('datasetSelect').value])) renderDatasets();
  if (sig('samples', [state.samples, state.clips && state.clips.map(c => [c.id, c.enabled]), state.subtitles, $('datasetSelect').value])) renderSamples();
  if (sig('clips', [state.clips, $('datasetSelect').value])) renderClips();
  if (sig('exp', [state.experiments, state.datasets])) {
    renderExperiments();
    const items = experiments().filter(e => !e.archived).map(e => [e.id, `${e.name} · ${e.updates} updates`]);
    fill($('trainExperiment'), items.filter(([id]) => !experiment(id).imported)); fill($('monitorExperiment'), items);
    fill($('expDataset'), (state.datasets || []).map(d => [d.id, d.name])); $('expDataset').onchange();
    renderTraining(); renderVariant('A'); renderVariant('B');
  }
  if (sig('trash', [state.trash, state.converted])) renderManage();
  if (sig('outputs', state.outputs)) renderHistory().catch(e => notice(e.message));
}
async function refresh() {
  try {
    state = await api('status');
    const j = state.job; $('status').hidden = !j;
    if (j) { $('message').textContent = `${j.action}: ${j.message}`; $('progress').value = j.progress || 0; $('status').classList.toggle('error', j.status === 'error'); $('stop').hidden = !state.busy }
    document.querySelectorAll('.job').forEach(b => b.disabled = state.busy || uploading);
    $('files').disabled = state.busy || uploading; $('importSrt').disabled = state.busy || uploading;
    $('stt').textContent = `STT: ${state.stt.model} · language: English. Weights may download on first use.`;
    refreshViews(); renderUploadQueue();
    if (document.querySelector('#monitor.active') && (state.busy || !detail)) loadDetail();
  } catch (e) { notice('Studio connection: ' + e.message) }
}
async function init() {
  try { meta = await api('backends') } catch (e) { notice('Backend information unavailable: ' + e.message) }
  if (meta) {
    renderFixedFields();
    fill($('lrPreset'), [['', 'Custom (number below)'], ...Object.entries(meta.training.lr_presets).map(([k, v]) => [v, `${k.replace('_', ' ')} · ${v}`])]);
    fill($('evalPick'), [['', 'Pick a fixed English evaluation sentence…'], ...meta.evaluation.map(s => [s, s])]);
  }
  await refresh(); gpu(); setInterval(refresh, 3000); setInterval(gpu, 5000);
}
init();
