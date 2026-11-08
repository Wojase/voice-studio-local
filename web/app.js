const $=id=>document.getElementById(id);
let state={},limit=20,uploading=false,settingsLoaded=false,clipSig='',sampleSig='',outSig='',manageSig='',uploadQueue=[];
function notice(s){$('toast').textContent=s;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,8000)}
function el(tag,text,cls){let e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
async function api(path,body){let r=await fetch('api/'+path,body===undefined?{}:{method:'POST',headers:{'X-Studio':'local',...(body instanceof FormData?{}:{'Content-Type':'application/json'})},body:body instanceof FormData?body:JSON.stringify(body)});let d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
function button(text,fn,cls=''){let b=el('button',text,cls);b.onclick=async()=>{try{b.disabled=true;await fn()}catch(e){notice(e.message)}finally{b.disabled=false}};return b}
function audio(kind,name){let a=el('audio');a.controls=true;a.preload='none';a.src='audio/'+kind+'/'+name;return a}
function link(kind,name,text){let a=el('a',text,'button');a.href='audio/'+kind+'/'+name;a.download=name;return a}
function invalidate(){clipSig=sampleSig=outSig=manageSig=''}
async function remove(kind,key=''){if(!confirm(`Move ${key||kind} to Trash? Model weights are only removed by a model/studio reset.`))return;await api('manage',{kind,key});invalidate();await refresh();notice('Moved to Trash. Restore in Manage.')}
async function job(action,args={}){await api('jobs',{action,...args});await refresh()}
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('nav button,section').forEach(e=>e.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active')});
document.querySelectorAll('[data-delete]').forEach(b=>b.onclick=()=>remove(b.dataset.delete).catch(e=>notice(e.message)));
$('stop').onclick=()=>api('stop',{}).then(d=>notice(d.message)).catch(e=>notice(e.message));
$('prepare').onclick=()=>job('prepare').catch(e=>notice(e.message));
$('train').onclick=()=>job('train',{epochs:Number($('epochs').value),accumulation:Number($('accumulation').value),batch_size:Number($('batch').value),lr:Number($('lr').value)}).catch(e=>notice(e.message));
$('generate').onclick=()=>job('generate',{text:$('text').value}).catch(e=>notice(e.message));
function stageFiles(files){
  if(uploading)return notice('Wait for the current upload to finish.');
  for(const file of files){
    if(!/\.(wav|mp3|mp4|m4v|mov|mkv|ogg|flac)$/i.test(file.name)){notice('Choose audio/video first. Attach SRT in the staged recording row.');continue}
    uploadQueue.push({file,srt:null,mode:'use'});
  }
  $('files').value='';renderUploadQueue();
}
function renderUploadQueue(){
  $('uploadStage').hidden=!uploadQueue.length;$('uploadQueue').replaceChildren();
  for(const item of uploadQueue){
    let row=el('div',undefined,'clip');row.append(el('strong',item.file.name));
    let label=el('label','Optional subtitle file (.srt)'),input=el('input');input.type='file';input.accept='.srt';
    // Preserve the actual File object when staging more recordings; file inputs cannot be filled programmatically.
    input.onchange=()=>{item.srt=input.files[0]||null;renderUploadQueue()};label.append(input);
    row.append(label,el('p',item.srt?item.srt.name:'No subtitles attached.','small'));
    if(item.srt){let modeLabel=el('label','Subtitle processing'),select=el('select');
      for(const [value,text]of [['use','Use SRT — trust text, no STT'],['check','SRT + check — verify using English STT']]){let o=el('option',text);o.value=value;select.append(o)}
      select.value=item.mode;select.onchange=()=>item.mode=select.value;modeLabel.append(select);row.append(modeLabel,button('Remove SRT',()=>{item.srt=null;renderUploadQueue()}));}
    row.append(button('Remove recording',()=>{uploadQueue=uploadQueue.filter(q=>q!==item);renderUploadQueue()},'danger'));
    $('uploadQueue').append(row);
  }
  $('confirmUpload').textContent=`Confirm upload (${uploadQueue.length})`;
  document.querySelectorAll('#uploadStage button,#uploadStage input,#uploadStage select').forEach(e=>e.disabled=uploading);
  $('confirmUpload').disabled=uploading||state.busy||!uploadQueue.length;
}
async function submitUploads(){
  if(uploading||state.busy)return notice('Wait for the current operation.');
  uploading=true;renderUploadQueue();const errors=[],checkIds=[];
  const staged=[...uploadQueue];
  for(const [i,item]of staged.entries()){
    try{notice(`Uploading ${i+1}/${staged.length}: ${item.file.name}`);const form=new FormData();form.append('file',item.file);
      if(item.srt){form.append('subtitles',item.srt);form.append('subtitle_mode',item.mode)}
      const result=await api('upload',form);if(result.check_requested)checkIds.push(result.id);
      uploadQueue=uploadQueue.filter(q=>q!==item);
    }catch(e){errors.push(`${item.file.name}: ${e.message}`)}
  }
  uploading=false;renderUploadQueue();invalidate();await refresh();
  if(checkIds.length){try{await job('check_subtitles',{sample_ids:[...new Set(checkIds)]})}catch(e){errors.push('Files uploaded, but STT check could not start: '+e.message+'. Use Audio + SRT to run it again.')}}
  notice(errors.length?errors.join('\n'):checkIds.length?'Uploaded. SRT checking started.':'Upload complete. SRT-only clips are ready; recordings without SRT still need preparation.');
}
$('confirmUpload').onclick=()=>submitUploads().catch(e=>notice(e.message));
$('cancelUpload').onclick=()=>{if(!uploading){uploadQueue=[];renderUploadQueue()}};
$('files').onchange=e=>stageFiles([...e.target.files]);
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('drag')};
$('drop').ondragleave=()=>$('drop').classList.remove('drag');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('drag');stageFiles([...e.dataTransfer.files])};
$('importSrt').onclick=async()=>{try{let f=$('srtFile').files[0],s=$('srtAudio').value;if(!f||!s)return notice('Choose a recording and SRT first.');if(!confirm('Replace this recording’s clips? Existing clips go to Trash.'))return;let form=new FormData();form.append('file',f);let d=await api('subtitles/'+s,form);invalidate();await refresh();notice(`Imported ${d.cues} cues as ${d.clips} clips. Review or run SRT + check.`)}catch(e){notice(e.message)}};
$('checkSrt').onclick=()=>job('check_subtitles',{sample_id:$('srtAudio').value}).catch(e=>notice(e.message));
$('more').onclick=()=>{limit+=20;renderClips()};
function renderClips(){$('clips').replaceChildren();if(!state.clips.length)$('clips').append(el('p','No prepared clips yet.','empty'));for(let c of state.clips.slice(0,limit)){let box=el('div',undefined,'clip'),row=el('div',undefined,'row');row.append(el('span',`${c.seconds.toFixed(1)} s · ${c.source||'STT'} · ${c.review||'manual review'}`,c.review==='needs_review'?'review':'small'),audio('clips',c.id+'.wav'));let ta=el('textarea');ta.value=c.text;ta.rows=2;ta.setAttribute('aria-label','Clip transcript');let check=el('input');check.type='checkbox';check.checked=c.enabled;let label=el('label',' Include in training');label.prepend(check);let tools=el('div',undefined,'tools');tools.append(button('Save transcript / review',async()=>{await api('clips/'+c.id,{text:ta.value,enabled:check.checked});notice('Clip saved.')}),button('Delete clip',()=>remove('clips',c.id),'danger'));box.append(row,ta,label);if(c.check)box.append(el('p',`STT heard: ${c.check.recognized} · word error ${(c.check.word_error*100).toFixed(0)}%`,'small'));box.append(tools);$('clips').append(box)}$('more').hidden=state.clips.length<=limit}
async function refresh(){try{state=await api('status');if(!settingsLoaded){$('batch').value=state.settings.batch_size;settingsLoaded=true}let j=state.job;$('status').hidden=!j;if(j){$('message').textContent=j.message;$('progress').value=j.progress||0;$('status').classList.toggle('error',j.status==='error');$('stop').hidden=!state.busy}document.querySelectorAll('.job').forEach(b=>b.disabled=state.busy||uploading);$('generate').disabled=state.busy||!state.model;$('files').disabled=state.busy||uploading;$('confirmUpload').disabled=state.busy||uploading||!uploadQueue.length;$('importSrt').disabled=state.busy||uploading;$('summary').textContent=`${state.samples.length} recordings · ${(state.samples.reduce((n,c)=>n+c.seconds,0)/60).toFixed(1)} minutes · ${state.clips.filter(c=>c.enabled).length} enabled clips`;$('stt').textContent=`STT: ${state.stt.model} · language: English. Weights may download on first use.`;$('trainTitle').textContent=state.model?'Continue training your voice':'Train your voice';$('train').textContent=state.model?'Continue training':'Start training';$('modelInfo').textContent=state.model?`Saved updates: ${state.model.steps} · completed epochs: ${state.model.epochs} · clips: ${state.model.samples}`:'No saved model yet.';
let cs=state.clips.map(c=>c.id+(c.check?JSON.stringify(c.check):'')).join(',');if(cs!==clipSig||!$('clips').childNodes.length){clipSig=cs;renderClips()}
let ss=JSON.stringify([state.samples,state.subtitles]);if(ss!==sampleSig){sampleSig=ss;$('sampleList').replaceChildren();let selected=$('srtAudio').value;$('srtAudio').replaceChildren();for(let s of state.samples){let row=el('div',undefined,'clip');row.append(el('strong',s.name),el('p',`${(s.seconds/60).toFixed(1)} minutes`,'small'),audio('samples',s.id+'.wav'),button('Delete clips / redo STT',()=>remove('sample_clips',s.id),'danger'),button('Delete recording + its clips',()=>remove('samples',s.id),'danger'));$('sampleList').append(row);let option=el('option',s.name);option.value=s.id;$('srtAudio').append(option)}if(state.samples.some(s=>s.id===selected))$('srtAudio').value=selected;$('subtitleList').replaceChildren(...state.subtitles.map(s=>el('p',`${s.filename}: ${s.cues} cues → ${s.clips} clips`,'small')))}
let os=state.outputs.join(',');if(os!==outSig||!$('outputs').childNodes.length){outSig=os;$('outputs').replaceChildren();if(!state.outputs.length)$('outputs').append(el('p','No generated audio yet.','empty'));for(let name of state.outputs){let row=el('div',undefined,'clip');row.append(audio('outputs',name),link('outputs',name,'Download WAV'),button('Delete output',()=>remove('outputs',name),'danger'));$('outputs').append(row)}}
let ms=JSON.stringify([state.trash,state.converted]);if(ms!==manageSig){manageSig=ms;$('trash').replaceChildren();$('converted').replaceChildren();for(let t of state.trash){let row=el('div',undefined,'clip');row.append(el('p',`${t.label} · ${t.paths.length} files`),button('Restore',async()=>{await api('manage',{action:'restore',key:t.id});invalidate();await refresh()}),button('Permanently delete',async()=>{let confirm=prompt('This cannot be undone. Type DELETE to permanently remove this trash entry.');if(confirm!=='DELETE')return;await api('manage',{action:'purge',key:t.id,confirm});invalidate();await refresh()},'danger'));$('trash').append(row)}for(let name of state.converted){let row=el('div',undefined,'tools');row.append(link('converted',name,'Download MP3'),button('Delete MP3',()=>remove('converted',name),'danger'));$('converted').append(row)}}
}catch(e){notice('Studio connection: '+e.message)}}
async function gpu(){try{let g=await api('gpu');$('gpu').replaceChildren(...g.cards.map((c,i)=>el('span',`GPU ${i}: ${c.name} · ${c.use}% · ${(c.used/1024).toFixed(1)}/${(c.total/1024).toFixed(0)} GB · ${c.temperature}°C`)))}catch(e){$('gpu').textContent='GPU monitoring unavailable.'}}refresh();gpu();setInterval(refresh,2500);setInterval(gpu,5000);
