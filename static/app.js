'use strict';
const $ = id => document.getElementById(id);
const state = { editingLayer: null, layers: new Set(), layerAlphas: {}, alpha: 1, mode: 'baseline', count: 28, ready: false, busy: false, controller: null, lastPrompt: '', runs: [] };
let toastTimer, lastFocus;
function icon(id) { return `<svg aria-hidden="true"><use href="#i-${id}"/></svg>`; }
function toast(text) { $('toast').textContent = text; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 2500); }
function active() { return state.mode === 'custom'; }
function sortedLayers() { return [...state.layers].sort((a,b) => a-b); }
function layerScales() { return Object.fromEntries(sortedLayers().map(i=>[i,state.layerAlphas[i] ?? state.alpha])); }
function settingsLabel(layers, alpha, mode = state.mode) {
  if(mode==='baseline')return 'Original projections';
  if(!layers.length)return 'Trained curves · α 1 on all layers';
  const values=layers.map(i=>state.layerAlphas[i] ?? alpha);
  if(new Set(values).size>1)return layers.map((i,n)=>`L${i}: ${values[n].toFixed(1)}`).join(' · ');
  return `α ${(values[0] ?? alpha).toFixed(1)} · ${layers.length===state.count?'All layers':layers.map(i=>'L'+i).join(', ')}`;
}
function renderLayerScales() {
  $('layerScales').replaceChildren();
  for(const i of sortedLayers()) {
    const row=document.createElement('div');row.className='layer-scale-row';
    const edit=document.createElement('button');edit.className='text-button';edit.textContent=`Layer ${i}: ${state.layerAlphas[i].toFixed(1)}`;edit.disabled=state.busy;
    edit.onclick=()=>{state.editingLayer=i;updateSettings();};
    const remove=document.createElement('button');remove.className='text-button';remove.textContent='Reset';remove.setAttribute('aria-label',`Reset layer ${i}`);remove.disabled=state.busy;
    remove.onclick=()=>{state.layers.delete(i);delete state.layerAlphas[i];updateSettings();};
    row.append(edit,remove);$('layerScales').append(row);
  }
  $('individualScales').hidden=!state.layers.size;
}
function saveSettings() { try { sessionStorage.setItem('manifolds-ab-v1', JSON.stringify({layers:sortedLayers(),alpha:1,mode:state.mode,layerAlphas:layerScales()})); } catch {} }
function buildLayers() {
  $('layerGrid').replaceChildren();
  for (let i=0;i<state.count;i++) {
    const b=document.createElement('button');b.className='layer-button';b.textContent=i;b.setAttribute('aria-label',`Layer ${i}`);
    b.onclick=()=>{state.editingLayer=i;updateSettings();};
    $('layerGrid').append(b);
  }
}
function updateSettings(preserveInput = false) {
  [...$('layerGrid').children].forEach((b,i)=>{b.setAttribute('aria-pressed',String(state.editingLayer===i));b.classList.toggle('configured',state.layers.has(i));b.title=state.layers.has(i)?`Layer ${i}: ${state.layerAlphas[i]}`:`Layer ${i}: no custom scale`;b.disabled=state.busy;});
  $('layerCount').textContent=`${state.layers.size} configured`;
  const i=state.editingLayer;
  $('layerEditor').hidden=i===null;
  $('editorHint').hidden=i!==null;
  if(i!==null){
    $('editorTitle').textContent=`Layer ${i}`;
    $('layerValue').setAttribute('aria-label',`Layer ${i} strength`);
    if(!preserveInput)$('layerValue').value=state.layers.has(i)?state.layerAlphas[i].toFixed(1):'';
    $('layerSlider').value=state.layers.has(i)?state.layerAlphas[i]:1;
    $('layerSlider').setAttribute('aria-label',`Layer ${i} scale slider`);
    $('layerSlider').setAttribute('aria-valuetext',state.layers.has(i)?String(state.layerAlphas[i]):'Not set; move to apply');
    $('editorStatus').textContent=state.layers.has(i)?'Saved for this layer. Other layers stay unchanged.':'No custom scale set. Enter a value to change this layer.';
  }
  $('layerValue').disabled=state.busy;$('layerSlider').disabled=state.busy;
  $('baselinePreset').classList.toggle('active',state.mode==='baseline');
  $('customPreset').classList.toggle('active',state.mode==='custom');
  $('customControls').hidden=state.mode!=='custom';
  $('pillText').textContent=active()?(state.layers.size?settingsLabel(sortedLayers(),1):'BSpline · trained curves'):'Llama-3.2-3B-Instruct';
  $('settingsPill').classList.toggle('active',active());$('selectionDot').hidden=!active();
  $('summaryTitle').textContent=active()?`${state.layers.size} layers configured`:'Original Llama weights';
  $('summaryBody').textContent=active()?'Only values you explicitly set are included. Unconfigured layers use trained curves.':'Running original Llama-3.2-3B-Instruct without spline curves.';
  $('openControls').hidden=false;
  ['baselinePreset','customPreset','resetSettings','newChat','bulkValue','bulkSlider','applyAllLayers','tokenLimit'].forEach(id=>$(id).disabled=state.busy);
  renderLayerScales();saveSettings();
}
$('bulkSlider').oninput=e=>{$('bulkValue').value=e.target.value;};
$('bulkValue').oninput=e=>{if(e.target.value!==''&&e.target.validity.valid)$('bulkSlider').value=e.target.value;};
$('applyAllLayers').onclick=()=>{
  if(state.busy)return;
  const input=$('bulkValue');
  if(input.value===''||!input.reportValidity())return;
  const value=Math.round(Number(input.value)*10)/10;
  if(!Number.isFinite(value)||value < -3||value > 3)return;
  state.mode='custom';state.layers=new Set(Array.from({length:state.count},(_,i)=>i));
  state.layerAlphas=Object.fromEntries(sortedLayers().map(i=>[i,value]));
  updateSettings();toast(`Applied α ${value.toFixed(1)} to all ${state.count} layers`);
};
$('layerValue').oninput=e=>{const v=Number(e.target.value),i=state.editingLayer;if(i!==null&&e.target.value!==''&&Number.isFinite(v)&&v>=-3&&v<=3){state.layers.add(i);state.layerAlphas[i]=Math.round(v*10)/10;updateSettings(true);}};
$('layerValue').onchange=()=>updateSettings();
$('layerSlider').oninput=e=>{const i=state.editingLayer;if(i===null)return;state.layers.add(i);state.layerAlphas[i]=Number(e.target.value);updateSettings();};
function baseline() {state.mode='baseline';state.editingLayer=null;updateSettings();reusePrompt();}
function reusePrompt(){if(!$('prompt').value&&state.lastPrompt){$('prompt').value=state.lastPrompt;resizeInput();}}
function syncDrawerAccessibility(){const narrow=matchMedia('(max-width:760px)').matches,opened=$('controls').classList.contains('open');$('controls').inert=narrow&&!opened;document.querySelector('.chat-pane').inert=narrow&&opened;document.querySelector('.topbar').inert=narrow&&opened;}
function openControls() {lastFocus=document.activeElement;$('controls').classList.add('open');syncDrawerAccessibility();$('openControls').setAttribute('aria-expanded','true');if(matchMedia('(max-width:760px)').matches){$('backdrop').hidden=false;$('closeControls').focus();}}
function closeControls() {$('controls').classList.remove('open');syncDrawerAccessibility();$('backdrop').hidden=true;$('openControls').setAttribute('aria-expanded','false');if(lastFocus)lastFocus.focus();}
$('openControls').onclick=openControls;$('settingsPill').onclick=openControls;$('closeControls').onclick=closeControls;$('backdrop').onclick=closeControls;
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeControls();if(e.key==='Tab'&&!$('backdrop').hidden){const els=[...$('controls').querySelectorAll('button:not(:disabled),input:not(:disabled),a,summary')];const first=els[0],last=els[els.length-1];if(e.shiftKey&&document.activeElement===first){last.focus();e.preventDefault();}else if(!e.shiftKey&&document.activeElement===last){first.focus();e.preventDefault();}}});
window.addEventListener('resize',()=>{syncDrawerAccessibility();if(!matchMedia('(max-width:760px)').matches)$('backdrop').hidden=true;else if($('controls').classList.contains('open'))$('backdrop').hidden=false;});
$('baselinePreset').onclick=baseline;$('resetSettings').onclick=()=>{baseline();toast('Switched to Llama-3.2-3B-Instruct');};
$('customPreset').onclick=()=>{state.mode='custom';updateSettings();reusePrompt();openControls();};
function updateSend() {$('compare').disabled=state.busy||!state.ready||!$('prompt').value.trim();$('exportRuns').disabled=!state.runs.length||state.busy;$('send').disabled=!state.busy&&(!state.ready||!$('prompt').value.trim());$('send').classList.toggle('stopping',state.busy);$('send').setAttribute('aria-label',state.busy?'Stop generating':'Send message');$('send').innerHTML=state.busy?'<span class="stop-square"></span>':icon('arrow');$('prompt').disabled=state.busy;}
function resizeInput() {$('prompt').style.height='auto';$('prompt').style.height=Math.min($('prompt').scrollHeight,140)+'px';updateSend();}
$('prompt').addEventListener('input',resizeInput);
$('prompt').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();if(!$('send').disabled&&!state.busy)send();}});
$('send').onclick=()=>state.busy?state.controller?.abort():send();
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>{$('prompt').value=b.dataset.prompt;resizeInput();$('prompt').focus();});
$('newChat').onclick=()=>{if(state.busy)return;state.lastPrompt='';state.runs=[];state.layers.clear();state.layerAlphas={};state.alpha=1;baseline();$('messages').replaceChildren();$('welcome').hidden=false;$('error').hidden=true;$('prompt').value='';resizeInput();$('prompt').focus();toast('New chat · Llama-3.2-3B-Instruct');};
function scrollBottom(force=false) {const s=$('scrollArea');if(force||s.scrollHeight-s.scrollTop-s.clientHeight<180)s.scrollTop=s.scrollHeight;}
function addUser(text) {const el=document.createElement('article');el.className='message user';const bubble=document.createElement('div');bubble.className='user-bubble';bubble.textContent=text;el.append(bubble);$('messages').append(el);}
function addAssistant(layers,alpha,mode) {const el=document.createElement('article');el.className='message assistant';const head=document.createElement('div');head.className='assistant-heading';head.innerHTML=icon('logo');const name=document.createElement('strong');name.textContent=mode==='baseline'?'Llama-3.2-3B-Instruct':'BSpline · interventions';head.append(name);const badge=document.createElement('span');badge.className='answer-settings';badge.textContent='Awaiting server confirmation';head.append(badge);const content=document.createElement('div');content.className='answer-content waiting';content.textContent='Thinking…';const actions=document.createElement('div');actions.className='answer-actions';el.append(head,content,actions);$('messages').append(el);return {el,content,actions,badge,name};}
function formatAnswer(container,text){
  // Text nodes only: model output is never interpreted as HTML.
  container.replaceChildren();
  const chunks=text.split(/```/);
  chunks.forEach((chunk,i)=>{if(i%2){const pre=document.createElement('pre');const code=document.createElement('code');code.textContent=chunk.replace(/^\w*\n/,'');pre.append(code);container.append(pre);}else{chunk.split(/(\*\*[^*]+\*\*)/g).forEach(part=>{if(part.startsWith('**')&&part.endsWith('**')){const b=document.createElement('strong');b.textContent=part.slice(2,-2);container.append(b);}else container.append(document.createTextNode(part));});}});
}
function configurationLabel(config){
  if(config.mode==='baseline')return 'Original Llama · splines off';
  const entries=Object.entries(config.layer_alphas);
  const overrides=Object.entries(config.overrides);
  if(!overrides.length)return 'Trained curves · α 1 on all layers';
  if(overrides.length===entries.length&&new Set(overrides.map(x=>x[1])).size===1)return `All ${entries.length} layers · α ${overrides[0][1]}`;
  return overrides.map(([i,a])=>`L${i}: ${a}`).join(' · ');
}
async function send(compare=false){
  const prompt=$('prompt').value.trim();if(!prompt||state.busy||!state.ready)return;
  const layers=sortedLayers(),alpha=state.alpha,layer_alphas=layerScales(),max_tokens=Number($('tokenLimit').value);
  const modes=compare?['baseline','custom']:[state.mode];
  state.lastPrompt=prompt;state.busy=true;state.controller=new AbortController();$('error').hidden=true;$('welcome').hidden=true;
  addUser(prompt);$('prompt').value='';resizeInput();updateSettings();scrollBottom(true);
  try{
    for(const mode of modes){
      if(state.controller.signal.aborted)break;
      const request={mode,messages:[{role:'user',content:prompt}],layers:mode==='baseline'?[]:layers,alpha,layer_alphas:mode==='baseline'?{}:layer_alphas,max_tokens};
      const answer=addAssistant(request.layers,alpha,mode);
      let text='',finish=null,config=null,failure=null;
      try{
        const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request),signal:state.controller.signal});
        if(!res.ok){const data=await res.json();throw new Error(typeof data.detail==='string'?data.detail:'Request rejected; check the settings.');}
        const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='';
        while(true){
          const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});
          const events=buffer.split('\n\n');buffer=events.pop();
          for(const event of events){
            if(!event.startsWith('data: '))continue;const data=JSON.parse(event.slice(6));
            if(data.type==='configuration'){
              config=data;answer.name.textContent=data.mode==='baseline'?data.model:'BSpline · interventions';answer.badge.textContent=configurationLabel(data);answer.badge.title=`Server verified · build ${data.build}`;
            }else if(data.type==='token'){text+=data.text;answer.content.classList.remove('waiting');answer.content.textContent=text;scrollBottom();}
            else if(data.type==='done')finish=data;
            else if(data.type==='error')throw new Error(data.message);
          }
        }
        if(!config)throw new Error('Server version mismatch: this server did not confirm the applied model settings. Restart it with the updated source.');
        if(!finish)throw new Error('The response stream ended before generation completed.');
        if(!text)throw new Error('The model returned an empty response.');
      }catch(e){
        failure=e.name==='AbortError'?'Stopped':e.message;
        if(e.name!=='AbortError'){$('error').textContent=failure;$('error').hidden=false;}
      }
      answer.content.classList.remove('waiting');
      if(text){formatAnswer(answer.content,text);const copy=document.createElement('button');copy.className='copy-button';copy.textContent='Copy';copy.onclick=async()=>{try{await navigator.clipboard.writeText(text);toast('Response copied');}catch{toast('Select the response to copy it.');}};answer.actions.append(copy);}
      else answer.content.textContent=failure||'No response generated.';
      const info=document.createElement('span');info.textContent=failure||`${finish.seconds}s · ${finish.tokens} tokens${finish.truncated?' · token limit reached':''}`;answer.actions.append(info);
      state.runs.push({prompt,request,configuration:config,response:text,finish,error:failure});
      scrollBottom(true);if(failure)break;
    }
  }finally{state.busy=false;state.controller=null;updateSettings();resizeInput();$('prompt').focus();}
}
$('compare').onclick=()=>send(true);
$('exportRuns').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({exported_at:new Date().toISOString(),runs:state.runs},null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='attention-manifolds-comparison.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function pollStatus(){try{const r=await fetch('/api/status');const data=await r.json();state.ready=data.state==='ready';$('connection').className=`connection ${data.state}`;$('statusText').textContent=state.ready?'Model ready':data.state==='error'?'Model unavailable':'Loading model';$('connection').title=state.ready?`${data.model} · running locally`:'The model is loading into memory';$('runtimeInfo').textContent=`${data.device || 'unknown device'} · build ${data.build || 'unknown (restart server)'}`;if(data.layers!==state.count){state.count=data.layers;state.layers=new Set([...state.layers].filter(i=>i<state.count));buildLayers();updateSettings();}if(data.state==='error'){$('error').textContent=data.message;$('error').hidden=false;}updateSend();if(!state.ready&&data.state!=='error')setTimeout(pollStatus,1800);}catch{$('connection').className='connection error';$('statusText').textContent='Disconnected';state.ready=false;updateSend();setTimeout(pollStatus,3000);}}
try{const saved=JSON.parse(sessionStorage.getItem('manifolds-ab-v1'));if(saved&&Number.isFinite(saved.alpha)&&saved.alpha>=-3&&saved.alpha<=3&&Array.isArray(saved.layers)){state.mode=saved.mode==='custom'?'custom':'baseline';state.alpha=saved.alpha;state.layers=new Set(saved.layers.filter(i=>Number.isInteger(i)&&i>=0&&i<state.count));for(const i of state.layers)state.layerAlphas[i]=saved.alpha;if(saved.layerAlphas&&typeof saved.layerAlphas==='object')for(const i of state.layers){const v=saved.layerAlphas[i];if(Number.isFinite(v)&&v>=-3&&v<=3)state.layerAlphas[i]=Math.round(v*10)/10;}}}catch{}
if(state.mode==='baseline'){state.layers.clear();state.layerAlphas={};state.alpha=1;}
buildLayers();updateSettings();syncDrawerAccessibility();pollStatus();
$('openHelp').onclick=()=>{$('helpOverlay').hidden=false;};
if($('openGuideLeft'))$('openGuideLeft').onclick=()=>{$('helpOverlay').hidden=false;};
if($('openGuideRight'))$('openGuideRight').onclick=()=>{$('helpOverlay').hidden=false;};
$('closeHelp').onclick=()=>{$('helpOverlay').hidden=true;};
$('helpOverlay').onclick=e=>{if(e.target===$('helpOverlay'))$('helpOverlay').hidden=true;};
document.querySelectorAll('.help-prompt[data-prompt]').forEach(b=>b.onclick=()=>{$('helpOverlay').hidden=true;$('prompt').value=b.dataset.prompt;resizeInput();$('prompt').focus();});
