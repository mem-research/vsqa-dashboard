/* Training DAG: uses canonical Variant records and the same drawer/write API as the table. */
'use strict';
const recipeGraph = { data: null, selected: null, focus: false, zoom: 1, width: 1, height: 1, active: false };
async function showRecipeTab(active) {
  recipeGraph.active = active;
  $('#table-view').hidden = active; $('#dag-view').hidden = !active;
  document.body.classList.toggle('dag-active', active);
  for (const [id, on] of [['tab-table', !active], ['tab-dag', active]]) {
    $(`#${id}`).classList.toggle('primary', on); $(`#${id}`).setAttribute('aria-pressed', String(on));
  }
  $('#bottom').classList.add('hidden'); window.scrollTo(0, 0); updatePinnedOffsets();
  if (active) await refreshRecipes();
}
$('#tab-table').onclick = () => showRecipeTab(false);
$('#tab-dag').onclick = () => showRecipeTab(true);
async function refreshRecipes() {
  $('#dag-summary').textContent = 'Loading lineage…';
  try { recipeGraph.data = await api('/api/lineage'); renderRecipes(); }
  catch (e) { $('#dag-summary').textContent = 'Graph unavailable'; $('#dag-warnings').textContent = e.message; }
}
function relatedStages(id, direction) {
  const found = new Set([id]), pending = [id];
  while (pending.length) {
    const n = pending.pop();
    for (const e of recipeGraph.data.edges) {
      const next = direction === 'ancestors' ? (e.to === n ? e.from : null) : (e.from === n ? e.to : null);
      if (next && !found.has(next)) { found.add(next); pending.push(next); }
    }
  }
  return found;
}
function graphSelection() {
  const data = recipeGraph.data; if (!data) return [];
  const q = $('#dag-search').value.trim().toLowerCase();
  let ids = new Set(data.nodes.map(n => n.id));
  if (recipeGraph.focus && recipeGraph.selected) ids = new Set([...relatedStages(recipeGraph.selected, 'ancestors'), ...relatedStages(recipeGraph.selected, 'descendants')]);
  if (q) {
    const hits = data.nodes.filter(n => `${n.id} ${n.name} ${n.method || ''} ${n.checkpoints.map(c => c.step).join(' ')}`.toLowerCase().includes(q));
    const keep = new Set(hits.flatMap(n => [...relatedStages(n.id, 'ancestors')]));
    ids = new Set([...ids].filter(id => keep.has(id)));
  }
  return data.nodes.filter(n => ids.has(n.id));
}
function renderRecipes() {
  const data = recipeGraph.data; if (!data) return;
  const nodes = graphSelection(), positions = new Map(), heights = new Map(), ports = new Map();
  const layers = [...new Set(nodes.map(n => n.layer))].sort((a,b) => a-b);
  nodes.sort((a,b) => a.layer-b.layer || a.id.localeCompare(b.id, undefined, {numeric:true}));
  for (const n of nodes) {
    // Include historical outgoing steps even when their checkpoint payload was removed.
    const steps = [...new Set([...n.checkpoints.map(c=>c.step), ...data.edges.filter(e=>e.from===n.id).map(e=>e.step??null)])]
      .sort((a,b)=>(a??-1)-(b??-1));
    ports.set(n.id, new Map(steps.map((step,i)=>[step,180+i*26])));
    const y=heights.get(n.layer)||40, height=Math.max(186,190+steps.length*26);
    positions.set(n.id, {x:30+layers.indexOf(n.layer)*440,y,height});
    heights.set(n.layer,y+height+32);
  }
  recipeGraph.width = Math.max(600, layers.length*440); recipeGraph.height = Math.max(400, ...heights.values());
  const highlight = recipeGraph.selected ? new Set([...relatedStages(recipeGraph.selected,'ancestors'), ...relatedStages(recipeGraph.selected,'descendants')]) : null;
  const edges = data.edges.filter(e => positions.has(e.from) && positions.has(e.to));
  const paths = edges.map(e => {
    const a=positions.get(e.from), b=positions.get(e.to), x=a.x+328, y=a.y+ports.get(e.from).get(e.step??null), end=b.x, ey=b.y+88;
    const child=data.nodes.find(n=>n.id===e.to), planned=child?.status==='PLANNED';
    const label=e.step==null?'weights':`@${e.step}${e.ema?' EMA':' student'}`;
    return `<g class="${highlight && !(highlight.has(e.from)&&highlight.has(e.to))?'dag-dim':''}"><path class="dag-edge ${planned?'dag-planned':''}" data-from="${esc(e.from)}" data-to="${esc(e.to)}" data-step="${e.step??'weights'}" d="M ${x} ${y} C ${x+60} ${y}, ${end-60} ${ey}, ${end} ${ey}" marker-end="url(#dag-arrow)"/><title>${esc(label)} · ${esc(e.kind)}</title></g>`;
  }).join('');
  $('#dag-canvas').innerHTML = `<svg class="dag-edges" width="${recipeGraph.width}" height="${recipeGraph.height}" aria-hidden="true"><defs><marker id="dag-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#8a94a3"/></marker></defs>${paths}</svg>` + nodes.map(n=>{
    const pos=positions.get(n.id), available=n.checkpoints.filter(c=>c.available), target=n.target??'—';
    return `<article class="dag-stage ${n.status==='PLANNED'?'dag-planned':''} ${recipeGraph.selected===n.id?'dag-selected':''} ${highlight&&!highlight.has(n.id)?'dag-dim':''}" style="left:${pos.x}px;top:${pos.y}px;height:${pos.height}px" data-stage="${esc(n.id)}">
      <button class="dag-title" data-open-stage="${esc(n.id)}">${esc(n.external?n.status:n.id)} ${badge(n.status)}</button>
      <div class="dag-name" title="${esc(n.name)}">${esc(n.name)}</div>
      ${n.external?'<p class="muted">Source checkpoint / base model</p>':`<div>${esc(n.method)} · steps ${n.start} → ${target}</div><div class="muted">${n.has_execution?'Observed '+n.observed:'Not launched'}</div><div class="dag-checkpoints" title="${esc(available.map(c=>c.step+(c.ema?' EMA':'')).join(', '))}">ckpt: ${esc(available.slice(-4).map(c=>c.step+(c.ema?'e':'')).join(' · ')||'none yet')} · ${available.length} available</div><button class="small" data-open-stage="${esc(n.id)}">Recipe / checkpoints / branch</button>`}
      ${[...ports.get(n.id)].map(([step,y])=>`<button class="dag-port" data-port-stage="${esc(n.id)}" data-step="${step??'weights'}" style="top:${y-1}px" aria-label="${esc(n.id)} checkpoint ${step??'base weights'}" title="Checkpoint ${step??'base weights'}: view lineage / branch"><span>${step==null?'weights':'@'+step}</span></button>`).join('')}
    </article>`;
  }).join('');
  for(const button of $$('[data-open-stage]',$('#dag-canvas'))) button.onclick=()=>openRecipeStage(button.dataset.openStage);
  for(const button of $$('[data-port-stage]',$('#dag-canvas'))) button.onclick=async()=>{
    const id=button.dataset.portStage, step=button.dataset.step==='weights'?null:Number(button.dataset.step);
    await openRecipeStage(id);
    if(recipeGraph.selected!==id)return;
    const node=data.nodes.find(n=>n.id===id), select=$('#dag-checkpoint');
    if(!select)return;
    const choices=node.checkpoints.map((c,i)=>({c,i})).filter(({c})=>c.step===step);
    for(const option of [...select.options])if(!choices.some(({i})=>String(i)===option.value))option.remove();
    $('#dag-branch').disabled=!choices.length;
    if(!choices.length)select.innerHTML='<option>Historical checkpoint — artifact unavailable</option>';
  };
  $('#dag-summary').textContent=`${nodes.filter(n=>!n.external).length} stages · ${edges.length} edges`;
  $('#dag-warnings').textContent=data.warnings.join('\n'); applyGraphZoom();
}
function applyGraphZoom() {
  const g=recipeGraph; $('#dag-canvas').style.transform=`scale(${g.zoom})`;
  $('#dag-space').style.width=`${g.width*g.zoom}px`; $('#dag-space').style.height=`${g.height*g.zoom}px`;
  $('#dag-reset').textContent=`${Math.round(g.zoom*100)}%`;
}
function zoomGraph(value) { recipeGraph.zoom=Math.max(.1,Math.min(2,value));applyGraphZoom(); }
$('#dag-in').onclick=()=>zoomGraph(recipeGraph.zoom*1.2); $('#dag-out').onclick=()=>zoomGraph(recipeGraph.zoom/1.2);
$('#dag-reset').onclick=()=>zoomGraph(1);
$('#dag-fit').onclick=()=>{zoomGraph(Math.min($('#dag-viewport').clientWidth/recipeGraph.width,$('#dag-viewport').clientHeight/recipeGraph.height));$('#dag-viewport').scrollTo(0,0);};
$('#dag-all').onclick=()=>{recipeGraph.focus=false;recipeGraph.selected=null;$('#dag-search').value='';renderRecipes();};
$('#dag-focus').onclick=()=>{if(!recipeGraph.selected){toast('Select a stage first');return;}recipeGraph.focus=true;renderRecipes();$('#dag-viewport').scrollTo(0,0);};
$('#dag-search').oninput=renderRecipes; $('#dag-refresh').onclick=refreshRecipes;
$('#dag-viewport').addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();zoomGraph(recipeGraph.zoom*(e.deltaY<0?1.1:1/1.1));}},{passive:false});
let graphDrag=null;
$('#dag-viewport').onpointerdown=e=>{if(e.button!==0||e.target.closest('button,input,select,textarea,a'))return;const v=$('#dag-viewport');graphDrag={x:e.clientX,y:e.clientY,left:v.scrollLeft,top:v.scrollTop};v.setPointerCapture(e.pointerId);v.classList.add('dag-dragging');e.preventDefault();};
$('#dag-viewport').onpointermove=e=>{if(graphDrag){const v=$('#dag-viewport');v.scrollLeft=graphDrag.left+graphDrag.x-e.clientX;v.scrollTop=graphDrag.top+graphDrag.y-e.clientY;}};
$('#dag-viewport').onpointerup=$('#dag-viewport').onpointercancel=$('#dag-viewport').onlostpointercapture=()=>{graphDrag=null;$('#dag-viewport').classList.remove('dag-dragging');};
async function openRecipeStage(id) {
  recipeGraph.selected=id;renderRecipes();
  const node=recipeGraph.data.nodes.find(n=>n.id===id);if(!node)return;
  const ancestorIds=relatedStages(id,'ancestors'),descendantIds=relatedStages(id,'descendants');
  const stageLinks=ids=>recipeGraph.data.nodes.filter(n=>ids.has(n.id)).sort((a,b)=>a.layer-b.layer).map(n=>`<button class="small" data-lineage-stage="${esc(n.id)}">${esc(n.external?n.name:n.id)}</button>`).join(' → ');
  const edgeLines=recipeGraph.data.edges.filter(e=>ancestorIds.has(e.from)&&ancestorIds.has(e.to)).map(e=>`${e.from} → ${e.to} · checkpoint ${e.step??'base'}${e.ema?' EMA':' student'} · ${e.kind}${e.run?' · '+e.run:''}`).join('\n');
  openDrawer(`${node.external?'Source':id} · lineage / recipe`, `<div class="card"><h4>${esc(node.name)}</h4><button id="dag-focus-stage" class="small">Focus this lineage</button><h4>Ancestors → selected</h4><div class="chips">${stageLinks(ancestorIds)}</div><pre class="detail">${esc(edgeLines)}</pre><h4>Selected → descendants</h4><div class="chips">${stageLinks(descendantIds)}</div></div>` +
    (node.external?'':`<div class="card"><h4>Checkpoints · new weight-only branch</h4><p class="muted">Choose an exact Run + step. Planned targets can define future stages but cannot launch before their checkpoint is produced. This is not optimizer-state resume.</p><select id="dag-checkpoint">${node.checkpoints.map((c,i)=>`<option value="${i}">@${c.step} ${c.ema?'EMA':'student'} · ${c.available?(c.exported?'export':'DCP; export required'):'planned target'} · ${esc(c.run?.split('/').pop()||'not produced')}</option>`).join('')}</select><button id="dag-branch" class="primary" ${node.checkpoints.length?'':'disabled'}>New training stage</button></div><div class="card"><h4>Current recipe / run records</h4><div id="dag-recipe-detail" class="detail">Loading…</div></div>`));
  for(const b of $$('[data-lineage-stage]',$('#drawer')))b.onclick=()=>openRecipeStage(b.dataset.lineageStage);
  $('#dag-focus-stage').onclick=()=>{recipeGraph.focus=true;renderRecipes();$('#drawer').classList.add('hidden');$('#dag-viewport').scrollTo(0,0);};
  if(node.external)return;
  $('#dag-branch').onclick=()=>openBranchEditor(node,node.checkpoints[Number($('#dag-checkpoint').value)]);
  try{const d=await api(`/api/variant/${id}`);if(recipeGraph.selected===id&&$('#dag-recipe-detail'))$('#dag-recipe-detail').textContent=JSON.stringify({parameters:d.variant.parameters,provenance:d.variant.provenance,runs:node.runs},null,2);}catch(e){toast(e.message);}
}
async function openBranchEditor(node, checkpoint) {
  if(!checkpoint)return;
  openDrawer(`Branch from ${node.id}@${checkpoint.step}${checkpoint.ema?' EMA':''}`,`<form id="dag-branch-form" class="card">
    <p class="muted">Creates a PLANNED stage in E0029; no GPU launch. Parent identity is fixed below. Review parameters and launcher environment before saving.</p>
    <pre class="detail">${esc(JSON.stringify(checkpoint,null,2))}</pre>
    <div class="kv"><label>Training template</label><select name="clone_from">${recipeGraph.data.nodes.filter(n=>!n.external).map(n=>`<option value="${n.id}" ${n.id===node.id?'selected':''}>${n.id} · ${esc(n.method)} · ${esc(n.name)}</option>`).join('')}</select>
    <label>New stage name</label><input name="name" required value="${esc(node.id+'@'+checkpoint.step+' → new stage')}">
    <label>Description</label><textarea name="description" rows="2"></textarea>
    <label>Launcher entry</label><input name="entry" required></div>
    <label>Recipe parameters (JSON)</label><textarea name="parameters" class="dag-json" rows="15" required></textarea>
    <label>Launcher environment (JSON; review for this recipe)</label><textarea name="env" class="dag-json" rows="8" required></textarea>
    <p class="muted">Parent initializer / resume_step are set by the server. Raw DCP or future checkpoints remain blocked on export preparation. No checkpoint metadata is modified.</p>
    <div class="row"><button id="dag-preview" type="button">Preview stage</button><button type="submit" class="primary">Create PLANNED stage</button></div><pre id="dag-branch-result" class="detail"></pre></form>`);
  const form=$('#dag-branch-form'), f=form.elements;
  const fill=async()=>{try{const d=await api('/api/train/template?clone_from='+encodeURIComponent(f.clone_from.value));const v=d.template;if(!v)throw Error('Template missing');const p={...v.parameters},env={...(v.provenance?.env||{})};for(const k of ['initializer','resume_step','init_step','dense_qat_step','sparse_ft_step'])delete p[k];for(const k of ['RUN_NAME','ALLOCATION_ID','INITIALIZER','RESUME','RESUME_FROM','RESUME_FROM_CHECKPOINT'])delete env[k];f.parameters.value=JSON.stringify(p,null,2);f.env.value=JSON.stringify(env,null,2);f.entry.value=v.provenance?.entry||'';}catch(e){$('#dag-branch-result').textContent=e.message;}};
  f.clone_from.onchange=fill;await fill();
  const payload=()=>({source:{variant:node.id,step:checkpoint.step,ema:checkpoint.ema,run:checkpoint.run},clone_from:f.clone_from.value,name:f.name.value,description:f.description.value,entry:f.entry.value,parameters:JSON.parse(f.parameters.value),env:JSON.parse(f.env.value)});
  $('#dag-preview').onclick=async()=>{try{const d=await post('/api/recipes/branch',{...payload(),dry_run:true});$('#dag-branch-result').textContent=JSON.stringify(d.preview,null,2);}catch(e){$('#dag-branch-result').textContent=e.message;}};
  form.onsubmit=async e=>{e.preventDefault();if(!confirm('Create a new PLANNED E0029 stage from this checkpoint? No training will be launched.'))return;const submit=form.querySelector('[type=submit]');submit.disabled=true;try{const d=await post('/api/recipes/branch',payload());$('#dag-branch-result').textContent=JSON.stringify(d,null,2);toast(`Created ${d.id} · lint ${d.lint_rc}, journal ${d.journal_rc}`,8000);await loadOverview();await refreshRecipes();recipeGraph.selected=d.id;recipeGraph.focus=true;$('#dag-search').value='';renderRecipes();}catch(err){$('#dag-branch-result').textContent=err.message;submit.disabled=false;}};
}
