/* VSQA E0029/E0030 dashboard: one big Variant table with parameter families and a step timeline. */
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 4) => (v === null || v === undefined || v === '') ? '' : (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : String(+v.toFixed(d))) : (typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v)));
const badge = (s) => `<span class="badge st-${esc(s)}">${esc(s)}</span>`;
const toast = (msg, ms = 4000) => { const t = $('#toast'); t.textContent = msg; t.style.display = 'block'; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = 'none', ms); };
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) { /* ignore */ } throw new Error(`${path}: ${m}`); }
  return r.json();
}
const post = (path, body) => api(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body || {}) });
const store = (k, v) => localStorage.setItem(k, JSON.stringify(v));
const load = (k, d) => { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } };
// Display groups used to be `<kernel>×vbench[<subset>]`; VBench datapoints are now grouped by
// evaluation dataset only, so a persisted pinned/hidden group keeps its dataset and loses the
// kernel half of the name. Backends are provenance on each datapoint, never a group.
const OLD_VBENCH_KIND = /^(?:vsqa|dense-bf16)×vbench\[(.+)\]$/;
const datasetKind = (k) => { const m = OLD_VBENCH_KIND.exec(String(k ?? '')); return m ? `vbench[${m[1]}]` : k; };
function migrateKinds(v) {
  if (!Array.isArray(v)) return v;
  const out = [...new Set(v.map(datasetKind))];
  if (out.length !== v.length || out.some((k, i) => k !== v[i])) store('kinds', out);
  return out;
}
function migratePin(v) { const out = v ? datasetKind(v) : v; if (out !== v) store('pinKind', out); return out; }

// ------------------------------------------------------------------ state
const state = {
  rows: [], famSpec: [], dpKinds: {}, filterGroups: [], dpRefs: [],
  order: load('famOrder', null), collapsed: new Set(load('famCollapsed', ['model', 'cube', 'kernel', 'lr', 'schedule'])),
  selected: new Set(load('selected', [])), sortMode: load('sortMode', 'newest'),
  tlScale: load('tlScale', 300), maxSteps: 1000, videos: {}, scalars: {}, expanded: new Map(), kinds: migrateKinds(load('kinds', null)),
  pin: migratePin(load('pinKind', null)),
  hidePvScale: load('hidePvScale', false) === true,
  hideDmd8: load('hideDmd8', false) === true,
  hideDmd3: load('hideDmd3', false) === true,
  hideDmd4: load('hideDmd4', false) === true,
  hideNoFake: load('hideNoFake', false) === true,
  // One selected option per group, or absent for "no restriction from this group".
  picks: load('filterPicks', {}) || {},
};
const famByKey = () => Object.fromEntries(state.famSpec.map((f) => [f.key, f]));
function famOrder() {
  const keys = state.famSpec.map((f) => f.key);
  const o = (state.order || []).filter((k) => keys.includes(k));
  return [...o, ...keys.filter((k) => !o.includes(k))];
}
const saveSel = () => { store('selected', [...state.selected]); $('#selection-summary').textContent = state.selected.size ? `${state.selected.size} selected: ${[...state.selected].sort().join(', ')}` : 'no variants selected'; };
for (const [id, key] of [['hide-pv-scale', 'hidePvScale'], ['hide-dmd8', 'hideDmd8'], ['hide-dmd4', 'hideDmd4'], ['hide-dmd3', 'hideDmd3'], ['hide-no-fake', 'hideNoFake']]) {
  const control = $(`#${id}`);
  control.checked = state[key];
  control.onchange = () => { state[key] = control.checked; store(key, state[key]); renderTable(); };
}
function updatePinnedOffsets() {
  document.documentElement.style.setProperty('--viewport-width', `${document.documentElement.clientWidth}px`);
  document.documentElement.style.setProperty('--header-height', `${$('header').getBoundingClientRect().height}px`);
}
const headerResize = new ResizeObserver(updatePinnedOffsets);
headerResize.observe($('header'));
window.addEventListener('resize', updatePinnedOffsets);

// ------------------------------------------------------------------ data
async function loadOverview() {
  const d = await api('/api/overview');
  state.rows = d.variants; state.famSpec = d.families; state.dpKinds = d.datapoint_kinds;
  state.filterGroups = d.filter_groups || [];
  state.vbenchSubmission = d.vbench_submission === true;
  // Video manifests can change after a validation replay while this tab stays open.
  const openVideos = [...state.expanded.values()].filter((ex) => ex.videos);
  await Promise.all([...new Set(openVideos.map((ex) => ex.vid))].map(async (id) => {
    state.videos[id] = await api(`/api/videos/${id}`);
  }));
  for (const ex of openVideos) {
    const source = ex.dp.kind === 'inf-kernel×val12' ? 'inf_val' : /×val12@\d+step$/.test(ex.dp.kind) ? 'forced_val' : 'videos';
    ex.videos = state.videos[ex.vid][source][String(ex.dp.step)] || [];
  }
  state.maxSteps = Math.max(1000, ...state.rows.map((r) => Math.max(r.timeline.target || 0, ...r.timeline.segments.map((s) => s.end))));
  renderFilterGroups(); renderLegend(); renderTable();
}
function renderFilterGroups() {
  const host = $('#row-filters');
  host.innerHTML = state.filterGroups.map((g) => {
    const counts = state.rows.reduce((acc, r) => {
      const v = r.filters?.[g.key];
      if (v) acc[v] = (acc[v] || 0) + 1;
      return acc;
    }, {});
    const chips = g.options.map((o) => `<span class="chip ${state.picks[g.key] === o.value ? 'on' : ''}" data-group="${esc(g.key)}" data-value="${esc(o.value)}" title="Show only this category; click again to clear">${esc(o.label)} <span class="muted">${counts[o.value] || 0}</span></span>`).join('');
    return `<span class="filter-group"><span class="muted">${esc(g.label)}</span>${chips}</span>`;
  }).join('');
  // Clicking the selected option clears the group, so each group stays a 0-or-1 choice.
  $$('#row-filters .chip').forEach((c) => c.onclick = () => {
    const { group, value } = c.dataset;
    if (state.picks[group] === value) delete state.picks[group]; else state.picks[group] = value;
    store('filterPicks', state.picks); renderFilterGroups(); renderTable();
  });
}

// ------------------------------------------------------------------ table
function sortedRows() {
  const q = $('#filter').value.toLowerCase().split(/\s+/).filter(Boolean);
  let rows = state.rows.filter((r) => {
    if (state.hidePvScale && r.families.kernel?.expanded.pv_scale === true) return false;
    if (state.hideNoFake && r.families.kernel?.expanded.mm === 'native NVFP4·FP8') return false;
    const objective = r.families.objective?.expanded;
    if (state.hideDmd8 && objective?.objective === 'DMD-K8' && objective.K === 8) return false;
    if (state.hideDmd3 && objective?.objective === 'DMD-K3' && objective.K === 3) return false;
    if (state.hideDmd4 && objective?.objective === 'DMD-K4' && objective.K === 4) return false;
    // Selected filter groups intersect with each other and with the hide switches above.
    for (const [group, value] of Object.entries(state.picks)) if (r.filters?.[group] !== value) return false;
    const hay = `${r.id} ${r.name} ${r.status} ${r.resume.label} ${Object.values(r.families).map((f) => f.collapsed).join(' ')}`.toLowerCase();
    return q.every((t) => hay.includes(t));
  });
  const picked = Object.entries(state.picks).map(([g, v]) => v).join(' + ');
  $('#filter-summary').textContent = `showing ${rows.length} / ${state.rows.length} rows${picked ? ` · ${picked}` : ''}`;
  if (state.sortMode === 'newest') return rows.sort((a, b) => b.id.localeCompare(a.id));
  const order = famOrder();
  const compare = (a, b) => {
    if (Array.isArray(a) && Array.isArray(b)) {
      for (let i = 0; i < Math.min(a.length, b.length); i++) {
        const c = compare(a[i], b[i]); if (c) return c;
      }
      return a.length - b.length;
    }
    return String(a ?? '').localeCompare(String(b ?? ''), undefined, { numeric: true });
  };
  // Build lineage keys from all rows so hiding a parent does not reorder its children.
  const byId = new Map(state.rows.map((r) => [r.id, r]));
  const keys = new Map(), visiting = new Set();
  const key = (r) => {
    if (keys.has(r.id)) return keys.get(r.id);
    const source = r.resume || {};
    let group;
    if (source.kind === 'base') group = [0, 0, source.label];
    else if (source.kind === 'variant') {
      const parent = byId.get(source.variant);
      const parentKey = parent && !visiting.has(parent.id) && parent.id !== r.id
        ? (visiting.add(r.id), key(parent)) : [[0, 2, source.variant], [], source.variant];
      group = [parentKey[0][0] + 1, 0, parentKey, source.step ?? -1, !!source.ema];
    } else group = [0, 1, source.label || '', source.step ?? -1];
    visiting.delete(r.id);
    const result = [group, order.map((k) => String(r.families[k].collapsed ?? '')), r.id];
    keys.set(r.id, result);
    return result;
  };
  for (const r of state.rows) key(r);
  return rows.sort((a, b) => compare(keys.get(a.id), keys.get(b.id)));
}
function renderTable() {
  const order = famOrder(); const spec = famByKey();
  const tlW = tlWidth(); const pin = pinnedKind();
  let h1 = `<tr class="fam"><th rowspan="2" class="sticky-l"><input type="checkbox" id="sel-all" title="select all shown"></th><th rowspan="2" class="sticky-l2">Variant</th><th rowspan="2">Resume from</th>`;
  let h2 = '<tr class="sub">';
  for (const k of order) {
    const f = spec[k]; const col = state.collapsed.has(k);
    h1 += `<th class="fam-head" draggable="true" data-fam="${k}" colspan="${col ? 1 : f.columns.length}" title="click: ${col ? 'expand' : 'collapse'} · drag: reorder">${esc(f.label)} <span class="muted">${col ? '▸' : '▾'}</span></th>`;
    h2 += col ? '<th></th>' : f.columns.map((c) => `<th>${esc(c.label)}</th>`).join('');
  }
  h1 += `<th rowspan="2" class="tl" style="min-width:${tlW + 20}px">${tlAxis(tlW)}</th>`;
  if (pin) h1 += `<th rowspan="2" class="pin sticky-r">${pinHeadHtml(pin)}</th>`;
  h1 += '</tr>';
  h2 += '</tr>';
  const rows = sortedRows();
  state.dpRefs = [];
  const ncols = 3 + order.reduce((n, k) => n + (state.collapsed.has(k) ? 1 : spec[k].columns.length), 0) + 1 + (pin ? 1 : 0);
  document.body.classList.toggle('pinned', !!pin);
  const body = rows.map((r) => {
    const isDmd = r.families.objective?.expanded.objective?.startsWith('DMD-');
    const eb = r.families.batch?.expanded.effective_batch;
    const steps = r.families.schedule?.expanded.steps;
    const tint = !isDmd ? '' : eb === 4 && steps === 500 ? 'dmd-green ' : eb === 16 && steps === 1000 ? 'dmd-blue ' : '';
    let cells = `<td class="sticky-l"><input type="checkbox" data-sel="${r.id}" ${state.selected.has(r.id) ? 'checked' : ''}></td>`;
    cells += `<td class="id sticky-l2"><div><a data-detail="${r.id}">${r.id}</a>${r.warnings.length ? `<span class="wicon" title="${esc(r.warnings.join('\n'))}">⚠</span>` : ''} ${statusCell(r)}</div><span class="name ${badKernel(r) ? 'bad-kernel' : ''}" title="${badKernel(r) ? 'PV scale trick kernel: training/inference mismatch\n\n' : ''}${esc(r.name)}\n\n${esc(r.description || '')}">${esc(r.name)}</span></td>`;
    cells += `<td>${resumeCell(r)}</td>`;
    for (const k of order) {
      const f = r.families[k]; const col = state.collapsed.has(k);
      const bad = k === 'kernel' && badKernel(r);
      const cls = (f.error ? 'err' : (f.warning ? 'warn' : '')) + (bad ? ' bad-kernel' : '');
      const title = f.error || f.warning || (bad ? 'PV scale trick kernel: training/inference mismatch (wrong training kernel)' : '');
      if (col) cells += `<td class="${cls}" title="${esc(title)}">${esc(f.collapsed ?? '')}</td>`;
      else cells += spec[k].columns.map((c) => `<td class="${cls} ${typeof f.expanded[c.key] === 'number' ? 'num' : ''}" title="${esc(title)}">${esc(fmt(f.expanded[c.key]))}</td>`).join('');
    }
    cells += `<td class="tl">${tlRow(r, tlW)}</td>`;
    if (pin) cells += pinCell(r, pin);
    let html = `<tr data-id="${r.id}" class="${tint}${state.selected.has(r.id) ? 'sel' : ''}">${cells}</tr>`;
    for (const [key, ex] of state.expanded) if (ex.vid === r.id) html += `<tr class="expand" data-key="${esc(key)}"><td class="sticky-l ex-step"><a data-close-ex="${esc(key)}" title="${esc(ex.dp.kind)}${ex.dp.eval ? ' · ' + ex.dp.eval : ''}${ex.dp.run ? '\n' + esc(ex.dp.run) : ''}\nclick to close">${ex.dp.step}</a></td><td colspan="${ncols - 1 - (pin ? 1 : 0)}"><div class="ex-wrap">${expansionHtml(ex)}</div></td>${pin ? '<td class="pin sticky-r"></td>' : ''}</tr>`;
    return html;
  }).join('');
  $('#grid').innerHTML = `<thead>${h1}${h2}</thead><tbody>${body}</tbody>`;
  bindTable(rows);
  $('#btn-warnings').textContent = `warnings (${state.rows.reduce((n, r) => n + r.warnings.length, 0)})`;
  updatePinnedOffsets();
}
const badKernel = (r) => r.families.kernel && r.families.kernel.expanded.pv_scale === true;
function resumeCell(r) {
  const s = r.resume;
  const kids = r.children.length ? ` <span class="muted" title="${esc(r.children.map((c) => `${c.id} @${c.step}`).join('\n'))}">→${r.children.length}</span>` : '';
  if (s.kind === 'variant') return `<a data-jump="${s.variant}" title="jump to ${s.variant}; run ${esc(s.run || '')}">${esc(s.label)}</a>${kids}`;
  if (s.kind === 'base') return `<span>${esc(s.label)}</span>${kids}`;
  return `<span class="muted" title="${esc(s.label)}">${esc(s.label.replace('logs/', '').slice(0, 30))}</span>${kids}`;
}
function averageStep(run) {
  const p = run && run.progress;
  const value = p && p.avg_sec_per_step;
  const title = value == null ? 'No measured step interval in run.log' : `${run.id}: ${p.timed_seconds}s / ${p.timed_steps} observed steps; includes validation/checkpoint waits within the logged interval; excludes unobserved startup and steps before resume.`;
  return ` <span class="muted step-average" title="${esc(title)}">avg ${value == null ? '—' : value.toFixed(2) + ' s/step'}</span>`;
}
function statusCell(r) {
  const live = r.run; const p = live && live.progress;
  const stale = live && live.state === 'RUNNING' && live.log_mtime && (Date.now() / 1000 - live.log_mtime) > Math.max(900, 6 * ((p && p.sec_per_it) || 0));
  const wb = r.wandb.url ? ` <a href="${esc(r.wandb.url)}" target="_blank" title="W&B ${esc(r.wandb.state || '')} @${r.wandb.summary_step ?? '?'}">W&B</a>` : '';
  const runs = r.run_rows.length > 1 ? ` <span class="muted" title="${esc(r.run_rows.map((x) => `${x.id} ${x.state}`).join('\n'))}">${r.run_rows.length} runs</span>` : '';
  return `${badge(r.status)}${live && live.state !== r.status ? ' ' + badge(live.state) : ''}${p && live.state === 'RUNNING' ? ` <span class="muted">${p.sec_per_it}s/it ETA ${esc(p.eta)}</span>` : ''}${stale ? ' <span class="wicon" title="run.log not updated for >max(15 min, 6 steps)">stale</span>' : ''}${wb}${averageStep(live)}${runs}`;
}
function bindTable(rows) {
  const tbl = $('#grid');
  $$('input[data-sel]', tbl).forEach((cb) => cb.onchange = () => { cb.checked ? state.selected.add(cb.dataset.sel) : state.selected.delete(cb.dataset.sel); saveSel(); cb.closest('tr').classList.toggle('sel', cb.checked); });
  $('#sel-all').onchange = (e) => { rows.forEach((r) => e.target.checked ? state.selected.add(r.id) : state.selected.delete(r.id)); saveSel(); renderTable(); };
  $$('a[data-detail]', tbl).forEach((a) => a.onclick = () => showVariantDetail(a.dataset.detail));
  $$('a[data-jump]', tbl).forEach((a) => a.onclick = () => jumpTo(a.dataset.jump));
  $$('th.fam-head', tbl).forEach((th) => {
    th.onclick = () => { const k = th.dataset.fam; state.collapsed.has(k) ? state.collapsed.delete(k) : state.collapsed.add(k); store('famCollapsed', [...state.collapsed]); renderTable(); };
    th.ondragstart = (e) => { e.dataTransfer.setData('text/plain', th.dataset.fam); th.classList.add('dragging'); };
    th.ondragend = () => th.classList.remove('dragging');
    th.ondragover = (e) => { e.preventDefault(); th.classList.add('drop-target'); };
    th.ondragleave = () => th.classList.remove('drop-target');
    th.ondrop = (e) => { e.preventDefault(); th.classList.remove('drop-target'); const from = e.dataTransfer.getData('text/plain'), to = th.dataset.fam; if (!from || from === to) return; const o = famOrder().filter((k) => k !== from); o.splice(o.indexOf(to), 0, from); state.order = o; store('famOrder', o); renderTable(); };
  });
  $$('a[data-close-ex]', tbl).forEach((a) => a.onclick = () => { state.expanded.delete(a.dataset.closeEx); renderTable(); });
  const unpin = $('#pin-clear', tbl); if (unpin) unpin.onclick = (e) => { e.stopPropagation(); setPin(state.pin); };
}
// Datapoints carry only an index into state.dpRefs (rebuilt by every renderTable); one delegated
// listener replaces a handler and an inline JSON copy per marker, which dominated the table's size.
function dpAttrs(vid, d) {
  state.dpRefs.push({ vid, d });
  return `data-i="${state.dpRefs.length - 1}"`;
}
$('#grid').addEventListener('click', (e) => {
  const el = e.target.closest('.dp[data-i], .pin-dp[data-i]'); if (!el) return;
  const ref = state.dpRefs[+el.dataset.i]; if (!ref) return;
  e.stopPropagation(); onDatapoint(ref.vid, structuredClone(ref.d), e.altKey);
});
function jumpTo(id) {
  const tr = $(`tr[data-id="${id}"]`); if (!tr) { toast(`${id} is not in the table (filtered or smoke)`); return; }
  tr.scrollIntoView({ block: 'center' }); tr.style.outline = '2px solid var(--accent)'; setTimeout(() => tr.style.outline = '', 1800);
}
$('#filter').oninput = renderTable;
$('#sort-mode').value = state.sortMode; $('#sort-mode').onchange = () => { state.sortMode = $('#sort-mode').value; store('sortMode', state.sortMode); renderTable(); };

// ------------------------------------------------------------------ timeline
const TL_PAD = 8;
function visibleKinds() { const all = Object.keys(state.dpKinds); const sel = state.kinds ? all.filter((k) => state.kinds.includes(k)) : all; return sel; }
function laneOf() { const m = {}; visibleKinds().forEach((k, i) => m[k] = 9 + i * 13); return m; }
function tlHeight() { return Math.max(30, 6 + visibleKinds().length * 13 + 6); }
function tlWidth() { return Math.round(state.tlScale * state.maxSteps / 1000); }
const sx = (step, w) => TL_PAD + (w * step / state.maxSteps);
function tlAxis(w) {
  const step = niceStep(state.maxSteps / Math.max(2, w / 90));
  let s = `<svg class="tl" width="${w + 2 * TL_PAD}" height="18"><g class="axis">`;
  for (let v = 0; v <= state.maxSteps; v += step) s += `<line x1="${sx(v, w)}" y1="12" x2="${sx(v, w)}" y2="17"/><text x="${sx(v, w)}" y="10" text-anchor="middle">${v}</text>`;
  return s + '</g></svg>';
}
function niceStep(raw) { const p = Math.pow(10, Math.floor(Math.log10(raw))); const m = raw / p; return (m < 1.5 ? 1 : m < 3.5 ? 2.5 : m < 7.5 ? 5 : 10) * p; }
function marker(kind, x, y, cls, extra = '', title = '') {
  const k = state.dpKinds[kind] || { marker: 'circle', color: '#e5646a' }; const r = 4.5;
  const shapes = {
    circle: ['circle', `cx="${x}" cy="${y}" r="${r}"`], square: ['rect', `x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}"`],
    diamond: ['polygon', `points="${x},${y - r - 1} ${x + r + 1},${y} ${x},${y + r + 1} ${x - r - 1},${y}"`],
    triangle: ['polygon', `points="${x},${y - r - 1} ${x + r + 1},${y + r} ${x - r - 1},${y + r}"`],
  };
  const [tag, geom] = shapes[k.marker] || shapes.circle;
  const stroke = cls.includes('EVALUABLE') || cls.includes('QUEUED') ? ` style="stroke:${k.color}"` : '';
  if (!extra) return `<${tag} ${geom} class="dp ${cls}" fill="${k.color}"${stroke}>${title ? `<title>${esc(title)}</title>` : ''}</${tag}>`;
  const shape = `<${tag} ${geom} class="dp ${cls}" fill="${k.color}"${stroke}/>`;
  // Invisible 12x12 hit pad on top of the marker so dashed/hollow points are as easy to click as filled ones.
  return `<g class="dpg">${shape}<rect x="${x - 6}" y="${y - 6}" width="12" height="12" fill="transparent" class="dp hit ${cls}" ${extra}>${title ? `<title>${esc(title)}</title>` : ''}</rect></g>`;
}
// One expansion per datapoint. Two checkpoint-addressed evaluations of the same checkpoint on the
// same dataset (1 seed and 5 seeds) carry no eval id, so the result path is what separates them.
function dpKey(vid, d) {
  return `${vid}|${d.kind}|${d.step}|${d.eval || d.result || ''}`;
}

function tlRow(r, w) {
  const t = r.timeline; const lanes = laneOf(); const H = tlHeight(); const yMid = H / 2;
  let s = `<svg class="tl" width="${w + 2 * TL_PAD}" height="${H}">`;
  s += `<line class="target" x1="${sx(0, w)}" y1="${yMid}" x2="${sx(t.target || 0, w)}" y2="${yMid}"/>`;
  for (const seg of t.segments) s += `<line class="seg ${esc(seg.state)}${seg.attempt ? ' attempt' : ''}" x1="${sx(seg.start, w)}" y1="${yMid}" x2="${sx(Math.max(seg.end, seg.start + 1), w)}" y2="${yMid}"><title>${esc(seg.run)}\n${esc(seg.state)} ${seg.start}→${seg.end}</title></line>`;
  for (const c of t.checkpoints) s += `<line class="ckpt" x1="${sx(c, w)}" y1="${yMid - 5}" x2="${sx(c, w)}" y2="${yMid + 5}"><title>checkpoint-${c}</title></line>`;
  for (const e of t.exports) s += `<rect class="export" x="${sx(e.step, w) - 1.5}" y="${yMid - 6}" width="3" height="12"><title>export-step${e.step}${e.ema ? ' (+ema)' : ''}</title></rect>`;
  // A dataset lane can hold several evaluations of one checkpoint (different attention forward, or a
  // re-run): they stay separate datapoints, fanned out horizontally so each one is visible and clickable.
  const seen = new Map();
  for (const d of t.datapoints) if (d.kind in lanes) seen.set(`${d.kind}|${d.step}`, (seen.get(`${d.kind}|${d.step}`) || 0) + 1);
  const shown = new Map();
  for (const d of t.datapoints) {
    if (!(d.kind in lanes)) continue;
    const slot = `${d.kind}|${d.step}`; const n = seen.get(slot); const i = shown.get(slot) || 0; shown.set(slot, i + 1);
    // Each marker owns a 12 px transparent hit rect, so siblings must be at least that far apart
    // or the neighbour wins the click and opens the wrong evaluation.
    const dx = n > 1 ? (i - (n - 1) / 2) * 13 : 0;
    const y = lanes[d.kind]; const cls = `${d.status}${d.error ? ' error' : ''}${d.requeued ? ' requeued' : ''}${state.expanded.has(dpKey(r.id, d)) ? ' open' : ''}`;
    const queueable = state.dpKinds[d.kind]?.source === 'inf_val' || state.vbenchSubmission;
    const hint = d.status === 'EVALUABLE' ? (!queueable ? 'not evaluated (VBench submission from the dashboard is disabled)' : d.needs_export ? 'click: queue (DCP checkpoint will be exported first)' : 'click: queue') : d.status === 'QUEUED' ? `queued #${d.queue_id} · click: cancel` : d.status === 'RUNNING' && d.queue_id ? `running #${d.queue_id}` : (d.kind === 'inf-kernel×val12' || (d.eval && state.vbenchSubmission)) ? 'click: open · alt+click: re-run & overwrite' : 'click: open';
    // The forward is provenance, not part of the group: the tooltip names it on every datapoint.
    const prov = d.attention_backend ? `\nforward: ${d.attention_backend}` : d.forward ? `\nforward: ${d.forward}` : '';
    const tip = d.error ? d.error : `${d.kind} @${d.step} · ${d.status}${d.backend ? ' · ' + d.backend : ''}${d.count != null ? ` ${d.count}/${d.expected}` : ''}${d.eval ? ' · ' + d.eval : ''}${d.quality_score != null ? ' · Q ' + d.quality_score.toFixed(4) : ''}${d.requeued ? ` · re-render ${d.requeued} #${d.queue_id}` : ''}${n > 1 ? `\n${n} evaluations of this checkpoint on this dataset` : ''}${prov}\n${hint}`;
    s += marker(d.kind, sx(d.step, w) + dx, y, cls, dpAttrs(r.id, d), tip);
  }
  return s + '</svg>';
}
function renderLegend() {
  const vis = visibleKinds();
  const chip = ([k, v]) => {
    const lane = vis.includes(k); const pinned = state.pin === k;
    // The chip body keeps its original meaning (show/hide the timeline lane); pinning is a separate
    // explicit button so that asking for scores never silently hides the lane and vice versa.
    return `<span class="chip ${lane ? 'on' : ''}${pinned ? ' pinned' : ''}" data-kind="${esc(k)}" title="${esc(v.label)} — click the chip to show/hide this timeline lane">`
      + `<svg width="12" height="12" style="vertical-align:-2px">${marker(k, 6, 6, 'COMPLETED', '')}</svg> ${esc(k)}`
      + (v.source !== 'e0030' ? '' : `<button class="pin-btn${pinned ? ' on' : ''}" data-pin="${esc(k)}" title="${pinned ? 'unpin' : 'pin'} the VBench scores of every evaluation on ${esc(k)} as a column group at the right edge of the table (the timeline lane is not touched)">${pinned ? 'unpin' : 'pin'}</button>`)
      + '</span>';
  };
  $('#legend').innerHTML = Object.entries(state.dpKinds).map(chip).join('') + '<span class="chip" title="export exists but this kind has not been produced: click to queue">dashed = evaluable</span>' + (state.vbenchSubmission ? '<button class="small" id="new-subset" title="plan a new VBench evaluation dataset or forward (E0030 Variant)">+ VBench evaluation</button>' : '');
  $$('#legend .chip[data-kind]').forEach((c) => c.onclick = () => { const k = c.dataset.kind; const cur = visibleKinds(); state.kinds = cur.includes(k) ? cur.filter((x) => x !== k) : [...cur, k]; store('kinds', state.kinds); renderLegend(); renderTable(); });
  $$('#legend button[data-pin]').forEach((b) => b.onclick = (e) => { e.stopPropagation(); setPin(b.dataset.pin); });
  if ($('#new-subset')) $('#new-subset').onclick = async () => { await openQueueDrawer(); $('#ev-form').scrollIntoView({ block: 'start' }); };
}

// --------------------------------------------------- pinned VBench dataset score columns
// One pinned evaluation dataset at a time: every evaluation of every checkpoint of the Variant on
// that dataset is stacked inside a single sticky cell at the right edge of the row (whatever forward
// produced it, one line each — never averaged), so the row keeps one table-row height and the
// timeline can be scrolled horizontally with the scores staying in view.
const VB_COLS = [
  ['imaging_quality', 'IQ', 'imaging quality'],
  ['aesthetic_quality', 'AQ', 'aesthetic quality'],
  ['subject_consistency', 'SC', 'subject consistency'],
  ['background_consistency', 'BC', 'background consistency'],
  ['temporal_flickering', 'TF', 'temporal flickering'],
  ['motion_smoothness', 'MS', 'motion smoothness'],
  ['dynamic_degree', 'DD', 'dynamic degree'],
  ['quality_score', 'OQ', 'overall quality score — VBench weighted, normalized aggregate of the seven quality dimensions (not their arithmetic mean)'],
];
const OQ_KEY = 'quality_score';
// Placeholder datapoints (EVALUABLE) exist for every export step and carry no result: they are not
// part of the score history, the queued/planned/running/failed ones are.
const PIN_STATUS = new Set(['COMPLETED', 'PARTIAL', 'RUNNING', 'QUEUED', 'PLANNED', 'PENDING', 'FAILED']);
function pinnedKind() { return state.pin && state.dpKinds[state.pin]?.source === 'e0030' ? state.pin : null; }
function setPin(kind) {
  state.pin = state.pin === kind ? null : kind;
  store('pinKind', state.pin);
  renderLegend(); renderTable();
  toast(state.pin ? `pinned the ${state.pin} scores (every forward) at the right edge of the table` : 'unpinned the score columns');
}
// '0.839623 ± 0.008752' → {mean, std}; a bare number (older result) → {mean, std: null}; nothing → null.
function parseStat(metrics, key) {
  const m = metrics || {};
  for (const raw of [m[`${key}_summary`], m[key]]) {
    if (typeof raw === 'number') { if (Number.isFinite(raw)) return { mean: raw, std: null }; continue; }
    if (typeof raw !== 'string' || !raw.trim()) continue;
    // '0.839623 ± 0.008752', '0.800000 ± —' (a single eligible seed: mean without uncertainty), or a bare number.
    const hit = raw.match(/^\s*([-+0-9.eE]+)\s*(?:(?:±|\+\/-)\s*(\S+))?/);
    if (hit && Number.isFinite(Number(hit[1]))) return { mean: Number(hit[1]), std: Number.isFinite(Number(hit[2])) ? Number(hit[2]) : null };
  }
  return null;
}
function dpStat(d, key) {
  const stat = parseStat(d.metrics, key);
  if (stat) return stat;
  if (key === OQ_KEY && typeof d.quality_score === 'number' && Number.isFinite(d.quality_score)) return { mean: d.quality_score, std: null };
  return null;
}
function statHtml(stat, key, full) {
  if (!stat) return `<span class="pin-v na" title="${esc(full)}: not recorded for this checkpoint">—</span>`;
  const std = stat.std == null
    ? '<i class="na" title="no standard deviation recorded (single eligible seed, or a result produced before per-seed dispersion was stored)">±—</i>'
    : `<i>±${stat.std.toFixed(4).replace(/^0\./, '.')}</i>`;
  const tip = `${full}: ${stat.mean}${stat.std == null ? '\nno standard deviation recorded' : `\n± ${stat.std}`}`;
  return `<span class="pin-v${key === OQ_KEY ? ' oq' : ''}" title="${esc(tip)}">${stat.mean.toFixed(4)}${std}</span>`;
}
function pinHeadHtml(kind) {
  const spec = state.dpKinds[kind] || {};
  const heads = VB_COLS.map(([key, ab, full]) => `<span class="pin-h${key === OQ_KEY ? ' oq' : ''}" title="${esc(full)}">${ab}</span>`).join('');
  return `<div class="pin-title"><svg width="12" height="12" style="vertical-align:-2px">${marker(kind, 6, 6, 'COMPLETED', '')}</svg>`
    + `<span title="${esc(spec.label || kind)}">${esc(kind)}</span>`
    + `<button class="small" id="pin-clear" title="unpin these score columns (the timeline lane stays as it is)">unpin</button></div>`
    + `<div class="pin-grid pin-hrow"><span class="pin-step" title="training step of the evaluated checkpoint; execution provenance is available on hover and in details">step</span>${heads}</div>`;
}
function pinCell(r, kind) {
  // Several evaluations of one checkpoint on this dataset (different forward, or a re-run) are
  // several lines: sorted by step, stable within a step, each keeping its own E0030 identity.
  const dps = r.timeline.datapoints.filter((d) => d.kind === kind && PIN_STATUS.has(d.status)).sort((a, b) => a.step - b.step);
  if (!dps.length) return '<td class="pin sticky-r"><div class="pin-body"><span class="pin-none muted">no evaluation</span></div></td>';
  const rows = dps.map((d) => {
    const attrs = dpAttrs(r.id, d);
    const stats = VB_COLS.map(([key, , full]) => [key, full, dpStat(d, key)]);
    const prov = d.attention_backend || d.backend || d.forward || '';
    const step = `<span class="pin-step" title="${esc(`${kind} @${d.step} · ${d.status}${d.eval ? ' · ' + d.eval : ''}${prov ? `\nforward: ${prov}` : ''}\nclick to open this datapoint`)}">${d.step}</span>`;
    // A finished datapoint without metrics (e.g. rendered inf_val videos) is not the same as one still to come.
    const note = d.status === 'COMPLETED' || d.status === 'PARTIAL'
      ? `${d.count != null ? `${d.count}/${d.expected} videos · ` : ''}no VBench metrics`
      : d.error ? 'failed' : 'no score yet';
    if (!stats.some(([, , s]) => s)) return `<div class="pin-row pin-grid pin-dp" ${attrs}>${step}<span class="pin-note">${badge(d.status)} <span class="muted">${esc(note)}</span></span></div>`;
    return `<div class="pin-row pin-grid pin-dp" ${attrs}>${step}${stats.map(([key, full, s]) => statHtml(s, key, full)).join('')}</div>`;
  }).join('');
  return `<td class="pin sticky-r"><div class="pin-body">${rows}</div></td>`;
}
$('#tl-scale').value = state.tlScale;
$('#tl-scale').oninput = () => { state.tlScale = parseInt($('#tl-scale').value, 10); store('tlScale', state.tlScale); renderTable(); };
$('#tl-fit').onclick = () => { const th = $('th.tl'); const avail = Math.max(200, document.documentElement.clientWidth - (th ? th.getBoundingClientRect().left : 900) - 120); state.tlScale = Math.max(40, Math.floor(avail * 1000 / state.maxSteps)); $('#tl-scale').value = state.tlScale; store('tlScale', state.tlScale); renderTable(); };

// ------------------------------------------------------------------ datapoint / detail panels
function openBottom(title, html) { $('#bottom-title').innerHTML = title; $('#bottom-body').innerHTML = html; $('#bottom').classList.remove('hidden'); }
function openDrawer(title, html) { $('#drawer-title').innerHTML = title; $('#drawer-body').innerHTML = html; $('#drawer').classList.remove('hidden'); }
$$('button[data-close]').forEach((b) => b.onclick = () => $(`#${b.dataset.close}`).classList.add('hidden'));
const DIMS7 = ['imaging_quality', 'aesthetic_quality', 'subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree'];
const previewHtml = (src) => `<img src="${esc(src.replace('/media/', '/thumbnail/'))}" alt="视频第一帧" loading="lazy" decoding="async">`;
const videoEl = (src) => `<div class="video-slot" data-src="${esc(src)}">${previewHtml(src)}</div><a class="video-download" href="${esc(src)}" download>下载 MP4</a>`;
const validationProtocol = (x) => x.cfg != null && x.shift != null ? `<div class="muted">重生成 · ${esc(x.inference_steps)}步 / CFG ${esc(x.cfg)} / shift ${esc(x.shift)} / seed ${esc(x.seed)}</div>` : '';
const videoRow = (content, layout = 'strip') => `<div class="playback-row" data-video-row><button type="button" class="row-play" aria-pressed="false">播放此行</button><div class="${layout}">${content}</div></div>`;
// Patch one datapoint in the cached rows and re-render at once; the next loadOverview reconciles with the server.
function patchDp(d, changes) {
  for (const r of state.rows) for (const x of r.timeline.datapoints) {
    if (x.kind === d.kind && x.step === d.step && (x.eval || '') === (d.eval || '') && (x.ref || '') === (d.ref || '')) Object.assign(x, changes);
  }
  renderTable();
}
async function queueFor(d, overwrite) {
  const spec = state.dpKinds[d.kind] || {};
  if (spec.source === 'inf_val') {
    if (overwrite) patchDp(d, { requeued: 'queued' }); else patchDp(d, { status: 'QUEUED', queue_id: '…' });
    try { const r = await post('/api/inf_val/enqueue', { ref: d.ref, overwrite }); toast(`queued inf_val #${r.queue_id} → ${r.out_dir}`); patchDp(d, { queue_id: r.queue_id }); loadOverview(); }
    catch (e) { toast(e.message, 8000); patchDp(d, overwrite ? { requeued: null } : { status: 'EVALUABLE', queue_id: null }); }
    return;
  }
  if (!state.vbenchSubmission) { toast('VBench submission from the dashboard is disabled; the training session runs evaluations'); return; }
  if (d.eval && overwrite) {
    try { const r = await post(`/api/queue/enqueue/${d.eval}`); toast(`re-queued ${d.eval} as #${r.queue_id}`); loadOverview(); } catch (e) { toast(e.message, 8000); }
    return;
  }
  await openQueueDrawer(); prefillEvalForm(d.ref, spec);
}
async function onDatapoint(vid, d, alt) {
  if (d.status === 'QUEUED' && d.queue_id) {
    if (!confirm(`cancel queued job #${d.queue_id}?`)) return;
    patchDp(d, { status: 'EVALUABLE', queue_id: null });
    try { await post(`/api/queue/${d.queue_id}/cancel`); } catch (e) { toast(e.message); }
    loadOverview(); return;
  }
  if (d.status === 'EVALUABLE') { await queueFor(d, false); return; }
  if (alt && (d.kind === 'inf-kernel×val12' || d.eval)) { if (confirm(`re-run ${d.kind} @${d.step} and overwrite the existing result?`)) await queueFor(d, true); return; }
  if (d.error) { toast(d.error); return; }
  const key = dpKey(vid, d);
  if (state.expanded.has(key)) { state.expanded.delete(key); renderTable(); return; }
  const ex = { vid, dp: d, loading: true };
  state.expanded.set(key, ex); renderTable();
  try {
    if (d.kind === 'inf-kernel×val12' && !d.eval) {
      state.videos[vid] = await api(`/api/videos/${vid}`);
      ex.videos = state.videos[vid].inf_val[String(d.step)] || [];
    } else if (d.kind.endsWith('×val12') && !d.eval) {
      state.videos[vid] = await api(`/api/videos/${vid}`);
      ex.videos = state.videos[vid].videos[String(d.step)] || [];
    } else if (/×val12@\d+step$/.test(d.kind) && !d.eval) {
      state.videos[vid] = await api(`/api/videos/${vid}`);
      ex.videos = state.videos[vid].forced_val[String(d.step)] || [];
    } else if (d.analysis) {
      if (!state.analysis) state.analysis = {};
      if (!state.analysis[vid]) state.analysis[vid] = await api(`/api/analysis/${vid}`);
      const all = state.analysis[vid].steps[String(d.step)] || [];
      // Several evaluations of one checkpoint can share a lane (1 seed vs 5 seeds); the datapoint's
      // own video count identifies which one was clicked.
      ex.analysis = all.find((x) => x.videos_scored === d.count) || all[0];
    } else if (d.eval) {
      ex.perVideo = await api(`/api/eval/per_video/${d.eval}`);
    }
  } catch (e) { ex.error = e.message; }
  ex.loading = false;
  if (state.expanded.has(key)) renderTable();
}
function expansionHtml(ex) {
  const d = ex.dp; const head = '';
  if (ex.loading) return head + '<div class="muted">loading…</div>';
  if (ex.error) return head + `<div class="warnbox">${esc(ex.error)}</div>`;
  if (ex.videos) {
    return head + videoRow(ex.videos.map((x) => `<div class="cell"><div class="cap" title="${esc(x.caption || '')}${x.extra ? `\n补测 caption #${x.caption_idx}（${esc(x.backend || 'training kernel')}）` : ''}">${x.extra ? '<span class="chip on">补</span> ' : ''}#${x.caption_idx} ${esc((x.caption || '').slice(0, 40))}</div>${validationProtocol(x)}${videoEl(`/media/${x.run}/${x.file}`)}</div>`).join('') || '<span class="muted">no videos found</span>');
  }
  if (ex.analysis) {
    const a = ex.analysis; const m = a.dimension_means || {}; const cnt = a.dimension_counts || {};
    const per = a.per_seed || {}; const seeds = a.seeds || [];
    const head2 = `<table class="mini" style="width:auto"><tr>${DIMS7.map((x) => `<th>${x.replace('_', ' ')}</th>`).join('')}<th>Quality</th><th>videos</th></tr>` +
      `<tr>${DIMS7.map((x) => `<td class="num" title="${cnt[x] != null ? cnt[x] + ' videos' : ''}">${m[x] != null ? m[x].toFixed(4) : ''}</td>`).join('')}` +
      `<td class="num"><b>${a.quality_score.toFixed(4)}</b></td><td class="num">${a.videos_scored}</td></tr></table>`;
    // Several seeds: the per-seed spread is the only honest uncertainty this evaluation has.
    const spread = seeds.length > 1
      ? `<table class="mini" style="width:auto"><tr><th>seed</th>${seeds.map((s) => `<th>${s}</th>`).join('')}</tr>` +
        `<tr><td>Quality</td>${seeds.map((s) => `<td class="num">${(per[String(s)] || {}).quality_score != null ? per[String(s)].quality_score.toFixed(4) : '—'}</td>`).join('')}</tr></table>`
      : '';
    return head + head2 + spread +
      `<div class="detail">${esc([`forward: ${a.attention_backend}`,
        `cube: ${JSON.stringify(a.cube_shape)}  sparsity: ${a.sparsity}  linear: ${a.linear_quantization || 'BF16'}`,
        `sampling: ${a.sampling_steps} steps  cfg ${a.cfg}  ladder ${JSON.stringify(a.timesteps)}`,
        `geometry: ${a.geometry}  prompt set: ${a.prompt_set}  seeds: ${JSON.stringify(seeds)}`,
        `result: ${a.path}`].join('\n'))}</div>`;
  }
  if (ex.perVideo) {
    const m = d.metrics || {}; const pv = ex.perVideo; const scores = Object.entries(pv.scores || {});
    const table = `<table class="mini" style="width:auto"><tr>${DIMS7.map((x) => `<th>${x.replace('_', ' ')}</th>`).join('')}<th>Quality</th><th>videos</th></tr><tr>${DIMS7.map((x) => `<td class="num">${m[x] != null ? m[x].toFixed(4) : ''}</td>`).join('')}<td class="num"><b>${m.quality_score != null ? m.quality_score.toFixed(4) : '—'}</b></td><td class="num">${m.videos_scored ?? ''}</td></tr></table>`;
    if (!scores.length) return head + table + `<div class="muted">no per-video scores (${esc(pv.run || 'no run')})</div>`;
    const strips = DIMS7.map((dim) => {
      const worst = scores.filter(([, s]) => s.scores && s.scores[`vbench.${dim}`] != null).sort((a, b) => a[1].scores[`vbench.${dim}`] - b[1].scores[`vbench.${dim}`]).slice(0, 8);
      if (!worst.length) return '';
      return `<div class="strip-label">${dim} — lowest ${worst.length}</div>` + videoRow(worst.map(([f, s]) => `<div class="cell"><div class="cap" title="${esc(s.prompt || f)}"><code>${s.scores[`vbench.${dim}`].toFixed(3)}</code> ${esc((s.prompt || f).slice(0, 40))}</div>${videoEl(`/media/${pv.run}/videos/${f.split('/').pop()}`)}</div>`).join(''));
    }).join('');
    return head + table + strips;
  }
  return head;
}

// Inactive rows contain images only. At most one row owns video decoders.
let activePlayback = null;
function stopPlayback() {
  const previous = activePlayback;
  if (!previous) return;
  activePlayback = null;
  previous.abort.abort();
  for (const slot of $$('.video-slot', previous.row)) {
    const video = $('video', slot);
    if (video) { video.pause(); video.removeAttribute('src'); video.load(); }
    slot.innerHTML = previewHtml(slot.dataset.src);
  }
  previous.row.classList.remove('playing');
  const button = $('.row-play', previous.row);
  button.textContent = '播放此行'; button.setAttribute('aria-pressed', 'false');
}
async function playRow(row) {
  if (activePlayback?.row === row) return;
  stopPlayback();
  const current = { row, abort: new AbortController(), videos: [] };
  activePlayback = current;
  const button = $('.row-play', row);
  button.textContent = '加载此行…'; button.setAttribute('aria-pressed', 'true');
  row.classList.add('playing');
  const ready = await Promise.all($$('.video-slot', row).map((slot) => new Promise((resolve) => {
    const video = document.createElement('video');
    video.muted = true; video.playsInline = true; video.preload = 'auto';
    const signal = current.abort.signal;
    video.addEventListener('canplay', () => resolve(video), { once: true, signal });
    video.addEventListener('error', () => { slot.textContent = '视频加载失败'; resolve(null); }, { once: true, signal });
    signal.addEventListener('abort', () => resolve(null), { once: true });
    current.videos.push(video);
    slot.replaceChildren(video);
    video.src = slot.dataset.src;
  })));
  if (activePlayback !== current || !row.isConnected) return;
  current.videos = ready.filter(Boolean);
  if (!current.videos.length) { button.textContent = '视频加载失败'; return; }
  const restart = () => {
    if (activePlayback !== current) return;
    for (const video of current.videos) {
      video.currentTime = 0;
      video.play().catch(() => { if (activePlayback === current) button.textContent = '播放被阻止，请重新点击'; });
    }
  };
  for (const video of current.videos) video.addEventListener('ended', () => {
    if (current.videos.every((v) => v.ended || v.error)) restart();
  }, { signal: current.abort.signal });
  button.textContent = '停止此行'; restart();
}
document.addEventListener('click', (event) => {
  const row = event.target.closest('[data-video-row]');
  if (!row || event.target.closest('a')) return;
  if (event.target.closest('.row-play') && activePlayback?.row === row) stopPlayback();
  else playRow(row);
});
// Rerendering, closing a panel or leaving the tab also releases the decoders.
new MutationObserver(() => {
  if (activePlayback && (!activePlayback.row.isConnected || activePlayback.row.closest('.hidden'))) stopPlayback();
}).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
document.addEventListener('visibilitychange', () => { if (document.hidden) stopPlayback(); });
async function showVariantDetail(id) {
  const d = await api(`/api/variant/${id}`); const v = d.variant; const r = state.rows.find((x) => x.id === id);
  const runs = (r ? r.run_rows : []).map((x) => `<tr><td><code>${esc(x.id)}</code></td><td>${badge(x.state)}</td><td title="Slurm job / Slurm step">${esc(x.exec.host || '')} / ${x.exec.job ?? '—'}${x.exec.slurm_step != null ? '.' + esc(x.exec.slurm_step) : ''}</td><td>${x.progress?.step != null ? `${x.progress.step}/${x.progress.total}` : '—'}</td><td>${x.exit_code ?? ''}</td><td>${(x.checkpoint_steps || []).join(',')}</td><td>${(x.exports || []).map((e) => e.step + (e.ema ? 'e' : '')).join(',')}</td><td>${x.gpu_peak_mib ? Math.max(...x.gpu_peak_mib) + ' MiB' : ''}</td><td>${x.wandb_url ? `<a href="${esc(x.wandb_url)}" target="_blank">${esc(x.wandb_url.split('/').pop())}</a>` : ''}${averageStep(x)}</td><td>${x.matched_by ? `<span class="wicon">matched by ${x.matched_by}</span>` : ''}</td></tr>`).join('');
  openBottom(`${id} <span class="muted">${esc(v.name)}</span> ${badge(v.status)}`,
    `<p>${esc(v.description || '')}</p>` +
    `<h4>runs</h4><table class="mini"><tr><th>run dir</th><th>state</th><th>host/job</th><th>step</th><th>exit</th><th>ckpts</th><th>exports</th><th>peak mem</th><th>W&B</th><th></th></tr>${runs || '<tr><td colspan="10" class="muted">no run directories</td></tr>'}</table>` +
    (r && r.warnings.length ? `<h4>warnings</h4><div class="warnbox">${esc(r.warnings.join('\n'))}</div>` : '') +
    `<h4>parameters</h4><div class="detail">${esc(Object.entries(v.parameters || {}).map(([k, x]) => `${k}: ${JSON.stringify(x)}`).join('\n'))}</div>` +
    `<h4>metrics</h4><div class="detail">${esc(JSON.stringify(v.metrics || {}, null, 1))}</div><h4>provenance</h4><div class="detail">${esc(JSON.stringify(v.provenance || {}, null, 1))}</div>`);
}

// ------------------------------------------------------------------ compare selected
$('#btn-compare').onclick = openCompare;
async function openCompare() {
  const ids = [...state.selected].sort(); if (!ids.length) { toast('select rows first'); return; }
  openBottom(`compare ${ids.join(', ')}`, `<div class="toolbar" style="margin-bottom:8px"><span class="chips" id="cmp-tabs"><span class="chip on" data-t="curves">curves</span><span class="chip" data-t="videos">validation videos</span><span class="chip" data-t="vbench">VBench by step</span></span><span id="cmp-status" class="muted"></span></div><div id="cmp-body"></div>`);
  $$('#cmp-tabs .chip').forEach((c) => c.onclick = () => { $$('#cmp-tabs .chip').forEach((x) => x.classList.toggle('on', x === c)); ({ curves: cmpCurves, videos: cmpVideos, vbench: cmpVbench })[c.dataset.t](ids); });
  cmpCurves(ids);
}
const KEYS = ['generator_loss', 'fake_score_loss', 'grad_norm/student', 'grad_norm/critic', 'step_time_sec', 'loss', 'grad_norm'];
const rowLabel = (id) => { const r = state.rows.find((x) => x.id === id); return r ? `${id} ${r.families.objective.collapsed} ${r.families.cube.collapsed} ${r.families.batch.collapsed}` : id; };
let plotlyLoading = null;
function loadPlotly() {
  if (window.Plotly) return Promise.resolve();
  if (!plotlyLoading) plotlyLoading = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = '/plotly.min.js';
    script.onload = resolve;
    script.onerror = () => { script.remove(); plotlyLoading = null; reject(new Error('Unable to load the curve renderer')); };
    document.head.appendChild(script);
  });
  return plotlyLoading;
}
async function cmpCurves(ids) {
  const body = $('#cmp-body'); body.innerHTML = `<div class="toolbar"><span class="chips" id="cmp-keys">${KEYS.map((k) => `<span class="chip ${load('cmpKeys', ['generator_loss', 'fake_score_loss', 'grad_norm/student']).includes(k) ? 'on' : ''}" data-k="${k}">${k}</span>`).join('')}</span><label>smooth <input id="cmp-smooth" type="range" min="0" max="0.98" step="0.02" value="${load('cmpSmooth', 0.6)}"></label><button class="small" id="cmp-refresh">refresh W&amp;B</button></div><div id="cmp-plots" style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px"></div>`;
  const keys = () => $$('#cmp-keys .chip.on').map((c) => c.dataset.k);
  const draw = async (force) => {
    $('#cmp-status').textContent = 'loading…';
    let d;
    try {
      [d] = await Promise.all([api(`/api/scalars?ids=${ids.join(',')}&keys=${keys().join(',')}${force ? '&refresh=1' : ''}`), loadPlotly()]);
    } catch (error) { $('#cmp-status').textContent = error.message; return; }
    if (!$('#cmp-smooth')) return; // The user switched away while the curve resources loaded.
    $('#cmp-status').textContent = ids.map((id) => { const m = d[id] && d[id].meta; return m ? `${id}@${m.last_step}${m.error ? ' ERR' : ''}` : `${id}: none`; }).join(' · ');
    const a = parseFloat($('#cmp-smooth').value); store('cmpSmooth', a); store('cmpKeys', keys());
    const plots = $('#cmp-plots'); plots.innerHTML = '';
    for (const key of keys()) {
      const traces = [];
      for (const id of ids) { const pts = d[id] && d[id].series && d[id].series[key]; if (!pts || !pts.length) continue; let s = null; const sm = pts.map(([x, y]) => { s = s === null ? y : a * s + (1 - a) * y; return [x, s]; }); traces.push({ x: sm.map((p) => p[0]), y: sm.map((p) => p[1]), name: rowLabel(id), mode: 'lines', line: { width: 1.5 } }); }
      if (!traces.length) continue;
      const div = document.createElement('div'); plots.appendChild(div);
      Plotly.newPlot(div, traces, { title: { text: key, font: { size: 12 } }, height: 260, margin: { l: 45, r: 8, t: 28, b: 30 }, paper_bgcolor: '#171b21', plot_bgcolor: '#171b21', font: { color: '#d9dee5', size: 10 }, xaxis: { gridcolor: '#2a313b', range: [0, state.maxSteps] }, yaxis: { gridcolor: '#2a313b' }, legend: { orientation: 'h', y: -0.25 }, hovermode: 'x unified' }, { responsive: true, displaylogo: false });
    }
    if (!plots.children.length) plots.innerHTML = '<p class="muted">no scalar data</p>';
  };
  $$('#cmp-keys .chip').forEach((c) => c.onclick = () => { c.classList.toggle('on'); draw(false); });
  $('#cmp-smooth').onchange = () => draw(false); $('#cmp-refresh').onclick = () => draw(true);
  draw(false);
}
async function cmpVideos(ids) {
  const body = $('#cmp-body'); body.innerHTML = '<p class="muted">loading…</p>';
  await Promise.all(ids.map(async (id) => { state.videos[id] = await api(`/api/videos/${id}`); }));
  const caps = [...new Set(ids.flatMap((id) => state.videos[id].captions || []))]; const steps = [...new Set(ids.flatMap((id) => state.videos[id].steps || []))].sort((a, b) => a - b);
  body.innerHTML = `<div class="toolbar" style="margin-bottom:6px"><label>caption <select id="cv-cap">${caps.map((c) => `<option>${esc(c)}</option>`).join('')}</select></label><label>step <select id="cv-step"><option value="latest">latest each</option>${steps.map((s) => `<option>${s}</option>`).join('')}</select></label></div><div id="cv-grid"></div>`;
  const draw = () => {
    const cap = $('#cv-cap').value, st = $('#cv-step').value;
    const cells = ids.map((id) => { const d = state.videos[id]; const step = st === 'latest' ? Math.max(-1, ...(d.steps || [])) : parseInt(st, 10); const v = (d.videos[String(step)] || []).find((x) => x.caption === cap); return `<div class="cell"><div class="cap"><b>${id}</b><span class="muted">${esc(rowLabel(id).slice(6))}</span><span>step ${step}</span></div>${v ? validationProtocol(v) + videoEl(`/media/${v.run}/${v.file}`) : `<div class="muted">no video (steps ${(d.steps || []).join(',')})</div>`}</div>`; });
    $('#cv-grid').innerHTML = Array.from({ length: Math.ceil(cells.length / 4) }, (_, i) => videoRow(cells.slice(i * 4, i * 4 + 4).join(''), 'video-grid')).join('');
  };
  $('#cv-cap').onchange = draw; $('#cv-step').onchange = draw; draw();
}
function cmpVbench(ids) {
  const dims = ['imaging_quality', 'aesthetic_quality', 'subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree'];
  const rows = ids.flatMap((id) => (state.rows.find((x) => x.id === id) || { evals: [] }).evals.map((e) => ({ id, ...e })));
  $('#cmp-body').innerHTML = rows.length ? `<table class="mini"><tr><th>Variant</th><th>step</th><th>dataset</th><th>forward</th><th>E0030</th><th>status</th><th>prompts</th>${dims.map((d) => `<th>${d.replace('_', ' ')}</th>`).join('')}<th>Quality</th></tr>${rows.sort((a, b) => a.id.localeCompare(b.id) || a.step - b.step).map((e) => `<tr><td><b>${e.id}</b></td><td class="num">${e.step}</td><td>${esc(e.kind || e.kind_error)}</td><td title="${esc(e.attention_backend || '')}">${esc(e.backend || '')}</td><td>${e.id ? e.id : ''}${e.eval || ''}</td><td>${badge(e.status)}</td><td class="num">${e.prompts ?? ''}</td>${dims.map((d) => `<td class="num">${e.metrics[d] != null ? e.metrics[d].toFixed(4) : ''}</td>`).join('')}<td class="num"><b>${e.quality_score != null ? e.quality_score.toFixed(4) : ''}</b></td></tr>`).join('')}</table>` : '<p class="muted">no E0030 evaluations for the selected Variants</p>';
}

// ------------------------------------------------------------------ queue & planning drawer
$('#btn-queue').onclick = openQueueDrawer;
$('#btn-cluster').onclick = openClusterDrawer;
$('#btn-events').onclick = async () => { const ev = await api('/api/events?limit=200'); openDrawer('events', `<div class="detail">${esc(ev.map((e) => `${e.ts}  ${e.level.padEnd(5)}  ${e.message}`).join('\n') || 'no events')}</div>`); };
$('#btn-warnings').onclick = () => {
  const text = state.rows.filter((r) => r.warnings.length).map((r) => `${r.id}: ${r.warnings.join('; ')}`).join('\n');
  openDrawer('warnings', `<div class="warnbox">${esc(text || 'No warnings')}</div>`);
};
const DIMS = ['imaging_quality', 'aesthetic_quality', 'subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree'];
const DIM_SHORT = { imaging_quality: 'imaging', aesthetic_quality: 'aesthetic', subject_consistency: 'subject', background_consistency: 'background', temporal_flickering: 'flicker', motion_smoothness: 'smooth', dynamic_degree: 'dynamic' };
async function openQueueDrawer() {
  if ($('#drawer').classList.contains('hidden') || !$('#q-settings')) {
    openDrawer('queue & planning', `<h3>Scheduler</h3><div id="q-settings" class="card"></div><h3>Evaluation queue</h3><div style="overflow:auto"><table id="q-table" class="mini"></table></div>${state.vbenchSubmission ? '<h3>New evaluation Variant (E0030)</h3><form id="ev-form" class="card"></form>' : ''}<h3>New training Variant (E0029, PLANNED only)</h3><form id="tr-form" class="card"></form>`);
    if (state.vbenchSubmission) buildEvalForm();
    await buildTrainForm();
  }
  await renderQueue();
}
async function renderQueue() {
  const q = await api('/api/queue'); const s = q.settings; const probe = s.last_probe;
  $('#q-settings').innerHTML = `<div class="row"><label>eval allocation (Slurm job id) <input id="s-alloc" type="number" value="${s.eval_allocation_id ?? ''}" style="width:7em"></label><label>GPUs <input id="s-gpus" type="number" value="${s.num_gpus}" min="1" max="4" style="width:3em"></label><button id="s-save" class="small">save</button><button id="s-pause" class="small ${s.paused ? 'primary' : 'danger'}">${s.paused ? 'resume scheduler' : 'pause scheduler'}</button><button id="s-tick" class="small">tick now</button></div>
    <div class="row muted">state: <b>${s.paused ? 'PAUSED' : 'ACTIVE'}</b> · last tick ${esc(s.last_tick || '-')} · decision: ${esc(s.last_decision || '-')}</div>
    <div class="row muted">last probe: ${probe ? `${esc(probe.ts)} ${probe.ok ? probe.gpus.map((g) => `gpu${g.gpu}=${g.used_mib}MiB/${g.util}%`).join(' ') : 'ERR ' + esc(probe.error)}` : '-'}</div>`;
  $('#s-save').onclick = async () => { await post('/api/settings', { eval_allocation_id: parseInt($('#s-alloc').value, 10) || 0, num_gpus: parseInt($('#s-gpus').value, 10) }); toast('settings saved'); renderQueue(); };
  $('#s-pause').onclick = async () => { await post('/api/settings', { paused: !s.paused }); renderQueue(); };
  $('#s-tick').onclick = async () => { const r = await post('/api/scheduler/tick'); toast(`tick: ${r.decision}`); renderQueue(); };
  $('#q-table').innerHTML = `<tr><th>#</th><th>Kind</th><th>Status</th><th>Variant</th><th>Checkpoint</th><th>Fwd</th><th>Dims</th><th>Mode</th><th>Output</th><th>Error</th><th></th></tr>` + (q.queue.map((r) => `<tr><td>${r.id}</td><td>${esc(r.kind || 'vbench')}${r.env.OVERWRITE === '1' ? ' <span class="muted">overwrite</span>' : ''}</td><td>${badge(r.status)}</td><td><b>${r.variant_id}</b></td><td>${esc(r.env.CHECKPOINT)}</td><td>${r.kind === 'inf_val' ? 'VSQA' : esc(r.env.ATTENTION_BACKEND || 'dense')}</td><td title="${esc(r.env.DIMENSIONS || '')}">${r.env.DIMENSIONS ? r.env.DIMENSIONS.split(',').length + '/7' : '7/7'}${r.env.LIMIT ? ' lim ' + r.env.LIMIT : ''}</td><td>${esc(r.env.SAMPLE_MODE)}${r.env.SAMPLE_MODE === 'fixed' ? ' x' + r.env.SAMPLES_PER_PROMPT : ''}</td><td>${r.run_dir ? `<code title="${esc(r.run_dir)}">${esc(r.run_dir.replace('logs/', '').slice(-34))}</code>` : ''}</td><td title="${esc(r.error || '')}">${esc((r.error || '').slice(0, 30))}</td>
    <td>${r.status === 'queued' ? `<button class="small" data-act="up" data-id="${r.id}">▲</button> <button class="small" data-act="down" data-id="${r.id}">▼</button> <button class="small danger" data-act="cancel" data-id="${r.id}">cancel</button>` : ''}${r.status === 'running' ? `<button class="small danger" data-act="kill" data-id="${r.id}">kill</button>` : ''}${['failed', 'cancelled'].includes(r.status) && (r.kind === 'inf_val' || state.vbenchSubmission) ? `<button class="small" data-act="requeue" data-id="${r.id}">requeue</button>` : ''}</td></tr>`).join('') || '<tr><td colspan="11" class="muted">queue empty</td></tr>');
  $$('#q-table button[data-act]').forEach((b) => b.onclick = async () => { if (b.dataset.act === 'kill' && !confirm('SIGTERM the running launcher?')) return; try { await post(`/api/queue/${b.dataset.id}/${b.dataset.act}`); } catch (e) { toast(e.message, 8000); } renderQueue(); });
}
function buildEvalForm() {
  const f = $('#ev-form');
  f.innerHTML = `<div class="kv">
    <label>checkpoint source</label><input name="checkpoint_source" list="ck-list" placeholder="E0029/V0040@500 or HF id" required><datalist id="ck-list"></datalist>
    <label>attention forward</label><select name="attention_backend"><option value="">dense BF16 (TORCH_SDPA)</option><option value="VSQA">VSQA (FVFA4-v3 NVFP4-QK/FP8-PV)</option></select>
    <label>VSA sparsity (VSQA)</label><input name="sparsity" placeholder="from checkpoint (0.9)">
    <label>dimensions</label><div>${DIMS.map((d) => `<label><input type="checkbox" name="dimensions" value="${d}" checked> ${DIM_SHORT[d]}</label>`).join(' ')}</div>
    <label>sample mode</label><select name="sample_mode"><option value="fixed">fixed N per prompt (VBench-lite)</option><option value="official">official (5 / 25 for flicker)</option></select>
    <label>samples per prompt</label><input name="samples_per_prompt" type="number" value="1" min="1">
    <label>seed base</label><input name="seed_base" type="number" value="0">
    <label>limit prompts</label><input name="limit" type="number" placeholder="empty = all">
    <label>frames × H × W · fps</label><div class="row"><input name="frames" type="number" value="81" style="width:5em"><input name="height" type="number" value="480" style="width:5em"><input name="width" type="number" value="832" style="width:5em"><input name="fps" type="number" value="16" style="width:4em"></div>
    <label>steps / CFG / flow shift</label><div class="row"><input name="sampling_steps" placeholder="auto" style="width:5em"><input name="cfg" placeholder="auto" style="width:5em"><input name="flow_shift" placeholder="auto" style="width:5em"></div>
    <label>note</label><input name="note" placeholder="why this evaluation"></div>
  <div class="row"><span class="muted" id="ev-form-counts"></span></div>
  <div class="row"><button type="button" class="small" id="ev-preview">preview row</button><button type="submit" class="primary">create E0030 Variant + enqueue</button><label><input type="checkbox" name="enqueue" checked> enqueue</label></div>
  <pre class="detail" id="ev-preview-out"></pre>`;
  const refreshCounts = async () => { const dims = $$('input[name=dimensions]:checked', f).map((x) => x.value); if (!dims.length) { $('#ev-form-counts').textContent = 'select at least one dimension'; return; } const c = await api(`/api/eval/prompt_counts?dims=${dims.join(',')}`); const spp = parseInt(f.samples_per_prompt.value, 10) || 1; const lim = parseInt(f.limit.value, 10); const prompts = lim ? Math.min(lim, c.union) : c.union; const n = f.sample_mode.value === 'official' ? c.official_videos : prompts * spp; $('#ev-form-counts').textContent = `${c.union} prompts (${Object.entries(c.per_dimension).map(([k, v]) => `${DIM_SHORT[k]} ${v}`).join(', ')}); ${n} videos ≈ ${(n * 81 / 4 / 3600).toFixed(1)} h on 4 GPUs${dims.length < 7 ? ' · Quality Score will be null (incomplete dimension set)' : ''}`; };
  f.addEventListener('input', refreshCounts); refreshCounts();
  api('/api/checkpoints').then((cks) => { $('#ck-list').innerHTML = cks.filter((c) => c.export || c.ema_export).map((c) => `<option value="${esc(c.ref)}">${esc(c.name.slice(0, 50))}</option>`).join(''); });
  const formData = () => { const fd = new FormData(f); const o = {}; for (const [k, v] of fd.entries()) { if (k === 'dimensions') (o.dimensions = o.dimensions || []).push(v); else if (v !== '') o[k] = v; } o.enqueue = f.enqueue.checked; ['sparsity', 'cfg', 'flow_shift'].forEach((k) => { if (o[k] != null) o[k] = parseFloat(o[k]); }); ['samples_per_prompt', 'seed_base', 'limit', 'frames', 'height', 'width', 'fps', 'sampling_steps'].forEach((k) => { if (o[k] != null) o[k] = parseInt(o[k], 10); }); return o; };
  $('#ev-preview').onclick = async () => { try { const r = await post('/api/variants/eval', { ...formData(), dry_run: true }); $('#ev-preview-out').textContent = `will create ${r.preview.id}\n` + JSON.stringify(r.preview, null, 1) + '\nlauncher env: ' + JSON.stringify(r.env); } catch (e) { toast(e.message); } };
  f.onsubmit = async (e) => { e.preventDefault(); try { const r = await post('/api/variants/eval', formData()); toast(`created ${r.id}${r.queue_id ? `, queued #${r.queue_id}` : ''}; lint rc ${r.lint_rc}`); $('#ev-preview-out').textContent = JSON.stringify(r, null, 1); renderQueue(); loadOverview(); } catch (err) { toast(err.message, 8000); } };
}
// Dataset lanes do not select a forward. This generic form requires explicit confirmation
// and cannot declare a pinned prompt manifest.
function prefillEvalForm(ref, spec) {
  const f = $('#ev-form'); const r = state.rows.find((x) => ref.startsWith(`E0029/${x.id}@`));
  f.checkpoint_source.value = ref;
  if (r) { const p = r.params; f.attention_backend.value = String(p.attention_kind || '').toLowerCase() === 'vsa' ? 'VSQA' : ''; if (p.frames) { f.frames.value = p.frames; f.height.value = p.height; f.width.value = p.width; } }
  if (spec && spec.subset) {
    const sub = spec.subset; const dims = sub.split('-')[0];
    const longOf = Object.fromEntries(Object.entries(DIM_SHORT).map(([k, v]) => [v, k]));
    const want = dims === 'q7' ? DIMS : dims.split('+').map((x) => longOf[x]).filter(Boolean);
    // A pinned-manifest label ('q7tiny-v1') names no dimension set: keep the form's defaults rather
    // than clearing every dimension and leaving an unsubmittable form.
    if (want.length) $$('input[name=dimensions]', f).forEach((cb) => cb.checked = want.includes(cb.value));
    const lim = sub.match(/-lim(\d+)/); f.limit.value = lim ? lim[1] : '';
    f.sample_mode.value = sub.includes('-official') ? 'official' : 'fixed';
  }
  f.dispatchEvent(new Event('input')); f.scrollIntoView({ block: 'start' });
  const fwd = f.attention_backend.options[f.attention_backend.selectedIndex].text;
  const datasetNote = spec?.subset === 'q7tiny-v1' ? 'generic VBench form; q7tiny-v1 requires a pinned-manifest declaration' : spec?.subset || 'generic VBench form';
  toast(`prefilled ${ref} · ${datasetNote} · selected forward: ${fwd} — verify before submitting; nothing is queued yet`, 10000);
}
async function buildTrainForm() {
  const f = $('#tr-form'); const t = await api('/api/train/template');
  f.innerHTML = `<div class="kv">
    <label>clone parameters from</label><select name="clone_from"><option value="">(none)</option>${state.rows.map((v) => `<option value="${v.id}">${v.id} ${esc(v.name.slice(0, 60))}</option>`).join('')}</select>
    <label>next id</label><span><b>${t.next_id}</b> <span class="muted">PLANNED, runs [] — an agent launches it later</span></span>
    <label>name</label><input name="name" required placeholder="93f INFER NATIVE C256T8 FT-250 -> DMD 1000, ...">
    <label>description</label><textarea name="description" rows="3" placeholder="what changes versus the parent Variant; not a quality claim"></textarea>
    <label>launcher entry</label><input name="entry" list="entry-list"><datalist id="entry-list">${t.entries.map((e) => `<option value="${esc(e)}">`).join('')}</datalist></div>
  <h4>parameters <span class="muted">(blank removes the key)</span></h4><div class="kv" id="tr-params"></div>
  <div class="row"><input id="tr-new-param" placeholder="new parameter key" style="width:14em"><button type="button" class="small" id="tr-add-param">add</button></div>
  <h4>launcher env (provenance.env)</h4><div class="kv" id="tr-env"></div>
  <div class="row"><input id="tr-new-env" placeholder="NEW_ENV_KEY" style="width:14em"><button type="button" class="small" id="tr-add-env">add</button></div>
  <div class="row"><button type="button" class="small" id="tr-preview">preview row</button><button type="submit" class="primary">create PLANNED E0029 Variant</button></div>
  <pre class="detail" id="tr-preview-out"></pre>`;
  const colLabel = Object.fromEntries(t.columns.map((c) => [c.key, c.label]));
  const kvRows = (el, obj, labels) => { el.innerHTML = Object.entries(obj).map(([k, v]) => `<label title="${esc(labels[k] || '')}">${esc(k)}</label><input data-k="${esc(k)}" value="${esc(typeof v === 'object' ? JSON.stringify(v) : v)}">`).join(''); };
  const fill = async () => { const id = f.clone_from.value; const v = id ? (await api(`/api/variant/${id}`)).variant : null; kvRows($('#tr-params'), v ? v.parameters : {}, colLabel); kvRows($('#tr-env'), v && v.provenance ? v.provenance.env || {} : {}, {}); if (v && v.provenance && v.provenance.entry) f.entry.value = v.provenance.entry; };
  f.clone_from.onchange = fill; fill();
  const addRow = (el, key) => { if (key) el.insertAdjacentHTML('beforeend', `<label>${esc(key)}</label><input data-k="${esc(key)}" value="">`); };
  $('#tr-add-param').onclick = () => { addRow($('#tr-params'), $('#tr-new-param').value.trim()); $('#tr-new-param').value = ''; };
  $('#tr-add-env').onclick = () => { addRow($('#tr-env'), $('#tr-new-env').value.trim()); $('#tr-new-env').value = ''; };
  const collect = (el) => Object.fromEntries($$('input[data-k]', el).map((i) => [i.dataset.k, i.value === '' ? null : i.value]));
  const payload = () => ({ clone_from: f.clone_from.value || null, name: f.name.value, description: f.description.value || null, entry: f.entry.value || null, parameters: collect($('#tr-params')), env: collect($('#tr-env')) });
  $('#tr-preview').onclick = async () => { try { const r = await post('/api/variants/train', { ...payload(), dry_run: true }); $('#tr-preview-out').textContent = JSON.stringify(r.preview, null, 1); } catch (e) { toast(e.message); } };
  f.onsubmit = async (e) => { e.preventDefault(); if (!confirm('Append a PLANNED Variant to E0029 results.yaml?')) return; try { const r = await post('/api/variants/train', payload()); toast(`created ${r.id}; lint rc ${r.lint_rc}`); $('#tr-preview-out').textContent = JSON.stringify(r, null, 1); loadOverview(); } catch (err) { toast(err.message, 8000); } };
}
async function openClusterDrawer() {
  const d = await api('/api/cluster');
  const gpuBars = (gpus) => gpus && gpus.length ? `<div class="chips">${gpus.map((g) => `<span>gpu${g.gpu} <span class="bar"><i style="width:${100 * g.used_mib / g.total_mib}%"></i></span> ${(g.used_mib / 1024).toFixed(0)}G</span>`).join('')}</div>` : '';
  openDrawer('cluster', d.allocations.map((a) => `<div class="alloc"><h4>job ${a.job} <span class="muted">${esc(a.name)}</span> — ${esc(a.node)} ${badge(a.state)} <span class="muted">${esc(a.gres)} · ${esc(a.elapsed)}</span></h4>${a.runs.length ? a.runs.map((r) => `<div>${badge(r.kind)} ${r.variant ? `<b>${r.variant}</b> ` : ''}<code>${esc(r.id)}</code> ${r.progress ? `${r.progress.step}/${r.progress.total} · ${r.progress.sec_per_it}s/it · ETA ${esc(r.progress.eta)}` : ''}${gpuBars(r.gpu_latest)}</div>`).join('') : '<div class="muted">no RUNNING run directory reports this allocation</div>'}</div>`).join('') + (d.unassigned.length ? `<div class="alloc"><h4>RUNNING run dirs without a job id</h4>${d.unassigned.map((r) => `<div><code>${esc(r.id)}</code></div>`).join('')}</div>` : ''));
}

// ------------------------------------------------------------------ boot
setInterval(() => { $('#clock').textContent = new Date().toLocaleTimeString(); }, 1000);
setInterval(() => { if (!document.hidden) loadOverview(); }, 120000);
saveSel();
loadOverview();
