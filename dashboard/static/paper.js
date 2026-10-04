// Live paper viewer. The PDF is loaded once per page load (or on "Reload PDF"); the header polls
// /api/paper and compares three times: newest source edit, newest complete PDF, and the PDF this
// tab loaded. Source newer than PDF: the server is compiling. Server PDF newer than the loaded
// one: this copy is outdated. Otherwise it is up to date.
const $ = (s) => document.querySelector(s);
let loaded = null;       // pdf_mtime of the copy shown in the iframe
let last = null;         // last /api/paper answer
let fetchedAt = 0;       // Date.now() when `last` arrived, to age server times without clock skew

function stamp(t) {
  if (t == null) return '—';
  const age = last.now - t + (Date.now() - fetchedAt) / 1000;
  const rel = age < 60 ? `${Math.max(0, Math.round(age))} s ago` : age < 3600 ? `${Math.round(age / 60)} min ago`
    : age < 86400 ? `${(age / 3600).toFixed(1)} h ago` : `${Math.round(age / 86400)} d ago`;
  return `${new Date(t * 1000).toLocaleString()} (${rel})`;
}

function render() {
  const d = last; if (!d) return;
  const badge = $('#paper-state');
  if (!d.available) {
    badge.className = 'paper-state error'; badge.textContent = 'no PDF';
    $('#paper-error').hidden = false; $('#paper-error').textContent = d.error || `no complete PDF in ${d.paper_dir}`;
    return;
  }
  $('#paper-error').hidden = true;
  const outdated = loaded != null && d.pdf_mtime > loaded;
  const [cls, text, tip] = outdated ? ['outdated', 'Newer PDF available — reload', 'The server has a newer PDF than the one shown here.']
    : d.state === 'compiling' ? ['compiling', 'Compiling on server…', 'A source file changed after the last complete PDF; the server is rebuilding it.']
    : ['current', 'Up to date', 'The PDF shown here is the newest build, and no source changed since.'];
  badge.className = `paper-state ${cls}`; badge.textContent = text; badge.title = tip;
  $('#source-time').textContent = stamp(d.source_mtime) + (d.source_file ? ` · ${d.source_file}` : '');
  $('#pdf-time').textContent = stamp(d.pdf_mtime);
  $('#loaded-time').textContent = loaded == null ? '—' : outdated ? `${new Date(loaded * 1000).toLocaleString()} (older)` : 'latest build';
}

async function poll() {
  try {
    const r = await fetch('/api/paper', { cache: 'no-store' });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    last = await r.json(); fetchedAt = Date.now();
  } catch (e) {
    $('#paper-state').className = 'paper-state error'; $('#paper-state').textContent = 'status unavailable';
    $('#paper-state').title = e.message; return false;
  }
  render(); return true;
}

async function loadPdf() {
  if (!(await poll()) || !last.available) return;
  loaded = last.pdf_mtime;
  $('#paper-frame').src = `/paper.pdf?v=${encodeURIComponent(loaded)}`;
  render();
}

$('#reload').onclick = loadPdf;
loadPdf();
setInterval(() => { if (!document.hidden) poll(); }, 5000);
setInterval(render, 1000);  // keep the relative ages ticking between polls
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
