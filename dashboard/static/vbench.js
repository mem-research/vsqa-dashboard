(() => {
  'use strict';
  const $ = (selector) => document.querySelector(selector);
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
  const params = new URLSearchParams(location.search);
  let dataset = ['all', 'q7', 'q7tiny'].includes(params.get('dataset')) ? params.get('dataset') : 'all';
  let data, byId, q7, tiny;
  const search = $('#prompt-search');
  search.value = params.get('q') || '';

  function render() {
    const query = search.value.trim().toLowerCase();
    const selection = data.datasets[dataset];
    const rows = selection.ids.map((id) => byId.get(id)).filter((row) => !query || row.prompt.toLowerCase().includes(query) || String(row.global_id).padStart(4, '0').includes(query));
    for (const button of document.querySelectorAll('[data-dataset]')) button.setAttribute('aria-pressed', String(button.dataset.dataset === dataset));
    $('#dataset-note').textContent = dataset === 'all'
      ? '完整 VBench 语料，按原始 Global ID 排序；相同文本的不同原始记录不会合并。'
      : dataset === 'q7'
        ? 'q7：保留至少属于一个 quality-7 维度的原始记录，取并集；不是要求同时具备七个维度。'
        : 'q7tiny-v1：固定的 32 条文本选取监测子集，按评测清单顺序展示；不是具有统计代表性的完整 benchmark。';
    $('#prompt-status').textContent = `显示 ${rows.length} / ${selection.ids.length} 条 · ${selection.label} · 完整语料 ${data.prompts.length} 条`;
    $('#prompt-rows').innerHTML = rows.map((row) => `<tr data-global-id="${row.global_id}"><td class="prompt-id">${String(row.global_id).padStart(4, '0')}</td><td>${escapeHtml(row.prompt)}</td><td>${row.dimensions.map((dim) => `<span class="prompt-dimension">${escapeHtml(dim)}</span>`).join('')}</td><td>${q7.has(row.global_id) ? '<span class="prompt-subset">q7</span>' : ''}${tiny.has(row.global_id) ? '<span class="prompt-subset">q7tiny-v1</span>' : ''}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">没有匹配的 prompt。</td></tr>';
    const url = new URL(location.href);
    url.searchParams.set('dataset', dataset);
    if (search.value) url.searchParams.set('q', search.value); else url.searchParams.delete('q');
    history.replaceState(null, '', url);
  }

  async function load() {
    try {
      const response = await fetch('/api/vbench/prompts');
      if (!response.ok) throw new Error(`语料加载失败（HTTP ${response.status}），请刷新重试。`);
      data = await response.json();
      byId = new Map(data.prompts.map((row) => [row.global_id, row]));
      q7 = new Set(data.datasets.q7.ids);
      tiny = new Set(data.datasets.q7tiny.ids);
      for (const button of document.querySelectorAll('[data-dataset]')) {
        const definition = data.datasets[button.dataset.dataset];
        button.textContent = `${definition.label} · ${definition.ids.length}`;
        button.disabled = false;
        button.onclick = () => { dataset = button.dataset.dataset; render(); };
      }
      search.disabled = false;
      $('#clear-search').disabled = false;
      search.oninput = render;
      $('#clear-search').onclick = () => { search.value = ''; render(); search.focus(); };
      $('#source-info').innerHTML = `<p>Global ID = VBench_full_info.json 中的 0-based 索引。前导零仅用于显示。</p><p>q7 维度：${data.quality_dimensions.map(escapeHtml).join(' · ')}</p><p>语料：<code>${escapeHtml(data.corpus_path)}</code><br>SHA-256：<code>${escapeHtml(data.corpus_sha256)}</code></p><p>q7tiny 定义：<code>${escapeHtml(data.dataset_path)}</code><br>SHA-256：<code>${escapeHtml(data.dataset_sha256)}</code></p><p>子集 ID 由评测脚本的 load_rows() 生成；q7tiny 的语料摘要、原文、维度及 ID 均通过同一校验。</p>`;
      $('#prompt-provenance').hidden = false;
      render();
    } catch (error) {
      $('#prompt-status').textContent = '语料不可用';
      $('#prompt-error').hidden = false;
      $('#prompt-error').textContent = error.message;
    }
  }
  load();
})();
