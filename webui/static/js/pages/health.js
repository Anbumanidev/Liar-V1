// Health & logs: engine status, diagnostics findings (LLM / pipeline / data / DB), grouped errors,
// and a live log tail viewer (engine.log + JSONL categories).
import { api, on, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { pageTimers } from '../router.js';
import { getScope } from '../scope.js';

export const title = 'Health & logs';

export function mount(root, params) {
  const t = pageTimers();
  const scopeDay = () => (getScope().kind === 'today' ? getScope().date : f.todayIST());
  let day = params.date || scopeDay();
  const dateIn = h('input', { type: 'date', class: 'input sm', value: day, max: f.todayIST(), 'aria-label': 'Day' });
  const engineCard = ui.card('Engine', { body: ui.skeleton(4) });
  const findingsCard = ui.card('Findings & recommended actions', { body: ui.skeleton(3) });
  const llmCard = ui.card('AI models (LLM)', { body: ui.skeleton(3) });
  const pipeCard = ui.card('Pipeline · data · database', { body: ui.skeleton(3) });
  const errCard = ui.card('Top errors', { flush: true, hint: 'last 50 error rows, grouped', body: ui.skeleton(4) });
  const logCard = ui.card('Logs', { flush: true, body: h('div') });
  dateIn.addEventListener('change', () => { if (dateIn.value) { day = dateIn.value; load(); loadLogs(); } });
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Health & logs'), dateIn, h('span', { class: 'grow' }),
    h('button', { class: 'btn sm', onclick: () => { load(); loadLogs(); } }, icon('refresh', 14), 'Refresh')),
  h('div', { class: 'grid cols-2' }, engineCard, findingsCard), h('div', { class: 'grid cols-2' }, llmCard, pipeCard), errCard, logCard);

  const load = async () => {
    let d;
    try { d = await api.get('/api/health' + qs({ date: day })); } catch (err) { engineCard.body.replaceChildren(ui.errorBox(err)); return; }
    const e = d.engine;
    const flag = (v, okText, badText) => (v === null || v === undefined ? ui.badge('unknown') : v ? ui.badge(okText, 'good') : ui.badge(badText, 'bad'));
    engineCard.body.replaceChildren(h('div', { class: 'kv' },
      h('div', { class: 'k' }, 'State'), h('div', { class: 'v' }, ui.badge(e.label, e.alive ? (e.state === 'RUNNING' ? 'good' : 'warn') : 'bad')),
      h('div', { class: 'k' }, 'Heartbeat'), h('div', { class: 'v' }, e.heartbeat_at ? `${f.time(e.heartbeat_at)} IST (${f.ago(e.heartbeat_age_s)} ago)` : 'never'),
      h('div', { class: 'k' }, 'Started'), h('div', { class: 'v' }, e.started_at ? f.dateTime(e.started_at) : '–'),
      h('div', { class: 'k' }, 'Next cycle'), h('div', { class: 'v' }, e.next_cycle_at ? `${f.time(e.next_cycle_at)} IST` : '–'),
      h('div', { class: 'k' }, 'Activity'), h('div', { class: 'v' }, e.activity || '–'),
      h('div', { class: 'k' }, 'Market data'), h('div', { class: 'v' }, flag(e.data_feed_ok, 'OK', 'DOWN')),
      h('div', { class: 'k' }, 'Local LLM'), h('div', { class: 'v' }, flag(e.llm_ok, 'OK', 'DOWN')),
      h('div', { class: 'k' }, 'Errors on this day'), h('div', { class: 'v' }, String(d.errors_today)),
      e.last_error ? h('div', { class: 'k' }, 'Last error') : null, e.last_error ? h('div', { class: 'v neg' }, e.last_error) : null,
      d.web_data ? h('div', { class: 'k' }, 'Dashboard data feed') : null,
      d.web_data ? h('div', { class: 'v' }, d.web_data.healthy ? ui.badge('OK', 'good') : ui.badge(d.web_data.last_error || 'failing', 'bad')) : null));
    const dg = d.diagnostics;
    if (!dg) {
      const na = ui.notice('info', 'Diagnostics not available yet', d.diagnostics_error?.status === 500 ? d.diagnostics_error.detail
        : 'LLM, pipeline and data diagnostics appear here once the analysis module is installed. Engine status, errors and logs already work.');
      findingsCard.body.replaceChildren(na);
      llmCard.body.replaceChildren(ui.empty('No LLM statistics', ''));
      pipeCard.body.replaceChildren(ui.empty('No pipeline statistics', ''));
    } else {
      const fnd = dg.findings || [];
      const order = { high: 0, medium: 1, low: 2 };
      findingsCard.body.replaceChildren(fnd.length ? h('div', { class: 'issues' }, [...fnd].sort((a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3)).map((x) =>
        h('div', { class: ['issue', x.severity] },
          h('div', { class: 'head' }, ui.badge(String(x.severity || '').toUpperCase(), x.severity === 'high' ? 'bad' : x.severity === 'medium' ? 'warn' : 'info'),
            h('span', { class: 'title' }, x.title)),
          x.detail ? h('div', { class: 'detail' }, x.detail) : null,
          x.action ? h('div', { class: 'fix' }, h('b', null, 'Do this: '), x.action) : null)))
        : ui.notice('good', 'All healthy', 'No problems detected on this day.'));
      const llm = dg.llm || {};
      llmCard.body.replaceChildren(h('div', { class: 'facts-grid' },
        ui.fact('Calls', f.num(llm.calls, 0)), ui.fact('Failed', f.num(llm.failed, 0), llm.failed ? 'neg' : ''),
        ui.fact('Timeouts', f.num(llm.timeouts, 0), llm.timeouts ? 'neg' : ''), ui.fact('Invalid output', f.num(llm.invalid, 0)),
        ui.fact('Failure rate', f.isNum(llm.fail_pct) ? `${f.num(llm.fail_pct, 1)}%` : '–'),
        ui.fact('Avg latency', f.isNum(llm.avg_latency_s) ? `${f.num(llm.avg_latency_s, 1)} s` : '–'),
        ui.fact('p90 latency', f.isNum(llm.p90_latency_s) ? `${f.num(llm.p90_latency_s, 1)} s` : '–')));
      const pl = dg.pipeline || {}; const dt = dg.data || {}; const db = dg.db || {};
      const bySym = Object.entries(dt.by_symbol || {}).sort((a, b) => b[1] - a[1]).slice(0, 12);
      pipeCard.body.replaceChildren(h('div', { class: 'stack' },
        h('div', { class: 'facts-grid' },
          ui.fact('Analyses', f.num(pl.analyses, 0)), ui.fact('Avg time', f.isNum(pl.avg_s) ? `${f.num(pl.avg_s, 1)} s` : '–'),
          ui.fact('Slowest', f.isNum(pl.max_s) ? `${f.num(pl.max_s, 1)} s` : '–'), ui.fact('Slow analyses', f.num(pl.slow, 0)),
          ui.fact('Data errors', f.num(dt.errors, 0), dt.errors ? 'neg' : ''), ui.fact('DB locked', f.num(db.locked, 0), db.locked ? 'neg' : '')),
        bySym.length ? h('div', null, h('div', { class: 'section-title' }, 'Data errors by symbol'),
          h('div', { class: 'chips' }, bySym.map(([s, n]) => ui.badge(`${s} × ${n}`, 'warn')))) : null));
    }
    errCard.body.replaceChildren(ui.table([
      { key: 'count', label: 'Count', num: true, fmt: (e) => h('b', null, String(e.count)) },
      { key: 'component', label: 'Component', fmt: (e) => h('span', { class: 'mono' }, e.component) },
      { key: 'severity', label: 'Severity', fmt: (e) => ui.badge(e.severity, e.severity === 'ERROR' || e.severity === 'CRITICAL' ? 'bad' : 'warn') },
      { key: 'message', label: 'Message', wrap: true },
      { key: 'last_ts', label: 'Last seen', fmt: (e) => f.dateTime(e.last_ts) },
    ], d.errors, { empty: ui.empty('No errors recorded', 'The engine records data/LLM/DB problems here.', 'check') }));
  };

  // ------------------------------------------------------------------ log viewer
  const fileSel = h('select', { class: 'input sm', 'aria-label': 'Log file' },
    ['engine', 'activity', 'errors', 'decisions', 'orders', 'trades', 'risk', 'scans', 'agents', 'debates', 'approvals', 'control', 'capital', 'shadow', 'audit', 'web']
      .map((x) => h('option', { value: x }, x === 'engine' ? 'engine.log' : x === 'web' ? 'web.log' : `${x}.jsonl`)));
  const linesSel = h('select', { class: 'input sm', 'aria-label': 'Lines' }, [100, 200, 500, 1000].map((n) => h('option', { value: n, selected: n === 200 }, `${n} lines`)));
  const levelSel = h('select', { class: 'input sm', 'aria-label': 'Level' }, h('option', { value: '' }, 'All levels'),
    ['ERROR', 'WARNING', 'INFO', 'DEBUG'].map((l) => h('option', { value: l }, l)));
  const search = h('input', { class: 'input sm', placeholder: 'Search…', 'aria-label': 'Search logs', style: { width: '180px' } });
  let autoScroll = true;
  let live = true;
  const logbox = h('div', { class: 'logbox', role: 'log' });
  const status = h('span', { class: 'small faint' }, '');
  logCard.head.replaceChildren(h('h2', null, 'Logs'), h('div', { class: 'actions' }, fileSel, linesSel, levelSel, search,
    ui.toggle('Auto-scroll', autoScroll, (v) => { autoScroll = v; }), ui.toggle('Live', live, (v) => { live = v; }), status));
  logCard.body.replaceChildren(logbox);

  const lineEl = (e, kind) => {
    if (kind === 'text') {
      return h('div', { class: ['logline', e.level] }, h('span', { class: 'ts' }, (e.ts || '').slice(11, 19)),
        h('span', { class: ['lv', e.level] }, e.level), h('span', { class: 'msg' }, e.source ? h('span', { class: 'src' }, `${e.source} `) : null, e.message));
    }
    const { ts, category, ...rest } = e;
    const sev = String(rest.severity || rest.level || '').toUpperCase();
    const summary = Object.entries(rest).filter(([, v]) => v !== null && typeof v !== 'object').slice(0, 10)
      .map(([k, v]) => `${k}=${typeof v === 'string' && v.length > 120 ? v.slice(0, 120) + '…' : v}`).join('  ');
    const hasObj = Object.values(rest).some((v) => v && typeof v === 'object');
    return h('div', { class: ['logline', sev] }, h('span', { class: 'ts' }, ts ? f.time(ts) : ''), h('span', { class: ['lv', sev] }, sev || category || ''),
      h('span', { class: 'msg' }, summary || e._raw || '',
        hasObj ? h('details', { class: 'more' }, h('summary', null, 'details'), h('div', { class: 'json' }, JSON.stringify(rest, null, 2))) : null));
  };
  let lastKind = 'text';
  const loadLogs = async () => {
    const file = fileSel.value;
    levelSel.disabled = false;
    try {
      const d = await api.get('/api/logs' + qs({ file, date: day, lines: linesSel.value, q: search.value.trim(), level: levelSel.value }));
      lastKind = d.kind;
      logbox.replaceChildren(...(d.entries.length ? d.entries.map((e) => lineEl(e, d.kind))
        : [ui.empty(d.exists ? 'No matching lines' : `No ${file} log for ${day}`, d.exists ? 'Try another filter.' : 'The engine writes logs into logs/<date>/ while it runs.', 'search')]));
      status.textContent = `${d.count} line(s) · ${d.date}`;
      if (autoScroll) logbox.scrollTop = logbox.scrollHeight;
    } catch (err) {
      logbox.replaceChildren(ui.errorBox(err));
    }
  };
  for (const el of [fileSel, linesSel, levelSel]) el.addEventListener('change', loadLogs);
  search.addEventListener('input', debounce(loadLogs, 400));

  t.track(on('sse:log', (d) => {
    if (!live || fileSel.value !== 'engine' || day !== f.todayIST() || search.value.trim() || lastKind !== 'text') return;
    logbox.querySelector('.empty')?.remove();
    const lv = levelSel.value;
    for (const e of d.entries || []) {
      if (lv && e.level !== lv) continue;
      logbox.appendChild(lineEl(e, 'text'));
    }
    while (logbox.children.length > 2000) logbox.firstChild.remove();
    if (autoScroll) logbox.scrollTop = logbox.scrollHeight;
  }));
  t.track(on('scope', () => { const d = scopeDay(); if (d !== day) { day = d; dateIn.value = d; load(); loadLogs(); } }));
  t.every(30000, () => { if (day === f.todayIST()) load(); });
  t.every(15000, () => { if (live && fileSel.value !== 'engine' && day === f.todayIST()) loadLogs(); });
  const onRefresh = () => { load(); loadLogs(); };
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  load();
  loadLogs();
  return () => t.clear();
}
