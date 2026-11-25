// Live log: what the engine and the AI models are doing right now.
// Tab "AI activity" = logs/<day>/activity.jsonl (human one-liners, "Running now" AI calls with live timers);
// tab "Terminal" = engine.log exactly as printed in the engine's terminal. Last ~300 lines on open, then live (SSE).
import { api, on, qs } from '../api.js';
import { h, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { pageTimers } from '../router.js';
import { store } from '../store.js';
import { openDecision } from '../components/decision.js';
import { STAGES, STAGE_LABEL, STAGE_HELP, activityRow, levelRank, RunningTracker, runningStrip, LoadGate } from '../components/activity.js';
import { scanNowControls } from '../components/controls.js';

export const title = 'Live log';

const MAX_ROWS = 3000;

function pref(key, dflt) {
  try { const v = localStorage.getItem(`liar.log.${key}`); return v === null ? dflt : JSON.parse(v); } catch { return dflt; }
}
function savePref(key, v) { try { localStorage.setItem(`liar.log.${key}`, JSON.stringify(v)); } catch { /* ignore */ } }

export function mount(root, params) {
  const t = pageTimers();
  let tab = params.tab === 'terminal' ? 'terminal' : pref('tab', 'activity');

  // ------------------------------------------------------------------ header: engine status + scan now
  const statusLine = h('div', { class: 'status-line' });
  let engine = store.overview?.engine || null;
  const paintStatus = () => {
    const e = engine;
    const nxt = e?.next_cycle_at ? Date.parse(e.next_cycle_at) : null;
    const left = nxt ? (nxt - f.nowMs()) / 1000 : null;
    const tone = !e ? '' : !e.alive ? 'bad' : e.state === 'RUNNING' ? 'good' : 'warn';
    statusLine.replaceChildren(ui.badge(e ? e.label : '…', tone), h('span', { class: 'act' }, e?.activity || (e?.alive ? 'running' : 'engine not running')),
      nxt && e?.alive ? h('span', { class: 'faint mono' }, left > 0 ? `next cycle in ${f.countdown(left)}` : 'next cycle due') : null,
      e?.heartbeat_age_s !== undefined && e?.heartbeat_at ? h('span', { class: 'faint small' }, `heartbeat ${f.time(e.heartbeat_at)}`) : null,
      e && !e.alive ? h('span', { class: 'small muted' }, 'Start it in a terminal:  .\\tasks.ps1 run') : null);
  };
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Live log'), h('span', { class: 'sub' }, 'what the engine and the AI are doing, live'),
    h('span', { class: 'grow' }), scanNowControls()),
  ui.card(null, { flush: true, body: statusLine }));

  // ------------------------------------------------------------------ AI activity tab
  const tracker = new RunningTracker();
  const strip = runningStrip(tracker);
  const list = h('div', { class: 'act-list' });
  const scroll = h('div', { class: 'act-scroll' }, list);
  const st = { stages: new Set(pref('hiddenStages', [])), symbol: '', minLevel: pref('minLevel', 0), paused: false, auto: true, buffered: [], dropped: 0 };
  let entries = [];
  const chips = h('div', { class: 'chips' });
  const drawChips = () => chips.replaceChildren(...STAGES.map((s) => h('button', { class: ['chip', !st.stages.has(s) && 'on'],
    title: `${st.stages.has(s) ? 'Show' : 'Hide'}: ${STAGE_HELP[s] || s}`,
    onclick: () => { if (st.stages.has(s)) st.stages.delete(s); else st.stages.add(s); savePref('hiddenStages', [...st.stages]); drawChips(); redraw(); } },
  STAGE_LABEL[s])));
  const symIn = h('input', { class: 'input sm mono', placeholder: 'Symbol', 'aria-label': 'Filter by symbol', style: { width: '110px' }, 'data-search': '' });
  const levelSel = h('select', { class: 'input sm', 'aria-label': 'Minimum level' },
    [[0, 'All (incl. debug)'], [1, 'Info and above'], [2, 'Warnings + errors'], [3, 'Errors only']]
      .map(([v, l]) => h('option', { value: String(v), selected: v === st.minLevel }, l)));
  const pausedNote = h('span', { class: 'small warn-text' });
  const visible = (e) => !st.stages.has(e.stage) && levelRank(e.level) >= st.minLevel
    && (!st.symbol || String(e.symbol || '').toUpperCase().includes(st.symbol));
  const rowOf = (e) => activityRow(e, { onDecision: openDecision });
  const redraw = () => {
    const rows = entries.filter(visible).slice(-800);
    list.replaceChildren(...(rows.length ? rows.map(rowOf)
      : [ui.empty('Nothing yet', entries.length ? 'No events match these filters.' : 'When the engine runs, every step (scans, AI agent calls, debate, risk checks) appears here live.', 'live')]));
    if (st.auto) scroll.scrollTop = scroll.scrollHeight;
  };
  const appendEntries = (items) => {
    entries.push(...items);
    if (entries.length > MAX_ROWS) entries = entries.slice(-MAX_ROWS);
    list.querySelector('.empty')?.remove();
    for (const e of items) if (visible(e)) list.appendChild(rowOf(e));
    while (list.children.length > 800) list.firstChild.remove();
    if (st.auto) scroll.scrollTop = scroll.scrollHeight;
  };
  symIn.addEventListener('input', debounce(() => { st.symbol = symIn.value.trim().toUpperCase(); redraw(); }, 250));
  levelSel.addEventListener('change', () => { st.minLevel = Number(levelSel.value); savePref('minLevel', st.minLevel); redraw(); });
  const pauseTog = ui.toggle('Pause', false, (v) => {
    st.paused = v;
    if (!v && st.buffered.length) appendEntries(st.buffered);
    st.buffered = [];
    st.dropped = 0;
    pausedNote.textContent = '';
  });
  const bufferWhilePaused = (items) => {
    st.buffered.push(...items);
    if (st.buffered.length > MAX_ROWS) { st.dropped += st.buffered.length - MAX_ROWS; st.buffered = st.buffered.slice(-MAX_ROWS); }
    const total = st.buffered.length + st.dropped;
    pausedNote.textContent = `${total} new while paused${st.dropped ? ` (showing the last ${MAX_ROWS})` : ''}`;
  };
  const autoTog = ui.toggle('Auto-scroll', true, (v) => { st.auto = v; if (v) scroll.scrollTop = scroll.scrollHeight; });
  const activityPane = h('div', null, strip,
    h('div', { class: 'card-b row' }, chips, h('span', { class: 'grow' }), symIn, levelSel, pauseTog, autoTog, pausedNote),
    scroll);

  // ------------------------------------------------------------------ Terminal tab
  const term = h('div', { class: 'logbox', role: 'log' });
  const tst = { minLevel: pref('termLevel', 'INFO'), debug: pref('termDebug', false), q: '', auto: true };
  let termLines = [];
  const RANK = { DEBUG: 0, TRACE: 0, INFO: 1, SUCCESS: 1, WARNING: 2, ERROR: 3, CRITICAL: 3 };
  const termVisible = (e) => {
    const r = RANK[e.level] ?? 1;
    if (r === 0 && !tst.debug) return false;
    if (r !== 0 && r < (RANK[tst.minLevel] ?? 1)) return false;
    return !tst.q || `${e.source} ${e.message}`.toLowerCase().includes(tst.q);
  };
  const termRow = (e) => h('div', { class: ['logline', e.level] }, h('span', { class: 'ts' }, (e.ts || '').slice(11, 19)),
    h('span', { class: ['lv', e.level] }, e.level), h('span', { class: 'msg' }, e.source ? h('span', { class: 'src' }, `${e.source} `) : null, e.message));
  const termRedraw = () => {
    const rows = termLines.filter(termVisible).slice(-1500);
    term.replaceChildren(...(rows.length ? rows.map(termRow) : [ui.empty('No lines', termLines.length ? 'No lines match these filters.' : 'engine.log for today is empty or missing.', 'search')]));
    if (tst.auto) term.scrollTop = term.scrollHeight;
  };
  const termAppend = (items) => {
    termLines.push(...items);
    if (termLines.length > MAX_ROWS) termLines = termLines.slice(-MAX_ROWS);
    term.querySelector('.empty')?.remove();
    for (const e of items) if (termVisible(e)) term.appendChild(termRow(e));
    while (term.children.length > 1500) term.firstChild.remove();
    if (tst.auto) term.scrollTop = term.scrollHeight;
  };
  const termLevel = h('select', { class: 'input sm', 'aria-label': 'Minimum level' },
    [['INFO', 'Info and above'], ['WARNING', 'Warnings + errors'], ['ERROR', 'Errors only']]
      .map(([v, l]) => h('option', { value: v, selected: v === tst.minLevel }, l)));
  termLevel.addEventListener('change', () => { tst.minLevel = termLevel.value; savePref('termLevel', tst.minLevel); termRedraw(); });
  const termSearch = h('input', { class: 'input sm', placeholder: 'Search…', 'aria-label': 'Search terminal', style: { width: '180px' } });
  termSearch.addEventListener('input', debounce(() => { tst.q = termSearch.value.trim().toLowerCase(); termRedraw(); }, 250));
  const terminalPane = h('div', null,
    h('div', { class: 'card-b row' }, h('span', { class: 'small muted' }, 'engine.log - exactly what the engine terminal prints'), h('span', { class: 'grow' }),
      termLevel, ui.toggle('Show DEBUG', tst.debug, (v) => { tst.debug = v; savePref('termDebug', v); termRedraw(); }), termSearch,
      ui.toggle('Auto-scroll', true, (v) => { tst.auto = v; if (v) term.scrollTop = term.scrollHeight; })),
    term);

  // ------------------------------------------------------------------ tabs
  const tabs = h('div', { class: 'tabs', role: 'tablist' });
  const body = h('div');
  const drawTabs = () => {
    tabs.replaceChildren(...[['activity', 'AI activity'], ['terminal', 'Terminal']].map(([v, l]) => h('button', {
      class: tab === v ? 'on' : null, role: 'tab', 'aria-selected': String(tab === v),
      onclick: () => { tab = v; savePref('tab', v); drawTabs(); } }, l)));
    body.replaceChildren(tab === 'activity' ? activityPane : terminalPane);
    if (tab === 'activity' && st.auto) scroll.scrollTop = scroll.scrollHeight;
    if (tab === 'terminal' && tst.auto) term.scrollTop = term.scrollHeight;
  };
  root.append(h('section', { class: 'card' }, tabs, body));
  drawTabs();
  drawChips();

  // ------------------------------------------------------------------ load + live
  // The tail load races the live stream: SSE items that arrive while it is in flight are held by the
  // gate and released afterwards - only the ones newer than the last loaded line (deduped).
  const gate = new LoadGate();
  const ingestLive = (items) => {
    if (!items.length) return;
    tracker.ingest(items);
    strip.tick();
    if (st.paused) { bufferWhilePaused(items); return; }
    appendEntries(items);
  };
  const load = async () => {
    gate.begin();
    let loaded = null;
    try {
      const d = await api.get('/api/logs' + qs({ file: 'activity', lines: 300 }));
      loaded = d.entries || [];
      entries = loaded.slice(-MAX_ROWS);
      tracker.calls.clear();
      tracker.ingest(entries);
      strip.tick();
      redraw();
    } catch (err) { list.replaceChildren(ui.errorBox(err, 'Could not load the activity log')); }
    ingestLive(gate.end(loaded || []));
    try {
      const d = await api.get('/api/logs' + qs({ file: 'engine', lines: 300 }), { background: true });
      termLines = d.entries || [];
      termRedraw();
    } catch (err) { term.replaceChildren(ui.errorBox(err, 'Could not load engine.log')); }
  };
  t.track(on('sse:activity', (d) => {
    const items = d?.entries || [];
    if (gate.hold(items)) return;
    ingestLive(items);
  }));
  t.track(on('sse:log', (d) => termAppend(d.entries || [])));
  t.track(on('overview', (o) => { engine = o.engine; paintStatus(); }));
  t.track(on('sse:status', (d) => { if (d.engine) engine = d.engine; paintStatus(); }));
  t.every(1000, () => { strip.tick(); paintStatus(); });
  const onRefresh = () => load();
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  paintStatus();
  load();
  return () => t.clear();
}

