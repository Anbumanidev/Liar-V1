// Live: analysis feed (every analysed stock) | candlestick chart | open positions + watch list.
import { api, on, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { pageTimers } from '../router.js';
import { CandleChart } from '../charts.js';
import { registerChart } from '../theme.js';
import { openDecision } from '../components/decision.js';
import { positionCard } from '../components/positions.js';
import { store } from '../store.js';

export const title = (p) => (p?.symbol ? `${p.symbol} · Live` : 'Live');

const TF_INTRADAY = [['1m', '1m'], ['5m', '5m'], ['10m', '10m'], ['15m', '15m'], ['20m', '20m'], ['30m', '30m']];
const TF_LONG = [['1h', '1H'], ['5h', '5H'], ['1d', '1D'], ['1wk', '1W']];
const ACTIONS = ['EXECUTED', 'QUEUED', 'RISK_REJECTED', 'NO_TRADE', 'ANALYZED', 'SIGNAL_ONLY', 'FILTERED'];
const SYM_RE = /^[A-Z0-9][A-Z0-9&\-_.^]{0,23}$/;

function pref(key, dflt) {
  try { const v = localStorage.getItem(`liar.live.${key}`); return v === null ? dflt : JSON.parse(v); } catch { return dflt; }
}
function savePref(key, v) { try { localStorage.setItem(`liar.live.${key}`, JSON.stringify(v)); } catch { /* ignore */ } }

export function mount(root, params) {
  const t = pageTimers();
  const st = {
    symbol: (params.symbol || pref('symbol', '') || '').toUpperCase(),
    tf: params.tf || pref('tf', '10m'),
    toggles: pref('toggles', { ema9: true, ema21: true, vwap: true, volume: true }),
    hidden: new Set(pref('hiddenActions', [])),
    oldest: null,
  };

  // ------------------------------------------------------------------ feed column
  const feedList = h('div', { class: 'feed' });
  const chips = h('div', { class: 'chips' });
  const counts = Object.fromEntries(ACTIONS.map((a) => [a, 0]));
  const drawChips = () => chips.replaceChildren(...ACTIONS.filter((a) => !['ANALYZED', 'SIGNAL_ONLY'].includes(a) || counts[a] > 0).map((a) => h('button', {
    class: ['chip', !st.hidden.has(a) && 'on'], title: st.hidden.has(a) ? `Show ${a}` : `Hide ${a}`,
    onclick: () => { if (st.hidden.has(a)) st.hidden.delete(a); else st.hidden.add(a); savePref('hiddenActions', [...st.hidden]); drawChips(); applyFilter(); },
  }, ui.ACTION_LABEL[a], h('span', { class: 'n' }, String(counts[a] || 0)))));
  const applyFilter = () => {
    for (const el of feedList.children) el.classList.toggle('hidden', st.hidden.has(el.dataset.action));
  };
  const moreBtn = h('button', { class: 'btn sm block' }, 'Load older');
  const feedCard = ui.card('Analysis feed', { hint: 'every analysed stock, newest first', flush: true,
    body: h('div', null, h('div', { class: 'card-b' }, chips), h('div', { class: 'feed-scroll' }, feedList, h('div', { class: 'card-b' }, moreBtn))) });

  const feedRow = (r, isNew) => h('div', { class: ['feed-item', isNew && 'new', st.hidden.has(r.final_action) && 'hidden'],
    dataset: { action: r.final_action, id: String(r.id) },
    onclick: () => { openDecision(r.id); setSymbol(r.symbol); } },
  h('div', { class: 't' }, f.hm(r.time)),
  h('div', { class: 'body' },
    h('div', { class: 'line1' }, h('span', { class: 'sym' }, r.symbol), ui.dirBadge(r.direction),
      h('span', { class: 'meta' }, [r.strategy, r.mode].filter(Boolean).join(' · ')), ui.gradeBadge(r.grade, r.confluence),
      ui.actionChip(r.final_action)),
    h('div', { class: 'why clamp-2' }, r.reason || '–')));

  const loadFeed = async (older = false) => {
    try {
      const d = await api.get('/api/decisions' + qs({ limit: 80, before_id: older ? st.oldest : null }), { background: !older });
      const rows = d.rows || [];
      if (!older) { feedList.replaceChildren(); for (const a of ACTIONS) counts[a] = 0; }
      for (const r of rows) { feedList.appendChild(feedRow(r, false)); counts[r.final_action] = (counts[r.final_action] || 0) + 1; }
      if (rows.length) st.oldest = rows[rows.length - 1].id;
      moreBtn.classList.toggle('hidden', rows.length < 80);
      if (!feedList.children.length) feedList.replaceChildren(ui.empty('No analysis yet', 'When the engine scans stocks, each one appears here with its outcome and the reason.', 'live'));
      drawChips();
      if (!st.symbol && rows[0]) setSymbol(rows.find((r) => r.final_action !== 'FILTERED')?.symbol || rows[0].symbol);
    } catch (err) { feedList.replaceChildren(ui.errorBox(err)); }
  };
  moreBtn.addEventListener('click', () => ui.busy(moreBtn, () => loadFeed(true)));

  // ------------------------------------------------------------------ chart column
  const symInput = h('input', { class: 'input sm mono', placeholder: 'Symbol e.g. RELIANCE', value: st.symbol, list: 'live-symbols',
    'data-search': '', autocomplete: 'off', spellcheck: 'false', style: { width: '170px' }, 'aria-label': 'Symbol search' });
  const datalist = h('datalist', { id: 'live-symbols' });
  symInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') setSymbol(symInput.value); });
  symInput.addEventListener('change', () => setSymbol(symInput.value));
  const tfBtns = new Map();
  const tfGroup = (label, items) => h('div', { class: 'row nowrap' }, h('span', { class: 'grp-label' }, label),
    h('div', { class: 'seg sm' }, items.map(([v, l]) => { const b = h('button', { type: 'button', onclick: () => setTf(v) }, l); tfBtns.set(v, b); return b; })));
  const tog = (k, label) => ui.toggle(label, st.toggles[k], (v) => { st.toggles[k] = v; savePref('toggles', st.toggles); chart.setToggles({ [k]: v }); });
  const head = h('div', { class: 'chart-head' });
  const modeSel = h('select', { class: 'input sm', style: { width: '120px' }, 'aria-label': 'Mode for analysis' },
    ['INTRADAY', 'BTST', 'SWING', 'POSITIONAL'].map((m) => h('option', { value: m }, m)));
  const analyzeBtn = h('button', { class: 'btn sm', title: 'Ask the engine to analyse this symbol now (analysis only, no trade)' }, icon('bolt', 13), 'Analyze now');
  analyzeBtn.addEventListener('click', () => {
    if (!st.symbol) return;
    ui.sendCommand(analyzeBtn, `Analyze ${st.symbol}`, () => api.post('/api/control', { action: 'analyze_now', symbol: st.symbol, mode: modeSel.value }));
  });
  const box = h('div', { class: 'chart-box' });
  const chartCard = h('section', { class: 'card chart-card' },
    h('div', { class: 'card-h' }, head, h('div', { class: 'actions' }, modeSel, analyzeBtn)),
    h('div', { class: 'chart-tools' }, symInput, datalist,
      h('div', { class: 'tf-groups' }, tfGroup('Intraday', TF_INTRADAY), tfGroup('Long-term', TF_LONG)),
      h('span', { class: 'grow' }),
      h('div', { class: 'row nowrap' }, tog('ema9', 'EMA 9'), tog('ema21', 'EMA 21'), tog('vwap', 'VWAP'), tog('volume', 'Volume'))),
    box,
    h('div', { class: 'card-b' }, h('div', { class: 'legend-keys' },
      h('span', { class: 'ema9' }, h('i'), 'EMA 9'), h('span', { class: 'ema21' }, h('i'), 'EMA 21'), h('span', { class: 'vwap' }, h('i'), 'VWAP (intraday)'),
      h('span', { class: 'ent' }, h('i'), 'Entry'), h('span', { class: 'sl' }, h('i'), 'Stop-loss'), h('span', { class: 'tgt' }, h('i'), 'Targets'),
      h('span', { class: 'faint' }, '▲▼ trade entries/exits · times in IST · closed bars only'))));
  const chart = new CandleChart(box, { intraday: true });
  chart.setToggles(st.toggles);
  t.track(registerChart(chart));

  let loadSeq = 0;
  const loadChart = async (fit = false) => {
    if (!st.symbol) { chart.setMessage('Pick a symbol from the feed or type one above.'); paintHead(null); return; }
    const my = ++loadSeq;
    if (fit) chart.setMessage('Loading…');
    try {
      const d = await api.get('/api/candles' + qs({ symbol: st.symbol, tf: st.tf, bars: 300 }), { background: !fit });
      if (my !== loadSeq) return;
      await chart.ready;
      chart.setIntraday(d.intraday);
      const extra = d.quote && f.isNum(d.quote.ltp) ? [{ price: d.quote.ltp, kind: 'ltp', title: 'LTP' }] : [];
      chart.setData(d, { keepRange: !fit, extraLines: extra });
      paintHead(d);
    } catch (err) {
      if (my !== loadSeq) return;
      chart.setMessage(err.status === 503 ? 'Market data service unavailable (check internet / data provider).' : `Could not load candles: ${err.message}`);
    }
  };
  const paintHead = (d) => {
    const q = d?.quote;
    head.replaceChildren(h('span', { class: 'sym' }, st.symbol || '—'),
      q ? h('span', { class: 'ltp' }, f.price(q.ltp)) : null,
      q && f.isNum(q.change_pct) ? h('span', { class: ['strong', f.tone(q.change)] }, `${f.signed(q.change)} (${f.pct(q.change_pct, { sign: true })})`) : null,
      d ? h('span', { class: 'faint small' }, `${d.tf.toUpperCase()} · ${d.bars} bars · as of ${f.time(d.as_of)} IST`) : null);
  };
  const setTf = (tf) => {
    st.tf = tf; savePref('tf', tf);
    for (const [v, b] of tfBtns) b.classList.toggle('on', v === tf);
    loadChart(true);
  };
  const setSymbol = (raw) => {
    const s = String(raw || '').trim().toUpperCase().replace(/\.NS$/, '');
    if (!SYM_RE.test(s)) { if (raw) ui.toast(`"${raw}" is not a valid NSE symbol`, 'warn'); return; }
    if (s === st.symbol) return;
    st.symbol = s; savePref('symbol', s);
    symInput.value = s;
    history.replaceState(null, '', `#/live?symbol=${encodeURIComponent(s)}`);
    loadChart(true);
  };
  for (const [v, b] of tfBtns) b.classList.toggle('on', v === st.tf);

  // ------------------------------------------------------------------ side column
  const posBody = h('div', null, ui.skeleton(3));
  const watchBody = h('div', null, ui.skeleton(5));
  const posCard = ui.card('Open positions', { body: posBody });
  const watchCard = ui.card('Watch list', { hint: 'recently analysed', body: watchBody, flush: true });
  const renderPositions = (o) => {
    const rows = o?.positions || [];
    posCard.head.querySelector('h2').textContent = `Open positions (${rows.length})`;
    posBody.replaceChildren(rows.length ? h('div', { class: 'pos-list' }, rows.map((p) => positionCard(p, { compact: true, onClick: () => setSymbol(p.symbol) })))
      : ui.empty('No open positions', '', 'chart'));
  };
  const loadWatch = async () => {
    try {
      const w = await api.get('/api/watchlist?limit=24', { background: true });
      datalist.replaceChildren(...w.rows.map((r) => h('option', { value: r.symbol })));
      watchBody.replaceChildren(ui.table([
        { key: 'symbol', label: 'Symbol', fmt: (r) => h('span', { class: 'sym' }, r.symbol, r.open_position ? ' ●' : '') },
        { key: 'ltp', label: 'LTP', num: true, sort: (r) => r.quote?.ltp, fmt: (r) => f.price(r.quote?.ltp) },
        { key: 'chg', label: 'Chg %', num: true, sort: (r) => r.quote?.change_pct,
          fmt: (r) => h('span', { class: f.tone(r.quote?.change_pct) }, f.pct(r.quote?.change_pct, { sign: true })) },
        { key: 'final_action', label: 'Last', fmt: (r) => (r.final_action ? ui.actionChip(r.final_action) : null) },
      ], w.rows, { onRow: (r) => setSymbol(r.symbol), sortable: true, empty: ui.empty('Nothing analysed yet') }));
    } catch (err) { watchBody.replaceChildren(ui.errorBox(err)); }
  };

  root.append(h('div', { class: 'live-grid' },
    feedCard,
    chartCard,
    h('div', { class: 'live-side grid' }, posCard, watchCard)));

  // ------------------------------------------------------------------ live wiring
  const chartSoon = debounce(() => loadChart(false), 1500);
  t.track(on('sse:decision', (r) => {
    counts[r.final_action] = (counts[r.final_action] || 0) + 1;
    feedList.querySelector('.empty')?.remove();
    feedList.prepend(feedRow(r, true));
    while (feedList.children.length > 400) feedList.lastChild.remove();
    drawChips();
  }));
  t.track(on('sse:order', (o) => { if (o.symbol === st.symbol) chartSoon(); }));
  t.track(on('sse:trade', (tr) => { if (tr.symbol === st.symbol) chartSoon(); }));
  t.track(on('overview', renderPositions));
  const onRefresh = () => { loadFeed(); loadChart(false); loadWatch(); };
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  t.every(20000, () => loadChart(false));
  t.every(30000, loadWatch);

  if (store.overview) renderPositions(store.overview);
  paintHead(null);
  loadFeed();
  loadWatch();
  if (st.symbol) loadChart(true);
  return () => { t.clear(); chart.destroy(); };
}

