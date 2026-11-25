// Batches & capital: overall money in / out, every batch's result, start a new batch (with a live
// top-up preview), and a batch detail view (equity curve + per-mode / per-strategy breakdown).
import { api, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { go, pageTimers } from '../router.js';
import { LineChart } from '../charts.js';
import { registerChart } from '../theme.js';
import { tradeColumns } from './today.js';

export const title = (p) => (p?.id ? `Batch #${p.id}` : 'Batches & capital');

export function mount(root, params) {
  return params.id ? mountDetail(root, Number(params.id)) : mountList(root);
}

// ------------------------------------------------------------------ list
function mountList(root) {
  const t = pageTimers();
  const overall = h('div', { class: 'kpis k4' }, Array.from({ length: 4 }, () => h('div', { class: 'kpi' }, ui.skeleton(2))));
  const tableCard = ui.card('All batches', { flush: true, body: ui.skeleton(4), hint: 'click a batch for its equity curve' });
  const newCard = ui.card('Start a new batch', { cls: 'accent', body: ui.skeleton(4) });
  const eventsCard = ui.card('Money in / out', { flush: true, body: ui.skeleton(3) });
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Batches & capital'),
    h('span', { class: 'sub' }, 'each batch is a fresh paper account; history is never deleted')),
  overall, h('div', { class: 'home-grid' }, h('div', { class: 'grid' }, tableCard, eventsCard), newCard));

  let formBuilt = false;
  const load = async () => {
    let d;
    try { d = await api.get('/api/batches'); } catch (err) { overall.replaceChildren(ui.errorBox(err)); return; }
    overall.replaceChildren(
      ui.kpi({ label: 'Net invested', value: f.inr(d.net_invested), tone: 'info',
        sub: h('span', null, 'deposited ', h('b', null, f.inr(d.total_deposited)), d.source === 'fallback' ? ' · approx.' : '') }),
      ui.kpi({ label: 'Total withdrawn', value: f.inr(d.total_withdrawn), sub: 'taken out when a new batch started lower' }),
      ui.kpi({ label: 'Current capital', value: f.inr(d.current_capital), sub: d.current_batch_name ? `in ${d.current_batch_name}` : 'no active batch' }),
      ui.kpi({ label: 'Overall P&L', value: ui.money(d.overall_pnl), tone: f.kpiTone(d.overall_pnl),
        sub: h('span', null, h('b', { class: f.tone(d.overall_pnl_pct) }, f.pct(d.overall_pnl_pct, { sign: true })), ' on net invested') }));
    const rows = [...(d.batches || [])].reverse();
    tableCard.body.replaceChildren(ui.table([
      { key: 'name', label: 'Batch', fmt: (b) => ui.twoLine(h('span', { class: 'strong' }, b.name),
        ui.badge(b.status, b.status === 'ACTIVE' ? 'good' : b.status === 'HALTED' ? 'bad' : b.status === 'CLOSING' ? 'warn' : '')) },
      { key: 'starting_capital', label: 'Start capital', num: true, fmt: (b) => f.inr(b.starting_capital) },
      { key: 'top_up', label: 'Top-up', num: true, title: '+ money added / − withdrawn when the batch started',
        fmt: (b) => h('span', { class: f.tone(b.top_up) }, f.inr(b.top_up, { sign: true })) },
      { key: 'equity', label: 'Final / current', num: true, fmt: (b) => f.inr(b.equity) },
      { key: 'pnl', label: 'P&L', num: true, fmt: (b) => ui.twoLine(ui.money(b.pnl), h('span', { class: f.tone(b.pnl_pct) }, f.pct(b.pnl_pct, { sign: true }))) },
      { key: 'trades', label: 'Trades', num: true },
      { key: 'win_rate', label: 'Win rate', num: true, fmt: (b) => (f.isNum(b.win_rate) ? `${f.num(b.win_rate, 1)}%` : '–') },
      { key: 'started_at', label: 'Period', fmt: (b) => ui.twoLine(f.date(b.started_at), b.ended_at ? `→ ${f.date(b.ended_at)}` : '→ now') },
    ], rows, { onRow: (b) => go(`/batches/${b.batch_id}`), initialSort: null, empty: ui.empty('No batches yet', 'The engine creates Batch 1 on first start.') }));
    eventsCard.body.replaceChildren(ui.table([
      { key: 'ts', label: 'When', fmt: (e) => f.date(e.ts) },
      { key: 'kind', label: 'Type', fmt: (e) => ui.badge(e.kind.replace('_', ' '), e.kind === 'WITHDRAW' ? 'bad' : e.kind === 'TOP_UP' ? 'good' : 'info') },
      { key: 'amount', label: 'Amount', num: true, fmt: (e) => h('span', { class: e.kind === 'WITHDRAW' ? 'neg' : 'pos' }, `${e.kind === 'WITHDRAW' ? '−' : '+'}${f.inr(e.amount)}`) },
      { key: 'batch_id', label: 'Batch', fmt: (e) => `#${e.batch_id}` },
      { key: 'note', label: 'Note', wrap: true },
    ], [...(d.events || [])].reverse(), { sortable: false, empty: ui.empty('No money movements yet') }));
    if (!formBuilt) { newCard.body.replaceChildren(newBatchForm(d, load)); formBuilt = true; }
  };
  load();
  const onRefresh = () => load();
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  t.every(60000, load);
  return () => t.clear();
}

function newBatchForm(d, reload) {
  const capital = h('input', { class: 'input', type: 'number', min: '1', step: '0.01', value: String(d.default_starting_capital ?? 1000), inputmode: 'decimal' });
  const name = h('input', { class: 'input', placeholder: d.suggested_name || 'Batch N', maxlength: '120' });
  const confirm = h('input', { class: 'input mono', placeholder: 'Type CONFIRM', autocomplete: 'off', spellcheck: 'false' });
  const preview = h('div', null, ui.skeleton(2));
  const submit = h('button', { class: 'btn primary block lg', disabled: true }, 'Start new batch');
  let cnc = 'EXIT';
  const cncCards = h('div', { class: 'radio-cards' });
  const drawCnc = () => cncCards.replaceChildren(...[
    ['EXIT', 'Exit open positions', 'All open positions of the current batch are closed at the next available price.'],
    ['RUN_TO_COMPLETION', 'Let delivery (CNC) trades finish', 'Intraday positions exit now; swing/positional CNC positions keep running under the old batch until their own exit.'],
  ].map(([v, tt, dd]) => h('label', { class: ['radio-card', cnc === v && 'on'] },
    h('input', { type: 'radio', name: 'cnc', value: v, checked: cnc === v, onchange: () => { cnc = v; drawCnc(); } }),
    h('div', null, h('div', { class: 't' }, tt), h('div', { class: 'd' }, dd)))));
  drawCnc();

  const refreshPreview = debounce(async () => {
    const v = Number(capital.value);
    if (!Number.isFinite(v) || v <= 0) { preview.replaceChildren(ui.notice('warn', 'Enter a starting capital above ₹0')); return; }
    try {
      const p = await api.get('/api/batches/topup-preview' + qs({ capital: v }), { background: true });
      preview.replaceChildren(previewBox(p, false));
    } catch (err) {
      if (err.status === 501 && err.body?.estimate) preview.replaceChildren(previewBox(err.body.estimate, true));
      else preview.replaceChildren(ui.errorBox(err, 'Preview failed'));
    }
  }, 350);
  const validate = () => {
    const ok = confirm.value.trim() === 'CONFIRM' || (d.current_batch_name && confirm.value.trim() === d.current_batch_name);
    submit.disabled = !ok || !(Number(capital.value) > 0);
  };
  capital.addEventListener('input', () => { refreshPreview(); validate(); });
  confirm.addEventListener('input', validate);
  submit.addEventListener('click', async () => {
    const body = { starting_capital: Number(capital.value), name: name.value.trim() || null, open_cnc_handling: cnc, confirm_text: confirm.value.trim() };
    const res = await ui.sendCommand(submit, 'New batch', () => api.post('/api/batches/new', body), { onDone: () => reload() });
    if (res) { confirm.value = ''; validate(); ui.toast(`"${res.name}" with ${f.inr(res.starting_capital)} queued.`, 'info'); }
  });
  refreshPreview();
  const last = d.last_new_batch_command;
  return h('div', { class: 'stack' },
    h('div', { class: 'small muted' }, `Archives ${d.current_batch_name || 'the current batch'} (its history stays) and starts a fresh paper account. The engine executes it.`),
    h('label', { class: 'field' }, h('span', null, 'Starting capital (₹)'), capital, h('span', { class: 'help' }, 'Default from settings; change it freely.')),
    preview,
    h('div', { class: 'field' }, h('span', null, 'Open positions'), cncCards),
    h('label', { class: 'field' }, h('span', null, 'Name (optional)'), name),
    h('label', { class: 'field' }, h('span', null, 'Type CONFIRM to proceed'), confirm),
    submit,
    last ? h('div', { class: 'tiny faint' }, `Last request: #${last.id} ${last.status}${last.message ? ' - ' + last.message : ''} (${f.dateTime(last.created_at)})`) : null);
}

function previewBox(p, estimate) {
  const tone = p.action === 'ADD' ? 'info' : p.action === 'WITHDRAW' ? 'warn' : 'good';
  return h('div', { class: 'stack' },
    ui.notice(tone, p.message || '', null),
    h('div', { class: 'kv' },
      h('div', { class: 'k' }, 'Current equity'), h('div', { class: 'v' }, f.inr(p.current_equity)),
      h('div', { class: 'k' }, 'New starting capital'), h('div', { class: 'v' }, f.inr(p.starting_capital)),
      h('div', { class: 'k' }, p.action === 'WITHDRAW' ? 'Withdrawn' : 'Top-up'), h('div', { class: ['v', f.tone(p.top_up)] }, f.inr(p.top_up, { sign: true })),
      h('div', { class: 'k' }, 'Open positions'), h('div', { class: 'v' }, String(p.open_positions ?? 0))),
    estimate ? h('div', { class: 'tiny faint' }, 'Estimate (funding module not available yet).') : null);
}

// ------------------------------------------------------------------ detail
function mountDetail(root, id) {
  const t = pageTimers();
  const head = h('div', { class: 'page-head' }, h('a', { href: '#/batches', class: 'btn sm ghost' }, icon('arrowLeft', 14), 'Batches'), h('h1', null, `Batch #${id}`));
  const kpis = h('div', { class: 'kpis' }, Array.from({ length: 6 }, () => h('div', { class: 'kpi' }, ui.skeleton(2))));
  const box = h('div', { class: 'chart-box medium' });
  const eqCard = ui.card('Equity curve', { flush: true, body: box, hint: 'start → ledger / end-of-day points → live' });
  const modeCard = ui.card('By mode', { flush: true, body: ui.skeleton(3) });
  const stratCard = ui.card('By strategy', { flush: true, body: ui.skeleton(3) });
  const recentCard = ui.card('Recent trades', { flush: true, body: ui.skeleton(4) });
  root.append(head, kpis, eqCard, h('div', { class: 'grid cols-2' }, modeCard, stratCard), recentCard);
  const chart = new LineChart(box);
  t.track(registerChart(chart));
  const breakdownCols = (key, label) => [
    { key, label, fmt: (r) => h('span', { class: 'strong' }, r[key]) }, { key: 'trades', label: 'Trades', num: true },
    { key: 'win_rate', label: 'Win %', num: true, fmt: (r) => (f.isNum(r.win_rate) ? f.num(r.win_rate, 1) : '–') },
    { key: 'net', label: 'Net', num: true, fmt: (r) => ui.money(r.net) }, { key: 'charges', label: 'Charges', num: true, fmt: (r) => f.inr(r.charges) },
    { key: 'avg_r', label: 'Avg R', num: true, fmt: (r) => f.rmult(r.avg_r) },
  ];
  (async () => {
    let d;
    try { d = await api.get(`/api/batches/${id}`); } catch (err) { kpis.replaceChildren(ui.errorBox(err)); return; }
    const b = d.batch; const c = d.capital || {}; const tt = d.totals;
    head.replaceChildren(h('a', { href: '#/batches', class: 'btn sm ghost' }, icon('arrowLeft', 14), 'Batches'), h('h1', null, b.name),
      ui.badge(b.status, b.status === 'ACTIVE' ? 'good' : ''), h('span', { class: 'sub' }, `${f.date(b.started_at)} → ${b.ended_at ? f.date(b.ended_at) : 'now'}`));
    kpis.replaceChildren(
      ui.kpi({ label: 'Starting capital', value: f.inr(c.starting_capital ?? b.starting_capital),
        sub: d.funding && f.isNum(d.funding.top_up) ? `top-up ${f.inr(d.funding.top_up, { sign: true })}` : null }),
      ui.kpi({ label: b.status === 'ACTIVE' ? 'Equity now' : 'Final equity', value: f.inr(c.equity), sub: `reserve ${f.inr(c.reserve)} · unrealized ${f.inr(c.unrealized, { sign: true })}` }),
      ui.kpi({ label: 'P&L', value: ui.money(c.pnl), tone: f.kpiTone(c.pnl), sub: h('span', { class: f.tone(c.pnl_pct) }, f.pct(c.pnl_pct, { sign: true })) }),
      ui.kpi({ label: 'Trades', value: String(tt.closed), sub: `${tt.wins} won · ${tt.losses} lost` }),
      ui.kpi({ label: 'Win rate', value: f.isNum(tt.win_rate) ? `${f.num(tt.win_rate, 1)}%` : '–', sub: f.isNum(tt.profit_factor) ? `profit factor ${f.num(tt.profit_factor, 2)}` : null }),
      ui.kpi({ label: 'Drawdown now', value: f.isNum(c.drawdown_pct) ? `${f.num(c.drawdown_pct, 2)}%` : '–', sub: `tier ${c.tier || '–'} · HWM ${f.inr(c.hwm)}` }));
    chart.baseline = c.starting_capital ?? b.starting_capital;
    chart.setData(d.equity_curve);
    modeCard.body.replaceChildren(ui.table(breakdownCols('mode', 'Mode'), d.by_mode, { empty: ui.empty('No trades yet') }));
    stratCard.body.replaceChildren(ui.table(breakdownCols('strategy', 'Strategy'), d.by_strategy, { empty: ui.empty('No trades yet') }));
    recentCard.body.replaceChildren(ui.table(tradeColumns({ withDate: true }), d.recent_trades, { onRow: (r) => go(`/trade/${r.position_id}`),
      empty: ui.empty('No completed trades in this batch') }));
  })();
  return () => { t.clear(); chart.destroy(); };
}
