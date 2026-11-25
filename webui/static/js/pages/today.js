// Today (or any day): summary strip, trade completed log, open positions, decision funnel,
// post-mortem "what went wrong", event timeline and the day's equity curve.
import { api, on, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { go, pageTimers } from '../router.js';
import { LineChart } from '../charts.js';
import { registerChart } from '../theme.js';
import { openDecision } from '../components/decision.js';
import { positionCard } from '../components/positions.js';

export const title = 'Today';

export function tradeColumns({ withDate = false, withBatch = false } = {}) {
  const tm = withDate ? f.dateTime : f.hm;
  return [
    withBatch ? { key: 'batch_id', label: 'Batch', fmt: (r) => `#${r.batch_id}` } : null,
    { key: 'symbol', label: 'Name', fmt: (r) => ui.twoLine(h('span', { class: 'sym' }, r.symbol), `${r.direction === 'SHORT' ? 'Short' : 'Long'} · ${r.grade || '–'}`) },
    { key: 'mode', label: 'Mode · strategy', fmt: (r) => ui.twoLine(ui.modeBadge(r.mode), r.strategy || 'manual') },
    { key: 'entry_time', label: 'Entry', fmt: (r) => ui.twoLine(f.price(r.entry_price), tm(r.entry_time)) },
    { key: 'exit_time', label: 'Exit', fmt: (r) => ui.twoLine(f.price(r.exit_price), `${tm(r.exit_time)}${r.legs > 1 ? ` · ${r.legs} legs` : ''}`) },
    { key: 'initial_sl', label: 'Stop-loss', num: true, fmt: (r) => ui.twoLine(f.price(r.initial_sl),
      f.isNum(r.final_sl) && r.final_sl !== r.initial_sl ? `final ${f.price(r.final_sl)}` : null) },
    { key: 'target_1', label: 'T1 / T2', num: true, fmt: (r) => ui.twoLine(f.price(r.target_1), f.isNum(r.target_2) ? f.price(r.target_2) : null) },
    { key: 'exit_reason', label: 'Exit reason', wrap: true, fmt: (r) => r.exit_reason || '–' },
    { key: 'qty', label: 'Qty', num: true },
    { key: 'gross', label: 'Gross', num: true, fmt: (r) => ui.money(r.gross) },
    { key: 'charges', label: 'Charges', num: true, fmt: (r) => f.inr(r.charges) },
    { key: 'net', label: 'Net P&L', num: true, fmt: (r) => h('b', null, ui.money(r.net)) },
    { key: 'r_multiple', label: 'R', num: true, fmt: (r) => h('span', { class: f.tone(r.r_multiple) }, f.rmult(r.r_multiple)) },
    { key: 'holding_minutes', label: 'Duration', num: true, fmt: (r) => f.duration(r.holding_minutes) },
    { key: 'result', label: 'Result', fmt: (r) => ui.resultBadge(r.result) },
  ].filter(Boolean);
}

export function mount(root, params) {
  const t = pageTimers();
  let day = params.date || f.todayIST();
  const dateInput = h('input', { type: 'date', class: 'input sm', value: day, max: f.todayIST(), 'aria-label': 'Day' });
  const strip = h('div', { class: 'strip' }, ui.skeleton(2));
  const tradesCard = ui.card('Trade completed log', { hint: 'one row per trade (all exit legs combined) · click a row for the full story', flush: true, body: ui.skeleton(5) });
  const openCard = ui.card('Open positions', { body: ui.skeleton(2) });
  const funnelCard = ui.card('Decisions funnel', { hint: 'how many stocks made it through each stage', body: ui.skeleton(5) });
  const pmCard = ui.card('What went wrong today', { hint: 'post-mortem · top issues and the fix', body: ui.skeleton(3) });
  const tlCard = ui.card('Timeline', { hint: 'orders, exits, stop moves, risk rejections, approvals', body: ui.skeleton(6) });
  const eqBox = h('div', { class: 'chart-box small' });
  const eqCard = ui.card('Equity through the day', { hint: 'realized (capital ledger)', flush: true, body: eqBox });
  const eqChart = new LineChart(eqBox);
  t.track(registerChart(eqChart));

  const setDay = (d) => {
    day = d;
    dateInput.value = d;
    history.replaceState(null, '', d === f.todayIST() ? '#/today' : `#/today?date=${d}`);
    load(false);
  };
  dateInput.addEventListener('change', () => dateInput.value && setDay(dateInput.value));
  root.append(
    h('div', { class: 'page-head' }, h('h1', null, 'Day report'),
      h('div', { class: 'row' },
        h('button', { class: 'btn sm', title: 'Previous day', onclick: () => setDay(f.shiftDay(day, -1)) }, icon('chevronLeft', 14)),
        dateInput,
        h('button', { class: 'btn sm', title: 'Next day', onclick: () => { const n = f.shiftDay(day, 1); if (n <= f.todayIST()) setDay(n); } }, icon('chevronRight', 14)),
        h('button', { class: 'btn sm', onclick: () => setDay(f.todayIST()) }, 'Today')),
      h('span', { class: 'grow' }), h('span', { class: 'sub', id: 'today-sub' }, '')),
    strip, tradesCard,
    h('div', { class: 'grid cols-2' }, pmCard, funnelCard),
    h('div', { class: 'grid cols-2' }, h('div', { class: 'grid' }, openCard, eqCard), tlCard));

  const load = async (background = true) => {
    let d;
    try {
      d = await api.get('/api/today' + qs({ date: day }), { background });
    } catch (err) {
      strip.replaceChildren(ui.errorBox(err, 'Could not load the day'));
      return;
    }
    const s = d.summary;
    document.getElementById('today-sub').textContent = `${f.weekday(day + 'T12:00:00+05:30')} · updated ${f.time(d.server_time)} IST`;
    const cell = (label, value, sub = null) => h('div', { class: 'cell' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value),
      sub ? h('div', { class: 'sub' }, sub) : null);
    strip.replaceChildren(
      cell('Net P&L (realized)', ui.money(s.net), `gross ${f.inr(s.gross, { sign: true })}`),
      d.is_today ? cell('Unrealized now', ui.money(s.unrealized), `${s.open} open position(s)`) : null,
      cell('Trades', String(s.closed), `${s.wins} won · ${s.losses} lost${s.flat ? ` · ${s.flat} flat` : ''}`),
      cell('Win rate', f.isNum(s.win_rate) ? `${f.num(s.win_rate, 1)}%` : '–', f.isNum(s.avg_r) ? `avg ${f.rmult(s.avg_r)}` : null),
      cell('Charges', f.inr(s.charges), 'brokerage, STT, GST, stamp…'),
      cell('Best', s.best ? h('span', { class: 'pos' }, f.inr(s.best.net, { sign: true })) : '–', s.best?.symbol),
      cell('Worst', s.worst ? h('span', { class: 'neg' }, f.inr(s.worst.net, { sign: true })) : '–', s.worst?.symbol));

    tradesCard.head.querySelector('h2').textContent = `Trade completed log (${d.trades.length})`;
    const tot = d.trades.length ? { symbol: 'Total', gross: ui.money(s.gross), charges: f.inr(s.charges), net: h('b', null, ui.money(s.net)),
      r_multiple: f.isNum(s.avg_r) ? `avg ${f.rmult(s.avg_r)}` : '' } : null;
    tradesCard.body.replaceChildren(ui.table(tradeColumns(), d.trades, {
      onRow: (r) => go(`/trade/${r.position_id}`), footer: tot, initialSort: { key: 'exit_time', dir: -1 },
      empty: ui.empty(d.is_today ? 'No completed trades yet today' : 'No completed trades on this day',
        'Closed positions appear here with entry, exit, stop-loss, targets, charges and net P&L.', 'trades') }));

    openCard.head.querySelector('h2').textContent = `${d.is_today ? 'Open positions' : 'Still open from this day'} (${d.open_positions.length})`;
    openCard.body.replaceChildren(d.open_positions.length
      ? h('div', { class: 'pos-list' }, d.open_positions.map((p) => positionCard(p, { onClick: () => go(`/trade/${p.id}`) })))
      : ui.empty('No open positions', ''));

    funnelCard.body.replaceChildren(renderFunnel(d.funnel));
    tlCard.body.replaceChildren(renderTimeline(d.timeline));
    eqChart.setData(d.equity);
    loadPostmortem();
  };

  const loadPostmortem = async () => {
    try {
      const pm = await api.get('/api/postmortem/day' + qs({ date: day }), { background: true });
      pmCard.body.replaceChildren(renderIssues(pm));
    } catch (err) {
      pmCard.body.replaceChildren(err.status === 501
        ? ui.notice('info', 'Post-mortem analysis is not available yet', 'Each losing trade will be diagnosed here (stop too tight, late entry, …) with the fix. Trade-level details already work: click a trade above.')
        : ui.errorBox(err, 'Post-mortem failed'));
    }
  };

  const soon = debounce(() => { if (day === f.todayIST()) load(); }, 1500);
  t.track(on('sse:trade', soon));
  t.track(on('sse:order', soon));
  const onRefresh = () => load(false);
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  t.every(30000, () => { if (day === f.todayIST()) load(); });
  load(false);
  return () => { t.clear(); eqChart.destroy(); };
}

export function renderFunnel(fn) {
  if (!fn || !fn.counts?.TOTAL) return ui.empty('No stocks analysed on this day', 'The funnel shows how many stocks passed each stage.', 'live');
  const max = Math.max(1, ...fn.stages.map((s) => s.count));
  const tone = { risk_rejected: 'bad', executed: 'good', queued: 'warn' };
  return h('div', { class: 'stack' },
    h('div', { class: 'funnel' }, fn.stages.map((s) => h('div', { class: 'stage', title: s.help },
      h('div', { class: 'name' }, s.label, h('span', { class: 'h' }, s.help)),
      h('div', { class: ['bar', tone[s.key]] }, h('i', { style: { width: `${(s.count / max) * 100}%` } })),
      h('div', { class: 'n' }, String(s.count))))),
    h('div', { class: 'small faint' }, `${fn.scanned} scan rows · NO TRADE (AI said no) ${fn.counts.NO_TRADE} · filtered ${fn.counts.FILTERED}`),
    fn.risk_rules_top?.length ? h('div', null, h('div', { class: 'section-title' }, 'Top risk rules that blocked trades'),
      h('div', { class: 'chips' }, fn.risk_rules_top.map((r) => ui.badge(`${r.rule} × ${r.count}`, 'bad')))) : null,
    fn.filter_reasons_top?.length ? h('div', null, h('div', { class: 'section-title' }, 'Top filter reasons'),
      h('div', { class: 'chips' }, fn.filter_reasons_top.map((r) => ui.badge(`${r.reason} × ${r.count}`)))) : null);
}

export function renderIssues(pm) {
  const issues = pm?.top_issues || [];
  const sm = pm?.summary || {};
  const head = h('div', { class: 'small muted' }, `${sm.trades ?? 0} trade(s) analysed · ${sm.wins ?? 0} won · ${sm.losses ?? 0} lost`,
    f.isNum(sm.net_pnl) ? h('span', null, ' · net ', h('b', { class: f.tone(sm.net_pnl) }, f.inr(sm.net_pnl, { sign: true }))) : null);
  if (!issues.length) return h('div', { class: 'stack' }, head, ui.notice('good', 'No recurring problems found', 'Nothing systematic went wrong on this day.'));
  return h('div', { class: 'stack' }, head, h('div', { class: 'issues' }, issues.map((i) => h('div', { class: ['issue', i.count >= 3 ? 'high' : i.count === 2 ? 'medium' : 'low'] },
    h('div', { class: 'head' }, ui.badge(`× ${i.count}`, i.count >= 3 ? 'bad' : 'warn'), h('span', { class: 'title' }, i.title || i.code)),
    i.fix ? h('div', { class: 'fix' }, h('b', null, 'Fix: '), i.fix) : null))));
}

export function renderTimeline(items) {
  if (!items?.length) return ui.empty('Nothing happened yet', 'Orders, exits, stop-loss moves and risk rejections appear here.', 'today');
  const list = h('div', { class: 'timeline' });
  for (const e of [...items].reverse()) {
    const ref = e.ref || {};
    const click = ref.type === 'position' && ref.id ? () => go(`/trade/${ref.id}`) : ref.type === 'decision' && ref.id ? () => openDecision(ref.id) : null;
    list.appendChild(h('div', { class: ['tl-item', e.tone, click && 'click'], onclick: click },
      h('div', { class: 't' }, f.hm(e.ts)), h('div', { class: 'rail' }, h('span', { class: 'd' })),
      h('div', null, h('div', { class: 'row' }, h('span', { class: 'strong' }, e.symbol || ''), ui.badge(e.kind.replace('_', ' '), ''), h('span', { class: 'title' }, e.title)),
        e.detail ? h('div', { class: 'detail' }, e.detail) : null)));
  }
  return h('div', { class: 'feed-scroll' }, list);
}
