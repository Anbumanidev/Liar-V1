// Scope-aware KPI block (Today | Batch | Overall) built from GET /api/summary.
import { h } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';

/** Money context row (4 tiles) - differs per scope. */
export function moneyTiles(d) {
  const m = d.money || {};
  const k = d.kpis || {};
  if (m.kind === 'today') {
    const names = (m.batches || []).map((b) => `${b.name} (${f.inr(b.net, { sign: true })})`).join(', ');
    return [
      ui.kpi({ label: `${d.scope.label === 'Today' ? "Today's" : 'Day'} P&L`, value: ui.money(m.day_pnl), tone: f.kpiTone(m.day_pnl),
        sub: h('span', null, 'realized ', h('b', { class: f.tone(m.realized) }, f.inr(m.realized, { sign: true })),
          ' + unrealized ', h('b', { class: f.tone(m.unrealized) }, f.inr(m.unrealized, { sign: true }))) }),
      ui.kpi({ label: 'Realized (closed trades)', value: ui.money(k.realized), tone: f.kpiTone(k.realized),
        sub: `gross ${f.inr(k.gross, { sign: true })} · charges ${f.inr(k.charges)}` }),
      ui.kpi({ label: 'Unrealized (open now)', value: ui.money(k.unrealized), sub: `${k.open_positions} open position(s)` }),
      ui.kpi({ label: 'Batches that traded', value: String((m.batches || []).length), tone: 'info', sub: names || 'no trades on this day' }),
    ];
  }
  if (m.kind === 'batch') {
    const topUp = f.isNum(m.top_up) ? (m.funding_kind === 'INITIAL' || m.carried_in === null ? `initial deposit ${f.inr(m.top_up)}`
      : `${m.top_up >= 0 ? 'added' : 'withdrawn'} ${f.inr(Math.abs(m.top_up))} on top of carried-in ${f.inr(m.carried_in)}`) : '–';
    return [
      ui.kpi({ label: 'Starting capital', value: f.inr(m.starting_capital), tone: 'info', sub: topUp }),
      ui.kpi({ label: m.ended_at ? 'Final equity' : 'Equity now', value: f.inr(m.equity),
        sub: `reserve ${f.inr(m.reserve)}${f.isNum(m.drawdown_pct) ? ` · drawdown ${f.num(m.drawdown_pct, 2)}% (${m.tier})` : ''}` }),
      ui.kpi({ label: 'Batch P&L', value: ui.money(m.pnl), tone: f.kpiTone(m.pnl),
        sub: h('span', null, h('b', { class: f.tone(m.pnl_pct) }, f.pct(m.pnl_pct, { sign: true })), ' on starting capital') }),
      ui.kpi({ label: 'Status', value: m.status || '–', tone: m.status === 'ACTIVE' ? 'good' : m.status === 'HALTED' ? 'bad' : '',
        sub: `${f.date(m.started_at)} → ${m.ended_at ? f.date(m.ended_at) : 'now'}` }),
    ];
  }
  return [
    ui.kpi({ label: 'Total invested (net)', value: f.inr(m.net_invested), tone: 'info',
      sub: h('span', null, 'deposited ', h('b', null, f.inr(m.total_deposited)), ' · withdrawn ', h('b', null, f.inr(m.total_withdrawn)),
        m.source === 'fallback' ? ' · approx.' : '') }),
    ui.kpi({ label: 'Current capital', value: f.inr(m.current_capital), sub: `${m.batches} batch(es)` }),
    ui.kpi({ label: 'Overall P&L', value: ui.money(m.overall_pnl), tone: f.kpiTone(m.overall_pnl),
      sub: h('span', null, h('b', { class: f.tone(m.overall_pnl_pct) }, f.pct(m.overall_pnl_pct, { sign: true })), ' on money deposited') }),
    ui.kpi({ label: 'Realized, all batches', value: ui.money(k.realized), tone: f.kpiTone(k.realized),
      sub: `+ unrealized ${f.inr(k.unrealized, { sign: true })} · charges ${f.inr(k.charges)}` }),
  ];
}

/** Trading performance row (6 tiles) - same for every scope. */
export function perfTiles(d) {
  const k = d.kpis || {};
  return [
    ui.kpi({ label: 'Net realized P&L', value: ui.money(k.realized), tone: f.kpiTone(k.realized), sub: `gross ${f.inr(k.gross, { sign: true })}` }),
    ui.kpi({ label: 'Trades', value: String(k.trades ?? 0),
      sub: h('span', null, h('b', { class: 'pos' }, `${k.wins ?? 0} won`), ' · ', h('b', { class: 'neg' }, `${k.losses ?? 0} lost`),
        k.flat ? ` · ${k.flat} flat` : '') }),
    ui.kpi({ label: 'Win rate', value: f.isNum(k.win_rate) ? `${f.num(k.win_rate, 1)}%` : '–', sub: `avg ${f.rmult(k.avg_r)} per trade` }),
    ui.kpi({ label: 'Profit factor', value: f.isNum(k.profit_factor) ? f.num(k.profit_factor, 2) : '–',
      sub: h('span', null, 'avg win ', h('b', { class: 'pos' }, f.inr(k.avg_win, { sign: true })), ' · loss ',
        h('b', { class: 'neg' }, f.inr(k.avg_loss, { sign: true }))) }),
    ui.kpi({ label: 'Charges paid', value: f.inr(k.charges), sub: 'brokerage, STT, GST, stamp…' }),
    ui.kpi({ label: 'Best / worst trade', value: h('span', null, h('span', { class: 'pos' }, k.best ? f.inr(k.best.net, { sign: true }) : '–'),
      ' / ', h('span', { class: 'neg' }, k.worst ? f.inr(k.worst.net, { sign: true }) : '–')),
    sub: [k.best?.symbol, k.worst?.symbol].filter(Boolean).join(' / ') || 'no closed trades' }),
  ];
}

export function summaryBlock() {
  const money = h('div', { class: 'kpis k4' }, Array.from({ length: 4 }, () => h('div', { class: 'kpi' }, ui.skeleton(2))));
  const perf = h('div', { class: 'kpis' }, Array.from({ length: 6 }, () => h('div', { class: 'kpi' }, ui.skeleton(2))));
  const root = h('div', { class: 'grid' }, money, perf);
  root.render = (d) => { money.replaceChildren(...moneyTiles(d)); perf.replaceChildren(...perfTiles(d)); };
  root.fail = (err) => money.replaceChildren(ui.errorBox(err, 'Could not load the summary'));
  return root;
}
