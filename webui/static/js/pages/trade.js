// Trade detail: chart around the trade with entry/exit markers and SL/target lines, facts grid,
// "what happened" story, "why it lost / what to fix" diagnosis and the decision context.
import { api, qs } from '../api.js';
import { h, icon } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { pageTimers } from '../router.js';
import { CandleChart } from '../charts.js';
import { registerChart } from '../theme.js';
import { agentsBlock, debateBlock, metaBlock, riskBlock, openDecision } from '../components/decision.js';
import { closePositionButton } from '../components/controls.js';

export const title = (p) => `Trade #${p.id}`;

const TFS = ['1m', '5m', '10m', '15m', '1h'];

export function mount(root, params) {
  const t = pageTimers();
  const id = Number(params.id);
  const headBox = h('div', { class: 'page-head' }, h('a', { href: '#/trades', class: 'btn sm ghost' }, icon('arrowLeft', 14), 'Journal'),
    h('h1', null, `Trade #${id}`));
  const box = h('div', { class: 'chart-box medium' });
  const tfSeg = h('div');
  const chartCard = h('section', { class: 'card chart-card' },
    h('div', { class: 'card-h' }, h('h2', null, 'Price action around the trade'), h('div', { class: 'actions' }, tfSeg)), box,
    h('div', { class: 'card-b' }, h('div', { class: 'legend-keys' },
      h('span', { class: 'ent' }, h('i'), 'Entry'), h('span', { class: 'sl' }, h('i'), 'Initial SL (dotted) / final SL'),
      h('span', { class: 'tgt' }, h('i'), 'T1 / T2'), h('span', { class: 'ema9' }, h('i'), 'EMA 9'), h('span', { class: 'ema21' }, h('i'), 'EMA 21'),
      h('span', { class: 'vwap' }, h('i'), 'VWAP'), h('span', { class: 'faint' }, 'times in IST'))));
  const factsCard = ui.card('Facts', { body: ui.skeleton(4) });
  const storyCard = ui.card('What happened', { body: ui.skeleton(4) });
  const diagCard = ui.card('Why it lost / what to fix', { body: ui.skeleton(3) });
  const ctxCard = ui.card('Decision context', { hint: 'why the system took this trade', body: ui.skeleton(4) });
  const legsCard = ui.card('Exit legs & orders', { flush: true, body: ui.skeleton(3) });
  root.append(headBox, h('div', { class: 'strip', id: 'trade-strip' }, ui.skeleton(1)), chartCard,
    h('div', { class: 'grid cols-2' }, factsCard, storyCard), diagCard, ctxCard, legsCard);

  const chart = new CandleChart(box, { intraday: true });
  t.track(registerChart(chart));
  let detail = null;
  let tf = '5m';

  const lines = (d) => {
    const p = d.position;
    const out = [{ price: p.avg_entry, kind: 'entry', title: 'Entry' }, { price: p.initial_stop_loss, kind: 'sl0', title: 'Initial SL' }];
    if (f.isNum(p.stop_loss) && p.stop_loss !== p.initial_stop_loss) out.push({ price: p.stop_loss, kind: 'sl', title: 'Final SL' });
    if (f.isNum(p.target_1)) out.push({ price: p.target_1, kind: 't1', title: 'T1' });
    if (f.isNum(p.target_2)) out.push({ price: p.target_2, kind: 't2', title: 'T2' });
    if (d.row && f.isNum(d.row.exit_price)) out.push({ price: d.row.exit_price, kind: 'exit', title: 'Avg exit' });
    return out;
  };

  const loadChart = async () => {
    if (!detail) return;
    const c = detail.chart || {};
    chart.setMessage('Loading…');
    try {
      const d = await api.get('/api/candles' + qs({ symbol: detail.position.symbol, tf, from: c.from, to: c.to, bars: 600, position_id: id }));
      await chart.ready;
      chart.setIntraday(d.intraday);
      chart.setData({ ...d, lines: [] }, { keepRange: false, extraLines: lines(detail) });
      if (!d.candles.length) chart.setMessage('No candles for this window (the data provider may not keep intraday history this far back). Try 15m or 1h.');
    } catch (err) {
      chart.setMessage(`Could not load candles: ${err.message}`);
    }
  };

  const drawTf = () => tfSeg.replaceChildren(ui.segmented(TFS.map((v) => ({ value: v, label: v.toUpperCase() })), tf, (v) => { tf = v; drawTf(); loadChart(); }, { cls: 'sm' }));

  const load = async () => {
    try {
      detail = await api.get(`/api/trades/${id}`);
    } catch (err) {
      root.replaceChildren(headBox, ui.errorBox(err, `Could not load trade #${id}`));
      return;
    }
    const d = detail;
    const p = d.position;
    const r = d.row;
    const pm = d.postmortem;
    tf = TFS.includes(d.chart?.timeframe) ? d.chart.timeframe : '5m';
    drawTf();
    const result = pm?.result || r?.result || (p.status === 'OPEN' ? 'OPEN' : '–');
    const net = pm?.net_pnl ?? r?.net;
    headBox.replaceChildren(h('a', { href: '#/trades', class: 'btn sm ghost' }, icon('arrowLeft', 14), 'Journal'),
      h('h1', null, p.symbol), ui.modeBadge(p.mode), ui.dirBadge(p.direction), ui.resultBadge(result),
      h('span', { class: 'sub' }, `#${p.id} · ${p.strategy || 'manual'} · ${f.dateTime(p.opened_at)} IST`), h('span', { class: 'grow' }),
      p.status === 'OPEN' ? closePositionButton({ ...p, qty: p.qty }, { onDone: load }) : null,
      p.decision_id ? h('button', { class: 'btn sm', onclick: () => openDecision(p.decision_id) }, 'Decision details') : null);
    const strip = document.getElementById('trade-strip');
    const cell = (label, value, sub = null) => h('div', { class: 'cell' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value),
      sub ? h('div', { class: 'sub' }, sub) : null);
    strip.replaceChildren(
      cell('Net P&L', ui.money(net), r ? `gross ${f.inr(r.gross, { sign: true })} · charges ${f.inr(r.charges)}` : 'open'),
      cell('R multiple', h('span', { class: f.tone(r?.r_multiple ?? pm?.r_multiple) }, f.rmult(r?.r_multiple ?? pm?.r_multiple)),
        f.isNum(pm?.r_multiple) && r && pm.r_multiple !== r.r_multiple ? `price move ÷ initial risk · net of charges ${f.rmult(pm.r_multiple)}` : 'price move ÷ initial risk'),
      cell('Exit reason', (r?.exit_reason_labels || pm?.exit_reasons || []).join(' + ') || (p.status === 'OPEN' ? 'still open' : '–')),
      cell('Duration', f.duration(pm?.holding_minutes ?? r?.holding_minutes), r ? `${f.hm(r.entry_time)} → ${f.hm(r.exit_time)}` : null));

    const mfe = pm?.mfe; const mae = pm?.mae;
    factsCard.body.replaceChildren(h('div', { class: 'facts-grid' },
      ui.fact('Entry', `${f.price(p.avg_entry)} · ${f.stamp(p.opened_at)}`),
      ui.fact('Exit (avg)', r ? `${f.price(r.exit_price)} · ${f.stamp(r.exit_time)}` : 'open'),
      ui.fact('Initial stop-loss', f.price(p.initial_stop_loss), 'neg'),
      ui.fact('Final stop-loss', f.price(p.stop_loss), 'neg'),
      ui.fact('Target 1', f.price(p.target_1), 'pos'), ui.fact('Target 2', f.price(p.target_2), 'pos'),
      ui.fact('Quantity', `${p.initial_qty}${p.qty ? ` (${p.qty} open)` : ''}`),
      ui.fact('Gross P&L', ui.money(pm?.gross_pnl ?? r?.gross)), ui.fact('Charges', f.inr(pm?.charges ?? r?.charges)),
      ui.fact('Net P&L', ui.money(net)), ui.fact('R multiple (price)', f.rmult(r?.r_multiple ?? pm?.r_multiple)),
      f.isNum(pm?.r_multiple) ? ui.fact('R after charges', f.rmult(pm.r_multiple)) : null,
      ui.fact('MFE · best move for you', f.isNum(mfe) ? `${f.num(mfe)} /share${f.isNum(pm?.mfe_r) ? ` · ${f.num(pm.mfe_r)}R` : ''}` : (f.isNum(p.highest_price) ? `high ${f.price(p.highest_price)}` : '–'), 'pos'),
      ui.fact('MAE · worst move against', f.isNum(mae) ? `${f.num(mae)} /share${f.isNum(pm?.mae_r) ? ` · ${f.num(pm.mae_r)}R` : ''}` : (f.isNum(p.lowest_price) ? `low ${f.price(p.lowest_price)}` : '–'), 'neg'),
      ui.fact('Duration', f.duration(pm?.holding_minutes ?? r?.holding_minutes)),
      ui.fact('Strategy · grade', `${p.strategy || '–'} · ${p.grade || '–'}`), ui.fact('Regime', f.title(p.regime) || '–'),
      ui.fact('Trailing', p.trailing_rule || '–'), ui.fact('Source', p.source)));

    storyCard.body.replaceChildren(renderStory(d));
    diagCard.body.replaceChildren(renderDiagnosis(d));
    renderContext(d);
    legsCard.body.replaceChildren(h('div', { class: 'stack card-b' },
      h('div', { class: 'section-title' }, 'Exit legs'),
      ui.table([
        { key: 'exit_time', label: 'Exit time', fmt: (l) => f.dateTime(l.exit_time) }, { key: 'qty', label: 'Qty', num: true },
        { key: 'exit_price', label: 'Exit price', num: true, fmt: (l) => f.price(l.exit_price) },
        { key: 'exit_reason', label: 'Reason', fmt: (l) => `${l.exit_reason}${l.is_partial ? ' (partial)' : ''}` },
        { key: 'gross', label: 'Gross', num: true, fmt: (l) => ui.money(l.gross) }, { key: 'charges', label: 'Charges', num: true, fmt: (l) => f.inr(l.charges) },
        { key: 'net', label: 'Net', num: true, fmt: (l) => ui.money(l.net) }, { key: 'r_multiple', label: 'R', num: true, fmt: (l) => f.rmult(l.r_multiple) },
      ], d.legs, { sortable: false, empty: ui.empty('No exits yet', 'The position is still open.') }),
      h('div', { class: 'section-title' }, 'Orders'),
      ui.table([
        { key: 'id', label: '#' }, { key: 'time', label: 'Time', fmt: (o) => f.dateTime(o.time) }, { key: 'side', label: 'Side' },
        { key: 'purpose', label: 'Purpose' }, { key: 'type', label: 'Type' }, { key: 'qty', label: 'Qty', num: true },
        { key: 'status', label: 'Status', fmt: (o) => ui.badge(o.status, o.status === 'FILLED' ? 'good' : o.status === 'PENDING' ? 'warn' : 'bad') },
        { key: 'fill_price', label: 'Fill', num: true, fmt: (o) => f.price(o.fill_price) },
        { key: 'slippage', label: 'Slippage', num: true, fmt: (o) => f.price(o.slippage) },
        { key: 'reason', label: 'Note', wrap: true },
      ], d.orders, { sortable: false })));
    loadChart();
  };

  const renderContext = async (d) => {
    const dec = d.postmortem?.decision;
    if (dec) {
      ctxCard.body.replaceChildren(h('div', { class: 'stack' },
        h('div', { class: 'row small muted' }, `Decided ${f.dateTime(dec.created_at)} IST · confidence ${dec.confidence ?? '–'}% · confluence ${f.num(dec.confluence, 1)} (${dec.grade || '–'})`,
          f.isNum(dec.latency_ms) ? ` · pipeline ${f.num(dec.latency_ms / 1000, 1)}s` : ''),
        dec.agents?.length ? agentsBlock(null, dec.agents) : null,
        dec.debate ? debateBlock({ bull: { stance: dec.debate.bull_stance, conviction: dec.debate.bull },
          bear: { stance: dec.debate.bear_stance, conviction: dec.debate.bear }, transcript: [], rounds: '–', status: '' }) : null,
        metaBlock({ reasons_for: dec.reasons_for, reasons_against: dec.reasons_against })));
      return;
    }
    if (!d.position.decision_id) {
      ctxCard.body.replaceChildren(ui.empty('No system decision', 'This was a manual trade or the decision was not stored.', 'user'));
      return;
    }
    try {
      const full = await api.get(`/api/decisions/${d.position.decision_id}`, { background: true });
      ctxCard.body.replaceChildren(h('div', { class: 'stack' },
        h('div', { class: 'row small muted' }, `Decided ${f.dateTime(full.time)} IST · confidence ${full.confidence}% · confluence ${f.num(full.confluence, 1)} (${full.grade || '–'})`),
        full.vote || full.agents?.length ? agentsBlock(full.vote, full.agents) : null,
        full.debate ? debateBlock(full.debate) : null,
        full.meta ? metaBlock(full.meta) : null,
        full.risk ? h('details', { class: 'more' }, h('summary', null, 'Risk checks at entry'), riskBlock(full.risk)) : null));
    } catch (err) {
      ctxCard.body.replaceChildren(ui.errorBox(err));
    }
  };

  load();
  return () => { t.clear(); chart.destroy(); };
}

function renderStory(d) {
  const story = d.postmortem?.story;
  if (story?.length) return h('ol', { class: 'list' }, story.map((s) => h('li', null, s)));
  // Fallback built from DB facts (no analysis module): entry, stop moves, exits.
  const ev = [];
  for (const o of d.orders) {
    if (o.purpose === 'ENTRY' && o.status === 'FILLED') ev.push([o.filled_at, `Entered ${o.side === 'BUY' ? 'long' : 'short'} ${o.qty} @ ${f.price(o.fill_price)}${f.isNum(o.slippage) ? ` (slippage ${f.price(o.slippage)})` : ''}.`]);
    else if (o.status !== 'FILLED' && o.purpose === 'ENTRY') ev.push([o.created_at, `Entry order ${o.status.toLowerCase()}: ${o.reason || ''}`]);
  }
  for (const s of d.position.sl_history || []) {
    if ((s.by || '').toUpperCase() === 'ENTRY') ev.push([s.ts, `Initial stop-loss ${f.price(s.new)}.`]);
    else ev.push([s.ts, `Stop-loss moved ${f.price(s.old)} → ${f.price(s.new)} (${s.by || 'system'}).`]);
  }
  for (const l of d.legs) ev.push([l.exit_time, `${l.is_partial ? 'Partial exit' : 'Exit'} ${l.qty} @ ${f.price(l.exit_price)} - ${l.exit_reason} - net ${f.inr(l.net, { sign: true })}.`]);
  ev.sort((a, b) => String(a[0]).localeCompare(String(b[0])));
  if (!ev.length) return ui.empty('No events recorded', '');
  return h('div', { class: 'stack' },
    h('div', { class: 'timeline' }, ev.map(([ts, text]) => h('div', { class: 'tl-item' }, h('div', { class: 't' }, f.hm(ts)),
      h('div', { class: 'rail' }, h('span', { class: 'd' })), h('div', { class: 'detail' }, text)))),
    d.postmortem_error ? h('div', { class: 'tiny faint' }, 'Detailed post-mortem story: not available yet.') : null);
}

function renderDiagnosis(d) {
  const diag = d.postmortem?.diagnosis;
  if (!d.postmortem) {
    return ui.notice('info', 'Post-mortem analysis is not available yet',
      d.postmortem_error?.status === 500 ? d.postmortem_error.detail : 'When the analysis module is installed, each trade gets a diagnosis (e.g. stop too tight, entered late) with the fix.');
  }
  if (!diag?.length) return ui.notice('good', 'No problems found', 'The trade followed the plan.');
  const order = { high: 0, medium: 1, low: 2 };
  return h('div', { class: 'issues' }, [...diag].sort((a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3)).map((x) =>
    h('div', { class: ['issue', x.severity] },
      h('div', { class: 'head' }, ui.badge(String(x.severity || '').toUpperCase(), x.severity === 'high' ? 'bad' : x.severity === 'medium' ? 'warn' : 'info'),
        h('span', { class: 'title' }, x.title || x.code)),
      x.detail ? h('div', { class: 'detail' }, x.detail) : null,
      x.fix ? h('div', { class: 'fix' }, h('b', null, 'Fix: '), x.fix) : null,
      x.fixed_by ? h('span', { class: 'fixed' }, ui.badge(`Fixed in system: ${x.fixed_by}`, 'good')) : null)));
}
