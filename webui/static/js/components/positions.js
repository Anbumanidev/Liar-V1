// Open position card with live LTP, unrealized P&L and a SL -> Entry -> T1 -> T2 progress bar.
import { h } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';

export function progressBar(p) {
  const pr = p.progress;
  if (!pr) return null;
  const wrap = h('div', { class: 'progress', title: 'Stop-loss → Entry → Target 1 → Target 2 (dot = last price)' });
  wrap.appendChild(h('div', { class: 'track' }));
  if (f.isNum(pr.ltp_pct) && f.isNum(pr.entry_pct)) {
    const a = Math.min(pr.entry_pct, pr.ltp_pct);
    const b = Math.max(pr.entry_pct, pr.ltp_pct);
    wrap.appendChild(h('div', { class: ['fill', pr.ltp_pct >= pr.entry_pct ? 'pos' : 'neg'],
      style: { left: `${a}%`, width: `${Math.max(0.5, b - a)}%` } }));
  }
  const tick = (pctv, cls, label, pos = '') => {
    if (!f.isNum(pctv)) return;
    wrap.appendChild(h('div', { class: ['tick', cls], style: { left: `${pctv}%` } }));
    wrap.appendChild(h('div', { class: ['lbl', pos], style: { left: `${pctv}%` } }, label));
  };
  tick(0, 'sl', `SL ${f.price(p.stop_loss)}`, 'first');
  tick(pr.entry_pct, 'entry', 'Entry');
  if (f.isNum(pr.t2_pct)) {
    tick(pr.t1_pct, 't1', 'T1');
    tick(pr.t2_pct, 't2', `T2 ${f.price(p.target_2)}`, 'last');
  } else {
    tick(pr.t1_pct, 't1', `T1 ${f.price(p.target_1)}`, 'last');
  }
  if (f.isNum(pr.ltp_pct)) wrap.appendChild(h('div', { class: 'ltp', style: { left: `${pr.ltp_pct}%` } }));
  return wrap;
}

export function positionCard(p, { onClick = null, compact = false } = {}) {
  const ltpNote = p.ltp_source === 'live' ? 'live' : p.ltp_source === 'engine' ? `engine ${f.stamp(p.ltp_at)}` : 'no price';
  return h('div', { class: ['pos-card', onClick && 'clickable'], onclick: onClick ? () => onClick(p) : null },
    h('div', { class: 'top' },
      h('span', { class: 'sym' }, p.symbol), ui.modeBadge(p.mode), ui.dirBadge(p.direction),
      p.grade ? ui.gradeBadge(p.grade) : null,
      h('div', { class: 'pnl' },
        h('div', { class: ['v', f.tone(p.unrealized)] }, f.inr(p.unrealized, { sign: true })),
        h('div', { class: ['s', f.tone(p.unrealized_pct)] }, `${f.pct(p.unrealized_pct, { sign: true })} · ${f.rmult(p.r_now)}`))),
    compact ? null : h('div', { class: 'facts' },
      ui.fact('Qty', String(p.qty)), ui.fact('Avg entry', f.price(p.avg_entry)),
      ui.fact(`LTP (${ltpNote})`, f.price(p.ltp)), ui.fact('Stop-loss', f.price(p.stop_loss), 'neg'),
      ui.fact('Opened', f.stamp(p.opened_at))),
    progressBar(p),
    compact ? null : h('div', { class: 'row tiny faint' },
      p.strategy ? `Strategy ${p.strategy}` : null, p.trailing_rule && p.trailing_rule !== 'NONE' ? `· trailing ${p.trailing_rule}` : null,
      p.partial_done ? '· partial booked' : null, p.breakeven_moved ? '· SL at breakeven' : null,
      `· risk to SL ${f.inr(p.risk_to_sl)}`));
}

export function positionsList(rows, opts = {}) {
  if (!rows?.length) return ui.empty('No open positions', 'New trades appear here with live P&L.', 'chart');
  return h('div', { class: 'pos-list' }, rows.map((p) => positionCard(p, opts)));
}
