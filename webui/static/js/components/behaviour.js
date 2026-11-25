// "Price behaviour" block: the deterministic check stored with every trade decision
// (context_json.proposal.behaviour) - where the price sits in today's range, how far the day already moved
// vs its normal range, the distance from VWAP and how similar moves played out in the past 20 sessions.
// Renders nothing for older decisions without the snapshot.
import { h } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';

const VERDICT = {
  OK: ['OK', 'good', 'Price behaviour does not argue against the trade.'],
  CHASING: ['CHASING', 'bad', 'Entry is chasing an extended move (near the day extreme / far from VWAP).'],
  FADES: ['FADES', 'warn', 'Similar moves in the past sessions mostly faded.'],
};

export const RULE_HELP = {
  CHASING_HIGH: 'Price-behaviour check: the entry chases an extended move (near the day high/low, far from VWAP, the day already moved more than its normal range).',
  HISTORY_FADES: 'Price-behaviour check: in the past 20 sessions, similar moves in this stock mostly faded instead of continuing.',
};

export function verdictChip(verdict) {
  const v = VERDICT[String(verdict || '').toUpperCase()];
  return v ? ui.badge(v[0], v[1], v[2]) : null;
}

const pct0 = (v) => `${f.num(v, 0)}%`;
const asPct = (v) => (Math.abs(v) <= 1.0001 ? v * 100 : v);   // rates may come as 0-1 or 0-100

function rangeBar(b) {
  const lo = b.day_low; const hi = b.day_high;
  const usable = f.isNum(lo) && f.isNum(hi) && hi > lo;
  if (!usable && !f.isNum(b.range_pos)) return null;
  const pos = (p) => Math.max(0, Math.min(100, ((p - lo) / (hi - lo)) * 100));
  const marks = [];
  const add = (price, cls, label, title) => {
    if (!usable || !f.isNum(price)) return;
    const outside = price > hi ? ' ↑' : price < lo ? ' ↓' : '';
    marks.push({ left: pos(price), cls, label: label + outside, title: `${title} ${f.price(price)}${outside ? (outside === ' ↑' ? ' (above today\'s high)' : ' (below today\'s low)') : ''}` });
  };
  add(b.pdl, 'pd', 'PDL', 'Previous day low');
  add(b.pdh, 'pd', 'PDH', 'Previous day high');
  add(b.vwap, 'vwap', 'VWAP', 'VWAP');
  if (f.isNum(b.entry)) add(b.entry, 'entry', 'Entry', 'Planned entry');
  if (f.isNum(b.ltp) && (!f.isNum(b.entry) || Math.abs(b.ltp - b.entry) > 1e-6)) add(b.ltp, 'ltp', 'LTP', 'Price at decision');
  if (!usable && f.isNum(b.range_pos)) {
    marks.push({ left: Math.max(0, Math.min(100, b.range_pos * 100)), cls: 'entry', label: 'Now', title: 'Position in the day range' });
  }
  // stagger labels that would overlap
  marks.sort((a, c) => a.left - c.left);
  let lastLeft = -100; let row = 0;
  for (const m of marks) { row = m.left - lastLeft < 9 ? (row + 1) % 2 : 0; m.row = row; lastLeft = m.left; }
  return h('div', { class: 'brange', role: 'img', 'aria-label': 'Day range from low to high with entry, VWAP and previous-day levels' },
    h('div', { class: 'track' }),
    marks.map((m) => [
      h('div', { class: ['mk', m.cls], style: { left: `${m.left}%` }, title: m.title }),
      h('div', { class: ['ml', m.cls, m.row ? 'r2' : '', m.left < 6 ? 'first' : m.left > 94 ? 'last' : ''], style: { left: `${m.left}%` }, title: m.title }, m.label),
    ]),
    h('div', { class: 'ends' },
      h('span', null, 'Low ', h('b', null, usable ? f.price(lo) : '–')),
      f.isNum(b.day_open) ? h('span', { class: 'faint' }, `open ${f.price(b.day_open)}`) : null,
      h('span', null, 'High ', h('b', null, usable ? f.price(hi) : '–'))));
}

/** Full block (decision drawer / trade page). Returns null when there is no behaviour data. */
export function behaviourBlock(b) {
  if (!b || typeof b !== 'object') return null;
  const lines = [];
  if (f.isNum(b.range_pos)) {
    lines.push(h('li', null, 'Price at ', h('b', null, pct0(b.range_pos * 100)), ' of the day\'s range (0% = day low, 100% = day high)'));
  }
  if (f.isNum(b.range_vs_adr)) {
    lines.push(h('li', null, 'The day already moved ', h('b', { class: b.range_vs_adr >= 1 ? 'warn-text' : null }, pct0(b.range_vs_adr * 100)),
      ' of its normal daily range', f.isNum(b.adr) ? ` (average ₹${f.num(b.adr, 2)})` : ''));
  }
  if (f.isNum(b.vwap_ext_atr)) {
    lines.push(h('li', null, h('b', null, `${f.num(Math.abs(b.vwap_ext_atr), 1)} ATR`), b.vwap_ext_atr >= 0 ? ' above' : ' below', ' VWAP',
      f.isNum(b.vwap) ? ` (${f.price(b.vwap)})` : ''));
  }
  if (f.isNum(b.continuation_rate)) {
    lines.push(h('li', null, 'Past 20 sessions: ', h('b', null, pct0(asPct(b.continuation_rate))), ' of similar moves continued',
      f.isNum(b.samples) ? ` (${b.samples} sample${b.samples === 1 ? '' : 's'})` : ''));
  }
  if (f.isNum(b.gap_pct) && Math.abs(b.gap_pct) >= 0.05) {
    lines.push(h('li', null, `Opened with a ${b.gap_pct > 0 ? 'gap up' : 'gap down'} of `, h('b', null, f.pct(b.gap_pct, { sign: true })),
      f.isNum(b.pdc) ? ` vs previous close ${f.price(b.pdc)}` : ''));
  }
  const bar = rangeBar(b);
  if (!bar && !lines.length && !(b.notes || []).length && !b.verdict) return null;
  return h('div', { class: 'stack behaviour' },
    h('div', { class: 'row' }, verdictChip(b.verdict), b.direction ? ui.dirBadge(b.direction) : null,
      b.verdict && VERDICT[b.verdict] ? h('span', { class: 'small muted' }, VERDICT[b.verdict][2]) : null),
    bar,
    lines.length ? h('ul', { class: 'list small' }, lines) : null,
    (b.notes || []).length ? h('ul', { class: 'list small muted' }, b.notes.map((n) => h('li', null, n))) : null);
}
