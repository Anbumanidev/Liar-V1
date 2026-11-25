// Decision detail: everything that explains one analysed stock (signals, confluence, agent votes,
// debate, meta reasons, every risk check). Shown in a right-hand drawer; sections are reused by
// the trade detail page.
import { api } from '../api.js';
import { h, icon } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { followUpButton } from './followup.js';
import { behaviourBlock, RULE_HELP } from './behaviour.js';

const SIGNAL_TONE = { BULLISH: 'good', BEARISH: 'bad', NEUTRAL: '' };

export async function openDecision(id) {
  const titleNode = h('div', { class: 'row' }, h('h2', null, `Decision #${id}`));
  const body = ui.openDrawer(titleNode, ui.skeleton(8));
  try {
    const d = await api.get(`/api/decisions/${id}`);
    titleNode.replaceChildren(...decisionHeader(d));
    body.replaceChildren(...renderDecision(d));
  } catch (err) {
    body.replaceChildren(ui.errorBox(err, 'Could not load the decision'));
  }
}

export function decisionHeader(d) {
  return [
    h('h2', null, d.symbol), ui.modeBadge(d.mode), ui.dirBadge(d.direction), ui.actionChip(d.final_action),
    h('span', { class: 'faint small' }, `${f.dateTime(d.time)} IST · #${d.id}`),
    followUpButton(d.symbol, { decisionId: d.id, mode: d.mode, compact: true }),
  ];
}

export function renderDecision(d) {
  const out = [];
  out.push(h('div', { class: ['notice', d.final_action === 'EXECUTED' ? 'good' : d.final_action === 'RISK_REJECTED' ? 'bad'
    : d.final_action === 'QUEUED' ? 'warn' : 'info'] }, icon('info', 16),
  h('div', { class: 'grow' }, h('div', { class: 'strong' }, ui.ACTION_LABEL[d.final_action] || d.final_action),
    h('div', null, d.reason || '–'))));

  out.push(h('div', { class: 'facts-grid' },
    ui.fact('Decision', d.decision || '–'), ui.fact('Confidence', f.isNum(d.confidence) ? `${d.confidence}%` : '–'),
    ui.fact('Confluence', f.isNum(d.confluence) ? `${f.num(d.confluence, 1)} (${d.grade || '–'})` : '–'),
    ui.fact('Strategy', d.strategy || '–'), ui.fact('Regime', f.title(d.regime) || '–'),
    ui.fact('Entry', f.price(d.entry)), ui.fact('Stop-loss', f.price(d.stop_loss), 'neg'),
    ui.fact('Target 1', f.price(d.target_1), 'pos'), ui.fact('Target 2', f.price(d.target_2), 'pos'),
    ui.fact('Qty', d.qty ?? '–'), ui.fact('Pipeline time', f.isNum(d.latency_ms) ? `${f.num(d.latency_ms / 1000, 1)} s` : '–')));

  const links = [];
  for (const p of d.positions || []) links.push(h('a', { href: `#/trade/${p.id}`, class: 'btn sm' }, `Trade #${p.id} (${p.status})`));
  if (d.approval) links.push(ui.badge(`Approval ${d.approval.status}${d.approval.decided_by ? ' by ' + d.approval.decided_by : ''}`, 'warn'));
  if (links.length) out.push(h('div', { class: 'row' }, links));

  if (d.signals?.length) {
    out.push(section('Strategy signals', ui.table([
      { key: 'strategy', label: 'Strategy', fmt: (s) => h('span', { class: 'strong' }, s.strategy) },
      { key: 'direction', label: 'Dir', fmt: (s) => ui.dirBadge(s.direction) },
      { key: 'entry_price', label: 'Entry', num: true, fmt: (s) => f.price(s.entry_price) },
      { key: 'stop_loss', label: 'SL', num: true, fmt: (s) => f.price(s.stop_loss) },
      { key: 'target_1', label: 'T1', num: true, fmt: (s) => f.price(s.target_1) },
      { key: 'strength', label: 'Strength', num: true },
      { key: 'notes', label: 'Notes', wrap: true, fmt: (s) => (s.notes || []).join('; ') },
    ], d.signals, { sortable: false })));
  }
  if (d.confluence_detail) out.push(section('Confluence', confluenceBars(d.confluence_detail)));
  if (d.vote || d.agents?.length) out.push(section('AI agent votes', agentsBlock(d.vote, d.agents)));
  if (d.debate) out.push(section('Bull vs Bear debate', debateBlock(d.debate)));
  if (d.meta) out.push(section('Meta agent (final AI decision)', metaBlock(d.meta)));
  const beh = behaviourBlock(d.behaviour);
  if (beh) out.push(section('Price behaviour (deterministic check)', beh));
  if (d.risk) out.push(section('Risk manager (deterministic, can veto anything)', riskBlock(d.risk)));
  if (d.sl_candidates?.length) {
    out.push(section('Stop-loss candidates', ui.table([
      { key: 'method', label: 'Method' }, { key: 'price', label: 'Price', num: true, fmt: (c) => f.price(c.price) },
      { key: 'distance_atr', label: 'ATR x', num: true, fmt: (c) => f.num(c.distance_atr, 2) },
      { key: 'note', label: 'Note', wrap: true },
    ], d.sl_candidates, { sortable: false })));
  }
  if (d.orders?.length) {
    out.push(section('Orders', ui.table([
      { key: 'id', label: '#' }, { key: 'side', label: 'Side' }, { key: 'purpose', label: 'Purpose' },
      { key: 'status', label: 'Status', fmt: (o) => ui.badge(o.status, o.status === 'FILLED' ? 'good' : o.status === 'PENDING' ? 'warn' : 'bad') },
      { key: 'fill_price', label: 'Fill', num: true, fmt: (o) => f.price(o.fill_price) },
      { key: 'time', label: 'Time', fmt: (o) => f.time(o.time) }, { key: 'reason', label: 'Reason', wrap: true },
    ], d.orders, { sortable: false })));
  }
  const market = d.market || {};
  if (Object.keys(market).length) {
    out.push(section('Market context', h('div', { class: 'kv' }, Object.entries(market).slice(0, 14).flatMap(([k, v]) =>
      [h('div', { class: 'k' }, f.title(k)), h('div', { class: 'v' }, typeof v === 'number' ? f.num(v, 2) : String(v ?? '–'))]))));
  }
  return out;
}

function section(titleText, body) {
  return h('div', { class: 'stack' }, h('div', { class: 'section-title' }, titleText), body);
}

export function confluenceBars(c) {
  const comps = Object.entries(c.components || {});
  const max = Math.max(1, ...comps.map(([, v]) => Math.abs(v)));
  return h('div', { class: 'stack' },
    h('div', { class: 'row' }, h('span', { class: 'strong' }, `Score ${f.num(c.score, 1)} / 100`), ui.gradeBadge(c.grade)),
    h('div', { class: 'bars' }, comps.map(([k, v]) => h('div', { class: 'b' },
      h('span', { class: 'muted' }, f.title(k)),
      h('div', { class: 'track' }, h('i', { style: { width: `${Math.max(2, (Math.abs(v) / max) * 100)}%` } })),
      h('span', { class: 'right strong' }, f.num(v, 1))))),
    c.notes?.length ? h('ul', { class: 'list small muted' }, c.notes.map((n) => h('li', null, n))) : null);
}

export function agentsBlock(vote, agents) {
  const items = [];
  if (vote) {
    items.push(h('div', { class: 'row' },
      ui.badge(`Bullish ${vote.bullish ?? 0}`, 'good'), ui.badge(`Bearish ${vote.bearish ?? 0}`, 'bad'),
      ui.badge(`Neutral ${vote.neutral ?? 0}`), vote.failed ? ui.badge(`Failed ${vote.failed}`, 'warn') : null,
      h('span', { class: 'muted small' }, `Weighted score ${f.num(vote.score, 2)}`),
      vote.tradeable === false ? ui.badge('NOT TRADEABLE', 'bad') : null));
    if (vote.failed_agents?.length) items.push(h('div', { class: 'small faint' }, `Failed agents: ${vote.failed_agents.join(', ')}`));
  }
  if (agents?.length) {
    const contrib = vote?.contributions || {};
    items.push(ui.table([
      { key: 'agent', label: 'Agent', fmt: (a) => h('span', { class: 'strong' }, f.title(a.agent)) },
      { key: 'signal', label: 'Signal', fmt: (a) => ui.badge(a.signal, SIGNAL_TONE[a.signal] ?? '') },
      { key: 'confidence', label: 'Conf', num: true, fmt: (a) => `${a.confidence}%` },
      { key: 'weight', label: 'Vote', num: true, title: 'Contribution to the weighted vote', sort: (a) => contrib[a.agent],
        fmt: (a) => f.isNum(contrib[a.agent]) ? f.signed(contrib[a.agent], 2) : '–' },
      { key: 'status', label: 'Status', fmt: (a) => ui.badge(a.status, a.status === 'OK' ? '' : 'warn') },
      { key: 'latency_ms', label: 'Time', num: true, fmt: (a) => f.isNum(a.latency_ms) ? `${f.num(a.latency_ms / 1000, 1)}s` : '–' },
      { key: 'key_findings', label: 'Key findings', wrap: true, sortable: false,
        fmt: (a) => a.error ? h('span', { class: 'neg' }, a.error) : (a.key_findings || []).slice(0, 3).join(' · ') },
    ], agents, { sortable: true }));
  }
  return h('div', { class: 'stack' }, items);
}

export function debateBlock(db) {
  const side = (s, cls, label) => h('div', { class: ['side', cls] },
    h('div', { class: 'row' }, h('span', { class: ['strong', cls === 'bull' ? 'pos' : 'neg'] }, label),
      h('span', { class: 'grow' }), f.isNum(s?.conviction) ? h('span', { class: 'strong' }, `${s.conviction}/100`) : null),
    f.isNum(s?.conviction) ? h('div', { class: ['meter', cls] }, h('i', { style: { width: `${Math.max(0, Math.min(100, s.conviction))}%` } })) : null,
    h('div', { class: 'stance' }, s?.stance || '–'),
    s?.arguments?.length ? h('ul', { class: 'list small muted' }, s.arguments.map((a) => h('li', null, a))) : null);
  const transcript = (db.transcript || []).filter((t) => t.text || t.arguments?.length);
  return h('div', { class: 'stack' },
    h('div', { class: 'debate' }, side(db.bull, 'bull', '▲ Bull'), side(db.bear, 'bear', '▼ Bear')),
    h('div', { class: 'small faint' }, `${db.rounds} round(s) · ${db.status}${db.model ? ' · ' + db.model : ''}`),
    transcript.length ? h('details', { class: 'more' }, h('summary', null, 'Full transcript'),
      h('div', { class: 'stack small' }, transcript.map((t) => h('div', null,
        h('span', { class: 'strong' }, `${f.title(t.speaker)} (${t.phase || 'round ' + t.round}): `),
        t.text || [t.stance, ...(t.arguments || [])].filter(Boolean).join(' · '))))) : null);
}

export function metaBlock(m) {
  return h('div', { class: 'stack' },
    h('div', { class: 'row small muted' }, `Decision ${m.decision || '–'} · confidence ${m.confidence ?? '–'}%`,
      m.model ? ` · ${m.model}` : '', f.isNum(m.latency_ms) ? ` · ${f.num(m.latency_ms / 1000, 1)}s` : ''),
    m.reasons_for?.length ? h('div', null, h('div', { class: 'small strong pos' }, 'Reasons for'),
      h('ul', { class: 'reasons for' }, m.reasons_for.map((r) => h('li', null, h('span', null, r))))) : null,
    m.reasons_against?.length ? h('div', null, h('div', { class: 'small strong neg' }, 'Reasons against (considered)'),
      h('ul', { class: 'reasons against' }, m.reasons_against.map((r) => h('li', null, h('span', null, r))))) : null,
    m.why_not ? ui.notice('warn', 'Why not', m.why_not) : null,
    m.error ? ui.notice('bad', 'Meta agent error', m.error) : null);
}

export function riskBlock(r) {
  const checks = [...(r.checks || [])].sort((a, b) => Number(a.passed) - Number(b.passed));
  return h('div', { class: 'stack' },
    h('div', { class: 'row' },
      ui.badge(r.approved ? 'APPROVED' : 'REJECTED', r.approved ? 'good' : 'bad'),
      r.rule ? h('span', { class: 'mono', title: RULE_HELP[r.rule] || null }, r.rule) : null,
      f.isNum(r.qty) ? h('span', { class: 'small muted' }, `qty ${r.qty} / max ${r.max_qty ?? '–'}`) : null,
      f.isNum(r.risk_amount) ? h('span', { class: 'small muted' }, `risk ${f.inr(r.risk_amount)}`) : null,
      f.isNum(r.charges_estimate_total) ? h('span', { class: 'small muted' }, `est. charges ${f.inr(r.charges_estimate_total)}`) : null,
      f.isNum(r.expected_net_at_t1) ? h('span', { class: 'small muted' }, `net at T1 ${f.inr(r.expected_net_at_t1, { sign: true })}`) : null,
      r.tier ? ui.badge(`Tier ${r.tier}`) : null),
    r.message && !r.approved ? h('div', { class: 'small' }, r.message) : null,
    checks.length ? h('div', { class: 'checks' }, checks.map((c) => h('div', { class: ['check', c.passed ? 'ok' : 'fail'] },
      h('span', { class: 'ic' }, c.passed ? '✓' : '✗'), h('span', { class: 'rule', title: RULE_HELP[c.rule] || null }, c.rule),
      h('span', { class: 'det' }, c.passed ? (c.detail ? 'ok' : '') : (c.detail || 'failed'))))) : null,
    checks.length ? h('div', { class: 'tiny faint' }, `${checks.filter((c) => c.passed).length} of ${checks.length} checks passed`) : null);
}
