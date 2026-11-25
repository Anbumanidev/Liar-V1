// Live activity feed pieces: one row per activity.jsonl event, and the "Running now" tracker of
// AI calls that started but have not finished yet (with a live elapsed counter).
import { h } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';

export const STAGES = ['engine', 'cycle', 'screen', 'agents', 'llm', 'debate', 'meta', 'behaviour', 'risk', 'route', 'command',
  'watchlist', 'plan'];
export const STAGE_LABEL = { engine: 'ENGINE', cycle: 'CYCLE', screen: 'SCREEN', agents: 'AGENTS', llm: 'AI', debate: 'DEBATE',
  meta: 'META', behaviour: 'BEHAVIOUR', risk: 'RISK', route: 'ROUTE', command: 'COMMAND', watchlist: 'LISTS', plan: 'PLAN' };
export const STAGE_HELP = {
  engine: 'Engine start / stop / idle status', cycle: 'Scheduled scans starting and finishing', screen: 'Deterministic setup screening',
  agents: 'The 10 specialist AI agents analysing a stock (start + summary)', llm: 'Individual AI model calls',
  debate: 'Bull vs bear debate', meta: 'Meta agent (final AI decision)', behaviour: 'Price-behaviour check (chasing / fading moves)',
  risk: 'Risk manager verdicts', route: 'Final outcome per stock', command: 'Your commands (scan now, analyze now, …)',
  watchlist: 'Follow-up list changes', plan: 'Next-day plan and the 09:20 opening check',
};
const STAGE_TONE = { engine: 'info', cycle: 'violet', screen: 'outline', agents: 'cyan', llm: 'cyan', debate: 'warn', meta: 'violet',
  behaviour: 'warn', risk: 'bad', route: 'good', command: 'info', watchlist: 'outline', plan: 'good' };
const LEVEL_RANK = { debug: 0, info: 1, success: 1, warning: 2, error: 3 };

export function levelRank(level) { return LEVEL_RANK[String(level || 'info').toLowerCase()] ?? 1; }

export function stageChip(stage) {
  return ui.badge(STAGE_LABEL[stage] || String(stage || '?').toUpperCase(), STAGE_TONE[stage] ?? '', STAGE_HELP[stage] || null);
}

// ------------------------------------------------------------------ log tail + live stream merge
/** Identity of one activity line (the tail load and the SSE stream can both deliver it). */
export const actKey = (e) => `${e.ts}|${e.stage}|${e.message}`;

/** Items of `incoming` that are newer than the newest `loaded` entry; items with the same timestamp are
 * kept only if they are not already loaded (deduped by ts|stage|message). */
export function newerThan(loaded, incoming) {
  let last = -Infinity;
  for (const e of loaded) { const t = Date.parse(e.ts); if (Number.isFinite(t) && t > last) last = t; }
  if (last === -Infinity) return incoming.slice();
  const same = new Set(loaded.filter((e) => Date.parse(e.ts) === last).map(actKey));
  return incoming.filter((e) => {
    const t = Date.parse(e.ts);
    return Number.isFinite(t) && (t > last || (t === last && !same.has(actKey(e))));
  });
}

/** Holds SSE activity items while the initial /api/logs tail load is in flight, then releases only the
 * ones newer than what the load returned (no duplicates, nothing lost, no stale "Running now" calls). */
export class LoadGate {
  constructor() { this.loading = false; this.buf = []; }
  begin() { this.loading = true; this.buf = []; }
  /** true = held (the caller must not process the items now) */
  hold(items) { if (!this.loading) return false; this.buf.push(...items); return true; }
  end(loaded) { this.loading = false; const out = newerThan(loaded || [], this.buf); this.buf = []; return out; }
}

/** One activity line: time · stage · symbol · message (+ small facts). */
export function activityRow(e, { onDecision = null, compact = false } = {}) {
  const lvl = String(e.level || 'info').toLowerCase();
  const extras = [];
  if (e.stage === 'llm' && e.event === 'done' && f.isNum(e.latency_ms)) extras.push(`${f.num(e.latency_ms / 1000, 1)}s`);
  if (e.stage === 'llm' && e.model && !compact) extras.push(e.model);
  if (e.stage === 'cycle' && f.isNum(e.seconds)) extras.push(`${f.num(e.seconds, 0)}s`);
  const click = e.stage === 'route' && e.decision_id && onDecision ? () => onDecision(e.decision_id) : null;
  return h('div', { class: ['act-row', `lv-${lvl}`, click && 'click'], onclick: click, title: click ? 'Open the decision' : null },
    h('span', { class: 't' }, f.time(e.ts)),
    stageChip(e.stage),
    e.symbol ? h('span', { class: 'sym' }, e.symbol) : h('span', { class: 'sym faint' }, ''),
    h('span', { class: 'msg' }, e.message || e._raw || '',
      extras.length ? h('span', { class: 'x' }, ` · ${extras.join(' · ')}`) : null));
}

/** Tracks AI calls that have a `start` but no `done` yet. */
export class RunningTracker {
  constructor() { this.calls = new Map(); this.engine = null; }

  key(e) { return `${e.symbol || ''}|${e.mode || ''}|${e.agent || ''}`; }

  ingest(entries) {
    for (const e of entries || []) {
      if (e.stage === 'llm' && e.agent) {
        const k = this.key(e);
        if (e.event === 'start') this.calls.set(k, { symbol: e.symbol, mode: e.mode, agent: e.agent, model: e.model, started: Date.parse(e.ts) });
        else if (e.event === 'done') this.calls.delete(k);
      } else if (e.stage === 'engine') {
        this.engine = e;
        if (String(e.state || '').toUpperCase() === 'STOPPED') this.calls.clear();
      }
    }
    // an AI call nobody closed within 20 minutes is assumed dead (engine restarted / crashed)
    const cutoff = f.nowMs() - 20 * 60 * 1000;
    for (const [k, c] of this.calls) if (!(c.started > cutoff)) this.calls.delete(k);
  }

  list() { return [...this.calls.values()].sort((a, b) => a.started - b.started); }
}

/** "Running now" strip; call .tick() every second to refresh the elapsed counters. */
export function runningStrip(tracker) {
  const root = h('div', { class: 'running' });
  root.tick = () => {
    const calls = tracker.list();
    if (!calls.length) {
      root.replaceChildren(h('span', { class: 'faint small' }, 'No AI call running right now.'));
      return;
    }
    root.replaceChildren(h('span', { class: 'lbl' }, h('span', { class: 'dot pulse' }), `Running now (${calls.length})`),
      ...calls.map((c) => h('span', { class: 'run' }, h('b', null, c.symbol || '–'), ` ${f.title(c.agent)}`,
        c.model ? h('span', { class: 'faint' }, ` · ${c.model}`) : null,
        h('span', { class: 'el mono' }, ` ${f.num((f.nowMs() - c.started) / 1000, 0)}s`))));
  };
  root.tick();
  return root;
}
