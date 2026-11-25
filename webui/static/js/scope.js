// Global result scope: Today (a date) | Batch (one batch) | Overall (all batches).
// Persisted per viewer in localStorage (best effort); every page listens to the `scope` bus event.
import { api, emit } from './api.js';
import * as f from './format.js';

const KEY = 'liar.scope';
const state = { kind: 'today', date: null, batch_id: null };   // date null = follow "today" (IST)
let options = { batches: [], active_batch_id: null, today: null };

try {
  const saved = JSON.parse(localStorage.getItem(KEY) || 'null');
  if (saved && ['today', 'batch', 'all'].includes(saved.kind)) {
    state.kind = saved.kind;
    state.date = typeof saved.date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(saved.date) ? saved.date : null;
    state.batch_id = Number.isInteger(saved.batch_id) ? saved.batch_id : null;
  }
} catch { /* private mode / corrupt value: defaults */ }

function persist() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* ignore */ }
}

/** Current scope with the date resolved (Today) and the batch resolved to the active one when unset. */
export function getScope() {
  const date = state.date && state.date !== f.todayIST() ? state.date : f.todayIST();
  const batchId = state.batch_id ?? options.active_batch_id ?? null;
  return { kind: state.kind, date, isLiveToday: date === f.todayIST(), batch_id: batchId };
}

export function setScope(patch) {
  if ('kind' in patch) state.kind = patch.kind;
  if ('date' in patch) state.date = patch.date && patch.date !== f.todayIST() ? patch.date : null;
  if ('batch_id' in patch) state.batch_id = patch.batch_id === null || patch.batch_id === '' ? null : Number(patch.batch_id);
  persist();
  emit('scope', getScope());
}

/** Query params for the scoped endpoints. */
export function scopeParams() {
  const s = getScope();
  return { scope: s.kind, date: s.kind === 'today' ? s.date : null, batch_id: s.kind === 'batch' ? s.batch_id : null };
}

export function batchName(id) {
  return options.batches.find((b) => b.id === id)?.name || (id ? `Batch #${id}` : 'active batch');
}

export function scopeLabel() {
  const s = getScope();
  if (s.kind === 'today') return s.isLiveToday ? 'Today' : f.date(`${s.date}T12:00:00+05:30`);
  if (s.kind === 'batch') return batchName(s.batch_id);
  return 'Overall (all batches)';
}

/** Does a row (with `time` / `batch_id`) belong to the current scope? Used for live SSE appends. */
export function inScope(row) {
  const s = getScope();
  if (s.kind === 'all') return true;
  if (s.kind === 'batch') return !s.batch_id || row.batch_id === s.batch_id;
  return f.isoDay(new Date(row.time || row.ts || Date.now())) === s.date;
}

export function scopeOptions() { return options; }

/** Reload the batch list. A stale / deleted saved batch falls back to the active one; when that changes the
 * effective scope it is persisted and a `scope` event tells every page to reload. */
export async function loadScopeOptions() {
  try {
    const before = getScope();
    options = await api.get('/api/scope/options', { background: true });
    if (state.batch_id && !options.batches.some((b) => b.id === state.batch_id)) {
      state.batch_id = null;
      persist();
    }
    emit('scope:options', options);
    const after = getScope();
    // before.batch_id === null means the pages already asked for the server default (= the active batch)
    if (after.kind === 'batch' && before.batch_id !== null && after.batch_id !== before.batch_id) emit('scope', after);
  } catch { /* keep the previous list */ }
  return options;
}
