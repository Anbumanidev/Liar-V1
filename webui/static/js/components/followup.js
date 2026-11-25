// Follow-up lists: the "Add to follow-up" split button (+ list picker), result-toast helpers and the
// next-day plan table. Used by the Follow-up page, the decision drawer, Live, Trade detail, Results and Home.
import { api, emit } from '../api.js';
import { h, icon } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';

const SYM_RE = /^[A-Z0-9][A-Z0-9&\-_.^]{0,23}$/;
const LAST_KEY = 'liar.followup.last';
export const NOT_READY = 'Follow-up lists are not available yet (backend update pending).';

export function cleanSymbol(raw) {
  const s = String(raw || '').trim().toUpperCase().replace(/\.NS$/, '');
  return SYM_RE.test(s) ? s : null;
}

let cache = null;
let cacheAt = 0;
/** Owner (MANUAL) lists, cached for 30 s. */
export async function manualLists(force = false) {
  if (!force && cache && Date.now() - cacheAt < 30000) return cache;
  const d = await api.get('/api/watchlists', { background: true });
  cache = (d.lists || []).filter((l) => String(l.kind || 'MANUAL').toUpperCase() === 'MANUAL' && !l.archived);
  cacheAt = Date.now();
  return cache;
}
export function invalidateLists() { cache = null; }

function lastList() { try { return Number(localStorage.getItem(LAST_KEY)) || null; } catch { return null; } }
function rememberList(id) { try { if (id) localStorage.setItem(LAST_KEY, String(id)); else localStorage.removeItem(LAST_KEY); } catch { /* ignore */ } }

/** Add a stock to a list (the "Follow-up" list when listId is null). Toasts the outcome; returns the item or null. */
export async function addToFollowUp(symbol, { listId = null, listName = null, note = null, mode = null, decisionId = null } = {}) {
  const sym = cleanSymbol(symbol);
  if (!sym) { ui.toast(`"${symbol}" is not a valid NSE symbol`, 'warn'); return null; }
  const body = { symbol: sym, note: note || null, mode: mode || null, decision_id: decisionId || null };
  try {
    let item; let name = listName; let id = listId;
    if (listId) {
      item = await api.post(`/api/watchlists/${listId}/items`, body);
    } else {
      const res = await api.post('/api/watchlists/default/items', body);
      item = res.item;
      name = res.watchlist?.name || 'Follow-up';
      id = res.watchlist?.id || null;
    }
    invalidateLists();
    const at = f.isNum(item?.added_price) ? ` at ${f.price(item.added_price)}` : '';
    ui.toast(`${sym} is on "${name || 'your list'}"${at}. You can follow it (and remove it) on the Follow-up page.`, 'good', {
      title: 'Added to follow-up', timeout: 7000,
      extra: h('a', { href: id ? `#/watchlists?id=${id}` : '#/watchlists', class: 'btn sm ghost' }, 'Open Follow-up →') });
    emit('watchlists:changed', { id, symbol: sym });
    return item || {};
  } catch (err) {
    const na = err.status === 501;
    ui.toast(na ? NOT_READY : err.message, na ? 'info' : 'bad', { title: `Could not add ${sym}` });
    return null;
  }
}

/** Modal: pick a list (default "Follow-up" / the last one used), optionally a new list and a note, then add. */
export async function pickAndAdd(symbol, { mode = null, decisionId = null } = {}) {
  const sym = cleanSymbol(symbol);
  if (!sym) { ui.toast(`"${symbol}" is not a valid NSE symbol`, 'warn'); return null; }
  let lists;
  try { lists = await manualLists(true); } catch (err) {
    ui.toast(err.status === 501 ? NOT_READY : err.message, err.status === 501 ? 'info' : 'bad', { title: 'Follow-up lists' });
    return null;
  }
  const last = lastList();
  const preferred = lists.find((l) => l.id === last) || lists.find((l) => l.name === 'Follow-up') || null;
  const sel = h('select', { class: 'input', 'aria-label': 'List' },
    lists.some((l) => l.name === 'Follow-up') ? null : h('option', { value: '' }, 'Follow-up (created automatically)'),
    lists.map((l) => h('option', { value: String(l.id), selected: preferred?.id === l.id },
      `${l.name} (${l.items ?? 0})${l.use_for_trading ? ' · used for trading' : ''}`)),
    h('option', { value: 'new' }, '+ New list…'));
  const newName = h('input', { class: 'input hidden', placeholder: 'Name of the new list', maxlength: '80', autocomplete: 'off' });
  sel.addEventListener('change', () => { newName.classList.toggle('hidden', sel.value !== 'new'); if (sel.value === 'new') newName.focus(); });
  const note = h('input', { class: 'input', placeholder: 'e.g. wait for a breakout above the day high', maxlength: '500', autocomplete: 'off' });
  const ok = await ui.confirmModal({ title: `Follow up ${sym}`, confirmLabel: 'Add',
    body: h('div', { class: 'stack' },
      h('label', { class: 'field' }, h('span', null, 'List'), sel, newName),
      h('label', { class: 'field' }, h('span', null, 'Note (optional)'), note),
      h('div', { class: 'small muted' }, 'The current price is saved so the list shows the move since you added it. Nothing is traded.')) });
  if (!ok) return null;
  let listId = sel.value && sel.value !== 'new' ? Number(sel.value) : null;
  let listName = lists.find((l) => l.id === listId)?.name || null;
  if (sel.value === 'new') {
    const name = newName.value.trim();
    if (!name) { ui.toast('Give the new list a name.', 'warn'); return null; }
    try {
      const wl = await api.post('/api/watchlists', { name });
      listId = wl.id; listName = wl.name;
    } catch (err) { ui.toast(err.message, 'bad', { title: 'Could not create the list' }); return null; }
  }
  rememberList(listId);
  return addToFollowUp(sym, { listId, listName, note: note.value.trim() || null, mode, decisionId });
}

/** Split button: main part adds to "Follow-up" in one click, ▾ opens the list picker. */
export function followUpButton(symbolOrGetter, { decisionId = null, mode = null, compact = false } = {}) {
  const get = () => (typeof symbolOrGetter === 'function' ? symbolOrGetter() : symbolOrGetter);
  const main = h('button', { class: 'btn sm', type: 'button', title: 'Add to your "Follow-up" list to watch it later (nothing is traded)' },
    icon('plus', 13), compact ? 'Follow-up' : 'Add to follow-up');
  const more = h('button', { class: 'btn sm more', type: 'button', title: 'Choose a list / add a note', 'aria-label': 'Choose a follow-up list' }, '▾');
  main.addEventListener('click', (e) => {
    e.stopPropagation();
    const s = get();
    if (!s) { ui.toast('Pick a symbol first.', 'warn'); return; }
    ui.busy(main, () => addToFollowUp(s, { mode: typeof mode === 'function' ? mode() : mode, decisionId }));
  });
  more.addEventListener('click', (e) => {
    e.stopPropagation();
    const s = get();
    if (!s) { ui.toast('Pick a symbol first.', 'warn'); return; }
    pickAndAdd(s, { mode: typeof mode === 'function' ? mode() : mode, decisionId });
  });
  return h('span', { class: 'split-btn', onclick: (e) => e.stopPropagation() }, main, more);
}

/** Extra node for an Analyze-now / Scan-now result toast: "Add to follow-up" per analysed symbol. */
export function followUpExtra(st, { symbol = null, mode = null } = {}) {
  const data = st?.data || {};
  const rows = [];
  if (Array.isArray(data.analysed)) {
    for (const a of data.analysed) if (cleanSymbol(a?.symbol)) rows.push({ symbol: cleanSymbol(a.symbol), decisionId: a.decision_id || null });
  }
  if (!rows.length && cleanSymbol(symbol)) rows.push({ symbol: cleanSymbol(symbol), decisionId: data.decision_id || null });
  if (!rows.length) return null;
  const shown = rows.slice(0, 6);
  return h('div', { class: 'stack fu-extra' },
    shown.map((r) => h('div', { class: 'row nowrap' }, h('span', { class: 'sym' }, r.symbol), h('span', { class: 'grow' }),
      followUpButton(r.symbol, { decisionId: r.decisionId, mode, compact: true }))),
    rows.length > shown.length ? h('div', { class: 'tiny faint' }, `+${rows.length - shown.length} more on the Live feed`) : null);
}

// ------------------------------------------------------------------ next-day plan
export const PLAN_STATUS_TONE = { ACTIVE: 'info', TRIGGERED: 'good', INVALIDATED: 'bad' };

export function statusChip(status, note = null) {
  const s = String(status || 'ACTIVE').toUpperCase();
  return ui.badge(s, PLAN_STATUS_TONE[s] ?? '', note || null);
}

/** Plan rows: symbol, status (+ note), setup, direction, trigger, stop, target, R:R, score, reasons. */
export function planColumns({ quotes = null, onSymbol = null } = {}) {
  const pv = (r) => r.plan_view || {};
  return [
    { key: 'symbol', label: 'Stock', fmt: (r) => h('div', { class: 'two' },
      h('div', null, onSymbol ? h('a', { href: `#/live?symbol=${encodeURIComponent(r.symbol)}`, class: 'sym',
        onclick: (e) => { e.stopPropagation(); } }, r.symbol) : h('span', { class: 'sym' }, r.symbol)),
      pv(r).affordable === false ? h('div', null, ui.badge('not affordable with current capital', 'warn')) : null) },
    { key: 'status', label: 'Status', sort: (r) => r.status, wrap: true, fmt: (r) => h('div', { class: 'two' }, statusChip(r.status, r.status_note),
      r.status_note ? h('div', { class: 's clamp-2' }, r.status_note) : null) },
    { key: 'setup', label: 'Setup', sort: (r) => pv(r).setup, fmt: (r) => pv(r).setup || '–' },
    { key: 'direction', label: 'Dir', sort: (r) => pv(r).direction, fmt: (r) => ui.dirBadge(pv(r).direction) },
    quotes ? { key: 'ltp', label: 'LTP', num: true, sort: (r) => quotes.get(r.symbol)?.ltp, fmt: (r) => f.price(quotes.get(r.symbol)?.ltp) } : null,
    { key: 'trigger', label: 'Trigger', num: true, sort: (r) => pv(r).trigger, fmt: (r) => f.price(pv(r).trigger) },
    { key: 'stop', label: 'Stop', num: true, sort: (r) => pv(r).stop, fmt: (r) => h('span', { class: 'neg' }, f.price(pv(r).stop)) },
    { key: 'target', label: 'Target', num: true, sort: (r) => pv(r).target, fmt: (r) => h('span', { class: 'pos' }, f.price(pv(r).target)) },
    { key: 'rr', label: 'R:R', num: true, sort: (r) => pv(r).rr, fmt: (r) => (f.isNum(pv(r).rr) ? `1:${f.num(pv(r).rr, 1)}` : '–') },
    { key: 'score', label: 'Score', num: true, sort: (r) => pv(r).score, fmt: (r) => (f.isNum(pv(r).score) ? f.num(pv(r).score, 0) : '–') },
    { key: 'reasons', label: 'Why', sortable: false, wrap: true, fmt: (r) => {
      const reasons = pv(r).reasons || [];
      if (!reasons.length) return null;
      return h('details', { class: 'more', title: reasons.join(' · ') }, h('summary', null, `${reasons.length} reason${reasons.length > 1 ? 's' : ''}`),
        h('ul', { class: 'list small muted' }, reasons.map((x) => h('li', null, x))));
    } },
  ].filter(Boolean);
}

export function planTable(items, opts = {}) {
  return ui.table(planColumns(opts), items || [], { sortable: true, initialSort: { key: 'score', dir: -1 },
    empty: ui.empty('No stocks in this plan', 'The end-of-day scan found no setup worth planning for.', 'target'), ...opts.table });
}
