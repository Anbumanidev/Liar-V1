// Reusable UI pieces: cards, KPI tiles, badges, tables, toasts, modals, drawer, skeletons.
import { api, ApiError } from './api.js';
import { h, icon, clear, append } from './dom.js';
import * as f from './format.js';

// ------------------------------------------------------------------ basic blocks
export function card(titleText, { actions = null, hint = null, cls = '', body = null, flush = false } = {}) {
  const b = h('div', { class: ['card-b', flush && 'flush'] });
  if (body) append(b, [body]);
  const head = titleText !== null ? h('div', { class: 'card-h' },
    typeof titleText === 'string' ? h('h2', null, titleText) : titleText,
    hint ? h('span', { class: 'hint' }, hint) : null,
    actions ? h('div', { class: 'actions' }, actions) : null) : null;
  const el = h('section', { class: ['card', cls] }, head, b);
  el.body = b;
  el.head = head;
  return el;
}

export function kpi({ label, value, sub = null, tone = '', help = null }) {
  return h('div', { class: ['kpi', tone], title: help || null },
    h('div', { class: 'label' }, label),
    h('div', { class: 'value' }, value),
    sub !== null ? h('div', { class: 'sub' }, sub) : null);
}

export function badge(text, tone = '', title = null) {
  return h('span', { class: ['badge', tone], title }, text);
}

export function money(v, { sign = true, cls = '' } = {}) {
  return h('span', { class: [f.tone(v), cls] }, f.inr(v, { sign }));
}

export function signedSpan(text, v) { return h('span', { class: f.tone(v) }, text); }

export const ACTION_TONE = {
  EXECUTED: 'good', QUEUED: 'warn', RISK_REJECTED: 'bad', NO_TRADE: '', FILTERED: 'outline', SIGNAL_ONLY: 'info',
  ANALYZED: 'cyan',
};
export const ACTION_LABEL = {
  EXECUTED: 'EXECUTED', QUEUED: 'QUEUED', RISK_REJECTED: 'RISK REJECTED', NO_TRADE: 'NO TRADE', FILTERED: 'FILTERED',
  SIGNAL_ONLY: 'SIGNAL ONLY', ANALYZED: 'ANALYZED',
};
export function actionChip(action) {
  return badge(ACTION_LABEL[action] || action || '–', ACTION_TONE[action] ?? '');
}

export function gradeBadge(grade, confluence) {
  if (!grade && !f.isNum(confluence)) return null;
  const tone = grade === 'A+' || grade === 'A' ? 'good' : grade === 'B' ? 'info' : grade === 'C' ? 'warn' : '';
  const label = f.isNum(confluence) ? `${grade || '?'} · ${Math.round(confluence)}` : grade;
  return badge(label, tone, 'Confluence grade · score (0-100)');
}

export function resultBadge(result) {
  const map = { WIN: ['WIN', 'good'], LOSS: ['LOSS', 'bad'], FLAT: ['FLAT', ''], OPEN: ['OPEN', 'info'] };
  const [t, tone] = map[result] || [result || '–', ''];
  return badge(t, tone);
}

export function dirBadge(direction) {
  if (!direction) return null;
  return badge(direction === 'SHORT' ? 'SHORT ▼' : 'LONG ▲', direction === 'SHORT' ? 'bad' : 'good');
}

export function modeBadge(mode) { return mode ? badge(mode, 'violet') : null; }

export function fact(label, value, cls = '') {
  return h('div', { class: 'fact' }, h('div', { class: 'l' }, label), h('div', { class: ['v', cls] }, value));
}

export function skeleton(lines = 3, cls = '') {
  const s = h('div', { class: 'skel' });
  for (let i = 0; i < lines; i++) s.appendChild(h('div', { class: ['sk', i === 0 ? 'h20 w60' : (i % 3 === 2 ? 'w80' : ''), cls] }));
  return s;
}

export function empty(titleText, text = '', iconName = 'inbox') {
  return h('div', { class: 'empty' }, icon(iconName, 26), h('div', { class: 'et' }, titleText), text ? h('div', null, text) : null);
}

export function notice(tone, titleText, text = null, actions = null) {
  const ic = tone === 'bad' || tone === 'warn' ? 'alert' : tone === 'good' ? 'check' : 'info';
  return h('div', { class: ['notice', tone] }, icon(ic, 16),
    h('div', { class: 'grow' }, h('div', { class: 'strong' }, titleText), text ? h('div', { class: 'muted' }, text) : null),
    actions);
}

export function errorBox(err, what = 'Could not load data') {
  if (err instanceof ApiError && err.status === 501) {
    return notice('info', 'Not available yet', 'This analysis module is still being built. The rest of the page works.');
  }
  return notice('bad', what, err?.message || String(err));
}

export function segmented(options, value, onChange, { cls = '', disabled = () => false } = {}) {
  const root = h('div', { class: ['seg', cls], role: 'radiogroup' });
  const render = (val) => {
    clear(root);
    for (const o of options) {
      const opt = typeof o === 'string' ? { value: o, label: o } : o;
      root.appendChild(h('button', {
        type: 'button', class: [opt.value === val && 'on', opt.cls], role: 'radio', 'aria-checked': String(opt.value === val),
        title: opt.title || null, disabled: disabled(opt.value),
        onclick: () => { if (opt.value !== val) onChange(opt.value); },
      }, opt.label));
    }
  };
  render(value);
  root.set = render;
  return root;
}

export function toggle(label, checked, onChange) {
  const input = h('input', { type: 'checkbox', checked, onchange: (e) => onChange(e.target.checked) });
  return h('label', { class: 'toggle' }, input, label);
}

// ------------------------------------------------------------------ tables
/**
 * columns: [{key, label, num, fmt(row) -> node|string, sort(row) -> value, cls, wrap, title}]
 * opts: {rows, onRow(row), empty, footer: row-like object for tfoot, sortable, initialSort}
 */
export function table(columns, rows, opts = {}) {
  const wrap = h('div', { class: ['table-wrap', opts.tall && 'tall'] });
  let sortKey = opts.initialSort?.key || null;
  let sortDir = opts.initialSort?.dir || -1;
  const draw = () => {
    clear(wrap);
    if (!rows || rows.length === 0) {
      wrap.appendChild(opts.empty || empty('Nothing to show'));
      return;
    }
    let data = rows;
    if (sortKey) {
      const col = columns.find((c) => c.key === sortKey);
      const val = col?.sort || ((r) => r[sortKey]);
      data = [...rows].sort((a, b) => {
        const x = val(a); const y = val(b);
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return (x > y ? 1 : -1) * sortDir;
      });
    }
    const thead = h('thead', null, h('tr', null, columns.map((c) => {
      const sortable = opts.sortable !== false && c.sortable !== false;
      return h('th', {
        class: [c.num && 'num', sortable && 'sortable'], title: c.title || null,
        onclick: sortable ? () => { if (sortKey === c.key) sortDir = -sortDir; else { sortKey = c.key; sortDir = -1; } draw(); } : null,
      }, c.label, sortKey === c.key ? h('span', { class: 'arrow' }, sortDir > 0 ? '▲' : '▼') : null);
    })));
    const tbody = h('tbody');
    for (const r of data) {
      const tr = h('tr', { class: opts.onRow ? 'click' : null, onclick: opts.onRow ? () => opts.onRow(r) : null });
      for (const c of columns) {
        const v = c.fmt ? c.fmt(r) : r[c.key];
        tr.appendChild(h('td', { class: [c.num && 'num', c.wrap && 'wrap', c.cls && (typeof c.cls === 'function' ? c.cls(r) : c.cls)] },
          v === null || v === undefined || v === '' ? '–' : v));
      }
      tbody.appendChild(tr);
    }
    const t = h('table', { class: 'tbl' }, thead, tbody);
    if (opts.footer) {
      t.appendChild(h('tfoot', null, h('tr', null, columns.map((c) => {
        const v = opts.footer[c.key];
        return h('td', { class: [c.num && 'num'] }, v === undefined || v === null ? '' : v);
      }))));
    }
    wrap.appendChild(t);
  };
  draw();
  wrap.update = (newRows) => { rows = newRows; draw(); };
  return wrap;
}

export function twoLine(main, sub) {
  return h('div', { class: 'two' }, h('div', null, main), sub ? h('div', { class: 's' }, sub) : null);
}

// ------------------------------------------------------------------ toasts
export function toast(message, tone = 'info', { title = null, timeout = 5000 } = {}) {
  const root = document.getElementById('toasts');
  if (!root) return;
  const el = h('div', { class: ['toast', tone], role: tone === 'bad' ? 'alert' : 'status' },
    icon(tone === 'good' ? 'check' : tone === 'bad' || tone === 'warn' ? 'alert' : 'info', 16),
    h('div', { class: 'grow' }, title ? h('div', { class: 'tt' }, title) : null, h('div', null, message)),
    h('button', { class: 'x', 'aria-label': 'Dismiss', onclick: () => el.remove() }, '×'));
  root.appendChild(el);
  while (root.children.length > 5) root.firstChild.remove();
  if (timeout) setTimeout(() => el.remove(), timeout);
}

// ------------------------------------------------------------------ modal (typed confirmation)
export function confirmModal({ title, body, confirmLabel = 'Confirm', danger = false, typed = null, extra = null }) {
  return new Promise((resolve) => {
    const root = document.getElementById('overlay-root');
    const input = typed ? h('input', { class: 'input mono', placeholder: `Type ${typed}`, autocomplete: 'off', spellcheck: 'false' }) : null;
    const ok = h('button', { class: ['btn', danger ? 'danger' : 'primary'], disabled: !!typed }, confirmLabel);
    const close = (v) => { back.remove(); document.removeEventListener('keydown', onKey); resolve(v); };
    const onKey = (e) => { if (e.key === 'Escape') close(false); if (e.key === 'Enter' && !ok.disabled) close(true); };
    if (input) input.addEventListener('input', () => { ok.disabled = input.value.trim() !== typed; });
    ok.addEventListener('click', () => close(true));
    const back = h('div', { class: 'modal-back', onclick: (e) => { if (e.target === back) close(false); } },
      h('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true' },
        h('div', { class: 'mh' }, icon(danger ? 'alert' : 'info', 18), h('h2', null, title)),
        h('div', { class: 'mb' }, body, extra,
          input ? h('label', { class: 'field' }, h('span', null, `Type ${typed} to continue`), input) : null),
        h('div', { class: 'mf' }, h('button', { class: 'btn ghost', onclick: () => close(false) }, 'Cancel'), ok)));
    root.appendChild(back);
    document.addEventListener('keydown', onKey);
    setTimeout(() => (input || ok).focus(), 30);
  });
}

// ------------------------------------------------------------------ drawer
let drawerEl = null;
export function openDrawer(titleNode, contentNode) {
  closeDrawer();
  const root = document.getElementById('overlay-root');
  const body = h('div', { class: 'db' }, contentNode);
  const onKey = (e) => { if (e.key === 'Escape') closeDrawer(); };
  const back = h('div', { class: 'drawer-back', onclick: () => closeDrawer() });
  const panel = h('aside', { class: 'drawer', role: 'dialog', 'aria-modal': 'true' },
    h('div', { class: 'dh' }, h('div', { class: 'grow' }, titleNode),
      h('button', { class: 'icon-btn', 'aria-label': 'Close', title: 'Close (Esc)', onclick: () => closeDrawer() }, icon('close', 16))),
    body);
  drawerEl = { back, panel, onKey };
  root.append(back, panel);
  document.addEventListener('keydown', onKey);
  return body;
}
export function closeDrawer() {
  if (!drawerEl) return;
  drawerEl.back.remove();
  drawerEl.panel.remove();
  document.removeEventListener('keydown', drawerEl.onKey);
  drawerEl = null;
}

// ------------------------------------------------------------------ command tracking
/** Run an async action with a busy button. */
export async function busy(btn, fn) {
  const old = Array.from(btn.childNodes);
  btn.disabled = true;
  clear(btn).appendChild(h('span', { class: 'spin' }));
  try { return await fn(); } finally { clear(btn); append(btn, old); btn.disabled = false; }
}

/** Poll /api/commands/{id} until the engine finishes it, then toast the outcome. */
export async function trackCommand(cmd, label, { onDone = null, timeoutMs = 90000 } = {}) {
  const id = cmd?.command_id;
  if (!id) return null;
  toast(`${label} sent to the engine (command #${id}). Waiting for it to run…`, 'info', { timeout: 3500 });
  const t0 = Date.now();
  let delay = 1200;
  while (Date.now() - t0 < timeoutMs) {
    await new Promise((r) => setTimeout(r, delay));
    delay = Math.min(delay * 1.3, 4000);
    let st;
    try { st = await api.get(`/api/commands/${id}`, { background: true }); } catch { continue; }
    if (st.finished) {
      if (st.ok) toast(st.message || 'Done.', 'good', { title: `${label}: done` });
      else if (st.rule) toast(`${st.message || ''}`, 'bad', { title: `${label}: blocked by rule ${st.rule}`, timeout: 9000 });
      else toast(st.message || st.status, 'bad', { title: `${label}: ${st.status}`, timeout: 9000 });
      if (onDone) onDone(st);
      return st;
    }
  }
  toast('The engine has not picked it up yet. Is the engine running? It will run when the engine starts.', 'warn',
    { title: `${label}: still pending`, timeout: 9000 });
  return null;
}

export async function sendCommand(btn, label, fn, opts = {}) {
  try {
    const cmd = await (btn ? busy(btn, fn) : fn());
    trackCommand(cmd, label, opts);
    return cmd;
  } catch (err) {
    toast(err.message || String(err), 'bad', { title: `${label} failed` });
    return null;
  }
}
