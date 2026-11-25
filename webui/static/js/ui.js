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
      const tr = h('tr', { class: [opts.onRow && 'click', opts.rowClass && opts.rowClass(r)], onclick: opts.onRow ? () => opts.onRow(r) : null });
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
const MAX_TOASTS = 6;
/** Show a toast. Returns a handle {update({message, tone, title, timeout, extra}), close()} (sticky when timeout=0).
 * Sticky toasts (command tracking) are never evicted by the 6-toast limit; the oldest non-sticky one goes first. */
export function toast(message, tone = 'info', { title = null, timeout = 5000, extra = null } = {}) {
  const root = document.getElementById('toasts');
  if (!root) return { update() {}, close() {} };
  const iconFor = (t) => icon(t === 'good' ? 'check' : t === 'bad' || t === 'warn' ? 'alert' : 'info', 16);
  const titleEl = h('div', { class: 'tt' });
  const msgEl = h('div');
  const extraEl = h('div', { class: 'toast-x' });
  const ic = h('span');
  let dismissed = false;
  let timer = null;
  const el = h('div', { role: 'status' }, ic, h('div', { class: 'grow' }, titleEl, msgEl, extraEl),
    h('button', { class: 'x', 'aria-label': 'Dismiss', onclick: () => { dismissed = true; clearTimeout(timer); el.remove(); } }, '×'));
  const handle = {
    update({ message: m = null, tone: t = null, title: tt, timeout: to = null, extra: ex } = {}) {
      if (t) { el.className = `toast ${t}`; el.setAttribute('role', t === 'bad' ? 'alert' : 'status'); ic.replaceChildren(iconFor(t)); }
      if (m !== null) msgEl.textContent = m;
      if (tt !== undefined) { titleEl.textContent = tt || ''; titleEl.hidden = !tt; }
      if (ex !== undefined) { extraEl.replaceChildren(ex || null); extraEl.hidden = !ex; }
      if (to !== null) {
        clearTimeout(timer);
        el.dataset.sticky = to ? '' : '1';
        if (to) timer = setTimeout(() => { dismissed = true; el.remove(); }, to);
      }
      // evicted by the toast limit (not dismissed by the owner): bring it back so the result is seen
      if (!dismissed && !el.isConnected) attach();
    },
    close() { dismissed = true; clearTimeout(timer); el.remove(); },
  };
  const attach = () => {
    root.appendChild(el);
    while (root.children.length > MAX_TOASTS) {
      const victim = Array.from(root.children).find((c) => c !== el && c.dataset.sticky !== '1');
      if (!victim) break;
      victim.remove();
    }
  };
  handle.update({ message, tone, title, timeout, extra: extra || null });
  if (!el.isConnected) attach();
  return handle;
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

/** Poll /api/commands/{id} until the engine finishes it, keeping one sticky toast updated
 * ("waiting" -> "running… 42s") and ending with the engine's result message.
 * Only time spent PENDING counts toward `pendingTimeoutMs` (is the engine running?); once the engine has
 * picked the command up (RUNNING) it is followed for up to `runningTimeoutMs` (SWING evening scans can
 * take 15-20 min). `resultExtra(st)` may return a node (e.g. "Add to follow-up" buttons) for the final toast. */
export async function trackCommand(cmd, label, { onDone = null, pendingTimeoutMs = 300000, runningTimeoutMs = 3600000,
  resultExtra = null } = {}) {
  const id = cmd?.command_id;
  if (!id) return null;
  const t = toast(`Sent to the engine (command #${id}). Waiting for it to pick it up…`, 'info', { title: label, timeout: 0 });
  const t0 = Date.now();
  let delay = 1000;
  let started = null;
  let pendingMs = 0;
  let last = Date.now();
  let lastStatus = 'PENDING';
  for (;;) {
    await new Promise((r) => setTimeout(r, delay));
    const now = Date.now();
    if (lastStatus !== 'RUNNING') pendingMs += now - last;
    last = now;
    delay = Math.min(delay * 1.25, started ? 5000 : 3000);
    let st = null;
    try { st = await api.get(`/api/commands/${id}`, { background: true }); } catch { st = null; }
    if (st?.finished) {
      const ok = st.ok === true;
      const title = st.rule ? `${label}: blocked by rule ${st.rule}` : `${label}: ${ok ? 'done' : st.status === 'DONE' ? 'finished' : st.status}`;
      let extra = null;
      try { extra = resultExtra ? resultExtra(st) : null; } catch { extra = null; }
      t.update({ message: st.message || (ok ? 'Done.' : st.status), tone: ok ? 'good' : (st.status === 'FAILED' ? 'bad' : 'warn'),
        title, timeout: extra ? 30000 : (ok ? 9000 : 14000), extra });
      if (onDone) onDone(st);
      return st;
    }
    if (st) lastStatus = st.status;
    if (lastStatus === 'RUNNING') {
      started = started || now;
      const runFor = now - started;
      if (runFor >= runningTimeoutMs) {
        t.update({ message: `Still running after ${f.ago(runFor / 1000)} - follow it on the Live log / Settings > Recent commands.`,
          tone: 'info', title: `${label}: still running`, timeout: 20000 });
        return null;
      }
      t.update({ message: `running… ${f.ago(runFor / 1000)} - follow it on the Live log page.`, tone: 'info' });
    } else {
      if (pendingMs >= pendingTimeoutMs) {
        t.update({ message: `Still waiting after ${f.ago(pendingMs / 1000)}. Is the engine running? The result will show in Settings > Recent commands.`,
          tone: 'warn', title: `${label}: still pending`, timeout: 15000 });
        return null;
      }
      t.update({ message: `queued (#${id}), waiting for the engine… ${Math.round((now - t0) / 1000)}s` });
    }
  }
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
