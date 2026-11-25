// App bootstrap: theme, auth (login screen / setup instructions), shell (sidebar + status bar),
// one shared SSE stream, hash router and keyboard shortcuts.
import { api, on, setCsrf, ApiError } from './api.js';
import { h, icon, clear } from './dom.js';
import * as f from './format.js';
import * as sse from './sse.js';
import { startRouter, go, parse } from './router.js';
import { store, loadOverview, refreshOverviewSoon } from './store.js';
import * as ui from './ui.js';
import { applyThemeToCharts } from './theme.js';

const app = document.getElementById('app');
let stopRouter = null;
let clockTimer = null;
let overviewTimer = null;

// ------------------------------------------------------------------ theme
function initialTheme() {
  try { return localStorage.getItem('liar.theme') || 'dark'; } catch { return 'dark'; }
}
export function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('liar.theme', t); } catch { /* private mode */ }
  applyThemeToCharts();
  document.querySelectorAll('[data-theme-icon]').forEach((b) => b.replaceChildren(icon(t === 'dark' ? 'sun' : 'moon', 16)));
}
document.documentElement.dataset.theme = initialTheme();

// ------------------------------------------------------------------ login
function renderLogin({ notice = null, setup = false } = {}) {
  sse.stop();
  stopRouter?.();
  stopRouter = null;
  clearInterval(clockTimer);
  clearInterval(overviewTimer);
  app.className = 'login-wrap';
  const err = h('div', { class: 'notice bad hidden', role: 'alert' });
  const user = h('input', { class: 'input', name: 'username', autocomplete: 'username', placeholder: 'admin', required: true });
  const pass = h('input', { class: 'input', name: 'password', type: 'password', autocomplete: 'current-password', required: true });
  const submit = h('button', { class: 'btn primary lg block', type: 'submit' }, 'Log in');
  const form = h('form', null,
    h('label', { class: 'field' }, h('span', null, 'Username'), user),
    h('label', { class: 'field' }, h('span', null, 'Password'), pass), err, submit);
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    err.classList.add('hidden');
    try {
      const res = await ui.busy(submit, () => api.post('/api/login', { username: user.value, password: pass.value }));
      setCsrf(res.csrf);
      boot();
    } catch (ex) {
      if (ex instanceof ApiError && ex.body?.setup_required) { renderLogin({ setup: true }); return; }
      err.replaceChildren(icon('alert', 16), h('div', null, ex.message));
      err.classList.remove('hidden');
      pass.value = '';
      pass.focus();
    }
  });
  const setupBox = h('div', { class: 'stack' },
    ui.notice('warn', 'Dashboard credentials are not set up yet',
      'Create the owner login once from a terminal in the project folder, then reload this page:'),
    h('div', { class: 'code-line' }, '.venv\\Scripts\\python.exe scripts/set_password.py'),
    h('button', { class: 'btn block', onclick: () => location.reload() }, icon('refresh', 14), 'I have done it - reload'));
  app.replaceChildren(h('div', { class: 'login-card' },
    h('div', { class: 'brand' }, h('div', { class: 'brand-mark' }),
      h('div', null, h('div', { class: 'brand-name' }, 'Liar-V1 ', h('span', { class: 'paper-tag' }, 'PAPER')),
        h('div', { class: 'brand-sub' }, 'Owner trading terminal · NSE India'))),
    notice ? h('div', { class: 'notice info' }, icon('info', 16), h('div', null, notice)) : null,
    setup ? setupBox : form,
    h('div', { class: 'login-foot' }, 'Paper trading only - no real orders are ever placed. Owner-only, localhost.')));
  if (!setup) setTimeout(() => user.focus(), 30);
}

// ------------------------------------------------------------------ shell
const NAV = [
  { key: 'home', label: 'Home', icon: 'home', href: '#/home', kbd: 'g h' },
  { key: 'live', label: 'Live', icon: 'live', href: '#/live', kbd: 'g l' },
  { key: 'today', label: 'Today', icon: 'today', href: '#/today', kbd: 'g t' },
  { key: 'trades', label: 'Trades journal', icon: 'trades', href: '#/trades', kbd: 'g j' },
  { key: 'batches', label: 'Batches & capital', icon: 'batches', href: '#/batches', kbd: 'g b' },
  { sep: 'System' },
  { key: 'health', label: 'Health & logs', icon: 'health', href: '#/health', kbd: 'g x' },
  { key: 'settings', label: 'Settings', icon: 'settings', href: '#/settings', kbd: 'g s' },
];

const shell = {};

function renderShell() {
  app.className = 'shell';
  const navLinks = {};
  const nav = h('nav', { class: 'nav', 'aria-label': 'Main' }, NAV.map((n) => {
    if (n.sep) return h('div', { class: 'nav-sep' }, n.sep);
    const a = h('a', { href: n.href, title: `${n.label} (${n.kbd})`, onclick: () => closeSidebar() },
      icon(n.icon, 17), h('span', null, n.label), h('span', { class: 'kbd' }, n.kbd));
    navLinks[n.key] = a;
    return a;
  }));
  const homeCount = h('span', { class: 'count hidden' });
  navLinks.home.appendChild(homeCount);
  const sidebar = h('aside', { class: 'sidebar' },
    h('div', { class: 'brand' }, h('div', { class: 'brand-mark' }),
      h('span', { class: 'brand-name' }, 'Liar-V1'), h('span', { class: 'paper-tag' }, 'PAPER')),
    nav,
    h('div', { class: 'sidebar-foot' }, h('span', null, 'Paper trading only'), h('span', { id: 'whoami' }, ''),
      h('span', null, 'Press ', h('span', { class: 'kbd-key' }, '?'), ' for shortcuts')));
  const scrim = h('div', { class: 'scrim hidden', onclick: () => closeSidebar() });

  const title = h('div', { class: 'page-title' }, '');
  const enginePill = h('button', { class: 'pill', title: 'Engine status (heartbeat age)', onclick: () => go('/health') },
    h('span', { class: 'dot' }), h('span', { class: 'v' }, '…'));
  const marketPill = h('span', { class: 'pill hide-sm', title: 'NSE market (IST)' }, h('span', { class: 'dot' }), h('span', { class: 'v' }, '…'));
  const clock = h('span', { class: 'pill hide-sm', title: 'Indian Standard Time' }, h('span', { class: 'v mono' }, f.istClock()), ' IST');
  const autonomy = h('button', { class: 'pill hide-sm', title: 'Who decides - click to change', onclick: () => go('/settings') }, '…');
  const streamPill = h('span', { class: 'pill hide-sm', title: 'Live updates' }, h('span', { class: 'dot' }), h('span', { class: 'v' }, 'connecting'));
  const bellCount = h('span', { class: 'badge-count hidden' });
  const bell = h('button', { class: 'icon-btn', title: 'Pending approvals', 'aria-label': 'Pending approvals',
    onclick: () => { go('/home'); setTimeout(() => document.getElementById('approvals')?.scrollIntoView({ behavior: 'smooth' }), 250); } },
  icon('bell', 16), bellCount);
  const themeBtn = h('button', { class: 'icon-btn', title: 'Toggle light / dark theme', 'aria-label': 'Toggle theme', 'data-theme-icon': '',
    onclick: () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark') },
  icon(document.documentElement.dataset.theme === 'dark' ? 'sun' : 'moon', 16));
  const logoutBtn = h('button', { class: 'icon-btn', title: 'Log out', 'aria-label': 'Log out', onclick: logout }, icon('logout', 16));
  const menuBtn = h('button', { class: 'icon-btn menu-btn', 'aria-label': 'Menu', onclick: () => openSidebar() }, icon('menu', 16));
  const topbar = h('header', { class: 'topbar' }, menuBtn, title, h('div', { class: 'spacer' }),
    h('div', { class: 'status-group' }, enginePill, marketPill, clock, autonomy, streamPill), bell, themeBtn, logoutBtn);
  const content = h('main', { class: 'content', id: 'content' });
  app.replaceChildren(sidebar, scrim, h('div', { class: 'main' }, topbar, content));
  Object.assign(shell, { sidebar, scrim, navLinks, title, enginePill, marketPill, clock, autonomy, streamPill, bellCount, homeCount, content });
}

function openSidebar() { shell.sidebar.classList.add('open'); shell.scrim.classList.remove('hidden'); }
function closeSidebar() { shell.sidebar?.classList.remove('open'); shell.scrim?.classList.add('hidden'); }

// ------------------------------------------------------------------ status bar updates
const status = { heartbeatAt: null, engine: null, market: null };

function updateEngine(engine) {
  if (!engine) return;
  status.engine = engine;
  status.heartbeatAt = engine.heartbeat_at ? Date.parse(engine.heartbeat_at) : null;
  paintEngine();
}

function paintEngine() {
  const e = status.engine;
  if (!e || !shell.enginePill) return;
  const age = status.heartbeatAt ? (f.nowMs() - status.heartbeatAt) / 1000 : null;
  const stale = age === null || age > (e.stale_after_s || 30);
  const state = e.state || 'UNKNOWN';
  const tone = stale || state === 'STOPPED' || state === 'UNKNOWN' ? 'bad' : state === 'RUNNING' ? 'good' : 'warn';
  shell.enginePill.className = `pill ${tone}`;
  const label = state === 'UNKNOWN' ? 'NO HEARTBEAT' : stale && state !== 'STOPPED' ? `${state} (stale)` : state;
  shell.enginePill.replaceChildren(h('span', { class: ['dot', tone === 'good' && 'pulse'] }),
    h('span', { class: 'hide-sm' }, 'Engine '), h('span', { class: 'v' }, label), age !== null ? h('span', { class: 'mono' }, `· ${f.ago(age)}`) : null);
}

function updateMarket(market) {
  if (!market || !shell.marketPill) return;
  status.market = market;
  const open = market.open;
  shell.marketPill.className = `pill hide-sm ${open ? 'good' : ''}`;
  const sub = open ? (market.closes_at ? `till ${f.hm(market.closes_at)}` : '') :
    market.holiday ? market.holiday : market.next_open ? `opens ${f.weekday(market.next_open)} ${f.hm(market.next_open)}` : '';
  shell.marketPill.replaceChildren(h('span', { class: 'dot' }), h('span', null, 'Market '),
    h('span', { class: 'v' }, open === null || open === undefined ? '?' : open ? 'OPEN' : 'CLOSED'),
    sub ? h('span', { class: 'faint' }, `· ${sub}`) : null);
}

function updateAutonomy(a) {
  if (!a || !shell.autonomy) return;
  const auto = a.value === 'FULL_AUTO' || a.value === 'HYBRID';
  shell.autonomy.className = `pill hide-sm ${auto ? 'warn' : a.value === 'HUMAN_IN_THE_LOOP' ? 'info' : ''}`;
  shell.autonomy.replaceChildren(icon(auto ? 'bolt' : 'user', 13), h('span', null, 'Decides: '), h('span', { class: 'v' }, a.label));
}

function updatePending(n) {
  store.pending = n || 0;
  for (const el of [shell.bellCount, shell.homeCount]) {
    if (!el) continue;
    el.textContent = String(store.pending);
    el.classList.toggle('hidden', !store.pending);
  }
}

function updateStream(state) {
  if (!shell.streamPill) return;
  const map = { live: ['good', 'Live'], reconnecting: ['warn', 'Reconnecting'], polling: ['warn', 'Polling'], idle: ['', 'Offline'] };
  const [tone, label] = map[state] || ['', state];
  shell.streamPill.className = `pill hide-sm ${tone}`;
  shell.streamPill.replaceChildren(h('span', { class: ['dot', state === 'live' && 'pulse'] }), h('span', { class: 'v' }, label));
}

// ------------------------------------------------------------------ boot / logout
async function boot() {
  let me;
  try {
    me = await api.get('/api/me');
  } catch (err) {
    if (err.status === 401) { renderLogin({ setup: !!err.body?.setup_required, notice: sessionNotice(err) }); return; }
    app.replaceChildren(h('div', { class: 'login-card' }, ui.errorBox(err, 'Dashboard server unreachable'),
      h('button', { class: 'btn block', onclick: () => location.reload() }, 'Retry')));
    return;
  }
  setCsrf(me.csrf);
  f.setServerTime(me.server_time);
  store.me = me;
  renderShell();
  document.getElementById('whoami').textContent = `Signed in as ${me.username}`;
  stopRouter = startRouter(shell.content, {
    onRoute: (route) => {
      for (const [k, a] of Object.entries(shell.navLinks)) a.classList.toggle('active', k === route.nav);
      shell.title.textContent = NAV.find((n) => n.key === route.nav)?.label || '';
    },
  });
  sse.start();
  loadOverview({ background: true }).catch(() => {});
  clockTimer = setInterval(() => {
    shell.clock.firstChild.textContent = f.istClock();
    paintEngine();
  }, 1000);
  overviewTimer = setInterval(() => { if (!document.hidden) loadOverview().catch(() => {}); }, 30000);
}

function sessionNotice(err) {
  const d = err?.body?.detail || '';
  return d && d !== 'not authenticated' ? d : null;
}

async function logout() {
  try { await api.post('/api/logout', {}); } catch { /* ignore */ }
  setCsrf(null);
  renderLogin({ notice: 'You have been logged out.' });
}

// ------------------------------------------------------------------ global events
on('auth:required', (body) => {
  if (app.classList.contains('login-wrap')) return;
  renderLogin({ setup: !!body?.setup_required, notice: body?.detail && body.detail !== 'not authenticated' ? body.detail : 'Please log in again.' });
});
on('overview', (o) => {
  updateEngine(o.engine);
  updateMarket(o.market);
  updateAutonomy(o.autonomy);
  updatePending(o.approvals_pending);
});
on('sse:status', (s) => {
  updateEngine(s.engine);
  if (s.capital) refreshOverviewSoon();
});
on('sse:approvals', (a) => {
  updatePending(a.pending);
  if (a.changed && a.pending > 0) ui.toast('A trade is waiting for your approval.', 'warn', { title: 'Approval needed' });
});
on('sse:order', () => refreshOverviewSoon());
on('sse:trade', (t) => {
  refreshOverviewSoon();
  ui.toast(`${t.symbol}: ${t.exit_label} @ ${f.price(t.exit_price)} · net ${f.inr(t.net, { sign: true })}`, t.net >= 0 ? 'good' : 'bad',
    { title: 'Position exit' });
});
on('stream:state', updateStream);
on('net:down', () => updateStream('idle'));

// ------------------------------------------------------------------ keyboard shortcuts
const KEYMAP = { h: '/home', l: '/live', t: '/today', j: '/trades', b: '/batches', x: '/health', s: '/settings' };
let gPending = 0;
document.addEventListener('keydown', (e) => {
  if (!shell.content || app.classList.contains('login-wrap')) return;
  const tag = (e.target?.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target?.isContentEditable) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === 'g') { gPending = Date.now(); return; }
  if (gPending && Date.now() - gPending < 1200 && KEYMAP[e.key]) { gPending = 0; e.preventDefault(); go(KEYMAP[e.key]); return; }
  gPending = 0;
  if (e.key === '/') {
    const search = document.querySelector('[data-search]');
    if (search) { e.preventDefault(); search.focus(); search.select?.(); }
  } else if (e.key === '?') {
    showShortcuts();
  } else if (e.key === 'r' && parse().route) {
    window.dispatchEvent(new CustomEvent('page:refresh'));
  }
});

export function showShortcuts() {
  const rows = [['g h', 'Home'], ['g l', 'Live'], ['g t', 'Today'], ['g j', 'Trades journal'], ['g b', 'Batches'],
    ['g x', 'Health & logs'], ['g s', 'Settings'], ['/', 'Focus symbol search (Live)'], ['r', 'Refresh page data'],
    ['Esc', 'Close drawer / dialog'], ['?', 'This help']];
  ui.confirmModal({ title: 'Keyboard shortcuts', confirmLabel: 'Close',
    body: h('div', { class: 'kv' }, rows.flatMap(([k, d]) => [h('div', null, k.split(' ').map((x) => h('span', { class: 'kbd-key' }, x))),
      h('div', { class: 'v' }, d)])) });
}

boot();
