// Tiny hash router: #/home, #/live, #/watchlists, #/today, #/trade/12, #/trades, #/batches, #/batches/3, #/health, #/settings.
// Each page module exports `title` and `mount(root, params) -> cleanup()`; only the visible page runs timers.

export const ROUTES = [
  { name: 'home', re: /^\/?(home)?$/, load: () => import('./pages/home.js'), nav: 'home' },
  { name: 'live', re: /^\/live$/, load: () => import('./pages/live.js'), nav: 'live' },
  { name: 'activity', re: /^\/activity$/, load: () => import('./pages/activity.js'), nav: 'activity' },
  { name: 'watchlists', re: /^\/(watchlists|follow-up|followup)$/, load: () => import('./pages/watchlists.js'), nav: 'watchlists' },
  { name: 'results', re: /^\/(results|today)$/, load: () => import('./pages/today.js'), nav: 'results' },
  { name: 'trade', re: /^\/trade\/(\d+)$/, keys: ['id'], load: () => import('./pages/trade.js'), nav: 'trades' },
  { name: 'trades', re: /^\/trades$/, load: () => import('./pages/trades.js'), nav: 'trades' },
  { name: 'batch', re: /^\/batches\/(\d+)$/, keys: ['id'], load: () => import('./pages/batches.js'), nav: 'batches' },
  { name: 'batches', re: /^\/batches$/, load: () => import('./pages/batches.js'), nav: 'batches' },
  { name: 'health', re: /^\/health$/, load: () => import('./pages/health.js'), nav: 'health' },
  { name: 'settings', re: /^\/settings$/, load: () => import('./pages/settings.js'), nav: 'settings' },
];

export function parse(hash = location.hash) {
  const raw = (hash || '').replace(/^#/, '');
  const [path, query = ''] = raw.split('?');
  for (const r of ROUTES) {
    const m = path.match(r.re);
    if (m) {
      const params = Object.fromEntries(new URLSearchParams(query));
      (r.keys || []).forEach((k, i) => { params[k] = m[i + 1]; });
      return { route: r, params };
    }
  }
  return { route: ROUTES[0], params: {} };
}

export function go(path) {
  const target = path.startsWith('#') ? path : `#${path}`;
  if (location.hash === target) window.dispatchEvent(new HashChangeEvent('hashchange'));
  else location.hash = target;
}

let current = null;
let seq = 0;

export function startRouter(root, { onRoute } = {}) {
  const render = async () => {
    const { route, params } = parse();
    const my = ++seq;
    if (current?.cleanup) { try { current.cleanup(); } catch (e) { console.error(e); } }
    current = null;
    root.replaceChildren();
    onRoute?.(route, params);
    let mod;
    try {
      mod = await route.load();
    } catch (err) {
      root.textContent = `Could not load page: ${err.message}`;
      return;
    }
    if (my !== seq) return;
    const pageRoot = document.createElement('div');
    pageRoot.className = 'grid';
    root.appendChild(pageRoot);
    const cleanup = mod.mount(pageRoot, params, route) || null;
    current = { route, cleanup };
    document.title = `${typeof mod.title === 'function' ? mod.title(params) : mod.title} · Liar-V1 (PAPER)`;
    window.scrollTo(0, 0);
  };
  window.addEventListener('hashchange', render);
  render();
  return () => window.removeEventListener('hashchange', render);
}

export function currentRoute() { return current?.route?.name || null; }

/** Page helper: intervals that pause while the tab is hidden and stop on unmount. */
export function pageTimers() {
  const timers = [];
  const offs = [];
  return {
    every(ms, fn) {
      const t = setInterval(() => { if (!document.hidden) fn(); }, ms);
      timers.push(t);
      return t;
    },
    later(ms, fn) { const t = setTimeout(fn, ms); timers.push(t); return t; },
    track(off) { if (typeof off === 'function') offs.push(off); return off; },
    clear() {
      timers.forEach((t) => { clearInterval(t); clearTimeout(t); });
      offs.forEach((off) => { try { off(); } catch { /* ignore */ } });
      timers.length = 0;
      offs.length = 0;
    },
  };
}
