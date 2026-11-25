// Settings: autonomy + engine controls, appearance, session info, link to the advanced Streamlit
// console for everything else, recent commands and attributions.
import { api, on } from '../api.js';
import { h, icon } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { pageTimers } from '../router.js';
import { store, loadOverview } from '../store.js';
import { autonomyControl, engineControls } from '../components/controls.js';

export const title = 'Settings';

export function mount(root) {
  const t = pageTimers();
  const autonomyCard = ui.card('Who decides?', { hint: 'global autonomy switch (PAPER only)', body: ui.skeleton(3) });
  const engineCard = ui.card('Engine controls', { body: ui.skeleton(2) });
  const cmdCard = ui.card('Recent commands', { flush: true, hint: 'everything the dashboards asked the engine to do', body: ui.skeleton(4) });
  const me = store.me || {};
  const themeSeg = ui.segmented([{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }], document.documentElement.dataset.theme,
    async (v) => { const m = await import('../app.js'); m.setTheme(v); themeSeg.set(v); });
  const consoleCard = ui.card('Advanced console', { body: h('div', { class: 'stack' },
    h('div', { class: 'muted' }, 'Modes, risk limits, capital rules, strategies, agents & models, universe, schedule, notifications, prompt templates, manual trades and backups are edited in the Streamlit console. Every change is versioned and capped by config/hard_caps.yaml.'),
    h('a', { class: 'btn primary', href: me.streamlit_url || 'http://127.0.0.1:8501', target: '_blank', rel: 'noopener noreferrer' },
      icon('external', 14), 'Open Streamlit console'),
    h('div', { class: 'tiny faint' }, 'Start it with:  .\\tasks.ps1 dashboard   (or: streamlit run dashboard/app.py --server.address 127.0.0.1)')) });
  const prefsCard = ui.card('Appearance & session', { body: h('div', { class: 'stack' },
    h('div', { class: 'row' }, h('span', { class: 'strong' }, 'Theme'), themeSeg),
    h('div', { class: 'kv' },
      h('div', { class: 'k' }, 'Signed in as'), h('div', { class: 'v' }, me.username || '–'),
      h('div', { class: 'k' }, 'Idle timeout'), h('div', { class: 'v' }, `${Math.round((me.idle_timeout_s || 1800) / 60)} minutes (auto-refresh does not count as activity)`),
      h('div', { class: 'k' }, 'Trading'), h('div', { class: 'v' }, ui.badge('PAPER ONLY', 'warn')),
      h('div', { class: 'k' }, 'Full auto allowed'), h('div', { class: 'v' }, me.allow_full_auto ? 'yes (config/hard_caps.yaml)' : 'no - locked by config/hard_caps.yaml')),
    h('button', { class: 'btn', onclick: async () => (await import('../app.js')).showShortcuts() }, icon('keyboard', 14), 'Keyboard shortcuts')) });
  const aboutCard = ui.card('About', { body: h('div', { class: 'stack small muted' },
    h('div', null, 'Liar-V1 multi-agent PAPER trading system for NSE India. This dashboard never places orders: every action is a command the engine executes after the deterministic risk manager vets it. All numbers are computed by code from the database - never by an LLM.'),
    h('div', null, 'Charts by ', h('a', { href: 'https://www.tradingview.com/', target: '_blank', rel: 'noopener noreferrer' }, 'TradingView'),
      ' Lightweight Charts™ v4.2.3 (Apache License 2.0), served locally.')) });
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Settings')),
    h('div', { class: 'grid cols-2' }, h('div', { class: 'grid' }, autonomyCard, engineCard), h('div', { class: 'grid' }, consoleCard, prefsCard, aboutCard)),
    cmdCard);

  let ctl = null;
  const render = (o) => {
    if (!ctl) { ctl = autonomyControl(o.autonomy); autonomyCard.body.replaceChildren(ctl); } else ctl.update(o.autonomy);
    engineCard.body.replaceChildren(engineControls(o.engine));
  };
  const loadCommands = async () => {
    try {
      const d = await api.get('/api/commands?limit=25', { background: true });
      cmdCard.body.replaceChildren(ui.table([
        { key: 'id', label: '#' }, { key: 'created_at', label: 'When', fmt: (c) => f.dateTime(c.created_at) },
        { key: 'type', label: 'Command', fmt: (c) => h('span', { class: 'mono' }, c.type) },
        { key: 'status', label: 'Status', fmt: (c) => ui.badge(c.status, c.status === 'DONE' ? 'good' : c.status === 'PENDING' || c.status === 'RUNNING' ? 'warn' : 'bad') },
        { key: 'requested_by', label: 'By' }, { key: 'rule', label: 'Rule', fmt: (c) => (c.rule ? h('span', { class: 'mono neg' }, c.rule) : '–') },
        { key: 'message', label: 'Result', wrap: true },
      ], d.items, { sortable: false, empty: ui.empty('No commands yet') }));
    } catch (err) { cmdCard.body.replaceChildren(ui.errorBox(err)); }
  };
  t.track(on('overview', render));
  if (store.overview) render(store.overview);
  loadOverview({ background: true }).catch(() => {});
  loadCommands();
  t.every(10000, loadCommands);
  return () => t.clear();
}

