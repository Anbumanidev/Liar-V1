// "Who decides?" autonomy switch + engine controls (pause / resume / kill switch / analyze now).
// Every action goes through the engine's control-command queue; nothing trades directly.
import { api } from '../api.js';
import { h, icon } from '../dom.js';
import * as ui from '../ui.js';
import { loadOverview } from '../store.js';
import { followUpExtra } from './followup.js';

const ORDER = ['HUMAN_IN_THE_LOOP', 'HYBRID', 'FULL_AUTO', 'SIGNALS_ONLY', 'PER_MODE'];
const SHORT = { HUMAN_IN_THE_LOOP: 'Human', HYBRID: 'Hybrid', FULL_AUTO: 'Full auto', SIGNALS_ONLY: 'Signals', PER_MODE: 'Per mode' };

export function autonomyControl(autonomy) {
  const root = h('div', { class: 'stack' });
  const render = (a) => {
    const opts = new Map((a.options || []).map((o) => [o.value, o]));
    const desc = h('div', { class: 'small muted' }, a.description);
    const seg = ui.segmented(ORDER.filter((v) => opts.has(v)).map((v) => ({
      value: v, label: SHORT[v] + (opts.get(v).locked ? ' 🔒' : ''), title: opts.get(v).description,
      cls: v === 'FULL_AUTO' || v === 'HYBRID' ? 'auto' : '',
    })), a.value, (v) => choose(a, v), { cls: 'block', disabled: (v) => opts.get(v)?.locked });
    const perMode = ui.table([
      { key: 'mode', label: 'Mode' },
      { key: 'enabled', label: 'On', fmt: (m) => (m.enabled ? ui.badge('ON', 'good') : ui.badge('OFF')) },
      { key: 'policy', label: 'Mode policy', fmt: (m) => m.policy.replace('_', ' ') },
      { key: 'effective', label: 'Applied now', fmt: (m) => ui.badge(m.effective.replace('_', ' '),
        m.effective === 'AUTO' ? 'warn' : m.effective === 'MANUAL_APPROVE' ? 'info' : m.effective === 'HYBRID' ? 'violet' : '') },
    ], (a.per_mode || []).filter((m) => m.enabled), { sortable: false, empty: ui.empty('No mode enabled') });
    root.replaceChildren(
      seg, desc,
      a.value === 'HYBRID' ? h('div', { class: 'small' }, `Auto-executes grade ≥ ${a.hybrid_min_grade} and confidence ≥ ${a.hybrid_min_confidence}%; the rest waits for you.`) : null,
      !a.allow_full_auto ? h('div', { class: 'tiny faint' }, '🔒 Fully automatic / Hybrid are locked by config/hard_caps.yaml (allow_full_auto: false).') : null,
      h('details', { class: 'more' }, h('summary', null, 'What each option means'),
        h('div', { class: 'stack small' }, ORDER.filter((v) => opts.has(v)).map((v) => h('div', null,
          h('b', null, opts.get(v).label + ': '), opts.get(v).description)))),
      perMode,
      h('div', { class: 'tiny faint' }, 'PAPER trades only. The risk manager vetoes every trade in every setting.'));
  };
  const choose = async (a, value) => {
    const opt = (a.options || []).find((o) => o.value === value);
    const auto = value === 'FULL_AUTO' || value === 'HYBRID';
    const body = h('div', { class: 'stack' }, h('div', null, opt?.description || ''),
      auto ? ui.notice('warn', 'Automatic execution', 'Trades will be placed on PAPER without asking you. The risk manager and capital protection still veto everything.') : null);
    const ok = await ui.confirmModal({ title: `Switch to "${opt?.label || value}"?`, body, confirmLabel: 'Apply',
      danger: auto, typed: auto ? 'CONFIRM' : null });
    if (!ok) return;
    try {
      const res = await api.post('/api/settings/autonomy', { autonomy: value, confirm_text: auto ? 'CONFIRM' : null });
      if (!res.changed) { ui.toast('Nothing changed.', 'info'); return; }
      ui.toast(`Saved as settings version ${res.version_id}. Applies to NEW trades.`, 'good', { title: `Autonomy: ${opt?.label || value}` });
      ui.trackCommand({ command_id: res.command_id }, 'Reload settings');
      const o = await loadOverview({ background: false });
      render(o.autonomy);
    } catch (err) {
      ui.toast(err.message, 'bad', { title: 'Could not change autonomy' });
    }
  };
  render(autonomy);
  root.update = render;
  return root;
}

export function engineControls(engine) {
  const pause = h('button', { class: 'btn' }, icon('pause', 14), 'Pause new entries');
  const resume = h('button', { class: 'btn' }, icon('play', 14), 'Start / Resume');
  const kill = h('button', { class: 'btn danger' }, icon('stop', 14), 'Kill switch');
  pause.addEventListener('click', () => ui.sendCommand(pause, 'Pause', () => api.post('/api/control', { action: 'pause' }),
    { onDone: () => loadOverview() }));
  resume.addEventListener('click', () => ui.sendCommand(resume, 'Resume', () => api.post('/api/control', { action: 'resume' }),
    { onDone: () => loadOverview() }));
  kill.addEventListener('click', async () => {
    const exitCnc = h('input', { type: 'checkbox' });
    const ok = await ui.confirmModal({ title: 'Emergency kill switch', danger: true, typed: 'CONFIRM', confirmLabel: 'Kill now',
      body: h('div', { class: 'stack' }, h('div', null, 'Halts new trades and exits all intraday (MIS) positions at the next available price.'),
        h('label', { class: 'toggle' }, exitCnc, 'Also exit all delivery (CNC) positions'),
        h('div', { class: 'small muted' }, 'PAPER only. You can resume later from here.')) });
    if (!ok) return;
    ui.sendCommand(kill, 'Kill switch', () => api.post('/api/control', { action: 'kill_switch', confirm_text: 'CONFIRM',
      exit_cnc: exitCnc.checked }), { onDone: () => loadOverview() });
  });
  const state = engine?.state || 'UNKNOWN';
  return h('div', { class: 'stack' },
    h('div', { class: 'row' }, resume, pause, kill),
    h('div', { class: 'small muted' }, `Engine is ${engine?.label || state}. `, engine?.alive ? '' : 'If it is not running, start it in a terminal:  .\\tasks.ps1 run'));
}

export function closePositionButton(p, { onDone = null } = {}) {
  const btn = h('button', { class: 'btn sm danger-ghost' }, 'Close');
  btn.addEventListener('click', async (e) => {
    e.stopPropagation();
    const ok = await ui.confirmModal({ title: `Close ${p.symbol}?`, confirmLabel: 'Close position', danger: true,
      body: h('div', null, `Exit ${p.qty} ${p.symbol} (${p.mode}) at the next available price. PAPER trade.`) });
    if (!ok) return;
    ui.sendCommand(btn, `Close ${p.symbol}`, () => api.post('/api/control', { action: 'close_position', position_id: p.id }), { onDone });
  });
  return btn;
}

/** "Scan now (analysis only)" with a mode picker: screens the mode's watchlist now and sends the best
 * setups to the AI. Never trades. Progress shows on the Live log; the result message comes back as a toast. */
export function scanNowControls({ compact = false, modeSelect = null } = {}) {
  const modeSel = modeSelect || h('select', { class: 'input sm', 'aria-label': 'Mode to scan', style: { width: '118px' } },
    ['INTRADAY', 'BTST', 'SWING', 'POSITIONAL'].map((m) => h('option', { value: m }, m)));
  const ai = h('input', { type: 'checkbox', checked: true });
  const btn = h('button', { class: 'btn sm primary', title: 'Screen the watchlist now and run the AI on the best setups. Analysis only - never trades.' },
    icon('search', 13), compact ? 'Scan now' : 'Scan now (analysis only)');
  btn.addEventListener('click', () => {
    const mode = modeSel.value;
    ui.sendCommand(btn, `Scan ${mode}`, () => api.post('/api/control', { action: 'scan_now', mode, use_llm: ai.checked }),
      { resultExtra: (st) => followUpExtra(st, { mode }) });
  });
  return h('div', { class: 'row nowrap' }, modeSelect ? null : modeSel,
    h('label', { class: 'toggle', title: 'Send the best setups to the AI agents (slower)' }, ai, 'AI'), btn);
}
