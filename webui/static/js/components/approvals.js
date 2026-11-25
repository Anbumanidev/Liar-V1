// Pending approval card: plan, bull/bear points, charges + tax estimate, live countdown, Approve / Reject.
import { api } from '../api.js';
import { h } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { openDecision } from './decision.js';

export function approvalCard(a, { onChanged = null } = {}) {
  const deadline = Date.now() + (a.seconds_left || 0) * 1000;
  const timer = h('span', { class: 'timer', title: 'Time left to decide (then it expires)' });
  const tick = () => {
    const left = (deadline - Date.now()) / 1000;
    timer.textContent = left <= 0 ? 'EXPIRED' : `⏱ ${f.countdown(left)}`;
    if (left > 0 && timer.isConnected) setTimeout(tick, 1000);
  };
  tick();
  const t1 = a.tax?.scenarios?.find((s) => s.label === 'Target 1') || a.tax?.scenarios?.[0];
  const approveBtn = h('button', { class: 'btn success' }, 'Approve');
  const rejectBtn = h('button', { class: 'btn danger-ghost' }, 'Reject');
  approveBtn.addEventListener('click', async () => {
    const ok = await ui.confirmModal({ title: `Approve ${a.symbol}?`, confirmLabel: 'Approve trade',
      body: h('div', { class: 'stack' },
        h('div', null, `${a.direction} ${a.qty} × ${a.symbol} (${a.mode}) at ${f.price(a.entry_price)}, stop-loss ${f.price(a.stop_loss)}, target ${f.price(a.target_1)}.`),
        h('div', { class: 'muted small' }, 'PAPER trade. The risk manager re-checks everything before execution and can still block it.')) });
    if (!ok) return;
    ui.sendCommand(approveBtn, `Approve ${a.symbol}`, () => api.post(`/api/approvals/${a.id}/approve`), { onDone: onChanged });
  });
  rejectBtn.addEventListener('click', async () => {
    const reason = h('input', { class: 'input', placeholder: 'Reason (optional)', maxlength: '300' });
    const ok = await ui.confirmModal({ title: `Reject ${a.symbol}?`, confirmLabel: 'Reject', danger: true,
      body: h('div', { class: 'muted' }, 'The setup will be tracked as a shadow trade so you can see what it would have done.'),
      extra: h('label', { class: 'field' }, h('span', null, 'Reason'), reason) });
    if (!ok) return;
    ui.sendCommand(rejectBtn, `Reject ${a.symbol}`, () => api.post(`/api/approvals/${a.id}/reject`, { reason: reason.value || null }),
      { onDone: onChanged });
  });
  return h('div', { class: 'appr' },
    h('div', { class: 'top' },
      h('span', { class: 'sym' }, a.symbol), ui.modeBadge(a.mode), ui.dirBadge(a.direction), ui.gradeBadge(a.grade),
      h('span', { class: 'small muted' }, `${a.confidence}% conf`), timer),
    h('div', { class: 'facts-grid' },
      ui.fact('Qty', `${a.qty}${a.max_qty && a.max_qty !== a.qty ? ` (max ${a.max_qty})` : ''}`),
      ui.fact(`Entry (${a.entry_type})`, f.price(a.entry_price)), ui.fact('Stop-loss', f.price(a.stop_loss), 'neg'),
      ui.fact('Target 1', f.price(a.target_1), 'pos'), ui.fact('Reward : risk', f.isNum(a.rr) ? `${f.num(a.rr, 2)} : 1` : '–'),
      ui.fact('Risk (₹ to SL)', f.inr(a.risk_amount))),
    (a.bull_points?.length || a.bear_points?.length) ? h('div', { class: 'pts' },
      h('div', null, h('div', { class: 'bull' }, '▲ Bull points'), h('ul', null, (a.bull_points || []).map((p) => h('li', null, p)))),
      h('div', null, h('div', { class: 'bear' }, '▼ Bear points'), h('ul', null, (a.bear_points || []).map((p) => h('li', null, p))))) : null,
    h('div', { class: 'taxline' },
      f.isNum(a.charges_estimate) ? `Charges ≈ ${f.inr(a.charges_estimate)}` : 'Charges: –',
      a.tax ? ` · Tax: ${a.tax.head || '–'} ~${f.num(a.tax.rate_pct, 1)}%` : '',
      a.tax && f.isNum(a.tax.breakeven_move_pct) ? ` · needs ${f.num(a.tax.breakeven_move_pct, 2)}% move to cover costs` : '',
      t1 && f.isNum(t1.net_after_tax) ? h('span', null, ' · you keep at T1 ', h('b', { class: f.tone(t1.net_after_tax) },
        f.inr(t1.net_after_tax, { sign: true }))) : null),
    h('div', { class: 'actions' }, approveBtn, rejectBtn,
      a.decision_id ? h('button', { class: 'btn ghost', onclick: () => openDecision(a.decision_id) }, 'Full reasoning') : null));
}

export function approvalsList(items, opts) {
  if (!items?.length) return ui.empty('Nothing waiting for you', 'Trades that need your approval appear here instantly.', 'check');
  return h('div', { class: 'stack' }, items.map((a) => approvalCard(a, opts)));
}
