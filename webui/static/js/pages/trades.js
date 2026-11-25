// Trades journal: every completed trade across batches, filters, totals and CSV export.
import { api, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { go, pageTimers } from '../router.js';
import { tradeColumns } from './today.js';

export const title = 'Trades journal';

export function mount(root, params) {
  const t = pageTimers();
  const st = { batch_id: params.batch_id || '', from: params.from || '', to: params.to || '', mode: params.mode || '',
    result: params.result || '', symbol: params.symbol || '' };
  const batchSel = h('select', { class: 'input sm', 'aria-label': 'Batch' }, h('option', { value: '' }, 'All batches'));
  const fromIn = h('input', { type: 'date', class: 'input sm', value: st.from, 'aria-label': 'From date' });
  const toIn = h('input', { type: 'date', class: 'input sm', value: st.to, 'aria-label': 'To date' });
  const modeSel = h('select', { class: 'input sm', 'aria-label': 'Mode' }, h('option', { value: '' }, 'All modes'),
    ['INTRADAY', 'BTST', 'SWING', 'POSITIONAL'].map((m) => h('option', { value: m, selected: st.mode === m }, m)));
  const resSel = h('select', { class: 'input sm', 'aria-label': 'Result' }, h('option', { value: '' }, 'All results'),
    [['WIN', 'Wins'], ['LOSS', 'Losses'], ['FLAT', 'Flat'], ['OPEN', 'Partially open']].map(([v, l]) => h('option', { value: v, selected: st.result === v }, l)));
  const symIn = h('input', { class: 'input sm mono', placeholder: 'Symbol', value: st.symbol, 'data-search': '', style: { width: '120px' }, 'aria-label': 'Symbol' });
  const csvBtn = h('button', { class: 'btn sm' }, icon('download', 14), 'CSV');
  const summary = h('div', { class: 'strip' }, ui.skeleton(1));
  const tableCard = ui.card('Completed trades', { flush: true, body: ui.skeleton(8),
    hint: 'one row per position · newest first · click a row for the full story' });
  let rows = [];

  const filters = h('div', { class: 'row' },
    h('label', { class: 'field' }, h('span', null, 'Batch'), batchSel),
    h('label', { class: 'field' }, h('span', null, 'From'), fromIn),
    h('label', { class: 'field' }, h('span', null, 'To'), toIn),
    h('label', { class: 'field' }, h('span', null, 'Mode'), modeSel),
    h('label', { class: 'field' }, h('span', null, 'Result'), resSel),
    h('label', { class: 'field' }, h('span', null, 'Symbol'), symIn),
    h('div', { class: 'field' }, h('span', null, ' '), h('div', { class: 'row' },
      h('button', { class: 'btn sm ghost', onclick: () => { Object.keys(st).forEach((k) => { st[k] = ''; }); for (const el of [fromIn, toIn, symIn]) el.value = ''; batchSel.value = ''; modeSel.value = ''; resSel.value = ''; load(); } }, 'Reset'),
      csvBtn)));
  root.append(h('div', { class: 'page-head' }, h('h1', null, 'Trades journal'), h('span', { class: 'sub' }, 'all completed trades across batches')),
    ui.card(null, { body: filters }), summary, tableCard);

  const read = () => {
    st.batch_id = batchSel.value; st.from = fromIn.value; st.to = toIn.value; st.mode = modeSel.value; st.result = resSel.value;
    st.symbol = symIn.value.trim().toUpperCase();
  };
  const load = async () => {
    read();
    try {
      const d = await api.get('/api/trades' + qs({ ...st, limit: 2000 }));
      rows = d.rows;
      if (batchSel.options.length <= 1) {
        for (const b of d.batches) batchSel.appendChild(h('option', { value: String(b.id), selected: String(b.id) === String(st.batch_id) }, `${b.name} (${b.status})`));
      }
      const tt = d.totals;
      const cell = (label, value, sub = null) => h('div', { class: 'cell' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value),
        sub ? h('div', { class: 'sub' }, sub) : null);
      summary.replaceChildren(
        cell('Trades', String(tt.closed), d.truncated ? `showing ${rows.length} of ${d.count}` : `${tt.wins} won · ${tt.losses} lost`),
        cell('Net P&L', ui.money(tt.net), `gross ${f.inr(tt.gross, { sign: true })}`),
        cell('Win rate', f.isNum(tt.win_rate) ? `${f.num(tt.win_rate, 1)}%` : '–', f.isNum(tt.avg_r) ? `avg ${f.rmult(tt.avg_r)}` : null),
        cell('Avg win / loss', h('span', null, h('span', { class: 'pos' }, f.inr(tt.avg_win, { sign: true })), ' / ', h('span', { class: 'neg' }, f.inr(tt.avg_loss, { sign: true })))),
        cell('Profit factor', f.isNum(tt.profit_factor) ? f.num(tt.profit_factor, 2) : '–', 'gross wins ÷ gross losses'),
        cell('Charges paid', f.inr(tt.charges)));
      tableCard.head.querySelector('h2').textContent = `Completed trades (${d.count})`;
      tableCard.body.replaceChildren(ui.table(tradeColumns({ withDate: true, withBatch: true }), rows, {
        onRow: (r) => go(`/trade/${r.position_id}`), tall: true,
        footer: rows.length ? { symbol: 'Total', gross: ui.money(tt.gross), charges: f.inr(tt.charges), net: h('b', null, ui.money(tt.net)),
          r_multiple: f.isNum(tt.avg_r) ? `avg ${f.rmult(tt.avg_r)}` : '' } : null,
        empty: ui.empty('No trades match these filters', 'Completed trades show here once positions close.', 'trades') }));
    } catch (err) {
      tableCard.body.replaceChildren(ui.errorBox(err));
    }
  };
  const later = debounce(load, 350);
  for (const el of [batchSel, fromIn, toIn, modeSel, resSel]) el.addEventListener('change', load);
  symIn.addEventListener('input', later);
  csvBtn.addEventListener('click', () => exportCsv(rows));
  const onRefresh = () => load();
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  load();
  return () => t.clear();
}

function csvCell(v) {
  if (v === null || v === undefined) return '';
  const s = Array.isArray(v) ? v.join(' + ') : String(v);
  // neutralise spreadsheet formula injection (symbols/LLM text come from the DB)
  const safe = /^[=+\-@\t\r]/.test(s) && !/^-?\d+(\.\d+)?$/.test(s) ? `'${s}` : s;
  return /[",\n]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe;
}

export function exportCsv(rows) {
  if (!rows?.length) { ui.toast('Nothing to export.', 'info'); return; }
  const cols = ['position_id', 'batch_id', 'symbol', 'mode', 'direction', 'strategy', 'grade', 'qty', 'entry_time', 'entry_price',
    'exit_time', 'exit_price', 'initial_sl', 'final_sl', 'target_1', 'target_2', 'exit_reason', 'gross', 'charges', 'net',
    'r_multiple', 'holding_minutes', 'result'];
  const lines = [cols.join(',')];
  for (const r of rows) lines.push(cols.map((c) => csvCell(r[c])).join(','));
  const blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `liar-v1-trades-${f.todayIST()}.csv`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
