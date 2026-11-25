// Follow-up: the owner's lists of stocks to watch (create / rename / delete lists, add / remove stocks, live LTP and
// the move since a stock was added, latest analysis) plus the system's next-day plans (NEXT_DAY lists).
// Lists are owner data written straight through data.watchlists (audited); scanning a list is a SCAN_NOW command.
import { api, on, qs } from '../api.js';
import { h, icon } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { go, pageTimers } from '../router.js';
import { openDecision } from '../components/decision.js';
import { cleanSymbol, invalidateLists, planColumns, statusChip, NOT_READY } from '../components/followup.js';

export const title = 'Follow-up';

const SEL_KEY = 'liar.followup.sel';
const MODES = ['INTRADAY', 'BTST', 'SWING', 'POSITIONAL'];
const SOURCE_TONE = { MANUAL: '', ANALYSIS: 'cyan', SCAN: 'violet', EOD_PLAN: 'good' };
const TRADING_HELP = 'INTRADAY scans analyse these stocks first; every trade still needs the risk manager.';

function savedSel() { try { return Number(localStorage.getItem(SEL_KEY)) || null; } catch { return null; } }
function saveSel(id) { try { localStorage.setItem(SEL_KEY, String(id || '')); } catch { /* ignore */ } }
const isPlan = (l) => String(l?.kind || '').toUpperCase() === 'NEXT_DAY';
const planDay = (l) => (l?.for_date ? f.weekday(`${l.for_date}T12:00:00+05:30`) : '');

export function mount(root, params) {
  const t = pageTimers();
  const st = { lists: [], sel: Number(params.id) || savedSel(), detail: null, quotes: new Map(), wantPlans: params.plans === '1' };

  const listsBody = h('div', { class: 'wl-nav' }, ui.skeleton(4));
  const plansBody = h('div', { class: 'wl-nav' });
  const newBtn = h('button', { class: 'btn sm primary', onclick: () => createList() }, icon('plus', 13), 'New list');
  const listsCard = ui.card('Your lists', { actions: newBtn, body: listsBody, flush: true });
  const plansCard = ui.card('Plans', { hint: 'next-day setups from the end-of-day scan', body: plansBody, flush: true });
  const mainCard = h('section', { class: 'card wl-main' }, h('div', { class: 'card-b' }, ui.skeleton(6)));
  const pageNote = h('div');

  root.append(
    h('div', { class: 'page-head' }, h('h1', null, 'Follow-up'),
      h('span', { class: 'sub' }, 'stocks you want to keep an eye on · nothing here places a trade'), h('span', { class: 'grow' }),
      h('button', { class: 'btn sm', title: 'Refresh (r)', onclick: () => loadAll() }, icon('refresh', 14), 'Refresh')),
    pageNote,
    h('div', { class: 'wl-grid' }, h('div', { class: 'grid' }, listsCard, plansCard), mainCard));

  // ------------------------------------------------------------------ data
  const notReady = (err) => err?.status === 501;
  const showNotReady = () => {
    pageNote.replaceChildren(ui.notice('info', 'Follow-up lists are not available yet',
      'The backend for follow-up lists (data/watchlists.py) is still being installed. This page starts working as soon as it is there - nothing else is affected.'));
    listsBody.replaceChildren(ui.empty('Not available yet', '', 'list'));
    plansBody.replaceChildren();
    mainCard.replaceChildren(h('div', { class: 'card-b' }, ui.empty('Follow-up lists are coming', 'Then you can add any analysed stock here with one click and follow its price.', 'list')));
    newBtn.disabled = true;
  };

  let listSeq = 0;
  const loadLists = async () => {
    const my = ++listSeq;
    try {
      const d = await api.get('/api/watchlists', { background: true });
      if (my !== listSeq) return false;
      pageNote.replaceChildren();
      newBtn.disabled = false;
      st.lists = d.lists || [];
      if (st.wantPlans) {
        const plan = st.lists.filter(isPlan).sort((a, b) => String(b.for_date).localeCompare(String(a.for_date)))[0];
        if (plan) st.sel = plan.id;
        st.wantPlans = false;
      }
      if (!st.lists.some((l) => l.id === st.sel)) st.sel = (st.lists.find((l) => !isPlan(l)) || st.lists[0])?.id || null;
      drawNav();
      return true;
    } catch (err) {
      if (my !== listSeq) return false;
      if (notReady(err)) showNotReady();
      else listsBody.replaceChildren(ui.errorBox(err, 'Could not load your lists'));
      return false;
    }
  };

  const navItem = (l) => h('button', { type: 'button', class: ['wl-item', l.id === st.sel && 'on'], 'aria-current': l.id === st.sel ? 'true' : null,
    onclick: () => select(l.id) },
  h('span', { class: 'name ellipsis' }, isPlan(l) ? `Plan · ${planDay(l)}` : l.name),
  l.use_for_trading ? ui.badge('TRADING', 'warn', TRADING_HELP) : null,
  h('span', { class: 'n' }, String(l.items ?? 0)));

  const drawNav = () => {
    const manual = st.lists.filter((l) => !isPlan(l));
    const plans = st.lists.filter(isPlan).sort((a, b) => String(b.for_date).localeCompare(String(a.for_date)));
    listsBody.replaceChildren(manual.length ? manual.map(navItem)
      : h('div', { class: 'card-b' }, ui.empty('No lists yet', 'Create one with "New list", or press "Add to follow-up" on any analysed stock - a "Follow-up" list is created for you.', 'list')));
    plansBody.replaceChildren(plans.length ? plans.map(navItem)
      : h('div', { class: 'card-b small faint' }, 'No next-day plan yet. After the close the end-of-day scan saves tomorrow\'s setups here.'));
  };

  let detailSeq = 0;
  const loadDetail = async (background = true) => {
    const my = ++detailSeq;
    if (!st.sel) { st.detail = null; drawMain(); return; }
    try {
      const d = await api.get(`/api/watchlists/${st.sel}`, { background });
      if (my !== detailSeq) return;
      st.detail = d;
      drawMain();
      refreshQuotes();
    } catch (err) {
      if (my !== detailSeq) return;
      if (notReady(err)) { showNotReady(); return; }
      if (err.status === 404) { st.sel = null; await loadLists(); loadDetail(); return; }
      mainCard.replaceChildren(h('div', { class: 'card-b' }, ui.errorBox(err, 'Could not load the list')));
    }
  };

  const loadAll = async () => { if (await loadLists()) await loadDetail(false); };

  const select = (id) => {
    st.sel = id;
    saveSel(id);
    history.replaceState(null, '', `#/watchlists?id=${id}`);
    drawNav();
    mainCard.replaceChildren(h('div', { class: 'card-b' }, ui.skeleton(6)));
    loadDetail(false);
  };

  // ------------------------------------------------------------------ live quotes (10-15 s)
  let quoteSeq = 0;
  const refreshQuotes = async () => {
    const items = st.detail?.items || [];
    const syms = [...new Set(items.map((i) => i.symbol))];
    if (!syms.length) return;
    const my = ++quoteSeq;
    try {
      for (let i = 0; i < syms.length; i += 40) {
        const d = await api.get('/api/quotes' + qs({ symbols: syms.slice(i, i + 40).join(',') }), { background: true });
        if (my !== quoteSeq) return;
        for (const [s, q] of Object.entries(d.quotes || {})) if (q) st.quotes.set(s, q);
      }
      if (my === quoteSeq && tableEl) tableEl.update(items);
    } catch { /* keep the last prices */ }
  };

  // ------------------------------------------------------------------ actions
  const fail = (title) => (err) => ui.toast(notReady(err) ? NOT_READY : err.message, notReady(err) ? 'info' : 'bad', { title });
  const changed = () => { invalidateLists(); loadLists(); loadDetail(); };

  const createList = async () => {
    const name = h('input', { class: 'input', maxlength: '80', placeholder: 'e.g. Breakout candidates', autocomplete: 'off' });
    const desc = h('input', { class: 'input', maxlength: '400', placeholder: 'optional', autocomplete: 'off' });
    const trade = h('input', { type: 'checkbox' });
    const ok = await ui.confirmModal({ title: 'New follow-up list', confirmLabel: 'Create',
      body: h('div', { class: 'stack' }, h('label', { class: 'field' }, h('span', null, 'Name'), name),
        h('label', { class: 'field' }, h('span', null, 'Description'), desc),
        h('label', { class: 'toggle' }, trade, 'Use for trading'), h('div', { class: 'small muted' }, TRADING_HELP)) });
    if (!ok) return;
    if (!name.value.trim()) { ui.toast('Give the list a name.', 'warn'); return; }
    try {
      const wl = await api.post('/api/watchlists', { name: name.value.trim(), description: desc.value.trim() || null, use_for_trading: trade.checked });
      ui.toast(`List "${wl.name}" created.`, 'good');
      invalidateLists();
      await loadLists();
      select(wl.id);
    } catch (err) { fail('Could not create the list')(err); }
  };

  const renameList = async (wl) => {
    const name = h('input', { class: 'input', maxlength: '80', value: wl.name, autocomplete: 'off' });
    const desc = h('input', { class: 'input', maxlength: '400', value: wl.description || '', autocomplete: 'off' });
    const ok = await ui.confirmModal({ title: `Rename "${wl.name}"`, confirmLabel: 'Save',
      body: h('div', { class: 'stack' }, h('label', { class: 'field' }, h('span', null, 'Name'), name),
        h('label', { class: 'field' }, h('span', null, 'Description'), desc)) });
    if (!ok) return;
    try {
      await api.patch(`/api/watchlists/${wl.id}`, { name: name.value.trim(), description: desc.value.trim() });
      changed();
    } catch (err) { fail('Could not rename the list')(err); }
  };

  const deleteList = async (wl) => {
    const n = (wl.items || []).length;
    const ok = await ui.confirmModal({ title: `Delete "${wl.name}"?`, danger: true, confirmLabel: 'Delete list',
      body: h('div', null, `The list and its ${n} stock(s) are removed. Past analyses and trades are not affected.`) });
    if (!ok) return;
    try {
      await api.del(`/api/watchlists/${wl.id}`);
      ui.toast(`List "${wl.name}" deleted.`, 'good');
      st.sel = null;
      invalidateLists();
      await loadLists();
      loadDetail(false);
    } catch (err) { fail('Could not delete the list')(err); }
  };

  const setTrading = async (wl, value, input) => {
    try {
      await api.patch(`/api/watchlists/${wl.id}`, { use_for_trading: value });
      ui.toast(value ? `"${wl.name}" will be analysed first in INTRADAY scans. Every trade still needs the risk manager.`
        : `"${wl.name}" is no longer used for trading.`, 'good', { title: 'Use for trading' });
      changed();
    } catch (err) { input.checked = !value; fail('Could not change the list')(err); }
  };

  const addStock = async (wl, symIn, noteIn, modeSel, errEl) => {
    const sym = cleanSymbol(symIn.value);
    errEl.textContent = '';
    if (!sym) { errEl.textContent = symIn.value.trim() ? `"${symIn.value.trim()}" is not a valid NSE symbol (letters, digits, & - _ .)` : 'Type a symbol, e.g. RELIANCE'; symIn.focus(); return; }
    if ((wl.items || []).some((i) => i.symbol === sym)) { errEl.textContent = `${sym} is already on this list.`; return; }
    try {
      const item = await api.post(`/api/watchlists/${wl.id}/items`, { symbol: sym, note: noteIn.value.trim() || null, mode: modeSel.value || null });
      ui.toast(`${sym} added${f.isNum(item?.added_price) ? ` at ${f.price(item.added_price)}` : ''}.`, 'good');
      symIn.value = ''; noteIn.value = '';
      changed();
    } catch (err) { errEl.textContent = notReady(err) ? NOT_READY : err.message; }
  };

  const removeItem = async (wl, item) => {
    const ok = await ui.confirmModal({ title: `Remove ${item.symbol}?`, confirmLabel: 'Remove',
      body: h('div', null, `${item.symbol} is removed from "${wl.name}". You can add it again any time.`) });
    if (!ok) return;
    try {
      await api.del(`/api/watchlists/${wl.id}/items/${item.id}`);
      changed();
    } catch (err) { fail(`Could not remove ${item.symbol}`)(err); }
  };

  const analyze = (btn, item, wl) => {
    const mode = item.mode || item.plan_view?.mode || 'INTRADAY';
    ui.sendCommand(btn, `Analyze ${item.symbol}`, () => api.post('/api/control', { action: 'analyze_now', symbol: item.symbol, mode: MODES.includes(mode) ? mode : 'INTRADAY' }),
      { onDone: () => { if (st.sel === wl.id) loadDetail(); } });
  };

  const scanList = (btn, wl, modeSel, ai) => {
    ui.sendCommand(btn, `Scan "${wl.name}"`, () => api.post(`/api/watchlists/${wl.id}/scan`, { mode: modeSel.value, use_llm: ai.checked }),
      { onDone: () => { if (st.sel === wl.id) loadDetail(); } });
  };

  // ------------------------------------------------------------------ main panel
  let tableEl = null;
  const moveSinceAdded = (r) => {
    const q = st.quotes.get(r.symbol);
    if (!f.isNum(q?.ltp) || !f.isNum(r.added_price) || !r.added_price) return null;
    const d = q.ltp - r.added_price;
    return { abs: d, pct: (d / r.added_price) * 100 };
  };

  const actionsCell = (r, wl) => {
    const an = h('button', { class: 'btn sm', title: 'Analyse now (analysis only - never trades)' }, icon('bolt', 12), 'Analyze');
    an.addEventListener('click', (e) => { e.stopPropagation(); analyze(an, r, wl); });
    return h('div', { class: 'row nowrap' }, an,
      h('button', { class: 'btn sm', title: 'Open the chart on Live', onclick: (e) => { e.stopPropagation(); go(`/live?symbol=${encodeURIComponent(r.symbol)}`); } },
        icon('chart', 12), 'Chart'),
      h('button', { class: 'btn sm danger-ghost', title: `Remove ${r.symbol} from this list`, 'aria-label': `Remove ${r.symbol}`,
        onclick: (e) => { e.stopPropagation(); removeItem(wl, r); } }, icon('trash', 12)));
  };

  const manualColumns = (wl) => [
    { key: 'symbol', label: 'Stock', fmt: (r) => ui.twoLine(h('span', { class: 'row nowrap' }, h('span', { class: 'sym' }, r.symbol), ui.modeBadge(r.mode)),
      `added ${f.stamp(r.added_at)}${f.isNum(r.added_price) ? ` @ ${f.price(r.added_price)}` : ''}`) },
    { key: 'ltp', label: 'LTP', num: true, sort: (r) => st.quotes.get(r.symbol)?.ltp, fmt: (r) => {
      const q = st.quotes.get(r.symbol);
      return ui.twoLine(h('b', null, f.price(q?.ltp)), f.isNum(q?.change_pct) ? h('span', { class: f.tone(q.change_pct) }, `${f.pct(q.change_pct, { sign: true })} today`) : null);
    } },
    { key: 'since', label: 'Since added', num: true, sort: (r) => moveSinceAdded(r)?.pct, fmt: (r) => {
      const m = moveSinceAdded(r);
      if (!m) return h('span', { class: 'faint', title: f.isNum(r.added_price) ? 'waiting for a live price' : 'no price was saved when it was added' }, '–');
      return ui.twoLine(h('b', { class: f.tone(m.abs) }, f.inr(m.abs, { sign: true })), h('span', { class: f.tone(m.pct) }, f.pct(m.pct, { sign: true })));
    } },
    { key: 'note', label: 'Note', wrap: true, sortable: false, fmt: (r) => (r.note ? h('div', { class: 'clamp-2', title: r.note }, r.note) : null) },
    { key: 'source', label: 'Source', fmt: (r) => ui.badge(String(r.source || 'MANUAL').replace('_', ' '), SOURCE_TONE[r.source] ?? '') },
    { key: 'last_analysis_at', label: 'Latest analysis', wrap: true, sort: (r) => r.last_analysis_at, fmt: (r) => {
      if (!r.last_summary && !r.last_analysis_at) return h('span', { class: 'faint' }, 'not analysed yet');
      return h('div', { class: 'two' },
        h('div', { class: 'clamp-2', title: r.last_summary || null }, r.last_summary || 'analysed'),
        h('div', { class: 's' }, f.stamp(r.last_analysis_at),
          r.last_analysis_id ? h('a', { href: '#', class: 'ml', onclick: (e) => { e.preventDefault(); e.stopPropagation(); openDecision(r.last_analysis_id); } }, ' · details') : null));
    } },
    { key: 'status', label: 'Status', fmt: (r) => h('div', { class: 'two' }, statusChip(r.status, r.status_note),
      r.status_note ? h('div', { class: 's clamp-2' }, r.status_note) : null) },
    { key: 'actions', label: '', sortable: false, fmt: (r) => actionsCell(r, wl) },
  ];

  const drawMain = () => {
    tableEl = null;
    const wl = st.detail;
    if (!wl) {
      mainCard.replaceChildren(h('div', { class: 'card-b' }, ui.empty(st.lists.length ? 'Pick a list on the left' : 'Start your first follow-up list',
        'Add stocks you want to watch: press "Add to follow-up" on any analysed stock (Live feed, decision details, trade pages, Results) or create a list here.', 'list')));
      return;
    }
    const plan = isPlan(wl);
    const items = wl.items || [];
    const head = h('div', { class: 'card-h' },
      h('h2', null, plan ? `Plan for ${planDay(wl)}` : wl.name),
      plan ? ui.badge('NEXT DAY PLAN', 'good') : null,
      wl.use_for_trading ? ui.badge('USED FOR TRADING', 'warn', TRADING_HELP) : null,
      h('span', { class: 'hint' }, `${items.length} stock(s)`),
      h('div', { class: 'actions' },
        plan ? null : h('button', { class: 'btn sm ghost', title: 'Rename', onclick: () => renameList(wl) }, icon('edit', 13), 'Rename'),
        h('button', { class: 'btn sm danger-ghost', title: 'Delete this list', onclick: () => deleteList(wl) }, icon('trash', 13), 'Delete')));

    // scan + trading toggle
    const modeSel = h('select', { class: 'input sm', 'aria-label': 'Mode to analyse in', style: { width: '118px' } },
      MODES.map((m) => h('option', { value: m }, m)));
    const ai = h('input', { type: 'checkbox', checked: true });
    const scanBtn = h('button', { class: 'btn sm primary', disabled: !items.length,
      title: items.length ? 'Screen these stocks now and run the AI on the best setups. Analysis only - never trades.' : 'Add a stock first' },
    icon('search', 13), 'Scan this list now (analysis only)');
    scanBtn.addEventListener('click', () => scanList(scanBtn, wl, modeSel, ai));
    const tradeIn = h('input', { type: 'checkbox', checked: !!wl.use_for_trading });
    tradeIn.addEventListener('change', () => setTrading(wl, tradeIn.checked, tradeIn));
    const tools = h('div', { class: 'card-b stack wl-tools' },
      wl.description ? h('div', { class: 'small muted' }, wl.description) : null,
      h('div', { class: 'row' }, modeSel, h('label', { class: 'toggle', title: 'Send the best setups to the AI agents (slower)' }, ai, 'AI'), scanBtn,
        h('span', { class: 'grow' }),
        plan ? null : h('label', { class: 'toggle', title: TRADING_HELP }, tradeIn, 'Use for trading')),
      plan ? h('div', { class: 'tiny faint' }, 'Saved by the end-of-day scan. Each setup is checked again at 09:20; a plan is not an order - every entry still goes through the AI and the risk manager.')
        : h('div', { class: 'tiny faint' }, `Use for trading: ${TRADING_HELP}`));

    // add stock
    const symIn = h('input', { class: 'input sm mono', placeholder: 'Symbol e.g. TATAMOTORS', maxlength: '24', autocomplete: 'off', spellcheck: 'false',
      'aria-label': 'Symbol to add', style: { width: '170px' } });
    const noteIn = h('input', { class: 'input sm', placeholder: 'Note (optional)', maxlength: '500', autocomplete: 'off', 'aria-label': 'Note',
      style: { width: '220px' } });
    const addMode = h('select', { class: 'input sm', 'aria-label': 'Mode', style: { width: '118px' } },
      h('option', { value: '' }, 'Any mode'), MODES.map((m) => h('option', { value: m }, m)));
    const errEl = h('span', { class: 'small neg', role: 'alert' });
    const addBtn = h('button', { class: 'btn sm', type: 'submit' }, icon('plus', 13), 'Add stock');
    const form = h('form', { class: 'row wl-add' }, symIn, noteIn, addMode, addBtn, errEl);
    form.addEventListener('submit', (e) => { e.preventDefault(); ui.busy(addBtn, () => addStock(wl, symIn, noteIn, addMode, errEl)); });
    symIn.addEventListener('input', () => { errEl.textContent = ''; });

    const cols = plan ? [...planColumns({ quotes: st.quotes, onSymbol: true }), { key: 'actions', label: '', sortable: false, fmt: (r) => actionsCell(r, wl) }]
      : manualColumns(wl);
    tableEl = ui.table(cols, items, { sortable: true, initialSort: plan ? { key: 'score', dir: -1 } : { key: 'last_analysis_at', dir: -1 },
      empty: ui.empty(plan ? 'This plan is empty' : 'No stocks on this list yet',
        plan ? 'The end-of-day scan found nothing worth planning for.' : 'Add one with the box above, or press "Add to follow-up" on any analysed stock (Live feed, decision details, trade pages, Results).', 'list') });
    mainCard.replaceChildren(head, tools, plan ? null : h('div', { class: 'card-b wl-addbox' }, form), tableEl);
  };

  // ------------------------------------------------------------------ wiring
  t.track(on('watchlists:changed', () => { loadLists(); loadDetail(); }));
  const onRefresh = () => loadAll();
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  t.every(12000, refreshQuotes);
  t.every(60000, () => { loadLists(); loadDetail(); });
  loadAll();
  return () => t.clear();
}
