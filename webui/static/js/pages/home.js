// Home cockpit: capital & P&L KPIs, alerts, open positions, approvals, autonomy + engine controls, live activity.
import { api, on, qs } from '../api.js';
import { h, icon, debounce } from '../dom.js';
import * as f from '../format.js';
import * as ui from '../ui.js';
import { go, pageTimers } from '../router.js';
import { store, loadOverview } from '../store.js';
import { positionCard } from '../components/positions.js';
import { approvalsList } from '../components/approvals.js';
import { autonomyControl, engineControls, closePositionButton } from '../components/controls.js';
import { openDecision } from '../components/decision.js';
import { summaryBlock } from '../components/summary.js';
import { planTable } from '../components/followup.js';
import { scopeParams, scopeLabel, inScope } from '../scope.js';

export const title = 'Home';

export function mount(root) {
  const t = pageTimers();
  const kpis = summaryBlock();
  const scopeTitle = h('span', { class: 'scope-tag' });
  const alerts = h('div', { class: 'alerts' });
  const positions = ui.card('Open positions', { hint: 'live LTP · unrealized P&L net of entry charges', body: ui.skeleton(4) });
  const approvals = ui.card('Waiting for your approval', { body: ui.skeleton(3) });
  approvals.id = 'approvals';
  const autonomy = ui.card('Who decides?', { hint: 'auto vs human-in-the-loop', body: ui.skeleton(3) });
  const engine = ui.card('Engine', { body: ui.skeleton(2) });
  const planHint = h('span', { class: 'hint' });
  const planCard = ui.card(h('div', { class: 'row' }, h('h2', null, 'Plan'), planHint), { flush: true,
    actions: h('a', { href: '#/watchlists?plans=1', class: 'btn sm ghost' }, 'Follow-up →') });
  planCard.classList.add('hidden');
  const feedList = h('div', { class: 'feed' });
  const activity = ui.card('Latest activity', { hint: 'live', flush: true,
    actions: h('a', { href: '#/live', class: 'btn sm ghost' }, 'Open Live →'), body: h('div', { class: 'feed-scroll' }, feedList) });

  root.append(
    h('div', { class: 'page-head' }, h('h1', null, 'Cockpit'), scopeTitle, h('span', { class: 'sub', id: 'home-batch' }, ''), h('span', { class: 'grow' }),
      h('button', { class: 'btn sm', title: 'Refresh (r)', onclick: () => refresh(false) }, icon('refresh', 14), 'Refresh')),
    kpis, alerts,
    h('div', { class: 'home-grid' },
      h('div', { class: 'grid' }, positions, planCard, approvals),
      h('div', { class: 'grid' }, autonomy, engine, activity)));

  let autonomyCtl = null;
  let lastEngineKey = '';

  const renderOverview = (o) => {
    const cap = o.capital || {};
    document.getElementById('home-batch').textContent = o.batch ? `· active: ${o.batch.name} (${o.batch.status})` : '· no active batch';
    // alerts
    const al = [];
    if (cap.halted || o.batch?.status === 'HALTED') al.push(ui.notice('bad', 'HALTED - capital protection stopped trading', cap.halt_reason || o.batch?.halt_reason || 'Review before resuming from the Streamlit console.'));
    if (o.engine && !o.engine.alive) al.push(ui.notice('warn', `Engine is ${o.engine.label}`, 'No fresh heartbeat. Start it in a terminal:  .\\tasks.ps1 run   (or: python main.py engine)'));
    if (o.engine?.data_feed_ok === false) al.push(ui.notice('warn', 'Market data feed reported DOWN', 'Prices may be stale; the engine skips stale quotes.'));
    if (o.engine?.llm_ok === false) al.push(ui.notice('warn', 'Local LLM reported DOWN', 'Agents fall back or fail; check Ollama on the Health page.'));
    if (o.engine?.last_error && o.engine.alive) al.push(ui.notice('info', 'Last engine error', o.engine.last_error));
    alerts.replaceChildren(...al);
    // positions
    const rows = o.positions || [];
    positions.head.querySelector('h2').textContent = `Open positions (${rows.length})`;
    positions.body.replaceChildren(rows.length ? h('div', { class: 'pos-list' }, rows.map((p) => {
      const c = positionCard(p, { onClick: () => go(`/live?symbol=${encodeURIComponent(p.symbol)}`) });
      c.querySelector('.top').appendChild(closePositionButton(p, { onDone: () => loadOverview() }));
      return c;
    })) : ui.empty('No open positions', 'When the engine enters a trade it shows up here with live P&L and its stop/target progress.', 'chart'));
    // autonomy + engine (avoid re-rendering widgets the owner may be using)
    if (!autonomyCtl) { autonomyCtl = autonomyControl(o.autonomy); autonomy.body.replaceChildren(autonomyCtl); }
    else autonomyCtl.update(o.autonomy);
    const ek = `${o.engine?.state}|${o.engine?.alive}`;
    if (ek !== lastEngineKey) { lastEngineKey = ek; engine.body.replaceChildren(engineControls(o.engine)); }
  };

  const loadApprovals = async (background = true) => {
    try {
      const a = await api.get('/api/approvals', { background });
      approvals.head.querySelector('h2').textContent = `Waiting for your approval (${a.count})`;
      approvals.classList.toggle('accent', a.count > 0);
      approvals.body.replaceChildren(approvalsList(a.items, { onChanged: () => { loadApprovals(); loadOverview(); } }));
    } catch (err) {
      approvals.body.replaceChildren(ui.errorBox(err));
    }
  };

  // ---- activity feed
  const feedItems = [];
  const addFeed = (item, isNew) => {
    feedItems.unshift(item);
    const el = feedRow(item, isNew);
    feedList.prepend(el);
    while (feedList.children.length > 40) feedList.lastChild.remove();
    feedList.parentElement.querySelector('.empty')?.remove();
  };
  let feedSeq = 0;   // a stale response (scope changed meanwhile) never overwrites the feed
  const loadFeed = async () => {
    const my = ++feedSeq;
    try {
      const d = await api.get('/api/decisions' + qs({ limit: 20, ...scopeParams() }), { background: true });
      if (my !== feedSeq) return;
      feedList.replaceChildren();
      const rows = (d.rows || []).slice().reverse();
      for (const r of rows) addFeed({ kind: 'decision', ts: r.time, data: r }, false);
      if (!rows.length) feedList.replaceChildren(ui.empty('No analysis in this scope yet', 'Analysed stocks, orders and exits stream in here live.', 'live'));
    } catch (err) { if (my === feedSeq) feedList.replaceChildren(ui.errorBox(err)); }
  };

  // ---- next-day plan (hidden when there is none or the follow-up backend is not deployed yet)
  let planSeq = 0;
  const loadPlan = async () => {
    const my = ++planSeq;
    try {
      const d = await api.get('/api/plan', { background: true });
      if (my !== planSeq) return;
      const p = d.plan;
      if (!p) { planCard.classList.add('hidden'); return; }
      const c = p.counts || {};
      planCard.head.querySelector('h2').textContent = p.label || 'Plan';
      planHint.textContent = `${(p.items || []).length} stock(s) · ${c.ACTIVE ?? 0} active · ${c.TRIGGERED ?? 0} triggered · ${c.INVALIDATED ?? 0} invalidated`;
      planCard.body.replaceChildren(planTable(p.items || [], { onSymbol: true }),
        h('div', { class: 'card-b tiny faint' }, 'From the end-of-day scan. A plan is not an order: every entry still goes through the AI and the risk manager.'));
      planCard.classList.remove('hidden');
    } catch {
      if (my === planSeq) planCard.classList.add('hidden');
    }
  };

  let sumSeq = 0;
  const loadSummary = async (background = true) => {
    const my = ++sumSeq;
    scopeTitle.replaceChildren('Showing ', h('b', null, scopeLabel()));
    try {
      const d = await api.get('/api/summary' + qs(scopeParams()), { background });
      if (my === sumSeq) kpis.render(d);
    } catch (err) { if (my === sumSeq) kpis.fail(err); }
  };
  const summarySoon = debounce(() => loadSummary(), 1500);
  const refresh = (background = true) => {
    loadOverview({ background }).catch(() => {});
    loadSummary(background);
    loadApprovals(background);
  };

  t.track(on('overview', renderOverview));
  t.track(on('sse:approvals', () => loadApprovals()));
  t.track(on('sse:decision', (r) => { if (inScope(r)) addFeed({ kind: 'decision', ts: r.time, data: r }, true); }));
  t.track(on('sse:order', (o) => { addFeed({ kind: 'order', ts: o.time, data: o }, true); summarySoon(); }));
  t.track(on('sse:trade', (tr) => { addFeed({ kind: 'trade', ts: tr.exit_time, data: tr }, true); summarySoon(); }));
  t.track(on('scope', () => { loadSummary(false); loadFeed(); }));
  t.track(on('poll:tick', () => loadOverview().catch(() => {})));
  const onRefresh = () => refresh(false);
  window.addEventListener('page:refresh', onRefresh);
  t.track(() => window.removeEventListener('page:refresh', onRefresh));
  t.every(15000, () => loadOverview().catch(() => {}));
  t.every(20000, () => loadSummary());
  t.every(60000, loadPlan);
  t.track(on('watchlists:changed', () => loadPlan()));

  if (store.overview) renderOverview(store.overview);
  refresh(true);
  loadFeed();
  loadPlan();
  return () => t.clear();
}

function feedRow(item, isNew) {
  const d = item.data;
  let line1; let why; let onclick;
  if (item.kind === 'decision') {
    line1 = [h('span', { class: 'sym' }, d.symbol), h('span', { class: 'meta' }, [d.strategy, d.mode].filter(Boolean).join(' · ')),
      ui.gradeBadge(d.grade, d.confluence), ui.actionChip(d.final_action)];
    why = d.reason;
    onclick = () => openDecision(d.id);
  } else if (item.kind === 'order') {
    const tone = d.status === 'FILLED' ? 'good' : d.status === 'PENDING' ? 'warn' : 'bad';
    line1 = [h('span', { class: 'sym' }, d.symbol), h('span', { class: 'meta' }, `${d.side} ${d.purpose.replace('_', ' ').toLowerCase()} · ${d.qty} qty`),
      ui.badge(`ORDER ${d.status}`, tone)];
    why = d.fill_price ? `filled @ ${f.price(d.fill_price)}` : d.reason || `${d.type} order`;
    onclick = d.position_id ? () => go(`/trade/${d.position_id}`) : (d.decision_id ? () => openDecision(d.decision_id) : null);
  } else {
    line1 = [h('span', { class: 'sym' }, d.symbol), h('span', { class: 'meta' }, d.exit_label),
      ui.badge(`EXIT ${f.inr(d.net, { sign: true })}`, d.net >= 0 ? 'good' : 'bad')];
    why = `${d.qty} @ ${f.price(d.exit_price)} · gross ${f.inr(d.gross, { sign: true })} · charges ${f.inr(d.charges)}`;
    onclick = () => go(`/trade/${d.position_id}`);
  }
  return h('div', { class: ['feed-item', isNew && 'new'], onclick },
    h('div', { class: 't' }, f.hm(item.ts)),
    h('div', { class: 'body' }, h('div', { class: 'line1' }, line1), why ? h('div', { class: 'why clamp-2' }, why) : null));
}
