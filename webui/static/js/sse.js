// One shared EventSource for the whole app. Events are re-emitted on the bus as `sse:<type>`.
// Auto-reconnects with backoff; after repeated failures falls back to polling until SSE works again.
import { api, emit, qs } from './api.js';
import { setServerTime } from './format.js';

const TYPES = ['hello', 'decision', 'order', 'trade', 'status', 'approvals', 'log', 'session_expired'];
const cursors = { decision: null, order: null, trade: null };
let es = null;
let state = 'idle';            // idle | live | reconnecting | polling
let failures = 0;
let retryTimer = null;
let pollTimer = null;
let stopped = true;

function setState(s) {
  if (state === s) return;
  state = s;
  emit('stream:state', s);
}
export function streamState() { return state; }

function handle(type, raw) {
  let data = null;
  try { data = raw ? JSON.parse(raw) : null; } catch { return; }
  if (type === 'hello') {
    failures = 0;
    setState('live');
    if (data?.server_time) setServerTime(data.server_time);
    for (const k of Object.keys(cursors)) if (cursors[k] === null) cursors[k] = data?.cursors?.[k] ?? 0;
    stopPolling();
  }
  if (type === 'decision' && data?.id) cursors.decision = Math.max(cursors.decision || 0, data.id);
  if (type === 'order' && data?.id) cursors.order = Math.max(cursors.order || 0, data.id);
  if (type === 'trade' && data?.id) cursors.trade = Math.max(cursors.trade || 0, data.id);
  if (type === 'status' && data?.server_time) setServerTime(data.server_time);
  if (type === 'session_expired') {
    reconnect(200);
    return;
  }
  emit(`sse:${type}`, data);
}

function connect() {
  if (stopped) return;
  closeSource();
  const url = '/api/stream' + qs({ decision: cursors.decision, order: cursors.order, trade: cursors.trade });
  try {
    es = new EventSource(url, { withCredentials: true });
  } catch {
    onFailure();
    return;
  }
  for (const t of TYPES) es.addEventListener(t, (e) => handle(t, e.data));
  es.onerror = () => {
    // Never let the browser auto-reconnect with the stale cursors in the original URL: reconnect
    // ourselves with the latest ids so no event is replayed or lost.
    const closed = !es || es.readyState === EventSource.CLOSED;
    closeSource();
    if (closed) { onFailure(); return; }
    failures += 1;
    setState('reconnecting');
    if (failures >= 3) startPolling();
    reconnect(Math.min(15000, 1000 * failures));
  };
}

function closeSource() {
  if (es) { es.close(); es = null; }
}

async function onFailure() {
  closeSource();
  failures += 1;
  // Was it an expired session? A background /api/me returns 401 -> app shows the login screen.
  try { await api.get('/api/me', { background: true }); } catch (err) { if (err.status === 401) { stop(); return; } }
  if (failures >= 3) startPolling();
  else setState('reconnecting');
  reconnect(Math.min(30000, 1000 * 2 ** Math.min(failures, 5)));
}

function reconnect(ms) {
  clearTimeout(retryTimer);
  retryTimer = setTimeout(connect, ms);
}

// ------------------------------------------------------------------ polling fallback
function startPolling() {
  setState('polling');
  if (pollTimer) return;
  const tick = async () => {
    if (document.hidden) return;
    try {
      const d = await api.get('/api/decisions' + qs({ since_id: cursors.decision ?? undefined, limit: 50 }), { background: true });
      const rows = (d.rows || []).slice().reverse();
      if (cursors.decision === null) cursors.decision = d.latest_id || 0;
      else for (const r of rows) handle('decision', JSON.stringify(r));
      emit('poll:tick', {});
    } catch { /* keep trying */ }
  };
  tick();
  pollTimer = setInterval(tick, 5000);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

export function start() {
  if (!stopped) return;
  stopped = false;
  failures = 0;
  setState('reconnecting');
  connect();
}

export function stop() {
  stopped = true;
  clearTimeout(retryTimer);
  stopPolling();
  closeSource();
  for (const k of Object.keys(cursors)) cursors[k] = null;
  setState('idle');
}
