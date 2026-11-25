// fetch wrapper: JSON in/out, CSRF header on writes, X-Background for timer-driven calls
// (they do not extend the 30-minute idle timeout), 401 -> login screen.

export const bus = new EventTarget();
export function on(type, fn) {
  const h = (e) => fn(e.detail);
  bus.addEventListener(type, h);
  return () => bus.removeEventListener(type, h);
}
export function emit(type, detail) { bus.dispatchEvent(new CustomEvent(type, { detail })); }

let csrf = null;
export function setCsrf(token) { csrf = token; }

// Idle timeout: only real user input counts as activity (like the Streamlit dashboard, where
// auto-refresh reruns never extend the session). The first request after a click / key press /
// scroll is sent as a normal request (slides the 30-minute idle timeout); timer-driven requests
// without fresh input carry X-Background: 1 and do not.
let lastInput = Date.now();
let lastForeground = 0;
for (const ev of ['pointerdown', 'keydown', 'wheel', 'touchstart']) {
  window.addEventListener(ev, () => { lastInput = Date.now(); }, { passive: true, capture: true });
}

export class ApiError extends Error {
  constructor(status, detail, body) {
    super(typeof detail === 'string' ? detail : `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

async function request(method, path, { body, background = false, signal } = {}) {
  const headers = { Accept: 'application/json' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (method !== 'GET' && csrf) headers['X-CSRF-Token'] = csrf;
  if (background && lastInput <= lastForeground) headers['X-Background'] = '1';
  else lastForeground = Date.now();
  let res;
  try {
    res = await fetch(path, {
      method, headers, credentials: 'same-origin', cache: 'no-store', signal,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    if (err?.name === 'AbortError') throw err;
    emit('net:down', { path });
    throw new ApiError(0, 'Network error - is the dashboard server running?');
  }
  emit('net:up', {});
  let data = null;
  const ctype = res.headers.get('content-type') || '';
  if (ctype.includes('application/json')) {
    try { data = await res.json(); } catch { data = null; }
  }
  if (res.status === 401 && path !== '/api/login') {
    emit('auth:required', data || {});
    throw new ApiError(401, data?.detail || 'Please log in', data);
  }
  if (!res.ok) {
    const detail = data?.detail;
    const msg = typeof detail === 'string' ? detail : (detail ? JSON.stringify(detail) : `HTTP ${res.status}`);
    throw new ApiError(res.status, msg, data);
  }
  return data;
}

export const api = {
  get: (path, opts) => request('GET', path, opts),
  post: (path, body = {}, opts) => request('POST', path, { ...opts, body }),
};

export function qs(params) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params || {})) {
    if (v === null || v === undefined || v === '' || v === false) continue;
    p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}
