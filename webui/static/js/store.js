// Shared app state: the logged-in user and the latest /api/overview snapshot.
import { api, emit } from './api.js';
import { debounce } from './dom.js';
import { setServerTime } from './format.js';

export const store = { me: null, overview: null, pending: 0 };

let inflight = null;
export function loadOverview({ background = true } = {}) {
  if (inflight) return inflight;
  inflight = api.get('/api/overview', { background })
    .then((o) => {
      store.overview = o;
      store.pending = o.approvals_pending || 0;
      if (o.server_time) setServerTime(o.server_time);
      emit('overview', o);
      return o;
    })
    .finally(() => { inflight = null; });
  return inflight;
}

const soon = debounce(() => loadOverview().catch(() => {}), 1200);
/** Coalesce bursts of SSE events into one overview refresh. */
export function refreshOverviewSoon() { soon(); }
