// Indian formatting helpers: ₹ with en-IN grouping, signed values (+/−, never colour alone), IST times.

const MINUS = '−';
const nf = new Map();
function fmt(dp) {
  if (!nf.has(dp)) nf.set(dp, new Intl.NumberFormat('en-IN', { minimumFractionDigits: dp, maximumFractionDigits: dp }));
  return nf.get(dp);
}

export const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

export function num(v, dp = 2) {
  return isNum(v) ? fmt(dp).format(v).replace('-', MINUS) : '–';
}

export function signed(v, dp = 2) {
  if (!isNum(v)) return '–';
  if (Math.abs(v) < 0.5 * 10 ** -dp) return fmt(dp).format(0);
  return (v > 0 ? '+' : MINUS) + fmt(dp).format(Math.abs(v));
}

/** ₹1,23,456.78 ; with sign: +₹12.00 / −₹4.50 */
export function inr(v, { sign = false, dp = 2 } = {}) {
  if (!isNum(v)) return '–';
  const abs = fmt(dp).format(Math.abs(v));
  if (sign) {
    if (Math.abs(v) < 0.5 * 10 ** -dp) return '₹' + fmt(dp).format(0);
    return (v > 0 ? '+' : MINUS) + '₹' + abs;
  }
  return (v < 0 ? MINUS : '') + '₹' + abs;
}

export function price(v) { return num(v, 2); }

export function pct(v, { sign = false, dp = 2 } = {}) {
  if (!isNum(v)) return '–';
  return (sign ? signed(v, dp) : num(v, dp)) + '%';
}

export function rmult(v) { return isNum(v) ? signed(v, 2) + 'R' : '–'; }

export function tone(v) {
  if (!isNum(v) || Math.abs(v) < 0.005) return 'flat';
  return v > 0 ? 'pos' : 'neg';
}

export function kpiTone(v) {
  if (!isNum(v) || Math.abs(v) < 0.005) return '';
  return v > 0 ? 'good' : 'bad';
}

// ------------------------------------------------------------------ time (always IST)
let serverOffsetMs = 0;
export function setServerTime(iso) {
  const t = Date.parse(iso);
  if (Number.isFinite(t)) serverOffsetMs = t - Date.now();
}
export function nowMs() { return Date.now() + serverOffsetMs; }

const dtf = (opts) => new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', ...opts });
const F_TIME = dtf({ hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
const F_HM = dtf({ hour: '2-digit', minute: '2-digit', hour12: false });
const F_DATE = dtf({ day: '2-digit', month: 'short', year: 'numeric' });
const F_DSHORT = dtf({ day: '2-digit', month: 'short' });
const F_WD = dtf({ weekday: 'short', day: '2-digit', month: 'short' });
const F_ISO = dtf({ year: 'numeric', month: '2-digit', day: '2-digit' });

function toDate(v) {
  if (v === null || v === undefined || v === '') return null;
  const d = v instanceof Date ? v : new Date(typeof v === 'number' ? v : Date.parse(v));
  return Number.isFinite(d.getTime()) ? d : null;
}

export function time(v) { const d = toDate(v); return d ? F_TIME.format(d) : '–'; }
export function hm(v) { const d = toDate(v); return d ? F_HM.format(d) : '–'; }
export function date(v) { const d = toDate(v); return d ? F_DATE.format(d) : '–'; }
export function dshort(v) { const d = toDate(v); return d ? F_DSHORT.format(d) : '–'; }
export function weekday(v) { const d = toDate(v); return d ? F_WD.format(d) : '–'; }
export function dateTime(v) { const d = toDate(v); return d ? `${F_DSHORT.format(d)} ${F_HM.format(d)}` : '–'; }

/** Smart stamp: time only for today (IST), else date + time. */
export function stamp(v) {
  const d = toDate(v);
  if (!d) return '–';
  return isoDay(d) === todayIST() ? F_HM.format(d) : `${F_DSHORT.format(d)} ${F_HM.format(d)}`;
}

/** YYYY-MM-DD of a Date in IST. */
export function isoDay(d) {
  const parts = F_ISO.formatToParts(d);
  const get = (t) => parts.find((p) => p.type === t)?.value;
  return `${get('year')}-${get('month')}-${get('day')}`;
}
export function todayIST() { return isoDay(new Date(nowMs())); }
export function shiftDay(day, n) {
  const d = new Date(day + 'T12:00:00+05:30');
  d.setUTCDate(d.getUTCDate() + n);
  return isoDay(d);
}

export function ago(seconds) {
  if (!isNum(seconds)) return 'never';
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
  return `${Math.floor(s / 86400)}d`;
}

export function duration(minutes) {
  if (!isNum(minutes)) return '–';
  const m = Math.round(minutes);
  if (m < 60) return `${m}m`;
  if (m < 60 * 24) return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`;
  const d = Math.floor(m / 1440);
  return `${d}d ${Math.floor((m % 1440) / 60)}h`;
}

export function countdown(seconds) {
  if (!isNum(seconds)) return '–';
  const s = Math.max(0, Math.floor(seconds));
  if (s <= 0) return 'EXPIRED';
  if (s < 3600) return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  const hrs = Math.floor(s / 3600);
  if (hrs >= 24) return `${Math.floor(hrs / 24)}d ${hrs % 24}h`;
  return `${hrs}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
}

export function istClock() { return F_TIME.format(new Date(nowMs())); }

export function title(s) {
  if (!s) return '';
  return String(s).toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
