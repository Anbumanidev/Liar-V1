// Lightweight Charts (vendored v4.2.3, Apache-2.0) wrappers. Chart times from the API are IST
// wall-clock epochs (UTC epoch + 19800 s), so the default UTC axis reads IST.
import { h } from './dom.js';
import * as f from './format.js';

let libPromise = null;
export function ensureCharts() {
  if (window.LightweightCharts) return Promise.resolve(window.LightweightCharts);
  if (!libPromise) {
    libPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = '/static/vendor/lightweight-charts.standalone.production.js';
      s.async = true;
      s.onload = () => resolve(window.LightweightCharts);
      s.onerror = () => { libPromise = null; reject(new Error('chart library failed to load')); };
      document.head.appendChild(s);
    });
  }
  return libPromise;
}

export function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return {
    bg: v('--chart-bg'), grid: v('--chart-grid'), text: v('--chart-text'), border: v('--border'),
    up: v('--chart-up'), down: v('--chart-down'), volUp: v('--chart-vol-up'), volDown: v('--chart-vol-down'),
    ema9: v('--chart-ema9'), ema21: v('--chart-ema21'), vwap: v('--chart-vwap'),
    green: v('--green'), red: v('--red'), amber: v('--amber'), accent: v('--accent'), text2: v('--text-2'),
    violet: v('--violet'),
  };
}

function baseOptions(c, intraday = true) {
  return {
    autoSize: true,
    layout: { background: { type: 'solid', color: c.bg }, textColor: c.text, fontSize: 11,
      fontFamily: getComputedStyle(document.body).fontFamily, attributionLogo: false },
    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
    rightPriceScale: { borderColor: c.border, scaleMargins: { top: 0.08, bottom: 0.22 } },
    timeScale: { borderColor: c.border, timeVisible: intraday, secondsVisible: false, rightOffset: 4, barSpacing: 7,
      minBarSpacing: 0.1 },
    crosshair: { mode: 0 },
    localization: { priceFormatter: (p) => f.num(p, 2) },
  };
}

const MARKER_COLOR = (c) => ({ buy: c.green, sell: c.red, exit_win: c.green, exit_loss: c.red });
const LINE_STYLE = (c) => ({
  entry: { color: c.accent, style: 0, w: 1 }, sl: { color: c.red, style: 2, w: 1 }, sl0: { color: c.red, style: 3, w: 1 },
  t1: { color: c.green, style: 2, w: 1 }, t2: { color: c.green, style: 3, w: 1 }, exit: { color: c.amber, style: 2, w: 1 },
  ltp: { color: c.text2, style: 1, w: 1 },
});

// ------------------------------------------------------------------ history limits (mirror webui/candles.py)
export const TF_LABEL = { '1m': '1-minute', '5m': '5-minute', '10m': '10-minute', '15m': '15-minute', '20m': '20-minute',
  '30m': '30-minute', '1h': '1-hour', '5h': '5-hour', '1d': 'Daily', '1wk': 'Weekly' };
/** Calendar days of history the provider keeps per timeframe (Yahoo; resampled timeframes inherit their source). */
export const TF_HISTORY_DAYS = { '1m': 7, '5m': 59, '10m': 59, '15m': 59, '20m': 59, '30m': 59, '1h': 729, '5h': 729,
  '1d': 4383, '1wk': 10957 };
const TF_LIMIT_TEXT = { '1m': '~7 days', '5m': '~59 days', '10m': '~59 days', '15m': '~59 days', '20m': '~59 days',
  '30m': '~59 days', '1h': '~2 years', '5h': '~2 years', '1d': '~12 years', '1wk': 'the full listed history' };
export const RANGES = [['1D', 1], ['5D', 7], ['1M', 30], ['3M', 91], ['6M', 182], ['1Y', 365], ['5Y', 1826], ['MAX', 0]];
/** Range loaded when a symbol / timeframe is opened, and how much of it is shown (sessions or days). */
export const DEFAULT_LOAD = { '1m': '5D', '5m': '5D', '10m': '5D', '15m': '5D', '20m': '5D', '30m': '5D', '1h': '1Y', '5h': '1Y',
  '1d': '5Y', '1wk': 'MAX' };
export const DEFAULT_VIEW = { '1m': 'session', '5m': 'session', '10m': 'session', '15m': 'session', '20m': 'session',
  '30m': 'session', '1h': 91, '5h': 182, '1d': 365, '1wk': 1826 };
/** Bars fetched per scroll-back page. */
export const OLDER_BARS = { '1m': 1500, '5m': 750, '10m': 380, '15m': 250, '20m': 190, '30m': 130, '1h': 420, '5h': 240,
  '1d': 500, '1wk': 260 };

/** Is a range button usable for a timeframe? -> {ok, why} (why = tooltip). */
export function rangeState(tf, key) {
  const label = TF_LABEL[tf] || tf;
  const days = (RANGES.find(([k]) => k === key) || [key, 0])[1];
  if (key === 'MAX') return { ok: true, why: `All available ${label.toLowerCase()} history (${TF_LIMIT_TEXT[tf] || 'max'})` };
  if ((tf === '1d' && key === '1D') || (tf === '1wk' && ['1D', '5D', '1M'].includes(key))) {
    return { ok: false, why: `Too short for ${label.toLowerCase()} candles - pick a shorter timeframe` };
  }
  if (days > (TF_HISTORY_DAYS[tf] ?? Infinity) + 1) {
    const src = ['10m', '20m', '30m'].includes(tf) ? ' (built from 5-minute candles)' : tf === '5h' ? ' (built from 1-hour candles)' : '';
    return { ok: false, why: `${label} data only goes back ${TF_LIMIT_TEXT[tf]}${src}` };
  }
  return { ok: true, why: `Show the last ${key === '1D' ? 'session' : key === '5D' ? '5 sessions' : key}` };
}

/** Row of range buttons. onPick(key); `extra` = [{key, label, title}] shown first (e.g. "Trade"). el.set(tf, active). */
export function rangeBar(onPick, { extra = [] } = {}) {
  const root = h('div', { class: 'seg sm range-bar', role: 'group', 'aria-label': 'Date range' });
  root.set = (tf, active) => {
    const items = [...extra.map((x) => ({ key: x.key, label: x.label, ok: true, why: x.title || null })),
      ...RANGES.map(([k]) => ({ key: k, label: k, ...rangeState(tf, k) }))];
    root.replaceChildren(...items.map((it) => h('span', { class: 'rb', title: it.why },
      h('button', { type: 'button', class: [it.key === active && 'on'], disabled: !it.ok, 'aria-pressed': String(it.key === active),
        onclick: () => onPick(it.key) }, it.label))));
  };
  return root;
}

// ------------------------------------------------------------------ indicators (computed on the merged bars)
const round2 = (v) => Math.round(v * 100) / 100;

export function emaSeries(bars, n) {
  const out = [];
  const k = 2 / (n + 1);
  let e = null;
  for (let i = 0; i < bars.length; i++) {
    const c = bars[i].close;
    e = e === null ? c : c * k + e * (1 - k);
    if (i >= n - 1) out.push({ time: bars[i].time, value: round2(e) });
  }
  return out;
}

/** Session VWAP (reset each IST day; chart times are IST wall-clock epochs). */
export function vwapSeries(bars) {
  const out = [];
  let day = null; let pv = 0; let vol = 0;
  for (const b of bars) {
    const d = Math.floor(b.time / 86400);
    if (d !== day) { day = d; pv = 0; vol = 0; }
    const v = Number.isFinite(b.volume) ? b.volume : 0;
    pv += ((b.high + b.low + b.close) / 3) * v;
    vol += v;
    if (vol > 0) out.push({ time: b.time, value: round2(pv / vol) });
  }
  return out;
}

const markerKey = (m) => `${m.time}|${m.kind}|${m.text}`;

/** Candlestick chart with volume, EMA9/EMA21/VWAP overlays, markers, price lines and lazy scroll-back history.
 *
 * setData() replaces everything; prepend() merges an older page (the view does not jump); mergeLatest() merges a
 * refresh of the newest bars. Indicators are recomputed on the merged bars so they stay continuous. */
export class CandleChart {
  constructor(container, { intraday = true } = {}) {
    this.container = container;
    this.intraday = intraday;
    this.toggles = { ema9: true, ema21: true, vwap: true, volume: true };
    this.priceLines = [];
    this.payload = null;
    this.bars = [];
    this.markers = [];
    this.lines = [];
    this.extraLines = [];
    this.markerFilter = null;
    this.hasMore = false;
    this.historyNote = '';
    this.gen = 0;
    this.loadingOlder = false;
    this.fetchOlder = null;
    this.olderBlockedUntil = 0;
    this.legend = h('div', { class: 'chart-legend' });
    this.msg = h('div', { class: 'chart-msg' });
    this.edge = h('div', { class: 'chart-edge', hidden: true, role: 'status' });
    container.append(this.legend, this.msg, this.edge);
    this.ready = ensureCharts().then((LC) => this._init(LC));
  }

  _init(LC) {
    if (this.destroyed) return;
    const c = themeColors();
    this.LC = LC;
    this.chart = LC.createChart(this.container, baseOptions(c, this.intraday));
    this.candles = this.chart.addCandlestickSeries({ upColor: c.up, downColor: c.down, borderUpColor: c.up,
      borderDownColor: c.down, wickUpColor: c.up, wickDownColor: c.down, priceLineVisible: true });
    this.volume = this.chart.addHistogramSeries({ priceScaleId: 'vol', priceFormat: { type: 'volume' },
      lastValueVisible: false, priceLineVisible: false });
    this.chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    const line = (color) => this.chart.addLineSeries({ color, lineWidth: 1.5, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false });
    this.ema9 = line(c.ema9);
    this.ema21 = line(c.ema21);
    this.vwap = line(c.vwap);
    this.chart.subscribeCrosshairMove((p) => this._legend(p));
    this.chart.timeScale().subscribeVisibleLogicalRangeChange((r) => this._onRange(r));
    this._applyVisibility();
    if (this.payload) this._render({ fit: true });
  }

  _legend(param) {
    const bar = param && param.time !== undefined ? param.seriesData.get(this.candles) : null;
    const last = this.bars[this.bars.length - 1];
    const b = bar || last;
    this.legend.replaceChildren();
    if (!b) return;
    const chg = b.close - b.open;
    const parts = [['O', f.num(b.open)], ['H', f.num(b.high)], ['L', f.num(b.low)], ['C', f.num(b.close)]];
    for (const [k, v] of parts) this.legend.append(h('span', null, `${k} `, h('b', null, v)));
    this.legend.append(h('span', { class: f.tone(chg) }, f.signed(chg)));
    const vol = (bar ? param.seriesData.get(this.volume)?.value : last?.volume);
    if (f.isNum(vol)) this.legend.append(h('span', null, 'Vol ', h('b', null, f.num(vol, 0))));
  }

  setMessage(text) {
    this.msg.textContent = text || '';
    this.msg.hidden = !text;
  }

  setIntraday(intraday) {
    this.intraday = intraday;
    this.chart?.applyOptions({ timeScale: { timeVisible: intraday } });
    this._applyVisibility();
  }

  setToggles(t) {
    Object.assign(this.toggles, t);
    this._applyVisibility();
  }

  _applyVisibility() {
    if (!this.chart) return;
    this.ema9.applyOptions({ visible: !!this.toggles.ema9 });
    this.ema21.applyOptions({ visible: !!this.toggles.ema21 });
    this.vwap.applyOptions({ visible: !!this.toggles.vwap && this.intraday });
    this.volume.applyOptions({ visible: !!this.toggles.volume });
  }

  /** Replace everything. payload = /api/candles response; extraLines = [{price, kind, title}] drawn on top.
   * fit: true/false overrides the default (fit when !keepRange or nothing was shown before). */
  setData(payload, { keepRange = true, extraLines = null, markerFilter = null, fit = null } = {}) {
    const prevLen = this.bars.length;
    this.gen += 1;
    this.loadingOlder = false;
    this.olderBlockedUntil = 0;
    this.settleUntil = Date.now() + 700;   // the caller positions the view now; no scroll-back for that transient range
    this.payload = payload;
    this.bars = (payload.candles || []).slice();
    this.markers = (payload.markers || []).slice();
    this.lines = payload.lines || [];
    this.extraLines = extraLines || [];
    this.markerFilter = markerFilter;
    this.hasMore = !!payload.has_more;
    this.historyNote = payload.history_note || '';
    if (payload.intraday !== undefined) this.intraday = !!payload.intraday;
    this._render({ fit: fit === null ? (!keepRange || prevLen === 0) : !!fit });
  }

  /** Merge an older page (bars before the first loaded one). Returns the number of bars added. */
  prepend(payload) {
    const first = this.bars.length ? this.bars[0].time : Infinity;
    const older = (payload.candles || []).filter((b) => b.time < first);
    this.hasMore = !!payload.has_more && older.length > 0;
    if (payload.history_note) this.historyNote = payload.history_note;
    if (!older.length) { this._edgeState(); return 0; }
    const ts = this.chart?.timeScale();
    const lr = ts?.getVisibleLogicalRange();
    this.bars = older.concat(this.bars);
    const seen = new Set(this.markers.map(markerKey));
    this.markers = (payload.markers || []).filter((m) => m.time < first && !seen.has(markerKey(m))).concat(this.markers);
    this._render({ fit: false });
    if (ts && lr) ts.setVisibleLogicalRange({ from: lr.from + older.length, to: lr.to + older.length });
    return older.length;
  }

  /** Merge a refresh of the newest bars (replaces bars/markers from its first bar on; keeps older history). */
  mergeLatest(payload, { extraLines = null } = {}) {
    const fresh = payload.candles || [];
    if (extraLines) this.extraLines = extraLines;
    this.lines = payload.lines || this.lines;
    this.payload = { ...payload, has_more: this.hasMore };
    if (fresh.length) {
      const cut = fresh[0].time;
      if (!this.bars.length || cut <= this.bars[0].time) {
        this.bars = fresh.slice();
        this.markers = (payload.markers || []).slice();
      } else {
        this.bars = this.bars.filter((b) => b.time < cut).concat(fresh);
        this.markers = this.markers.filter((m) => m.time < cut).concat(payload.markers || []);
      }
    }
    this._render({ fit: false });
  }

  _render({ fit = false } = {}) {
    if (!this.chart) return;
    const c = themeColors();
    const bars = this.bars;
    this.candles.setData(bars);
    this.volume.setData(bars.map((b, i) => ({ time: b.time, value: b.volume,
      color: (i > 0 ? b.close >= bars[i - 1].close : b.close >= b.open) ? c.volUp : c.volDown })));
    this.ema9.setData(emaSeries(bars, 9));
    this.ema21.setData(emaSeries(bars, 21));
    this.vwap.setData(this.intraday ? vwapSeries(bars) : []);
    this._applyVisibility();
    const colors = MARKER_COLOR(c);
    const times = new Set(bars.map((b) => b.time));
    let markers = this.markers.filter((m) => times.has(m.time));
    if (this.markerFilter) markers = markers.filter(this.markerFilter);
    markers.sort((a, b) => a.time - b.time);
    this.candles.setMarkers(markers.map((m) => ({ time: m.time, position: m.position, shape: m.shape,
      color: colors[m.color] || c.accent, text: m.text, size: 1 })));
    this.setLines([...(this.lines || []), ...(this.extraLines || [])]);
    this.setMessage(bars.length ? '' : 'No candles for this symbol / timeframe yet.');
    if (fit) this.chart.timeScale().fitContent();
    this._legend(null);
    this._edgeState();
  }

  /** Fit everything loaded. */
  fit() { this.chart?.timeScale().fitContent(); }

  /** Show the newest part of the data: 'session' = the last session (two if it has < 20 bars) or N days. */
  showRecent(span) {
    const n = this.bars.length;
    if (!this.chart || !n) return;
    let from = 0;
    const dayOf = (i) => Math.floor(this.bars[i].time / 86400);
    if (span === 'session') {
      let i = n - 1;
      const back = (d) => { while (i > 0 && dayOf(i - 1) === d) i -= 1; };
      back(dayOf(n - 1));
      if (n - i < 20 && i > 0) { i -= 1; back(dayOf(i)); }
      from = i;
    } else if (f.isNum(span)) {
      const cutoff = this.bars[n - 1].time - span * 86400;
      from = Math.max(0, this.bars.findIndex((b) => b.time >= cutoff));
    }
    this.chart.timeScale().setVisibleLogicalRange({ from: from - 0.5, to: n - 1 + 4 });
  }

  /** Lazy scroll-back: fetchOlder(beforeChartTime) -> Promise</api/candles page>. */
  enableHistory(fetchOlder) { this.fetchOlder = fetchOlder; }

  /** A new symbol / timeframe / range is being loaded: drop any in-flight older page and stop paging. */
  resetHistory() { this.gen += 1; this.loadingOlder = false; this.hasMore = false; this._edgeState(); }

  _onRange(r) {
    this._edgeState(r);
    if (!r || !this.bars.length || !this.fetchOlder || !this.hasMore || this.loadingOlder) return;
    if (r.from > 12 || Date.now() < this.olderBlockedUntil || Date.now() < (this.settleUntil || 0)) return;
    this._loadOlder();
  }

  async _loadOlder() {
    const gen = this.gen;
    this.loadingOlder = true;
    this._edgeState();
    try {
      const d = await this.fetchOlder(this.bars[0].time);
      if (gen !== this.gen || this.destroyed) return;
      this.loadingOlder = false;
      this.prepend(d);
    } catch (err) {
      if (gen !== this.gen || this.destroyed) return;
      this.loadingOlder = false;
      this.olderBlockedUntil = Date.now() + 10000;
      this.edge.textContent = `Could not load older candles: ${err?.message || err}`;
      this.edge.className = 'chart-edge warn';
      this.edge.hidden = false;
    } finally {
      if (gen === this.gen) this.loadingOlder = false;
    }
  }

  _edgeState(range) {
    if (this.destroyed) return;
    const r = range || this.chart?.timeScale().getVisibleLogicalRange();
    if (this.loadingOlder) {
      this.edge.textContent = 'Loading older candles…';
      this.edge.className = 'chart-edge';
      this.edge.hidden = false;
      return;
    }
    if (this.edge.classList.contains('warn') && Date.now() < this.olderBlockedUntil) return;
    const atStart = r && this.bars.length && r.from <= 0.5 && !this.hasMore;
    if (atStart) {
      this.edge.textContent = `Start of available history${this.historyNote ? ` (${this.historyNote})` : ''}`;
      this.edge.className = 'chart-edge info';
      this.edge.hidden = false;
    } else {
      this.edge.hidden = true;
    }
  }

  setLines(lines) {
    if (!this.candles) return;
    for (const pl of this.priceLines) this.candles.removePriceLine(pl);
    this.priceLines = [];
    const styles = LINE_STYLE(themeColors());
    for (const l of lines || []) {
      if (!f.isNum(l.price)) continue;
      const st = styles[l.kind] || styles.entry;
      this.priceLines.push(this.candles.createPriceLine({ price: l.price, color: st.color, lineWidth: st.w,
        lineStyle: st.style, axisLabelVisible: true, title: l.title || l.kind }));
    }
  }

  applyTheme() {
    if (!this.chart) return;
    const c = themeColors();
    this.chart.applyOptions(baseOptions(c, this.intraday));
    this.candles.applyOptions({ upColor: c.up, downColor: c.down, borderUpColor: c.up, borderDownColor: c.down,
      wickUpColor: c.up, wickDownColor: c.down });
    this.ema9.applyOptions({ color: c.ema9 });
    this.ema21.applyOptions({ color: c.ema21 });
    this.vwap.applyOptions({ color: c.vwap });
    if (this.payload) this._render({ fit: false });
  }

  destroy() {
    this.destroyed = true;
    this.gen += 1;
    if (this.chart) this.chart.remove();
    this.chart = null;
  }
}

/** Simple area/line chart (equity curves). points = [{time, equity|value}] */
export class LineChart {
  constructor(container, { color = null, baseline = null } = {}) {
    this.container = container;
    this.colorVar = color;
    this.baseline = baseline;
    this.msg = h('div', { class: 'chart-msg' });
    container.append(this.msg);
    this.ready = ensureCharts().then((LC) => {
      if (this.destroyed) return;
      const c = themeColors();
      this.chart = LC.createChart(container, { ...baseOptions(c, true),
        rightPriceScale: { borderColor: c.border, scaleMargins: { top: 0.12, bottom: 0.08 } } });
      this.series = this.chart.addAreaSeries({ lineColor: c.accent, topColor: c.accent + '55', bottomColor: c.accent + '05',
        lineWidth: 2, priceLineVisible: false });
      if (this.points) this.setData(this.points);
    });
  }

  setData(points) {
    this.points = points;
    if (!this.chart) return;
    const data = (points || []).filter((p) => f.isNum(p.equity ?? p.value)).map((p) => ({ time: p.time, value: p.equity ?? p.value }));
    this.series.setData(data);
    if (this._base) { this.series.removePriceLine(this._base); this._base = null; }
    if (f.isNum(this.baseline)) {
      const c = themeColors();
      this._base = this.series.createPriceLine({ price: this.baseline, color: c.text2, lineStyle: 2, lineWidth: 1,
        axisLabelVisible: true, title: 'start' });
    }
    this.msg.textContent = data.length ? '' : 'No equity points yet.';
    this.chart.timeScale().fitContent();
  }

  applyTheme() {
    if (!this.chart) return;
    const c = themeColors();
    this.chart.applyOptions(baseOptions(c, true));
    this.series.applyOptions({ lineColor: c.accent, topColor: c.accent + '55', bottomColor: c.accent + '05' });
    if (this.points) this.setData(this.points);
  }

  destroy() {
    this.destroyed = true;
    if (this.chart) this.chart.remove();
    this.chart = null;
  }
}
