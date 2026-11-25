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
    timeScale: { borderColor: c.border, timeVisible: intraday, secondsVisible: false, rightOffset: 4, barSpacing: 7 },
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

/** Candlestick chart with volume, EMA9/EMA21/VWAP overlays, markers and price lines. */
export class CandleChart {
  constructor(container, { intraday = true } = {}) {
    this.container = container;
    this.intraday = intraday;
    this.toggles = { ema9: true, ema21: true, vwap: true, volume: true };
    this.priceLines = [];
    this.payload = null;
    this.legend = h('div', { class: 'chart-legend' });
    this.msg = h('div', { class: 'chart-msg' });
    container.append(this.legend, this.msg);
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
    if (this.payload) this.setData(this.payload, { keepRange: false });
  }

  _legend(param) {
    const bar = param && param.time !== undefined ? param.seriesData.get(this.candles) : null;
    const last = this.payload?.candles?.[this.payload.candles.length - 1];
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
  }

  setToggles(t) {
    Object.assign(this.toggles, t);
    if (!this.chart) return;
    this.ema9.applyOptions({ visible: !!this.toggles.ema9 });
    this.ema21.applyOptions({ visible: !!this.toggles.ema21 });
    this.vwap.applyOptions({ visible: !!this.toggles.vwap });
    this.volume.applyOptions({ visible: !!this.toggles.volume });
  }

  /** payload = /api/candles response; extraLines = [{price, kind, title}] drawn on top. */
  setData(payload, { keepRange = true, extraLines = null, markerFilter = null } = {}) {
    this.payload = payload;
    if (!this.chart) return;
    const c = themeColors();
    const candles = payload.candles || [];
    const prevLen = this._len || 0;
    this.candles.setData(candles);
    this.volume.setData(candles.map((b, i) => ({ time: b.time, value: b.volume,
      color: (i > 0 ? b.close >= candles[i - 1].close : b.close >= b.open) ? c.volUp : c.volDown })));
    this.ema9.setData(payload.ema9 || []);
    this.ema21.setData(payload.ema21 || []);
    this.vwap.setData(payload.vwap || []);
    this.setToggles({ vwap: this.toggles.vwap && !!payload.vwap });
    const colors = MARKER_COLOR(c);
    let markers = (payload.markers || []);
    if (markerFilter) markers = markers.filter(markerFilter);
    this.candles.setMarkers(markers.map((m) => ({ time: m.time, position: m.position, shape: m.shape,
      color: colors[m.color] || c.accent, text: m.text, size: 1 })));
    this.setLines([...(payload.lines || []), ...(extraLines || [])]);
    this._len = candles.length;
    this.setMessage(candles.length ? '' : 'No candles for this symbol / timeframe yet.');
    if (!keepRange || prevLen === 0) this.chart.timeScale().fitContent();
    this._legend(null);
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
    if (this.payload) this.setData(this.payload, { keepRange: true });
  }

  destroy() {
    this.destroyed = true;
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
