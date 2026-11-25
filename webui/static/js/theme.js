// Registry of live chart instances so a theme switch can recolour them without a reload.
const charts = new Set();

export function registerChart(chart) {
  charts.add(chart);
  return () => charts.delete(chart);
}

export function applyThemeToCharts() {
  // wait one frame so the new CSS variables are in effect
  requestAnimationFrame(() => {
    for (const c of charts) {
      try { c.applyTheme(); } catch { /* chart may be destroyed */ }
    }
  });
}
