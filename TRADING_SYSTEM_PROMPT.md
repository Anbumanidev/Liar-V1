# PROJECT: Multi-Agent, Multi-Mode Paper Trading System for Indian Equities (NSE)

You are building a production-quality, end-to-end, multi-agent PAPER TRADING system for the Indian stock market (NSE cash equity). It supports multiple trading modes (Intraday, BTST, Swing, Positional), and I control everything from an owner-only dashboard, including which modes run, which trades are taken, and every setting. Starting capital is ₹1,000. This system must NEVER place real orders. Build it incrementally, phase by phase (section 17), with tests, and confirm each phase works before moving on.

---

## 0. NON-NEGOTIABLE RULES
1. PAPER TRADING ONLY. No real order placement code anywhere. Broker APIs may be used ONLY for market data. Add a startup assertion and a code-level guard that raises an error if any order-placement endpoint is ever called.
2. OPEN-SOURCE ONLY. All LLMs are open-weight models run locally (Ollama preferred, vLLM optional). All libraries open source. No paid LLM APIs.
3. LLMs NEVER DO MATH. Indicators, patterns, levels, position sizes, charges, capital movements and P&L are computed by deterministic Python. LLMs only interpret, reason and argue.
4. ALL LLM OUTPUTS ARE STRUCTURED: JSON validated by Pydantic. On failure retry up to 2 times, then mark the agent "FAILED" and treat its vote as neutral. One agent failing never crashes the loop.
5. A DETERMINISTIC RISK MANAGER HAS FINAL VETO over the meta agent AND over my manual actions (section 5). CAPITAL PROTECTION (section 8) overrides everything.
6. I AM IN CONTROL. Every setting, mode, strategy, agent, model, risk limit, schedule and trade is controllable from the dashboard (section 14). Only the hard caps in config/hard_caps.yaml (section 14.2) cannot be changed from the dashboard.
7. EVERYTHING IS LOGGED: inputs, agent outputs, debates, decisions, rejected stocks and reasons, my approvals/rejections/edits, capital movements, every settings change.
8. All times IST (Asia/Kolkata). Respect NSE trading hours and holidays.
9. Secrets (email password, broker API keys, dashboard credentials) in .env / secrets files, never hard-coded, git-ignored.
10. No averaging down. A stop-loss may only be moved closer to price (tighter), never further away, by the system OR by me. Hard-coded guards.

---

## 1. TECH STACK
- Python 3.11+
- Orchestration: LangGraph (state graph, parallel sub-agents, checkpoints)
- LLM runtime: Ollama. Model names in config/models.yaml, swappable from the dashboard.
- Live/intraday data: free broker data APIs, preference order: Angel One SmartAPI or Upstox API. Abstract, pluggable data-provider interface.
- Historical/fallback data: yfinance (".NS"), jugaad-data / nsepython (bhavcopy, indices, India VIX, FII/DII, ASM/GSM lists, corporate announcements, results calendar, corporate actions).
- Indicators: pandas-ta; candlestick patterns: TA-Lib (CDL* functions).
- Sentiment: FinBERT (ProsusAI/finbert) on headlines from Indian financial news RSS feeds + NSE announcements.
- Fundamentals: yfinance + NSE filings, cached daily.
- Storage: SQLite via SQLAlchemy (switchable to PostgreSQL). Raw JSONL logs in /logs/YYYY-MM-DD/.
- Scheduler: APScheduler. Email: smtplib + Gmail App Password, Jinja2 HTML templates.
- Dashboard: Streamlit + Plotly, streamlit-authenticator + bcrypt, streamlit-autorefresh for live pages.
- Backtest/replay: vectorbt or backtrader + custom REPLAY mode.
- pytest, YAML + pydantic-settings, loguru.

---

## 2. MODEL SELECTION (config/models.yaml)
Check the current Ollama model library and pick the strongest open-weight models that fit my hardware (ask me for GPU/RAM). By task:
- Reasoning roles (Bull, Bear, Meta): strongest open-weight reasoning model that fits (latest Qwen, DeepSeek, gpt-oss or Llama class).
- Sub-agents: smaller, fast instruct model with reliable JSON output.
- Sentiment: FinBERT.
- Provide "low-resource" and "high-resource" profiles. Log latency per agent call; warn if one stock's full cycle exceeds 60 s.
- The dashboard lets me assign any installed Ollama model to any agent.

---

## 3. TRADING MODES
Each mode is a pluggable configuration: timeframes, strategies, agent weights, schedule, charges model, exit rules.

| Mode | Holding | Product | Direction | Analysis timeframes |
|---|---|---|---|---|
| INTRADAY | same day, square-off 15:15 | MIS | long + short | 5m, 15m, 1h |
| BTST | buy today 14:45–15:20, sell next trading day | CNC | long only | 15m, 1h, daily |
| SWING | 2–10 trading days (configurable) | CNC | long only | 1h, daily |
| POSITIONAL | weeks to ~3 months (configurable) | CNC | long only | daily, weekly |

Rules:
- Overnight short positions are not possible in the Indian cash market, so BTST/SWING/POSITIONAL are long-only (hard-coded).
- Each mode can be ENABLED/DISABLED from the dashboard, per batch.
- Capital allocation per mode as % of trading capital, set from the dashboard (default: INTRADAY 50%, SWING 50%, BTST 0%, POSITIONAL 0%). Allocations must sum to ≤ 100%.
- Mode selection policy (dashboard setting): FIXED (each candidate is analysed only for enabled modes as configured) or AUTO (the meta agent recommends the best-fitting enabled mode for each setup, with reasons). Either way I can change the mode of a trade in the approval queue.
- Execution policy per mode (dashboard setting): AUTO (system executes after risk manager approval), MANUAL_APPROVE (trade goes to my approval queue), SIGNALS_ONLY (logged and emailed, never executed). Default: MANUAL_APPROVE for all modes.
- Approval timeout per mode (default INTRADAY 3 min, BTST 10 min, SWING/POSITIONAL until 09:45 next session). Expired = auto-rejected as "APPROVAL_TIMEOUT".
- Mode conversion of an open position from the dashboard: INTRADAY → SWING/POSITIONAL (long only, before 15:10, only if the SWING/POSITIONAL allocation and risk limits allow; charges and rules switch to the new mode). Any CNC position can be exited at any time.
- BTST note: log the T+1 settlement / short-delivery risk of selling before shares are credited; simulate it as a configurable small probability event in replay only.

---

## 4. AGENT ARCHITECTURE (mode-aware)

### Layer 1 — Specialist Sub-Agents (parallel per stock, per mode)
Deterministic feature computation + LLM interpretation. Output: {agent, symbol, mode, timestamp, signal: BULLISH|BEARISH|NEUTRAL, confidence: 0-100, key_findings: [...], data_used: {...}, reasoning}. Each agent uses the timeframes of the mode being analysed. Agent weights per mode are configurable.

1. Fundamental Agent: INTRADAY/BTST = quality filter only (ASM/GSM, pledge, results today, corporate actions, debt). SWING/POSITIONAL = scored signal (growth, profitability, valuation vs sector, debt, promoter holding trend). Always outputs TRADEABLE: true/false.
2. Technical Analysis Agent: EMA 9/21/50/200, VWAP (intraday), RSI, MACD, Supertrend, ADX, Bollinger, ATR, pivots/CPR, support/resistance, multi-timeframe.
3. Candlestick & Price Action Agent: TA-Lib patterns, opening range, gaps, inside bars, trend structure, daily/weekly patterns for longer modes.
4. Volume & Liquidity Agent: relative volume, spikes, VWAP deviation, delivery % trend, spread/liquidity.
5. Market Context Agent: NIFTY 50 / BANK NIFTY trend, India VIX, sector strength, advance/decline, FII/DII flows, global cues.
6. News & Sentiment Agent: FinBERT on last 24h (intraday) or 7 days (longer modes).
7. Event & Overnight Risk Agent (BTST/SWING/POSITIONAL): results dates, ex-dates, board meetings, index rebalancing, scheduled macro events (RBI policy, budget, US Fed), weekends/holidays inside the holding window.
8. Entry Timing Agent: entry type, price zone, or "wait".
9. Stop-Loss Agent: deterministic candidates (ATR, swing points, VWAP, opening range, daily structure); LLM picks and justifies; code validates.
10. Exit/Target Agent: targets from R:R, levels, trailing rule, max holding period. Minimum R:R 1:1.5 intraday, 1:2 for longer modes (configurable).
Registry-based; add more agents later without code changes elsewhere.

### Layer 2 — Debate Agents
Bull Agent (case FOR) and Bear Agent (case AGAINST, including charges and overnight risk), 2 rebuttal rounds (configurable from dashboard), full transcript stored. Output: {stance, top_arguments, rebuttals, conviction: 0-100}.

### Layer 3 — Meta Decision Agent
Receives sub-agent reports, strategy candidates, confluence score, regime, capital state, mode settings and debate. Outputs:
{decision: BUY | SELL_SHORT | NO_TRADE, symbol, mode, recommended_mode_reason (AUTO policy), strategy, entry_price, entry_type: MARKET|LIMIT, stop_loss, target_1, target_2, trailing_rule, max_holding_days, confidence: 0-100, time_validity, reasons_for, reasons_against_considered, why_not}
Confidence threshold per mode (default 70). Quantity is never decided by the LLM.

---

## 5. DETERMINISTIC RISK MANAGER (Python, FINAL VETO — also checks my manual trades and edits)
All values in config/risk.yaml, editable from the dashboard within hard caps.
- Risk per trade: 1% of the mode's SIZING CAPITAL (section 6), reduced by drawdown tiers and loss budget (section 8).
- Quantity = floor(min(risk amount / |entry − SL|, mode capital available for the position / entry)). Reject if < 1.
- Small-capital settings (while trading capital < ₹10,000): max 2 open positions in total, max 60% of trading capital per position, max 1 position per sector.
- Leverage: MIS_LEVERAGE = 1 (off) by default; configurable for INTRADAY only, with loud warnings. Never leverage for CNC modes.
- Max trades per day per mode; daily loss limit 3% of trading capital → no new entries that day.
- INTRADAY: no entries before 09:30 or after 14:45; force square-off at 15:15.
- Overnight (BTST/SWING/POSITIONAL): max overnight exposure 50% of trading capital; no new entry if a results/board meeting/ex-date falls inside the holding window (configurable); each position's risk for the loss budget = max(SL risk, qty × entry × GAP_BUFFER_PCT [default 3%]) to account for gap-downs.
- Reject: near circuit limits; R:R below the mode minimum; invalid SL distance (zero, < 0.3 × ATR, > 3 × ATR of the mode's timeframe); TRADEABLE=false; confluence below minimum; mode disabled; mode allocation exhausted.
- CHARGES CHECK with the mode's charges model: reject if (expected profit at target_1 − round-trip charges) ≤ 0, or round-trip charges > 1% of position value. Log "CHARGES_EXCEED_EDGE".
- My manual trades and edits go through the same checks. If I am blocked, the dashboard shows the exact rule.
Every veto logged with the exact rule.

---

## 6. CAPITAL MANAGEMENT — ₹1,000 START, COMPOUNDING WITH A PROTECTED RESERVE

### 6.1 Accounts (virtual, per batch)
- STARTING_CAPITAL = ₹1,000 (set per batch from the dashboard).
- TRADING account: available for trading, split across modes by allocation %. Capital held in open CNC positions is BLOCKED until they exit.
- RESERVE account: locked profits, starts ₹0, never traded directly.
- BASE_CAPITAL = STARTING_CAPITAL.
- TOTAL EQUITY = TRADING (cash + mark-to-market of open positions) + RESERVE. Track the high-water mark.

### 6.2 EOD processing (15:30, after intraday square-off)
Uses REALIZED net P&L of the day only (open positions are marked-to-market for equity/drawdown but never locked).
- Realized net P&L > 0: PROFIT_LOCK_PCT (default 30%) moves to RESERVE; the rest compounds in TRADING.
- Realized net P&L < 0: TRADING absorbs it. If TRADING cash + blocked capital falls below BASE_CAPITAL, refill up to BASE from RESERVE, but RESERVE never drops below RESERVE_PROTECTED_PCT (default 50%) of its peak.
- All movements logged in capital_ledger.

### 6.3 Sizing capital: grow slowly, shrink fast
- Increases only after TRADING stays ≥ 10% above current sizing capital for 3 consecutive trading days without a drawdown tier; decreases immediately when TRADING falls below it. Per-mode sizing capital = sizing capital × mode allocation %. All changes logged.

### 6.4 Guarantees (tests)
- RESERVE only grows by profit locks; only shrinks by the capped refill or a confirmed, audit-logged manual transfer.
- The engine can never treat RESERVE as available capital.
- Ledger always balances: STARTING_CAPITAL + Σ realized net P&L + Σ confirmed manual adjustments = TRADING (cash + cost basis of open positions) + RESERVE.

---

## 7. STRATEGY LIBRARY & PROFIT OPTIMIZATION

### 7.1 Regime detection (deterministic, every cycle, per timeframe)
TRENDING_UP / TRENDING_DOWN (ADX > 25, price vs EMA 21/VWAP, index aligned), RANGING (ADX < 20, Bollinger squeeze/inside CPR), HIGH_VOLATILITY (India VIX above 20-day average by configurable %, ATR expansion). Stored with every decision.

### 7.2 Strategies (pluggable; each declares valid modes, regimes, time windows, entry/SL/target rules)
INTRADAY: Opening Range Breakout; VWAP Pullback; EMA 9/21 Trend Pullback; CPR/Pivot Breakout; Gap-and-Go / Gap-Fill; Relative Strength Momentum; Mean Reversion (ranging only); Supertrend + ADX on 15m.
BTST: strong close near day high with above-average volume and delivery %, sector strength, positive market close.
SWING: daily breakout from base/consolidation with volume; pullback to 20/50 EMA in uptrend; bullish reversal at support with RSI divergence; relative strength leaders vs NIFTY.
POSITIONAL: weekly trend + fundamentals score; 52-week-high breakouts with volume; sector rotation leaders.
All strategy parameters editable from the dashboard within defined ranges.

### 7.3 Confluence scoring (deterministic)
0–100 from trend alignment across the mode's timeframes, volume, regime fit, market/sector alignment, candlestick confirmation, distance to levels, sentiment, and (longer modes) fundamentals score and event risk. Grades A+ / A / B. Minimum per mode (default 60).

### 7.4 Trade management
- Book 50% at 1R and move SL to breakeven (if qty is 1, move SL to breakeven at 1R).
- Trailing: INTRADAY = ATR trail or 9 EMA on 5m; SWING/POSITIONAL = trail on daily close (ATR or 20 EMA).
- Time stop: INTRADAY exit if not +0.5R within N minutes; SWING/POSITIONAL exit at 15:20 on the max holding day.
- Pre-event exit: CNC positions exit before results/ex-dates if configured.

### 7.5 Universe & filters
- Universe per mode (dashboard): NIFTY 50/100/200, F&O stocks, or custom list; plus my whitelist and blacklist.
- Only stocks the risk manager can size ≥ 1 share; liquidity and spread filters; log exclusions.
- Max 1 position per sector; no two positions with 20-day correlation > 0.7.
- After 3 consecutive losses in a mode, pause that mode's entries for 60 min (intraday) / 1 day (others).
- Daily profit lock (intraday): after +2% of trading capital, only A+ setups at half size; if open profit falls back 50% from the day's peak, stop intraday for the day.

### 7.6 Self-evaluation (no overfitting)
- Metrics per mode, strategy, regime, agent: win rate, average R, expectancy, profit factor, max drawdown.
- Auto-disable a strategy in the batch if expectancy is negative after 30+ trades (re-enable from dashboard).
- Optional 0.25 fractional Kelly after 50+ trades per strategy, always capped by the risk manager.
- SHADOW TRACKING: every rejected setup (by the system, risk manager, or me, including timeouts) is followed to its would-be SL/target outcome without capital. Lets me compare "my decisions vs system decisions".
- Tuning only via walk-forward replay. Log every parameter change with before/after metrics.

---

## 8. CAPITAL PROTECTION — MAX 10% LOSS (overrides everything)
- MAX_LOSS_PCT = 10, HARD_STOP_PCT = 9 (1% buffer for slippage/gaps). Hard cap: cannot be raised above 10 from the dashboard.
- Both floors always active, stricter wins:
  - Absolute: TOTAL EQUITY ≥ STARTING_CAPITAL × (1 − 9%) = ₹910 on a ₹1,000 start.
  - Trailing: TOTAL EQUITY ≥ high-water mark × (1 − 9%).
- Loss budget = distance from current equity (minus open-position risk, using gap-adjusted risk for overnight positions) to the stricter floor. New trade max risk = min(normal risk, remaining budget / 3). Block entries if a 1-share trade can't be funded.
- Drawdown tiers from the high-water mark: 0–3% normal; 3–6% size × 0.5 and A/A+ only; 6–9% size × 0.25, A+ only, max 1 open position, no new overnight positions; ≥ 9% or floor hit → HALT: exit all positions (CNC positions exit at next available price), block new trades, urgent email; resume only from the dashboard.
- Tests must prove no sequence of SL-hit trades (without slippage beyond SL/gap buffer) can breach either floor.

---

## 9. PAPER TRADING ENGINE
- Realistic fills: market orders at next tick/candle + configurable slippage (default 0.05%); limit orders only if price trades through.
- Overnight positions: next-day open price used for gap fills on SL (if the stock gaps below SL, fill at the open, not at SL).
- Monitor positions every tick/minute: SL, targets, partial exits, trailing, time stops, square-off, max holding period.
- Corporate actions on held CNC positions: adjust price/quantity for splits/bonuses, credit dividends on ex-date.
- Charges module (config/charges.yaml, "last verified" date, note telling me to verify current rates), separate models per product:
  - MIS (intraday): brokerage default min(₹20, 0.03% of order value) per executed order; STT on sell side; exchange charges; SEBI fee; stamp duty on buy; GST.
  - CNC (delivery): brokerage configurable (default ₹0, common for discount brokers); STT on both buy and sell; exchange charges; SEBI fee; higher stamp duty on buy; GST; DP charge per scrip on the sell day.
  - Brokerage models supported: flat-per-order, percentage-with-cap, zero.
- Track gross, charges, net P&L per trade, day, mode, batch.

---

## 10. SCHEDULE (IST, trading days only)
- 08:30 Pre-market: holiday check, fundamentals/news/global cues/FII-DII refresh, corporate actions and event calendar, overnight gap check on open CNC positions (pre-open prices 09:00–09:08).
- 09:00–09:15 Intraday watchlist build (log every stock and reason).
- 09:15–09:30 Observe opening range.
- 09:20–10:30 Execute approved SWING/POSITIONAL entries from yesterday's scan (limit orders); exit BTST positions per rules.
- 09:30–14:45 INTRADAY cycle every 5 min (configurable). CNC positions monitored all session.
- 14:30–15:15 BTST scan and entries (if enabled).
- 15:15 Intraday square-off. 15:20 time-stop exits for CNC positions at max holding.
- 15:30 EOD capital processing.
- 15:45 EOD report + email.
- 16:00–18:00 SWING/POSITIONAL scan on daily data → candidates to approval queue (or auto per policy) for next session.
- "Analyze now" from the dashboard runs the pipeline for any symbol and mode on demand.

---

## 11. BATCHES
- Batch = one experiment: starting capital (default ₹1,000), full settings snapshot (modes, allocations, policies, models, strategies, risk, capital rules, prompt versions), start/end, status ACTIVE | PAUSED | HALTED | CLOSED.
- Everything carries batch_id. One ACTIVE batch at a time.
- New batch: exits open positions (or, my choice, lets CNC positions run to completion under the old batch in "closing" state), archives the old batch, starts fresh (TRADING = starting capital, RESERVE = 0). History never deleted by a reset.

---

## 12. STORAGE (SQLAlchemy)
batches, settings_versions (full config snapshot per change), runs, stocks_analyzed, agent_reports (JSON, model, prompt version, latency, tokens), debates, decisions (meta output, mode, strategy, regime, confluence, risk verdict + rule), approval_queue (status PENDING/APPROVED/REJECTED/MODIFIED/TIMEOUT, my edits, my reason, decided_at), orders, positions (mode, product, holding days), trades (entry/exit, mode, qty, SL, targets, exit reason SL/TARGET/TRAIL/TIME/SQUAREOFF/GAP/EVENT/HALT/MANUAL, gross, charges, net, source SYSTEM/MANUAL), shadow_trades, corporate_action_adjustments, capital_ledger, daily_summary (per mode and total, TRADING, RESERVE, equity, HWM, drawdown, tier), control_commands, audit_log, login_attempts, errors. Plus JSONL logs per day.

---

## 13. EMAIL NOTIFICATIONS (each type on/off from the dashboard)
- PENDING APPROVAL: symbol, mode, side, entry, SL, targets, qty, confidence, grade, top 3 bull/bear points, approval deadline. (Approval itself happens only in the dashboard.)
- ENTRY / EXIT: details, mode, strategy, exit reason, holding time, gross/charges/net P&L.
- EOD (15:45): trades by mode with P&L, charges, win rate; open CNC positions with unrealized P&L and days held; capital section (profit locked or loss absorbed/refilled, TRADING, RESERVE, TOTAL EQUITY, HWM, drawdown, tier, sizing capital); stocks analysed but not traded with reasons; my approvals vs rejections and shadow outcomes; agent failures; equity chart.
- SWING SCAN (evening): next-session candidates.
- URGENT: HALT, gap through SL on an overnight position, data feed down during market hours.
- Retry 3 times; never block trading on email failure; recipients and quiet hours configurable.

---

## 14. DASHBOARD — FULL CONTROL, OWNER-ONLY

### 14.1 Authentication
- streamlit-authenticator, single admin user, bcrypt-hashed password in a secrets file. `python scripts/set_password.py` to set/change.
- 30-min idle timeout, 15-min lockout after 5 failed logins, all attempts logged.
- Bind to 127.0.0.1 by default; README documents safe remote/mobile access (Tailscale, or Caddy reverse proxy with HTTPS). Never plain HTTP on the internet.
- Every page requires login. Layout must work on a phone screen (approvals are time-sensitive).

### 14.2 Control architecture
- Dashboard and engine are separate processes. Dashboard writes control_commands; engine polls every 2–5 s, executes, writes status. Dashboard never calls agents or the engine directly.
- config/hard_caps.yaml (editable only on disk, NOT from the dashboard): paper-only flag, MAX_LOSS_PCT ≤ 10, max leverage, max risk per trade, no overnight shorts, SL-tighten-only rule. Dashboard inputs are clamped to these caps and show the cap.
- Every settings change creates a new settings_versions row; I can view diffs and REVERT to any previous version. Changes apply to new trades; open trades keep their settings unless I edit them explicitly.
- Every action goes to audit_log. Destructive actions require typing "CONFIRM" or the batch name.

### 14.3 Control panel (all of these must exist)
1. Engine: Start / Pause / Resume / Stop; per-mode pause; EMERGENCY KILL SWITCH (pause everything, square off intraday, optional exit of all CNC positions).
2. Modes: enable/disable each mode; capital allocation %; selection policy (FIXED/AUTO); execution policy per mode (AUTO / MANUAL_APPROVE / SIGNALS_ONLY); approval timeouts; max holding days; entry windows.
3. Approval queue (live, auto-refresh, countdown per item): full reasoning (agent reports, debate, meta, confluence, charges), and buttons: Approve / Reject (with optional reason) / Modify (mode, entry, SL [tighter only after entry], targets, qty up to the risk-manager maximum) / Approve as different mode. Bulk reject.
4. Manual trade: I can open my own paper trade (symbol, mode, side, entry type, qty, mandatory SL, targets). Option "Analyze first" runs the full agent pipeline before I submit. Goes through the risk manager; tagged MANUAL.
5. Positions: close fully or partially; tighten SL; change targets; change trailing rule; convert INTRADAY → SWING/POSITIONAL; extend or shorten holding period; pin "do not auto-exit before event" toggle (still subject to SL and HALT).
6. Watchlist & universe: universe per mode; add/remove symbols; whitelist-only mode; blacklist; price, liquidity and sector filters.
7. Strategies: enable/disable per mode; edit parameters within ranges; view live performance; re-enable auto-disabled strategies.
8. Agents & models: enable/disable each agent; assign model per agent from installed Ollama models; agent weights per mode; confidence thresholds per mode; debate rounds; view/edit prompt templates (versioned, with revert).
9. Risk: all limits from section 5 (within hard caps); drawdown tier thresholds; gap buffer; overnight exposure; event-avoidance rules.
10. Capital: PROFIT_LOCK_PCT; RESERVE_PROTECTED_PCT; sizing step rules; manual RESERVE → TRADING transfer (confirmed, audited).
11. Schedule: cycle interval, scan times, square-off time (within NSE hours), holiday overrides.
12. Notifications: each email type on/off, recipients, quiet hours, "send test email".
13. Batches: start new (name, starting capital default ₹1,000, copy settings from current or a previous batch, how to handle open CNC positions); resume HALTED (warning); close; export CSV/Excel; delete (double confirmation, audit retained).
14. Data & system: data provider selection; "Analyze now" for any symbol/mode; log viewer with filters; DB backup/restore; health status.

### 14.4 Analytics pages
1. Overview: TRADING, RESERVE, TOTAL EQUITY, HWM, drawdown & tier, net P&L (total and per mode), charges, win rate, profit factor, expectancy; equity curve (TRADING vs RESERVE stacked, HWM line, floor lines, tier bands); daily P&L bars; mode split.
2. Live: open positions with live and unrealized P&L, pending approvals, loss budget remaining, regime, engine heartbeat, next cycle.
3. Capital ledger: every lock, refill, sizing change, manual transfer with running balances.
4. Day-wise log: full timeline for any date (scans, decisions, approvals, entries, exits, EOD).
5. Trades: filter by mode/strategy/source/date; click → candlestick chart (mode's timeframe) with entry/SL/targets/exit, plus full agent reports, debate, meta reasoning and my approval/edit history.
6. Rejected / No-trade: everything not traded with the reason (meta, risk rule, my rejection, timeout) and shadow outcome.
7. Me vs System: P&L of trades I approved, rejected, or modified vs what the system alone would have done (shadow tracking).
8. Strategy, mode & agent analytics: metrics per mode, strategy, regime, agent; latency and failure rates.
9. Batch comparison: side-by-side metrics, overlaid equity curves, settings diff between batches.
A batch selector at the top filters every page.

---

## 15. PROJECT STRUCTURE
config/ (incl. hard_caps.yaml), data/, features/ (indicators, patterns, regime), modes/, strategies/, agents/ (sub_agents/, debate/, meta/, prompts/), risk/, capital/, paper_engine/ (incl. charges, corporate actions), approvals/, shadow/, storage/, notifications/, dashboard/, control/, scheduler/, scripts/, tests/, logs/, main.py, README.md, .env.example, requirements.txt (pinned), Makefile (setup, pull-models, run, dashboard, test, replay).

---

## 16. QUALITY REQUIREMENTS
- Type hints, docstrings, no silent excepts.
- Versioned prompt templates; version saved with every agent report.
- No look-ahead bias in replay/backtest.
- Graceful degradation: any failure → log it and default to NO_TRADE; never leave a position without an active SL.
- README: setup, free broker data API keys, Gmail App Password, dashboard password, safe mobile access, how modes/approvals/capital/reserve rules work, how to read the dashboard.

---

## 17. BUILD PHASES (in order; stop and show results after each)
1. Skeleton, config + hard caps, DB models, settings versioning, logging, paper-trading guard.
2. Data layer (yfinance + one broker provider), NSE holidays, market hours, event/corporate-action calendar.
3. Features: indicators, patterns, levels, regime detection, all timeframes. Unit tests.
4. Charges (MIS + CNC), capital module, capital protection, risk manager, paper engine with overnight/gap handling and mode conversion. Unit tests for every rule (profit lock, loss absorb, capped refill, sizing steps, both floors, gap-adjusted risk, halt/resume, charges rejection, ledger balancing, SL-tighten-only, no overnight shorts).
5. Modes + strategy modules + confluence scoring.
6. Sub-agents (mode-aware) with Ollama + Pydantic, one at a time.
7. Debate + meta + LangGraph wiring.
8. Approval queue, manual trades, shadow tracking.
9. Scheduler + full multi-day loop + REPLAY mode (historical days minute-by-minute, including overnight holds).
10. Email notifications.
11. Dashboard: auth, control architecture, every control in 14.3, every page in 14.4.
12. Walk-forward replay over past weeks/months per mode before relying on live paper results.

Start with Phase 1. Before writing code, show me the planned folder structure and ask me for: my hardware (GPU/RAM), which free broker data API I have, my email for alerts, and confirm starting capital ₹1,000 and default mode allocations.
