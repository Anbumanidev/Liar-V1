# Liar-V1 — Multi-Agent, Multi-Mode **Paper** Trading System for NSE

A local, open-source, multi-agent paper-trading system for Indian cash equities (NSE). It trades
four modes (INTRADAY, BTST, SWING, POSITIONAL) with a starting capital of ₹1,000. You control
everything from an owner-only dashboard.

> **PAPER TRADING ONLY.** No code path places a real order. Broker APIs are used for market data
> only. Real order placement is blocked three ways:
> 1. a hard-coded constant plus a startup assertion on `config/hard_caps.yaml`;
> 2. broker SDK order methods are patched to raise `RealTradingForbiddenError`;
> 3. the HTTP layer refuses any order, GTT or position-conversion endpoint.
>
> Every order row is constrained in the DB to `is_paper = true`.

---

## 1. How it works (one minute)

```
data (yfinance / Angel One / Upstox, NSE calendar, events, news, fundamentals)
  -> features (indicators, candlestick patterns, levels, regime)          deterministic Python
  -> strategies (17) + confluence score  -> pre-filter                    deterministic Python
  -> 10 specialist agents (parallel) -> weighted vote                     local LLMs (Ollama), JSON only
  -> Bull vs Bear debate (rebuttal rounds) -> Meta decision               local LLMs, choose among options
  -> RISK MANAGER (final veto, sizing, charges, capital protection)       deterministic Python
  -> execution policy: AUTO | your approval queue | signals only
  -> paper engine (fills, SL/targets/partials/trailing/time stops, gaps)  deterministic Python
  -> capital ledger (TRADING / RESERVE, profit lock, refill, sizing)      deterministic Python
```

* **LLMs never do math.** Prices, sizes, charges and P&L are computed in Python. LLMs only
  interpret facts and pick among options that the code generated.
* **Every LLM output is JSON validated by Pydantic.** An invalid output is retried up to 2 times.
  After that the agent is marked FAILED and its vote counts as neutral; a failed meta agent means
  NO_TRADE.
* **The risk manager has final veto.** It overrides the meta agent *and your own manual trades and
  edits*. When it blocks you, the dashboard shows the exact rule.
* **Capital protection overrides everything.** Total equity is kept above the stricter of two
  floors: ₹910 absolute on a ₹1,000 start, and the high-water mark × 0.91.

---

## 2. Setup (Windows 11; Linux/macOS similar)

Requirements: Python 3.11+ (tested on 3.14), ~10 GB disk for models, an NVIDIA GPU is optional.

```powershell
git clone <this repo>; cd Liar-V1
.\tasks.ps1 setup            # venv + pinned requirements + DB init (or: make setup)
copy .env.example .env       # then edit .env (see below)
.\.venv\Scripts\python.exe scripts\set_password.py   # dashboard login (bcrypt)
.\tasks.ps1 pull-models      # Ollama models for your hardware profile
.\.venv\Scripts\python.exe scripts\health_check.py
```

No `make` on Windows? `tasks.ps1` has the same targets: `setup, pull-models, run, dashboard, test,
replay, walk-forward, backup, health`.

### 2.1 Local LLMs (Ollama)
1. Install Ollama from <https://ollama.com/download> and keep it running.
2. `python scripts/pull_models.py --profile low` pulls the default profile:
   * `low` (default; fits a 6 GB GPU such as an RTX 4050): `qwen3:4b` for the 10 specialist
     agents and `qwen3:8b` for Bull, Bear and Meta.
   * `high` (≥ 12 GB VRAM): `qwen3:8b` / `qwen3:14b`.
3. On the dashboard (Control → Agents & models) you can assign any installed model to any agent.
   Each agent call's latency is logged, and a warning is logged if one stock's full cycle takes
   more than 60 s.
4. Without Ollama, set `engine.llm_backend: rules`. Agents then use their deterministic
   interpretation, which is fast and is the default for replay and walk-forward.

### 2.2 Market data
* **yfinance** works with no keys (default). It has delayed quotes and limited intraday history:
  1-minute bars for about 7 days and 5-minute for about 60 days.
* **Angel One SmartAPI** (free): create an app at <https://smartapi.angelbroking.com>, enable TOTP,
  and fill `ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_PIN` and `ANGEL_TOTP_SECRET` in `.env`.
  Then select `angelone` under Control → Data & system. It is used for **data only**; its order
  methods are disabled at runtime.
* **Upstox**: set `UPSTOX_ACCESS_TOKEN` (plus key/secret).
* **Behind a corporate proxy or antivirus HTTPS scanning?** The system builds a CA bundle from
  `certifi` plus the Windows certificate store automatically (`engine.use_os_cert_store`). If
  that's not enough, set `CA_BUNDLE_PATH`. TLS verification is never disabled.

### 2.3 Email alerts (Gmail App Password)
Google Account → Security → 2-Step Verification → **App passwords** → create one. Put it in
`.env` as `SMTP_APP_PASSWORD`, along with `SMTP_USER` and `ALERT_EMAIL_TO`. Use "send test
email" on the dashboard to check it.
Email types:
* PENDING APPROVAL, ENTRY/EXIT, EOD report (15:45), SWING SCAN, URGENT (HALT, gap through SL,
  data feed down).
* Each type can be switched on/off; quiet hours are supported (URGENT can bypass them).
* Each email is retried 3 times, and email never blocks trading.

### 2.4 Dashboard password
`python scripts/set_password.py` writes a bcrypt hash to `secrets/dashboard_auth.yaml`, which is
git-ignored.
* Single admin user.
* 30-minute idle timeout.
* 15-minute lockout after 5 failed logins; every attempt is logged.

---

## 3. Running

```powershell
.\tasks.ps1 run          # engine (scheduler + control loop):  python main.py engine
.\tasks.ps1 dashboard    # dashboard on http://127.0.0.1:8501 (separate process)
```

The engine and dashboard are **separate processes**. The dashboard only writes `control_commands`
and settings versions. The engine polls those every 2–5 s, executes them, and writes the results
and its heartbeat back.

Other commands:

```powershell
python main.py init-db                                   # tables, settings v1, prompts, first batch
python main.py analyze RELIANCE --mode SWING             # full agent pipeline now (analysis only)
python main.py replay --start 2026-10-01 --end 2026-10-31 --modes INTRADAY,SWING --symbols INFY,TCS
python main.py walk-forward --start 2026-06-01 --end 2026-10-31 --modes SWING --train-days 20 --test-days 5
python scripts/backup_db.py --keep 30
```

### Daily schedule (IST, NSE trading days only; holidays from NSE's own list)
| Time | Job |
|---|---|
| 08:30 | pre-market: event calendar, corporate actions on held positions |
| 09:00 | intraday watchlist (every stock and the reason it was included or excluded is logged) |
| 09:08 | overnight gap check on CNC positions (urgent email if pre-open is through the SL) |
| 09:20–10:30 | yesterday's approved SWING/POSITIONAL limit orders can fill; BTST positions exit |
| 09:30–14:45 | INTRADAY cycle every 5 min; all positions monitored every minute |
| 14:30–15:15 | BTST scan (if enabled) |
| 15:15 / 15:20 | intraday square-off / CNC max-holding time stop |
| 15:30 / 15:45 | EOD capital processing / EOD report email |
| 16:00 | SWING + POSITIONAL scan → candidates for the next session |

---

## 4. The rules you should know

### Modes
| Mode | Holding | Product | Direction | Default |
|---|---|---|---|---|
| INTRADAY | same day, square-off 15:15 | MIS | long + short | **on, 50 %** |
| BTST | buy 14:45–15:20, sell next morning | CNC | long only | off, 0 % |
| SWING | 2–10 trading days | CNC | long only | **on, 50 %** |
| POSITIONAL | up to ~3 months | CNC | long only | off, 0 % |

* **Long-only CNC is hard-coded:** there are no overnight shorts in the Indian cash market.
* **Execution policy per mode:** AUTO / MANUAL_APPROVE (the default) / SIGNALS_ONLY.
* **Approval deadlines:** INTRADAY 3 min, BTST 10 min, SWING/POSITIONAL until 09:45 of the next
  session. An expired item is recorded as `APPROVAL_TIMEOUT`.
* **Converting a position:** an open INTRADAY long can be converted to SWING/POSITIONAL before
  15:10, if the target mode's allocation and the risk limits allow it.

### Who decides: fully automatic or human in the loop
One switch, on the dashboard **Home** page (also Settings → Modes), in `config/modes.yaml`
(`autonomy:`) or from the command line (`python main.py engine --autonomy FULL_AUTO`):

| Setting | What happens to a risk-approved trade |
|---|---|
| `PER_MODE` (default) | each mode's own execution policy (all default to MANUAL_APPROVE) |
| `HUMAN_IN_THE_LOOP` | every trade waits for your Approve / Reject / Modify |
| `HYBRID` | auto-executes only top setups (default grade A+ and confidence ≥ 85), the rest wait for you |
| `FULL_AUTO` | every trade executes automatically (typed `CONFIRM` required on the dashboard) |
| `SIGNALS_ONLY` | logged + emailed, never executed |

Notes:
* The risk manager and capital protection still veto every trade in every setting.
* `config/hard_caps.yaml: allow_full_auto: false` locks FULL_AUTO and HYBRID off. That file can
  only be edited on disk.
* Every switch is an audited settings version.

### Tax & charges (estimates, not tax advice)
**Rules used** (FY 2026-27; Income-tax Act 2025 in force from 1 April 2026; Budget 2026 kept the
rates):

| Trade | Tax head | Rate |
|---|---|---|
| Intraday | speculative business income | your slab rate |
| Delivery held ≤ 12 months | STCG (old s.111A → new s.196) | 20% |
| Delivery held > 12 months | LTCG (old s.112A → new s.198) | 12.5% above ₹1.25 lakh per FY |

* 4% cess (plus surcharge if you set one) on top.
* Short-term losses offset STCG and then LTCG; long-term losses offset only LTCG; intraday losses
  offset only intraday profit.
* STT is not deductible from capital gains.

**Before a trade**, every approval card (and the manual-trade preview) shows:
* the tax head;
* charges, tax and **what you keep** at the SL, T1 and T2 exits;
* the price move needed just to pay the charges;
* the after-tax R:R.

**After trades**, the **Tax & charges** page shows:
* the FY estimate, with set-off and carry-forward;
* STT paid;
* per-trade tax lines;
* tips with estimated savings. The tips are rule-based: tax-loss harvesting before 31 March,
  carry-forward reminders, the LTCG exemption left, advance-tax dates (15 Jun / Sep / Dec / Mar),
  the business-income option, and cutting DP charges by selling the whole quantity in one go.

The EOD email includes the FY estimate and the top tips. Set your slab rate under Tax & charges →
Tax settings (or `config/tax.yaml`).

### Risk manager (final veto)
**Sizing:**
* Risk per trade is 1 % of the mode's sizing capital.
* Quantity = `floor(min(risk / |entry − SL|, mode capital / entry))`, further capped by the
  gap-adjusted loss budget, free cash, max position % and overnight exposure.

**Small-capital rules (below ₹10,000):** max 2 open positions, max 60 % of capital per position,
max 1 position per sector.

**Other limits:** daily loss limit 3 %; INTRADAY entries only between 09:30 and 14:45.

**Rejections:**
* near a circuit limit;
* R:R below 1.5 (intraday) or 2 (other modes);
* SL distance below 0.3 × ATR or above 3 × ATR;
* TRADEABLE = false;
* low confluence or confidence;
* events (results, ex-dates) inside the holding window;
* correlation above 0.7 with an open position;
* averaging down.

**Charges check:** a trade is rejected (`CHARGES_EXCEED_EDGE`) if expected profit at T1 minus
round-trip charges is ≤ 0, or if charges exceed 1 % of the position value.

**Stop-loss:** a stop-loss can only ever move closer to price — by the system *or by you*.

> **Heads-up for ₹1,000 accounts:** the delivery DP charge (~₹15.34 per scrip per sell day) is
> about 2.5 % of a ₹600 position. Most SWING/BTST trades will therefore be rejected by the
> 1 %-charges rule until capital grows, or until you change that limit (within the hard caps).
> INTRADAY (MIS) charges are much lower.

### Capital (TRADING vs RESERVE)
* **TRADING** = cash + capital blocked in open positions. **RESERVE** = locked profits; it is
  never traded and never counted as available.
* **EOD on a profit day:** 30 % of realized net P&L moves to RESERVE.
* **EOD on a loss day:** TRADING absorbs the loss. If TRADING falls below the starting capital,
  it is refilled from RESERVE, but RESERVE never drops below 50 % of its peak.
* **Sizing capital grows slowly:** only after TRADING has stayed ≥ 10 % above it for 3
  consecutive days with no drawdown tier. **It shrinks immediately** when TRADING falls below it.
* **Ledger invariant (tested):** starting capital + Σ realized net P&L + Σ confirmed adjustments =
  TRADING + RESERVE.

### Capital protection (max 10 % loss)
* Floors: absolute (₹910) and trailing (HWM × 0.91). The stricter one applies.
* **New trade risk** = min(normal risk, remaining loss budget / 3). Overnight positions count as
  max(SL risk, 3 % gap risk).
* Drawdown tiers:
  * 3–6 %: half size, A/A+ setups only;
  * 6–9 %: quarter size, A+ only, max 1 open position, no new overnight positions;
  * ≥ 9 % or a floor hit: **HALT** — all positions are exited, new trades are blocked and you get
    an urgent email. Trading resumes **only from the dashboard**.

### Trade management
* 50 % is booked at 1R and the SL moves to breakeven.
* Trailing: ATR or EMA9 on 5m (INTRADAY); daily-close EMA20 or ATR (SWING/POSITIONAL).
* Time stops; pre-event exits; next-day open fills when a position gaps through its SL.
* Stocks are paused after 3 consecutive losses; there is an intraday profit lock (only A+ setups
  at half size after +2 %, and the mode stops if half of the day's peak profit is given back).

### Self-evaluation without overfitting
* **Shadow tracking:** every rejected setup — by the system, the risk manager, you, or a timeout —
  is followed to its would-be outcome without capital (dashboard: "Me vs System").
* **Auto-disable:** a strategy with negative expectancy after 30 trades is disabled; you can
  re-enable it from the dashboard.
* **Tuning only via walk-forward replay:** parameters are chosen on one window and scored on the
  next unseen window, and every change is logged with before/after metrics.

---

### Advanced methods
* **"Stocks in play" ranking** (`strategies/ranking.py`): each intraday cycle analyses the
  watchlist stocks most likely to move first. The ranking uses relative volume at the same time of
  day, the gap or move from the previous close, the range versus ATR, and strength versus NIFTY.
  It refreshes every 15 minutes (`engine.watchlist_ranking`).
* **Monte Carlo risk-of-ruin** (Analytics → Risk lab): your own trade results are resampled
  thousands of times, with the same HALT floors as live trading. It shows the chance of profit, the
  chance of a HALT, final-equity bands and the 95th-percentile drawdown.
* **Agent calibration** (Risk lab): the hit rate and Brier score of every AI agent are measured
  against what the price did next, on both traded and shadow setups. It suggests agent weights that
  you apply with one click; nothing changes automatically. Validate them with a walk-forward replay.

## 5. The dashboard
The navigation is grouped as **Home · Trading · Analytics · Settings · System**. Home is a
one-screen cockpit:
* alerts (HALT, engine down, data feed);
* equity, today's P&L, drawdown, TRADING / RESERVE;
* Start / Pause buttons;
* the autonomy switch;
* trades waiting for you, with one-tap Approve / Reject and their after-tax value;
* open positions;
* a **setup checklist** (engine, Ollama models, email, data, charges verification) with the
  command that fixes each item;
* a plain-language "how it works".

* A batch selector at the top filters every page. Every page needs login, and the layout works on
  a phone.
* **Analytics:**
  * Overview (equity curve with TRADING/RESERVE, HWM, floors and tier bands);
  * Live (positions, approvals, loss budget, regime, heartbeat);
  * Capital ledger; Day-wise log;
  * Trades (chart with entry/SL/targets/exit plus the full agent reasoning);
  * Rejected / No-trade (with shadow outcomes); Me vs System;
  * Strategy/mode/agent analytics (incl. LLM latency and failure rates);
  * Batch comparison.
* **Control:**
  * Engine (start/pause/stop, per-mode pause, **EMERGENCY KILL SWITCH**);
  * Modes; Approval queue (approve / reject / modify / approve as another mode, countdowns);
  * Manual trade (optionally "Analyze first"); Positions (close, tighten SL, targets, trailing,
    convert, holding, event pin);
  * Watchlist & universe; Strategies; Agents & models (models, weights, prompt templates with
    versions and revert);
  * Risk; Capital (RESERVE → TRADING transfer); Schedule; Notifications;
  * Batches (new / resume HALTED / close / export / delete); Data & system (provider, Analyze now,
    logs, backup/restore, health).
* **Hard caps** in `config/hard_caps.yaml` can only be changed on disk: paper-only, MAX_LOSS ≤ 10 %,
  max leverage, max risk per trade, no overnight shorts, SL-tighten-only. Dashboard inputs are
  clamped to these caps and show them.
* **Settings changes** create a new version (diff + revert) and apply to **new** trades; open
  positions keep the settings they were opened with.
* **Destructive actions** require typing `CONFIRM` or the batch name.

### Safe remote / mobile access
The dashboard binds to **127.0.0.1** only. **Never expose plain HTTP to the internet.** Options:
* **Tailscale (recommended):** install it on the PC and your phone, then open
  `http://<pc-tailscale-name>:8501` after running the dashboard with
  `--server.address <tailscale-ip>`. Traffic is end-to-end encrypted, with no open ports.
* **Caddy reverse proxy with HTTPS:** keep Streamlit on 127.0.0.1 and use a Caddyfile:
  `trading.example.com { reverse_proxy 127.0.0.1:8501 }`. Caddy obtains TLS certificates
  automatically. Restrict the source IPs if you can; the dashboard login and lockout still apply.

---

## 6. Data, logs and storage
* **Database:** SQLite at `runtime/trading.db` (WAL mode, shared by the engine and dashboard). For
  PostgreSQL, set `DATABASE_URL`.
* **Tables:** every table in spec section 12 — batches, settings versions, runs, agent reports,
  debates, decisions, approvals, orders, positions, trades, shadow trades, capital ledger, daily
  summary, control commands, audit log, login attempts, errors, …
* **Batches:** everything carries `batch_id`, and history is never deleted by a reset.
* **Logs:** raw JSONL in `logs/YYYY-MM-DD/<category>.jsonl` (agents, debates, decisions, risk,
  orders, trades, capital, approvals, control, settings, errors, …), plus loguru logs per process.
* **Replay:** each replay uses its own DB in `runtime/replay/<name>.db` and writes a JSON report
  next to it, so your live paper account is never touched.

## 7. Project structure
```
config/        YAML settings + hard_caps.yaml, typed schema, versioned settings store, secrets
core/          enums, shared types, clock (simulated for replay), paper-only guard, logging, TLS
data/          NSE calendar, providers (yfinance / Angel One / Upstox / replay), service, universe,
               NSE client, events, news, fundamentals
features/      indicators, patterns, levels, regime, feature builder, SL candidates, market, FinBERT
modes/         hard-coded mode specs (products, long-only CNC)
strategies/    17 strategies, confluence scoring, performance / auto-disable / Kelly
agents/        LLM client, prompt store, 10 sub-agents, debate, meta, LangGraph pipeline (graph.py)
risk/          risk manager (final veto), risk state, hard guards (SL tighten-only, no averaging down)
capital/       accounts + ledger, capital protection (floors, tiers, loss budget)
paper_engine/  charges, fills, broker, position monitor, corporate actions
approvals/     approval queue, execution-policy router, manual trades
shadow/        shadow tracking of rejected setups
notifications/ email sender, notifier, EOD report, templates
dashboard/     Streamlit app, auth, control pages, analytics pages
control/       command payloads/queue, processor, batches, bootstrap, engine loop
scheduler/     daily plan, jobs, REPLAY, walk-forward
tax/           Indian tax estimates (pre-trade, FY set-off, tips)
scripts/       set_password, pull_models, backup_db, health_check
tests/         ~1,140 tests (pytest)
```

## 8. Tests
```powershell
.\tasks.ps1 test                                  # full suite (network/Ollama tests skipped)
$env:RUN_NETWORK_TESTS=1; $env:RUN_OLLAMA_TESTS=1; .\.venv\Scripts\python.exe -m pytest tests
```
Tested guarantees include:
* the paper-only guard;
* the ledger invariant under randomized sequences;
* a 200- and 60-sequence proof that stop-loss and gap hits never breach either capital floor;
* every risk rule; SL tighten-only; no overnight shorts;
* profit lock, refill and sizing rules; charges math;
* no-look-ahead replay data;
* the full LangGraph pipeline; the dashboard → engine command loop; a multi-day replay.

## 9. Known limitations
* **Charges:** rates in `config/charges.yaml` follow common discount-broker tariffs. STT was
  verified for 2026; **re-verify** the exchange, stamp duty and DP charges with your broker.
* **Static lists:** NSE holiday lists, index constituents and the macro-event calendar
  (RBI/Budget/FOMC) are refreshed from NSE where possible; the static fallbacks are marked
  "verify".
* **yfinance:** intraday history is limited (1m ≈ 7 days, 5m ≈ 60 days). Long INTRADAY replays
  need a broker data API. SWING/POSITIONAL replays fall back to daily bars.
* **Promoter holding:** this uses yfinance "insider holding" as a proxy.
* **Live results:** paper results depend on data quality and delays. Run walk-forward replays over
  past weeks or months per mode before relying on live paper results.
