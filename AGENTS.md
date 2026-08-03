# Project continuity guide

This file applies to the entire repository. It is the durable handoff for future
Codex sessions, especially when the user switches between Personal and Business
accounts/workspaces. Chat history and memory may not migrate; the local repo,
Git history, this file, and other checked-in documentation do.

## Start here

- Repository on the current Windows machine:
  `D:\Projects\个人兴趣\crypto-prediction`.
- Communicate with the user in Chinese unless the user asks otherwise.
- Inspect the worktree, current branch, recent commits, and this file before
  changing anything. Do not assume a fresh worktree.
- Treat current code and focused tests as authoritative. `README.md` contains
  useful setup material but some strategy lists, email behavior, example CLI
  flags, and schedules are older than the implementation.
- Preserve user changes and untracked files. Never use destructive Git or file
  commands unless the user explicitly requests them and the exact targets have
  been verified.

## Mission and user intent

This is a personal crypto and US-stock research/signal bot. The primary research
objective is a robust, explainable strategy that produces **excess return over
same-cost buy-and-hold**, not merely a positive nominal return or an impressive
in-sample fit.

The user manually controls the real dollar allocation to each asset. A strategy
that is “100% invested” means 100% of the amount the user assigned to that asset,
not the user's entire portfolio.

Known user-held / deliberately enabled strategy assets are BTC, ETH, SOL, TCEHY,
COIN, and MSFT. The user explicitly confirmed holdings in TCEHY, COIN, and MSFT.
Other assets in `CURS` or `STOCKS` may still be fetched or run with legacy generic
strategies, but must not silently inherit one of the asset-specific strategies.

## Non-negotiable safety rules

- Keep `COMMIT = False` in `app/core/config.py` unless the user explicitly asks
  to enable real trading. The normal daily job is simulation/recommendation only.
- Never place live orders, enable a live-trading switch, or expand permissions
  based only on an inferred desire to deploy.
- Never print, copy into docs, or commit `app/core/secret.ini`, API keys, Gmail
  app passwords, database credentials, browser state, or session tokens.
- `secret.ini` and local database files are intentionally Git-ignored.
- The user's required execution friction is 2% per side. Fixed defensive
  strategies also use 10 bps adverse slippage and execute a prior-close signal
  at the next open. Keep strategy and buy-and-hold execution assumptions equal.
- No lookahead. For example, COIN may use only a completed prior-calendar-day
  BTC candle and then execute at the following US stock-session open.
- Do not select or tune parameters on the final test interval. Avoid large
  parameter grids and report selection bias or weak subperiods honestly.
- Historical outperformance is evidence, not a promise of future returns.

## Current enabled asset-specific strategies

The source of truth is `app/core/config.py`; implementation registration is in
`app/trading/strategies.py`, construction in `app/trading/strat_trader.py` and
`app/trading/trader_driver.py`.

| Asset | Registered strategy | Frozen rule |
| --- | --- | --- |
| BTC | `BTC-SMA200-DEFENSIVE` | Enter above SMA200 +5%; exit below SMA200 -5%; no minimum hold. |
| ETH | `ETH-120D-BREAKOUT-DEFENSIVE` | Enter on a 120-day high; exit 5% below the position peak; no BTC gate. |
| SOL | `SOL-30D-BREAKOUT-DEFENSIVE` | Enter on a 30-day high; exit 10% below peak; requires the BTC defensive regime. |
| TCEHY | `TCEHY-REGIME-DEFENSIVE` | SMA200 trend, 20-day SMA200 slope, and SMA20 ±2 sigma decisions in range regimes. |
| COIN | `COIN-BTC-SMA200-DEFENSIVE` | Follow BTC-USD SMA200 ±5%, using a one-calendar-day lag; bootstrap buy-and-hold during context warmup. |
| MSFT | `MSFT-20D-BREAKOUT-DEFENSIVE` | Bootstrap invested; exit 10% below peak; re-enter on a 20-day high. |

Important isolation behavior:

- `CRYPTO_STRATEGIES` contains only the BTC, ETH, and SOL fixed strategies.
- `crypto_strategies_for_asset()` plus `CRYPTO_STRATEGY_ASSET_ALLOWLIST` prevent
  any of those strategies from trading another crypto.
- TCEHY, COIN, and MSFT have exclusive stock mappings. For these tickers,
  `stock_strategies_for_asset()` returns exactly one fixed strategy and excludes
  the legacy generic stock strategies.
- Fixed strategies force `buy_pct = sell_pct = 1.0` inside their simulation
  bucket. The user controls the real per-asset allocation separately.
- `MA-BOLL-BANDS` was not robustly superior across bull, bear, and range crypto
  regimes. Do not restart parameter tuning for deployment without new evidence.

## Research and validation protocol

Use this sequence for strategy work:

1. Obtain roughly three calendar years of daily data when the API supports it.
2. Establish same-cost buy-and-hold before evaluating strategy return.
3. Keep train, validation, purge gap, and final test intervals separated.
4. Tune only on train/validation; freeze parameters before looking at final test.
5. Execute close-derived signals at the next open with 2% friction per side and
   10 bps adverse slippage.
6. Compare excess return, max drawdown, transaction count, turnover, exposure,
   and behavior across bull/bear/range subperiods.
7. Verify both the pure offline simulator and the registered runtime path. Their
   return and transaction count must match.
8. Add asset-isolation, frozen-parameter, execution-cost, next-open, and
   profitability-gate tests before enabling a strategy.
9. Do not describe a strategy as robust merely because full-period/test excess
   is positive. Disclose subperiod underperformance.

The main reports are:

- `artifacts/backtests/tcehy_regime_defensive_validation.json`
- `artifacts/backtests/held_stock_defensive_validation.json`
- validation code under `app/backtesting/`

### Last audited stock results (data through 2026-07-31)

All numbers include 2% per-side friction and 10 bps slippage.

| Asset | Strategy | Full return | Buy/hold | Excess | Strategy DD | Buy/hold DD | Trades |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TCEHY | `TCEHY-REGIME-DEFENSIVE` | 58.69% | 42.10% | +16.59 pp | 26.59% | 38.23% | 2 |
| COIN | `COIN-BTC-SMA200-DEFENSIVE` | 196.31% | 64.21% | +132.10 pp | 47.84% | 66.39% | 6 |
| MSFT | `MSFT-20D-BREAKOUT-DEFENSIVE` | 47.85% | 41.83% | +6.02 pp | 21.27% | 34.50% | 9 |

Caveats that must survive account changes:

- COIN underperformed buy-and-hold in the early selection interval by 42.33 pp
  and in validation by 2.70 pp, but strongly outperformed in the held-out test.
- MSFT underperformed in selection by 8.92 pp and validation by 0.44 pp, while
  outperforming in the held-out test by 11.15 pp.
- These rules are defensive/regime-dependent, not uniformly superior in every
  market interval.

## Repository map

- `app/core/config.py`: safety flags, enabled strategies, frozen parameters,
  allowlists, data windows, assets, and execution settings.
- `app/core/main.py`: one-time/cron entrypoint, crypto and stock simulations,
  logging, email composition, and DEFI monitoring.
- `app/trading/strategies.py`: strategy functions and `STRATEGY_REGISTRY`.
- `app/trading/strat_trader.py`: portfolio state, 2% brokerage, pending
  next-open orders, slippage, and indicator state.
- `app/trading/trader_driver.py`: fixed candidate construction, asset checks,
  BTC context alignment, and historical data feed.
- `app/trading/us_stock_client.py`: Yahoo Finance daily data and SQLite cache.
- `app/db/database.py`: SQLAlchemy models and SQLite/PostgreSQL connection.
- `app/db/db_management.py`: init/test/stats/backfill/clear/drop CLI.
- `app/backtesting/`: auditable pure simulations and registered-runtime checks.
- `tests/strategy/` and `tests/backtesting/`: strategy, isolation, execution,
  and validation gates.
- `artifacts/backtests/`: checked-in evidence reports plus some local-only
  research artifacts.

## Windows and Poetry workflow

The normal shell is PowerShell. Python is managed by Poetry.

```powershell
poetry install
poetry env activate
python app/core/main.py
```

Running without activation is usually clearer:

```powershell
poetry run python app/core/main.py
poetry run python app/core/main.py --asset=crypto
poetry run python app/core/main.py --asset=stock
```

If `poetry` is not on `PATH`, discover it with `Get-Command poetry -All`. On the
current machine the known executable is:

```powershell
& "C:\Users\Administrator\AppData\Roaming\Python\Scripts\poetry.exe" run python app/core/main.py
```

`python app/core/main.py` is the trading/signal job. `python start_server.py`
starts the Flask dashboard and is not a replacement for that job.

The repo's `--cronjob` mode currently schedules 13:00 according to the local
process clock and logs it as 21:00 SGT. Separately, the user requested an
external daily trigger at **08:45 Asia/Shanghai** with continued output
monitoring. That external trigger is not defined by this repository and may be
account/workspace-specific. After switching accounts, inspect or recreate the
08:45 automation before assuming it still runs. Do not silently change either
schedule.

Runtime output is written to `./log.txt` in addition to console logs. A process
monitor should report crashes, missing market data, Gmail failures, and the
final recommendations, not just that the process exists.

## Database and market data

SQLite is the current preferred local database. The default URL is
`sqlite:///./crypto_trading.db`; the database file is local and Git-ignored.
PostgreSQL remains optional and is available through `docker-compose.yml` or a
local server.

PowerShell commands:

```powershell
$env:DATABASE_URL = "sqlite:///./crypto_trading.db"
poetry run python app/db/db_management.py test
poetry run python app/db/db_management.py stats
poetry run python app/db/db_management.py backfill --symbols BTC ETH SOL --days 1095
poetry run python app/db/db_management.py backfill-stocks --symbols TCEHY COIN MSFT BTC-USD --days 1095
```

Crypto daily cache keys use forms such as `BTCUSDT__1d`; stocks use their ticker.
Three calendar years are normally about 1,095 crypto rows and about 750 US
stock sessions. Always set/check `DATABASE_URL` before interpreting `stats`;
earlier “no data” incidents came from querying a different/default database.

`clear` and `drop` are destructive. Never run them merely to troubleshoot an
empty query. First check the URL, resolved database file, symbol keys, row count,
and date range.

## Email behavior

Credentials live in `app/core/secret.ini`. Gmail must use a 16-character Google
App Password with 2-Step Verification, not the normal account password. A 535
BadCredentials response usually means the app password/account configuration is
wrong or stale; never paste the password into chat, logs, Git, or this file.

After a normal `--asset=all` run:

- the first address in `GMAIL_RECIPIENTS` is the admin and receives crypto plus
  stock summaries/recommendations;
- later recipients receive crypto-only recommendations;
- when there is no BUY/SELL at all, only the admin receives the NO ACTION
  heartbeat and process summary;
- a missing/invalid Gmail configuration can fail email delivery even when the
  simulations themselves succeeded.

## Verification commands

Focused defensive-strategy regression suite (32 tests passed on 2026-08-03):

```powershell
$env:DATABASE_URL = "sqlite:///./crypto_trading.db"
poetry run pytest -q `
  tests/strategy/test_held_stock_defensive_strategies.py `
  tests/backtesting/test_held_stock_defensive_validation.py `
  tests/strategy/test_tcehy_regime_defensive_strategy.py `
  tests/backtesting/test_tencent_defensive_validation.py `
  tests/strategy/test_altcoin_defensive_strategies.py `
  tests/backtesting/test_altcoin_defensive_validation.py `
  tests/strategy/test_sma200_strategy.py
```

Rebuild auditable reports:

```powershell
poetry run python app/backtesting/tencent_defensive_validation.py
poetry run python app/backtesting/held_stock_defensive_validation.py
```

Also run `python -m py_compile` on edited modules and `git diff --check`. Run the
broader test suite when appropriate, but note that as of 2026-08-03 the legacy
full suite had known unrelated failures, chiefly old assumptions that
`STRATEGIES` still contains generic strategies and older mocks. Do not claim the
full suite is green without rerunning and examining it; do not weaken current
asset isolation merely to satisfy stale tests.

## Git and delivery conventions

- Development happens on `dev`; push to `origin/dev` when the user asks.
- Create/merge a PR to `master` only when requested. Admin bypass for GitHub PR
  approval is permission-sensitive and must be explicitly granted for that PR.
- After a requested merge, update local `master`, merge/synchronize `dev`, and
  settle on the branch the user requested (historically usually `dev`).
- Never stage `secret.ini`, databases, generated dashboards, or unrelated user
  files.
- Current continuity checkpoint (2026-08-03): `dev` and `origin/dev` point to
  `0973c60 add defensive stock strategies`. `master`/`origin/master` point to
  `96dc394` and do not yet include commits `d6c1bfb` or `0973c60`.
- Four old local research artifacts were explicitly left untracked and should
  not be deleted or committed without a new request:
  - `artifacts/backtests/altcoin_defensive_validation.json`
  - `artifacts/backtests/btc_sma200_defensive_registered_smoke.json`
  - `artifacts/backtests/portfolio_strategy_validation.json`
  - `artifacts/backtests/sma200_variant_all_8_assets.json`

## Keep this handoff current

Update `AGENTS.md` whenever a strategy is enabled/retired, frozen parameters or
cost assumptions change, a new profit gate becomes authoritative, the runtime
command/schedule changes, the database location changes, or the Git workflow
changes. Record important decisions and caveats in the repo rather than relying
on a single ChatGPT/Codex thread.
