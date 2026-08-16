"""Complete daily stock simulation runner."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

try:
    from core.config import *
    from utils.util import compute_option_signal_win_rates
    from visualization.visualization import create_comprehensive_dashboard
except ImportError:
    from app.core.config import *
    from app.utils.util import compute_option_signal_win_rates
    from app.visualization.visualization import create_comprehensive_dashboard


class StockSimulationRunner:
    def __init__(
        self,
        *,
        logger,
        market_client_factory,
        trader_driver_factory,
        simulation_service,
    ):
        self._logger = logger
        self._market_client_factory = market_client_factory
        self._trader_driver_factory = trader_driver_factory
        self._simulation_service = simulation_service

    def run(self, all_actions: list, best_summaries: Optional[list] = None) -> None:
        """
        Run stock simulation only (daily candles).

        Args:
            all_actions (list): List to append formatted action lines for logging/email.

        Returns:
            None
        """
        self._logger.info("\n" + "=" * 50)
        self._logger.info("STARTING STOCK TRADING SIMULATION")
        self._logger.info("=" * 50)

        # Check if it's a weekend day (Sunday or Monday) to skip stock simulation
        # US stock market is closed on weekends, and there's a one-day delay in data
        current_weekday = datetime.now().weekday()  # Monday=0, Sunday=6
        if current_weekday in [6, 0]:  # Sunday (6) or Monday (0)
            self._logger.info(
                f"Skipping stock simulation - current day is {'Sunday' if current_weekday == 6 else 'Monday'}"
            )
            self._logger.info(
                "US stock market is closed on weekends, and data has one-day delay"
            )
            return

        self._logger.info(
            f"Proceeding with stock simulation - current day is {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][current_weekday]}"
        )

        # Initialize US Stock Client
        stock_client = self._market_client_factory.stocks(STOCK_SIMULATION_ASSETS)

        # only 1 stock for debugging purposes
        stock_list = STOCK_SIMULATION_ASSETS[:1] if DEBUG else STOCK_SIMULATION_ASSETS

        # Simulate stock trading for each stock
        for stock in stock_list:
            self._logger.info(f"\n\n# --- Simulating for Stock: {stock} --- #")

            try:
                # Most assets use a dedicated three-year history. NFLX loads its
                # frozen pre-test history so the monthly regime can be reconstructed.
                end_date = datetime.now().strftime("%Y-%m-%d")
                history_lookback_days = (
                    NFLX_RUNTIME_HISTORY_LOOKBACK_DAYS
                    if stock == "NFLX"
                    else STOCK_HISTORY_LOOKBACK_DAYS
                )
                start_date = (
                    datetime.now() - timedelta(days=history_lookback_days)
                ).strftime("%Y-%m-%d")
                data_stream = stock_client.get_historic_data(
                    stock, start=start_date, end=end_date
                )
                self._logger.info(
                    f"Retrieved {len(data_stream)} data points for {stock} "
                    f"(last {history_lookback_days} days)"
                )

                # Validate data stream before creating trader driver
                if not data_stream:
                    self._logger.error(
                        f"No historical data available for stock {stock}"
                    )
                    continue

                if len(data_stream) < 2:
                    self._logger.error(
                        f"Insufficient historical data for stock {stock}: only {len(data_stream)} data points available"
                    )
                    continue

                warmup_points = 0
                performance_start_index = 0
                reporting_data_stream = data_stream
                if stock == "NFLX":
                    if len(data_stream) <= NFLX_RUNTIME_EVALUATION_ROWS:
                        self._logger.error(
                            "Insufficient NFLX pre-test history for monthly SMA state "
                            f"reconstruction: {len(data_stream)} rows"
                        )
                        continue
                    performance_start_index = (
                        len(data_stream) - NFLX_RUNTIME_EVALUATION_ROWS
                    )
                    warmup_points = performance_start_index - 1
                    reporting_data_stream = data_stream[performance_start_index:]
                    self._logger.info(
                        f"[NFLX] Warmup rows: {warmup_points}; evaluation rows: "
                        f"{len(reporting_data_stream)}"
                    )

                # For stock simulation, we'll use a fixed initial amount
                # You can modify this based on your stock portfolio value
                initial_stock_amount = 10000  # $10,000 initial investment
                current_stock_amount = 0  # Assume no current holdings for simulation

                # Run simulation for stocks
                if DEBUG:
                    SIM_BUY_PCTS = [BUY_PCTS[0]]
                    SIM_SELL_PCTS = [SELL_PCTS[0]]
                else:
                    SIM_BUY_PCTS = BUY_PCTS
                    SIM_SELL_PCTS = SELL_PCTS

                stock_strategies = stock_strategies_for_asset(stock)
                if not stock_strategies:
                    self._logger.info(
                        f"No enabled stock strategies are allowed for {stock}"
                    )
                    continue

                btc_data_stream = None
                if COIN_BTC_SMA200_DEFENSIVE_STRATEGY in stock_strategies:
                    context_symbol = COIN_BTC_SMA200_DEFENSIVE_PARAMETERS[
                        "context_symbol"
                    ]
                    btc_data_stream = stock_client.get_historic_data(
                        context_symbol, start=start_date, end=end_date
                    )
                    if not btc_data_stream:
                        self._logger.error(
                            f"{COIN_BTC_SMA200_DEFENSIVE_STRATEGY} requires "
                            f"{context_symbol} daily context data"
                        )
                        continue
                    self._logger.info(
                        f"Retrieved {len(btc_data_stream)} {context_symbol} daily "
                        f"context points for {stock}"
                    )

                trader_driver = self._trader_driver_factory.create(
                    name=stock,
                    initial_cash=initial_stock_amount,
                    initial_coin=current_stock_amount,
                    strategies=(
                        stock_strategies if DEBUG is not True else stock_strategies[:5]
                    ),
                    buy_pcts=SIM_BUY_PCTS,
                    sell_pcts=SIM_SELL_PCTS,
                    btc_data_stream=btc_data_stream,
                )
                trader_driver.feed_data(
                    data_stream,
                    warmup_points=warmup_points,
                    performance_start_index=performance_start_index,
                )
                selection = self._simulation_service.select_and_record(
                    trader_driver=trader_driver,
                    asset_type="STOCK",
                    exchange="STOCK",
                    asset=stock,
                    data_stream=reporting_data_stream,
                )
                best_t, signal, best_info = (
                    selection.trader,
                    selection.signal,
                    selection.best_info,
                )
                if best_summaries is not None:
                    # Signal frequency stats from the best trader's full trade history
                    th = getattr(best_t, "trade_history", []) or []
                    num_buy = len(
                        [x for x in th if str(x.get("action", "")).upper() == "BUY"]
                    )
                    num_sell = len(
                        [x for x in th if str(x.get("action", "")).upper() == "SELL"]
                    )
                    num_intervals = len(th)
                    signal_rate_pct = (
                        100.0 * (num_buy + num_sell) / num_intervals
                        if num_intervals > 0
                        else 0.0
                    )
                    # Time-based stats (more interpretable)
                    sig_dates = [
                        x.get("date")
                        for x in th
                        if str(x.get("action", "")).upper() in ("BUY", "SELL")
                    ]
                    sig_dates_dt = []
                    for d0 in sig_dates:
                        try:
                            dt0 = (
                                d0
                                if isinstance(d0, datetime)
                                else datetime.fromisoformat(str(d0))
                            )
                        except Exception:
                            dt0 = None
                        if dt0 is not None:
                            sig_dates_dt.append(dt0)
                    sig_dates_dt = sorted(sig_dates_dt)

                    span_days = 0.0
                    if th:
                        try:
                            d_start = th[0].get("date")
                            d_end = th[-1].get("date")
                            dt_start = (
                                d_start
                                if isinstance(d_start, datetime)
                                else datetime.fromisoformat(str(d_start))
                            )
                            dt_end = (
                                d_end
                                if isinstance(d_end, datetime)
                                else datetime.fromisoformat(str(d_end))
                            )
                            span_days = max(
                                0.0, (dt_end - dt_start).total_seconds() / 86400.0
                            )
                        except Exception:
                            span_days = float(num_intervals)

                    num_signals = num_buy + num_sell
                    signals_per_30d = (
                        (num_signals / span_days * 30.0) if span_days > 0 else 0.0
                    )
                    avg_days_between = ""
                    if len(sig_dates_dt) >= 2:
                        deltas = [
                            (sig_dates_dt[i] - sig_dates_dt[i - 1]).total_seconds()
                            / 86400.0
                            for i in range(1, len(sig_dates_dt))
                        ]
                        avg_days_between = float(np.mean(deltas)) if deltas else ""

                    def _fmt_last_dt(dt_obj) -> str:
                        if dt_obj is None:
                            return ""
                        try:
                            if dt_obj.time() == datetime.min.time():
                                return dt_obj.strftime("%Y-%m-%d")
                            return dt_obj.strftime("%Y-%m-%d %H:%M")
                        except Exception:
                            return str(dt_obj)

                    def _latest_action_dt(action: str):
                        for evt in reversed(th):
                            if str(evt.get("action", "")).upper() != action:
                                continue
                            d0 = evt.get("date")
                            try:
                                dt0 = (
                                    d0
                                    if isinstance(d0, datetime)
                                    else datetime.fromisoformat(
                                        str(d0).replace("Z", "+00:00")
                                    )
                                )
                            except Exception:
                                try:
                                    dt0 = datetime.strptime(
                                        str(d0), "%Y-%m-%d %H:%M:%S"
                                    )
                                except Exception:
                                    dt0 = None
                            if dt0 is not None:
                                return dt0
                        return None

                    last_buy_dt = _latest_action_dt("BUY")
                    last_sell_dt = _latest_action_dt("SELL")

                    opt_stats = compute_option_signal_win_rates(
                        trade_history=th,
                        hold_days=OPTION_SIGNAL_HOLD_DAYS,
                    )
                    opt_settlements = []
                    for opt in getattr(best_t, "option_history", []) or []:
                        if not opt.get("settled"):
                            continue
                        settled_on = opt.get("settled_on")
                        if isinstance(settled_on, datetime):
                            settled_str = settled_on.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            settled_str = str(settled_on) if settled_on else ""
                        opt_settlements.append(
                            {
                                "asset_type": "STOCK",
                                "exchange": "STOCK",
                                "asset": stock,
                                "option_type": opt.get("option_type", ""),
                                "leverage_multiple": opt.get("leverage_multiple", ""),
                                "entry_price": opt.get("entry_price", ""),
                                "exit_price": opt.get("exit_price", ""),
                                "pnl": opt.get("pnl", ""),
                                "settled_on": settled_str,
                            }
                        )
                    opt_settlements = opt_settlements[-5:]
                    best_summaries.append(
                        {
                            "asset_type": "STOCK",
                            "exchange": "STOCK",
                            "asset": stock,
                            "best_strategy": best_t.high_strategy,
                            "buy_pct": best_info.get("buy_pct", ""),
                            "sell_pct": best_info.get("sell_pct", ""),
                            "num_buy": num_buy,
                            "num_sell": num_sell,
                            "num_intervals": num_intervals,
                            "signal_rate_pct": round(signal_rate_pct, 2),
                            "signals_per_30d": round(signals_per_30d, 2),
                            "avg_days_between_signals": (
                                round(avg_days_between, 2)
                                if isinstance(avg_days_between, (int, float))
                                else ""
                            ),
                            "last_buy_date": _fmt_last_dt(last_buy_dt),
                            "last_sell_date": _fmt_last_dt(last_sell_dt),
                            "call_win_rate_pct": opt_stats.get("call_win_rate_pct", ""),
                            "put_win_rate_pct": opt_stats.get("put_win_rate_pct", ""),
                            "call_trials": opt_stats.get("call_trials", 0),
                            "put_trials": opt_stats.get("put_trials", 0),
                            "call_wins": opt_stats.get("call_wins", 0),
                            "put_wins": opt_stats.get("put_wins", 0),
                            "option_settlements": opt_settlements,
                        }
                    )

                # Log best trader summary for stock
                init_value = best_info.get("init_value", 0)
                max_final_value = best_info.get("max_final_value", 0)
                self._logger.info(
                    f"\n{'★'*10} BEST TRADER SUMMARY (STOCK: {stock}) {'★'*10}\n"
                    f"Best trader performance:\n"
                    f"  Strategy Parameters:\n"
                    f"    - Buy percentage: {best_info.get('buy_pct', 'N/A')}\n"
                    f"    - Sell percentage: {best_info.get('sell_pct', 'N/A')}\n"
                    f"    - Tolerance percentage: {best_info.get('tol_pct', 'N/A')}\n"
                    f"    - Bollinger sigma: {best_info.get('bollinger_sigma', 'N/A')}\n"
                    f"    - Buy strategy: {best_info.get('buy', 'N/A')}\n"
                    f"    - Sell strategy: {best_info.get('sell', 'N/A')}\n"
                    f"  Performance Metrics:\n"
                    f"    - Initial value: ${init_value:,.2f}\n"
                    f"    - Final value: ${max_final_value:,.2f}\n"
                    f"    - Rate of return: {best_info.get('rate_of_return', 'N/A')}\n"
                    f"    - Baseline rate of return: {best_info.get('baseline_rate_of_return', 'N/A')}\n"
                    f"    - Coin rate of return: {best_info.get('coin_rate_of_return', 'N/A')}\n"
                    f"  Trading Statistics:\n"
                    f"    - Max drawdown: {best_t.max_drawdown * 100:.2f}%\n"
                    f"    - Transactions: {best_t.num_transaction}\n"
                    f"    - Buys: {best_t.num_buy_action}, Sells: {best_t.num_sell_action}\n"
                    f"    - Strategy: {best_t.high_strategy}\n"
                    f"    - Today's signal: {signal} for stock={best_t.crypto_name}\n"
                    f"{'★'*36}\n"
                )

                # Save visualizations for stock
                strategy_performance = trader_driver.get_all_strategy_performance()
                dashboard_filename = f"app/visualization/plots/trading_dashboard_{stock}_STOCK_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                create_comprehensive_dashboard(
                    trader_instance=best_t,
                    save_html=True,
                    filename=dashboard_filename,
                    strategy_performance=strategy_performance,
                )

                # Gather recommended action for email/log
                action_line = f"{datetime.now()} | STOCK | {stock} | Action: {signal['action']} | Buy %: {signal.get('buy_percentage', '')} | Sell %: {signal.get('sell_percentage', '')}"
                all_actions.append(action_line)

                # Log recommended action to log.txt
                with open(LOG_FILE, "a") as outfile:
                    outfile.write(action_line + "\n")

            except ValueError as e:
                self._logger.error(f"Data validation failed for stock {stock}: {e}")
                continue
            except Exception as e:
                self._logger.error(f"Stock simulation failed for {stock}: {e}")
                continue
