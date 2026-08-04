"""Complete daily crypto simulation runner and exchange data fallbacks."""

import os
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from core.config import *
    from core.logger import get_logger
    from trading.binance_client import BinanceClient
    from trading.cbpro_client import CBProClient
    from trading.trader_driver import TraderDriver
    from utils.util import (
        compute_option_signal_win_rates,
        display_port_msg,
        run_moving_window_simulation,
    )
    from visualization.visualization import (
        create_comprehensive_dashboard,
        create_moving_window_signals_report,
    )
except ImportError:
    from app.core.config import *
    from app.core.logger import get_logger
    from app.trading.binance_client import BinanceClient
    from app.trading.cbpro_client import CBProClient
    from app.trading.trader_driver import TraderDriver
    from app.utils.util import (
        compute_option_signal_win_rates,
        display_port_msg,
        run_moving_window_simulation,
    )
    from app.visualization.visualization import (
        create_comprehensive_dashboard,
        create_moving_window_signals_report,
    )

logger = get_logger(__name__)


def fetch_historical_data_with_fallback(
    asset: str,
    binance_client: BinanceClient,
    coinbase_client: CBProClient,
    exchange_configs: list,
    interval_hours: int = DATA_INTERVAL_HOURS,
    lookback_days: int = TIMESPAN,
):
    """
    Fetch historical data for an asset, trying Binance first, then falling back to Coinbase.

    Args:
        asset (str): The asset symbol (e.g., 'BTC', 'ETH')
        binance_client (BinanceClient): Binance client instance
        coinbase_client (CBProClient): Coinbase client instance
        exchange_configs (list): List of exchange configurations

    Returns:
        tuple: (data_stream, source_exchange_name) or (None, None) if both fail
    """

    def validate_and_format_data(data_stream):
        """
        Validate and format data to ensure it matches the expected format for trader_driver.
        Expected format: (price, date, open, low, high)
        """
        if not data_stream:
            return None

        formatted_data = []
        for item in data_stream:
            if len(item) >= 5:
                # Take only the first 5 elements: (price, date, open, low, high)
                formatted_item = (item[0], item[1], item[2], item[3], item[4])
                formatted_data.append(formatted_item)
            else:
                logger.warning(
                    f"Skipping data point with insufficient elements: {item}"
                )

        return formatted_data if formatted_data else None

    # Try Binance first
    binance_config = next(
        (
            config
            for config in exchange_configs
            if config["name"] == ExchangeName.BINANCE
        ),
        None,
    )
    if binance_config:
        try:
            binance_symbol = binance_config["symbol_format"](asset)
            logger.info(
                f"Attempting to fetch {asset} data from Binance using symbol: {binance_symbol}"
            )
            data_stream = binance_client.get_historic_data(
                binance_symbol,
                interval_hours=interval_hours,
                lookback_days=lookback_days,
            )

            # Validate and format the data
            formatted_data = validate_and_format_data(data_stream)
            if formatted_data and len(formatted_data) >= 2:
                logger.info(
                    f"Successfully fetched {len(formatted_data)} data points from Binance for {asset}"
                )
                return formatted_data, ExchangeName.BINANCE
            else:
                logger.warning(
                    f"Binance returned insufficient data for {asset}: {len(formatted_data) if formatted_data else 0} points"
                )
        except Exception as e:
            logger.warning(f"Failed to fetch {asset} data from Binance: {e}")

    # Fall back to Coinbase
    coinbase_config = next(
        (
            config
            for config in exchange_configs
            if config["name"] == ExchangeName.COINBASE
        ),
        None,
    )
    if coinbase_config:
        try:
            coinbase_symbol = coinbase_config["symbol_format"](asset)
            logger.info(
                f"Attempting to fetch {asset} data from Coinbase using symbol: {coinbase_symbol}"
            )
            data_stream = coinbase_client.get_historic_data(
                coinbase_symbol,
                interval_hours=interval_hours,
                lookback_days=lookback_days,
            )

            # Validate and format the data
            formatted_data = validate_and_format_data(data_stream)
            if formatted_data and len(formatted_data) >= 2:
                logger.info(
                    f"Successfully fetched {len(formatted_data)} data points from Coinbase for {asset}"
                )
                return formatted_data, ExchangeName.COINBASE
            else:
                logger.warning(
                    f"Coinbase returned insufficient data for {asset}: {len(formatted_data) if formatted_data else 0} points"
                )
        except Exception as e:
            logger.warning(f"Failed to fetch {asset} data from Coinbase: {e}")

    logger.error(f"Both Binance and Coinbase failed to provide data for {asset}")
    return None, None


def fetch_intraday_data_with_fallback(
    asset: str,
    binance_client: BinanceClient,
    coinbase_client: CBProClient,
    exchange_configs: list,
    source_exchange: Optional[ExchangeName] = None,
    interval_hours: int = 1,
):
    """
    Fetch intraday data for an asset, preferring the source exchange when provided.

    Args:
        asset (str): The asset symbol (e.g., 'BTC', 'ETH')
        binance_client (BinanceClient): Binance client instance
        coinbase_client (CBProClient): Coinbase client instance
        exchange_configs (list): List of exchange configurations
        source_exchange (Optional[ExchangeName]): Preferred exchange to pull intraday data from.
        interval_hours (int): Intraday interval in hours. Defaults to 1.

    Returns:
        Optional[list]: Intraday data stream or None if not available.

    Raises:
        None
    """

    def validate_and_format_data(data_stream):
        if not data_stream:
            return None
        formatted_data = []
        for item in data_stream:
            if len(item) >= 5:
                formatted_item = (item[0], item[1], item[2], item[3], item[4])
                formatted_data.append(formatted_item)
        return formatted_data if formatted_data else None

    def _fetch_from_exchange(exchange_name: ExchangeName):
        if exchange_name == ExchangeName.BINANCE:
            cfg = next(
                (c for c in exchange_configs if c["name"] == ExchangeName.BINANCE), None
            )
            if not cfg:
                return None
            symbol = cfg["symbol_format"](asset)
            data = binance_client.get_historic_data(
                symbol, interval_hours=interval_hours
            )
            return validate_and_format_data(data)
        if exchange_name == ExchangeName.COINBASE:
            cfg = next(
                (c for c in exchange_configs if c["name"] == ExchangeName.COINBASE),
                None,
            )
            if not cfg:
                return None
            symbol = cfg["symbol_format"](asset)
            data = coinbase_client.get_historic_data(
                symbol, interval_hours=interval_hours
            )
            return validate_and_format_data(data)
        return None

    if source_exchange is not None:
        data = _fetch_from_exchange(source_exchange)
        if data:
            return data

    # Fallback: try both exchanges
    data = _fetch_from_exchange(ExchangeName.BINANCE)
    if data:
        return data
    return _fetch_from_exchange(ExchangeName.COINBASE)


def align_asset_stream_to_context(asset_stream, context_stream):
    """Drop asset candles whose calendar date has no completed context candle."""
    context_dates = {str(item[1])[:10] for item in context_stream or []}
    aligned = []
    dropped_dates = []
    for item in asset_stream or []:
        date_key = str(item[1])[:10]
        if date_key in context_dates:
            aligned.append(item)
        else:
            dropped_dates.append(date_key)
    return aligned, dropped_dates


class CryptoSimulationRunner:
    def __init__(self, *, logger, trader_driver_factory, simulation_service):
        self._logger = logger
        self._trader_driver_factory = trader_driver_factory
        self._simulation_service = simulation_service

    def run(
        self, *, all_actions, best_summaries, binance_client, coinbase_client, exchanges
    ) -> None:
        simulated_assets = set()

        # Display portfolio information for both exchanges
        self._logger.info("=== Portfolio Overview ===")
        for exchange in exchanges:
            self._logger.info(f"--- {exchange['name'].value} Portfolio ---")
            display_port_msg(
                v_c=exchange["crypto_value"],
                v_s=exchange["stablecoin_value"],
                before=True,
            )

        configured_assets = CURS[:1] if DEBUG else CURS
        asset_list = [
            candidate
            for candidate in configured_assets
            if crypto_strategies_for_asset(candidate)
        ]
        for asset in asset_list:
            asset_strategies = crypto_strategies_for_asset(asset)
            # Only simulate each asset once, regardless of exchange
            if asset in simulated_assets:
                self._logger.info(f"Skipping duplicate simulation for asset: {asset}")
                continue
            simulated_assets.add(asset)

            self._logger.info(f"\n\n# --- Simulating for asset: {asset} --- #")

            # Use fallback approach: try Binance first, then Coinbase
            data_stream, source_exchange = fetch_historical_data_with_fallback(
                asset,
                binance_client,
                coinbase_client,
                EXCHANGE_CONFIGS,
                interval_hours=CRYPTO_SIGNAL_INTERVAL_HOURS,
                lookback_days=CRYPTO_SIGNAL_LOOKBACK_DAYS,
            )

            if data_stream is None:
                self._logger.error(
                    f"Failed to fetch historical data for {asset} from both exchanges"
                )
                continue

            self._logger.info(f"Using data from {source_exchange.value} for {asset}")

            btc_data_stream = None
            if SOL_30D_BREAKOUT_DEFENSIVE_STRATEGY in asset_strategies:
                btc_data_stream, btc_source_exchange = (
                    fetch_historical_data_with_fallback(
                        "BTC",
                        binance_client,
                        coinbase_client,
                        EXCHANGE_CONFIGS,
                        interval_hours=CRYPTO_SIGNAL_INTERVAL_HOURS,
                        lookback_days=CRYPTO_SIGNAL_LOOKBACK_DAYS,
                    )
                )
                if not btc_data_stream:
                    self._logger.error(
                        "SOL defensive strategy requires aligned BTC daily data"
                    )
                    continue
                self._logger.info(
                    f"Using BTC regime data from {btc_source_exchange.value} for SOL"
                )

                data_stream, dropped_dates = align_asset_stream_to_context(
                    data_stream, btc_data_stream
                )
                if dropped_dates:
                    self._logger.warning(
                        f"Dropped {len(dropped_dates)} SOL candle(s) without same-day "
                        f"completed BTC context; first dropped date: {dropped_dates[0]}"
                    )
                if len(data_stream) < 200:
                    self._logger.error("Insufficient aligned SOL/BTC daily history")
                    continue
            intraday_stream = None
            if "MA-BOLL-BANDS" in asset_strategies and MA_BOLL_ZOOM_IN:
                try:
                    intraday_stream = fetch_intraday_data_with_fallback(
                        asset=asset,
                        binance_client=binance_client,
                        coinbase_client=coinbase_client,
                        exchange_configs=EXCHANGE_CONFIGS,
                        source_exchange=source_exchange,
                        interval_hours=MA_BOLL_ZOOM_IN_INTRADAY_HOURS,
                    )
                    if intraday_stream:
                        self._logger.info(
                            f"Fetched {len(intraday_stream)} intraday candles ({MA_BOLL_ZOOM_IN_INTRADAY_HOURS}h) "
                            f"for {asset} from {source_exchange.value}"
                        )
                except Exception as e:
                    self._logger.warning(f"Intraday fetch failed for {asset}: {e}")

            # Use the source exchange for wallet and portfolio data
            source_exchange_config = next(
                (config for config in exchanges if config["name"] == source_exchange),
                None,
            )
            if not source_exchange_config:
                self._logger.error(
                    f"Could not find configuration for {source_exchange.value}"
                )
                continue

            # Performance simulation must be independent of the live wallet. Using
            # a fallback 1-coin position made returns depend on the asset's starting
            # price and did not match the frozen validation's capital convention.
            simulation_initial_cash = CRYPTO_SIMULATION_INITIAL_CASH
            simulation_initial_coin = CRYPTO_SIMULATION_INITIAL_COIN
            self._logger.info(
                f"Using standardized simulation capital for {asset}: "
                f"cash=${simulation_initial_cash:.2f}, coin={simulation_initial_coin:.1f}"
            )

            # Run simulation
            # simulation configuration
            if DEBUG:
                SIM_BUY_PCTS = [BUY_PCTS[0]]
                SIM_SELL_PCTS = [SELL_PCTS[0]]
            else:
                SIM_BUY_PCTS = BUY_PCTS
                SIM_SELL_PCTS = SELL_PCTS
            try:
                # Validate data stream before creating trader driver
                if not data_stream:
                    self._logger.error(f"No historical data available for {asset}")
                    continue

                if len(data_stream) < 200:
                    self._logger.error(
                        f"Insufficient daily history for {asset}: {len(data_stream)} rows; SMA200 needs at least 200"
                    )
                    continue

                # The strategy was validated on daily candles. Use all fetched rows as one
                # evaluation window so SMA200 receives a complete warmup period.
                data_points_per_day = 24 / CRYPTO_SIGNAL_INTERVAL_HOURS
                window_size_data_points = len(data_stream)
                step_size_data_points = len(data_stream)

                self._logger.info(
                    f"Starting moving window simulation for {asset} using {source_exchange.value} data "
                    f"with {len(data_stream)} daily data points (single full-history evaluation window)"
                )
                self._logger.info(
                    f"Data interval configuration: {CRYPTO_SIGNAL_INTERVAL_HOURS}h intervals, "
                    f"~{data_points_per_day:.2f} data points per day, "
                    f"total data span covers ~{len(data_stream) / data_points_per_day:.1f} days"
                )

                # Run moving window simulation
                moving_window_results = run_moving_window_simulation(
                    trader_driver_class=TraderDriver,
                    data_stream=data_stream,
                    window_size=window_size_data_points,
                    step_size=step_size_data_points,
                    name=asset,
                    init_amount=simulation_initial_cash,
                    cur_coin=simulation_initial_coin,
                    # only test 1 strategy for debugging purposes
                    overall_stats=asset_strategies,
                    tol_pcts=TOL_PCTS,
                    ma_lengths=MA_LENGTHS,
                    ema_lengths=EMA_LENGTHS,
                    bollinger_mas=BOLLINGER_MAS,
                    bollinger_tols=BOLLINGER_TOLS,
                    buy_pcts=SIM_BUY_PCTS,
                    sell_pcts=SIM_SELL_PCTS,
                    buy_stas=BUY_STAS,
                    sell_stas=SELL_STAS,
                    rsi_periods=RSI_PERIODS,
                    rsi_oversold_thresholds=RSI_OVERSOLD_THRESHOLDS,
                    rsi_overbought_thresholds=RSI_OVERBOUGHT_THRESHOLDS,
                    kdj_oversold_thresholds=KDJ_OVERSOLD_THRESHOLDS,
                    kdj_overbought_thresholds=KDJ_OVERBOUGHT_THRESHOLDS,
                    mode="normal",
                    execute_on_next_open=CRYPTO_EXECUTE_ON_NEXT_OPEN,
                    slippage_bps=CRYPTO_SLIPPAGE_BPS,
                    enable_options=False,
                    btc_data_stream=btc_data_stream,
                )

                # Get aggregated metrics for best strategy
                best_strategy = moving_window_results["best_strategy"]
                best_metrics = moving_window_results["best_strategy_metrics"]
                best_window_result = moving_window_results["best_window_result"]

                # Create a TraderDriver with full data to get the current signal
                # Use the most recent data for signal generation
                trader_driver = self._trader_driver_factory.create(
                    name=asset,
                    initial_cash=simulation_initial_cash,
                    initial_coin=simulation_initial_coin,
                    strategies=asset_strategies,
                    buy_pcts=SIM_BUY_PCTS,
                    sell_pcts=SIM_SELL_PCTS,
                    zoom_in=MA_BOLL_ZOOM_IN,
                    zoom_in_min_move_pct=MA_BOLL_ZOOM_IN_MIN_MOVE_PCT,
                    ma_boll_simplify=MA_BOLL_SIMPLIFY,
                    execute_on_next_open=CRYPTO_EXECUTE_ON_NEXT_OPEN,
                    slippage_bps=CRYPTO_SLIPPAGE_BPS,
                    enable_options=False,
                    btc_data_stream=btc_data_stream,
                )
                trader_driver.feed_data(
                    data_stream,
                    intraday_stream=intraday_stream,
                    intraday_interval_hours=MA_BOLL_ZOOM_IN_INTRADAY_HOURS,
                )
                selection = self._simulation_service.select_and_record(
                    trader_driver=trader_driver,
                    asset_type="CRYPTO",
                    exchange=source_exchange.value,
                    asset=asset,
                    data_stream=data_stream,
                )
                best_t, signal, best_info = (
                    selection.trader,
                    selection.signal,
                    selection.best_info,
                )
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
                                dt0 = datetime.strptime(str(d0), "%Y-%m-%d %H:%M:%S")
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
                            "asset_type": "CRYPTO",
                            "exchange": source_exchange.value,
                            "asset": asset,
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
                        "asset_type": "CRYPTO",
                        "exchange": source_exchange.value,
                        "asset": asset,
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

                # Log aggregated best trader summary with exchange name
                self._logger.info(
                    f"\n{'★'*10} MOVING WINDOW SIMULATION RESULTS ({source_exchange.value}) {'★'*10}\n"
                    f"Total windows analyzed: {moving_window_results['num_windows']}\n"
                    f"Window size: {MOVING_WINDOW_DAYS} days ({window_size_data_points} data points at {DATA_INTERVAL_HOURS}h intervals)\n"
                    f"Best strategy (aggregated): {best_strategy}\n"
                    f"\n--- Aggregated Performance Metrics ---\n"
                    f"Mean rate of return: {best_metrics['mean_rate_of_return']:.2f}%\n"
                    f"Std dev of return: {best_metrics['std_rate_of_return']:.2f}%\n"
                    f"Min rate of return: {best_metrics['min_rate_of_return']:.2f}%\n"
                    f"Max rate of return: {best_metrics['max_rate_of_return']:.2f}%\n"
                    f"Median rate of return: {best_metrics['median_rate_of_return']:.2f}%\n"
                    f"Risk-adjusted return (mean - std): {best_metrics['risk_adjusted_return']:.2f}%\n"
                    f"Win rate: {best_metrics['win_rate']*100:.1f}%\n"
                    f"Mean baseline rate: {best_metrics['mean_baseline_rate']:.2f}%\n"
                    f"Mean coin rate: {best_metrics['mean_coin_rate']:.2f}%\n"
                    f"Mean max drawdown: {best_metrics['mean_drawdown']:.2f}%\n"
                    f"Mean transactions: {best_metrics['mean_transactions']:.1f}\n"
                    f"\n--- Best Window Performance ---\n"
                    f"Best window period: {best_window_result['window_start_date']} to {best_window_result['window_end_date']}\n"
                    f"Best window rate of return: {best_window_result['rate_of_return']:.2f}%\n"
                    f"Today's signal: {signal} for crypto={best_t.crypto_name}\n"
                    f"{'★'*50}\n"
                )

                # Save visualizations
                strategy_performance = trader_driver.get_all_strategy_performance()
                dashboard_filename = f"app/visualization/plots/trading_dashboard_{asset}_{source_exchange.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                create_comprehensive_dashboard(
                    trader_instance=best_t,
                    save_html=True,
                    filename=dashboard_filename,
                    strategy_performance=strategy_performance,
                )

                # Save moving-window buy/sell plots as a single stacked HTML per asset
                try:
                    plots_dir = os.path.join("app", "visualization", "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    window_report_filename = os.path.join(
                        plots_dir,
                        f"moving_window_signals_{asset}_{source_exchange.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    )
                    create_moving_window_signals_report(
                        asset_name=asset,
                        window_chart_data=moving_window_results.get(
                            "window_chart_data", []
                        ),
                        filename=window_report_filename,
                        title=f"Moving Window Buy/Sell Signal Report - {asset} ({source_exchange.value})",
                    )
                except Exception as e:
                    self._logger.error(
                        f"Failed to create moving window signals report for {asset}: {e}"
                    )

                # Gather recommended action for email/log
                action_line = f"{datetime.now()} | {source_exchange.value} | {asset} | Action: {signal['action']} | Buy %: {signal.get('buy_percentage', '')} | Sell %: {signal.get('sell_percentage', '')}"
                all_actions.append(action_line)

                # Log recommended action to log.txt
                with open(LOG_FILE, "a") as outfile:
                    outfile.write(action_line + "\n")

            except ValueError as e:
                self._logger.error(
                    f"Data validation failed for {asset} using {source_exchange.value}: {e}"
                )
            except Exception as e:
                self._logger.error(
                    f"Simulation failed for {asset} using {source_exchange.value}: {e}"
                )
