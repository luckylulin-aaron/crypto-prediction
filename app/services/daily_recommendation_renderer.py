"""Plain-text and HTML rendering for daily trading recommendations."""

from datetime import datetime
from typing import Optional, Tuple

try:
    from core.config import OPTION_SIGNAL_HOLD_DAYS
    from core.logger import get_logger
    from utils.email_util import send_email
except ImportError:
    from app.core.config import OPTION_SIGNAL_HOLD_DAYS
    from app.core.logger import get_logger
    from app.utils.email_util import send_email

logger = get_logger(__name__)


def send_daily_recommendations_email(
    log_file,
    recipient_list,
    from_email,
    app_password,
    best_summaries: Optional[list] = None,
    pending_signals: Optional[list] = None,
):
    """
    Send daily recommendations email from log.txt for today's actions.

    Sends different content based on recipient:
    - First recipient (you): Gets all recommendations (crypto + stocks)
    - Other recipients: Gets only crypto recommendations (stocks filtered out)

    Args:
        log_file: path to the log file
        recipient_list: list of email addresses to send the email to
        from_email: email address to send the email from
        app_password: app password for the email account

    Returns:
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    best_summaries = best_summaries or []
    ledger_authoritative = pending_signals is not None
    pending_signals = pending_signals or []

    def _to_dt(x):
        try:
            if isinstance(x, datetime):
                return x
            return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

    def _fmt_pct(v) -> str:
        try:
            if v is None or v == "":
                return ""
            return f"{float(v):.2f}"
        except Exception:
            return str(v)

    def _fmt_int(v) -> str:
        try:
            if v is None or v == "":
                return ""
            return str(int(v))
        except Exception:
            return str(v)

    def _leverage_suggestion(action: str) -> Tuple[str, str, str]:
        try:
            act = str(action).upper()
        except Exception:
            act = ""
        if act == "BUY":
            return "CALL", "x3/x5", "10d"
        if act == "SELL":
            return "PUT", "x3/x5", "10d"
        return "", "", ""

    def _render_best_table(rows: list) -> Tuple[str, str]:
        """
        Return (plain_text, html) table with best strategy + signal frequency per asset.
        Expected keys:
          asset_type, exchange, asset, best_strategy, buy_pct, sell_pct,
          num_buy, num_sell, num_intervals, signal_rate_pct,
          signals_per_30d, avg_days_between_signals,
          last_buy_date, last_sell_date
          call_win_rate_pct, put_win_rate_pct, call_trials, put_trials
        """
        if not rows:
            return "", ""

        header = f"{'Type':<6} | {'Exchange':<8} | {'Asset':<8} | {'Best Strategy':<20} | {'Buy %':<6} | {'Sell %':<6} | {'BUY':<3} | {'SELL':<4} | {'Sig%':<5} | {'Sig/30d':<7} | {'AvgDays':<6} | {'Last BUY':<16} | {'Last SELL':<16} | {'CallWin%':<8} | {'PutWin%':<8}"
        sep = "-" * len(header)
        lines = []
        for r in rows:
            lines.append(
                f"{r.get('asset_type',''):<6} | {r.get('exchange',''):<8} | {r.get('asset',''):<8} | {r.get('best_strategy',''):<20} | {_fmt_pct(r.get('buy_pct')):<6} | {_fmt_pct(r.get('sell_pct')):<6} | {_fmt_int(r.get('num_buy')):<3} | {_fmt_int(r.get('num_sell')):<4} | {_fmt_pct(r.get('signal_rate_pct')):<5} | {_fmt_pct(r.get('signals_per_30d')):<7} | {_fmt_pct(r.get('avg_days_between_signals')):<6} | {str(r.get('last_buy_date','')):<16} | {str(r.get('last_sell_date','')):<16} | {_fmt_pct(r.get('call_win_rate_pct')):<8} | {_fmt_pct(r.get('put_win_rate_pct')):<8}"
            )
        plain = "\n".join([header, sep] + lines) + "\n"

        html = (
            '<table style="border-collapse:collapse;width:100%;margin-top:10px;">'
            "<thead>"
            '<tr style="background:#0f172a;color:#ffffff;">'
            '<th style="padding:10px;text-align:left;font-weight:600;">Type</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Exchange</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Asset</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Best strategy</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Buy %</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Sell %</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">BUY</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">SELL</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Sig%</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Sig/30d</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">AvgDays</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Last BUY</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Last SELL</th>'
            f'<th style="padding:10px;text-align:right;font-weight:600;" title="Call option win rate: BUY?SELL within {OPTION_SIGNAL_HOLD_DAYS}d, sell_price > buy_price">CallWin%</th>'
            f'<th style="padding:10px;text-align:right;font-weight:600;" title="Put option win rate: SELL?BUY within {OPTION_SIGNAL_HOLD_DAYS}d, buy_price < sell_price">PutWin%</th>'
            "</tr></thead><tbody>"
        )
        for idx, r in enumerate(rows):
            bg = "#f8fafc" if idx % 2 == 0 else "#ffffff"
            icon = "??" if r.get("asset_type") == "CRYPTO" else "??"
            html += (
                f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;">'
                f"<td style=\"padding:10px;color:#0f172a;\">{icon} {r.get('asset_type','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('exchange','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;font-weight:600;\">{r.get('asset','')}</td>"
                f"<td style=\"padding:10px;color:#334155;font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;\">{r.get('best_strategy','')}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('buy_pct'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('sell_pct'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_int(r.get('num_buy'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_int(r.get('num_sell'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('signal_rate_pct'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('signals_per_30d'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('avg_days_between_signals'))}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('last_buy_date','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('last_sell_date','')}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\" title=\"{_fmt_int(r.get('call_wins'))}/{_fmt_int(r.get('call_trials'))} wins\">{_fmt_pct(r.get('call_win_rate_pct'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\" title=\"{_fmt_int(r.get('put_wins'))}/{_fmt_int(r.get('put_trials'))} wins\">{_fmt_pct(r.get('put_win_rate_pct'))}</td>"
                "</tr>"
            )
        html += "</tbody></table>"
        return plain, html

    def _render_option_settlements(rows: list) -> Tuple[str, str]:
        """
        Return (plain_text, html) table for option settlements.
        Expected keys:
          asset_type, exchange, asset, option_type, leverage_multiple,
          entry_price, exit_price, pnl, settled_on
        """
        if not rows:
            return "", ""

        header = (
            f"{'Type':<6} | {'Exchange':<8} | {'Asset':<8} | {'Opt':<5} | {'Lev':<4} | "
            f"{'Entry':<10} | {'Exit':<10} | {'PnL':<10} | {'Settled':<19}"
        )
        sep = "-" * len(header)
        lines = []
        for r in rows:
            lines.append(
                f"{r.get('asset_type',''):<6} | {r.get('exchange',''):<8} | {r.get('asset',''):<8} | "
                f"{r.get('option_type',''):<5} | {str(r.get('leverage_multiple','')):<4} | "
                f"{_fmt_pct(r.get('entry_price')):<10} | {_fmt_pct(r.get('exit_price')):<10} | "
                f"{_fmt_pct(r.get('pnl')):<10} | {str(r.get('settled_on','')):<19}"
            )
        plain = "\n".join([header, sep] + lines) + "\n"

        html = (
            '<table style="border-collapse:collapse;width:100%;margin-top:10px;">'
            "<thead>"
            '<tr style="background:#0f172a;color:#ffffff;">'
            '<th style="padding:10px;text-align:left;font-weight:600;">Type</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Exchange</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Asset</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Option</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Lev</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Entry</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">Exit</th>'
            '<th style="padding:10px;text-align:right;font-weight:600;">PnL</th>'
            '<th style="padding:10px;text-align:left;font-weight:600;">Settled</th>'
            "</tr></thead><tbody>"
        )
        for idx, r in enumerate(rows):
            bg = "#f8fafc" if idx % 2 == 0 else "#ffffff"
            html += (
                f'<tr style="background:{bg};border-bottom:1px solid #e2e8f0;">'
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('asset_type','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('exchange','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;font-weight:600;\">{r.get('asset','')}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('option_type','')}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{r.get('leverage_multiple','')}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('entry_price'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('exit_price'))}</td>"
                f"<td style=\"padding:10px;text-align:right;color:#0f172a;\">{_fmt_pct(r.get('pnl'))}</td>"
                f"<td style=\"padding:10px;color:#0f172a;\">{r.get('settled_on','')}</td>"
                "</tr>"
            )
        html += "</tbody></table>"
        return plain, html

    buy_sell_lines = []
    no_action_entries = []

    # Find latest simulation time (best-effort) from the log footer:
    # `Finish job at time <timestamp>`
    latest_sim_time = None
    try:
        with open(log_file, "r") as f:
            all_log_lines = f.readlines()
        for line in reversed(all_log_lines):
            if "Finish job at time" in line:
                latest_sim_time = line.split("Finish job at time", 1)[1].strip()
                break
    except Exception as e:
        logger.warning(f"Could not parse latest simulation time from log: {e}")

    with open(log_file, "r") as infile:
        for line in infile:
            if line.strip() and line[:10] == today_str:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) == 6:
                    time, exch, asset, action, buy, sell = parts
                    action_val = action.replace("Action: ", "")
                    buy_val = buy.replace("Buy %: ", "")
                    sell_val = sell.replace("Sell %: ", "")
                    if (
                        action_val.upper() in ("BUY", "SELL")
                        and not ledger_authoritative
                    ):
                        buy_sell_lines.append(
                            (time, exch, asset, action_val, buy_val, sell_val)
                        )
                    else:
                        no_action_entries.append((exch, asset))
                else:
                    # fallback: treat as no action
                    no_action_entries.append(("?", "?"))

    # Pending ledger rows are authoritative for actionable reminders. Their time
    # is the signal candle date, not the date on which this process happens to run.
    for row in pending_signals:
        signal_dt = _to_dt(row.get("signal_date"))
        signal_time = (
            signal_dt.strftime("%Y-%m-%d %H:%M:%S")
            if signal_dt is not None
            else str(row.get("signal_date", ""))
        )
        buy_sell_lines.append(
            (
                signal_time,
                row.get("exchange", ""),
                row.get("asset", ""),
                row.get("action", ""),
                _fmt_pct(row.get("buy_percentage")),
                _fmt_pct(row.get("sell_percentage")),
            )
        )

    # If we have no parsed log entries for today, we can still send the best summary table
    # (when simulations ran in this process and provided `best_summaries`).
    if not buy_sell_lines and not no_action_entries and not best_summaries:
        logger.info("No trading actions found for today, skipping email notification.")
        return True

    # Separate crypto and stock recommendations
    crypto_lines = []
    stock_lines = []
    crypto_no_action = []
    stock_no_action = []

    # Filter buy/sell lines
    for line in buy_sell_lines:
        time, exch, asset, action, buy, sell = line
        if exch == "STOCK":
            stock_lines.append(line)
        else:
            crypto_lines.append(line)

    # Filter no action entries
    for exch, asset in no_action_entries:
        if exch == "STOCK":
            stock_no_action.append((exch, asset))
        else:
            crypto_no_action.append((exch, asset))

    # If there are NO BUY/SELL recommendations at all, mute notifications for non-admin recipients
    # to avoid spamming with "NO ACTION" emails. We still send a heartbeat to the first recipient
    # (admin) so you can see the latest simulation time and confirm the bot is running.
    if not buy_sell_lines:
        if not recipient_list:
            logger.info("Recipient list is empty; skipping email notification.")
            return False

        admin_recipient = recipient_list[0]
        sim_ts = latest_sim_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"Latest simulation time: {sim_ts}\n\nNo BUY/SELL recommendations today.\n\n"
        html_body = (
            "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, Arial, sans-serif;\">"
            f'<div style="font-size:14px;color:#334155;">?? Latest simulation time: <b>{sim_ts}</b></div>'
            '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? No BUY/SELL recommendations today</h2>'
        )

        if no_action_entries:
            from collections import defaultdict

            exch_assets = defaultdict(list)
            for exch, asset in no_action_entries:
                exch_assets[exch].append(asset)
            body += "No action recommended for the following assets today:\n"
            for exch, assets in exch_assets.items():
                asset_list = ", ".join(sorted(set(assets)))
                body += f"- {exch}: {asset_list}\n"
            html_body += '<div style="margin-top:10px;font-size:13px;color:#334155;"><b>No action</b> for:</div>'
            html_body += (
                '<ul style="margin:6px 0 0 18px;color:#334155;font-size:13px;">'
            )
            for exch, assets in exch_assets.items():
                asset_list = ", ".join(sorted(set(assets)))
                html_body += f"<li><b>{exch}</b>: {asset_list}</li>"
            html_body += "</ul>"

        if best_summaries:
            summary_rows = sorted(
                list(best_summaries),
                key=lambda r: (r.get("asset_type") != "STOCK", r.get("asset", "")),
            )
            body += "\nBest strategy + signal frequency (per asset):\n"
            plain_tbl, html_tbl = _render_best_table(summary_rows)
            body += plain_tbl + "\n"
            html_body += '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Strategy + signal frequency</h2>'
            html_body += html_tbl
        html_body += "</div>"

        subject = f"Daily Trading Bot Recommendations ({today_str}) - NO ACTION"
        sent = send_email(
            subject=subject,
            body=body.strip(),
            to_emails=[admin_recipient],
            from_email=from_email,
            app_password=app_password,
            html_body=html_body,
        )
        if sent:
            logger.info(
                "No BUY/SELL recommendations today; sent heartbeat to admin only and muted other recipients."
            )
        else:
            logger.error("Failed to deliver NO ACTION heartbeat to admin.")
        return bool(sent)

    # Send different emails to different recipients
    if not recipient_list:
        logger.info("Recipient list is empty; pending signals remain undelivered.")
        return False
    admin_sent = False
    for i, recipient in enumerate(recipient_list):
        sim_ts = latest_sim_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"Latest simulation time: {sim_ts}\n\n"
        html_body = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,Arial,sans-serif;\">"
            f'<div style="font-size:14px;color:#334155;">?? Latest simulation time: <b>{sim_ts}</b></div>'
        )

        if i == 0:
            # First recipient (admin) - gets everything
            all_lines = crypto_lines + stock_lines
            all_no_action = crypto_no_action + stock_no_action

            summary_rows = sorted(
                list(best_summaries),
                key=lambda r: (r.get("asset_type") != "STOCK", r.get("asset", "")),
            )
            if summary_rows:
                body += "Best strategy summary (per asset):\n"
                plain_tbl, html_tbl = _render_best_table(summary_rows)
                body += plain_tbl + "\n"
                html_body += (
                    '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Best strategy summary (per asset)</h2>'
                    + html_tbl
                )

            if all_lines:
                header = (
                    f"{'Time':<19} | {'Exchange':<8} | {'Asset':<8} | {'Action':<10} | "
                    f"{'Buy %':<6} | {'Sell %':<6} | {'Option':<6} | {'Lev':<5} | {'Exp':<4}"
                )
                sep = "-" * len(header)
                formatted_lines = []
                for t, e, a, ac, b, s in all_lines:
                    opt, lev, exp = _leverage_suggestion(ac)
                    formatted_lines.append(
                        f"{t:<19} | {e:<8} | {a:<8} | {ac:<10} | {b:<6} | {s:<6} | {opt:<6} | {lev:<5} | {exp:<4}"
                    )
                body += f"{header}\n{sep}\n" + "\n".join(formatted_lines) + "\n"
                body += (
                    "\nBuy %: Recommended proportion of available funds to use for buying this asset.\n"
                    "Sell %: Recommended proportion of current holdings of this asset to sell.\n"
                    "Option/Lev/Exp: Suggested option type, leverage, and max expiration horizon.\n\n"
                )
                html_body += '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Today\'s recommendations</h2>'
                html_body += (
                    "<pre style=\"font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
                    'font-size:12px;white-space:pre;background:#0b1020;color:#e2e8f0;padding:12px;border-radius:10px;">'
                    + f"{header}\n{sep}\n"
                    + "\n".join(formatted_lines)
                    + "</pre>"
                )

            option_rows = []
            for r in summary_rows:
                for opt in r.get("option_settlements", []) or []:
                    option_rows.append(opt)
            if option_rows:
                body += "Option settlements:\n"
                plain_tbl, html_tbl = _render_option_settlements(option_rows)
                body += plain_tbl + "\n"
                html_body += (
                    '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Option settlements</h2>'
                    + html_tbl
                )

            if all_no_action:
                from collections import defaultdict

                exch_assets = defaultdict(list)
                for exch, asset in all_no_action:
                    exch_assets[exch].append(asset)
                body += "No action recommended for the following assets today:\n"
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    body += f"- {exch}: {asset_list}\n"

                html_body += '<div style="margin-top:12px;font-size:13px;color:#334155;"><b>? No action</b> for:</div><ul style="margin:6px 0 0 18px;color:#334155;font-size:13px;">'
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    html_body += f"<li><b>{exch}</b>: {asset_list}</li>"
                html_body += "</ul>"

        else:
            # Other recipients - crypto only
            if not crypto_lines:
                continue

            summary_rows = sorted(
                [r for r in best_summaries if r.get("asset_type") == "CRYPTO"],
                key=lambda r: r.get("asset", ""),
            )
            if summary_rows:
                body += "Best strategy summary (crypto only):\n"
                plain_tbl, html_tbl = _render_best_table(summary_rows)
                body += plain_tbl + "\n"
                html_body += (
                    '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Best strategy summary (crypto)</h2>'
                    + html_tbl
                )

            if crypto_lines:
                header = (
                    f"{'Time':<19} | {'Exchange':<8} | {'Asset':<8} | {'Action':<10} | "
                    f"{'Buy %':<6} | {'Sell %':<6} | {'Option':<6} | {'Lev':<5} | {'Exp':<4}"
                )
                sep = "-" * len(header)
                formatted_lines = []
                for t, e, a, ac, b, s in crypto_lines:
                    opt, lev, exp = _leverage_suggestion(ac)
                    formatted_lines.append(
                        f"{t:<19} | {e:<8} | {a:<8} | {ac:<10} | {b:<6} | {s:<6} | {opt:<6} | {lev:<5} | {exp:<4}"
                    )
                body += f"{header}\n{sep}\n" + "\n".join(formatted_lines) + "\n"
                body += (
                    "\nBuy %: Recommended proportion of available stablecoin to use for buying this asset.\n"
                    "Sell %: Recommended proportion of current holdings of this asset to sell.\n"
                    "Option/Lev/Exp: Suggested option type, leverage, and max expiration horizon.\n\n"
                )
                html_body += '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Today\'s recommendations</h2>'
                html_body += (
                    "<pre style=\"font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
                    'font-size:12px;white-space:pre;background:#0b1020;color:#e2e8f0;padding:12px;border-radius:10px;">'
                    + f"{header}\n{sep}\n"
                    + "\n".join(formatted_lines)
                    + "</pre>"
                )

            option_rows = []
            for r in summary_rows:
                for opt in r.get("option_settlements", []) or []:
                    option_rows.append(opt)
            if option_rows:
                body += "Option settlements:\n"
                plain_tbl, html_tbl = _render_option_settlements(option_rows)
                body += plain_tbl + "\n"
                html_body += (
                    '<h2 style="margin:14px 0 6px 0;font-size:18px;color:#0f172a;">?? Option settlements</h2>'
                    + html_tbl
                )

            if crypto_no_action:
                from collections import defaultdict

                exch_assets = defaultdict(list)
                for exch, asset in crypto_no_action:
                    exch_assets[exch].append(asset)
                body += "No action recommended for the following assets today:\n"
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    body += f"- {exch}: {asset_list}\n"

                html_body += '<div style="margin-top:12px;font-size:13px;color:#334155;"><b>? No action</b> for:</div><ul style="margin:6px 0 0 18px;color:#334155;font-size:13px;">'
                for exch, assets in exch_assets.items():
                    asset_list = ", ".join(sorted(set(assets)))
                    html_body += f"<li><b>{exch}</b>: {asset_list}</li>"
                html_body += "</ul>"

        subject = f"Daily Trading Bot Recommendations ({today_str})"
        html_body += "</div>"
        sent = send_email(
            subject=subject,
            body=body.strip(),
            to_emails=[recipient],
            from_email=from_email,
            app_password=app_password,
            html_body=html_body,
        )
        if i == 0:
            admin_sent = bool(sent)

    return admin_sent
