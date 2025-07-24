import os
from typing import Dict, List, Optional

import requests

from app.core.logger import get_logger
from app.utils.email_util import send_email

logger = get_logger(__name__)

class DefiEventClient:
    """
    Client for fetching DEFI protocols' market cap and TVL, calculating R, and sending signals.
    """
    
    DEFI_LLAMA_PROTOCOLS_URL = "https://api.llama.fi/protocols"

    def __init__(self):
        pass

    def fetch_protocols(self) -> Optional[List[Dict]]:
        """
        Fetch all DEFI protocols from DefiLlama.

        Returns:
            List of protocol dicts, or None if error.
        """
        try:
            response = requests.get(self.DEFI_LLAMA_PROTOCOLS_URL, timeout=20)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched {len(data)} protocols from DefiLlama.")
            return data
        except Exception as e:
            logger.error(f"Error fetching protocols from DefiLlama: {e}")
            return None

    def process_protocols(self, protocols: List[Dict]) -> List[Dict]:
        """
        For each protocol, calculate R and generate a signal.

        Args:
            protocols: List of protocol dicts from DefiLlama.
        Returns:
            List of dicts with name, symbol, market cap, TVL, R, and signal.
        """
        results = []
        for p in protocols:
            name = p.get("name")
            symbol = p.get("symbol") or "-"
            tvl = p.get("tvl")
            mcap = p.get("mcap")
            # Some protocols may not have both values
            if tvl is None or mcap is None or tvl == 0:
                continue
            try:
                r = mcap / tvl
            except Exception:
                continue
            if r < 1:
                signal = "UNDERVALUED (Buy/Hold)"
            elif r > 1:
                signal = "OVERVALUED (Sell)"
            else:
                signal = "FAIR VALUE"
            results.append({
                "name": name,
                "symbol": symbol,
                "market_cap": mcap,
                "tvl": tvl,
                "R": round(r, 3),
                "signal": signal
            })
        logger.info(f"Processed {len(results)} protocols with valid data.")
        return results

    def format_email_body(self, processed: List[Dict], top_n: int = 20) -> str:
        """
        Format a detailed email body for the top N undervalued and overvalued assets.

        Args:
            processed: List of processed protocol dicts.
            top_n: Number of assets to show for each signal.
        Returns:
            String email body.
        """
        undervalued = [p for p in processed if p["R"] < 1]
        overvalued = [p for p in processed if p["R"] > 1]
        undervalued = sorted(undervalued, key=lambda x: x["R"])[:top_n]
        overvalued = sorted(overvalued, key=lambda x: -x["R"])[:top_n]
        lines = ["DEFI Asset Valuation Signals (via DefiLlama)", ""]
        lines.append(f"Top {top_n} Undervalued (R < 1):")
        for p in undervalued:
            lines.append(f"- {p['name']} ({p['symbol']}): R={p['R']} | Market Cap=${int(p['market_cap']):,} | TVL=${int(p['tvl']):,} | {p['signal']}")
        lines.append("")
        lines.append(f"Top {top_n} Overvalued (R > 1):")
        for p in overvalued:
            lines.append(f"- {p['name']} ({p['symbol']}): R={p['R']} | Market Cap=${int(p['market_cap']):,} | TVL=${int(p['tvl']):,} | {p['signal']}")
        return "\n".join(lines)

    def run_and_email(self, to_emails: List[str], from_email: str, app_password: str, top_n: int = 20):
        """
        Fetch, process, and email the DEFI asset signals.

        Args:
            to_emails: List of recipient emails.
            from_email: Sender email.
            app_password: App password for sender email.
            top_n: Number of assets to show for each signal.
        """
        protocols = self.fetch_protocols()
        if not protocols:
            logger.error("No protocols fetched; aborting email.")
            return
        processed = self.process_protocols(protocols)
        if not processed:
            logger.error("No protocols processed; aborting email.")
            return
        body = self.format_email_body(processed, top_n=top_n)
        subject = "DEFI Asset Valuation Signals (Daily Report)"
        send_email(subject, body, to_emails, from_email, app_password)
        logger.info("DEFI asset signals email sent.")

if __name__ == "__main__":
    # Example usage: set credentials via environment or config
    TO_EMAILS = os.environ.get("DEFI_REPORT_TO_EMAILS", "").split(",")
    FROM_EMAIL = os.environ.get("DEFI_REPORT_FROM_EMAIL", "")
    APP_PASSWORD = os.environ.get("DEFI_REPORT_APP_PASSWORD", "")
    # Remove empty strings from TO_EMAILS
    TO_EMAILS = [e for e in TO_EMAILS if e]
    client = DefiEventClient()
    client.run_and_email(TO_EMAILS, FROM_EMAIL, APP_PASSWORD, top_n=20) 