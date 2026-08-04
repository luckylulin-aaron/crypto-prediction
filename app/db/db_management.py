"""
Database management utilities for the cryptocurrency trading bot.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

from sqlalchemy import text

try:
    from database import Base, db_manager, engine

    from ..core.logger import get_logger
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.logger import get_logger
    from db.database import Base, db_manager, engine


def init_database():
    """Initialize the database by creating all tables."""
    logger = get_logger(__name__)
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def drop_database():
    """Drop all tables from the database."""
    logger = get_logger(__name__)
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        return False


def show_statistics():
    """Show database statistics."""
    logger = get_logger(__name__)
    try:
        stats = db_manager.get_data_statistics()
        if not stats:
            print("No historical data found in database.")
            return

        print("\n=== Database Statistics ===")
        print(f"{'Symbol':<20} {'Records':<10} {'First Date':<12} {'Last Date':<12}")
        print("-" * 58)

        for symbol, count, first_date, last_date in stats:
            print(
                f"{symbol:<20} {count:<10} "
                f"{first_date.strftime('%Y-%m-%d'):<12} "
                f"{last_date.strftime('%Y-%m-%d'):<12}"
            )

        print(f"\nTotal symbols: {len(stats)}")

    except Exception as e:
        logger.error(f"Failed to show statistics: {e}")


def clear_old_data(days: int = 365):
    """Clear historical data older than specified days."""
    logger = get_logger(__name__)
    try:
        deleted_count = db_manager.clear_old_data(days)
        print(f"Deleted {deleted_count} records older than {days} days")
        return True
    except Exception as e:
        logger.error(f"Failed to clear old data: {e}")
        return False


def backfill_daily_data(symbols=None, days: int = 1095):
    """Fetch completed daily candles and store them idempotently in the database."""
    logger = get_logger(__name__)

    try:
        from core.config import CURS
        from trading.binance_client import BinanceClient

        client = BinanceClient()
        requested_symbols = symbols or CURS
        failures = []

        for asset in requested_symbols:
            pair = asset.upper()
            if not pair.endswith("USDT"):
                pair = f"{pair}USDT"

            cache_key = f"{pair}__1d"
            try:
                data = client.get_historic_data(
                    pair,
                    use_cache=False,
                    interval_hours=24,
                    lookback_days=days,
                )
                if not data:
                    raise RuntimeError("Binance returned no completed daily candles")

                if not db_manager.store_historical_data(cache_key, data):
                    raise RuntimeError("database store operation failed")

                print(
                    f"{cache_key}: stored {len(data)} rows "
                    f"({data[0][1]} through {data[-1][1]})"
                )
            except Exception as exc:
                failures.append(pair)
                logger.error(f"Failed to backfill {pair}: {exc}")

        if failures:
            print(f"Backfill failed for: {', '.join(failures)}")
            return False

        print(f"Backfill completed for {len(requested_symbols)} symbols.")
        return True
    except Exception as exc:
        logger.error(f"Failed to backfill daily data: {exc}")
        return False


def backfill_stock_daily_data(symbols=None, days: int = 3 * 365):
    """Fetch stock daily candles from Yahoo Finance and upsert them by ticker/date."""
    logger = get_logger(__name__)
    if days <= 0:
        logger.error(f"Stock backfill days must be positive, got {days}")
        return False

    try:
        from core.config import STOCKS
        from trading.us_stock_client import USStockClient

        requested_symbols = [symbol.upper() for symbol in (symbols or STOCKS)]
        client = USStockClient(tickers=requested_symbols)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        failures = []

        for ticker in requested_symbols:
            try:
                data = client.get_historic_data(
                    ticker,
                    start=start_date,
                    end=end_date,
                    use_cache=False,
                )
                if not data:
                    raise RuntimeError("Yahoo Finance returned no daily candles")

                if not db_manager.store_historical_data(ticker, data):
                    raise RuntimeError("database store operation failed")

                print(
                    f"{ticker}: stored {len(data)} rows "
                    f"({data[0][1]} through {data[-1][1]})"
                )
            except Exception as exc:
                failures.append(ticker)
                logger.error(f"Failed to backfill stock {ticker}: {exc}")

        if failures:
            print(f"Stock backfill failed for: {', '.join(failures)}")
            return False

        print(f"Stock backfill completed for {len(requested_symbols)} symbols.")
        return True
    except Exception as exc:
        logger.error(f"Failed to backfill stock daily data: {exc}")
        return False


def test_connection():
    """Test database connection."""
    logger = get_logger(__name__)
    try:
        # Try to execute a simple query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        print("Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        print(f"Database connection failed: {e}")
        return False


def main():
    """Main function for database management CLI."""
    parser = argparse.ArgumentParser(
        description="Database management for crypto trading bot"
    )
    parser.add_argument(
        "command",
        choices=[
            "init",
            "drop",
            "stats",
            "clear",
            "test",
            "backfill",
            "backfill-stocks",
        ],
        help="Command to execute",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days (clear: 365; crypto/stock backfill: 1095)",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Tickers, assets, or USDT pairs to backfill (default: configured list)",
    )

    args = parser.parse_args()

    if args.command == "init":
        success = init_database()
        sys.exit(0 if success else 1)

    elif args.command == "drop":
        confirm = input("Are you sure you want to drop all tables? (y/N): ")
        if confirm.lower() == "y":
            success = drop_database()
            sys.exit(0 if success else 1)
        else:
            print("Operation cancelled.")
            sys.exit(0)

    elif args.command == "stats":
        show_statistics()
        sys.exit(0)

    elif args.command == "clear":
        success = clear_old_data(args.days or 365)
        sys.exit(0 if success else 1)

    elif args.command == "test":
        success = test_connection()
        sys.exit(0 if success else 1)

    elif args.command == "backfill":
        success = backfill_daily_data(args.symbols, args.days or 1095)
        sys.exit(0 if success else 1)

    elif args.command == "backfill-stocks":
        success = backfill_stock_daily_data(args.symbols, args.days or 3 * 365)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
