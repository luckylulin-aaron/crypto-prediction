"""
Database models and connection management for cryptocurrency trading data.
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

try:
    from ..core.config import TIMESPAN
    from ..core.logger import get_logger
except ImportError:
    # Fallback for when running as script
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from core.config import TIMESPAN
    from core.logger import get_logger

# Create base class for declarative models
Base = declarative_base()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./crypto_trading.db"  # SQLite fallback for development
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class HistoricalData(Base):
    """Model for storing historical cryptocurrency price data."""

    __tablename__ = "historical_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)  # e.g., "BTC-USD"
    date = Column(DateTime, nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    class Config:
        indexes = [
            ("symbol", "date"),  # Composite index for efficient queries
        ]


class DataCache(Base):
    """Model for caching API request metadata."""

    __tablename__ = "data_cache"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    last_updated = Column(DateTime, nullable=False)
    data_count = Column(Integer, nullable=False, default=0)
    cache_key = Column(String(100), nullable=False, unique=True)  # symbol + date range
    created_at = Column(DateTime, default=datetime.utcnow)


class SignalLedger(Base):
    """Durable, idempotent BUY/SELL notification record."""

    __tablename__ = "signal_ledger"
    __table_args__ = (
        UniqueConstraint(
            "asset_type",
            "asset",
            "strategy",
            "strategy_version",
            "signal_date",
            "action",
            name="uq_signal_ledger_identity",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    asset_type = Column(String(20), nullable=False, index=True)
    exchange = Column(String(40), nullable=False)
    asset = Column(String(20), nullable=False, index=True)
    strategy = Column(String(100), nullable=False)
    strategy_version = Column(String(40), nullable=False, default="v1")
    signal_date = Column(DateTime, nullable=False, index=True)
    action = Column(String(10), nullable=False)
    buy_percentage = Column(Float, nullable=True)
    sell_percentage = Column(Float, nullable=True)
    delivery_status = Column(String(20), nullable=False, default="pending", index=True)
    delivery_attempts = Column(Integer, nullable=False, default=0)
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)


class SignalLedgerCheckpoint(Base):
    """Last completed candle scanned for one asset/strategy runtime."""

    __tablename__ = "signal_ledger_checkpoint"
    __table_args__ = (
        UniqueConstraint(
            "asset_type",
            "asset",
            "strategy",
            "strategy_version",
            name="uq_signal_ledger_checkpoint",
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    asset_type = Column(String(20), nullable=False)
    asset = Column(String(20), nullable=False)
    strategy = Column(String(100), nullable=False)
    strategy_version = Column(String(40), nullable=False, default="v1")
    last_evaluated_candle = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def get_db_session():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Manages database operations for historical trading data."""

    def __init__(self):
        self.logger = get_logger(__name__)
        # Don't create tables immediately - let user call init_database() when ready
        self._initialized = False

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=engine)
            self._initialized = True
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            self._initialized = False
            raise

    def ensure_initialized(self):
        """Ensure database is initialized before operations."""
        if not self._initialized:
            self._create_tables()

    def store_historical_data(self, symbol: str, data: List[List]) -> bool:
        """
        Store historical data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            data: List of [close_price, date_str, open_price, low, high, volume]

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            db = SessionLocal()

            # Convert data format
            records = []
            for item in data:
                close_price, date_str, open_price, low, high, volume = item
                # Try parsing with datetime format first, fallback to date-only format for backward compatibility
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        self.logger.error(f"Unable to parse date string: {date_str}")
                        continue

                record = HistoricalData(
                    symbol=symbol,
                    date=date,
                    open_price=open_price,
                    high_price=high,
                    low_price=low,
                    close_price=close_price,
                    volume=volume,
                )
                records.append(record)

            # Use upsert to avoid duplicates
            for record in records:
                existing = (
                    db.query(HistoricalData)
                    .filter(
                        HistoricalData.symbol == record.symbol,
                        HistoricalData.date == record.date,
                    )
                    .first()
                )

                if existing:
                    # Update existing record
                    existing.open_price = record.open_price
                    existing.high_price = record.high_price
                    existing.low_price = record.low_price
                    existing.close_price = record.close_price
                    existing.volume = record.volume
                    existing.updated_at = datetime.utcnow()
                else:
                    # Insert new record
                    db.add(record)

            db.commit()

            # Update cache metadata
            self._update_cache_metadata(symbol, len(records))

            self.logger.info(f"Stored {len(records)} records for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing historical data for {symbol}: {e}")
            db.rollback()
            return False
        finally:
            db.close()

    def get_historical_data(
        self, symbol: str, days: int = TIMESPAN
    ) -> Optional[List[List]]:
        """
        Retrieve historical data for a symbol from the database.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            days: Number of days of data to retrieve

        Returns:
            List of [close_price, date_str, open_price, low, high, volume] or None
        """
        try:
            db = SessionLocal()

            # Calculate date range - use current time for sub-daily intervals
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            if symbol.endswith(("__1d", "__ONE_DAY")):
                start_date = start_date.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

            # Query database
            records = (
                db.query(HistoricalData)
                .filter(
                    HistoricalData.symbol == symbol,
                    HistoricalData.date >= start_date,
                    HistoricalData.date < end_date,
                )
                .order_by(HistoricalData.date.asc())
                .all()
            )

            if not records:
                self.logger.info(f"No historical data found for {symbol}")
                return None

            # Convert to expected format
            data = []
            for record in records:
                # Use datetime format if time component exists, otherwise use date-only format
                if record.date.hour == 0 and record.date.minute == 0 and record.date.second == 0:
                    date_str = record.date.strftime("%Y-%m-%d")
                else:
                    date_str = record.date.strftime("%Y-%m-%d %H:%M:%S")
                data.append(
                    [
                        record.close_price,
                        date_str,
                        record.open_price,
                        record.low_price,
                        record.high_price,
                        record.volume,
                    ]
                )

            self.logger.info(
                f"Retrieved {len(data)} records for {symbol} from database"
            )
            return data

        except Exception as e:
            self.logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return None
        finally:
            db.close()

    def is_data_fresh(self, symbol: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is fresh enough to use.

        Args:
            symbol: Trading pair symbol
            max_age_hours: Maximum age in hours before data is considered stale

        Returns:
            bool: True if data is fresh, False otherwise
        """
        try:
            db = SessionLocal()

            cache_record = (
                db.query(DataCache).filter(DataCache.symbol == symbol).first()
            )

            if not cache_record:
                return False

            # Check if data is within max_age_hours
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            return cache_record.last_updated >= cutoff_time

        except Exception as e:
            self.logger.error(f"Error checking data freshness for {symbol}: {e}")
            return False
        finally:
            db.close()

    def _update_cache_metadata(self, symbol: str, data_count: int):
        """Update cache metadata for a symbol."""
        try:
            db = SessionLocal()

            cache_record = (
                db.query(DataCache).filter(DataCache.symbol == symbol).first()
            )

            if cache_record:
                cache_record.last_updated = datetime.utcnow()
                cache_record.data_count = data_count
            else:
                cache_record = DataCache(
                    symbol=symbol,
                    last_updated=datetime.utcnow(),
                    data_count=data_count,
                    cache_key=f"{symbol}_{datetime.utcnow().strftime('%Y%m%d')}",
                )
                db.add(cache_record)

            db.commit()

        except Exception as e:
            self.logger.error(f"Error updating cache metadata for {symbol}: {e}")
            db.rollback()
        finally:
            db.close()

    def get_data_statistics(
        self,
    ) -> List[Tuple[str, int, datetime, datetime]]:
        """
        Get statistics from the historical price records.

        Returns:
            List of (symbol, record_count, first_date, last_date) tuples.
        """
        try:
            db = SessionLocal()

            stats = (
                db.query(
                    HistoricalData.symbol,
                    func.count(HistoricalData.id),
                    func.min(HistoricalData.date),
                    func.max(HistoricalData.date),
                )
                .group_by(HistoricalData.symbol)
                .order_by(HistoricalData.symbol)
                .all()
            )

            return [(stat[0], int(stat[1]), stat[2], stat[3]) for stat in stats]

        except Exception as e:
            self.logger.error(f"Error getting data statistics: {e}")
            return []
        finally:
            db.close()

    def clear_old_data(self, days_to_keep: int = 365) -> int:
        """
        Clear historical data older than specified days.

        Args:
            days_to_keep: Number of days of data to keep

        Returns:
            int: Number of records deleted
        """
        try:
            db = SessionLocal()

            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            deleted_count = (
                db.query(HistoricalData)
                .filter(HistoricalData.date < cutoff_date)
                .delete()
            )

            db.commit()

            self.logger.info(f"Deleted {deleted_count} old records")
            return deleted_count

        except Exception as e:
            self.logger.error(f"Error clearing old data: {e}")
            db.rollback()
            return 0
        finally:
            db.close()

    def record_signal_events(
        self,
        *,
        asset_type: str,
        exchange: str,
        asset: str,
        strategy: str,
        strategy_version: str,
        latest_candle_date,
        events: list,
        bootstrap_days: int = 2,
    ) -> int:
        """Idempotently record new signal events since the prior candle checkpoint."""

        def _as_datetime(value):
            if isinstance(value, datetime):
                return value.replace(tzinfo=None)
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None)

        self.ensure_initialized()
        latest_candle = _as_datetime(latest_candle_date)
        db = SessionLocal()
        try:
            checkpoint = (
                db.query(SignalLedgerCheckpoint)
                .filter(
                    SignalLedgerCheckpoint.asset_type == asset_type,
                    SignalLedgerCheckpoint.asset == asset,
                    SignalLedgerCheckpoint.strategy == strategy,
                    SignalLedgerCheckpoint.strategy_version == strategy_version,
                )
                .first()
            )
            scan_after = (
                checkpoint.last_evaluated_candle
                if checkpoint is not None
                else latest_candle - timedelta(days=max(1, int(bootstrap_days)))
            )

            inserted = 0
            now = datetime.utcnow()
            for event in events or []:
                action = str(event.get("action", "")).upper()
                if action not in ("BUY", "SELL"):
                    continue
                signal_date = _as_datetime(event.get("date"))
                if signal_date < scan_after or signal_date > latest_candle:
                    continue

                existing = (
                    db.query(SignalLedger)
                    .filter(
                        SignalLedger.asset_type == asset_type,
                        SignalLedger.asset == asset,
                        SignalLedger.strategy == strategy,
                        SignalLedger.strategy_version == strategy_version,
                        SignalLedger.signal_date == signal_date,
                        SignalLedger.action == action,
                    )
                    .first()
                )
                if existing is not None:
                    existing.last_seen_at = now
                    continue

                db.add(
                    SignalLedger(
                        asset_type=asset_type,
                        exchange=exchange,
                        asset=asset,
                        strategy=strategy,
                        strategy_version=strategy_version,
                        signal_date=signal_date,
                        action=action,
                        buy_percentage=event.get("buy_percentage"),
                        sell_percentage=event.get("sell_percentage"),
                        first_seen_at=now,
                        last_seen_at=now,
                    )
                )
                inserted += 1

            if checkpoint is None:
                checkpoint = SignalLedgerCheckpoint(
                    asset_type=asset_type,
                    asset=asset,
                    strategy=strategy,
                    strategy_version=strategy_version,
                    last_evaluated_candle=latest_candle,
                    created_at=now,
                    updated_at=now,
                )
                db.add(checkpoint)
            elif latest_candle >= checkpoint.last_evaluated_candle:
                checkpoint.last_evaluated_candle = latest_candle
                checkpoint.updated_at = now

            db.commit()
            return inserted
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def get_pending_signals(self) -> list:
        """Return detached pending/failed signals in signal-date order."""
        self.ensure_initialized()
        db = SessionLocal()
        try:
            rows = (
                db.query(SignalLedger)
                .filter(SignalLedger.delivery_status != "delivered")
                .order_by(SignalLedger.signal_date.asc(), SignalLedger.id.asc())
                .all()
            )
            return [
                {
                    "id": row.id,
                    "asset_type": row.asset_type,
                    "exchange": row.exchange,
                    "asset": row.asset,
                    "strategy": row.strategy,
                    "strategy_version": row.strategy_version,
                    "signal_date": row.signal_date,
                    "action": row.action,
                    "buy_percentage": row.buy_percentage,
                    "sell_percentage": row.sell_percentage,
                    "delivery_attempts": row.delivery_attempts,
                    "last_error": row.last_error,
                }
                for row in rows
            ]
        finally:
            db.close()

    def mark_signal_delivery(self, signal_ids: list, *, success: bool, error=None) -> int:
        """Record one delivery attempt for the selected ledger rows."""
        ids = [int(signal_id) for signal_id in signal_ids or []]
        if not ids:
            return 0
        self.ensure_initialized()
        db = SessionLocal()
        try:
            rows = db.query(SignalLedger).filter(SignalLedger.id.in_(ids)).all()
            now = datetime.utcnow()
            for row in rows:
                row.delivery_attempts = int(row.delivery_attempts or 0) + 1
                row.last_seen_at = now
                if success:
                    row.delivery_status = "delivered"
                    row.delivered_at = now
                    row.last_error = None
                else:
                    row.delivery_status = "failed"
                    row.last_error = str(error or "email delivery failed")[:2000]
            db.commit()
            return len(rows)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


# Global database manager instance
db_manager = DatabaseManager()
