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
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.config import TIMESPAN
    from core.logger import get_logger

# Create base class for declarative models
Base = declarative_base()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./crypto_trading.db"  # SQLite fallback for development
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=StaticPool,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False  # Set to True for SQL debugging
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
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
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
                date = datetime.strptime(date_str, "%Y-%m-%d")
                
                record = HistoricalData(
                    symbol=symbol,
                    date=date,
                    open_price=open_price,
                    high_price=high,
                    low_price=low,
                    close_price=close_price,
                    volume=volume
                )
                records.append(record)
            
            # Use upsert to avoid duplicates
            for record in records:
                existing = db.query(HistoricalData).filter(
                    HistoricalData.symbol == record.symbol,
                    HistoricalData.date == record.date
                ).first()
                
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
    
    def get_historical_data(self, symbol: str, days: int = TIMESPAN) -> Optional[List[List]]:
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
            
            # Calculate date range
            end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=days)
            
            # Query database
            records = db.query(HistoricalData).filter(
                HistoricalData.symbol == symbol,
                HistoricalData.date >= start_date,
                HistoricalData.date < end_date
            ).order_by(HistoricalData.date.asc()).all()
            
            if not records:
                self.logger.info(f"No historical data found for {symbol}")
                return None
            
            # Convert to expected format
            data = []
            for record in records:
                date_str = record.date.strftime("%Y-%m-%d")
                data.append([
                    record.close_price,
                    date_str,
                    record.open_price,
                    record.low_price,
                    record.high_price,
                    record.volume
                ])
            
            self.logger.info(f"Retrieved {len(data)} records for {symbol} from database")
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
            
            cache_record = db.query(DataCache).filter(
                DataCache.symbol == symbol
            ).first()
            
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
            
            cache_record = db.query(DataCache).filter(
                DataCache.symbol == symbol
            ).first()
            
            if cache_record:
                cache_record.last_updated = datetime.utcnow()
                cache_record.data_count = data_count
            else:
                cache_record = DataCache(
                    symbol=symbol,
                    last_updated=datetime.utcnow(),
                    data_count=data_count,
                    cache_key=f"{symbol}_{datetime.utcnow().strftime('%Y%m%d')}"
                )
                db.add(cache_record)
            
            db.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating cache metadata for {symbol}: {e}")
            db.rollback()
        finally:
            db.close()
    
    def get_data_statistics(self) -> List[Tuple[str, int, datetime]]:
        """
        Get statistics about cached data.
        
        Returns:
            List of (symbol, record_count, last_updated) tuples
        """
        try:
            db = SessionLocal()
            
            stats = db.query(
                DataCache.symbol,
                DataCache.data_count,
                DataCache.last_updated
            ).all()
            
            return [(stat.symbol, stat.data_count, stat.last_updated) for stat in stats]
            
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
            
            deleted_count = db.query(HistoricalData).filter(
                HistoricalData.date < cutoff_date
            ).delete()
            
            db.commit()
            
            self.logger.info(f"Deleted {deleted_count} old records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error clearing old data: {e}")
            db.rollback()
            return 0
        finally:
            db.close()


# Global database manager instance
db_manager = DatabaseManager() 