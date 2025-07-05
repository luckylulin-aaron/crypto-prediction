"""
Database management utilities for the cryptocurrency trading bot.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

from database import db_manager, engine, Base
from logger import get_logger


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
            print("No cached data found in database.")
            return
        
        print("\n=== Database Statistics ===")
        print(f"{'Symbol':<15} {'Records':<10} {'Last Updated':<20}")
        print("-" * 50)
        
        for symbol, count, last_updated in stats:
            print(f"{symbol:<15} {count:<10} {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
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


def test_connection():
    """Test database connection."""
    logger = get_logger(__name__)
    try:
        # Try to execute a simple query
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        print("Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        print(f"Database connection failed: {e}")
        return False


def main():
    """Main function for database management CLI."""
    parser = argparse.ArgumentParser(description="Database management for crypto trading bot")
    parser.add_argument(
        "command",
        choices=["init", "drop", "stats", "clear", "test"],
        help="Command to execute"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to keep when clearing old data (default: 365)"
    )
    
    args = parser.parse_args()
    
    if args.command == "init":
        success = init_database()
        sys.exit(0 if success else 1)
    
    elif args.command == "drop":
        confirm = input("Are you sure you want to drop all tables? (y/N): ")
        if confirm.lower() == 'y':
            success = drop_database()
            sys.exit(0 if success else 1)
        else:
            print("Operation cancelled.")
            sys.exit(0)
    
    elif args.command == "stats":
        show_statistics()
        sys.exit(0)
    
    elif args.command == "clear":
        success = clear_old_data(args.days)
        sys.exit(0 if success else 1)
    
    elif args.command == "test":
        success = test_connection()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 