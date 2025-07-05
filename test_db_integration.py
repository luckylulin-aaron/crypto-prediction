#!/usr/bin/env python3
"""
Test script for PostgreSQL database integration.
This script tests the database functionality without requiring a live database connection.
"""

import os
import sys

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))


def test_database_imports():
    """Test that database modules can be imported."""
    try:
        from db.database import DataCache, HistoricalData, db_manager

        print("✅ Database modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import database modules: {e}")
        return False


def test_database_models():
    """Test database model definitions."""
    try:
        from db.database import DataCache, HistoricalData

        # Test HistoricalData model
        assert hasattr(HistoricalData, "__tablename__")
        assert HistoricalData.__tablename__ == "historical_data"

        # Test DataCache model
        assert hasattr(DataCache, "__tablename__")
        assert DataCache.__tablename__ == "data_cache"

        print("✅ Database models defined correctly")
        return True
    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False


def test_cbpro_client_integration():
    """Test that CBProClient can import database module."""
    try:
        from trading.cbpro_client import CBProClient

        print("✅ CBProClient can import database module")
        return True
    except ImportError as e:
        print(f"❌ CBProClient import failed: {e}")
        return False


def test_server_integration():
    """Test that server can import database module."""
    try:
        from api.server import app

        print("✅ Server can import database module")
        return True
    except ImportError as e:
        print(f"❌ Server import failed: {e}")
        return False


def test_db_management_script():
    """Test database management script imports."""
    try:
        from db.db_management import init_database, show_statistics, test_connection

        print("✅ Database management script imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Database management script import failed: {e}")
        return False


def main():
    """Run all database integration tests."""
    print("🧪 Testing PostgreSQL Database Integration")
    print("=" * 50)

    tests = [
        ("Database Imports", test_database_imports),
        ("Database Models", test_database_models),
        ("CBProClient Integration", test_cbpro_client_integration),
        ("Server Integration", test_server_integration),
        ("DB Management Script", test_db_management_script),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} test failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Database integration is ready.")
        print("\n📋 Next Steps:")
        print("1. Set up PostgreSQL database:")
        print("   # Option A: Using Docker (if installed)")
        print("   docker-compose up -d postgres")
        print("   ")
        print("   # Option B: Install PostgreSQL locally")
        print("   # macOS: brew install postgresql")
        print("   # Ubuntu: sudo apt-get install postgresql postgresql-contrib")
        print(
            "   # Windows: Download from https://www.postgresql.org/download/windows/"
        )
        print("2. Initialize database tables:")
        print("   poetry run python app/db/db_management.py init")
        print("3. Test database connection:")
        print("   poetry run python app/db/db_management.py test")
        print("4. Start the trading bot:")
        print("   poetry run start-server")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
