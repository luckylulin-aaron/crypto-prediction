"""
Test to verify Poetry setup and basic imports work correctly.
"""

import sys
import os

# Add the app directory to the Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


def test_imports():
    """Test that all main modules can be imported."""
    try:
        import config
        import logger
        import util
        import cbpro_client
        import ma_trader
        import trader_driver
        import strategies
        import visualization
        import server
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_config():
    """Test that configuration can be loaded."""
    try:
        from config import CURS, STRATEGIES, TIMESPAN
        assert isinstance(CURS, list), "CURS should be a list"
        assert isinstance(STRATEGIES, list), "STRATEGIES should be a list"
        assert isinstance(TIMESPAN, int), "TIMESPAN should be an integer"
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_logger():
    """Test that logger can be initialized."""
    try:
        from logger import get_logger
        logger = get_logger("test")
        assert logger is not None, "Logger should not be None"
        print("✅ Logger initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Logger error: {e}")
        return False


def test_poetry_dependencies():
    """Test that Poetry dependencies are available."""
    try:
        import flask
        import plotly
        import pandas
        import numpy
        import requests
        print("✅ Poetry dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Poetry dependency error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Poetry setup...")
    
    tests = [
        test_poetry_dependencies,
        test_imports,
        test_config,
        test_logger,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Poetry setup is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1) 