"""
Test suite for Economic Indicators Strategy.
Tests the integration of macroeconomic indicators into trading decisions.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from data.economic_indicators_client import EconomicIndicatorsClient
from trading.strategies import EconomicIndicatorsStrategy


class TestEconomicIndicatorsClient(unittest.TestCase):
    """Test the Economic Indicators Client functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = EconomicIndicatorsClient(fred_api_key="test_key")
    
    @patch('requests.get')
    def test_get_m2_money_supply(self, mock_get):
        """Test M2 money supply data retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '1000.0'},
                {'date': '2024-02-01', 'value': '1010.0'},
                {'date': '2024-03-01', 'value': '1020.0'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.get_m2_money_supply(days_back=90)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn('value', result.columns)
        self.assertIn('daily_change', result.columns)
        self.assertIn('daily_change_pct', result.columns)
    
    @patch('requests.get')
    def test_get_overnight_reverse_repo(self, mock_get):
        """Test overnight reverse repo data retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            'observations': [
                {'date': '2024-01-01', 'value': '500.0'},
                {'date': '2024-01-02', 'value': '510.0'},
                {'date': '2024-01-03', 'value': '520.0'}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test
        result = self.client.get_overnight_reverse_repo(days_back=30)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn('value', result.columns)
        self.assertIn('daily_change', result.columns)
    
    @patch.object(EconomicIndicatorsClient, 'get_m2_money_supply')
    @patch.object(EconomicIndicatorsClient, 'get_overnight_reverse_repo')
    def test_get_liquidity_indicators(self, mock_rrp, mock_m2):
        """Test liquidity indicators calculation."""
        # Mock M2 data
        m2_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3, freq='M'),
            'value': [1000.0, 1010.0, 1020.0],
            'daily_change': [0, 10.0, 10.0],
            'daily_change_pct': [0, 1.0, 0.99]
        })
        mock_m2.return_value = m2_data
        
        # Mock ON RRP data
        rrp_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7, freq='D'),
            'value': [500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0],
            'daily_change': [0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'daily_change_pct': [0, 2.0, 1.96, 1.92, 1.89, 1.85, 1.82]
        })
        mock_rrp.return_value = rrp_data
        
        # Test
        result = self.client.get_liquidity_indicators()
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('m2_current', result)
        self.assertIn('on_rrp_current', result)
        self.assertIn('m2_change_1m_pct', result)
        self.assertIn('on_rrp_vs_7d_avg_pct', result)
    
    @patch.object(EconomicIndicatorsClient, 'get_liquidity_indicators')
    def test_get_trading_signals(self, mock_indicators):
        """Test trading signals generation."""
        # Mock indicators
        mock_indicators.return_value = {
            'm2_change_1m_pct': 1.5,  # Bullish M2 growth
            'on_rrp_vs_7d_avg_pct': -15.0  # Bullish ON RRP (easing)
        }
        
        # Test
        result = self.client.get_trading_signals()
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn('liquidity_signal', result)
        self.assertIn('m2_signal', result)
        self.assertIn('rrp_signal', result)


class TestEconomicIndicatorsStrategy(unittest.TestCase):
    """Test the Economic Indicators Strategy functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = EconomicIndicatorsStrategy()
    
    @patch.object(EconomicIndicatorsClient, 'get_liquidity_indicators')
    @patch.object(EconomicIndicatorsClient, 'get_trading_signals')
    def test_get_macro_signal_bullish(self, mock_signals, mock_indicators):
        """Test bullish macro signal generation."""
        # Mock bullish indicators
        mock_indicators.return_value = {
            'm2_change_1m_pct': 1.5,  # Strong bullish
            'on_rrp_vs_7d_avg_pct': -15.0  # Strong bullish
        }
        mock_signals.return_value = {
            'm2_signal': 'BULLISH',
            'rrp_signal': 'BULLISH'
        }
        
        # Test
        result = self.strategy.get_macro_signal()
        
        # Assertions
        self.assertEqual(result['signal'], 'BULLISH')
        self.assertGreater(result['strength'], 0.5)
        self.assertIn('M2 growing', result['reason'])
        self.assertIn('ON RRP', result['reason'])
    
    @patch.object(EconomicIndicatorsClient, 'get_liquidity_indicators')
    @patch.object(EconomicIndicatorsClient, 'get_trading_signals')
    def test_get_macro_signal_bearish(self, mock_signals, mock_indicators):
        """Test bearish macro signal generation."""
        # Mock bearish indicators
        mock_indicators.return_value = {
            'm2_change_1m_pct': -1.0,  # Strong bearish
            'on_rrp_vs_7d_avg_pct': 15.0  # Strong bearish
        }
        mock_signals.return_value = {
            'm2_signal': 'BEARISH',
            'rrp_signal': 'BEARISH'
        }
        
        # Test
        result = self.strategy.get_macro_signal()
        
        # Assertions
        self.assertEqual(result['signal'], 'BEARISH')
        self.assertGreater(result['strength'], 0.5)
        self.assertIn('M2 contracting', result['reason'])
        self.assertIn('ON RRP', result['reason'])
    
    def test_adjust_technical_signal_bullish_macro(self):
        """Test technical signal adjustment with bullish macro."""
        # Mock technical signal
        technical_signal = {
            'action': 'BUY',
            'buy_percentage': 50,
            'sell_percentage': 0
        }
        
        # Mock bullish macro signal
        macro_signal = {
            'signal': 'BULLISH',
            'strength': 0.6,
            'reason': 'Strong liquidity indicators'
        }
        
        # Test
        result = self.strategy.adjust_technical_signal(technical_signal, macro_signal)
        
        # Assertions
        self.assertEqual(result['action'], 'BUY')
        self.assertGreater(result['buy_percentage'], 50)  # Should be boosted
        self.assertIn('macro_overlay', result)
        self.assertEqual(result['macro_overlay']['macro_signal'], 'BULLISH')
    
    def test_adjust_technical_signal_bearish_macro(self):
        """Test technical signal adjustment with bearish macro."""
        # Mock technical signal
        technical_signal = {
            'action': 'BUY',
            'buy_percentage': 50,
            'sell_percentage': 0
        }
        
        # Mock bearish macro signal
        macro_signal = {
            'signal': 'BEARISH',
            'strength': 0.6,
            'reason': 'Tightening liquidity'
        }
        
        # Test
        result = self.strategy.adjust_technical_signal(technical_signal, macro_signal)
        
        # Assertions
        self.assertLess(result['buy_percentage'], 50)  # Should be reduced
        self.assertIn('macro_overlay', result)
        self.assertEqual(result['macro_overlay']['macro_signal'], 'BEARISH')
    
    @patch.object(EconomicIndicatorsStrategy, 'get_macro_signal')
    def test_get_combined_signal(self, mock_macro):
        """Test combined signal generation."""
        # Mock macro signal
        mock_macro.return_value = {
            'signal': 'BULLISH',
            'strength': 0.5,
            'reason': 'Strong liquidity'
        }
        
        # Mock technical strategies
        mock_technical = Mock()
        mock_technical.trade_signal = {
            'action': 'BUY',
            'buy_percentage': 60,
            'sell_percentage': 0
        }
        
        # Test
        result = self.strategy.get_combined_signal([mock_technical])
        
        # Assertions
        self.assertIn('action', result)
        self.assertIn('macro_overlay', result)
        self.assertIn('strategy_name', result)
        self.assertIn('macro_indicators', result)


class TestEconomicIndicatorsIntegration(unittest.TestCase):
    """Integration tests for economic indicators with trading system."""
    
    def test_strategy_registry_inclusion(self):
        """Test that economic indicators strategy is in registry."""
        from trading.strategies import STRATEGY_REGISTRY
        
        self.assertIn('ECONOMIC-INDICATORS', STRATEGY_REGISTRY)
        self.assertEqual(STRATEGY_REGISTRY['ECONOMIC-INDICATORS'], EconomicIndicatorsStrategy)
    
    def test_strategy_initialization(self):
        """Test strategy initialization without API key."""
        strategy = EconomicIndicatorsStrategy()
        
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.name, "Economic Indicators")
        self.assertIsNotNone(strategy.econ_client)


if __name__ == '__main__':
    unittest.main()
