"""
Economic Indicators Client for fetching macroeconomic data from FRED API.
Provides M2 money supply and Federal Reserve overnight reverse repo data.
"""

import configparser
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from app.core.logger import get_logger


class EconomicIndicatorsClient:
    """
    Client for fetching economic indicators from FRED API and other sources.
    
    Provides access to:
    - M2 Money Supply (monthly)
    - Federal Reserve Overnight Reverse Repo (daily)
    - Other macroeconomic indicators
    """
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the economic indicators client.
        
        Args:
            fred_api_key (str, optional): FRED API key. If not provided, 
                                        will try to get from environment variable FRED_API_KEY,
                                        or from secret.ini file.
        """
        self.logger = get_logger(__name__)
        
        # Try to get API key from different sources
        self.fred_api_key = fred_api_key or self._get_fred_api_key()
        
        if not self.fred_api_key:
            self.logger.warning("No FRED API key provided. Some features may be limited.")
        
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    def _get_fred_api_key(self) -> Optional[str]:
        """
        Get FRED API key from environment variable or secret.ini file.
        
        Returns:
            Optional[str]: FRED API key if found, None otherwise
        """
        # First try environment variable
        api_key = os.environ.get('FRED_API_KEY')
        if api_key:
            return api_key
        
        # Then try secret.ini file
        try:
            config = configparser.ConfigParser()
            config_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'secret.ini')
            if config.read(config_path):
                api_key = config.get('CONFIG', 'FRED_API_KEY', fallback=None)
                if api_key and api_key != "your_fred_api_key_here":
                    # Remove quotes if present
                    api_key = api_key.strip().strip('"').strip("'")
                    return api_key
        except Exception as e:
            self.logger.debug(f"Could not read FRED API key from secret.ini: {e}")
        
        return None
        
    def get_m2_money_supply(self, days_back: int = 365) -> pd.DataFrame:
        """
        Get M2 Money Supply data from FRED.
        
        Args:
            days_back (int): Number of days to look back for data
            
        Returns:
            pd.DataFrame: DataFrame with date and M2 values
        """
        try:
            if not self.fred_api_key:
                self.logger.error("FRED API key required for M2 data")
                return pd.DataFrame()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'series_id': 'M2SL',  # M2 Money Stock
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'limit': 12  # Get last 12 months of data
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            self.logger.debug(f"M2 API response: {len(observations)} observations")
            
            if not observations:
                self.logger.warning("No M2 data returned from FRED")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Calculate daily change (since M2 is monthly, we'll interpolate)
            df = df.sort_values('date')
            df['daily_change'] = df['value'].diff()
            df['daily_change_pct'] = df['value'].pct_change() * 100
            
            self.logger.info(f"Retrieved {len(df)} M2 data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching M2 data: {e}")
            return pd.DataFrame()
    
    def get_overnight_reverse_repo(self, days_back: int = 365) -> pd.DataFrame:
        """
        Get Federal Reserve Overnight Reverse Repo data from FRED.
        
        Args:
            days_back (int): Number of days to look back for data
            
        Returns:
            pd.DataFrame: DataFrame with date and ON RRP values
        """
        try:
            if not self.fred_api_key:
                self.logger.error("FRED API key required for ON RRP data")
                return pd.DataFrame()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'series_id': 'RRPONTTLD',  # Overnight Reverse Repurchase Agreements
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'frequency': 'd'  # Daily frequency
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get('observations', [])
            
            if not observations:
                self.logger.warning("No ON RRP data returned from FRED")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Calculate daily changes
            df = df.sort_values('date')
            df['daily_change'] = df['value'].diff()
            df['daily_change_pct'] = df['value'].pct_change() * 100
            
            self.logger.info(f"Retrieved {len(df)} ON RRP data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching ON RRP data: {e}")
            return pd.DataFrame()
    
    def get_liquidity_indicators(self, days_back: int = 30) -> Dict[str, float]:
        """
        Get current liquidity indicators for trading signals.
        
        Args:
            days_back (int): Number of days to look back for calculations
            
        Returns:
            Dict[str, float]: Dictionary with current liquidity metrics
        """
        try:
            indicators = {}
            
            # Get M2 data (use longer period for monthly data)
            m2_df = self.get_m2_money_supply(max(days_back, 365))  # At least 1 year for monthly data
            if not m2_df.empty:
                latest_m2 = m2_df.iloc[-1]
                indicators['m2_current'] = latest_m2['value']
                indicators['m2_change_1m'] = latest_m2['daily_change']
                indicators['m2_change_1m_pct'] = latest_m2['daily_change_pct']
                
                # Calculate 30-day trend
                if len(m2_df) >= 2:
                    month_ago = m2_df.iloc[-2]
                    indicators['m2_change_30d'] = latest_m2['value'] - month_ago['value']
                    indicators['m2_change_30d_pct'] = ((latest_m2['value'] - month_ago['value']) / month_ago['value']) * 100
            
            # Get ON RRP data
            on_rrp_df = self.get_overnight_reverse_repo(days_back)
            if not on_rrp_df.empty:
                latest_rrp = on_rrp_df.iloc[-1]
                indicators['on_rrp_current'] = latest_rrp['value']
                indicators['on_rrp_change_1d'] = latest_rrp['daily_change']
                indicators['on_rrp_change_1d_pct'] = latest_rrp['daily_change_pct']
                
                # Calculate 7-day average
                if len(on_rrp_df) >= 7:
                    week_avg = on_rrp_df.tail(7)['value'].mean()
                    indicators['on_rrp_7d_avg'] = week_avg
                    indicators['on_rrp_vs_7d_avg'] = latest_rrp['value'] - week_avg
                    indicators['on_rrp_vs_7d_avg_pct'] = ((latest_rrp['value'] - week_avg) / week_avg) * 100
            
            self.logger.info(f"Calculated {len(indicators)} liquidity indicators")
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity indicators: {e}")
            return {}
    
    def get_trading_signals(self) -> Dict[str, str]:
        """
        Generate trading signals based on economic indicators.
        
        Returns:
            Dict[str, str]: Dictionary with trading signals and reasoning
        """
        try:
            indicators = self.get_liquidity_indicators()
            signals = {}
            
            if not indicators:
                signals['liquidity_signal'] = 'NEUTRAL'
                signals['liquidity_reason'] = 'No economic data available'
                return signals
            
            # M2-based signals
            if 'm2_change_1m_pct' in indicators:
                m2_change = indicators['m2_change_1m_pct']
                if m2_change > 1.0:  # M2 growing > 1% monthly
                    signals['m2_signal'] = 'BULLISH'
                    signals['m2_reason'] = f'M2 money supply growing {m2_change:.2f}% monthly'
                elif m2_change < -0.5:  # M2 contracting > 0.5% monthly
                    signals['m2_signal'] = 'BEARISH'
                    signals['m2_reason'] = f'M2 money supply contracting {abs(m2_change):.2f}% monthly'
                else:
                    signals['m2_signal'] = 'NEUTRAL'
                    signals['m2_reason'] = f'M2 money supply stable ({m2_change:.2f}% monthly)'
            
            # ON RRP-based signals
            if 'on_rrp_vs_7d_avg_pct' in indicators:
                rrp_deviation = indicators['on_rrp_vs_7d_avg_pct']
                if rrp_deviation > 10:  # ON RRP significantly above 7-day average
                    signals['rrp_signal'] = 'BEARISH'
                    signals['rrp_reason'] = f'ON RRP {rrp_deviation:.1f}% above 7-day average (tightening)'
                elif rrp_deviation < -10:  # ON RRP significantly below 7-day average
                    signals['rrp_signal'] = 'BULLISH'
                    signals['rrp_reason'] = f'ON RRP {abs(rrp_deviation):.1f}% below 7-day average (easing)'
                else:
                    signals['rrp_signal'] = 'NEUTRAL'
                    signals['rrp_reason'] = f'ON RRP near 7-day average ({rrp_deviation:.1f}% deviation)'
            
            # Combined liquidity signal
            bullish_count = sum(1 for signal in signals.values() if signal == 'BULLISH')
            bearish_count = sum(1 for signal in signals.values() if signal == 'BEARISH')
            
            if bullish_count > bearish_count:
                signals['liquidity_signal'] = 'BULLISH'
                signals['liquidity_reason'] = f'{bullish_count} bullish vs {bearish_count} bearish indicators'
            elif bearish_count > bullish_count:
                signals['liquidity_signal'] = 'BEARISH'
                signals['liquidity_reason'] = f'{bearish_count} bearish vs {bullish_count} bullish indicators'
            else:
                signals['liquidity_signal'] = 'NEUTRAL'
                signals['liquidity_reason'] = 'Mixed economic indicators'
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {'liquidity_signal': 'NEUTRAL', 'liquidity_reason': f'Error: {e}'}
    
    def get_historical_data_for_backtest(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical economic data for backtesting.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with daily economic indicators
        """
        try:
            # Get both M2 and ON RRP data
            m2_df = self.get_m2_money_supply(365)  # Get full year
            on_rrp_df = self.get_overnight_reverse_repo(365)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if not m2_df.empty:
                m2_df = m2_df[(m2_df['date'] >= start_dt) & (m2_df['date'] <= end_dt)]
            
            if not on_rrp_df.empty:
                on_rrp_df = on_rrp_df[(on_rrp_df['date'] >= start_dt) & (on_rrp_df['date'] <= end_dt)]
            
            # Merge dataframes on date
            if not m2_df.empty and not on_rrp_df.empty:
                merged_df = pd.merge(m2_df, on_rrp_df, on='date', how='outer', suffixes=('_m2', '_rrp'))
            elif not m2_df.empty:
                merged_df = m2_df
            elif not on_rrp_df.empty:
                merged_df = on_rrp_df
            else:
                return pd.DataFrame()
            
            # Forward fill missing values (especially for M2 monthly data)
            merged_df = merged_df.sort_values('date').fillna(method='ffill')
            
            self.logger.info(f"Prepared {len(merged_df)} historical data points for backtesting")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error preparing historical data: {e}")
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    # Test the client
    client = EconomicIndicatorsClient()
    
    # Get current trading signals
    signals = client.get_trading_signals()
    print("Current Trading Signals:")
    for key, value in signals.items():
        print(f"  {key}: {value}")
    
    # Get liquidity indicators
    indicators = client.get_liquidity_indicators()
    print("\nLiquidity Indicators:")
    for key, value in indicators.items():
        print(f"  {key}: {value}")
