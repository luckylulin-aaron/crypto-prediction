"""
Fear & Greed Index client for fetching Bitcoin market sentiment data.
Uses the Alternative.me API to get daily fear and greed index values.
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import os

logger = logging.getLogger(__name__)

class FearGreedClient:
    """
    Client for fetching Bitcoin Fear & Greed Index data from Alternative.me API.
    """
    
    def __init__(self, db_path: str = "app/data/fear_greed.db"):
        """
        Initialize the Fear & Greed Index client.
        
        Args:
            db_path: Path to SQLite database for caching fear & greed data
        """
        self.base_url = "https://api.alternative.me/fng/"
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database for storing fear & greed data."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fear_greed_index (
                    date TEXT PRIMARY KEY,
                    value INTEGER,
                    classification TEXT,
                    timestamp INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_current_fear_greed(self) -> Optional[Dict]:
        """
        Get the current Fear & Greed Index value.
        
        Returns:
            Dict containing current fear & greed data or None if error
        """
        try:
            response = requests.get(f"{self.base_url}?limit=1", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("data") and len(data["data"]) > 0:
                return data["data"][0]
            return None
            
        except Exception as e:
            logger.error(f"Error fetching current fear & greed index: {e}")
            return None
    
    def get_historical_fear_greed(self, days: int = 120) -> List[Dict]:
        """
        Get historical Fear & Greed Index data.
        
        Args:
            days: Number of days to fetch (max 365)
            
        Returns:
            List of fear & greed data dictionaries
        """
        try:
            # Check cache first
            cached_data = self._get_cached_data(days)
            if cached_data and len(cached_data) >= days:
                logger.info(f"Using cached fear & greed data for {len(cached_data)} days")
                return cached_data[:days]
            
            # Fetch from API
            response = requests.get(f"{self.base_url}?limit={days}", timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("data"):
                # Add date field to each record
                for record in data["data"]:
                    timestamp = int(record["timestamp"])
                    record["date"] = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                
                # Cache the data
                self._cache_data(data["data"])
                logger.info(f"Fetched and cached {len(data['data'])} days of fear & greed data")
                return data["data"]
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching historical fear & greed data: {e}")
            return []
    
    def _get_cached_data(self, days: int) -> List[Dict]:
        """Get cached fear & greed data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date, value, classification, timestamp 
                    FROM fear_greed_index 
                    ORDER BY date DESC 
                    LIMIT ?
                """, (days,))
                
                rows = cursor.fetchall()
                return [
                    {
                        "date": row[0],
                        "value": str(row[1]),
                        "value_classification": row[2],
                        "timestamp": str(row[3])
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Error reading cached fear & greed data: {e}")
            return []
    
    def _cache_data(self, data: List[Dict]):
        """Cache fear & greed data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for record in data:
                    # Convert timestamp to date
                    timestamp = int(record["timestamp"])
                    date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO fear_greed_index 
                        (date, value, classification, timestamp) 
                        VALUES (?, ?, ?, ?)
                    """, (
                        date,
                        int(record["value"]),
                        record["value_classification"],
                        timestamp
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching fear & greed data: {e}")
    
    def get_fear_greed_by_date(self, date: str) -> Optional[Dict]:
        """
        Get Fear & Greed Index data for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Fear & greed data for the date or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date, value, classification, timestamp 
                    FROM fear_greed_index 
                    WHERE date = ?
                """, (date,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "date": row[0],
                        "value": str(row[1]),
                        "value_classification": row[2],
                        "timestamp": str(row[3])
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error fetching fear & greed data for date {date}: {e}")
            return None
    
    def get_fear_greed_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Get Fear & Greed Index data for a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of fear & greed data for the date range
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT date, value, classification, timestamp 
                    FROM fear_greed_index 
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date ASC
                """, (start_date, end_date))
                
                rows = cursor.fetchall()
                return [
                    {
                        "date": row[0],
                        "value": str(row[1]),
                        "value_classification": row[2],
                        "timestamp": str(row[3])
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error fetching fear & greed data for range {start_date} to {end_date}: {e}")
            return []
    
    def get_classification_color(self, classification: str) -> str:
        """
        Get color for fear & greed classification.
        
        Args:
            classification: Fear & greed classification string
            
        Returns:
            Hex color code
        """
        colors = {
            "Extreme Fear": "#FF0000",      # Red
            "Fear": "#FF6600",              # Orange
            "Neutral": "#FFFF00",           # Yellow
            "Greed": "#00FF00",             # Green
            "Extreme Greed": "#00FF66"      # Bright Green
        }
        return colors.get(classification, "#808080")  # Gray for unknown
    
    def get_classification_description(self, value: int) -> str:
        """
        Get description for fear & greed value.
        
        Args:
            value: Fear & greed index value (0-100)
            
        Returns:
            Description of the sentiment
        """
        if value <= 25:
            return "Extreme Fear - Market sentiment is very negative, potential buying opportunity"
        elif value <= 45:
            return "Fear - Market sentiment is negative, cautious buying"
        elif value <= 55:
            return "Neutral - Market sentiment is balanced"
        elif value <= 75:
            return "Greed - Market sentiment is positive, cautious selling"
        else:
            return "Extreme Greed - Market sentiment is very positive, potential selling opportunity" 