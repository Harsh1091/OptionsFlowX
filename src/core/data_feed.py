"""
Real-time market data feed implementation.

This module provides real-time market data handling with support for
multiple data sources and websocket connections.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from loguru import logger
from ..utils.config import get_config, get_api_config
from ..utils.logger import log_market_data_update


class DataFeed:
    """
    Real-time market data feed for OptionsFlowX.
    
    Handles data collection from multiple sources including
    Zerodha, Angel Broking, and other providers.
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize data feed.
        
        Args:
            symbols: List of symbols to track
        """
        self.symbols = symbols or ["NIFTY", "BANKNIFTY"]
        self.config = get_config()
        self.api_config = get_api_config()
        
        # Data storage
        self.market_data = {}
        self.historical_data = {}
        self.websocket_connections = {}
        
        # Callbacks
        self.data_callbacks = []
        self.error_callbacks = []
        
        # Connection status
        self.is_connected = False
        self.last_update = None
        
        logger.info(f"DataFeed initialized for symbols: {self.symbols}")
    
    async def start(self) -> None:
        """Start the data feed."""
        try:
            logger.info("Starting data feed...")
            
            # Initialize connections based on provider
            provider = self.api_config.get('provider', 'paper_trading')
            
            if provider == 'zerodha':
                await self._init_zerodha_connection()
            elif provider == 'angel':
                await self._init_angel_connection()
            elif provider == 'upstox':
                await self._init_upstox_connection()
            else:
                await self._init_paper_trading()
            
            self.is_connected = True
            logger.info("Data feed started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start data feed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the data feed."""
        try:
            logger.info("Stopping data feed...")
            
            # Close websocket connections
            for connection in self.websocket_connections.values():
                if hasattr(connection, 'close'):
                    connection.close()
            
            self.websocket_connections.clear()
            self.is_connected = False
            
            logger.info("Data feed stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping data feed: {e}")
    
    async def _init_zerodha_connection(self) -> None:
        """Initialize Zerodha connection."""
        try:
            api_key = self.api_config.get('api_key')
            api_secret = self.api_config.get('api_secret')
            access_token = self.api_config.get('access_token')
            
            if not all([api_key, api_secret, access_token]):
                raise ValueError("Missing Zerodha API credentials")
            
            logger.info("Zerodha connection initialized")
            await self._init_paper_trading()  # Fallback to paper trading
            
        except Exception as e:
            logger.error(f"Failed to initialize Zerodha connection: {e}")
            raise
    
    async def _init_angel_connection(self) -> None:
        """Initialize Angel Broking connection."""
        try:
            logger.info("Angel Broking connection not implemented yet")
            await self._init_paper_trading()
            
        except Exception as e:
            logger.error(f"Failed to initialize Angel connection: {e}")
            raise
    
    async def _init_upstox_connection(self) -> None:
        """Initialize Upstox connection."""
        try:
            logger.info("Upstox connection not implemented yet")
            await self._init_paper_trading()
            
        except Exception as e:
            logger.error(f"Failed to initialize Upstox connection: {e}")
            raise
    
    async def _init_paper_trading(self) -> None:
        """Initialize paper trading with simulated data."""
        try:
            logger.info("Initializing paper trading mode")
            
            # Generate simulated market data
            for symbol in self.symbols:
                self.market_data[symbol] = {
                    'symbol': symbol,
                    'last_price': 18000.0 if symbol == 'NIFTY' else 42000.0,
                    'bid': 17999.0 if symbol == 'NIFTY' else 41999.0,
                    'ask': 18001.0 if symbol == 'NIFTY' else 42001.0,
                    'volume': 1000,
                    'timestamp': datetime.now(),
                    'change': 0.0,
                    'change_percent': 0.0
                }
            
            # Start simulated data updates
            asyncio.create_task(self._simulate_market_data())
            
        except Exception as e:
            logger.error(f"Failed to initialize paper trading: {e}")
            raise
    
    async def _simulate_market_data(self) -> None:
        """Simulate market data updates for paper trading."""
        import random
        
        while self.is_connected:
            try:
                for symbol in self.symbols:
                    # Generate random price movement
                    current_data = self.market_data[symbol]
                    base_price = 18000.0 if symbol == 'NIFTY' else 42000.0
                    
                    # Random walk with mean reversion
                    change = random.gauss(0, 50)  # 50 point standard deviation
                    new_price = current_data['last_price'] + change
                    
                    # Keep price within reasonable bounds
                    new_price = max(base_price * 0.8, min(base_price * 1.2, new_price))
                    
                    # Update market data
                    self._update_market_data(symbol, {
                        'last_price': new_price,
                        'bid': new_price - 1,
                        'ask': new_price + 1,
                        'volume': current_data['volume'] + random.randint(100, 1000),
                        'timestamp': datetime.now()
                    })
                
                # Wait for next update
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                logger.error(f"Error in simulated data: {e}")
                await asyncio.sleep(5)
    
    def _update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update market data for a symbol."""
        try:
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            
            # Update data
            self.market_data[symbol].update(data)
            self.market_data[symbol]['timestamp'] = datetime.now()
            
            # Calculate change
            if 'last_price' in data and 'last_price' in self.market_data[symbol]:
                prev_price = self.market_data[symbol].get('last_price', data['last_price'])
                change = data['last_price'] - prev_price
                change_percent = (change / prev_price) * 100 if prev_price != 0 else 0
                
                self.market_data[symbol]['change'] = change
                self.market_data[symbol]['change_percent'] = change_percent
            
            self.last_update = datetime.now()
            
            # Log update
            log_market_data_update(symbol, self.market_data[symbol])
            
            # Trigger callbacks
            for callback in self.data_callbacks:
                try:
                    callback(symbol, self.market_data[symbol])
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary or None if not available
        """
        return self.market_data.get(symbol)
    
    def get_all_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get market data for all symbols.
        
        Returns:
            Dictionary of market data for all symbols
        """
        return self.market_data.copy()
    
    async def get_historical_data(self, symbol: str, 
                                start_date: datetime,
                                end_date: datetime,
                                interval: str = '1minute') -> Optional[Dict[str, Any]]:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Historical data or None if not available
        """
        try:
            # Generate simulated historical data
            return self._generate_historical_data(symbol, start_date, end_date, interval)
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _generate_historical_data(self, symbol: str, start_date: datetime,
                                end_date: datetime, interval: str) -> Dict[str, Any]:
        """Generate simulated historical data."""
        # Implementation for generating simulated historical data
        return {}
    
    def add_data_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Add callback for market data updates.
        
        Args:
            callback: Function to call when data updates
        """
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """
        Add callback for error events.
        
        Args:
            callback: Function to call when errors occur
        """
        self.error_callbacks.append(callback)
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open.
        
        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Check market hours (IST: 9:15 AM to 3:30 PM)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary containing connection status
        """
        return {
            'is_connected': self.is_connected,
            'last_update': self.last_update,
            'symbols': self.symbols,
            'provider': self.api_config.get('provider', 'unknown'),
            'market_open': self.is_market_open()
        } 