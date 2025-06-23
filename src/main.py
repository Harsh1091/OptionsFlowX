"""
Main application entry point for OptionsFlowX.

This module provides the main OptionsFlowX class and CLI interface
for running the high-frequency trading scanner.
"""

import asyncio
import signal
import sys
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
from .core import DataFeed, SignalProcessor, RiskManager
from .utils.config import get_config
from .utils.logger import log_performance_metrics


class OptionsFlowX:
    """
    Main OptionsFlowX application class.
    
    Orchestrates data feed, signal processing, and risk management
    for high-frequency options trading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OptionsFlowX application.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Load configuration
        self.config = config or get_config()
        
        # Initialize components
        self.data_feed = DataFeed()
        self.signal_processor = SignalProcessor()
        self.risk_manager = RiskManager()
        
        # Application state
        self.is_running = False
        self.start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'signals_generated': 0,
            'positions_opened': 0,
            'total_pnl': 0.0,
            'uptime': 0
        }
        
        logger.info("OptionsFlowX initialized")
    
    async def start(self) -> None:
        """Start the OptionsFlowX application."""
        try:
            logger.info("Starting OptionsFlowX...")
            
            # Start data feed
            await self.data_feed.start()
            
            # Register data callbacks
            self.data_feed.add_data_callback(self._on_market_data_update)
            self.data_feed.add_error_callback(self._on_error)
            
            # Set running state
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("OptionsFlowX started successfully")
            
            # Start main processing loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Error starting OptionsFlowX: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the OptionsFlowX application."""
        try:
            logger.info("Stopping OptionsFlowX...")
            
            self.is_running = False
            
            # Stop data feed
            await self.data_feed.stop()
            
            # Close all positions
            await self._close_all_positions()
            
            # Log final performance metrics
            self._log_final_metrics()
            
            logger.info("OptionsFlowX stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping OptionsFlowX: {e}")
    
    async def _main_loop(self) -> None:
        """Main processing loop."""
        try:
            while self.is_running:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check for stop-loss/take-profit triggers
                await self._check_position_triggers()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
    
    def _on_market_data_update(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Handle market data updates."""
        try:
            # Process market data for signals
            signal = self.signal_processor.process_market_data(symbol, market_data)
            
            if signal:
                self.performance_metrics['signals_generated'] += 1
                
                # Check if we can open a position
                if self.risk_manager.can_open_position(symbol, signal):
                    # Calculate position size
                    current_price = market_data.get('last_price', 0)
                    quantity, risk_amount = self.risk_manager.calculate_position_size(
                        signal, current_price
                    )
                    
                    if quantity > 0:
                        # Open position
                        success = self.risk_manager.open_position(
                            symbol, signal, current_price, quantity
                        )
                        
                        if success:
                            self.performance_metrics['positions_opened'] += 1
                            logger.info(f"Opened position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing market data update: {e}")
    
    def _on_error(self, error: Exception) -> None:
        """Handle errors."""
        logger.error(f"Data feed error: {error}")
    
    async def _check_position_triggers(self) -> None:
        """Check for stop-loss and take-profit triggers."""
        try:
            for symbol, position in self.risk_manager.positions.items():
                # Get current market data
                market_data = self.data_feed.get_market_data(symbol)
                if not market_data:
                    continue
                
                current_price = market_data.get('last_price', 0)
                
                # Check for triggers
                action = self.risk_manager.check_stop_loss_take_profit(symbol, current_price)
                
                if action:
                    # Close position
                    success = self.risk_manager.close_position(symbol, current_price, action)
                    
                    if success:
                        logger.info(f"Closed position for {symbol} due to {action}")
            
        except Exception as e:
            logger.error(f"Error checking position triggers: {e}")
    
    async def _close_all_positions(self) -> None:
        """Close all open positions."""
        try:
            for symbol in list(self.risk_manager.positions.keys()):
                market_data = self.data_feed.get_market_data(symbol)
                if market_data:
                    current_price = market_data.get('last_price', 0)
                    self.risk_manager.close_position(symbol, current_price, "SHUTDOWN")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            if self.start_time:
                self.performance_metrics['uptime'] = (
                    datetime.now() - self.start_time
                ).total_seconds()
            
            # Get portfolio summary
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            self.performance_metrics['total_pnl'] = portfolio_summary.get('total_pnl', 0.0)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _log_final_metrics(self) -> None:
        """Log final performance metrics."""
        try:
            log_performance_metrics(self.performance_metrics)
            
            # Log portfolio summary
            portfolio_summary = self.risk_manager.get_portfolio_summary()
            log_performance_metrics(portfolio_summary)
            
        except Exception as e:
            logger.error(f"Error logging final metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get application status.
        
        Returns:
            Status dictionary
        """
        try:
            return {
                'is_running': self.is_running,
                'start_time': self.start_time,
                'uptime': self.performance_metrics['uptime'],
                'data_feed_status': self.data_feed.get_connection_status(),
                'portfolio_summary': self.risk_manager.get_portfolio_summary(),
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {}
    
    def scan_symbols(self, symbols: List[str]) -> None:
        """
        Update symbols to scan.
        
        Args:
            symbols: List of symbols to scan
        """
        try:
            self.data_feed.symbols = symbols
            logger.info(f"Updated scan symbols: {symbols}")
            
        except Exception as e:
            logger.error(f"Error updating scan symbols: {e}")


def main():
    parser = argparse.ArgumentParser(description='OptionsFlowX Trading System')
    parser.add_argument('--backtest', action='store_true', help='Run backtest on sample data')
    args = parser.parse_args()

    if args.backtest:
        from src.utils.data_loader import load_historical_data
        from src.core.backtester import Backtester, simple_rsi_ema_strategy
        import pandas as pd
        df = load_historical_data('sample_data.csv')
        # Add prev_close for strategy
        df['prev_close'] = df['close'].shift(1)
        df = df.dropna().reset_index(drop=True)
        backtester = Backtester(simple_rsi_ema_strategy, df)
        backtester.run()
        backtester.report()
        return
    try:
        # Create application
        app = OptionsFlowX()
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            asyncio.create_task(app.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start application
        asyncio.run(app.start())
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 