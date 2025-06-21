"""
Signal processing engine for OptionsFlowX.

This module provides signal generation and filtering capabilities
using multiple technical indicators and advanced algorithms.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from loguru import logger
from ..utils.config import get_config, get_signal_processing_config
from ..indicators import RSI, EMA, VIXIndia, MACD, BollingerBands
from ..utils.logger import log_trading_signal


class SignalProcessor:
    """
    Signal processing engine for OptionsFlowX.
    
    Combines multiple technical indicators to generate high-quality
    trading signals with advanced filtering algorithms.
    """
    
    def __init__(self):
        """Initialize signal processor."""
        self.config = get_config()
        self.signal_config = get_signal_processing_config()
        
        # Initialize indicators
        self.rsi = RSI()
        self.ema = EMA()
        self.vix = VIXIndia()
        self.macd = MACD()
        self.bb = BollingerBands()
        
        # Signal storage
        self.signals = []
        self.signal_history = []
        
        # Performance tracking
        self.signal_stats: Dict[str, Any] = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'successful_signals': 0,
            'false_signals': 0
        }
        
        logger.info("SignalProcessor initialized")
    
    def process_market_data(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process market data and generate signals.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Signal dictionary or None if no signal
        """
        try:
            # Extract price data
            prices = self._extract_price_data(market_data)
            if not prices or len(prices) < 50:  # Need minimum data
                return None
            
            # Calculate indicators
            indicators = self._calculate_indicators(prices)
            
            # Generate signals
            signals = self._generate_signals(symbol, prices, indicators)
            
            # Filter signals
            filtered_signals = self._filter_signals(signals, market_data)
            
            # Combine signals
            final_signal = self._combine_signals(filtered_signals)
            
            if final_signal:
                # Add metadata
                final_signal.update({
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'price': market_data.get('last_price', 0),
                    'volume': market_data.get('volume', 0)
                })
                
                # Log signal
                log_trading_signal(final_signal)
                
                # Store signal
                self.signals.append(final_signal)
                self.signal_history.append(final_signal)
                
                # Update stats
                self._update_signal_stats(final_signal)
                
                return final_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing market data for {symbol}: {e}")
            return None
    
    def _extract_price_data(self, market_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract price data from market data."""
        # In a real implementation, this would extract historical prices
        # For now, we'll simulate price data
        import random
        
        base_price = market_data.get('last_price', 18000.0)
        prices = []
        
        # Generate simulated historical prices
        for i in range(100):
            change = random.gauss(0, base_price * 0.001)  # 0.1% standard deviation
            price = base_price + change
            prices.append(price)
            base_price = price
        
        return prices
    
    def _calculate_indicators(self, prices: List[float]) -> Dict[str, Any]:
        """Calculate all technical indicators."""
        try:
            prices_array = np.array(prices)
            
            indicators = {
                'rsi': self.rsi.calculate(prices_array),
                'ema_short': self.ema.calculate_short_ema(prices_array),
                'ema_long': self.ema.calculate_long_ema(prices_array),
                'macd': self.macd.calculate(prices_array),
                'bb': self.bb.calculate(prices_array)
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _generate_signals(self, symbol: str, prices: List[float], 
                         indicators: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate signals from indicators."""
        signals = []
        
        try:
            # RSI signals
            rsi_values = indicators.get('rsi', [])
            if len(rsi_values) > 0:
                rsi_signal, rsi_strength = self.rsi.get_signal(rsi_values[-1])
                if rsi_signal != "NEUTRAL":
                    signals.append({
                        'indicator': 'RSI',
                        'signal_type': rsi_signal,
                        'strength': rsi_strength,
                        'value': rsi_values[-1]
                    })
            
            # EMA signals
            ema_short = indicators.get('ema_short', [])
            ema_long = indicators.get('ema_long', [])
            if len(ema_short) > 0 and len(ema_long) > 0:
                ema_signal, ema_strength = self.ema.get_crossover_signal(ema_short, ema_long)
                if ema_signal != "NEUTRAL":
                    signals.append({
                        'indicator': 'EMA',
                        'signal_type': ema_signal,
                        'strength': ema_strength,
                        'short_ema': ema_short[-1],
                        'long_ema': ema_long[-1]
                    })
            
            # MACD signals
            macd_data = indicators.get('macd', {})
            if macd_data:
                macd_signal, macd_strength = self.macd.get_signal(macd_data)
                if macd_signal != "NEUTRAL":
                    signals.append({
                        'indicator': 'MACD',
                        'signal_type': macd_signal,
                        'strength': macd_strength,
                        'macd_line': macd_data.get('macd_line', [0])[-1],
                        'signal_line': macd_data.get('signal_line', [0])[-1]
                    })
            
            # Bollinger Bands signals
            bb_data = indicators.get('bb', {})
            if bb_data:
                prices_array = np.array(prices)
                bb_signal, bb_strength = self.bb.get_signal(prices_array, bb_data)
                if bb_signal != "NEUTRAL":
                    signals.append({
                        'indicator': 'BB',
                        'signal_type': bb_signal,
                        'strength': bb_strength,
                        'upper_band': bb_data.get('upper_band', [0])[-1],
                        'lower_band': bb_data.get('lower_band', [0])[-1]
                    })
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _filter_signals(self, signals: List[Dict[str, Any]], 
                       market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter signals based on various criteria."""
        filtered_signals = []
        
        min_strength = self.signal_config.get('min_signal_strength', 0.7)
        
        for signal in signals:
            # Filter by signal strength
            if signal.get('strength', 0) < min_strength:
                continue
            
            # Filter by volume confirmation
            if self.signal_config.get('volume_confirmation', True):
                volume = market_data.get('volume', 0)
                if volume < 1000:  # Minimum volume threshold
                    continue
            
            # Filter by time (avoid lunch hours)
            if self.signal_config.get('time_filter', True):
                if not self._is_good_trading_time():
                    continue
            
            # Filter by volatility
            if self.signal_config.get('volatility_filter', True):
                if not self._check_volatility_conditions(signal):
                    continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def _combine_signals(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Combine multiple signals into a final signal."""
        if not signals:
            return None
        
        # Count signal types
        buy_signals = [s for s in signals if s['signal_type'] in ['BUY', 'BULLISH']]
        sell_signals = [s for s in signals if s['signal_type'] in ['SELL', 'BEARISH']]
        
        # Calculate average strength
        avg_strength = sum(s.get('strength', 0) for s in signals) / len(signals)
        
        # Determine final signal type
        if len(buy_signals) > len(sell_signals):
            signal_type = "BUY"
            confidence = len(buy_signals) / len(signals)
        elif len(sell_signals) > len(buy_signals):
            signal_type = "SELL"
            confidence = len(sell_signals) / len(signals)
        else:
            return None  # Conflicting signals
        
        # Create final signal
        final_signal = {
            'signal_type': signal_type,
            'strength': avg_strength,
            'confidence': confidence,
            'indicators_used': [s['indicator'] for s in signals],
            'supporting_signals': signals
        }
        
        return final_signal
    
    def _is_good_trading_time(self) -> bool:
        """Check if current time is good for trading."""
        now = datetime.now()
        
        # Avoid lunch hours (12:00-13:00)
        lunch_start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        lunch_end = now.replace(hour=13, minute=0, second=0, microsecond=0)
        
        if lunch_start <= now <= lunch_end:
            return False
        
        return True
    
    def _check_volatility_conditions(self, signal: Dict[str, Any]) -> bool:
        """Check if volatility conditions are met."""
        # Implement volatility checks
        return True
    
    def _update_signal_stats(self, signal: Dict[str, Any]) -> None:
        """Update signal statistics."""
        self.signal_stats['total_signals'] += 1
        
        signal_type = signal.get('signal_type', '')
        if signal_type in ['BUY', 'BULLISH']:
            self.signal_stats['buy_signals'] += 1
        elif signal_type in ['SELL', 'BEARISH']:
            self.signal_stats['sell_signals'] += 1
    
    def get_signal_history(self, symbol: Optional[str] = None, 
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get signal history.
        
        Args:
            symbol: Filter by symbol (optional)
            limit: Maximum number of signals to return
            
        Returns:
            List of signal dictionaries
        """
        history = self.signal_history
        
        if symbol:
            history = [s for s in history if s.get('symbol') == symbol]
        
        return history[-limit:]
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """
        Get signal statistics.
        
        Returns:
            Dictionary containing signal statistics
        """
        stats = self.signal_stats.copy()
        
        if stats['total_signals'] > 0:
            stats['success_rate'] = stats['successful_signals'] / stats['total_signals']
            stats['false_signal_rate'] = stats['false_signals'] / stats['total_signals']
        else:
            stats['success_rate'] = 0.0
            stats['false_signal_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset signal statistics."""
        self.signal_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'successful_signals': 0,
            'false_signals': 0
        }
        logger.info("Signal statistics reset")
    
    def mark_signal_result(self, signal_id: str, success: bool) -> None:
        """
        Mark a signal as successful or failed.
        
        Args:
            signal_id: Signal identifier
            success: True if signal was successful, False otherwise
        """
        # Find signal in history and update result
        for signal in self.signal_history:
            if signal.get('id') == signal_id:
                signal['result'] = 'SUCCESS' if success else 'FAILED'
                signal['result_timestamp'] = datetime.now()
                
                if success:
                    self.signal_stats['successful_signals'] += 1
                else:
                    self.signal_stats['false_signals'] += 1
                
                break 