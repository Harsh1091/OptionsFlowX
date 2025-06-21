"""
Moving Average Convergence Divergence (MACD) indicator implementation.

This module provides MACD calculation with signal line and histogram
for trend analysis and momentum detection.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
from loguru import logger
from ..utils.config import get_indicators_config
from .ema import EMA


class MACD:
    """
    Moving Average Convergence Divergence (MACD) indicator.
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD indicator.
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._cache = {}
        
        # Initialize EMA calculator
        self.ema_calculator = EMA()
        
        # Load configuration
        config = get_indicators_config()
        macd_config = config.get('macd', {})
        
        self.fast_period = macd_config.get('fast_period', fast_period)
        self.slow_period = macd_config.get('slow_period', slow_period)
        self.signal_period = macd_config.get('signal_period', signal_period)
        
        logger.debug(f"MACD initialized with fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
    
    def calculate(self, prices: Union[pd.Series, np.ndarray, list]) -> Dict[str, np.ndarray]:
        """
        Calculate MACD values for given price data.
        
        Args:
            prices: Price data (Series, array, or list)
            
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        elif isinstance(prices, list):
            prices = np.array(prices)
        
        # Ensure prices is numpy array
        prices = np.asarray(prices)
        
        # Check cache for existing calculation
        cache_key = hash(str(prices[-max(self.fast_period, self.slow_period):]))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate fast and slow EMAs
        fast_ema = self.ema_calculator.calculate(prices, self.fast_period)
        slow_ema = self.ema_calculator.calculate(prices, self.slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self.ema_calculator.calculate(macd_line, self.signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        result = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def get_signal(self, macd_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Generate trading signal based on MACD.
        
        Args:
            macd_data: Dictionary containing MACD values
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        macd_line = macd_data['macd_line']
        signal_line = macd_data['signal_line']
        histogram = macd_data['histogram']
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return "NEUTRAL", 0.0
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        prev_histogram = histogram[-2]
        
        if np.isnan(current_macd) or np.isnan(current_signal):
            return "NEUTRAL", 0.0
        
        # Calculate signal strength based on histogram magnitude
        strength = min(abs(current_histogram) / abs(current_macd) if current_macd != 0 else 0, 1.0)
        
        # Check for MACD crossover
        if current_macd > current_signal and prev_macd <= prev_signal:
            return "BUY", strength
        elif current_macd < current_signal and prev_macd >= prev_signal:
            return "SELL", strength
        elif current_macd > current_signal:
            return "BULLISH", strength
        else:
            return "BEARISH", strength
    
    def get_divergence(self, prices: np.ndarray, macd_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect MACD divergence patterns.
        
        Args:
            prices: Price data
            macd_data: MACD values
            
        Returns:
            Dictionary containing divergence information
        """
        if len(prices) < 20 or len(macd_data['macd_line']) < 20:
            return {"divergence": False, "type": None, "strength": 0.0}
        
        macd_line = macd_data['macd_line']
        
        # Find peaks and troughs in prices and MACD
        price_peaks = self._find_peaks(prices)
        price_troughs = self._find_troughs(prices)
        macd_peaks = self._find_peaks(macd_line)
        macd_troughs = self._find_troughs(macd_line)
        
        # Check for bullish divergence (price lower lows, MACD higher lows)
        bullish_divergence = self._check_bullish_divergence(
            prices, macd_line, price_troughs, macd_troughs
        )
        
        # Check for bearish divergence (price higher highs, MACD lower highs)
        bearish_divergence = self._check_bearish_divergence(
            prices, macd_line, price_peaks, macd_peaks
        )
        
        if bullish_divergence:
            return {
                "divergence": True,
                "type": "BULLISH",
                "strength": bullish_divergence
            }
        elif bearish_divergence:
            return {
                "divergence": True,
                "type": "BEARISH",
                "strength": bearish_divergence
            }
        
        return {"divergence": False, "type": None, "strength": 0.0}
    
    def _find_peaks(self, data: np.ndarray) -> list:
        """Find peaks in data."""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray) -> list:
        """Find troughs in data."""
        troughs = []
        for i in range(1, len(data) - 1):
            if data[i] < data[i - 1] and data[i] < data[i + 1]:
                troughs.append(i)
        return troughs
    
    def _check_bullish_divergence(self, prices, macd_line, price_troughs, macd_troughs):
        """Check for bullish divergence."""
        if len(price_troughs) < 2 or len(macd_troughs) < 2:
            return 0.0
        
        # Check last two troughs
        price_trend = prices[price_troughs[-1]] < prices[price_troughs[-2]]
        macd_trend = macd_line[macd_troughs[-1]] > macd_line[macd_troughs[-2]]
        
        if price_trend and macd_trend:
            return min(1.0, abs(prices[price_troughs[-1]] - prices[price_troughs[-2]]) / prices[price_troughs[-2]])
        
        return 0.0
    
    def _check_bearish_divergence(self, prices, macd_line, price_peaks, macd_peaks):
        """Check for bearish divergence."""
        if len(price_peaks) < 2 or len(macd_peaks) < 2:
            return 0.0
        
        # Check last two peaks
        price_trend = prices[price_peaks[-1]] > prices[price_peaks[-2]]
        macd_trend = macd_line[macd_peaks[-1]] < macd_line[macd_peaks[-2]]
        
        if price_trend and macd_trend:
            return min(1.0, abs(prices[price_peaks[-1]] - prices[price_peaks[-2]]) / prices[price_peaks[-2]])
        
        return 0.0
    
    def get_histogram_signal(self, macd_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Generate signal based on histogram pattern.
        
        Args:
            macd_data: MACD values
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        histogram = macd_data['histogram']
        
        if len(histogram) < 3:
            return "NEUTRAL", 0.0
        
        current_hist = histogram[-1]
        prev_hist = histogram[-2]
        prev_prev_hist = histogram[-3]
        
        if np.isnan(current_hist) or np.isnan(prev_hist):
            return "NEUTRAL", 0.0
        
        # Calculate strength based on histogram magnitude
        strength = min(abs(current_hist) / 0.01, 1.0)  # Normalize to typical MACD values
        
        # Check for histogram reversal
        if current_hist > 0 and prev_hist < 0 and prev_prev_hist < 0:
            return "BUY", strength
        elif current_hist < 0 and prev_hist > 0 and prev_prev_hist > 0:
            return "SELL", strength
        elif current_hist > 0:
            return "BULLISH", strength
        else:
            return "BEARISH", strength
    
    def reset_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.debug("MACD cache cleared")
    
    def get_latest_values(self, prices: Union[pd.Series, np.ndarray, list]) -> Dict[str, float]:
        """
        Get the latest MACD values.
        
        Args:
            prices: Price data
            
        Returns:
            Dictionary containing latest MACD values
        """
        macd_data = self.calculate(prices)
        
        return {
            'macd_line': macd_data['macd_line'][-1] if len(macd_data['macd_line']) > 0 else np.nan,
            'signal_line': macd_data['signal_line'][-1] if len(macd_data['signal_line']) > 0 else np.nan,
            'histogram': macd_data['histogram'][-1] if len(macd_data['histogram']) > 0 else np.nan
        }
    
    def is_bullish(self, macd_data: Dict[str, np.ndarray]) -> bool:
        """
        Check if MACD indicates bullish trend.
        
        Args:
            macd_data: MACD values
            
        Returns:
            True if bullish, False otherwise
        """
        if len(macd_data['macd_line']) == 0:
            return False
        
        current_macd = macd_data['macd_line'][-1]
        current_signal = macd_data['signal_line'][-1]
        
        return current_macd > current_signal
    
    def is_bearish(self, macd_data: Dict[str, np.ndarray]) -> bool:
        """
        Check if MACD indicates bearish trend.
        
        Args:
            macd_data: MACD values
            
        Returns:
            True if bearish, False otherwise
        """
        if len(macd_data['macd_line']) == 0:
            return False
        
        current_macd = macd_data['macd_line'][-1]
        current_signal = macd_data['signal_line'][-1]
        
        return current_macd < current_signal 