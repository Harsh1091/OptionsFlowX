"""
Relative Strength Index (RSI) indicator implementation.

This module provides RSI calculation with various optimization techniques
for high-frequency trading applications.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numba import jit
from loguru import logger
from ..utils.config import get_indicators_config


class RSI:
    """
    Relative Strength Index (RSI) indicator.
    
    RSI is a momentum oscillator that measures the speed and magnitude
    of recent price changes to evaluate overbought or oversold conditions.
    """
    
    def __init__(self, period: int = 14, smoothing: int = 3):
        """
        Initialize RSI indicator.
        
        Args:
            period: Period for RSI calculation (default: 14)
            smoothing: Smoothing factor for RSI values (default: 3)
        """
        self.period = period
        self.smoothing = smoothing
        self._cache = {}
        
        # Load configuration
        config = get_indicators_config()
        rsi_config = config.get('rsi', {})
        
        self.overbought = rsi_config.get('overbought', 70)
        self.oversold = rsi_config.get('oversold', 30)
        
        logger.debug(f"RSI initialized with period={period}, smoothing={smoothing}")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi_fast(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Fast RSI calculation using Numba JIT compilation.
        
        Args:
            prices: Array of prices
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        n = len(prices)
        rsi = np.full(n, np.nan)
        
        if n < period + 1:
            return rsi
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Calculate RSI for first valid period
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        else:
            rsi[period] = 100
        
        # Calculate RSI for remaining periods using smoothing
        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
        
        return rsi
    
    def calculate(self, prices: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Calculate RSI values for given price data.
        
        Args:
            prices: Price data (Series, array, or list)
            
        Returns:
            Array of RSI values
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        elif isinstance(prices, list):
            prices = np.array(prices)
        
        # Ensure prices is numpy array
        prices = np.asarray(prices)
        
        # Check cache for existing calculation
        cache_key = hash(str(prices[-self.period:]))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate RSI
        rsi_values = self._calculate_rsi_fast(prices, self.period)
        
        # Apply smoothing if specified
        if self.smoothing > 1:
            rsi_values = self._apply_smoothing(rsi_values, self.smoothing)
        
        # Cache result
        self._cache[cache_key] = rsi_values
        
        return rsi_values
    
    def _apply_smoothing(self, rsi_values: np.ndarray, smoothing: int) -> np.ndarray:
        """
        Apply smoothing to RSI values.
        
        Args:
            rsi_values: Raw RSI values
            smoothing: Smoothing factor
            
        Returns:
            Smoothed RSI values
        """
        smoothed = np.full_like(rsi_values, np.nan)
        
        for i in range(smoothing - 1, len(rsi_values)):
            if not np.isnan(rsi_values[i - smoothing + 1:i + 1]).any():
                smoothed[i] = np.mean(rsi_values[i - smoothing + 1:i + 1])
        
        return smoothed
    
    def get_signal(self, rsi_value: float) -> Tuple[str, float]:
        """
        Generate trading signal based on RSI value.
        
        Args:
            rsi_value: Current RSI value
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if np.isnan(rsi_value):
            return "NEUTRAL", 0.0
        
        # Calculate signal strength based on distance from neutral (50)
        if rsi_value > 50:
            strength = min((rsi_value - 50) / 50, 1.0)
        else:
            strength = min((50 - rsi_value) / 50, 1.0)
        
        # Generate signal
        if rsi_value >= self.overbought:
            return "SELL", strength
        elif rsi_value <= self.oversold:
            return "BUY", strength
        elif rsi_value > 50:
            return "BULLISH", strength
        else:
            return "BEARISH", strength
    
    def get_divergence(self, prices: np.ndarray, rsi_values: np.ndarray) -> dict:
        """
        Detect RSI divergence patterns.
        
        Args:
            prices: Price data
            rsi_values: RSI values
            
        Returns:
            Dictionary containing divergence information
        """
        if len(prices) < 20 or len(rsi_values) < 20:
            return {"divergence": False, "type": None, "strength": 0.0}
        
        # Find peaks and troughs in prices and RSI
        price_peaks = self._find_peaks(prices)
        price_troughs = self._find_troughs(prices)
        rsi_peaks = self._find_peaks(rsi_values)
        rsi_troughs = self._find_troughs(rsi_values)
        
        # Check for bullish divergence (price lower lows, RSI higher lows)
        bullish_divergence = self._check_bullish_divergence(
            prices, rsi_values, price_troughs, rsi_troughs
        )
        
        # Check for bearish divergence (price higher highs, RSI lower highs)
        bearish_divergence = self._check_bearish_divergence(
            prices, rsi_values, price_peaks, rsi_peaks
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
    
    def _check_bullish_divergence(self, prices, rsi_values, price_troughs, rsi_troughs):
        """Check for bullish divergence."""
        if len(price_troughs) < 2 or len(rsi_troughs) < 2:
            return 0.0
        
        # Check last two troughs
        price_trend = prices[price_troughs[-1]] < prices[price_troughs[-2]]
        rsi_trend = rsi_values[rsi_troughs[-1]] > rsi_values[rsi_troughs[-2]]
        
        if price_trend and rsi_trend:
            return min(1.0, abs(prices[price_troughs[-1]] - prices[price_troughs[-2]]) / prices[price_troughs[-2]])
        
        return 0.0
    
    def _check_bearish_divergence(self, prices, rsi_values, price_peaks, rsi_peaks):
        """Check for bearish divergence."""
        if len(price_peaks) < 2 or len(rsi_peaks) < 2:
            return 0.0
        
        # Check last two peaks
        price_trend = prices[price_peaks[-1]] > prices[price_peaks[-2]]
        rsi_trend = rsi_values[rsi_peaks[-1]] < rsi_values[rsi_peaks[-2]]
        
        if price_trend and rsi_trend:
            return min(1.0, abs(prices[price_peaks[-1]] - prices[price_peaks[-2]]) / prices[price_peaks[-2]])
        
        return 0.0
    
    def reset_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.debug("RSI cache cleared")
    
    def get_latest_value(self, prices: Union[pd.Series, np.ndarray, list]) -> float:
        """
        Get the latest RSI value.
        
        Args:
            prices: Price data
            
        Returns:
            Latest RSI value
        """
        rsi_values = self.calculate(prices)
        return rsi_values[-1] if len(rsi_values) > 0 else np.nan
    
    def is_overbought(self, rsi_value: float) -> bool:
        """
        Check if RSI indicates overbought condition.
        
        Args:
            rsi_value: RSI value
            
        Returns:
            True if overbought, False otherwise
        """
        return rsi_value >= self.overbought
    
    def is_oversold(self, rsi_value: float) -> bool:
        """
        Check if RSI indicates oversold condition.
        
        Args:
            rsi_value: RSI value
            
        Returns:
            True if oversold, False otherwise
        """
        return rsi_value <= self.oversold 