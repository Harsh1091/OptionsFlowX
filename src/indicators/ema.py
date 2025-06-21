"""
Exponential Moving Average (EMA) indicator implementation.

This module provides EMA calculation with various optimization techniques
for high-frequency trading applications.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from numba import jit
from loguru import logger
from ..utils.config import get_indicators_config


class EMA:
    """
    Exponential Moving Average (EMA) indicator.
    
    EMA is a type of moving average that gives more weight to recent
    price data, making it more responsive to recent price changes.
    """
    
    def __init__(self, short_period: int = 9, long_period: int = 21, signal_period: int = 9):
        """
        Initialize EMA indicator.
        
        Args:
            short_period: Short EMA period (default: 9)
            long_period: Long EMA period (default: 21)
            signal_period: Signal line period (default: 9)
        """
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self._cache = {}
        
        # Load configuration
        config = get_indicators_config()
        ema_config = config.get('ema', {})
        
        self.short_period = ema_config.get('short_period', short_period)
        self.long_period = ema_config.get('long_period', long_period)
        self.signal_period = ema_config.get('signal_period', signal_period)
        
        logger.debug(f"EMA initialized with short={self.short_period}, long={self.long_period}, signal={self.signal_period}")
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_ema_fast(prices: np.ndarray, period: int) -> np.ndarray:
        """
        Fast EMA calculation using Numba JIT compilation.
        
        Args:
            prices: Array of prices
            period: EMA period
            
        Returns:
            Array of EMA values
        """
        n = len(prices)
        ema = np.full(n, np.nan)
        
        if n == 0:
            return ema
        
        # Calculate smoothing factor
        alpha = 2.0 / (period + 1)
        
        # Initialize EMA with first price
        ema[0] = prices[0]
        
        # Calculate EMA for remaining periods
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def calculate(self, prices: Union[pd.Series, np.ndarray, list], period: Optional[int] = None) -> np.ndarray:
        """
        Calculate EMA values for given price data.
        
        Args:
            prices: Price data (Series, array, or list)
            period: EMA period (if None, uses short_period)
            
        Returns:
            Array of EMA values
        """
        if isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        elif isinstance(prices, list):
            prices = np.array(prices)
        
        # Ensure prices is numpy array
        prices = np.asarray(prices)
        
        if period is None:
            period = self.short_period
        
        # Ensure period is valid
        period = int(period)  # type: ignore
        
        # Check cache for existing calculation
        cache_key = (hash(str(prices[-period:])), period)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Calculate EMA
        ema_values = self._calculate_ema_fast(prices, period)
        
        # Cache result
        self._cache[cache_key] = ema_values
        
        return ema_values
    
    def calculate_short_ema(self, prices: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Calculate short EMA values.
        
        Args:
            prices: Price data
            
        Returns:
            Array of short EMA values
        """
        return self.calculate(prices, self.short_period)
    
    def calculate_long_ema(self, prices: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Calculate long EMA values.
        
        Args:
            prices: Price data
            
        Returns:
            Array of long EMA values
        """
        return self.calculate(prices, self.long_period)
    
    def calculate_signal_line(self, ema_values: np.ndarray) -> np.ndarray:
        """
        Calculate signal line (EMA of EMA).
        
        Args:
            ema_values: EMA values
            
        Returns:
            Array of signal line values
        """
        return self.calculate(ema_values, self.signal_period)
    
    def get_crossover_signal(self, short_ema: np.ndarray, long_ema: np.ndarray) -> Tuple[str, float]:
        """
        Generate trading signal based on EMA crossover.
        
        Args:
            short_ema: Short EMA values
            long_ema: Long EMA values
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if len(short_ema) < 2 or len(long_ema) < 2:
            return "NEUTRAL", 0.0
        
        current_short = short_ema[-1]
        current_long = long_ema[-1]
        prev_short = short_ema[-2]
        prev_long = long_ema[-2]
        
        if np.isnan(current_short) or np.isnan(current_long):
            return "NEUTRAL", 0.0
        
        # Calculate signal strength based on distance between EMAs
        distance = abs(current_short - current_long) / current_long
        strength = min(distance * 10, 1.0)  # Scale distance to 0-1
        
        # Check for crossover
        if current_short > current_long and prev_short <= prev_long:
            return "BUY", strength
        elif current_short < current_long and prev_short >= prev_long:
            return "SELL", strength
        elif current_short > current_long:
            return "BULLISH", strength
        else:
            return "BEARISH", strength
    
    def get_trend_strength(self, short_ema: np.ndarray, long_ema: np.ndarray) -> float:
        """
        Calculate trend strength based on EMA separation.
        
        Args:
            short_ema: Short EMA values
            long_ema: Long EMA values
            
        Returns:
            Trend strength (0-1)
        """
        if len(short_ema) == 0 or len(long_ema) == 0:
            return 0.0
        
        current_short = short_ema[-1]
        current_long = long_ema[-1]
        
        if np.isnan(current_short) or np.isnan(current_long):
            return 0.0
        
        # Calculate percentage difference
        diff = abs(current_short - current_long) / current_long
        return min(diff * 5, 1.0)  # Scale to 0-1
    
    def get_support_resistance(self, prices: np.ndarray, ema_values: np.ndarray, 
                             lookback: int = 20) -> Tuple[float, float]:
        """
        Calculate support and resistance levels based on EMA.
        
        Args:
            prices: Price data
            ema_values: EMA values
            lookback: Number of periods to look back
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if len(prices) < lookback or len(ema_values) < lookback:
            return np.nan, np.nan
        
        recent_prices = prices[-lookback:]
        recent_ema = ema_values[-lookback:]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(recent_prices) | np.isnan(recent_ema))
        if not np.any(valid_mask):
            return np.nan, np.nan
        
        recent_prices = recent_prices[valid_mask]
        recent_ema = recent_ema[valid_mask]
        
        if len(recent_prices) == 0:
            return np.nan, np.nan
        
        # Calculate support and resistance
        support = float(np.percentile(recent_prices, 25))
        resistance = float(np.percentile(recent_prices, 75))
        
        return support, resistance
    
    def get_ema_ribbon(self, prices: np.ndarray, periods: Optional[List[int]] = None) -> dict:
        """
        Calculate EMA ribbon (multiple EMAs).
        
        Args:
            prices: Price data
            periods: List of EMA periods (default: [5, 10, 20, 50, 100, 200])
            
        Returns:
            Dictionary containing EMA values for each period
        """
        if periods is None:
            periods = [5, 10, 20, 50, 100, 200]
        
        ribbon = {}
        for period in periods:
            ribbon[f'EMA_{period}'] = self.calculate(prices, period)
        
        return ribbon
    
    def get_ribbon_signal(self, ribbon: dict) -> Tuple[str, float]:
        """
        Generate signal based on EMA ribbon alignment.
        
        Args:
            ribbon: Dictionary of EMA values
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if not ribbon:
            return "NEUTRAL", 0.0
        
        # Get current values
        current_values = {}
        for period, values in ribbon.items():
            if len(values) > 0:
                current_values[period] = values[-1]
        
        if len(current_values) < 3:
            return "NEUTRAL", 0.0
        
        # Check if EMAs are aligned (bullish: short > long, bearish: short < long)
        periods = sorted([int(p.split('_')[1]) for p in current_values.keys()])
        values = [current_values[f'EMA_{p}'] for p in periods]
        
        # Check for bullish alignment (ascending order)
        bullish_aligned = all(values[i] >= values[i-1] for i in range(1, len(values)))
        
        # Check for bearish alignment (descending order)
        bearish_aligned = all(values[i] <= values[i-1] for i in range(1, len(values)))
        
        # Calculate strength based on alignment quality
        if bullish_aligned:
            strength = self._calculate_alignment_strength(values, ascending=True)
            return "BULLISH", strength
        elif bearish_aligned:
            strength = self._calculate_alignment_strength(values, ascending=False)
            return "BEARISH", strength
        else:
            return "NEUTRAL", 0.0
    
    def _calculate_alignment_strength(self, values: List[float], ascending: bool = True) -> float:
        """
        Calculate the strength of EMA alignment.
        
        Args:
            values: EMA values
            ascending: Whether values should be in ascending order
            
        Returns:
            Alignment strength (0-1)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate average spacing between consecutive values
        spacings = []
        for i in range(1, len(values)):
            if ascending:
                spacing = (values[i] - values[i-1]) / values[i-1]
            else:
                spacing = (values[i-1] - values[i]) / values[i-1]
            spacings.append(max(0, spacing))
        
        # Return average spacing as strength
        return min(float(np.mean(spacings)) * 10, 1.0)
    
    def reset_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.debug("EMA cache cleared")
    
    def get_latest_value(self, prices: Union[pd.Series, np.ndarray, list], 
                        period: Optional[int] = None) -> float:
        """
        Get the latest EMA value.
        
        Args:
            prices: Price data
            period: EMA period (if None, uses short_period)
            
        Returns:
            Latest EMA value
        """
        ema_values = self.calculate(prices, period)
        return ema_values[-1] if len(ema_values) > 0 else np.nan
    
    def is_bullish(self, short_ema: float, long_ema: float) -> bool:
        """
        Check if EMA indicates bullish trend.
        
        Args:
            short_ema: Short EMA value
            long_ema: Long EMA value
            
        Returns:
            True if bullish, False otherwise
        """
        return short_ema > long_ema
    
    def is_bearish(self, short_ema: float, long_ema: float) -> bool:
        """
        Check if EMA indicates bearish trend.
        
        Args:
            short_ema: Short EMA value
            long_ema: Long EMA value
            
        Returns:
            True if bearish, False otherwise
        """
        return short_ema < long_ema 