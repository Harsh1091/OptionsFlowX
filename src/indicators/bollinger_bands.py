"""
Bollinger Bands indicator implementation.

This module provides Bollinger Bands calculation with upper, lower,
and middle bands for volatility and trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
from loguru import logger
from ..utils.config import get_indicators_config


class BollingerBands:
    """
    Bollinger Bands indicator.
    
    Bollinger Bands consist of a middle band (SMA) and upper/lower bands
    that are standard deviations away from the middle band.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands indicator.
        
        Args:
            period: Period for moving average calculation (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
        """
        self.period = period
        self.std_dev = std_dev
        self._cache = {}
        
        # Load configuration
        config = get_indicators_config()
        bb_config = config.get('bollinger_bands', {})
        
        self.period = bb_config.get('period', period)
        self.std_dev = bb_config.get('std_dev', std_dev)
        
        logger.debug(f"Bollinger Bands initialized with period={self.period}, std_dev={self.std_dev}")
    
    def calculate(self, prices: Union[pd.Series, np.ndarray, list]) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands for given price data.
        
        Args:
            prices: Price data (Series, array, or list)
            
        Returns:
            Dictionary containing upper, middle, and lower bands
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        elif isinstance(prices, list):
            prices = np.array(prices)
        
        # Check cache for existing calculation
        cache_key = hash(str(prices[-self.period:]))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        n = len(prices)
        upper_band = np.full(n, np.nan)
        middle_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        
        # Calculate bands
        for i in range(self.period - 1, n):
            window = prices[i - self.period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)
            
            middle_band[i] = sma
            upper_band[i] = sma + (self.std_dev * std)
            lower_band[i] = sma - (self.std_dev * std)
        
        result = {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def get_signal(self, prices: np.ndarray, bb_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Generate trading signal based on Bollinger Bands.
        
        Args:
            prices: Price data
            bb_data: Bollinger Bands data
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if len(prices) < 2 or len(bb_data['upper_band']) < 2:
            return "NEUTRAL", 0.0
        
        current_price = prices[-1]
        current_upper = bb_data['upper_band'][-1]
        current_lower = bb_data['lower_band'][-1]
        current_middle = bb_data['middle_band'][-1]
        
        prev_price = prices[-2]
        prev_upper = bb_data['upper_band'][-2]
        prev_lower = bb_data['lower_band'][-2]
        
        if np.isnan(current_price) or np.isnan(current_upper) or np.isnan(current_lower):
            return "NEUTRAL", 0.0
        
        # Calculate bandwidth (volatility measure)
        bandwidth = (current_upper - current_lower) / current_middle
        strength = min(bandwidth, 1.0)
        
        # Check for price position relative to bands
        if current_price >= current_upper:
            return "SELL", strength
        elif current_price <= current_lower:
            return "BUY", strength
        elif current_price > current_middle:
            return "BULLISH", strength
        else:
            return "BEARISH", strength
    
    def get_squeeze_signal(self, bb_data: Dict[str, np.ndarray], 
                          lookback: int = 20) -> Dict[str, Any]:
        """
        Detect Bollinger Band squeeze (low volatility period).
        
        Args:
            bb_data: Bollinger Bands data
            lookback: Number of periods to look back
            
        Returns:
            Dictionary containing squeeze information
        """
        if len(bb_data['upper_band']) < lookback:
            return {"squeeze": False, "strength": 0.0, "duration": 0}
        
        upper_band = bb_data['upper_band']
        lower_band = bb_data['lower_band']
        middle_band = bb_data['middle_band']
        
        # Calculate bandwidth for recent periods
        recent_bandwidths = []
        for i in range(max(0, len(upper_band) - lookback), len(upper_band)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]) and not np.isnan(middle_band[i]):
                bandwidth = (upper_band[i] - lower_band[i]) / middle_band[i]
                recent_bandwidths.append(bandwidth)
        
        if len(recent_bandwidths) < 5:
            return {"squeeze": False, "strength": 0.0, "duration": 0}
        
        # Calculate average bandwidth
        avg_bandwidth = np.mean(recent_bandwidths)
        
        # Check if current bandwidth is significantly lower than average
        current_bandwidth = recent_bandwidths[-1]
        squeeze_threshold = avg_bandwidth * 0.7  # 30% lower than average
        
        if current_bandwidth < squeeze_threshold:
            # Calculate squeeze strength
            strength = min((avg_bandwidth - current_bandwidth) / avg_bandwidth, 1.0)
            
            # Calculate squeeze duration
            duration = 0
            for i in range(len(recent_bandwidths) - 1, -1, -1):
                if recent_bandwidths[i] < squeeze_threshold:
                    duration += 1
                else:
                    break
            
            return {
                "squeeze": True,
                "strength": strength,
                "duration": duration,
                "current_bandwidth": current_bandwidth,
                "avg_bandwidth": avg_bandwidth
            }
        
        return {"squeeze": False, "strength": 0.0, "duration": 0}
    
    def get_percent_b(self, prices: np.ndarray, bb_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate %B indicator (price position within bands).
        
        Args:
            prices: Price data
            bb_data: Bollinger Bands data
            
        Returns:
            Array of %B values
        """
        upper_band = bb_data['upper_band']
        lower_band = bb_data['lower_band']
        
        percent_b = np.full(len(prices), np.nan)
        
        for i in range(len(prices)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                band_width = upper_band[i] - lower_band[i]
                if band_width != 0:
                    percent_b[i] = (prices[i] - lower_band[i]) / band_width
                else:
                    percent_b[i] = 0.5  # Middle of bands
        
        return percent_b
    
    def get_bandwidth(self, bb_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate bandwidth (volatility measure).
        
        Args:
            bb_data: Bollinger Bands data
            
        Returns:
            Array of bandwidth values
        """
        upper_band = bb_data['upper_band']
        lower_band = bb_data['lower_band']
        middle_band = bb_data['middle_band']
        
        bandwidth = np.full(len(upper_band), np.nan)
        
        for i in range(len(upper_band)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]) and not np.isnan(middle_band[i]):
                if middle_band[i] != 0:
                    bandwidth[i] = (upper_band[i] - lower_band[i]) / middle_band[i]
        
        return bandwidth
    
    def get_support_resistance(self, bb_data: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Get support and resistance levels from Bollinger Bands.
        
        Args:
            bb_data: Bollinger Bands data
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        if len(bb_data['lower_band']) == 0 or len(bb_data['upper_band']) == 0:
            return np.nan, np.nan
        
        current_lower = bb_data['lower_band'][-1]
        current_upper = bb_data['upper_band'][-1]
        
        return current_lower, current_upper
    
    def get_breakout_signal(self, prices: np.ndarray, bb_data: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """
        Detect breakout signals from Bollinger Bands.
        
        Args:
            prices: Price data
            bb_data: Bollinger Bands data
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if len(prices) < 2 or len(bb_data['upper_band']) < 2:
            return "NEUTRAL", 0.0
        
        current_price = prices[-1]
        prev_price = prices[-2]
        current_upper = bb_data['upper_band'][-1]
        current_lower = bb_data['lower_band'][-1]
        prev_upper = bb_data['upper_band'][-2]
        prev_lower = bb_data['lower_band'][-2]
        
        if np.isnan(current_price) or np.isnan(current_upper) or np.isnan(current_lower):
            return "NEUTRAL", 0.0
        
        # Check for upward breakout
        if prev_price <= prev_upper and current_price > current_upper:
            strength = min((current_price - current_upper) / current_upper, 1.0)
            return "BULLISH_BREAKOUT", strength
        
        # Check for downward breakout
        elif prev_price >= prev_lower and current_price < current_lower:
            strength = min((current_lower - current_price) / current_lower, 1.0)
            return "BEARISH_BREAKOUT", strength
        
        # Check for mean reversion
        elif current_price > current_upper:
            strength = min((current_price - current_upper) / current_upper, 1.0)
            return "MEAN_REVERSION_SELL", strength
        elif current_price < current_lower:
            strength = min((current_lower - current_price) / current_lower, 1.0)
            return "MEAN_REVERSION_BUY", strength
        
        return "NEUTRAL", 0.0
    
    def reset_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.debug("Bollinger Bands cache cleared")
    
    def get_latest_values(self, prices: Union[pd.Series, np.ndarray, list]) -> Dict[str, float]:
        """
        Get the latest Bollinger Bands values.
        
        Args:
            prices: Price data
            
        Returns:
            Dictionary containing latest band values
        """
        bb_data = self.calculate(prices)
        
        return {
            'upper_band': bb_data['upper_band'][-1] if len(bb_data['upper_band']) > 0 else np.nan,
            'middle_band': bb_data['middle_band'][-1] if len(bb_data['middle_band']) > 0 else np.nan,
            'lower_band': bb_data['lower_band'][-1] if len(bb_data['lower_band']) > 0 else np.nan
        }
    
    def is_overbought(self, price: float, bb_data: Dict[str, np.ndarray]) -> bool:
        """
        Check if price is overbought (above upper band).
        
        Args:
            price: Current price
            bb_data: Bollinger Bands data
            
        Returns:
            True if overbought, False otherwise
        """
        if len(bb_data['upper_band']) == 0:
            return False
        
        upper_band = bb_data['upper_band'][-1]
        return price >= upper_band
    
    def is_oversold(self, price: float, bb_data: Dict[str, np.ndarray]) -> bool:
        """
        Check if price is oversold (below lower band).
        
        Args:
            price: Current price
            bb_data: Bollinger Bands data
            
        Returns:
            True if oversold, False otherwise
        """
        if len(bb_data['lower_band']) == 0:
            return False
        
        lower_band = bb_data['lower_band'][-1]
        return price <= lower_band 