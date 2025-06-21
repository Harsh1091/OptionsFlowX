"""
VIX India indicator implementation.

This module provides VIX (Volatility Index) calculation and analysis
specifically for Indian markets, focusing on NIFTY and BANKNIFTY.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict, Any
from loguru import logger
from ..utils.config import get_indicators_config


class VIXIndia:
    """
    VIX India indicator for volatility analysis.
    
    VIX (Volatility Index) measures market volatility and is often
    called the "fear gauge" of the market.
    """
    
    def __init__(self, threshold: float = 20.0, lookback_period: int = 30, 
                 volatility_multiplier: float = 1.5):
        """
        Initialize VIX India indicator.
        
        Args:
            threshold: VIX threshold for signal generation (default: 20.0)
            lookback_period: Period for historical analysis (default: 30)
            volatility_multiplier: Multiplier for volatility calculations (default: 1.5)
        """
        self.threshold = threshold
        self.lookback_period = lookback_period
        self.volatility_multiplier = volatility_multiplier
        self._cache = {}
        
        # Load configuration
        config = get_indicators_config()
        vix_config = config.get('vix', {})
        
        self.threshold = vix_config.get('threshold', threshold)
        self.lookback_period = vix_config.get('lookback_period', lookback_period)
        self.volatility_multiplier = vix_config.get('volatility_multiplier', volatility_multiplier)
        
        logger.debug(f"VIX India initialized with threshold={self.threshold}, lookback={self.lookback_period}")
    
    def calculate_implied_volatility(self, option_prices: np.ndarray, 
                                   strike_prices: np.ndarray,
                                   spot_price: float,
                                   time_to_expiry: float,
                                   risk_free_rate: float = 0.05) -> np.ndarray:
        """
        Calculate implied volatility using Black-Scholes model.
        
        Args:
            option_prices: Array of option prices
            strike_prices: Array of strike prices
            spot_price: Current spot price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Array of implied volatilities
        """
        implied_vols = np.full(len(option_prices), np.nan)
        
        for i, (option_price, strike) in enumerate(zip(option_prices, strike_prices)):
            try:
                implied_vols[i] = self._black_scholes_implied_volatility(
                    option_price, spot_price, strike, time_to_expiry, risk_free_rate
                )
            except Exception as e:
                logger.debug(f"Failed to calculate implied volatility for strike {strike}: {e}")
                continue
        
        return implied_vols
    
    def _black_scholes_implied_volatility(self, option_price: float, spot_price: float,
                                        strike_price: float, time_to_expiry: float,
                                        risk_free_rate: float) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Option price
            spot_price: Current spot price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Implied volatility
        """
        # Initial guess for volatility
        sigma = 0.3
        
        for _ in range(100):  # Maximum 100 iterations
            # Calculate option price with current volatility
            calculated_price = self._black_scholes_call_price(
                spot_price, strike_price, time_to_expiry, risk_free_rate, sigma
            )
            
            # Calculate vega (derivative with respect to volatility)
            vega = self._black_scholes_vega(
                spot_price, strike_price, time_to_expiry, risk_free_rate, sigma
            )
            
            if abs(vega) < 1e-10:
                break
            
            # Newton-Raphson update
            sigma_new = sigma - (calculated_price - option_price) / vega
            
            # Check convergence
            if abs(sigma_new - sigma) < 1e-6:
                sigma = sigma_new
                break
            
            sigma = max(0.001, sigma_new)  # Ensure positive volatility
        
        return sigma
    
    def _black_scholes_call_price(self, spot_price: float, strike_price: float,
                                 time_to_expiry: float, risk_free_rate: float,
                                 volatility: float) -> float:
        """
        Calculate Black-Scholes call option price.
        
        Args:
            spot_price: Current spot price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility
            
        Returns:
            Call option price
        """
        d1 = (np.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        call_price = (spot_price * self._normal_cdf(d1) - 
                     strike_price * np.exp(-risk_free_rate * time_to_expiry) * self._normal_cdf(d2))
        
        return call_price
    
    def _black_scholes_vega(self, spot_price: float, strike_price: float,
                           time_to_expiry: float, risk_free_rate: float,
                           volatility: float) -> float:
        """
        Calculate Black-Scholes vega (derivative with respect to volatility).
        
        Args:
            spot_price: Current spot price
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free interest rate
            volatility: Volatility
            
        Returns:
            Vega value
        """
        d1 = (np.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        
        vega = spot_price * np.sqrt(time_to_expiry) * self._normal_pdf(d1)
        
        return vega
    
    def _normal_cdf(self, x: float) -> float:
        """Calculate cumulative distribution function of standard normal distribution."""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Calculate probability density function of standard normal distribution."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def calculate_vix(self, implied_volatilities: np.ndarray, 
                     weights: Optional[np.ndarray] = None) -> float:
        """
        Calculate VIX from implied volatilities.
        
        Args:
            implied_volatilities: Array of implied volatilities
            weights: Weights for different strikes (if None, equal weights)
            
        Returns:
            VIX value
        """
        if len(implied_volatilities) == 0:
            return np.nan
        
        # Remove NaN values
        valid_mask = ~np.isnan(implied_volatilities)
        if not np.any(valid_mask):
            return np.nan
        
        valid_vols = implied_volatilities[valid_mask]
        
        if weights is not None:
            valid_weights = weights[valid_mask]
            if len(valid_weights) != len(valid_vols):
                return np.nan
            # Weighted average
            vix = np.average(valid_vols, weights=valid_weights)
        else:
            # Simple average
            vix = np.mean(valid_vols)
        
        return vix * 100  # Convert to percentage
    
    def get_volatility_regime(self, vix_value: float) -> str:
        """
        Determine volatility regime based on VIX value.
        
        Args:
            vix_value: Current VIX value
            
        Returns:
            Volatility regime description
        """
        if np.isnan(vix_value):
            return "UNKNOWN"
        
        if vix_value < 15:
            return "LOW_VOLATILITY"
        elif vix_value < 25:
            return "NORMAL_VOLATILITY"
        elif vix_value < 35:
            return "HIGH_VOLATILITY"
        else:
            return "EXTREME_VOLATILITY"
    
    def get_signal(self, vix_value: float, historical_vix: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """
        Generate trading signal based on VIX value.
        
        Args:
            vix_value: Current VIX value
            historical_vix: Historical VIX values for comparison
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if np.isnan(vix_value):
            return "NEUTRAL", 0.0
        
        # Calculate signal strength based on deviation from threshold
        if vix_value > self.threshold:
            strength = min((vix_value - self.threshold) / self.threshold, 1.0)
            return "HIGH_VOLATILITY", strength
        else:
            strength = min((self.threshold - vix_value) / self.threshold, 1.0)
            return "LOW_VOLATILITY", strength
    
    def get_volatility_spike(self, vix_values: np.ndarray, 
                           spike_threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect volatility spikes in VIX data.
        
        Args:
            vix_values: Array of VIX values
            spike_threshold: Threshold for spike detection (standard deviations)
            
        Returns:
            Dictionary containing spike information
        """
        if len(vix_values) < 20:
            return {"spike_detected": False, "magnitude": 0.0, "direction": None}
        
        # Calculate rolling mean and standard deviation
        rolling_mean = np.convolve(vix_values, np.ones(20)/20, mode='valid')
        rolling_std = np.array([np.std(vix_values[max(0, i-19):i+1]) 
                               for i in range(19, len(vix_values))])
        
        if len(rolling_mean) == 0 or len(rolling_std) == 0:
            return {"spike_detected": False, "magnitude": 0.0, "direction": None}
        
        current_vix = vix_values[-1]
        current_mean = rolling_mean[-1]
        current_std = rolling_std[-1]
        
        if current_std == 0:
            return {"spike_detected": False, "magnitude": 0.0, "direction": None}
        
        # Calculate z-score
        z_score = (current_vix - current_mean) / current_std
        
        if abs(z_score) > spike_threshold:
            magnitude = min(abs(z_score) / spike_threshold, 1.0)
            direction = "UP" if z_score > 0 else "DOWN"
            return {
                "spike_detected": True,
                "magnitude": magnitude,
                "direction": direction,
                "z_score": z_score
            }
        
        return {"spike_detected": False, "magnitude": 0.0, "direction": None}
    
    def get_options_strategy_signal(self, vix_value: float, 
                                  market_trend: str = "NEUTRAL") -> Dict[str, Any]:
        """
        Generate options trading strategy signal based on VIX and market trend.
        
        Args:
            vix_value: Current VIX value
            market_trend: Current market trend (BULLISH, BEARISH, NEUTRAL)
            
        Returns:
            Dictionary containing strategy recommendations
        """
        volatility_regime = self.get_volatility_regime(vix_value)
        
        strategy = {
            "volatility_regime": volatility_regime,
            "vix_value": vix_value,
            "market_trend": market_trend,
            "recommended_strategies": [],
            "risk_level": "MEDIUM"
        }
        
        # Strategy recommendations based on volatility regime and market trend
        if volatility_regime == "LOW_VOLATILITY":
            if market_trend == "BULLISH":
                strategy["recommended_strategies"] = ["LONG_CALL", "BULL_CALL_SPREAD"]
                strategy["risk_level"] = "LOW"
            elif market_trend == "BEARISH":
                strategy["recommended_strategies"] = ["LONG_PUT", "BEAR_PUT_SPREAD"]
                strategy["risk_level"] = "LOW"
            else:
                strategy["recommended_strategies"] = ["IRON_CONDOR", "BUTTERFLY_SPREAD"]
                strategy["risk_level"] = "LOW"
        
        elif volatility_regime == "NORMAL_VOLATILITY":
            if market_trend == "BULLISH":
                strategy["recommended_strategies"] = ["LONG_CALL", "COVERED_CALL"]
                strategy["risk_level"] = "MEDIUM"
            elif market_trend == "BEARISH":
                strategy["recommended_strategies"] = ["LONG_PUT", "PROTECTIVE_PUT"]
                strategy["risk_level"] = "MEDIUM"
            else:
                strategy["recommended_strategies"] = ["STRADDLE", "STRANGLE"]
                strategy["risk_level"] = "MEDIUM"
        
        elif volatility_regime == "HIGH_VOLATILITY":
            if market_trend == "BULLISH":
                strategy["recommended_strategies"] = ["BULL_CALL_SPREAD", "COVERED_CALL"]
                strategy["risk_level"] = "HIGH"
            elif market_trend == "BEARISH":
                strategy["recommended_strategies"] = ["BEAR_PUT_SPREAD", "PROTECTIVE_PUT"]
                strategy["risk_level"] = "HIGH"
            else:
                strategy["recommended_strategies"] = ["IRON_CONDOR", "CALENDAR_SPREAD"]
                strategy["risk_level"] = "HIGH"
        
        else:  # EXTREME_VOLATILITY
            strategy["recommended_strategies"] = ["CASH", "DEFENSIVE_POSITIONS"]
            strategy["risk_level"] = "EXTREME"
        
        return strategy
    
    def reset_cache(self) -> None:
        """Clear the calculation cache."""
        self._cache.clear()
        logger.debug("VIX India cache cleared")
    
    def get_latest_value(self, implied_volatilities: np.ndarray, 
                        weights: Optional[np.ndarray] = None) -> float:
        """
        Get the latest VIX value.
        
        Args:
            implied_volatilities: Array of implied volatilities
            weights: Weights for different strikes
            
        Returns:
            Latest VIX value
        """
        return self.calculate_vix(implied_volatilities, weights)
    
    def is_high_volatility(self, vix_value: float) -> bool:
        """
        Check if VIX indicates high volatility.
        
        Args:
            vix_value: VIX value
            
        Returns:
            True if high volatility, False otherwise
        """
        return vix_value >= self.threshold
    
    def is_low_volatility(self, vix_value: float) -> bool:
        """
        Check if VIX indicates low volatility.
        
        Args:
            vix_value: VIX value
            
        Returns:
            True if low volatility, False otherwise
        """
        return vix_value < self.threshold 