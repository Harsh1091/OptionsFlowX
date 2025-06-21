"""
Options trading strategies for OptionsFlowX.

This module provides options-specific trading strategies including
straddles, strangles, spreads, and other options combinations.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from ..utils.config import get_config
from ..indicators import VIXIndia


class OptionsStrategy:
    """
    Options trading strategy implementation.
    
    Provides various options strategies based on market conditions,
    volatility, and technical signals.
    """
    
    def __init__(self):
        """Initialize options strategy."""
        self.config = get_config()
        self.vix_indicator = VIXIndia()
        
        logger.info("OptionsStrategy initialized")
    
    def get_strategy_recommendation(self, signal: Dict[str, Any], 
                                  market_data: Dict[str, Any],
                                  vix_value: float) -> Dict[str, Any]:
        """
        Get options strategy recommendation based on signal and market conditions.
        
        Args:
            signal: Trading signal
            market_data: Market data
            vix_value: Current VIX value
            
        Returns:
            Strategy recommendation dictionary
        """
        try:
            # Get volatility regime
            volatility_regime = self.vix_indicator.get_volatility_regime(vix_value)
            
            # Get market trend
            market_trend = self._determine_market_trend(market_data)
            
            # Generate strategy recommendation
            strategy = self._generate_strategy(signal, volatility_regime, market_trend)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendation: {e}")
            return {"strategy": "CASH", "reason": "Error in strategy generation"}
    
    def _determine_market_trend(self, market_data: Dict[str, Any]) -> str:
        """Determine current market trend."""
        try:
            # Simple trend determination based on price movement
            change_percent = market_data.get('change_percent', 0)
            
            if change_percent > 0.5:
                return "BULLISH"
            elif change_percent < -0.5:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Error determining market trend: {e}")
            return "NEUTRAL"
    
    def _generate_strategy(self, signal: Dict[str, Any], 
                          volatility_regime: str,
                          market_trend: str) -> Dict[str, Any]:
        """Generate options strategy based on conditions."""
        try:
            signal_type = signal.get('signal_type', 'NEUTRAL')
            signal_strength = signal.get('strength', 0.0)
            
            # Low volatility strategies
            if volatility_regime == "LOW_VOLATILITY":
                if signal_type == "BUY" and market_trend == "BULLISH":
                    return {
                        "strategy": "LONG_CALL",
                        "reason": "Low volatility bullish signal",
                        "risk_level": "LOW",
                        "expected_return": "MODERATE"
                    }
                elif signal_type == "SELL" and market_trend == "BEARISH":
                    return {
                        "strategy": "LONG_PUT",
                        "reason": "Low volatility bearish signal",
                        "risk_level": "LOW",
                        "expected_return": "MODERATE"
                    }
                else:
                    return {
                        "strategy": "IRON_CONDOR",
                        "reason": "Low volatility sideways market",
                        "risk_level": "LOW",
                        "expected_return": "LOW"
                    }
            
            # Normal volatility strategies
            elif volatility_regime == "NORMAL_VOLATILITY":
                if signal_type == "BUY" and signal_strength > 0.7:
                    return {
                        "strategy": "BULL_CALL_SPREAD",
                        "reason": "Strong bullish signal in normal volatility",
                        "risk_level": "MEDIUM",
                        "expected_return": "HIGH"
                    }
                elif signal_type == "SELL" and signal_strength > 0.7:
                    return {
                        "strategy": "BEAR_PUT_SPREAD",
                        "reason": "Strong bearish signal in normal volatility",
                        "risk_level": "MEDIUM",
                        "expected_return": "HIGH"
                    }
                else:
                    return {
                        "strategy": "STRADDLE",
                        "reason": "Normal volatility with mixed signals",
                        "risk_level": "MEDIUM",
                        "expected_return": "MODERATE"
                    }
            
            # High volatility strategies
            elif volatility_regime == "HIGH_VOLATILITY":
                if signal_type == "BUY":
                    return {
                        "strategy": "COVERED_CALL",
                        "reason": "High volatility bullish signal",
                        "risk_level": "HIGH",
                        "expected_return": "HIGH"
                    }
                elif signal_type == "SELL":
                    return {
                        "strategy": "PROTECTIVE_PUT",
                        "reason": "High volatility bearish signal",
                        "risk_level": "HIGH",
                        "expected_return": "HIGH"
                    }
                else:
                    return {
                        "strategy": "STRANGLE",
                        "reason": "High volatility with uncertainty",
                        "risk_level": "HIGH",
                        "expected_return": "HIGH"
                    }
            
            # Extreme volatility strategies
            else:  # EXTREME_VOLATILITY
                return {
                    "strategy": "CASH",
                    "reason": "Extreme volatility - stay in cash",
                    "risk_level": "EXTREME",
                    "expected_return": "NONE"
                }
                
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return {"strategy": "CASH", "reason": "Error in strategy generation"}
    
    def calculate_option_parameters(self, strategy: str, 
                                  current_price: float,
                                  days_to_expiry: int = 30) -> Dict[str, Any]:
        """
        Calculate option parameters for a given strategy.
        
        Args:
            strategy: Options strategy
            current_price: Current underlying price
            days_to_expiry: Days to option expiry
            
        Returns:
            Dictionary containing option parameters
        """
        try:
            if strategy == "LONG_CALL":
                return self._long_call_parameters(current_price, days_to_expiry)
            elif strategy == "LONG_PUT":
                return self._long_put_parameters(current_price, days_to_expiry)
            elif strategy == "BULL_CALL_SPREAD":
                return self._bull_call_spread_parameters(current_price, days_to_expiry)
            elif strategy == "BEAR_PUT_SPREAD":
                return self._bear_put_spread_parameters(current_price, days_to_expiry)
            elif strategy == "STRADDLE":
                return self._straddle_parameters(current_price, days_to_expiry)
            elif strategy == "STRANGLE":
                return self._strangle_parameters(current_price, days_to_expiry)
            elif strategy == "IRON_CONDOR":
                return self._iron_condor_parameters(current_price, days_to_expiry)
            else:
                return {"error": "Strategy not implemented"}
                
        except Exception as e:
            logger.error(f"Error calculating option parameters: {e}")
            return {"error": str(e)}
    
    def _long_call_parameters(self, current_price: float, 
                            days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for long call strategy."""
        try:
            # ATM call option
            strike_price = round(current_price / 50) * 50  # Round to nearest 50
            
            return {
                "strategy": "LONG_CALL",
                "strike_price": strike_price,
                "option_type": "CALL",
                "quantity": 1,
                "max_loss": "Premium paid",
                "max_profit": "Unlimited",
                "breakeven": strike_price + 50,  # Assuming 50 premium
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in long call parameters: {e}")
            return {}
    
    def _long_put_parameters(self, current_price: float, 
                           days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for long put strategy."""
        try:
            # ATM put option
            strike_price = round(current_price / 50) * 50  # Round to nearest 50
            
            return {
                "strategy": "LONG_PUT",
                "strike_price": strike_price,
                "option_type": "PUT",
                "quantity": 1,
                "max_loss": "Premium paid",
                "max_profit": strike_price - 50,  # Assuming 50 premium
                "breakeven": strike_price - 50,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in long put parameters: {e}")
            return {}
    
    def _bull_call_spread_parameters(self, current_price: float, 
                                   days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for bull call spread strategy."""
        try:
            # Buy lower strike call, sell higher strike call
            lower_strike = round(current_price / 50) * 50
            upper_strike = lower_strike + 100
            
            return {
                "strategy": "BULL_CALL_SPREAD",
                "buy_strike": lower_strike,
                "sell_strike": upper_strike,
                "buy_option_type": "CALL",
                "sell_option_type": "CALL",
                "quantity": 1,
                "max_loss": "Net premium paid",
                "max_profit": upper_strike - lower_strike - 30,  # Assuming 30 net premium
                "breakeven": lower_strike + 30,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in bull call spread parameters: {e}")
            return {}
    
    def _bear_put_spread_parameters(self, current_price: float, 
                                  days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for bear put spread strategy."""
        try:
            # Buy higher strike put, sell lower strike put
            upper_strike = round(current_price / 50) * 50
            lower_strike = upper_strike - 100
            
            return {
                "strategy": "BEAR_PUT_SPREAD",
                "buy_strike": upper_strike,
                "sell_strike": lower_strike,
                "buy_option_type": "PUT",
                "sell_option_type": "PUT",
                "quantity": 1,
                "max_loss": "Net premium paid",
                "max_profit": upper_strike - lower_strike - 30,  # Assuming 30 net premium
                "breakeven": upper_strike - 30,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in bear put spread parameters: {e}")
            return {}
    
    def _straddle_parameters(self, current_price: float, 
                           days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for straddle strategy."""
        try:
            # Buy ATM call and put
            strike_price = round(current_price / 50) * 50
            
            return {
                "strategy": "STRADDLE",
                "strike_price": strike_price,
                "call_quantity": 1,
                "put_quantity": 1,
                "max_loss": "Total premium paid",
                "max_profit": "Unlimited",
                "breakeven_up": strike_price + 100,  # Assuming 100 total premium
                "breakeven_down": strike_price - 100,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in straddle parameters: {e}")
            return {}
    
    def _strangle_parameters(self, current_price: float, 
                           days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for strangle strategy."""
        try:
            # Buy OTM call and put
            call_strike = round(current_price / 50) * 50 + 100
            put_strike = round(current_price / 50) * 50 - 100
            
            return {
                "strategy": "STRANGLE",
                "call_strike": call_strike,
                "put_strike": put_strike,
                "call_quantity": 1,
                "put_quantity": 1,
                "max_loss": "Total premium paid",
                "max_profit": "Unlimited",
                "breakeven_up": call_strike + 80,  # Assuming 80 total premium
                "breakeven_down": put_strike - 80,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in strangle parameters: {e}")
            return {}
    
    def _iron_condor_parameters(self, current_price: float, 
                              days_to_expiry: int) -> Dict[str, Any]:
        """Calculate parameters for iron condor strategy."""
        try:
            # Sell OTM call spread and OTM put spread
            put_short_strike = round(current_price / 50) * 50 - 50
            put_long_strike = put_short_strike - 100
            call_short_strike = round(current_price / 50) * 50 + 50
            call_long_strike = call_short_strike + 100
            
            return {
                "strategy": "IRON_CONDOR",
                "put_long_strike": put_long_strike,
                "put_short_strike": put_short_strike,
                "call_short_strike": call_short_strike,
                "call_long_strike": call_long_strike,
                "quantity": 1,
                "max_loss": 100 - 20,  # Spread width - net premium
                "max_profit": 20,  # Net premium received
                "breakeven_up": call_short_strike + 20,
                "breakeven_down": put_short_strike - 20,
                "days_to_expiry": days_to_expiry
            }
            
        except Exception as e:
            logger.error(f"Error in iron condor parameters: {e}")
            return {}
    
    def get_strategy_performance(self, strategy: str) -> Dict[str, Any]:
        """
        Get historical performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            
        Returns:
            Performance metrics dictionary
        """
        # This would typically query a database for historical performance
        # For now, return simulated data
        return {
            "strategy": strategy,
            "win_rate": 0.65,
            "avg_profit": 150,
            "avg_loss": -80,
            "profit_factor": 1.8,
            "max_drawdown": -200,
            "total_trades": 100
        } 