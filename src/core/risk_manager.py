"""
Risk management module for OptionsFlowX.

This module provides position sizing and risk control functionality.
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
from ..utils.config import get_config, get_trading_config


class RiskManager:
    """
    Risk management system for OptionsFlowX.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize risk manager."""
        self.config = get_config()
        self.trading_config = get_trading_config()
        
        # Portfolio state
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.position_history = []
        
        # Risk limits
        self.max_positions = self.trading_config.get('max_positions', 5)
        self.risk_per_trade = self.trading_config.get('risk_per_trade', 0.02)
        
        logger.info(f"RiskManager initialized with capital: {initial_capital}")
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              current_price: float) -> Tuple[int, float]:
        """Calculate position size based on signal."""
        try:
            # Simple fixed percentage position sizing
            risk_amount = self.current_capital * self.risk_per_trade
            quantity = int(risk_amount / current_price)
            
            return quantity, risk_amount
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0, 0.0
    
    def can_open_position(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """Check if a new position can be opened."""
        try:
            # Check maximum positions limit
            if len(self.positions) >= self.max_positions:
                return False
            
            # Check if already have position in this symbol
            if symbol in self.positions:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position eligibility: {e}")
            return False
    
    def open_position(self, symbol: str, signal: Dict[str, Any], 
                     entry_price: float, quantity: int) -> bool:
        """Open a new trading position."""
        try:
            if not self.can_open_position(symbol, signal):
                return False
            
            # Create position
            position = {
                'symbol': symbol,
                'signal_type': signal['signal_type'],
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': datetime.now(),
                'status': 'OPEN',
                'pnl': 0.0
            }
            
            # Add to positions
            self.positions[symbol] = position
            
            logger.info(f"Opened position: {symbol} at {entry_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, 
                      reason: str = "MANUAL") -> bool:
        """Close an existing position."""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            
            # Calculate P&L
            if position['signal_type'] == "BUY":
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            # Update position
            position.update({
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'status': 'CLOSED',
                'pnl': pnl,
                'close_reason': reason
            })
            
            # Update capital
            self.current_capital += pnl
            
            # Move to history
            self.position_history.append(position.copy())
            del self.positions[symbol]
            
            logger.info(f"Closed position: {symbol} at {exit_price}, P&L: {pnl}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if stop-loss or take-profit is triggered for a position.
        Returns "STOP_LOSS", "TAKE_PROFIT", or None.
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        stop_loss_percent = self.trading_config.get('stop_loss_percent', 2.0)
        entry_price = position['entry_price']
        signal_type = position['signal_type']
        threshold = stop_loss_percent / 100.0

        if signal_type == "BUY":
            if current_price <= entry_price * (1 - threshold):
                return "STOP_LOSS"
            elif current_price >= entry_price * (1 + threshold):
                return "TAKE_PROFIT"
        elif signal_type == "SELL":
            if current_price >= entry_price * (1 + threshold):
                return "STOP_LOSS"
            elif current_price <= entry_price * (1 - threshold):
                return "TAKE_PROFIT"
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        try:
            # Calculate total P&L
            total_pnl = self.current_capital - self.initial_capital
            
            # Calculate win rate
            closed_positions = [pos for pos in self.position_history if pos['status'] == 'CLOSED']
            if closed_positions:
                winning_trades = len([pos for pos in closed_positions if pos['pnl'] > 0])
                win_rate = winning_trades / len(closed_positions)
            else:
                win_rate = 0.0
            
            summary = {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'total_pnl': total_pnl,
                'total_return': (total_pnl / self.initial_capital) * 100,
                'open_positions': len(self.positions),
                'closed_positions': len(closed_positions),
                'win_rate': win_rate
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {} 