import pandas as pd
from typing import Callable, Dict, Any

class Backtester:
    def __init__(self, strategy: Callable, data: pd.DataFrame, initial_capital: float = 100000):
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.results = None

    def run(self, **strategy_kwargs):
        capital = self.initial_capital
        position = 0  # 1 for long, -1 for short, 0 for flat
        entry_price = 0
        trades = []
        equity_curve = [capital]

        for i, row in self.data.iterrows():
            signal = self.strategy(row, position, **strategy_kwargs)
            price = float(row['close'])
            if signal == 'buy' and position == 0:
                position = 1
                entry_price = price
                trades.append({'type': 'buy', 'price': price, 'datetime': row['datetime']})
            elif signal == 'sell' and position == 1:
                pnl = price - entry_price
                capital += pnl
                position = 0
                trades.append({'type': 'sell', 'price': price, 'datetime': row['datetime'], 'pnl': pnl})
            equity = capital + (price - entry_price if position == 1 else 0)
            equity_curve.append(float(equity))

        # Close any open position at the end
        if position == 1:
            price = self.data.iloc[-1]['close']
            pnl = price - entry_price
            capital += pnl
            trades.append({'type': 'sell', 'price': price, 'datetime': self.data.iloc[-1]['datetime'], 'pnl': pnl})

        self.results = {
            'final_capital': capital,
            'total_pnl': capital - self.initial_capital,
            'trades': trades,
            'equity_curve': equity_curve
        }
        return self.results

    def report(self):
        if not self.results:
            print('No results. Run the backtest first.')
            return
        print(f"Initial Capital: {self.initial_capital}")
        print(f"Final Capital: {self.results['final_capital']}")
        print(f"Total PnL: {self.results['total_pnl']}")
        print(f"Number of Trades: {len(self.results['trades'])//2}")
        wins = [t for t in self.results['trades'] if t.get('pnl', 0) > 0]
        print(f"Win Rate: {len(wins) / max(1, (len(self.results['trades'])//2)) * 100:.2f}%")

# Example simple strategy for demonstration

def simple_rsi_ema_strategy(row, position, rsi_period=2, ema_period=3):
    # This is a placeholder. In real use, you would calculate RSI/EMA over a window.
    # For demo, use price changes as a proxy for signals.
    # Buy if price increased from previous, sell if decreased.
    # You can replace this with real indicator logic.
    if 'prev_close' in row and row['close'] > row['prev_close']:
        if position == 0:
            return 'buy'
        else:
            return None
    elif 'prev_close' in row and row['close'] < row['prev_close']:
        if position == 1:
            return 'sell'
        else:
            return None
    return None 