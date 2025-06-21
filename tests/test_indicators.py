"""
Unit tests for technical indicators.

Tests RSI, EMA, VIX India, MACD, and Bollinger Bands indicators.
"""

import pytest
import numpy as np
from src.indicators import RSI, EMA, VIXIndia, MACD, BollingerBands


class TestRSI:
    """Test RSI indicator."""
    
    def setup_method(self):
        """Setup test data."""
        self.rsi = RSI(period=14)
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi_values = self.rsi.calculate(self.prices)
        assert len(rsi_values) == len(self.prices)
        assert not np.isnan(rsi_values[-1])
    
    def test_rsi_signal_generation(self):
        """Test RSI signal generation."""
        rsi_value = 75.0
        signal, strength = self.rsi.get_signal(rsi_value)
        assert signal in ["BUY", "SELL", "BULLISH", "BEARISH", "NEUTRAL"]
        assert 0.0 <= strength <= 1.0
    
    def test_rsi_overbought_oversold(self):
        """Test RSI overbought/oversold conditions."""
        assert self.rsi.is_overbought(80.0)
        assert not self.rsi.is_overbought(50.0)
        assert self.rsi.is_oversold(20.0)
        assert not self.rsi.is_oversold(50.0)


class TestEMA:
    """Test EMA indicator."""
    
    def setup_method(self):
        """Setup test data."""
        self.ema = EMA(short_period=9, long_period=21)
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    def test_ema_calculation(self):
        """Test EMA calculation."""
        ema_values = self.ema.calculate(self.prices, 9)
        assert len(ema_values) == len(self.prices)
        assert not np.isnan(ema_values[-1])
    
    def test_ema_crossover_signal(self):
        """Test EMA crossover signal."""
        short_ema = np.array([100, 101, 102, 103, 104])
        long_ema = np.array([99, 100, 101, 102, 103])
        signal, strength = self.ema.get_crossover_signal(short_ema, long_ema)
        assert signal in ["BUY", "SELL", "BULLISH", "BEARISH", "NEUTRAL"]
        assert 0.0 <= strength <= 1.0
    
    def test_ema_trend_detection(self):
        """Test EMA trend detection."""
        short_ema = 105.0
        long_ema = 100.0
        assert self.ema.is_bullish(short_ema, long_ema)
        assert not self.ema.is_bearish(short_ema, long_ema)


class TestVIXIndia:
    """Test VIX India indicator."""
    
    def setup_method(self):
        """Setup test data."""
        self.vix = VIXIndia(threshold=20.0)
    
    def test_vix_volatility_regime(self):
        """Test VIX volatility regime detection."""
        assert self.vix.get_volatility_regime(15.0) == "LOW_VOLATILITY"
        assert self.vix.get_volatility_regime(25.0) == "NORMAL_VOLATILITY"
        assert self.vix.get_volatility_regime(35.0) == "HIGH_VOLATILITY"
        assert self.vix.get_volatility_regime(45.0) == "EXTREME_VOLATILITY"
    
    def test_vix_signal_generation(self):
        """Test VIX signal generation."""
        signal, strength = self.vix.get_signal(25.0)
        assert signal in ["HIGH_VOLATILITY", "LOW_VOLATILITY"]
        assert 0.0 <= strength <= 1.0
    
    def test_vix_options_strategy(self):
        """Test VIX options strategy recommendation."""
        strategy = self.vix.get_options_strategy_signal(25.0, "BULLISH")
        assert "strategy" in strategy
        assert "risk_level" in strategy


class TestMACD:
    """Test MACD indicator."""
    
    def setup_method(self):
        """Setup test data."""
        self.macd = MACD(fast_period=12, slow_period=26)
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    def test_macd_calculation(self):
        """Test MACD calculation."""
        macd_data = self.macd.calculate(self.prices)
        assert "macd_line" in macd_data
        assert "signal_line" in macd_data
        assert "histogram" in macd_data
    
    def test_macd_signal_generation(self):
        """Test MACD signal generation."""
        macd_data = {
            'macd_line': np.array([0.1, 0.2]),
            'signal_line': np.array([0.05, 0.15]),
            'histogram': np.array([0.05, 0.05])
        }
        signal, strength = self.macd.get_signal(macd_data)
        assert signal in ["BUY", "SELL", "BULLISH", "BEARISH", "NEUTRAL"]
        assert 0.0 <= strength <= 1.0


class TestBollingerBands:
    """Test Bollinger Bands indicator."""
    
    def setup_method(self):
        """Setup test data."""
        self.bb = BollingerBands(period=20, std_dev=2.0)
        self.prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_data = self.bb.calculate(self.prices)
        assert "upper_band" in bb_data
        assert "middle_band" in bb_data
        assert "lower_band" in bb_data
    
    def test_bollinger_bands_signal(self):
        """Test Bollinger Bands signal generation."""
        bb_data = {
            'upper_band': np.array([110, 111]),
            'middle_band': np.array([105, 106]),
            'lower_band': np.array([100, 101])
        }
        signal, strength = self.bb.get_signal(self.prices, bb_data)
        assert signal in ["BUY", "SELL", "BULLISH", "BEARISH", "NEUTRAL"]
        assert 0.0 <= strength <= 1.0
    
    def test_bollinger_bands_squeeze(self):
        """Test Bollinger Bands squeeze detection."""
        bb_data = {
            'upper_band': np.array([105, 106]),
            'middle_band': np.array([100, 101]),
            'lower_band': np.array([95, 96])
        }
        squeeze_info = self.bb.get_squeeze_signal(bb_data)
        assert "squeeze" in squeeze_info
        assert "strength" in squeeze_info


if __name__ == "__main__":
    pytest.main([__file__]) 