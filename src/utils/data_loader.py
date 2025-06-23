import pandas as pd

def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load historical OHLCV+VIX data from a CSV file.
    Expects columns: datetime, open, high, low, close, volume, vix
    """
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    return df 