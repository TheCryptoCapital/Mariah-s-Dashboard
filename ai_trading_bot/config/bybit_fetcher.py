import requests
import pandas as pd

def get_bybit_ohlcv(symbol="BTCUSDT", interval="5", limit=1000):
    """
    Fetches OHLCV data from Bybit.
    Returns a pandas DataFrame with columns:
    timestamp, open, high, low, close, volume
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    rows = []
    for entry in data["result"]["list"]:
        t, o, h, l, c, v = entry[:6]
        rows.append({
            "timestamp": pd.to_datetime(int(t), unit="ms"),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v)
        })

    return pd.DataFrame(rows)
