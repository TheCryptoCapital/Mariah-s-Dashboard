from datetime import datetime

class MarketContext:
    """
    Extracts: session, weekend, high_vol/low_liq, and event windows.
    """
    def __init__(self, df):
        self.df = df
        self.flags = {}
        self.compute()

    def compute(self):
        ts = self.df["timestamp"].iloc[-1]
        self.flags["is_weekend"]      = ts.weekday() >= 5
        h = ts.hour
        self.flags["session"]         = (
            "Asia" if 2 <= h < 10 else
            "London" if 10 <= h < 17 else
            "US"
        )
        vol10 = self.df["volume"].iloc[-10:].mean()
        self.flags["low_liquidity"]   = vol10 < (self.df["volume"].mean() * 0.6)
        v10   = self.df["close"].rolling(10).std().iloc[-1]
        self.flags["high_volatility"] = v10 > (self.df["close"].std() * 1.5)
        self.flags["event_window"]    = 12 <= ts.hour < 14

    def get(self, key):
        return self.flags.get(key)

    def all(self):
        return self.flags

