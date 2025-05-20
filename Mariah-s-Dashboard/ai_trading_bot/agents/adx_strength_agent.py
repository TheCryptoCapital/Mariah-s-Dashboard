import warnings
# Silence the specific sklearn “valid feature names” UserWarning
warnings.filterwarnings(
    "ignore",
    message=".*does not have valid feature names.*",
    category=UserWarning
)

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from agents.base_agent import Agent

class ADXStrengthAgent(Agent):
    """
    Clusters ADX into trend-strength regimes and signals accordingly.
    """
    def __init__(self):
        super().__init__("ADXStrength")

    def process_data(self, data, context=None):
        self.context = context
        high = data["high"]
        low  = data["low"]
        close = data["close"]

        plus_dm  = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)

        # True range components as pandas Series
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di  = 100 * plus_dm.rolling(14).mean()  / (atr + 1e-6)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-6)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-6)
        data["adx"] = dx.rolling(14).mean()

        vals = data["adx"].dropna().values[-20:].reshape(-1,1)
        self.current = data["adx"].iloc[-1]
        if len(vals) >= 3:
            self.model   = KMeans(n_clusters=3, n_init=10).fit(vals)
            self.centers = self.model.cluster_centers_.flatten()
            # Keep the last fitted cluster for generate_signal()
            self.cluster = self.model.predict(vals[-1:].reshape(1, -1))[0]
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"

        # Predict again on the current value as an array
        X = np.array([[self.current]])
        self.cluster = self.model.predict(X)[0]

        # Order clusters by center magnitude
        order = np.argsort(self.centers)
        self.confidence = min(self.current / 50, 1.0)

        if self.cluster == order[2]:
            return "buy"
        if self.cluster == order[0]:
            return "avoid"
        return "hold"

