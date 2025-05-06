from agents.base_agent import Agent
from sklearn.cluster import KMeans
import numpy as np

class ADXStrengthAgent(Agent):
    """
    Clusters ADX into trend-strength regimes and signals accordingly.
    """
    def __init__(self):
        super().__init__("ADXStrength")

    def process_data(self, data, context=None):
        self.context = context
        high, low, close = data["high"], data["low"], data["close"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        tr = np.maximum.reduce([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ])
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr + 1e-6)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr + 1e-6)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-6)
        data["adx"] = dx.rolling(14).mean()
        vals = data["adx"].dropna().values[-20:].reshape(-1,1)
        self.current = data["adx"].iloc[-1]
        if len(vals) >= 3:
            self.model = KMeans(n_clusters=3, n_init=10).fit(vals)
            self.cluster = self.model.predict([[self.current]])[0]
            self.centers = self.model.cluster_centers_.flatten()
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"
        order = np.argsort(self.centers)
        self.confidence = self.current / 50
        if self.cluster == order[2]:
            return "buy"
        if self.cluster == order[0]:
            return "avoid"
        return "hold"

