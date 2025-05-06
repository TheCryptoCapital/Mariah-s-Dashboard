from agents.base_agent import Agent
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class RSIRegimeAgent(Agent):
    """
    Clusters RSI into oversold/neutral/overbought regimes,
    holds during news events.
    """
    def __init__(self):
        super().__init__("RSIRegime")

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        return 100 - (100 / (1 + rs))

    def process_data(self, data, context=None):
        self.context = context
        data["rsi"] = self.calculate_rsi(data["close"])
        vals = data["rsi"].dropna().values[-20:].reshape(-1,1)
        self.current = data["rsi"].iloc[-1]
        if len(vals) >= 3:
            self.model = KMeans(n_clusters=3, n_init=10).fit(vals)
            self.cluster = self.model.predict([[self.current]])[0]
            self.centers = self.model.cluster_centers_.flatten()
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        # avoid trades during event windows
        if self.context and self.context.get("event_window"):
            self.confidence = 0.1
            return "hold"
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"
        order = np.argsort(self.centers)
        # confidence based on distance from neutral RSI (50)
        self.confidence = 1 - abs(self.current - 50) / 50
        if self.cluster == order[0]:
            return "buy"
        if self.cluster == order[2]:
            return "sell"
        return "hold"

