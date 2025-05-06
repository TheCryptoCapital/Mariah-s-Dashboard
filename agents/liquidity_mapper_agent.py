from agents.base_agent import Agent
from sklearn.cluster import KMeans
import numpy as np

class LiquidityMapperAgent(Agent):
    """
    Clusters order-book liquidity (spread/volume) and signals on best regime,
    holds on weekends.
    """
    def __init__(self):
        super().__init__("LiquidityMapper")

    def process_data(self, data, context=None):
        self.context = context
        data["spread"] = data["high"] - data["low"]
        data["spread_vol"] = data["spread"] / (data["volume"] + 1e-6)
        feats = data[["volume","spread_vol"]].dropna().tail(20)
        if len(feats) >= 3:
            self.model = KMeans(n_clusters=3, n_init=10).fit(feats)
            self.cluster = self.model.predict([feats.iloc[-1].values])[0]
            self.centers = self.model.cluster_centers_
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        if self.context and self.context.get("is_weekend"):
            self.confidence = 0.1
            return "hold"
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"
        scores = self.centers[:,0] / (self.centers[:,1] + 1e-6)
        best, worst = np.argmax(scores), np.argmin(scores)
        self.confidence = scores[self.cluster] / (scores[best] + 1e-6)
        if self.cluster == best:
            return "buy"
        if self.cluster == worst:
            return "avoid"
        return "hold"

