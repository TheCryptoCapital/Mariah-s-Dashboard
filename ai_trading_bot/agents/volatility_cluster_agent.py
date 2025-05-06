from agents.base_agent import Agent
from sklearn.cluster import KMeans
import numpy as np

class VolatilityClusterAgent(Agent):
    """
    Clusters recent volatility into low/mid/high regimes,
    signals buy in high-vol, avoid in low-vol.
    """
    def __init__(self):
        super().__init__("VolatilityCluster")

    def process_data(self, data, context=None):
        self.context = context
        data["volatility"] = data["close"].rolling(window=10).std()
        vol = data["volatility"].dropna().values[-20:].reshape(-1,1)
        self.current = data["volatility"].iloc[-1]
        if len(vol) >= 3:
            self.model = KMeans(n_clusters=3, n_init=10).fit(vol)
            self.cluster = self.model.predict([[self.current]])[0]
            self.centers = self.model.cluster_centers_.flatten()
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"
        order = np.argsort(self.centers)
        self.confidence = float(self.current / (np.std(self.centers) + 1e-6))
        if self.cluster == order[2]:
            return "buy"
        if self.cluster == order[0]:
            return "avoid"
        return "hold"

