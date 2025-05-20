from agents.base_agent import Agent
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

class PnLClusterRebalancerAgent(Agent):
    """
    Clusters past PnL to boost or suppress current strategy confidence
    by session.
    """
    def __init__(self):
        super().__init__("PnLClusterRebalancer")
        self.trade_log = pd.DataFrame(columns=["pnl"])

    def update_trade_log(self, pnl_list):
        self.trade_log = pd.DataFrame(pnl_list, columns=["pnl"])

    def process_data(self, data, context=None):
        self.context = context
        pnl = self.trade_log["pnl"].dropna().values[-20:].reshape(-1, 1)
        self.current = pnl[-1][0] if len(pnl) else 0
        if len(pnl) >= 3:
            self.model = KMeans(n_clusters=3, n_init=10).fit(pnl)
            self.cluster = self.model.predict([[self.current]])[0]
            self.centers = self.model.cluster_centers_.flatten()
        else:
            self.cluster = -1

    def generate_signal(self) -> str:
        if self.cluster == -1:
            self.confidence = 0.1
            return "hold"
        best, worst = np.argmax(self.centers), np.argmin(self.centers)
        # scale down in Asia session
        adj = 0.85 if self.context and self.context.get("session") == "Asia" else 1.0
        self.confidence = adj * abs(self.current / (np.std(self.centers) + 1e-6))
        if self.cluster == best:
            return "buy"
        if self.cluster == worst:
            return "avoid"
        return "hold"

