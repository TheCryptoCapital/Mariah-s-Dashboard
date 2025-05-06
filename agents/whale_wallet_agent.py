from agents.base_agent import Agent
import numpy as np

class WhaleWalletAgent(Agent):
    """
    Tracks on-chain whale flows; boosts confidence in high-vol regimes.
    """
    def __init__(self):
        super().__init__("WhaleWallet")

    def process_data(self, data, context=None):
        self.context = context
        # Placeholder for real on-chain data
        self.activity = np.random.choice(["inflow", "outflow", "neutral"])
        base = np.random.uniform(0.6, 1.0) if self.activity != "neutral" else 0.1
        if self.context and self.context.get("high_volatility"):
            base *= 1.2
        self.confidence = base

    def generate_signal(self) -> str:
        if self.activity == "inflow":
            return "buy"
        if self.activity == "outflow":
            return "sell"
        return "hold"

