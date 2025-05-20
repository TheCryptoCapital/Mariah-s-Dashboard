from agents.base_agent import Agent
import numpy as np

class VolumeSpikeAgent(Agent):
    """
    Detects unusually large volume spikes via rolling z-score,
    boosts confidence in high-volatility contexts.
    """
    def __init__(self):
        super().__init__("VolumeSpike")

    def process_data(self, data, context=None):
        self.context = context
        vol = data["volume"]
        mu = vol.rolling(window=20).mean()
        sigma = vol.rolling(window=20).std() + 1e-6
        data["zscore"] = (vol - mu) / sigma
        self.zscore = data["zscore"].iloc[-1]

    def generate_signal(self) -> str:
        if self.zscore > 2:
            self.confidence = min(self.zscore / 3, 1.0)
            if self.context and self.context.get("high_volatility"):
                self.confidence = min(self.confidence * 1.2, 1.0)
            return "buy"
        self.confidence = 0.1
        return "hold"

