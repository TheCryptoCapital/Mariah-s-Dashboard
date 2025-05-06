from agents.base_agent import Agent
import numpy as np

class MACDDivergenceAgent(Agent):
    """
    Detects MACD/price divergences, suppressed during high-volatility.
    """
    def __init__(self):
        super().__init__("MACDDivergence")

    def process_data(self, data, context=None):
        self.context = context
        ema12 = data["close"].ewm(span=12, adjust=False).mean()
        ema26 = data["close"].ewm(span=26, adjust=False).mean()
        data["macd"] = ema12 - ema26
        self.data = data.dropna()

    def generate_signal(self) -> str:
        if self.context and self.context.get("high_volatility"):
            self.confidence = 0.1
            return "hold"
        if len(self.data) < 5:
            self.confidence = 0.1
            return "hold"
        p = self.data["close"].values[-5:]
        m = self.data["macd"].values[-5:]
        # bullish divergence
        if p[-1] < p[-3] and m[-1] > m[-3]:
            self.confidence = abs(m[-1] - m[-3]) / (np.std(m) + 1e-6)
            return "buy"
        # bearish divergence
        if p[-1] > p[-3] and m[-1] < m[-3]:
            self.confidence = abs(m[-1] - m[-3]) / (np.std(m) + 1e-6)
            return "sell"
        self.confidence = 0.1
        return "hold"

