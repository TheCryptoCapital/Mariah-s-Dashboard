from agents.base_agent import Agent
import pandas as pd

class TrendFinderAgent(Agent):
    """
    Detects trend vs. chop using SMA crossover plus slope confirmation,
    and suppresses signals in low-liquidity environments.
    """
    def __init__(self):
        super().__init__("TrendFinder")

    def process_data(self, data: pd.DataFrame, context=None):
        self.context = context
        data["sma_fast"] = data["close"].rolling(window=5).mean()
        data["sma_slow"] = data["close"].rolling(window=20).mean()
        data["slope_fast"] = data["sma_fast"].diff()
        self.data = data.dropna()

    def generate_signal(self) -> str:
        df = self.data
        if self.context and self.context.get("low_liquidity"):
            self.confidence = 0.1
            return "hold"

        recent_fast = df["sma_fast"].iloc[-1]
        recent_slow = df["sma_slow"].iloc[-1]
        slope       = df["slope_fast"].iloc[-1]

        if recent_fast > recent_slow and slope > 0:
            self.confidence = min(abs(slope) / recent_slow, 1.0)
            return "buy"
        if recent_fast < recent_slow and slope < 0:
            self.confidence = min(abs(slope) / recent_slow, 1.0)
            return "sell"

        self.confidence = 0.1
        return "hold"

