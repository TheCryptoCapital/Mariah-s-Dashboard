from agents.base_agent import Agent
import numpy as np

class FundingRateAgent(Agent):
    """
    Signals based on perpetual swap funding rate; deweights in low-liquidity.
    """
    def __init__(self):
        super().__init__("FundingRate")

    def process_data(self, data, context=None):
        self.context = context
        # Placeholder for real funding rate fetch
        self.rate = np.random.uniform(-0.01, 0.01)

    def generate_signal(self) -> str:
        self.confidence = abs(self.rate * 100)
        if self.context and self.context.get("low_liquidity"):
            self.confidence *= 0.7
        if self.rate > 0.005:
            return "sell"
        if self.rate < -0.005:
            return "buy"
        return "hold"

