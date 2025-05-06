from agents.base_agent import Agent
import numpy as np

class NewsSentimentAgent(Agent):
    """
    Processes NLP sentiment; boosts confidence around news windows.
    """
    def __init__(self):
        super().__init__("NewsSentiment")

    def process_data(self, data, context=None):
        self.context = context
        # Placeholder for real NLP sentiment score
        self.sentiment = np.random.uniform(-1, 1)

    def generate_signal(self) -> str:
        self.confidence = abs(self.sentiment)
        if self.context and self.context.get("event_window"):
            self.confidence *= 1.3
        if self.sentiment > 0.5:
            return "buy"
        if self.sentiment < -0.5:
            return "sell"
        return "hold"

