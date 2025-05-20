from agents.base_agent import Agent
import numpy as np

class CorrelationMonitorAgent(Agent):
    """
    Monitors correlation with BTC reference, de-emphasizes during events.
    """
    def __init__(self):
        super().__init__("CorrelationMonitor")
        self.reference = None

    def update_reference(self, ref_series):
        self.reference = ref_series

    def process_data(self, data, context=None):
        self.context = context
        if self.reference is None or len(data) != len(self.reference):
            self.corr = None
        else:
            self.corr = np.corrcoef(data["close"], self.reference)[0, 1]

    def generate_signal(self) -> str:
        if self.corr is None:
            self.confidence = 0.1
            return "hold"
        self.confidence = abs(self.corr)
        # lower weight during news events
        if self.context and self.context.get("event_window"):
            self.confidence *= 0.8
        if self.corr > 0.75:
            return "buy"
        if self.corr < -0.5:
            return "sell"
        return "hold"

