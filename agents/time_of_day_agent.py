from agents.base_agent import Agent

class TimeOfDayAgent(Agent):
    """
    Biases trading by session: Asia=buy, London=hold, US=sell.
    """
    def __init__(self):
        super().__init__("TimeOfDay")

    def process_data(self, data, context=None):
        self.context = context
        hour = data["timestamp"].iloc[-1].hour
        if 2 <= hour < 10:
            self.bias = "buy";  self.confidence = 0.7
        elif 10 <= hour < 17:
            self.bias = "hold"; self.confidence = 0.5
        else:
            self.bias = "sell"; self.confidence = 0.7

    def generate_signal(self) -> str:
        return self.bias

