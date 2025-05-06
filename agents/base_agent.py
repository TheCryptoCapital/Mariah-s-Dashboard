class Agent:
    """
    Base class for all trading agents.
    Tracks signal history, result history, win rate, and confidence.
    """
    def __init__(self, name):
        self.name = name
        self.last_signal = "hold"
        self.confidence = 0.1
        self.signal_history = []
        self.result_history = []

    def process_data(self, data, context=None):
        raise NotImplementedError

    def generate_signal(self):
        raise NotImplementedError

    def update_memory(self, signal, result=None):
        self.signal_history.append(signal)
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
        if result is not None:
            self.result_history.append(result)
            if len(self.result_history) > 100:
                self.result_history.pop(0)

    def get_win_rate(self):
        if not self.result_history:
            return 0.5
        wins = sum(1 for r in self.result_history if r > 0)
        return wins / len(self.result_history)


