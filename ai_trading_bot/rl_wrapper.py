import random
import pandas as pd
from collections import defaultdict

class SimpleRLWrapper:
    """
    ε-greedy Q-learning over the meta-controller’s state (agent signals + context).
    """
    def __init__(self,
                 actions=("buy","sell","hold","avoid"),
                 lr=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.lr       = lr
        self.gamma    = gamma
        self.epsilon  = epsilon
        self.q_table  = defaultdict(lambda: {a:0.0 for a in actions})

    def get_state(self, agent_signals: dict, context: dict) -> str:
        # Create a hashable state string
        return str(tuple(sorted(agent_signals.items())) + tuple(sorted(context.items())))

    def select_action(self, state: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q(self, state: str, action: str, reward: float):
        max_future = max(self.q_table[state].values())
        current_q = self.q_table[state][action]
        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * max_future - current_q)
        self.q_table[state][action] = new_q

    def export_q_table(self, path: str = "logs/q_table.csv"):
        rows = []
        for st, actions in self.q_table.items():
            for act, val in actions.items():
                rows.append({"state": st, "action": act, "value": val})
        pd.DataFrame(rows).to_csv(path, index=False)

