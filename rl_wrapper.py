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

    def get_state(self, agent_signals:dict, context:dict) -> str:
        return str(tuple(sorted(agent_signals.items())) + tuple(sorted(context.items())))

    def select_action(self, state:str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q(self, state:str, action:str, reward:float):
        max_f = max(self.q_table[state].values())
        old   = self.q_table[state][action]
        new   = old + self.lr * (reward + self.gamma * max_f - old)
        self.q_table[state][action] = new

    def export_q_table(self, path="logs/q_table.csv"):
        rows = []
        for st, acts in self.q_table.items():
            for a, v in acts.items():
                rows.append({"state": st, "action": a, "value": v})
        pd.DataFrame(rows).to_csv(path, index=False)

