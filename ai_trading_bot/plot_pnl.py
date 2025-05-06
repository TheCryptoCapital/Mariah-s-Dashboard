import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load & sort by timestamp
sig = pd.read_csv('logs/signal_log_rl.csv', parse_dates=['timestamp'])
sig = sig.sort_values('timestamp')

# 1) Cumulative PnL series
pnl = sig.groupby('timestamp')['reward'].sum().cumsum()

# 2) Total PnL
total_pnl = pnl.iloc[-1]

# 3) Daily returns
daily = pnl.resample('1D').last().diff().dropna()

# 4) Sharpe ratio (0% risk‐free)
sharpe = daily.mean() / daily.std() * np.sqrt(252)

# 5) Max drawdown
running_max = pnl.cummax()
drawdown   = pnl - running_max
max_dd     = drawdown.min()

print(f"Total PnL:    {total_pnl:,.0f}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:,.0f}")

# 6) Plot the equity curve
plt.figure(figsize=(10, 6))
pnl.plot(title="Cumulative PnL Over Time")
plt.ylabel("PnL")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

