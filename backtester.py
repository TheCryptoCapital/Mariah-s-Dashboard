import pandas as pd
from config.bybit_fetcher import get_bybit_ohlcv
from meta_controller import MetaAgentController
from market_context import MarketContext
from rl_wrapper import SimpleRLWrapper

def simulate_pnl(action, now, nxt):
    if action == "buy":  return nxt - now
    if action == "sell": return now - nxt
    return 0

def backtest(symbol="BTCUSDT", interval="5", limit=1000, window=50):
    # 1) Fetch historical OHLCV
    df = get_bybit_ohlcv(symbol, interval, limit)

    # 2) Initialize controller and RL agent
    controller = MetaAgentController()
    rl = SimpleRLWrapper()

    records = []
    # 3) Slide over data with a rolling window
    for i in range(window, len(df) - 1):
        slice_df = df.iloc[i-window:i].reset_index(drop=True)
        ctx = MarketContext(slice_df)
        result = controller.run_all(slice_df)

        # Build RL state from agent signals + context
        agent_signals = {ag.name: result[ag.name] for ag in controller.agents}
        state = rl.get_state(agent_signals, ctx.all())
        action = rl.select_action(state)

        # Simulate PnL using the next bar’s close
        now_price = slice_df["close"].iloc[-1]
        next_price = df["close"].iloc[i+1]
        reward = simulate_pnl(action, now_price, next_price)
        rl.update_q(state, action, reward)

        # Log every agent’s output plus RL action & reward
        for ag in controller.agents:
            records.append({
                "timestamp": slice_df["timestamp"].iloc[-1],
                "agent": ag.name,
                "signal": result[ag.name],
                "confidence": round(ag.confidence, 4),
                "win_rate": round(ag.get_win_rate(), 4),
                "rl_action": action,
                "reward": reward,
                **ctx.all()
            })

    # 4) Save the learned Q-table and signal log
    rl.export_q_table()
    pd.DataFrame(records).to_csv("logs/signal_log_rl.csv", index=False)
    print("✅ Backtest complete — logs/signal_log_rl.csv and logs/q_table.csv saved")

if __name__ == "__main__":
    backtest()

