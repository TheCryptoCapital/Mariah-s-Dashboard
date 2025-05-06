import pandas as pd
from datetime import datetime

# Load the bot's trade log
trades_file = "trades.csv"
daily_pnl_file = "daily_pnl.csv"

# Read trades.csv
try:
    df = pd.read_csv(trades_file)

    if df.empty:
        print("No trades found in trades.csv")
        exit()

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate Realized PnL for each trade
    df['Realized_PnL'] = (df['take_profit'] - df['entry_price']) * df['qty']

    # Group by date
    df['date'] = df['timestamp'].dt.date
    daily_pnl = df.groupby('date')['Realized_PnL'].sum().reset_index()

    # Save daily PnL to CSV
    daily_pnl.to_csv(daily_pnl_file, index=False)

    # Print results nicely
    print("\nDaily PnL Report:")
    for idx, row in daily_pnl.iterrows():
        print(f"{row['date']}: ${row['Realized_PnL']:.2f}")

    print(f"\nDaily PnL saved to {daily_pnl_file}")

except FileNotFoundError:
    print(f"Error: {trades_file} not found. Make sure trades.csv exists!")

