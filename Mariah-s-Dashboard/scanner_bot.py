import os
import time
import ccxt
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
import csv
from datetime import datetime

# Load API keys
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Connect to Bybit
session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

# Connect to Bybit for market data via ccxt
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

symbol = 'BTC/USDT'  # Trading pair
bot_symbol = 'BTCUSDT'  # API format

CSV_FILE = "trades.csv"

# Trade configuration
tp_percent = 0.02  # +2% take profit
sl_percent = 0.01  # -1% stop loss
qty = 0.001        # Fixed quantity per trade

# Functions

def get_klines():
    bars = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    return df

def place_trade(direction, entry_price):
    opposite = "Sell" if direction == "Buy" else "Buy"

    # Place market order
    session.place_order(
        category="linear",
        symbol=bot_symbol,
        side=direction,
        orderType="Market",
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=False,
        closeOnTrigger=False
    )

    # Set TP and SL
    tp_price = round(entry_price * (1 + tp_percent), 2)
    sl_price = round(entry_price * (1 - sl_percent), 2)

    session.place_order(
        category="linear",
        symbol=bot_symbol,
        side=opposite,
        orderType="Limit",
        price=tp_price,
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=True
    )

    session.place_order(
        category="linear",
        symbol=bot_symbol,
        side=opposite,
        orderType="Market",
        stopLoss=sl_price,
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=True
    )

    # Log to CSV
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            bot_symbol,
            direction,
            qty,
            entry_price,
            tp_price,
            sl_price
        ])

    print(f"✅ {direction} trade placed at {entry_price} | TP: {tp_price} | SL: {sl_price}")

# Live scanner
print("🚀 Live Scanner Bot Started!")

while True:
    try:
        df = get_klines()
        df = calculate_indicators(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        rsi = latest['RSI']
        macd = latest['MACD']
        macd_signal = latest['MACD_signal']

        print(f"RSI: {rsi:.2f} | MACD: {macd:.4f} | Signal: {macd_signal:.4f}")

        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker['last']

        # Buy signal
        if rsi < 30 and prev['MACD'] < prev['MACD_signal'] and macd > macd_signal:
            print("📈 BUY Signal detected!")
            place_trade("Buy", last_price)

        # Sell signal
        elif rsi > 70 and prev['MACD'] > prev['MACD_signal'] and macd < macd_signal:
            print("📉 SELL Signal detected!")
            place_trade("Sell", last_price)

        else:
            print("🔍 No signal, waiting...")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Wait 5 minutes
    time.sleep(300)

