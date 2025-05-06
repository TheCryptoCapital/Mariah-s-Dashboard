
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

# Connect to Bybit (for trading)
session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

# Connect to exchange for market data
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

# Config
symbols = ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'SOL/USDT', 'XRP/USDT']
bot_symbols = {
    'BTC/USDT': 'BTCUSDT',
    'ETH/USDT': 'ETHUSDT',
    'DOGE/USDT': 'DOGEUSDT',
    'SOL/USDT': 'SOLUSDT',
    'XRP/USDT': 'XRPUSDT'
}

CSV_FILE = "trades.csv"
tp_percent = 0.02
sl_percent = 0.01
qtys = {
    'BTCUSDT': 0.001,
    'ETHUSDT': 0.01,
    'DOGEUSDT': 100,
    'SOLUSDT': 0.5,
    'XRPUSDT': 10
}

def get_klines(symbol):
    bars = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df

def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    return df

def place_trade(symbol_key, side, entry_price):
    opposite = "Sell" if side == "Buy" else "Buy"
    qty = qtys[symbol_key]

    session.place_order(
        category="linear",
        symbol=symbol_key,
        side=side,
        orderType="Market",
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=False,
        closeOnTrigger=False
    )

    tp_price = round(entry_price * (1 + tp_percent), 6)
    sl_price = round(entry_price * (1 - sl_percent), 6)

    session.place_order(
        category="linear",
        symbol=symbol_key,
        side=opposite,
        orderType="Limit",
        price=tp_price,
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=True
    )

    session.place_order(
        category="linear",
        symbol=symbol_key,
        side=opposite,
        orderType="Market",
        stopLoss=sl_price,
        qty=round(qty, 3),
        timeInForce="GoodTillCancel",
        reduceOnly=True
    )

    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol_key,
            side,
            qty,
            entry_price,
            tp_price,
            sl_price
        ])
    print(f"✅ {side} trade placed for {symbol_key} at {entry_price} | TP: {tp_price} | SL: {sl_price}")

print("🚀 Multi-Symbol Scanner Bot Started")

while True:
    for symbol in symbols:
        try:
            df = get_klines(symbol)
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            rsi = latest['RSI']
            macd = latest['MACD']
            macd_signal = latest['MACD_signal']
            symbol_key = bot_symbols[symbol]
            ticker = exchange.fetch_ticker(symbol)
            last_price = ticker['last']

            print(f"{symbol} | RSI: {rsi:.2f} | MACD: {macd:.4f} | Signal: {macd_signal:.4f}")

            if rsi < 30 and prev['MACD'] < prev['MACD_signal'] and macd > macd_signal:
                print(f"📈 BUY Signal for {symbol}")
                place_trade(symbol_key, "Buy", last_price)

            elif rsi > 70 and prev['MACD'] > prev['MACD_signal'] and macd < macd_signal:
                print(f"📉 SELL Signal for {symbol}")
                place_trade(symbol_key, "Sell", last_price)
            else:
                print(f"🔍 No trade for {symbol}")

        except Exception as e:
            print(f"❌ Error scanning {symbol}: {e}")

    print("⏳ Sleeping 5 minutes before next scan...")
    time.sleep(300)
