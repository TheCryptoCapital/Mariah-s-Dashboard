# =======================
# 📦 IMPORTS & SETUP
# =======================

import asyncio
import sys

# macOS fix for "RuntimeError: Event loop is closed"
if sys.platform.startswith('darwin'):
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except RuntimeError:
        pass

import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np  # ✅ Use np.nan for missing values
import plotly.express as px  # ✅ Needed for interactive charts
import os
import time
import base64
import csv
import smtplib
import openai
import speech_recognition as sr
from email.message import EmailMessage
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from datetime import datetime, date
from streamlit_autorefresh import st_autorefresh
from PIL import Image
from strategy_mode import get_strategy_params
from mariah_voice import mariah_speak


# ✅ Load environment variables from .env explicitly
load_dotenv(dotenv_path="/root/mariah/.env")


# ✅ Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Etherscan imports
from onchain_feed import get_eth_gas, get_block_info

# ✅ Risk Threshold for the Day
MAX_DAILY_LOSS = -300  # 🔥 Adjust this as needed

# ✅ Risk Log Path
RISK_LOG_PATH = "risk_events.csv"

# ✅ Function to enforce daily loss limit
def check_max_daily_loss(df_bot_closed, df_manual_closed):
    if st.session_state.get("override_risk_lock"):
        return False

    today = date.today()
    df_all = pd.concat([df_bot_closed, df_manual_closed])
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_today = df_all[df_all["timestamp"].dt.date == today]
    pnl_today = df_today["Realized PnL ($)"].sum()

    if pnl_today <= MAX_DAILY_LOSS:
        log_risk_lock_event(today, pnl_today)
        return True

    return False

# ✅ Log daily shutdown to risk_events.csv
def log_risk_lock_event(today, pnl_today):
    if os.path.exists(RISK_LOG_PATH):
        df_log = pd.read_csv(RISK_LOG_PATH)
        if today in pd.to_datetime(df_log["date"]).dt.date.values:
            return  # Already logged today
    else:
        df_log = pd.DataFrame(columns=["date", "triggered_at_pnl"])

    new_row = pd.DataFrame([{"date": today, "triggered_at_pnl": pnl_today}])
    df_log = pd.concat([df_log, new_row], ignore_index=True)
    df_log.to_csv(RISK_LOG_PATH, index=False)

# ✅ Risk Banner Display
@st.cache_data(ttl=30)
def should_show_risk_banner(df_bot_closed, df_manual_closed):
    return check_max_daily_loss(df_bot_closed, df_manual_closed)


# ✅ Base64 image encoder (used for logo and brain images)
def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

from strategy_mode import get_strategy_params  # Required to get mode-specific settings

# ✅ Mode-Aware RSI Signal Evaluation
def get_rsi_signal(rsi_value, mode="Swing"):
    _, _, rsi_ob, rsi_os = get_strategy_params(mode)
    if rsi_value < rsi_os:
        return "buy"
    elif rsi_value > rsi_ob:
        return "sell"
    else:
        return "hold"

# ✅ Mode-Aware RSI Signal Scanner
def check_rsi_signal(symbol="BTCUSDT", interval="15", mode="Swing"):
    try:
        res = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=100
        )["result"]["list"]

        if not res or len(res) == 0:
            st.warning(f"📭 No kline data returned for {symbol}")
            return None, False

        # Create DataFrame
        df = pd.DataFrame(res, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["close"] = pd.to_numeric(df["close"])
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit='ms')

        # Calculate RSI
        df["RSI"] = ta.rsi(df["close"], length=14)
        latest_rsi = df["RSI"].iloc[-1]

        st.write(f"✅ {symbol} RSI (last candle):", latest_rsi)

        # Evaluate Signal
        signal = get_rsi_signal(latest_rsi, mode)
        return latest_rsi, signal == "buy"

    except Exception as e:
        st.error(f"❌ RSI scan failed for {symbol}: {e}")
        return None, False

# ✅ Log RSI Trade to CSV with Mode
def log_rsi_trade_to_csv(symbol, side, qty, entry_price, mode="Swing"):
    from datetime import datetime
    import csv
    import os

    log_file = "trades.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    trade_data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "stop_loss": 0,
        "take_profit": 0,
        "note": "RSI signal",
        "mode": mode  # ✅ Strategy mode now logged
    }

    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=trade_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_data)

# ✅ Mariah's Voice Engine
def mariah_speak(text):
    try:
        engine.setProperty('rate', 160)      # Speaking speed
        engine.setProperty('volume', 0.9)    # Volume level
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Mariah voice error: {e}")

def get_mariah_reply(prompt, open_pnl, closed_pnl, override_on):
    try:
        context = (
            f"Open PnL is ${open_pnl:,.2f}. "
            f"Closed PnL is ${closed_pnl:,.2f}. "
            f"Override is {'enabled' if override_on else 'off'}."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Mariah, an AI trading assistant. "
                        "You're smart, intuitive, and protective of Jonathan's capital. "
                        f"Live dashboard data: {context}"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Mariah GPT error: {e}"

# ✅ Mariah Voice Input (Speech-to-Text)
def listen_to_user():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        with mic as source:
            st.info("🎙 Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10)

            try:
                user_text = recognizer.recognize_google(audio)
                st.success(f"You said: {user_text}")
                return user_text
            except sr.UnknownValueError:
                st.error("❌ I couldn’t understand what you said.")
                return None
            except sr.RequestError:
                st.error("❌ Speech recognition service unavailable.")
                return None

    except sr.WaitTimeoutError:
        st.error("❌ No voice detected — try again.")
        return None
    except AssertionError as e:
        st.error(f"🎤 Microphone error: {e}")
        return None

# ✅ Background + Style
def set_dashboard_background(image_file):
    ...


# ✅ Cinematic dashboard background with glow
def set_dashboard_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(f"""
        <style>
        html, body, .stApp {{
            background:
                radial-gradient(circle at 35% 40%, rgba(80, 0, 150, 0.2) 0%, rgba(10, 0, 30, 0.85) 60%),
                linear-gradient(rgba(20, 10, 40, 0.4), rgba(20, 10, 40, 0.4)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-blend-mode: soft-light, overlay, normal;
        }}

        .blur-card, section.main .blur-card {{
            background-color: rgba(30, 0, 60, 0.4);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.5rem;
            margin-top: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.08);
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(30, 20, 60, 0.4) !important;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 16px;
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.05);
            padding: 1rem;
            margin: 1.5rem 0 1.5rem 0.5rem;
            transition: all 0.3s ease-in-out;
            width: 360px !important;
            min-width: 360px !important;
        }}

        [data-testid="stSidebar"]:hover {{
            box-shadow: 0 0 35px rgba(0, 255, 255, 0.2);
            border: 1px solid rgba(0, 255, 255, 0.3);
        }}

        section[data-testid="stSidebar"] > div:first-child {{
            border-top: none;
            box-shadow: none;
            padding-top: 0;
            margin-top: -10px;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background-color: #008d87;
            z-index: 9999;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(20, 16, 50, 0.85);
            border-bottom: 2px solid #008d87;
            box-shadow: 0 2px 10px #008d87;
            transition: all 0.3s ease;
        }}

        header[data-testid="stHeader"]:hover {{
            border-bottom: 2px solid #00fff5;
            box-shadow: 0 4px 20px #00fff5;
        }}

        div[data-testid="stAlert"] {{
            background-color: rgba(42, 30, 85, 0.8) !important;
            border: 1px solid #008d87;
            color: #e0f7ff !important;
            border-radius: 10px;
        }}

        h1, h2, h3, h4, h5, p, label {{
            color: #ffffff !important;
        }}

        .glow-on-hover:hover {{
            filter: drop-shadow(0 0 10px #00fff5);
        }}

        .pulse-brain {{
            animation: pulse 1.8s infinite ease-in-out;
            transform-origin: center;
        }}

        @keyframes pulse {{
            0%   {{ transform: scale(1); opacity: 1; }}
            50%  {{ transform: scale(1.1); opacity: 0.8; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}

        .override-glow {{
            animation: pulse-glow 1.8s infinite ease-in-out;
            transform-origin: center;
        }}

        .mariah-avatar {{
            animation: pulse-glow 2s infinite ease-in-out;
            transform-origin: center;
            border-radius: 12px;
            box-shadow: 0 0 20px #00fff5;
        }}

        @keyframes pulse-glow {{
            0%   {{ box-shadow: 0 0 0px #00fff5; }}
            50%  {{ box-shadow: 0 0 20px #00fff5; }}
            100% {{ box-shadow: 0 0 0px #00fff5; }}
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Set page layout
st.set_page_config(page_title="The Crypto Capital", layout="wide")

if "mariah_greeted" not in st.session_state:
    mariah_speak("System online. Welcome to the Crypto Capital.")
    st.session_state["mariah_greeted"] = True

# ✅ Load images FIRST (before using them)
logo_base64 = get_base64_image("logo.png")
brain_base64 = get_base64_image("updatedbrain.png")

# ✅ Apply background image
set_dashboard_background("Screenshot 2025.png")

# Load Mariah avatar (must be in same folder as dashboard.py)
mariah_img = Image.open("ChatGPT Image May 4, 2025, 07_29_01 PM.png")

header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.markdown(f"""
    <div class="blur-card" style="display: flex; align-items: center; gap: 20px; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{logo_base64}" width="180" class="glow-on-hover" />
        <div style="font-size: 4rem; font-weight: 800; color: white;">
            The <span style="color: #00fff5;">Crypto</span> Capital
        </div>
    </div>
    """, unsafe_allow_html=True)

import base64

# Convert Mariah's image to base64
with open("ChatGPT Image May 4, 2025, 07_29_01 PM.png", "rb") as img_file:
    mariah_base64 = base64.b64encode(img_file.read()).decode()

with header_col2:
    st.markdown(f"""
    <img src="data:image/png;base64,{mariah_base64}"
         class="mariah-avatar"
         width="130"
         style="margin-top: 0.5rem; border-radius: 12px;" />
    """, unsafe_allow_html=True)

with st.sidebar:
    # 🧠 Powered by AI Banner (Optional, Non-frosted)
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin: 1rem 0 0.5rem 0.5rem;">
        <span style="color: #00fff5; font-size: 1.1rem; font-weight: 600;">Powered by AI</span>
        <img src="data:image/png;base64,{brain_base64}" width="26" class="pulse-brain" />
    </div>
    """, unsafe_allow_html=True)

    # 📂 More Tools (Frosted Panel)
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.markdown("### 📂 More Tools")
    more_tab = st.selectbox(
        "Select a Tool",
        [
            "📆 Daily PnL",
            "📈 Performance Trends",
            "📆 Filter by Date",
            "📰 Crypto News",
            "📡 On-Chain Data",
            "📡 Signal Scanner"
        ],
        key="sidebar_more_tools_dropdown"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 📊 Dashboard Controls Panel (Frosted Panel)
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.title("📊 Dashboard Controls")

    # 🔇 Voice Settings
    st.markdown("### 🎤 Voice Settings")
    st.session_state["mute_mariah"] = st.sidebar.checkbox(
        "🔇 Mute Mariah’s Voice",
         value=st.session_state.get("mute_mariah", False)
    )

    # 🔊 Trigger Mariah's voice once per session
    if "mariah_greeted" not in st.session_state and not st.session_state["mute_mariah"]:
        mariah_speak("System online. Welcome to the Crypto Capital.")
        st.session_state["mariah_greeted"] = True

        st.audio("assets/mariah_output.mp3")

    # 🔁 Auto-Refresh Selector
    refresh_choice = st.selectbox(
        "🔁 Auto-Refresh Interval",
        options=["Every 10 sec", "Every 30 sec", "Every 1 min", "Every 5 min"],
        index=1,
        key="refresh_interval_selector_unique"
    )

    refresh_map = {
        "Every 10 sec": 10000,
        "Every 30 sec": 30000,
        "Every 1 min": 60000,
        "Every 5 min": 300000
    }
    refresh_interval = refresh_map[refresh_choice]
    st_autorefresh(interval=refresh_interval, key="auto_refresh_unique")

    # ⚙️ Strategy Mode Selector
    st.markdown("### ⚙️ Strategy Mode")
    mode = st.radio(
       "Choose a Strategy Mode:",
       ["Scalping", "Swing", "Momentum"],
       index=1,
       key="strategy_mode_selector"
    )


    # 📏 Position Sizing Section
    st.markdown("## 📏 Position Sizing")
    account_balance = st.number_input(
        "Account Balance ($)",
        value=5000.0,
        key="position_account_balance_input"
    )

    risk_percent = st.slider(
        "Risk % Per Trade",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        key="position_risk_slider"
    )

    entry_price_sidebar = st.number_input(
        "Entry Price",
        value=0.0,
        format="%.2f",
        key="position_entry_price_input"
    )

    stop_loss_sidebar = st.number_input(
        "Stop Loss Price",
        value=0.0,
        format="%.2f",
        key="position_stop_loss_input"
    )

    if entry_price_sidebar > 0 and stop_loss_sidebar > 0 and entry_price_sidebar != stop_loss_sidebar:
        qty_calc = position_size_from_risk(account_balance, risk_percent, entry_price_sidebar, stop_loss_sidebar)
        st.success(f"📊 Suggested Quantity: {qty_calc}")
    else:
        qty_calc = 0
        st.info("Enter valid entry and stop-loss.")

    # ✅ CLOSE the frosted panel
    st.markdown('</div>', unsafe_allow_html=True)

    # ✅ Risk Override Toggle
    st.markdown("---")
    st.markdown("### 🛑 Risk Controls")
    st.session_state["override_risk_lock"] = st.checkbox(
        "🚨 Manually override Mariah's risk lock (not recommended)",
        value=st.session_state.get("override_risk_lock", False)
    )

    st.session_state["test_mode"] = st.checkbox(
        "🧪 Enable Test Mode (force banners)",
        value=st.session_state.get("test_mode", False)
    )

# ✅ Email sender function (used in Tab 10)
def send_email_with_attachment(subject, body, to_email, filename):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = to_email
    msg.set_content(body)

    with open(filename, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(filename)

    msg.add_attachment(file_data, maintype="application", subtype="pdf", filename=file_name)

    with smtplib.SMTP(os.getenv("EMAIL_HOST"), int(os.getenv("EMAIL_PORT"))) as server:
        server.starttls()
        server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASSWORD"))
        server.send_message(msg)


# 📏 Sidebar Position Sizing Helper
if entry_price_sidebar > 0 and stop_loss_sidebar > 0 and entry_price_sidebar != stop_loss_sidebar:
    qty_calc = position_size_from_risk(account_balance, risk_percent, entry_price_sidebar, stop_loss_sidebar)
    st.sidebar.success(f"📊 Suggested Quantity: {qty_calc}")
else:
    qty_calc = 0
    st.sidebar.info("Enter valid entry and stop-loss.")

# ✅ Set image as full-page dashboard background
def set_dashboard_background(image_file):
    import base64
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(f"""
        <style>
        html, body, .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        .blur-card {{
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.08);   
            margin-top: 1rem;
        }}

        .pnl-positive {{
            color: #00d87f;
            font-weight: bold;
            text-align: center;
        }}
        .pnl-negative {{
            color: #ff4d4d;
            font-weight: bold;
            text-align: center;
        }}
        .pnl-label {{
            color: white;
            text-align: center;
            margin-bottom: 0.5rem;
            font-weight: 600;
            font-size: 1rem;
        }}

        [data-testid="stSidebar"] {{
            background-color: rgba(20, 16, 50, 0.85) !important;
            backdrop-filter: blur(10px);
            border-right: 3px solid #008d87;
            box-shadow: 2px 0 12px #008d87;
            transition: box-shadow 0.3s ease, border-color 0.3s ease;
            width: 375px !important;
            min-width: 375px !important;
        }}

        [data-testid="stSidebar"]:hover {{
            box-shadow: 4px 0 20px #00fff5;
            border-right: 3px solid #00fff5;
        }}

        section[data-testid="stSidebar"] > div:first-child {{
            border-top: none;
            box-shadow: none;
            padding-top: 0;
            margin-top: -10px;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background-color: #008d87;
            z-index: 9999;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(20, 16, 50, 0.85);
            border-bottom: 2px solid #008d87;
            box-shadow: 0 2px 10px #008d87;
            transition: all 0.3s ease;
        }}

        header[data-testid="stHeader"]:hover {{
            border-bottom: 2px solid #00fff5;
            box-shadow: 0 4px 20px #00fff5;
        }}

        div[data-testid="stAlert"] {{
            background-color: rgba(42, 30, 85, 0.8) !important;
            border: 1px solid #008d87;
            color: #e0f7ff !important;
            border-radius: 10px;
        }}

        h1, h2, h3, h4, h5, p, label {{
            color: #ffffff !important;
        }}

        .glow-on-hover:hover {{
            filter: drop-shadow(0 0 10px #00fff5);
        }}

        .pulse-brain {{
            animation: pulse 1.8s infinite ease-in-out;
            transform-origin: center;
        }}

        @keyframes pulse {{
            0%   {{ transform: scale(1); opacity: 1; }}
            50%  {{ transform: scale(1.1); opacity: 0.8; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
        </style>
    """, unsafe_allow_html=True)

# Load .env
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
session = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    recv_window=30000  # ← increase timeout window (ms)
)

CSV_FILE = "trades.csv"
if os.path.exists(CSV_FILE):
    df_trades = pd.read_csv(CSV_FILE)
else:
    df_trades = pd.DataFrame()

if os.path.exists("daily_pnl.csv"):
    df_daily_pnl = pd.read_csv("daily_pnl.csv")
else:
    df_daily_pnl = pd.DataFrame(columns=["date", "Realized_PnL"])

def load_open_positions():
    try:
        res = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )

        data = res["result"]["list"]

        def parse_float(value):
            try:
                return float(value)
            except:
                return 0.0

        return pd.DataFrame([{
            "Symbol": t.get("symbol", ""),
            "Size": parse_float(t.get("positionValue", 0)),
            "Entry Price": parse_float(t.get("avgPrice", 0)),
            "Mark Price": parse_float(t.get("markPrice", 0)),
            "PnL ($)": parse_float(t.get("unrealisedPnl", 0)),
            "Leverage": t.get("leverage", "")
        } for t in data if parse_float(t.get("positionValue", 0)) > 0])

    except Exception as e:
        st.error(f"❌ Failed to load open positions: {e}")
        return pd.DataFrame()

def load_trades():
    return pd.read_csv("trades.csv") if os.path.exists("trades.csv") else pd.DataFrame()


def load_closed_manual_trades():
    try:
        res = session.get_closed_pnl(category="linear", limit=50)
        data = res["result"]["list"]
        return pd.DataFrame([{
            "timestamp": t.get("createdTime", ""),
            "Symbol": t["symbol"],
            "Side": t["side"],
            "Size": float(t["qty"]),
            "Entry Price": float(t["avgEntryPrice"]),
            "Exit Price": float(t["avgExitPrice"]),
            "Realized PnL ($)": float(t["closedPnl"]),
            "Realized PnL (%)": (
                (float(t["closedPnl"]) / (float(t["qty"]) * float(t["avgEntryPrice"]) + 1e-9)) * 100
                if float(t["qty"]) > 0 else 0
            )
        } for t in data])
    except Exception as e:
        st.error(f"Error loading closed manual trades: {e}")
        return pd.DataFrame()


def trailing_stop_loss(threshold_pct=0.01, buffer_pct=0.015, log_slot=None):
    try:
        positions = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )

        updated_any = False

        for pos in positions["result"]["list"]:
            if float(pos["size"]) == 0:
                continue  # Skip closed positions

            symbol = pos["symbol"]

            try:
                entry_price = float(pos["avgPrice"])
                mark_price = float(pos["markPrice"])
                current_sl = float(pos.get("stopLoss") or 0)
            except (ValueError, TypeError):
                if log_slot:
                    log_slot.warning(f"⚠️ Skipped {symbol} due to invalid pricing")
                continue

            gain_pct = (mark_price - entry_price) / entry_price

            if log_slot:
                log_slot.write(f"🟢 Checking {symbol} | Entry: {entry_price:.2f} | Mark: {mark_price:.2f} | Gain: {gain_pct:.2%}")

            if gain_pct >= threshold_pct:
                new_sl = round(entry_price * (1 + gain_pct - buffer_pct), 2)

                if current_sl == 0 or new_sl > current_sl:
                    session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=new_sl
                    )
                    updated_any = True
                    if log_slot:
                        log_slot.success(f"📈 Trailing SL updated for {symbol}: ${new_sl}")

        if not updated_any and log_slot:
            log_slot.info("📭 No SL updated this round.")

    except Exception as e:
        if log_slot:
            log_slot.error(f"❌ SL trailing error: {e}")
    try:
        res = session.get_closed_pnl(category="linear", limit=50)
        data = res["result"]["list"]
        return pd.DataFrame([{
            "timestamp": t.get("createdTime", ""),
            "Symbol": t["symbol"],
            "Side": t["side"],
            "Size": float(t["qty"]),
            "Entry Price": float(t["avgEntryPrice"]),
            "Exit Price": float(t["avgExitPrice"]),
            "Realized PnL ($)": float(t["closedPnl"]),
            "Realized PnL (%)": (
                (float(t["closedPnl"]) / (float(t["qty"]) * float(t["avgEntryPrice"]) + 1e-9)) * 100
                if float(t["qty"]) > 0 else 0
            )
        } for t in data])
    except Exception as e:
        st.error(f"Error loading manual closed trades: {e}")
        return pd.DataFrame()

FEE_RATE = 0.00075  # 0.075% typical Bybit taker fee

@st.cache_data(ttl=15)
def split_bot_trades(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_open = df[df['take_profit'] == 0].copy()
    df_closed = df[df['take_profit'] != 0].copy()

    # Add 'note' column if it's missing
    if "note" not in df_open.columns:
        df_open["note"] = ""
    if "note" not in df_closed.columns:
        df_closed["note"] = ""

    # Fee-adjusted PnL ($)
    df_closed["Realized PnL ($)"] = (
        (df_closed["take_profit"] - df_closed["entry_price"]) * df_closed["qty"]
        - (df_closed["entry_price"] + df_closed["take_profit"]) * df_closed["qty"] * FEE_RATE
    )

    # PnL (%) remains unchanged
    df_closed["Realized PnL (%)"] = (
        ((df_closed["take_profit"] - df_closed["entry_price"]) / df_closed["entry_price"]) * 100
    )

    return df_open, df_closed


def log_daily_pnl_split(df_bot_closed, df_manual_closed, file_path="daily_pnl_split.csv"):
    today = pd.Timestamp.now().date()

    def calc_stats(df):
        if "timestamp" not in df.columns or df.empty:
            return 0.0, 0, 0, 0
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df_today = df[df["timestamp"].dt.date == today]
        if df_today.empty:
            return 0.0, 0, 0, 0
        pnl_sum = df_today["Realized PnL ($)"].sum()
        count = len(df_today)
        wins = len(df_today[df_today["Realized PnL ($)"] > 0])
        losses = len(df_today[df_today["Realized PnL ($)"] < 0])
        return pnl_sum, count, wins, losses

    # Calculate stats
    bot_pnl, bot_count, bot_wins, bot_losses = calc_stats(df_bot_closed)
    manual_pnl, manual_count, manual_wins, manual_losses = calc_stats(df_manual_closed)
    total_pnl = bot_pnl + manual_pnl

    if bot_count == 0 and manual_count == 0:
        return  # No trades to log

    # Load or create CSV
    if os.path.exists(file_path):
        df_log = pd.read_csv(file_path)
    else:
        df_log = pd.DataFrame(columns=[
            "date", "bot_pnl", "manual_pnl", "total_pnl",
            "bot_trades", "manual_trades",
            "bot_wins", "bot_losses", "manual_wins", "manual_losses"
        ])
    
    if today in pd.to_datetime(df_log["date"], errors='coerce').dt.date.values:
        return

    new_row = pd.DataFrame([{
        "date": today,
        "bot_pnl": bot_pnl,
        "manual_pnl": manual_pnl,
        "total_pnl": total_pnl,
        "bot_trades": bot_count,
        "manual_trades": manual_count,
        "bot_wins": bot_wins,
        "bot_losses": bot_losses,
        "manual_wins": manual_wins,
        "manual_losses": manual_losses
    }])

    df_log = pd.concat([df_log, new_row], ignore_index=True)
    df_log.to_csv(file_path, index=False)
# 🧮 Calculate dynamic position size based on risk %
def position_size_from_risk(account_balance, risk_percent, entry_price, stop_loss_price):
    risk_amount = account_balance * (risk_percent / 100)
    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return 0
    position_size = risk_amount / risk_per_unit
    return round(position_size, 3)


# ⚠️ Detect win rate or PnL trend changes
def get_trend_change_alerts(df):
    alerts = []

    if len(df) < 8:
        return alerts  # Not enough data

    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # Win rate drop (Bot)
    if not pd.isna(today["bot_win_rate_7d"]) and not pd.isna(yesterday["bot_win_rate_7d"]):
        drop = yesterday["bot_win_rate_7d"] - today["bot_win_rate_7d"]
        if drop >= 20:
            alerts.append(f"⚠️ Bot Win Rate dropped by {drop:.1f}% compared to 7-day trend.")

    # Win rate drop (Manual)
    if not pd.isna(today["manual_win_rate_7d"]) and not pd.isna(yesterday["manual_win_rate_7d"]):
        drop = yesterday["manual_win_rate_7d"] - today["manual_win_rate_7d"]
        if drop >= 20:
            alerts.append(f"⚠️ Manual Win Rate dropped by {drop:.1f}% compared to 7-day trend.")

    # PnL reversal detection
    if yesterday["bot_pnl_7d"] > 0 and today["bot_pnl_7d"] < 0:
        alerts.append("🔻 Bot 7-Day Avg PnL turned negative.")
    if yesterday["manual_pnl_7d"] > 0 and today["manual_pnl_7d"] < 0:
        alerts.append("🔻 Manual 7-Day Avg PnL turned negative.")

    return alerts

    # PnL reversal
    if yesterday["bot_pnl_7d"] > 0 and today["bot_pnl_7d"] < 0:
        alerts.append("🔻 Bot 7-Day Avg PnL turned negative.")
    if yesterday["manual_pnl_7d"] > 0 and today["manual_pnl_7d"] < 0:
        alerts.append("🔻 Manual 7-Day Avg PnL turned negative.")

    return alerts

df_trades = load_trades()

# Log area to display SL updates
sl_log = st.empty()
        
df_open_positions = load_open_positions()
trailing_stop_loss(log_slot=sl_log)  # ✅ This logs messages to dashboard

df_manual_closed = load_closed_manual_trades()
df_bot_open, df_bot_closed = split_bot_trades(df_trades)

# ✅ Ensure columns exist
if "Realized PnL ($)" not in df_manual_closed.columns:
    df_manual_closed["Realized PnL ($)"] = 0
if "Realized PnL ($)" not in df_bot_closed.columns:
    df_bot_closed["Realized PnL ($)"] = 0

# 🛑 Show loss-limit lock banner (only if override is OFF)
if should_show_risk_banner(df_bot_closed, df_manual_closed):
    mariah_speak("Warning. Mariah is pausing trades due to risk limit.")
    st.markdown("""
    <div style="background-color: rgba(255, 0, 0, 0.15); padding: 1rem; border-left: 6px solid red; border-radius: 8px;">
        <h4 style="color: red;">🚨 BOT DISABLED: Daily Loss Limit Reached</h4>
        <p style="color: #ffcccc;">Mariah has paused all trading for today to protect your capital. Override is OFF. 🛡️</p>
    </div>
    """, unsafe_allow_html=True)

# ✅ Show override ON badge if enabled
if st.session_state.get("override_risk_lock"):
    st.markdown("""
    <div class="override-glow" style="background-color: rgba(0, 255, 245, 0.15); padding: 1rem;
                 border-left: 6px solid #00fff5; border-radius: 8px;">
        <h4 style="color: #00fff5;">✅ Override Active</h4>
        <p style="color: #ccffff;">Mariah is trading today even though the risk lock was triggered. Use with caution. 😈</p>
    </div>
    """, unsafe_allow_html=True)

    # 🗣 Mariah speaks when override is turned on — only once
    if "override_voice_done" not in st.session_state:
        mariah_speak("Override active. Proceeding with caution.")
        st.session_state["override_voice_done"] = True

# 📭 Show banner if no trades today
if df_bot_closed.empty and df_manual_closed.empty:
    st.warning("📭 No trades recorded today. Your bot or manual log may be empty.")

# ✅ Safe to log daily stats now
log_daily_pnl_split(df_bot_closed, df_manual_closed)

# 🔢 Calculate PnL
open_pnl = df_open_positions["PnL ($)"].sum() if not df_open_positions.empty else 0
closed_pnl = df_bot_closed["Realized PnL ($)"].sum() + df_manual_closed["Realized PnL ($)"].sum()

# ============================
# ✅ Global PnL Summary in Frosted Glass
# ============================

st.markdown("""
<div class="blur-card">
    <h2>📊 Global PnL Summary</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("🌐 All Trades Summary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    pnl_class = "pnl-positive" if open_pnl >= 0 else "pnl-negative"
    st.markdown(f"""
    <div class="blur-card">
        <div class="pnl-label">📈 Open PnL (Unrealized)</div>
        <div class="{pnl_class}" style="font-size: 2rem;">${open_pnl:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pnl_class_closed = "pnl-positive" if closed_pnl >= 0 else "pnl-negative"
    st.markdown(f"""
    <div class="blur-card">
        <div class="pnl-label">✅ Closed PnL (Realized)</div>
        <div class="{pnl_class_closed}" style="font-size: 2rem;">${closed_pnl:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# ✅ Smart Tab Layout System
# ============================

# 🌟 Top 8 Tabs Including Mariah
main_tabs = [
    "🌐 All Trades", "📈 Bot Open Trades", "✅ Bot Closed Trades",
    "🔥 Manual Open Trades", "✅ Manual Closed Trades",
    "📊 Growth Curve", "🛒 Place Trade", "🧠 Mariah AI"
]

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(main_tabs)

# 🌐 All Trades
with tab1:
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("🌐 All Trades Summary")

    # ✅ Bot Closed Trades
    if df_bot_closed.empty:
        st.info("No bot closed trades yet.")
    else:
        df_bot_closed_display = df_bot_closed.copy()
        df_bot_closed_display["timestamp"] = df_bot_closed_display.get("timestamp", "")
        df_bot_closed_display["note"] = df_bot_closed_display.get("note", "")
        st.subheader("✅ Bot Closed Trades")
        st.dataframe(df_bot_closed_display[[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "note",
            "Realized PnL ($)", "Realized PnL (%)"
        ]])

    # ✅ Manual Closed Trades
    if df_manual_closed.empty:
        st.info("No manual closed trades yet.")
    else:
        df = df_manual_closed.copy()
        df_aligned = pd.DataFrame({
            "timestamp": df.get("timestamp", [""] * len(df)),
            "symbol": df.get("Symbol", df.get("symbol", "")),
            "side": df.get("Side", df.get("side", "")),
            "qty": df.get("Size", df.get("qty", "")),
            "entry_price": df.get("Entry Price", df.get("entry_price", "")),
            "stop_loss": 0,
            "take_profit": df.get("Exit Price", df.get("take_profit", "")),
            "note": df.get("note", ""),
            "Realized PnL ($)": df["Realized PnL ($)"],
            "Realized PnL (%)": df["Realized PnL (%)"]
        })
        st.subheader("✅ Manual Closed Trades")
        st.dataframe(df_aligned[[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "note",
            "Realized PnL ($)", "Realized PnL (%)"
        ]])

    # 🔥 Manual Open Trades
    try:
        res = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )
        live_positions = res["result"]["list"]
        parsed = []

        bot_open_keys = set(
            f"{row['symbol']}|{row['qty']}|{row['entry_price']}"
            for _, row in df_bot_open.iterrows()
        )

        for t in live_positions:
            try:
                size = float(t.get("positionValue") or 0)
                symbol = t.get("symbol", "")
                entry_price = float(t.get("avgPrice", 0))
                key = f"{symbol}|{size}|{entry_price}"

                if size > 0 and key not in bot_open_keys:
                    parsed.append({
                        "timestamp": t.get("updatedTime", ""),
                        "symbol": symbol,
                        "side": t.get("side", "Buy" if size > 0 else "Sell"),
                        "qty": size,
                        "entry_price": entry_price,
                        "stop_loss": float(t.get("stopLoss", 0) or 0),
                        "take_profit": float(t.get("markPrice", 0)),
                        "note": "manual",
                        "Realized PnL ($)": float(t.get("unrealisedPnl", 0)),
                        "Realized PnL (%)": 0.0
                    })
            except Exception:
                continue

        if parsed:
            st.subheader("🔥 Manual Open Trades (Live)")
            df_manual_open_all = pd.DataFrame(parsed)
            st.dataframe(df_manual_open_all[[
                "timestamp", "symbol", "side", "qty", "entry_price",
                "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
        else:
            st.info("No open manual trades found.")

    except Exception as e:
        st.error(f"❌ Error loading manual open trades: {e}")

    # ✅ Close the frosted glass panel
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📈 Bot Open Trades")

    if df_bot_open.empty:
        st.info("No active bot trades.")
    else:
        df_bot_open_display = df_bot_open.copy()
        df_bot_open_display["timestamp"] = df_bot_open_display.get("timestamp", "")
        df_bot_open_display["note"] = df_bot_open_display.get("note", "")
        df_bot_open_display["Realized PnL ($)"] = "" 
        df_bot_open_display["Realized PnL (%)"] = ""

        st.dataframe(df_bot_open_display[[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "note",
            "Realized PnL ($)", "Realized PnL (%)"
        ]])

    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("✅ Bot Closed Trades")
    if df_bot_closed.empty:
        st.info("No closed bot trades yet.")
    else:
        df_bot_closed_display = df_bot_closed.copy()
        df_bot_closed_display["timestamp"] = df_bot_closed_display.get("timestamp", "")
        df_bot_closed_display["note"] = df_bot_closed_display.get("note", "")
        st.dataframe(df_bot_closed_display[[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "note",
            "Realized PnL ($)", "Realized PnL (%)"
        ]])
with tab4:
    st.subheader("🔥 Manual Open Trades (Live positions not logged by bot)")

    try:
        res = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )
        live_positions = res["result"]["list"]
        parsed = []

        # Match against existing bot trades
        bot_open_keys = set(
            f"{row['symbol']}|{row['qty']}|{row['entry_price']}"
            for _, row in df_bot_open.iterrows()
        )

        for t in live_positions:
            try:
                size = float(t.get("positionValue") or 0)
                symbol = t.get("symbol", "")
                entry_price = float(t.get("avgPrice", 0))
                key = f"{symbol}|{size}|{entry_price}"

                if size > 0 and key not in bot_open_keys:
                    parsed.append({
                        "timestamp": t.get("updatedTime", ""),
                        "symbol": symbol,
                        "side": t.get("side", "Buy" if size > 0 else "Sell"),
                        "qty": size,
                        "entry_price": entry_price,
                        "stop_loss": float(t.get("stopLoss", 0) or 0),
                        "take_profit": float(t.get("markPrice", 0)),
                        "note": "manual",
                        "Realized PnL ($)": float(t.get("unrealisedPnl", 0)),
                        "Realized PnL (%)": 0.0
                    })
            except Exception:
                continue

        if not parsed:
            st.warning("No open manual trades found.")
        else:
            df_manual_open = pd.DataFrame(parsed)
            st.dataframe(df_manual_open[[
                "timestamp", "symbol", "side", "qty", "entry_price",
                "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])

    except Exception as e:
        st.error(f"❌ Failed to fetch open manual trades: {e}")

with tab5:
    st.subheader("✅ Manual Closed Trades")

    if df_manual_closed.empty:
        st.info("No closed manual trades found.")
    else:
        aligned_rows = []

        for i, row in df_manual_closed.iterrows():
            aligned_rows.append({
                "timestamp": row.get("timestamp", ""),
                "symbol": row.get("symbol", row.get("Symbol", "")),
                "side": row.get("side", row.get("Side", "")),
                "qty": row.get("qty", row.get("Size", "")),
                "entry_price": row.get("entry_price", row.get("Entry Price", "")),
                "stop_loss": 0,
                "take_profit": row.get("take_profit", row.get("Exit Price", "")),
                "note": row.get("note", ""),
                "Realized PnL ($)": row.get("Realized PnL ($)", ""),
                "Realized PnL (%)": row.get("Realized PnL (%)", "")
            })

        aligned_df = pd.DataFrame(aligned_rows)

        st.dataframe(aligned_df[[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "note",
            "Realized PnL ($)", "Realized PnL (%)"
        ]])


from plotly.subplots import make_subplots

with tab6:
    st.subheader("📊 Bot Trading Growth Curve (Cumulative + Daily PnL)")

    if df_trades.empty or "take_profit" not in df_trades.columns:
        st.warning("No bot trades available to plot.")
    else:
        # Ensure timestamp exists and is datetime
        df_trades["timestamp"] = pd.to_datetime(df_trades.get("timestamp", pd.Timestamp.now()), errors='coerce')

        # Filter for closed bot trades
        df_closed = df_trades[df_trades["take_profit"] != 0].copy()

        if df_closed.empty:
            st.info("No closed bot trades found to generate growth curve.")
        else:
            df_closed = df_closed.sort_values("timestamp")

            # Fee-adjusted realized PnL
            df_closed["Realized PnL ($)"] = (
                (df_closed["take_profit"] - df_closed["entry_price"]) * df_closed["qty"]
                - (df_closed["entry_price"] + df_closed["take_profit"]) * df_closed["qty"] * FEE_RATE
            )

            df_closed["Cumulative PnL"] = df_closed["Realized PnL ($)"].cumsum()

            # Daily PnL aggregation
            df_closed["date"] = df_closed["timestamp"].dt.date
            df_daily = df_closed.groupby("date").agg({
                "Realized PnL ($)": "sum"
            }).reset_index()

            # Create split subplot
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.08,
                subplot_titles=("📈 Cumulative Bot PnL", "📊 Daily Realized PnL")
            )

            # Line: Cumulative PnL
            fig.add_scatter(
                x=df_closed["timestamp"],
                y=df_closed["Cumulative PnL"],
                mode="lines+markers",
                name="Cumulative PnL",
                row=1, col=1
            )

            # Bar: Daily PnL
            colors = ["green" if v >= 0 else "red" for v in df_daily["Realized PnL ($)"]]
            fig.add_bar(
                x=df_daily["date"],
                y=df_daily["Realized PnL ($)"],
                name="Daily PnL",
                marker_color=colors,
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                showlegend=False,
                margin=dict(t=60, b=40),
                xaxis=dict(title=""),
                yaxis=dict(title="Cumulative $"),
                xaxis2=dict(title="Date"),
                yaxis2=dict(title="Daily PnL ($)")
            )

            st.plotly_chart(fig, use_container_width=True)

            # 📊 Trade Performance Metrics
            st.markdown("---")
            st.subheader("📌 Bot Trade Performance Summary")

            total_trades = len(df_closed)
            wins = df_closed[df_closed["Realized PnL ($)"] > 0]
            losses = df_closed[df_closed["Realized PnL ($)"] < 0]

            win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
            avg_win = wins["Realized PnL ($)"].mean() if not wins.empty else 0
            avg_loss = losses["Realized PnL ($)"].mean() if not losses.empty else 0
            profit_factor = wins["Realized PnL ($)"].sum() / abs(losses["Realized PnL ($)"].sum()) if not losses.empty else float('inf')

            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Total Bot Trades", total_trades)
            col2.metric("✅ Win Rate", f"{win_rate:.2f}%")
            col3.metric("⚖️ Profit Factor", f"{profit_factor:.2f}")

            col4, col5 = st.columns(2)
            col4.metric("🟢 Avg Win ($)", f"${avg_win:.2f}")
            col5.metric("🔻 Avg Loss ($)", f"${avg_loss:.2f}")

with tab7:
    st.subheader("🛒 Place Live Trade")

    # Dropdowns for symbol and side
    symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"])
    side = st.selectbox("Side", ["Buy", "Sell"])

    # Quantity from sidebar position sizing (but editable)
    qty = st.number_input(
        "Quantity (auto-filled from sidebar)",
        min_value=0.001,
        value=max(0.001, float(qty_calc)),
        step=0.001,
        format="%.3f"
    )

    # 🚀 Place market order
    if st.button("🚀 Place Market Order"):
        try:
            # ✅ Execute the order
            order = session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=round(qty, 3),
                timeInForce="GoodTillCancel",
                reduceOnly=False,
                closeOnTrigger=False
            )

            # ✅ Mariah confirms trade
            mariah_speak(f"Order executed. {side} {qty} {symbol}.")

            # ✅ Speak if override is active
            if st.session_state.get("override_risk_lock"):
                mariah_speak("Override active. Proceeding with caution.")

            # ✅ Log the trade
            log_manual_trade_to_csv(
                symbol=symbol,
                side=side,
                qty=round(qty, 3),
                entry_price=entry_price_sidebar,
                stop_loss=stop_loss_sidebar,
                take_profit=0
            )

            st.success(f"✅ Order placed: {side} {qty} {symbol}")
            st.write("Order Response:", order)

        except Exception as e:
            mariah_speak("Order failed. Check trade parameters.")
            st.error(f"❌ Order failed: {e}")

# ✅ Mariah GPT-powered reply engine
def get_mariah_reply(prompt, open_pnl, closed_pnl, override_on):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Mariah, an AI trading assistant. "
                        "You are protective, smart, and intuitive. "
                        "You speak with confidence, explain your decisions clearly, and always put Jonathan’s capital first."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Mariah GPT error: {e}"

# ✅ Tab 8: Mariah AI Chat
with tab8:   
    st.subheader("🧠 Talk to Mariah")

    # ✅ Mode Colors + Dynamic Styling
    mode_colors = {
        "Scalping": "#00ffcc",     # Aqua Green
        "Swing": "#ffaa00",        # Orange
        "Momentum": "#ff4d4d"      # Red
    }

    # ✅ Display Strategy Mode with Color
    st.markdown(
        f"<span style='font-size: 1.1rem; font-weight: 600;'>🚦 Current Strategy Mode: "
        f"<span style='color: {mode_colors[mode]};'>{mode}</span></span>",
        unsafe_allow_html=True
    )

    # 💬 Text input   
    user_input = st.chat_input("Ask Mariah anything...", key="mariah_chat_input")

    if user_input:
        st.chat_message("user").markdown(user_input)
        override_on = st.session_state.get("override_risk_lock", False)
        response = get_mariah_reply(user_input, open_pnl, closed_pnl, override_on)
        st.chat_message("assistant").markdown(response)
        mariah_speak(response)

    st.markdown("---")
    st.markdown("🎙 Or press below to speak:")
        
    # 🎤 Voice input (via mic)
    if st.button("🎙 Speak to Mariah"):
        voice_input = listen_to_user()
        if voice_input:
            st.chat_message("user").markdown(voice_input)  
            override_on = st.session_state.get("override_risk_lock", False)
            response = get_mariah_reply(voice_input, open_pnl, closed_pnl, override_on)
            st.chat_message("assistant").markdown(response)
            mariah_speak(response)

if more_tab == "📡 Signal Scanner":   
    with st.container():
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📡 RSI Signal Scanner")
        
        # 💰 Risk & Trade Settings
        account_balance = st.sidebar.number_input("Account Balance ($)", value=5000.0)
        interval = st.selectbox("Candle Interval", ["5", "15", "30", "60", "240"], index=1)
        symbols = st.multiselect("Symbols to Scan", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], default=["BTCUSDT", "ETHUSDT"])

        st.markdown("---")

        for symbol in symbols:
            # 🛡️ Daily loss guard
            if check_max_daily_loss(df_bot_closed, df_manual_closed):
                st.warning(f"🛑 Mariah skipped {symbol} — Daily loss limit reached.")
                continue

            try:
                # ✅ Scan for RSI signal
                rsi_value, trigger = check_rsi_signal(symbol=symbol, interval=interval, mode=mode)

                if trigger:
                    # ✅ Use mode-specific SL/TP from strategy_mode.py
                    sl_pct, tp_pct, rsi_overbought, rsi_oversold = get_strategy_params(mode)

                    entry_price = rsi_value  # If your scanner returns price, adjust accordingly
                    stop_loss = entry_price * (1 - sl_pct / 100)
                    take_profit = entry_price * (1 + tp_pct / 100)

                    risk_percent = get_risk_percent_from_strategy("RSI signal")
                    qty = position_size_from_risk(account_balance, risk_percent, entry_price, stop_loss)
 
                    # 🔊 Mariah speaks before placing the trade
                    mariah_speak(
                        f"Entering a {mode.lower()} trade on {symbol}. "
                        f"Stop loss at {sl_pct} percent. Take profit at {tp_pct} percent. "
                        f"Risking {risk_percent} percent of capital."
                    )

                    # 🚀 Place order
                    order = session.place_order(
                        category="linear",
                        symbol=symbol,
                        side="Buy",
                        orderType="Market",
                        qty=round(qty, 3),
                        timeInForce="GoodTillCancel",
                        reduceOnly=False,
                        closeOnTrigger=False
                    )

                    # 🧠 Log trade
                    log_manual_trade_to_csv(
                        symbol=symbol,
                        side="Buy",
                        qty=qty,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        note=f"RSI signal ({mode})"
                    )

                    st.success(f"✅ RSI trade executed: {symbol} @ ${entry_price:.2f}")

                else:
                    st.info(f"No RSI signal for {symbol}.")

            except Exception as e:
                st.error(f"❌ Error for {symbol}: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

if more_tab == "📆 Daily PnL":
    with st.container():
        st.subheader("📆 Daily PnL Breakdown (Bot vs Manual + Total + Trade Counts)")
        # ... same tab 8 content moved here

    pnl_file = "daily_pnl_split.csv"

    if not os.path.exists(pnl_file):
        st.warning("No daily PnL data found yet. Trade activity will log PnL automatically.")
    else:
        df_pnl = pd.read_csv(pnl_file)
        df_pnl["date"] = pd.to_datetime(df_pnl["date"])

        # Prepare melted df for grouped bars
        df_melted = df_pnl.melt(
            id_vars=["date", "total_pnl", "bot_trades", "manual_trades"],
            value_vars=["bot_pnl", "manual_pnl"],
            var_name="Type",
            value_name="PnL"
        )

        # Map labels for clarity
        type_labels = {
            "bot_pnl": "🤖 Bot PnL",
            "manual_pnl": "👤 Manual PnL"
        }
        df_melted["Type"] = df_melted["Type"].map(type_labels)

        # Tooltip data
        df_melted["Trades"] = df_melted.apply(
            lambda row: row["bot_trades"] if row["Type"] == "🤖 Bot PnL" else row["manual_trades"], axis=1
        )

        # Plot grouped bars
        fig = px.bar(
            df_melted,
            x="date",
            y="PnL",
            color="Type",
            barmode="group",
            title="Daily Bot vs Manual PnL with Total Line & Trade Counts",
            color_discrete_sequence=["green", "orange"],
            hover_data=["Trades"]
        )

        # Add total PnL line
        fig.add_scatter(
            x=df_pnl["date"],
            y=df_pnl["total_pnl"],
            mode="lines+markers",
            name="📈 Total PnL",
            line=dict(color="blue", width=2, dash="dash")
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="PnL ($)",
            legend_title="Trade Type",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # 🚨 Trade quality alerts
        alerts = get_trade_quality_alerts(df_pnl, df_bot_closed)
        if alerts:
            st.markdown("### ⚠️ Trade Quality Alerts")
            for alert in alerts:
                st.warning(alert)

        # 📋 Display daily summary table
        st.markdown("### 🧾 Daily Summary Table")
        df_table = df_pnl[[
            "date", "bot_pnl", "manual_pnl", "total_pnl", "bot_trades", "manual_trades"
        ]].copy()

        df_table.columns = [
            "Date", "🤖 Bot PnL", "👤 Manual PnL", "📈 Total PnL", "📊 Bot Trades", "📊 Manual Trades"
        ]

        st.dataframe(df_table.sort_values("Date", ascending=False), use_container_width=True)

        # ✅ Win/Loss Accuracy Table
        st.markdown("### ✅ Win/Loss Accuracy Summary")

        df_summary = df_pnl[[
            "date", "bot_wins", "bot_losses", "manual_wins", "manual_losses"
        ]].copy()

        df_summary["bot_win_rate"] = df_summary.apply(
            lambda row: (row["bot_wins"] / (row["bot_wins"] + row["bot_losses"])) * 100
            if (row["bot_wins"] + row["bot_losses"]) > 0 else 0,
            axis=1
        )

        df_summary["manual_win_rate"] = df_summary.apply(
            lambda row: (row["manual_wins"] / (row["manual_wins"] + row["manual_losses"])) * 100
            if (row["manual_wins"] + row["manual_losses"]) > 0 else 0,
            axis=1
        )

        df_summary = df_summary.rename(columns={
            "date": "Date",
            "bot_wins": "🤖 Bot Wins",
            "bot_losses": "❌ Bot Losses",
            "manual_wins": "👤 Manual Wins",
            "manual_losses": "❌ Manual Losses",
            "bot_win_rate": "🤖 Bot Win Rate (%)",
            "manual_win_rate": "👤 Manual Win Rate (%)"
        })

        st.dataframe(df_summary.sort_values("Date", ascending=False), use_container_width=True)

# ✅ More Tools: On-Chain Data (tab12 replacement)
if more_tab == "📡 On-Chain Data":
    with st.container():
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📡 ETH Gas + Block Data (Etherscan)")

        try:
            gas = get_eth_gas()
            block = get_block_info()

            if not gas:
                st.warning("Could not retrieve gas data.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⛽ Safe Gas", f"{gas['low']} Gwei")
                col2.metric("⚡ Avg Gas", f"{gas['avg']} Gwei")
                col3.metric("🚀 Fast Gas", f"{gas['high']} Gwei")
                col4.metric("📦 Last Block", f"{block}")
        except Exception as e:
            st.error(f"❌ Failed to load on-chain data: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

if more_tab == "📈 Performance Trends":
    with st.container():
        st.subheader("📈 Performance Trends (Win Rate, PnL, Profit Factor, Trend Alerts)")
        # Paste tab9 content here

    if not os.path.exists("daily_pnl_split.csv"):
        st.warning("No trend data yet — trades must be logged first.")
    else:
        df = pd.read_csv("daily_pnl_split.csv")
        df["date"] = pd.to_datetime(df["date"])

        # === Rolling Win Rate ===
        df["bot_win_rate"] = df.apply(
            lambda row: (row["bot_wins"] / (row["bot_wins"] + row["bot_losses"])) * 100
            if (row["bot_wins"] + row["bot_losses"]) > 0 else 0,
            axis=1
        )
        df["manual_win_rate"] = df.apply(
            lambda row: (row["manual_wins"] / (row["manual_wins"] + row["manual_losses"])) * 100
            if (row["manual_wins"] + row["manual_losses"]) > 0 else 0,
            axis=1
        )

        df["bot_win_rate_7d"] = df["bot_win_rate"].rolling(window=7).mean()
        df["manual_win_rate_7d"] = df["manual_win_rate"].rolling(window=7).mean()

        st.markdown("### 🧠 7-Day Rolling Win Rate (%)")
        fig_winrate = px.line(
            df,
            x="date",
            y=["bot_win_rate_7d", "manual_win_rate_7d"],
            labels={"value": "Win Rate (%)", "date": "Date"},
            title="Rolling 7-Day Win Rate (Bot vs Manual)"
        )
        fig_winrate.update_traces(mode="lines+markers")
        fig_winrate.update_layout(height=400)
        st.plotly_chart(fig_winrate, use_container_width=True)

        # === Cumulative PnL ===
        st.markdown("### 📈 Cumulative PnL Over Time")
        df["bot_cum_pnl"] = df["bot_pnl"].cumsum()
        df["manual_cum_pnl"] = df["manual_pnl"].cumsum()

        fig_cum = px.line(
            df,
            x="date",
            y=["bot_cum_pnl", "manual_cum_pnl"],
            labels={"value": "Cumulative PnL ($)", "date": "Date"},
            title="Cumulative Profit (Bot vs Manual)"
        )
        fig_cum.update_traces(mode="lines+markers")
        fig_cum.update_layout(height=400)
        st.plotly_chart(fig_cum, use_container_width=True)

        # === Rolling Avg PnL ===
        st.markdown("### 🔄 7-Day Rolling Average PnL")
        df["bot_pnl_7d"] = df["bot_pnl"].rolling(window=7).mean()
        df["manual_pnl_7d"] = df["manual_pnl"].rolling(window=7).mean()

        fig_rolling_pnl = px.line(
            df,
            x="date",
            y=["bot_pnl_7d", "manual_pnl_7d"],
            labels={"value": "Avg Daily PnL ($)", "date": "Date"},
            title="7-Day Rolling Average Daily PnL"
        )
        fig_rolling_pnl.update_traces(mode="lines+markers")
        fig_rolling_pnl.update_layout(height=400)
        st.plotly_chart(fig_rolling_pnl, use_container_width=True)

        # === Profit Factor Trend ===
        st.markdown("### ⚖️ Daily Profit Factor Trend")

        def safe_profit_factor(pnl):
            if pnl > 0:
                return pnl / 1  # treat win as normal
            elif pnl < 0:
                return pnl / abs(pnl)  # normalize loss
            return 0

        df["bot_profit_factor"] = df["bot_pnl"].apply(safe_profit_factor)
        df["manual_profit_factor"] = df["manual_pnl"].apply(safe_profit_factor)

        fig_pf = px.line(
            df,
            x="date",
            y=["bot_profit_factor", "manual_profit_factor"],
            labels={"value": "Profit Factor", "date": "Date"},
            title="Daily Profit Factor Trend (Bot vs Manual)"
        )
        fig_pf.update_traces(mode="lines+markers")
        fig_pf.update_layout(height=400, yaxis=dict(rangemode='tozero'))
        st.plotly_chart(fig_pf, use_container_width=True)

        # === Trend Change Alerts ===
        st.markdown("### ⚠️ Trend Change Alerts")

        trend_alerts = get_trend_change_alerts(df)
        if trend_alerts:
            for alert in trend_alerts:
                st.warning(alert)
        else:
            st.success("No major trend changes detected.")


if more_tab == "📆 Filter by Date":
    with st.container():
        st.subheader("📆 Filter Trades by Date Range")
        # Paste tab10 content here
    st.subheader("📆 Filter Trades by Date Range")

    if os.path.exists("trades.csv"):
        df_trades = pd.read_csv("trades.csv")
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df_trades = df_trades.dropna(subset=["timestamp"])
        df_trades = df_trades.sort_values("timestamp", ascending=False)

        if "Realized PnL ($)" not in df_trades.columns:
            df_trades["Realized PnL ($)"] = (
                (df_trades["take_profit"] - df_trades["entry_price"]) * df_trades["qty"]
                - (df_trades["entry_price"] + df_trades["take_profit"]) * df_trades["qty"] * 0.00075
            )

        min_date = df_trades["timestamp"].min().date()
        max_date = df_trades["timestamp"].max().date()

        start_date, end_date = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        trade_type = st.selectbox("Filter by Trade Type:", ["All", "bot", "manual"])

        df_filtered = df_trades[
            (df_trades["timestamp"].dt.date >= start_date) &
            (df_trades["timestamp"].dt.date <= end_date)
        ]

        if trade_type != "All":
            df_filtered = df_filtered[df_filtered["note"] == trade_type] if "note" in df_filtered.columns else pd.DataFrame()

        if df_filtered.empty:
            st.warning("No trades found for the selected range and filter.")
        else:
            from datetime import datetime
            st.markdown(f"""
                <div style='background-color:#0e1117;padding:1.2rem 1rem;border-radius:0.5rem;margin-bottom:1rem'>
                    <h2 style='color:white;text-align:center;margin:0'>📊 Trade Analytics Dashboard</h2>
                    <p style='color:#aaa;text-align:center;margin:0'>Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
                </div>
            """, unsafe_allow_html=True)

            pnl_total = df_filtered["Realized PnL ($)"].sum()
            win_rate = (df_filtered["Realized PnL ($)"] > 0).mean() * 100
            color = "green" if win_rate >= 50 else "red"

            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Total Trades", len(df_filtered))
            col2.markdown(f"✅ **Win Rate:** <span style='color:{color}'>{win_rate:.2f}%</span>", unsafe_allow_html=True)
            col3.metric("💰 Realized PnL", f"${pnl_total:.2f}")

            if "note" in df_filtered.columns:
                st.markdown("### 🧠 Strategy Breakdown")
                strategy_counts = df_filtered["note"].value_counts()
                strategy_pnl = df_filtered.groupby("note")["Realized PnL ($)"].sum()

                st.plotly_chart(go.Figure(
                    data=[go.Pie(labels=strategy_counts.index, values=strategy_counts.values, hole=0.3)],
                    layout_title_text="Trade Count by Strategy"
                ), use_container_width=True)

                st.plotly_chart(go.Figure(
                    data=[go.Bar(x=strategy_pnl.index, y=strategy_pnl.values, marker_color="purple")],
                    layout_title_text="Total PnL by Strategy"
                ), use_container_width=True)

            cols_to_show = ["timestamp", "symbol", "side", "qty", "entry_price", "take_profit", "Realized PnL ($)"]
            if "note" in df_filtered.columns:
                cols_to_show.append("note")

            st.markdown("### 🏆 Top 5 Trades by PnL")
            st.dataframe(df_filtered.nlargest(5, "Realized PnL ($)")[cols_to_show])

            st.markdown("### 💀 Worst 5 Trades by PnL")
            st.dataframe(df_filtered.nsmallest(5, "Realized PnL ($)")[cols_to_show])

            best_idx = df_filtered["Realized PnL ($)"].idxmax()
            worst_idx = df_filtered["Realized PnL ($)"].idxmin()
            bar_colors = [
                "green" if i == best_idx else "red" if i == worst_idx else "gray"
                for i in df_filtered.index
            ]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df_filtered["timestamp"],
                y=df_filtered["Realized PnL ($)"],
                marker_color=bar_colors,
                text=df_filtered["symbol"],
                hovertemplate="Symbol: %{text}<br>PnL: $%{y:.2f}<br>Date: %{x|%Y-%m-%d}"
            ))
            fig_bar.update_layout(title="📊 Daily Realized PnL (Best/Worst Highlighted)")
            st.plotly_chart(fig_bar, use_container_width=True)

            df_filtered.loc[:, "Cumulative PnL"] = df_filtered["Realized PnL ($)"].cumsum()
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_filtered["timestamp"],
                y=df_filtered["Cumulative PnL"],
                mode="lines+markers",
                name="Cumulative PnL",
                marker=dict(size=6, color="blue"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Cumulative: $%{y:.2f}"
            ))
            fig_line.add_trace(go.Scatter(
                x=[df_filtered.loc[best_idx, "timestamp"]],
                y=[df_filtered.loc[best_idx, "Cumulative PnL"]],
                mode="markers+text",
                name="Best",
                marker=dict(size=10, color="green", symbol="star"),
                text=["Best"],
                textposition="top center"
            ))
            fig_line.add_trace(go.Scatter(
                x=[df_filtered.loc[worst_idx, "timestamp"]],
                y=[df_filtered.loc[worst_idx, "Cumulative PnL"]],
                mode="markers+text",
                name="Worst",
                marker=dict(size=10, color="red", symbol="star"),
                text=["Worst"],
                textposition="top center"
            ))
            fig_line.update_layout(title="📈 Cumulative PnL with Best/Worst Marked")
            st.plotly_chart(fig_line, use_container_width=True)

            st.markdown("### 🧾 Filtered Trades Table")
            st.dataframe(df_filtered)

            csv = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Filtered Trades as CSV",
                data=csv,
                file_name=f"filtered_trades_{start_date}_to_{end_date}.csv",
                mime="text/csv"
            )

            if st.button("📧 Email This Report"):
                import matplotlib.pyplot as plt
                from fpdf import FPDF
                import os

                fig1, ax1 = plt.subplots()
                df_filtered.plot(x="timestamp", y="Realized PnL ($)", kind="bar", ax=ax1, title="Daily PnL")
                fig1_path = "daily_pnl_chart.png"
                fig1.savefig(fig1_path)
                plt.close(fig1)

                fig2, ax2 = plt.subplots()
                df_filtered["Cumulative PnL"] = df_filtered["Realized PnL ($)"].cumsum()
                df_filtered.plot(x="timestamp", y="Cumulative PnL", kind="line", ax=ax2, marker='o', title="Cumulative PnL")
                fig2_path = "cumulative_pnl_chart.png"
                fig2.savefig(fig2_path)
                plt.close(fig2)

                best_trade = df_filtered.loc[df_filtered["Realized PnL ($)"].idxmax()]
                worst_trade = df_filtered.loc[df_filtered["Realized PnL ($)"].idxmin()]
                strategy_pnl = df_filtered.groupby("note")["Realized PnL ($)"].sum().sort_values(ascending=False) if "note" in df_filtered.columns else {}

                pdf = FPDF()
                pdf.add_page()

                logo_path = "IMG_7006.PNG"
                if os.path.exists(logo_path):
                    pdf.image(logo_path, x=10, y=10, w=40)
                    pdf.ln(25)

                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "🔥 The Crypto Capital - Trade Report", ln=True)

                pdf.set_font("Arial", '', 12)
                pdf.cell(0, 10, f"Date Range: {start_date} to {end_date}", ln=True)
                pdf.cell(0, 10, f"Total Trades: {len(df_filtered)}", ln=True)
                pdf.cell(0, 10, f"Total PnL: ${pnl_total:.2f}", ln=True)
                pdf.cell(0, 10, f"Win Rate: {win_rate:.2f}%", ln=True)

                pdf.ln(6)
                if strategy_pnl:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, "📊 PnL by Strategy:", ln=True)
                    pdf.set_font("Arial", '', 11)
                    for strategy, pnl in strategy_pnl.items():
                        pdf.cell(0, 10, f"{strategy}: ${pnl:.2f}", ln=True)

                pdf.ln(6)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "🏆 Best Trade:", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 8, f"{best_trade.to_string()}")

                pdf.ln(6)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, "💀 Worst Trade:", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.multi_cell(0, 8, f"{worst_trade.to_string()}")

                pdf.ln(6)
                pdf.image(fig1_path, w=170)
                pdf.ln(6)
                pdf.image(fig2_path, w=170)

                pdf_path = f"/mnt/data/trade_report_{start_date}_to_{end_date}.pdf"
                pdf.output(pdf_path)

                try:
                    send_email_with_attachment(
                        subject="Your Daily Trade Report",
                        body="Attached is your trade report.",
                        to_email=os.getenv("EMAIL_TO"),
                        filename=pdf_path
                    )
                    st.success("📬 Email sent successfully!")
                    st.markdown(f"[📄 Download PDF Report]({pdf_path})")
                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")

    else:
        st.info("No trades.csv found. Place a trade or run the bot to generate data.")

from news_feed import get_crypto_news

if more_tab == "📰 Crypto News":
    with st.container():
        st.subheader("📰 Real-Time Crypto News")
        # Paste tab11 content here

    news_items = get_crypto_news()

    if not news_items:
        st.info("No news available right now.")
    else:
        for item in news_items:
            title = item["title"]
            title_lower = title.lower()
            alert = ""
            if any(word in title_lower for word in ["hack", "exploit", "rug pull", "lawsuit", "sec", "liquidation"]):
                alert = "🚨 "
            elif any(word in title_lower for word in ["etf", "partnership", "bullish", "upgrade"]):
                alert = "📈 "

            tags = ", ".join([c["code"] for c in item.get("tags", [])]) if item.get("tags") else ""
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <a href="{item['url']}" target="_blank" style="color:#00fff5; font-weight:bold;">{alert}{item['title']}</a><br>
                <span style="color:gray; font-size: 0.85rem;">{item['published']} | {item['source']} | {tags}</span>
            </div>
            """, unsafe_allow_html=True)

from datetime import date
import matplotlib.pyplot as plt
from fpdf import FPDF

from datetime import date

# ✅ Only send once per day
if st.session_state.get("last_email_sent") != str(date.today()):
    df = pd.read_csv("trades.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df_today = df[df["timestamp"].dt.date == date.today()]

    if not df_today.empty:
        # Calculate PnL if missing
        if "Realized PnL ($)" not in df_today.columns:
            df_today["Realized PnL ($)"] = (
                (df_today["take_profit"] - df_today["entry_price"]) * df_today["qty"]
                - (df_today["entry_price"] + df_today["take_profit"]) * df_today["qty"] * 0.00075
            )

        # Best/worst trades
        best_trade = df_today.loc[df_today["Realized PnL ($)"].idxmax()]
        worst_trade = df_today.loc[df_today["Realized PnL ($)"].idxmin()]
        pnl_total = df_today["Realized PnL ($)"].sum()
        win_rate = (df_today["Realized PnL ($)"] > 0).mean() * 100

        # Charts
        fig, ax = plt.subplots()
        df_today.set_index("timestamp")["Realized PnL ($)"].plot(kind="bar", ax=ax, title="Daily PnL")
        fig_path = "daily_auto_chart.png"
        fig.savefig(fig_path)
        plt.close(fig)

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Auto-Sent Daily Trade Report", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Date: {date.today()}", ln=True)
        pdf.cell(0, 10, f"Total Trades: {len(df_today)}", ln=True)
        pdf.cell(0, 10, f"Total PnL: ${pnl_total:.2f}", ln=True)
        pdf.cell(0, 10, f"Win Rate: {win_rate:.2f}%", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "🔝 Best Trade", ln=True)
        pdf.set_font("Arial", '', 11)
        for k, v in best_trade.to_dict().items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)

        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "🔻 Worst Trade", ln=True)
        pdf.set_font("Arial", '', 11)
        for k, v in worst_trade.to_dict().items():
            pdf.cell(0, 10, f"{k}: {v}", ln=True)

        pdf.ln(5)
        pdf.image(fig_path, w=180)
        auto_pdf_path = f"daily_report_{date.today()}.pdf"
        pdf.output(auto_pdf_path)

        # Send email
        try:
            send_email_with_attachment(
                subject="Daily Trade Report",
                body="Your automated daily report is attached.",
                to_email=os.getenv("EMAIL_TO"),
                filename=auto_pdf_path
            )
            st.session_state["last_email_sent"] = str(date.today())
            st.success("📧 Daily report emailed automatically.")
        except Exception as e:
            st.error(f"❌ Auto-email failed: {e}")

if more_tab == "📡 On-Chain Data":
    with st.container():
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📡 ETH Gas + Block Data (Etherscan)")

        # ✅ Debug message to confirm the logic is running
        st.warning("📡 DEBUG: Inside On-Chain tab")
        
        try:
            st.info("🚀 Fetching ETH gas + block data...")

            gas = get_eth_gas()
            block = get_block_info()

            # Debug output
            st.write("GAS RESULT:", gas)
            st.write("BLOCK RESULT:", block)

            if not gas:
                st.warning("Could not retrieve gas data.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("⛽ Safe Gas", f"{gas.get('low', '?')} Gwei")
                col2.metric("⚡ Avg Gas", f"{gas.get('avg', '?')} Gwei") 
                col3.metric("🚀 Fast Gas", f"{gas.get('high', '?')} Gwei")
                col4.metric("📦 Last Block", str(block) if block else "N/A")

        except Exception as e:
            st.error(f"❌ Failed to load on-chain data: {e}")

        # ✅ Always close blur-card
        st.markdown('</div>', unsafe_allow_html=True)


