"""
Crypto Capital Dashboard - A Streamlit-based cryptocurrency trading dashboard
with PnL tracking, trading signals, and various visualization features.

Author: Jonathan Ferrucci
"""

import os
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_SERVER_ENABLE_WEBSOCKET_COMPRESSION'] = 'false'

# =======================
# IMPORTS
# =======================
import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import base64
import csv
import smtplib
from email.message import EmailMessage
import pyttsx3
import openai
import speech_recognition as sr
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from datetime import datetime, date
from streamlit_autorefresh import st_autorefresh
from PIL import Image
import math
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Local module imports
from strategy_mode import get_strategy_params
from onchain_feed import get_eth_gas, get_block_info
from news_feed import get_crypto_news

# =======================
# CONSTANTS
# =======================
# Risk settings
MAX_DAILY_LOSS = -300  # Daily loss limit in USD
RISK_LOG_PATH = "risk_events.csv"
FEE_RATE = 0.00075  # 0.075% typical Bybit taker fee

# File paths
CSV_FILE = "trades.csv"
DAILY_PNL_FILE = "daily_pnl.csv"
DAILY_PNL_SPLIT_FILE = "daily_pnl_split.csv"

# =======================
# SETUP & CONFIGURATION
# =======================
# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page layout
st.set_page_config(page_title="The Crypto Capital", layout="wide")

# Initialize session state for tracking UI state
if "mariah_greeted" not in st.session_state:
    st.session_state["mariah_greeted"] = False

if "override_risk_lock" not in st.session_state:
    st.session_state["override_risk_lock"] = False

if "test_mode" not in st.session_state:
    st.session_state["test_mode"] = False

if "mute_mariah" not in st.session_state:
    st.session_state["mute_mariah"] = False

# Initialize Bybit API connection
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
session = HTTP(
    api_key=API_KEY,
    api_secret=API_SECRET,
    recv_window=30000  # Increase timeout window (ms)
)

# =======================
# ML SIGNAL GENERATOR
# =======================
class MLSignalGenerator:
    def __init__(self, model_path="models/rf_predictor.pkl"):
        """Initialize the ML signal generator."""
        self.model_path = model_path
        self.model = self._load_model()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self):
        """Load the trained model if exists, otherwise return None."""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                return None
        return None
        
    def train_model(self, historical_data, lookback_periods=14, prediction_horizon=5):
        """Train a new ML model using historical price data."""
        try:
            # Create features (technical indicators)
            df = self._create_features(historical_data)
            
            # Create target: 1 if price goes up by 2% within next N periods, 0 otherwise
            df['future_return'] = df['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            df['target'] = (df['future_return'] > 0.02).astype(int)
            
            # Drop NaNs and prepare data
            df = df.dropna()
            X = df.drop(['target', 'future_return', 'timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1)
            y = df['target']
            
            # Scale features
            self.scaler = StandardScaler().fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            
            accuracy = self.model.score(X_scaled, y)
            self.logger.info(f"Model trained with accuracy: {accuracy:.2f}")
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return None
            
    def _create_features(self, df):
        """Create technical features for the model."""
        df = df.copy()
        
        # Basic indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # Price action features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(14).std()
        df['distance_from_ma50'] = (df['close'] / df['close'].rolling(50).mean()) - 1
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Pattern detection
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                             (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        
        return df
        
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
        
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        middle_band = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = middle_band + std_dev * std
        lower_band = middle_band - std_dev * std
        return upper_band, middle_band, lower_band
        
    def get_signal(self, latest_data):
        """Generate trading signal using the ML model."""
        if self.model is None:
            self.logger.warning("ML model not loaded. Cannot generate signal.")
            return "hold", 0.0
            
        try:
            # Create features for latest data
            df = self._create_features(latest_data)
            df = df.dropna()
            
            if df.empty:
                return "hold", 0.0
                
            # Extract most recent feature set
            features = df.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1).iloc[-1:]
            
            # Scale features
            X_scaled = self.scaler.transform(features)
            
            # Make prediction
            pred_proba = self.model.predict_proba(X_scaled)[0]
            buy_confidence = pred_proba[1]  # Probability of price going up
            
            # Determine signal
            if buy_confidence > 0.7:
                signal = "buy"
            elif buy_confidence < 0.3:
                signal = "sell"
            else:
                signal = "hold"
                
            return signal, buy_confidence
            
        except Exception as e:
            self.logger.error(f"Error generating ML signal: {e}")
            return "hold", 0.0

    def get_feature_importance(self):
        """Return the feature importance from the model."""
        if self.model is None:
            return None
            
        try:
            # Get feature names from the latest created features
            feature_names = self._create_features(pd.DataFrame({
                'timestamp': [], 'open': [], 'high': [], 'low': [], 'close': [], 'volume': []
            })).columns.tolist()
            
            # Filter out non-feature columns
            feature_names = [f for f in feature_names if f not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Return as dictionary
            return dict(zip(feature_names[:len(importances)], importances))
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return None

# =======================
# UTILITY FUNCTIONS
# =======================
def get_base64_image(path):
    """Convert an image to base64 encoding."""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_dashboard_background(image_file):
    """Set the dashboard background with a cinematic glow effect."""
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
        filter: drop-shadow(0 0 12px #00fff5);
        transition: all 0.3s ease-in-out;

    }}
    .pulse-brain {{
        animation: pulse 1.8s infinite ease-in-out;
        transform-origin: center;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.1); opacity: 0.8; }}
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
        0% {{ box-shadow: 0 0 0px #00fff5; }}
        50% {{ box-shadow: 0 0 20px #00fff5; }}
        100% {{ box-shadow: 0 0 0px #00fff5; }}
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
    </style>
    """, unsafe_allow_html=True)

# =======================
# RISK MANAGEMENT FUNCTIONS
# =======================
def check_max_daily_loss(df_bot_closed, df_manual_closed):
    """Check if daily loss limit has been reached."""
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

def log_risk_lock_event(today, pnl_today):
    """Log daily shutdown to risk_events.csv."""
    if os.path.exists(RISK_LOG_PATH):
        df_log = pd.read_csv(RISK_LOG_PATH)
        if today in pd.to_datetime(df_log["date"]).dt.date.values:
            return  # Already logged today
    else:
        df_log = pd.DataFrame(columns=["date", "triggered_at_pnl"])
    
    new_row = pd.DataFrame([{"date": today, "triggered_at_pnl": pnl_today}])
    df_log = pd.concat([df_log, new_row], ignore_index=True)
    df_log.to_csv(RISK_LOG_PATH, index=False)

@st.cache_data(ttl=30)
def should_show_risk_banner(df_bot_closed, df_manual_closed):
    """Determine if risk banner should be displayed."""
    return check_max_daily_loss(df_bot_closed, df_manual_closed)

# =======================
# TRADING SIGNAL FUNCTIONS
# =======================
def get_rsi_signal(rsi_value, mode="Swing"):
    """Evaluate RSI signal based on the selected strategy mode."""
    _, _, rsi_ob, rsi_os = get_strategy_params(mode)
    if rsi_value < rsi_os:
        return "buy"
    elif rsi_value > rsi_ob:
        return "sell"
    else:
        return "hold"

def check_rsi_signal(symbol="BTCUSDT", interval="15", mode="Swing"):
    """Check RSI signal for the given symbol and timeframe."""
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

def log_rsi_trade_to_csv(symbol, side, qty, entry_price, mode="Swing"):
    """Log RSI-triggered trade to CSV file."""
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
        "mode": mode  # Strategy mode logged
    }
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=trade_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade_data)

def get_historical_data(symbol, interval, limit=100):
    """Get historical kline data for ML processing."""
    try:
        res = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=interval,
            limit=limit
        )["result"]["list"]
        
        # Create DataFrame
        df = pd.DataFrame(res, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        
        # Convert types
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"])
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit='ms')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to get historical data for {symbol}: {e}")
        return pd.DataFrame()

# =======================
# TRADE DATA FUNCTIONS
# =======================
def load_open_positions():
    """Load open positions from Bybit API."""
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
    """Load trades from CSV file."""
    return pd.read_csv("trades.csv") if os.path.exists("trades.csv") else pd.DataFrame()

def load_closed_manual_trades():
    """Load closed manual trades from Bybit API."""
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

@st.cache_data(ttl=15)
def split_bot_trades(df):
    """Split bot trades into open and closed trades."""
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

def log_daily_pnl_split(df_bot_closed, df_manual_closed, file_path=DAILY_PNL_SPLIT_FILE):
    """Log daily PnL statistics with split between bot and manual trades."""
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
            "bot_trades", "manual_trades", "bot_wins", "bot_losses",
            "manual_wins", "manual_losses"
        ])
    
    # Skip if already logged today
    if today in pd.to_datetime(df_log["date"], errors='coerce').dt.date.values:
        return
    
    # Add new row
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

def position_size_from_risk(account_balance, risk_percent, entry_price, stop_loss_price):
    """Calculate position size based on risk percentage."""
    risk_amount = account_balance * (risk_percent / 100)
    risk_per_unit = abs(entry_price - stop_loss_price)
    
    if risk_per_unit == 0:
        return 0
    
    position_size = risk_amount / risk_per_unit
    return round(position_size, 3)

def get_trend_change_alerts(df):
    """Detect win rate or PnL trend changes."""
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

def trailing_stop_loss(threshold_pct=0.01, buffer_pct=0.015, log_slot=None):
    """Update trailing stop loss based on profit threshold."""
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

# =======================
# MARIAH AI ASSISTANT FUNCTIONS
# =======================
def mariah_speak(text):
    """Text-to-speech function for Mariah's voice."""
    # Skip if muted
    if st.session_state.get("mute_mariah", False):
        return
        
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)  # Speaking speed
        engine.setProperty('volume', 0.9)  # Volume level
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Mariah voice error: {e}")

def get_mariah_reply(prompt, open_pnl, closed_pnl, override_on):
    """Get AI-generated response from Mariah assistant."""
    try:
        context = (
            f"Open PnL is ${open_pnl:,.2f}. "
            f"Closed PnL is ${closed_pnl:,.2f}. "
            f"Override is {'enabled' if override_on else 'off'}. "
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

def listen_to_user():
    """Speech-to-text function for voice input."""
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
                st.error("❌ I couldn't understand what you said.")
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

def send_email_with_attachment(subject, body, to_email, filename):
    """Send email with attachment."""
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

# =======================
# ADVANCED ANALYTICS FUNCTIONS
# =======================
def render_advanced_analytics(df_trades, df_pnl):
    """
    Render the advanced analytics dashboard
    
    Parameters:
    -----------
    df_trades : DataFrame
        Historical trades data
    df_pnl : DataFrame
        Daily PnL data
    """
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.title("🧠 Advanced Analytics")
    
    # Create tabs for different analytics views
    tabs = st.tabs([
        "📊 Key Metrics", 
        "📈 Equity Curve", 
        "🎯 Trade Distribution", 
        "🧮 Risk Analytics"
    ])
    
    # Prepare data
    if df_trades.empty:
        st.warning("No trade data available for analysis.")
        return
        
    # Ensure necessary columns
    if "Realized PnL ($)" not in df_trades.columns:
        df_trades["Realized PnL ($)"] = (
            (df_trades["take_profit"] - df_trades["entry_price"]) * df_trades["qty"]
            - (df_trades["entry_price"] + df_trades["take_profit"]) * df_trades["qty"] * 0.00075
        )
    
    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"])
    
    # Calculate key metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades["Realized PnL ($)"] > 0])
    losing_trades = len(df_trades[df_trades["Realized PnL ($)"] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    profit = df_trades[df_trades["Realized PnL ($)"] > 0]["Realized PnL ($)"].sum()
    loss = abs(df_trades[df_trades["Realized PnL ($)"] < 0]["Realized PnL ($)"].sum())
    profit_factor = profit / loss if loss > 0 else float('inf')
    
    avg_win = df_trades[df_trades["Realized PnL ($)"] > 0]["Realized PnL ($)"].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades["Realized PnL ($)"] < 0]["Realized PnL ($)"].mean() if losing_trades > 0 else 0
    
    expectancy = (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss) if total_trades > 0 else 0
    
    # Calculate rolling equity curve
    df_trades = df_trades.sort_values("timestamp")
    df_trades["Cumulative PnL"] = df_trades["Realized PnL ($)"].cumsum()
    
    # Calculate drawdowns
    df_trades["Peak"] = df_trades["Cumulative PnL"].cummax()
    df_trades["Drawdown"] = df_trades["Cumulative PnL"] - df_trades["Peak"]
    max_drawdown = abs(df_trades["Drawdown"].min())
    max_drawdown_pct = (max_drawdown / df_trades["Peak"].max() * 100) if df_trades["Peak"].max() > 0 else 0
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0)
    if len(df_trades) > 1:
        returns = df_trades["Realized PnL ($)"].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * math.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Tab 1: Key Metrics
    with tabs[0]:
        st.subheader("Key Performance Metrics")
        
        # Create 3x3 grid of metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Trades", f"{total_trades}")
            st.metric("Win Rate", f"{win_rate:.2f}%")
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            
        with col2:
            st.metric("Net Profit", f"${df_trades['Realized PnL ($)'].sum():.2f}")
            st.metric("Average Win", f"${avg_win:.2f}")
            st.metric("Average Loss", f"${avg_loss:.2f}")
            
        with col3:
            st.metric("Max Drawdown", f"${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
            st.metric("Expectancy", f"${expectancy:.2f}")
        
        # Show trades by symbol
        st.subheader("Performance by Symbol")
        
        symbol_metrics = df_trades.groupby("symbol").agg({
            "Realized PnL ($)": "sum",
            "symbol": "count"
        }).rename(columns={"symbol": "Trade Count"})
        
        symbol_win_rates = []
        for symbol in symbol_metrics.index:
            symbol_df = df_trades[df_trades["symbol"] == symbol]
            wins = len(symbol_df[symbol_df["Realized PnL ($)"] > 0])
            total = len(symbol_df)
            win_rate = (wins / total * 100) if total > 0 else 0
            symbol_win_rates.append(win_rate)
        
        symbol_metrics["Win Rate (%)"] = symbol_win_rates
        st.dataframe(symbol_metrics.sort_values("Realized PnL ($)", ascending=False))
        
        # Monthly performance
        st.subheader("Monthly Performance")
        df_trades["Month"] = df_trades["timestamp"].dt.to_period("M")
        monthly_pnl = df_trades.groupby("Month")["Realized PnL ($)"].sum().reset_index()
        monthly_pnl["Month"] = monthly_pnl["Month"].astype(str)
        
        fig = go.Figure()
        colors = ["green" if x >= 0 else "red" for x in monthly_pnl["Realized PnL ($)"]]
        fig.add_trace(go.Bar(
            x=monthly_pnl["Month"],
            y=monthly_pnl["Realized PnL ($)"],
            marker_color=colors
        ))
        fig.update_layout(title="Monthly Performance", xaxis_title="Month", yaxis_title="PnL ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Equity Curve
    with tabs[1]:
        st.subheader("Equity Curve and Drawdowns")
        
        # Create subplots for equity curve and drawdowns
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=("Equity Curve", "Drawdowns")
        )
        
        # Add equity curve trace
        fig.add_trace(
            go.Scatter(
                x=df_trades["timestamp"], 
                y=df_trades["Cumulative PnL"],
                mode="lines",
                name="Equity",
                line=dict(color="#00fff5", width=2)
            ),
            row=1, col=1
        )
        
        # Add drawdown trace
        fig.add_trace(
            go.Scatter(
                x=df_trades["timestamp"], 
                y=df_trades["Drawdown"],
                mode="lines",
                name="Drawdown",
                line=dict(color="red", width=2),
                fill="tozeroy"
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling metrics
        st.subheader("Rolling Performance")
        
        # Calculate rolling metrics
        window = st.slider("Rolling Window Size", min_value=5, max_value=50, value=20)
        
        df_trades["Rolling Win Rate"] = df_trades["Realized PnL ($)"].apply(
            lambda x: 1 if x > 0 else 0
        ).rolling(window).mean() * 100
        
        df_trades["Rolling PnL"] = df_trades["Realized PnL ($)"].rolling(window).sum()
        
        # Plot rolling metrics
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df_trades["timestamp"],
                y=df_trades["Rolling Win Rate"],
                mode="lines",
                name="Win Rate (%)",
                line=dict(color="green", width=2)
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_trades["timestamp"],
                y=df_trades["Rolling PnL"],
                mode="lines",
                name="PnL ($)",
                line=dict(color="blue", width=2)
            ),
            secondary_y=False
        )
        
        fig.update_layout(
            title=f"{window}-Trade Rolling Metrics",
            xaxis_title="Date",
            yaxis_title="PnL ($)",
            yaxis2_title="Win Rate (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Trade Distribution
    with tabs[2]:
        st.subheader("Trade Distribution Analysis")
        
        # PnL distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df_trades["Realized PnL ($)"],
            nbinsx=20,
            marker_color=["green" if x >= 0 else "red" for x in df_trades["Realized PnL ($)"]],
            opacity=0.7,
            name="PnL Distribution"
        ))
        
        fig.update_layout(
            title="PnL Distribution",
            xaxis_title="PnL ($)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade timing analysis
        st.subheader("Trade Timing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trades by day of week
            df_trades["Day of Week"] = df_trades["timestamp"].dt.day_name()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            day_counts = df_trades.groupby("Day of Week").size().reindex(day_order, fill_value=0)
            day_pnl = df_trades.groupby("Day of Week")["Realized PnL ($)"].sum().reindex(day_order, fill_value=0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=day_counts.index, 
                y=day_counts.values,
                name="Trade Count"
            ))
            
            fig.update_layout(
                title="Trades by Day of Week",
                xaxis_title="Day",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # PnL by day of week
            fig = go.Figure()
            
            colors = ["green" if x >= 0 else "red" for x in day_pnl.values]
            fig.add_trace(go.Bar(
                x=day_pnl.index, 
                y=day_pnl.values,
                marker_color=colors,
                name="PnL"
            ))
            
            fig.update_layout(
                title="PnL by Day of Week",
                xaxis_title="Day",
                yaxis_title="PnL ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade duration analysis (if we have exit timestamps)
        if "exit_timestamp" in df_trades.columns:
            st.subheader("Trade Duration Analysis")
            
            df_trades["Duration"] = (df_trades["exit_timestamp"] - df_trades["timestamp"]).dt.total_seconds() / 3600  # hours
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_trades["Duration"],
                y=df_trades["Realized PnL ($)"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=df_trades["Realized PnL ($)"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="PnL ($)")
                ),
                text=df_trades["symbol"],
                name="Trade Duration vs PnL"
            ))
            
            fig.update_layout(
                title="Trade Duration vs PnL",
                xaxis_title="Duration (hours)",
                yaxis_title="PnL ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Risk Analytics
    with tabs[3]:
        st.subheader("Risk Analytics")
        
        # Calculate risk of ruin
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=100)
        risk_per_trade_pct = st.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        
        # Simple Kelly criterion calculation
        if win_rate > 0 and avg_win != 0 and avg_loss != 0:
            w = win_rate / 100
            r = abs(avg_win / avg_loss)
            kelly_pct = (w*r - (1-w)) / r * 100
            optimal_f = max(0, kelly_pct) / 2  # Half-Kelly for safety
            
            st.metric("Kelly Criterion", f"{kelly_pct:.2f}%")
            st.metric("Half Kelly (Recommended Risk %)", f"{optimal_f:.2f}%")
        
        # Monte Carlo simulation
        st.subheader("Monte Carlo Simulation")
        
        num_simulations = st.slider("Number of Simulations", min_value=100, max_value=1000, value=200)
        num_trades = st.slider("Number of Future Trades", min_value=50, max_value=500, value=100)
        
        # Run Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        
        # Get win rate and PnL distribution for sampling
        win_prob = win_rate / 100
        win_pnl = df_trades[df_trades["Realized PnL ($)"] > 0]["Realized PnL ($)"].values
        loss_pnl = df_trades[df_trades["Realized PnL ($)"] < 0]["Realized PnL ($)"].values
        
        # If no wins or losses, use averages
        if len(win_pnl) == 0:
            win_pnl = np.array([10])
        if len(loss_pnl) == 0:
            loss_pnl = np.array([-10])
        
        # Run simulations
        all_equity_curves = []
        all_max_drawdowns = []
        all_final_equities = []
        
        for sim in range(num_simulations):
            equity = initial_capital
            equity_curve = [initial_capital]
            peak = initial_capital
            
            for _ in range(num_trades):
                # Determine if win or loss
                if np.random.random() < win_prob:
                    # Win: sample from win distribution
                    pnl = np.random.choice(win_pnl)
                else:
                    # Loss: sample from loss distribution
                    pnl = np.random.choice(loss_pnl)
                
                equity += pnl
                equity = max(0, equity)  # Prevent negative equity
                equity_curve.append(equity)
                
                # Track drawdown
                peak = max(peak, equity)
            
            all_equity_curves.append(equity_curve)
            all_final_equities.append(equity)
            
            # Calculate max drawdown for this simulation
            peaks = pd.Series(equity_curve).cummax()
            drawdowns = pd.Series(equity_curve) - peaks
            max_dd = abs(drawdowns.min())
            all_max_drawdowns.append(max_dd)
        
        # Calculate statistics from simulations
        median_equity = np.median(all_final_equities)
        pct_5 = np.percentile(all_final_equities, 5)
        pct_95 = np.percentile(all_final_equities, 95)
        
        profit_prob = sum(1 for eq in all_final_equities if eq > initial_capital) / num_simulations * 100
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Median Final Equity", f"${median_equity:.2f}")
            
        with col2:
            st.metric("5th Percentile", f"${pct_5:.2f}")
            
        with col3:
            st.metric("95th Percentile", f"${pct_95:.2f}")
            
        st.metric("Probability of Profit", f"{profit_prob:.2f}%")
        
        # Plot Monte Carlo simulation results
        fig = go.Figure()
        
        # Plot each simulation
        for i, curve in enumerate(all_equity_curves):
            if i == 0:  # First curve with visible legend
                fig.add_trace(go.Scatter(
                    y=curve,
                    mode="lines",
                    line=dict(color="rgba(0, 255, 245, 0.1)"),
                    name="Simulation Path"
                ))
            else:  # Rest without legend entries
                fig.add_trace(go.Scatter(
                    y=curve,
                    mode="lines",
                    line=dict(color="rgba(0, 255, 245, 0.1)"),
                    showlegend=False
                ))
        
        # Add median curve
        median_curve = np.median(np.array(all_equity_curves), axis=0)
        fig.add_trace(go.Scatter(
            y=median_curve,
            mode="lines",
            line=dict(color="white", width=2),
            name="Median Path"
        ))
        
        fig.update_layout(
            title=f"Monte Carlo Simulation ({num_simulations} runs)",
            xaxis_title="Trade Number",
            yaxis_title="Account Equity ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=all_max_drawdowns,
            nbinsx=20,
            marker_color="red",
            opacity=0.7,
            name="Drawdown Distribution"
        ))
        
        median_dd = np.median(all_max_drawdowns)
        pct_95_dd = np.percentile(all_max_drawdowns, 95)
        
        fig.add_vline(
            x=median_dd,
            line_width=2,
            line_dash="dash",
            line_color="white",
            annotation_text="Median"
        )
        
        fig.add_vline(
            x=pct_95_dd,
            line_width=2,
            line_dash="dash",
            line_color="yellow",
            annotation_text="95th percentile"
        )
        
        fig.update_layout(
            title="Maximum Drawdown Distribution",
            xaxis_title="Maximum Drawdown ($)",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Median Max Drawdown", f"${median_dd:.2f}")
        with col2:
            st.metric("95th Percentile Max Drawdown", f"${pct_95_dd:.2f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# MAIN DASHBOARD LAYOUT
# =======================
def main():
    """Main dashboard function."""
    # Initial greeting
    if "mariah_greeted" not in st.session_state:
        mariah_speak("System online. Welcome to the Crypto Capital.")
        st.session_state["mariah_greeted"] = True
    
    # Load images
    logo_base64 = get_base64_image("IMG_7006.PNG")
    brain_base64 = get_base64_image("updatedbrain1.png")
    
    # Apply background image
    set_dashboard_background("Screenshot 2025.png")
    
    # Header layout
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
    
    # Load and display Mariah's avatar
    with open("ChatGPT Image May 4, 2025, 07_29_01 PM.png", "rb") as img_file:
        mariah_base64 = base64.b64encode(img_file.read()).decode()
    
    with header_col2:
        st.markdown(f"""
        <img src="data:image/png;base64,{mariah_base64}"
            class="mariah-avatar"
            width="130"
            style="margin-top: 0.5rem; border-radius: 12px;" />
        """, unsafe_allow_html=True)
    
    # Sidebar layout
    with st.sidebar:
        # AI Banner
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 1rem 0 0.5rem 0.5rem;">
            <span style="color: #00fff5; font-size: 1.1rem; font-weight: 600;">Powered by AI</span>
            <img src="data:image/png;base64,{brain_base64}" width="26" class="pulse-brain" />
        </div>
        """, unsafe_allow_html=True)
        
        # More Tools Panel
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.markdown("### 📂 More Tools")
        more_tab = st.selectbox(
            "Select a Tool",
            [
                "📆 Daily PnL",
                "📈 Performance Trends",
                "📊 Advanced Analytics", # New option
                "📆 Filter by Date",
                "📰 Crypto News",
                "📡 On-Chain Data",
                "📡 Signal Scanner"
            ],
            key="sidebar_more_tools_dropdown"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dashboard Controls
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.title("📊 Dashboard Controls")
        
        # Voice Settings
        st.markdown("### 🎤 Voice Settings")
        st.session_state["mute_mariah"] = st.sidebar.checkbox(
            "🔇 Mute Mariah's Voice",
            value=st.session_state.get("mute_mariah", False)
        )
        
        # Auto-Refresh
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
        
        # Strategy Mode
        st.markdown("### ⚙️ Strategy Mode")
        mode = st.radio(
            "Choose a Strategy Mode:",
            ["Scalping", "Swing", "Momentum"],
            index=1,
            key="strategy_mode_selector"
        )
        
        # Position Sizing
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
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Controls
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
    
    # Load data
    sl_log = st.empty()  # Log area for SL updates
    df_open_positions = load_open_positions()
    trailing_stop_loss(log_slot=sl_log)  # Update trailing stops
    df_manual_closed = load_closed_manual_trades()
    df_trades = load_trades()
    df_bot_open, df_bot_closed = split_bot_trades(df_trades)
    
    # Ensure columns exist
    if "Realized PnL ($)" not in df_manual_closed.columns:
        df_manual_closed["Realized PnL ($)"] = 0
    if "Realized PnL ($)" not in df_bot_closed.columns:
        df_bot_closed["Realized PnL ($)"] = 0
    
    # Risk Banner
    if should_show_risk_banner(df_bot_closed, df_manual_closed):
        mariah_speak("Warning. Mariah is pausing trades due to risk limit.")
        st.markdown("""
        <div style="background-color: rgba(255, 0, 0, 0.15); padding: 1rem; border-left: 6px solid red; border-radius: 8px;">
            <h4 style="color: red;">🚨 BOT DISABLED: Daily Loss Limit Reached</h4>
            <p style="color: #ffcccc;">Mariah has paused all trading for today to protect your capital. Override is OFF. 🛡️</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Override Banner
    if st.session_state.get("override_risk_lock"):
        st.markdown("""
        <div class="override-glow" style="background-color: rgba(0, 255, 245, 0.15); padding: 1rem;
        border-left: 6px solid #00fff5; border-radius: 8px;">
            <h4 style="color: #00fff5;">✅ Override Active</h4>
            <p style="color: #ccffff;">Mariah is trading today even though the risk lock was triggered. Use with caution. 😈</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Speak override message once
        if "override_voice_done" not in st.session_state:
            mariah_speak("Override active. Proceeding with caution.")
            st.session_state["override_voice_done"] = True
    
    # No trades banner
    if df_bot_closed.empty and df_manual_closed.empty:
        st.warning("📭 No trades recorded today. Your bot or manual log may be empty.")
    
    # Log daily stats
    log_daily_pnl_split(df_bot_closed, df_manual_closed)
    
    # Calculate PnL
    open_pnl = df_open_positions["PnL ($)"].sum() if not df_open_positions.empty else 0
    closed_pnl = df_bot_closed["Realized PnL ($)"].sum() + df_manual_closed["Realized PnL ($)"].sum()
    
    # Global PnL Summary
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
    
    # Main tabs
    main_tabs = [
        "🌐 All Trades",
        "📈 Bot Open Trades",
        "✅ Bot Closed Trades",
        "🔥 Manual Open Trades",
        "✅ Manual Closed Trades",
        "📊 Growth Curve",
        "🛒 Place Trade",
        "🧠 Mariah AI"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(main_tabs)
    
    # Tab 1: All Trades
    with tab1:
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("🌐 All Trades Summary")
        
        # Bot Closed Trades
        if df_bot_closed.empty:
            st.info("No bot closed trades yet.")
        else:
            df_bot_closed_display = df_bot_closed.copy()
            df_bot_closed_display["timestamp"] = df_bot_closed_display.get("timestamp", "")
            df_bot_closed_display["note"] = df_bot_closed_display.get("note", "")
            
            st.subheader("✅ Bot Closed Trades")
            st.dataframe(df_bot_closed_display[[
                "timestamp", "symbol", "side", "qty", 
                "entry_price", "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
        
        # Manual Closed Trades
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
                "timestamp", "symbol", "side", "qty",
                "entry_price", "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
        
        # Manual Open Trades
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
                    "timestamp", "symbol", "side", "qty",
                    "entry_price", "stop_loss", "take_profit", "note",
                    "Realized PnL ($)", "Realized PnL (%)"
                ]])
            else:
                st.info("No open manual trades found.")
                
        except Exception as e:
            st.error(f"❌ Error loading manual open trades: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Bot Open Trades
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
                "timestamp", "symbol", "side", "qty",
                "entry_price", "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Bot Closed Trades
    with tab3:
        st.subheader("✅ Bot Closed Trades")
        
        if df_bot_closed.empty:
            st.info("No closed bot trades yet.")
        else:
            df_bot_closed_display = df_bot_closed.copy()
            df_bot_closed_display["timestamp"] = df_bot_closed_display.get("timestamp", "")
            df_bot_closed_display["note"] = df_bot_closed_display.get("note", "")
            
            st.dataframe(df_bot_closed_display[[
                "timestamp", "symbol", "side", "qty",
                "entry_price", "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
    
    # Tab 4: Manual Open Trades
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
                    "timestamp", "symbol", "side", "qty",
                    "entry_price", "stop_loss", "take_profit", "note",
                    "Realized PnL ($)", "Realized PnL (%)"
                ]])
                
        except Exception as e:
            st.error(f"❌ Failed to fetch open manual trades: {e}")
    
    # Tab 5: Manual Closed Trades
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
                "timestamp", "symbol", "side", "qty",
                "entry_price", "stop_loss", "take_profit", "note",
                "Realized PnL ($)", "Realized PnL (%)"
            ]])
    
    # Tab 6: Growth Curve
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
                
                # Trade Performance Metrics
                st.markdown("---")
                st.subheader("📌 Bot Trade Performance Summary")
                
                total_trades = len(df_closed)
                wins = df_closed[df_closed["Realized PnL ($)"] > 0]
                losses = df_closed[df_closed["Realized PnL ($)"] < 0]
                win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
                avg_win = wins["Realized PnL ($)"].mean() if not wins.empty else 0
                avg_loss = losses["Realized PnL ($)"].mean() if not losses.empty else 0
                profit_factor = abs(wins["Realized PnL ($)"].sum() / losses["Realized PnL ($)"].sum()) if not losses.empty else float('inf')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Total Bot Trades", total_trades)
                col2.metric("✅ Win Rate", f"{win_rate:.2f}%")
                col3.metric("⚖️ Profit Factor", f"{profit_factor:.2f}")
                
                col4, col5 = st.columns(2)
                col4.metric("🟢 Avg Win ($)", f"${avg_win:.2f}")
                col5.metric("🔻 Avg Loss ($)", f"${avg_loss:.2f}")
    
    # Tab 7: Place Trade
    with tab7:
        st.subheader("🛒 Place Live Trade")
        
        # Dropdowns for symbol and side
        symbol = st.selectbox(
            "Symbol", 
            ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"]
        )
        
        side = st.selectbox(
            "Side", 
            ["Buy", "Sell"]
        )
        
        # Quantity from sidebar position sizing (but editable)
        qty = st.number_input(
            "Quantity (auto-filled from sidebar)",
            min_value=0.001,
            value=max(0.001, float(qty_calc)),
            step=0.001,
            format="%.3f"
        )
        
        # Place market order
        if st.button("🚀 Place Market Order"):
            try:
                # Execute the order
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
                
                # Mariah confirms trade
                mariah_speak(f"Order executed. {side} {qty} {symbol}.")
                
                # Speak if override is active
                if st.session_state.get("override_risk_lock"):
                    mariah_speak("Override active. Proceeding with caution.")
                
                # Log the trade
                log_rsi_trade_to_csv(
                    symbol=symbol,
                    side=side,
                    qty=round(qty, 3),
                    entry_price=entry_price_sidebar,
                    mode=mode
                )
                
                st.success(f"✅ Order placed: {side} {qty} {symbol}")
                st.write("Order Response:", order)
                
            except Exception as e:
                mariah_speak("Order failed. Check trade parameters.")
                st.error(f"❌ Order failed: {e}")
    
    # Tab 8: Mariah AI
    with tab8:
        st.subheader("🧠 Talk to Mariah")
        
        # Mode Colors + Dynamic Styling
        mode_colors = {
            "Scalping": "#00ffcc",  # Aqua Green
            "Swing": "#ffaa00",     # Orange
            "Momentum": "#ff4d4d"   # Red
        }
        
        # Display Strategy Mode with Color
        st.markdown(
            f"<span style='font-size: 1.1rem; font-weight: 600;'>🚦 Current Strategy Mode: "
            f"<span style='color: {mode_colors[mode]};'>{mode}</span></span>",
            unsafe_allow_html=True
        )
        
        # Text input
        user_input = st.chat_input("Ask Mariah anything...", key="mariah_chat_input")
        
        if user_input:
            st.chat_message("user").markdown(user_input)
            override_on = st.session_state.get("override_risk_lock", False)
            response = get_mariah_reply(user_input, open_pnl, closed_pnl, override_on)
            st.chat_message("assistant").markdown(response)
            mariah_speak(response)
        
        st.markdown("---")
        st.markdown("🎙 Or press below to speak:")
        
        # Voice input (via mic)
        if st.button("🎙 Speak to Mariah"):
            voice_input = listen_to_user()
            
            if voice_input:
                st.chat_message("user").markdown(voice_input)
                override_on = st.session_state.get("override_risk_lock", False)
                response = get_mariah_reply(voice_input, open_pnl, closed_pnl, override_on)
                st.chat_message("assistant").markdown(response)
                mariah_speak(response)
    
    # Additional tabs based on sidebar selection
    if more_tab == "📡 Signal Scanner":
        render_signal_scanner(mode, account_balance, df_bot_closed, df_manual_closed)
    elif more_tab == "📆 Daily PnL":
        render_daily_pnl()
    elif more_tab == "📈 Performance Trends":
        render_performance_trends()
    elif more_tab == "📊 Advanced Analytics": # New tab
        # Load data for advanced analytics
        df_daily_pnl = pd.read_csv(DAILY_PNL_SPLIT_FILE) if os.path.exists(DAILY_PNL_SPLIT_FILE) else pd.DataFrame()
        render_advanced_analytics(df_trades, df_daily_pnl)
    elif more_tab == "📆 Filter by Date":
        render_filter_by_date()
    elif more_tab == "📰 Crypto News":
        render_crypto_news()
    elif more_tab == "📡 On-Chain Data":
        render_onchain_data()

# =======================
# MORE TOOLS TAB FUNCTIONS
# =======================
def render_signal_scanner(mode, account_balance, df_bot_closed, df_manual_closed):
    """Render the Signal Scanner tab with ML enhancement."""
    with st.container():
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📡 Signal Scanner")
        
        # Add ML option
        signal_type = st.radio(
            "Signal Type",
            ["RSI Only", "ML Enhanced"],
            index=0
        )
        
        # Risk & Trade Settings
        interval = st.selectbox(
            "Candle Interval", 
            ["5", "15", "30", "60", "240"], 
            index=1
        )
        
        symbols = st.multiselect(
            "Symbols to Scan", 
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"], 
            default=["BTCUSDT", "ETHUSDT"]
        )
        
        st.markdown("---")
        
        # Train ML model option
        if signal_type == "ML Enhanced":
            if st.button("🧠 Train ML Model"):
                with st.spinner("Training model..."):
                    for symbol in symbols:
                        historical_data = get_historical_data(symbol, interval, limit=500)
                        if not historical_data.empty:
                            ml_generator = MLSignalGenerator(model_path=f"models/{symbol}_{interval}_model.pkl")
                            accuracy = ml_generator.train_model(historical_data)
                            if accuracy:
                                st.success(f"✓ Model trained for {symbol} with accuracy: {accuracy:.2f}")
                            else:
                                st.error(f"Failed to train model for {symbol}")
        
        for symbol in symbols:
            # Daily loss guard
            if check_max_daily_loss(df_bot_closed, df_manual_closed):
                st.warning(f"🛑 Mariah skipped {symbol} — Daily loss limit reached.")
                continue
            
            try:
                # Traditional RSI signal
                rsi_value, rsi_trigger = check_rsi_signal(symbol=symbol, interval=interval, mode=mode)
                
                # Default to RSI-only trigger
                trigger = rsi_trigger
                
                # Add ML signal if selected
                if signal_type == "ML Enhanced":
                    # Get historical data for ML
                    historical_data = get_historical_data(symbol, interval, limit=100)
                    
                    if not historical_data.empty:
                        # Initialize ML signal generator
                        ml_generator = MLSignalGenerator(model_path=f"models/{symbol}_{interval}_model.pkl")
                        
                        # Get ML signal
                        ml_signal, confidence = ml_generator.get_signal(historical_data)
                        
                        # Show both signals
                        st.write(f"✅ {symbol} RSI: {rsi_value:.2f}, ML Signal: {ml_signal} (Confidence: {confidence:.2f})")
                        
                        # Use combined signal logic
                        trigger = rsi_trigger and (ml_signal == "buy") and (confidence > 0.65)
                        
                        # Show feature importance
                        if st.checkbox(f"Show ML feature importance for {symbol}", key=f"feat_imp_{symbol}"):
                            features = ml_generator.get_feature_importance()
                            if features:
                                st.write("Feature importance:")
                                features_df = pd.DataFrame({
                                    'Feature': list(features.keys()),
                                    'Importance': list(features.values())
                                }).sort_values('Importance', ascending=False)
                                st.dataframe(features_df)
                
                if trigger:
                    # Use mode-specific SL/TP from strategy_mode.py
                    sl_pct, tp_pct, rsi_overbought, rsi_oversold = get_strategy_params(mode)
                    entry_price = float(historical_data["close"].iloc[-1]) if 'historical_data' in locals() else rsi_value
                    stop_loss = entry_price * (1 - sl_pct / 100)
                    take_profit = entry_price * (1 + tp_pct / 100)
                    
                    # Get risk percent from strategy function
                    risk_percent = risk_percent  # Using the sidebar value
                    
                    qty = position_size_from_risk(account_balance, risk_percent, entry_price, stop_loss)
                    
                    # Mariah speaks before placing the trade
                    if signal_type == "ML Enhanced":
                        mariah_speak(
                            f"ML enhanced signal detected. Entering a {mode.lower()} trade on {symbol}. "
                            f"Stop loss at {sl_pct} percent. Take profit at {tp_pct} percent."
                        )
                    else:
                        mariah_speak(
                            f"Entering a {mode.lower()} trade on {symbol}. "
                            f"Stop loss at {sl_pct} percent. Take profit at {tp_pct} percent. "
                            f"Risking {risk_percent} percent of capital."
                        )
                    
                    # Place order button
                    if st.button(f"🚀 Execute Trade for {symbol}", key=f"exec_{symbol}"):
                        # Place order
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
                        
                        # Log trade
                        log_rsi_trade_to_csv(
                            symbol=symbol,
                            side="Buy",
                            qty=qty,
                            entry_price=entry_price,
                            mode=mode
                        )
                        
                        st.success(f"✅ {signal_type} trade executed: {symbol} @ ${entry_price:.2f}")
                else:
                    if signal_type == "RSI Only":
                        st.info(f"No RSI signal for {symbol}.")
                    elif signal_type == "ML Enhanced":
                        st.info(f"No combined signal for {symbol}.")
                    
            except Exception as e:
                st.error(f"❌ Error for {symbol}: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_daily_pnl():
    """Render the Daily PnL tab."""
    with st.container():
        st.subheader("📆 Daily PnL Breakdown (Bot vs Manual + Total + Trade Counts)")
        
        pnl_file = DAILY_PNL_SPLIT_FILE
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
                lambda row: row["bot_trades"] if row["Type"] == "🤖 Bot PnL" else row["manual_trades"],
                axis=1
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
            
            # Display daily summary table
            st.markdown("### 🧾 Daily Summary Table")
            df_table = df_pnl[[
                "date", "bot_pnl", "manual_pnl", "total_pnl", "bot_trades", "manual_trades"
            ]].copy()
            
            df_table.columns = [
                "Date",
                "🤖 Bot PnL",
                "👤 Manual PnL",
                "📈 Total PnL",
                "📊 Bot Trades",
                "📊 Manual Trades"
            ]
            
            st.dataframe(df_table.sort_values("Date", ascending=False), use_container_width=True)
            
            # Win/Loss Accuracy Table
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

def render_performance_trends():
    """Render the Performance Trends tab."""
    with st.container():
        st.subheader("📈 Performance Trends (Win Rate, PnL, Profit Factor, Trend Alerts)")
        
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

def render_filter_by_date():
    """Render the Filter by Date tab."""
    with st.container():
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
                    
                    st.plotly_chart(
                        go.Figure(
                            data=[go.Pie(labels=strategy_counts.index, values=strategy_counts.values, hole=0.3)],
                            layout_title_text="Trade Count by Strategy"
                        ), 
                        use_container_width=True
                    )
                    
                    st.plotly_chart(
                        go.Figure(
                            data=[go.Bar(x=strategy_pnl.index, y=strategy_pnl.values, marker_color="purple")],
                            layout_title_text="Total PnL by Strategy"
                        ), 
                        use_container_width=True
                    )
                
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
                    try:
                        import matplotlib.pyplot as plt
                        from fpdf import FPDF
                        
                        # Generate charts for PDF
                        fig1, ax1 = plt.subplots()
                        df_filtered.plot(x="timestamp", y="Realized PnL ($)", kind="bar", ax=ax1, title="Daily PnL")
                        fig1_path = "daily_pnl_chart.png"
                        fig1.savefig(fig1_path)
                        plt.close(fig1)
                        
                        fig2, ax2 = plt.subplots()
                        df_filtered.plot(x="timestamp", y="Cumulative PnL", kind="line", ax=ax2, marker='o', title="Cumulative PnL")
                        fig2_path = "cumulative_pnl_chart.png"
                        fig2.savefig(fig2_path)
                        plt.close(fig2)
                        
                        # Get best and worst trades
                        best_trade = df_filtered.loc[df_filtered["Realized PnL ($)"].idxmax()]
                        worst_trade = df_filtered.loc[df_filtered["Realized PnL ($)"].idxmin()]
                        
                        # Get PnL by strategy
                        strategy_pnl = df_filtered.groupby("note")["Realized PnL ($)"].sum().sort_values(ascending=False) if "note" in df_filtered.columns else {}
                        
                        # Create PDF
                        pdf = FPDF()
                        pdf.add_page()
                        
                        # Add logo if exists
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
                        
                        if strategy_pnl is not None:
                            pdf.set_font("Arial", 'B', 12)
                            pdf.cell(0, 10, "📊 PnL by Strategy:", ln=True)
                            
                            pdf.set_font("Arial", '', 11)
                            for strategy, pnl in strategy_pnl.items():
                                pdf.cell(0, 10, f"{strategy}: ${pnl:.2f}", ln=True)
                        
                        pdf.ln(6)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "🏆 Best Trade:", ln=True)
                        
                        pdf.set_font("Arial", '', 11)
                        for k, v in best_trade.to_dict().items():
                            pdf.cell(0, 10, f"{k}: {v}", ln=True)
                        
                        pdf.ln(6)
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(0, 10, "💀 Worst Trade:", ln=True)
                        
                        pdf.set_font("Arial", '', 11)
                        for k, v in worst_trade.to_dict().items():
                            pdf.cell(0, 10, f"{k}: {v}", ln=True)
                        
                        pdf.ln(6)
                        pdf.image(fig1_path, w=170)
                        pdf.ln(6)
                        pdf.image(fig2_path, w=170)
                        
                        pdf_path = f"trade_report_{start_date}_to_{end_date}.pdf"
                        pdf.output(pdf_path)
                        
                        # Send email
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

def render_crypto_news():
    """Render the Crypto News tab."""
    with st.container():
        st.subheader("📰 Real-Time Crypto News")
        
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
                    alert = "📈"
                
                tags = ", ".join([c["code"] for c in item.get("tags", [])]) if item.get("tags") else ""
                
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                <a href="{item['url']}" target="_blank" style="color:#00fff5; font-weight:bold;">{alert}{item['title']}</a><br>
                <span style="color:gray; font-size: 0.85rem;">{item['published']} | {item['source']} | {tags}</span>
                </div>
                """, unsafe_allow_html=True)

def render_onchain_data():
    """Render the On-Chain Data tab."""
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

# Run the dashboard
if __name__ == "__main__":
    main()
