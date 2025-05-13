# Updated on May 12, pushing correct VS Code version

"""
Crypto Capital Dashboard - ENHANCED VERSION with Visual Improvements
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

st.set_page_config(page_title="The Crypto Capital", layout="wide")

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
# Initialize session state for tracking UI state
if "mariah_greeted" not in st.session_state:
    st.session_state["mariah_greeted"] = False

if "override_risk_lock" not in st.session_state:
    st.session_state["override_risk_lock"] = False

if "test_mode" not in st.session_state:
    st.session_state["test_mode"] = False

if "mute_mariah" not in st.session_state:
    st.session_state["mute_mariah"] = False

if "current_tool" not in st.session_state:
    st.session_state.current_tool = None

if "active_tab_index" not in st.session_state:
    st.session_state.active_tab_index = 0

# Initialize Bybit API connection
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    st.error("❌ Missing API keys! Please check your .env file.")
    st.stop()

try:
    session = HTTP(
        api_key=API_KEY,
        api_secret=API_SECRET,
        recv_window=30000  # Increase timeout window (ms)
    )
except Exception as e:
    st.error(f"❌ Failed to initialize Bybit session: {e}")
    st.stop()

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
# ENHANCED MARIAH LEVEL 2
# =======================
class MariahLevel2:
    """Enhanced signal analyzer for Mariah using multiple indicators"""
    
    def __init__(self):
        self.last_analysis = {}
        
    def analyze_symbol(self, symbol, interval, session):
        """Enhanced analysis using multiple indicators"""
        try:
            # Get market data (using your existing session)
            res = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=100
            )["result"]["list"]
            
            # Convert to DataFrame
            df = pd.DataFrame(res, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            # Convert to numbers
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate all indicators
            signals = {
                "rsi": self._check_rsi(df),
                "macd": self._check_macd(df),
                "volume": self._check_volume(df),
                "moving_avg": self._check_moving_averages(df),
                "bollinger": self._check_bollinger_bands(df)
            }
            
            # Combine signals
            decision = self._combine_signals(signals)
            
            # Store result
            self.last_analysis[symbol] = {
                "signals": signals,
                "decision": decision,
                "timestamp": pd.Timestamp.now()
            }
            
            return decision
            
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}
    
    def _check_rsi(self, df):
        """Check RSI with dynamic levels"""
        rsi = ta.rsi(df['close'], length=14)
        current_rsi = rsi.iloc[-1]
        
        # Dynamic RSI levels based on recent volatility
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        
        # Adjust levels based on volatility
        if volatility > 0.03:  # High volatility
            overbought, oversold = 75, 25
        else:  # Normal volatility
            overbought, oversold = 70, 30
        
        if current_rsi > overbought:
            strength = (current_rsi - overbought) / (100 - overbought)
            return {"signal": "sell", "confidence": min(strength, 1.0), "value": current_rsi}
        elif current_rsi < oversold:
            strength = (oversold - current_rsi) / oversold
            return {"signal": "buy", "confidence": min(strength, 1.0), "value": current_rsi}
        else:
            return {"signal": "hold", "confidence": 0.0, "value": current_rsi}
    
    def _check_macd(self, df):
        """Check MACD crossovers and divergence"""
        macd = ta.macd(df['close'])
        
        macd_line = macd['MACD_12_26_9']
        signal_line = macd['MACDs_12_26_9']
        histogram = macd['MACDh_12_26_9']
        
        # Check for crossovers
        if len(macd_line) > 1:
            # Bullish crossover
            if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                # Stronger signal if above zero line
                confidence = 0.8 if macd_line.iloc[-1] > 0 else 0.6
                return {"signal": "buy", "confidence": confidence, "type": "crossover"}
            
            # Bearish crossover
            elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                confidence = 0.8 if macd_line.iloc[-1] < 0 else 0.6
                return {"signal": "sell", "confidence": confidence, "type": "crossover"}
            
            # Check histogram momentum
            elif len(histogram) > 3:
                recent_histogram = histogram.tail(3)
                if recent_histogram.iloc[-1] > recent_histogram.iloc[-2] > recent_histogram.iloc[-3]:
                    return {"signal": "buy", "confidence": 0.4, "type": "momentum"}
                elif recent_histogram.iloc[-1] < recent_histogram.iloc[-2] < recent_histogram.iloc[-3]:
                    return {"signal": "sell", "confidence": 0.4, "type": "momentum"}
        
        return {"signal": "hold", "confidence": 0.0, "type": "none"}
    
    def _check_volume(self, df):
        """Check volume patterns"""
        volume = df['volume']
        price = df['close']
        
        # Volume moving average
        volume_ma = volume.rolling(20).mean()
        
        # Current volume surge
        volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1]
        
        # Price change
        price_change_pct = (price.iloc[-1] - price.iloc[-2]) / price.iloc[-2] * 100
        
        # Strong volume with price movement
        if volume_ratio > 1.8 and abs(price_change_pct) > 0.5:
            confidence = min(volume_ratio / 3, 1.0)
            signal = "buy" if price_change_pct > 0 else "sell"
            return {"signal": signal, "confidence": confidence, "ratio": volume_ratio}
        
        # Unusual volume without much price movement (accumulation/distribution)
        elif volume_ratio > 1.5 and abs(price_change_pct) < 0.2:
            return {"signal": "hold", "confidence": 0.3, "ratio": volume_ratio, "note": "accumulation"}
        
        return {"signal": "hold", "confidence": 0.0, "ratio": volume_ratio}
    
    def _check_moving_averages(self, df):
        """Check multiple moving average alignment"""
        close = df['close']
        
        # Multiple EMAs
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        
        current_price = close.iloc[-1]
        current_ema8 = ema_8.iloc[-1]
        current_ema21 = ema_21.iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        
        # Bullish alignment: Price > EMA8 > EMA21 > EMA50
        if current_price > current_ema8 > current_ema21 > current_ema50:
            # Calculate strength of trend
            strength = ((current_price - current_ema50) / current_ema50) * 100
            confidence = min(abs(strength) / 5, 1.0)  # Scale 0-5% to 0-1
            return {"signal": "buy", "confidence": confidence, "alignment": "bullish"}
        
        # Bearish alignment: Price < EMA8 < EMA21 < EMA50
        elif current_price < current_ema8 < current_ema21 < current_ema50:
            strength = ((current_ema50 - current_price) / current_ema50) * 100
            confidence = min(abs(strength) / 5, 1.0)
            return {"signal": "sell", "confidence": confidence, "alignment": "bearish"}
        
        # Mixed signals
        return {"signal": "hold", "confidence": 0.0, "alignment": "mixed"}
    
    def _check_bollinger_bands(self, df):
        """Check Bollinger Band signals"""
        bbands = ta.bbands(df['close'], length=20, std=2)
        
        current_price = df['close'].iloc[-1]
        upper_band = bbands['BBU_20_2.0'].iloc[-1]
        lower_band = bbands['BBL_20_2.0'].iloc[-1]
        middle_band = bbands['BBM_20_2.0'].iloc[-1]
        
        # Calculate position within bands
        band_position = (current_price - lower_band) / (upper_band - lower_band)
        
        # Near upper band (overbought)
        if band_position > 0.95:
            return {"signal": "sell", "confidence": 0.6, "position": band_position}
        
        # Near lower band (oversold)
        elif band_position < 0.05:
            return {"signal": "buy", "confidence": 0.6, "position": band_position}
        
        # Bollinger squeeze (low volatility)
        band_width = (upper_band - lower_band) / middle_band
        avg_band_width = bbands['BBU_20_2.0'].rolling(10).mean().iloc[-1] - bbands['BBL_20_2.0'].rolling(10).mean().iloc[-1]
        avg_band_width /= bbands['BBM_20_2.0'].rolling(10).mean().iloc[-1]
        
        if band_width < avg_band_width * 0.7:
            return {"signal": "hold", "confidence": 0.3, "position": band_position, "note": "squeeze"}
        
        return {"signal": "hold", "confidence": 0.0, "position": band_position}
    
    def _combine_signals(self, signals):
        """Combine all signals with weighted scoring"""
        # Weight each signal type
        weights = {
            "rsi": 1.0,
            "macd": 1.2,  # MACD gets slightly higher weight
            "volume": 0.8,
            "moving_avg": 1.1,
            "bollinger": 0.7
        }
        
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        signal_details = {}
        
        for signal_type, signal_data in signals.items():
            if signal_data["signal"] != "hold":
                weight = weights[signal_type]
                confidence = signal_data["confidence"]
                weighted_score = weight * confidence
                
                if signal_data["signal"] == "buy":
                    buy_score += weighted_score
                elif signal_data["signal"] == "sell":
                    sell_score += weighted_score
                
                total_weight += weight
                signal_details[signal_type] = signal_data
        
        # Calculate final scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final signal
        min_confidence = 0.4  # Minimum confidence to act
        
        if buy_score > sell_score and buy_score > min_confidence:
            return {
                "action": "buy",
                "confidence": buy_score,
                "score_difference": buy_score - sell_score,
                "signals": signal_details,
                "summary": f"{len([s for s in signals.values() if s['signal'] == 'buy'])} buy signals"
            }
        elif sell_score > buy_score and sell_score > min_confidence:
            return {
                "action": "sell",
                "confidence": sell_score,
                "score_difference": sell_score - buy_score,
                "signals": signal_details,
                "summary": f"{len([s for s in signals.values() if s['signal'] == 'sell'])} sell signals"
            }
        else:
            return {
                "action": "hold",
                "confidence": 0.5,
                "score_difference": abs(buy_score - sell_score),
                "signals": signal_details,
                "summary": "Mixed or weak signals"
            }
    
    def get_detailed_analysis(self, symbol):
        """Get detailed breakdown of last analysis"""
        if symbol in self.last_analysis:
            return self.last_analysis[symbol]
        return None

# =======================
# UTILITY FUNCTIONS
# =======================
def create_crypto_ticker(session):
    """Create a seamless crypto ticker with no gaps"""
    
    # Initialize toggle state
    if 'hide_header' not in st.session_state:
        st.session_state.hide_header = False
    
    # Add CSS for seamless scrolling
    st.markdown(f"""
    <style>
    .crypto-ticker {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        width: 100%;
        background: linear-gradient(90deg, rgba(15,15,35,0.95) 0%, rgba(30,20,60,0.95) 100%);
        border-bottom: 1px solid #00fff5;
        padding: 8px 0;
        overflow: hidden;
        white-space: nowrap;
        z-index: 9999999;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .ticker-scroll {{
        display: flex;
        align-items: center;
        animation: scrollContinuous 60s linear infinite;
        /* Duplicate content for seamless loop */
        width: max-content;
    }}
    
    .ticker-content {{
        display: flex;
        align-items: center;
        white-space: nowrap;
    }}
    
    @keyframes scrollContinuous {{
        0% {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    
    .crypto-item {{
        display: inline-block;
        margin: 0 20px;
        padding: 5px 15px;
        background: rgba(255,255,255,0.05);
        border-radius: 5px;
        font-size: 14px;
        color: white;
        white-space: nowrap;
    }}
    
    .positive {{ color: #00d87f; }}
    .negative {{ color: #ff4d4d; }}
    
    /* Streamlit header styling */
    header[data-testid="stHeader"] {{
        {'display: none !important;' if st.session_state.hide_header else 'display: block !important;'}
        top: 40px !important;
        background: rgba(20, 16, 50, 0.95) !important;
        border-bottom: 3px solid #00fff5 !important;
        box-shadow: 0 0 20px #00fff5, 0 2px 10px rgba(0, 255, 245, 0.5) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* Enhanced green glow effect */
    header[data-testid="stHeader"]::after {{
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #00fff5, transparent);
        animation: glow 2s ease-in-out infinite alternate;
    }}
    
    @keyframes glow {{
        0% {{ box-shadow: 0 0 10px #00fff5; }}
        100% {{ box-shadow: 0 0 30px #00fff5, 0 0 40px #00fff5; }}
    }}
    
    /* Header button styling */
    [data-testid="stHeader"] button {{
        color: white !important;
        opacity: 0.8;
        transition: opacity 0.3s ease;
    }}
    
    [data-testid="stHeader"] button:hover {{
        opacity: 1;
        filter: drop-shadow(0 0 10px #00fff5);
    }}
    
    /* Main content positioning */
    .main .block-container {{
        padding-top: {80 if not st.session_state.hide_header else 40}px !important;
    }}
    
    /* Sidebar positioning */
    [data-testid="stSidebar"] {{
        top: {40 if not st.session_state.hide_header else 0}px !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Expanded to top 20 crypto symbols
    crypto_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
        "ADAUSDT", "MATICUSDT", "BNBUSDT", "LTCUSDT", "UNIUSDT",
        "XLMUSDT", "DOTUSDT", "DYDXUSDT", "NEARUSDT", "FILUSDT",
        "RNDRUSDT", "ETCUSDT", "BCHUSDT", "LDOUSDT", "GALAUSDT"
    ]
    
    ticker_items = []
    
    for symbol in crypto_symbols:
        try:
            response = session.get_tickers(category="linear", symbol=symbol)
            if response.get("retCode") == 0:
                data = response["result"]["list"][0]
                price = float(data["lastPrice"])
                change_pct = float(data["price24hPcnt"]) * 100
                
                color_class = "positive" if change_pct >= 0 else "negative"
                sign = "+" if change_pct >= 0 else ""
                
                # Create clean HTML string
                symbol_name = symbol.replace('USDT', '')
                price_formatted = f"{price:,.2f}"
                change_formatted = f"{sign}{change_pct:.2f}"
                
                item = f"""<span class="crypto-item">
                    <strong>{symbol_name}</strong> 
                    ${price_formatted} 
                    <span class="{color_class}">{change_formatted}%</span>
                </span>"""
                
                ticker_items.append(item)
        except Exception as e:
            continue
    
    # Render ticker only if we have items
    if ticker_items:
        # Create seamless content by duplicating items
        all_items = ''.join(ticker_items)
        
        ticker_html = f"""
        <div class="crypto-ticker">
            <div class="ticker-scroll">
                <div class="ticker-content">
                    {all_items}
                </div>
                <div class="ticker-content">
                    {all_items}
                </div>
            </div>
        </div>
        """
        
        # Render the complete ticker
        st.markdown(ticker_html, unsafe_allow_html=True)

def get_base64_image(path):
    """Convert an image to base64 encoding."""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"❌ Failed to load image {path}: {e}")
        return ""

def set_dashboard_background(image_file):
    """Set the dashboard background with a cinematic glow effect."""
    try:
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
            margin: 1.5rem 0 2.8rem 0.5rem;
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
        
        /* Custom CSS for radio buttons */
        div[data-testid="stRadio"] > div {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        div[data-testid="stRadio"] label {{
            background-color: rgba(255, 255, 255, 0.05);
            padding: 10px 15px;
            border-radius: 8px;
            border: 1px solid rgba(0, 255, 245, 0.2);
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        div[data-testid="stRadio"] label:hover {{
            background-color: rgba(0, 255, 245, 0.1);
            border-color: rgba(0, 255, 245, 0.4);
        }}
        div[data-testid="stRadio"] label input[type="radio"]:checked + span {{
            background-color: rgba(0, 255, 245, 0.2);
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Failed to set background: {e}")

# =======================
# RISK MANAGEMENT FUNCTIONS
# =======================
def check_max_daily_loss(df_bot_closed, df_manual_closed):
    """Check if daily loss limit has been reached."""
    if st.session_state.get("override_risk_lock"):
        return False
    
    today = date.today()
    
    # Safely handle empty DataFrames and ensure timestamp column exists
    df_bot_safe = df_bot_closed.copy() if not df_bot_closed.empty else pd.DataFrame(columns=["timestamp", "Realized PnL ($)"])
    df_manual_safe = df_manual_closed.copy() if not df_manual_closed.empty else pd.DataFrame(columns=["timestamp", "Realized PnL ($)"])
    
    # Ensure timestamp columns exist and are properly formatted
    for df, name in [(df_bot_safe, "bot"), (df_manual_safe, "manual")]:
        if "timestamp" not in df.columns and not df.empty:
            df["timestamp"] = pd.Timestamp.now()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    df_all = pd.concat([df_bot_safe, df_manual_safe])
    df_today = df_all[df_all["timestamp"].dt.date == today] if not df_all.empty else pd.DataFrame()
    
    pnl_today = df_today["Realized PnL ($)"].sum() if not df_today.empty else 0
    
    if pnl_today <= MAX_DAILY_LOSS:
        log_risk_lock_event(today, pnl_today)
        return True
    return False

def log_risk_lock_event(today, pnl_today):
    """Log daily shutdown to risk_events.csv."""
    try:
        if os.path.exists(RISK_LOG_PATH):
            df_log = pd.read_csv(RISK_LOG_PATH)
            if today in pd.to_datetime(df_log["date"]).dt.date.values:
                return  # Already logged today
        else:
            df_log = pd.DataFrame(columns=["date", "triggered_at_pnl"])
        
        new_row = pd.DataFrame([{"date": today, "triggered_at_pnl": pnl_today}])
        df_log = pd.concat([df_log, new_row], ignore_index=True)
        df_log.to_csv(RISK_LOG_PATH, index=False)
    except Exception as e:
        st.error(f"❌ Failed to log risk event: {e}")

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

def check_rsi_signal(session, symbol="BTCUSDT", interval="15", mode="Swing"):
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
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
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
    
    try:
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=trade_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
    except Exception as e:
        st.error(f"❌ Failed to log trade: {e}")

def get_historical_data(session, symbol, interval, limit=100):
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
        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["high"] = pd.to_numeric(df["high"], errors='coerce')
        df["low"] = pd.to_numeric(df["low"], errors='coerce')
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit='ms')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to get historical data for {symbol}: {e}")
        return pd.DataFrame()

# =======================
# TRADE DATA FUNCTIONS
# =======================
def load_open_positions(session):
    """Load open positions from Bybit API."""
    try:
        res = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )
        
        # Check for API errors
        if res.get("retCode") != 0:
            st.error(f"❌ API error: {res.get('retMsg', 'Unknown error')}")
            return pd.DataFrame()
        
        data = res["result"]["list"]
        
        def parse_float(value):
            try:
                return float(value)
            except (ValueError, TypeError):
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
    try:
        return pd.read_csv("trades.csv") if os.path.exists("trades.csv") else pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Failed to load trades: {e}")
        return pd.DataFrame()

def load_closed_manual_trades(session):
    """Load closed manual trades from Bybit API."""
    try:
        res = session.get_closed_pnl(category="linear", limit=50)
        
        # Check for API errors
        if res.get("retCode") != 0:
            st.error(f"❌ API error: {res.get('retMsg', 'Unknown error')}")
            return pd.DataFrame()
        
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
    
    # Ensure required columns exist
    if 'take_profit' not in df.columns:
        df['take_profit'] = 0
    
    df_open = df[df['take_profit'] == 0].copy()
    df_closed = df[df['take_profit'] != 0].copy()
    
    # Add 'note' column if it's missing
    for df_subset in [df_open, df_closed]:
        if "note" not in df_subset.columns:
            df_subset["note"] = ""
    
    # Safe calculation of Realized PnL for closed trades
    if not df_closed.empty:
        # Ensure required columns exist and have numeric values
        for col in ['entry_price', 'take_profit', 'qty']:
            if col not in df_closed.columns:
                df_closed[col] = 0
            df_closed[col] = pd.to_numeric(df_closed[col], errors='coerce').fillna(0)
        
        # Fee-adjusted PnL ($)
        df_closed["Realized PnL ($)"] = (
            (df_closed["take_profit"] - df_closed["entry_price"]) * df_closed["qty"]
            - (df_closed["entry_price"] + df_closed["take_profit"]) * df_closed["qty"] * FEE_RATE
        )
        
        # PnL (%) - avoid division by zero
        mask = df_closed["entry_price"] != 0
        df_closed["Realized PnL (%)"] = 0
        df_closed.loc[mask, "Realized PnL (%)"] = (
            ((df_closed.loc[mask, "take_profit"] - df_closed.loc[mask, "entry_price"]) 
             / df_closed.loc[mask, "entry_price"]) * 100
        )
    
    return df_open, df_closed

def log_daily_pnl_split(df_bot_closed, df_manual_closed, file_path=DAILY_PNL_SPLIT_FILE):
    """Log daily PnL statistics with split between bot and manual trades."""
    today = pd.Timestamp.now().date()
    
    def calc_stats(df):
        if "timestamp" not in df.columns or df.empty:
            return 0.0, 0, 0, 0
        
        # Ensure timestamp is datetime
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        df_today = df[df["timestamp"].dt.date == today]
        
        if df_today.empty:
            return 0.0, 0, 0, 0
        
        # Ensure PnL column exists and is numeric
        if "Realized PnL ($)" not in df_today.columns:
            df_today["Realized PnL ($)"] = 0
        df_today["Realized PnL ($)"] = pd.to_numeric(df_today["Realized PnL ($)"], errors='coerce').fillna(0)
        
        pnl_sum = df_today["Realized PnL ($)"].sum()
        count = len(df_today)
        wins = len(df_today[df_today["Realized PnL ($)"] > 0])
        losses = len(df_today[df_today["Realized PnL ($)"] < 0])
        
        return pnl_sum, count, wins, losses
    
    try:
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
        if not df_log.empty and today in pd.to_datetime(df_log["date"], errors='coerce').dt.date.values:
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
    except Exception as e:
        st.error(f"❌ Failed to log daily PnL: {e}")

def position_size_from_risk(account_balance, risk_percent, entry_price, stop_loss_price):
    """Calculate position size based on risk percentage."""
    if not all([account_balance, risk_percent, entry_price, stop_loss_price]):
        return 0
    
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
    
    try:
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        
        # Win rate drop (Bot)
        if ("bot_win_rate_7d" in df.columns and 
            not pd.isna(today["bot_win_rate_7d"]) and 
            not pd.isna(yesterday["bot_win_rate_7d"])):
            drop = yesterday["bot_win_rate_7d"] - today["bot_win_rate_7d"]
            if drop >= 20:
                alerts.append(f"⚠️ Bot Win Rate dropped by {drop:.1f}% compared to 7-day trend.")
        
        # Win rate drop (Manual)
        if ("manual_win_rate_7d" in df.columns and 
            not pd.isna(today["manual_win_rate_7d"]) and 
            not pd.isna(yesterday["manual_win_rate_7d"])):
            drop = yesterday["manual_win_rate_7d"] - today["manual_win_rate_7d"]
            if drop >= 20:
                alerts.append(f"⚠️ Manual Win Rate dropped by {drop:.1f}% compared to 7-day trend.")
        
        # PnL reversal detection
        if ("bot_pnl_7d" in df.columns and 
            yesterday["bot_pnl_7d"] > 0 and today["bot_pnl_7d"] < 0):
            alerts.append("🔻 Bot 7-Day Avg PnL turned negative.")
        
        if ("manual_pnl_7d" in df.columns and 
            yesterday["manual_pnl_7d"] > 0 and today["manual_pnl_7d"] < 0):
            alerts.append("🔻 Manual 7-Day Avg PnL turned negative.")
    except Exception as e:
        st.error(f"❌ Error checking trend alerts: {e}")
    
    return alerts

def trailing_stop_loss(session, threshold_pct=0.01, buffer_pct=0.015, log_slot=None):
    """Update trailing stop loss based on profit threshold."""
    try:
        positions = session.get_positions(
            category="linear",
            settleCoin="USDT",
            accountType="UNIFIED"
        )
        
        # Check for API errors
        if positions.get("retCode") != 0:
            if log_slot:
                log_slot.error(f"❌ API error: {positions.get('retMsg', 'Unknown error')}")
            return
        
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
                    try:
                        session.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            stopLoss=new_sl
                        )
                        updated_any = True
                        
                        if log_slot:
                            log_slot.success(f"📈 Trailing SL updated for {symbol}: ${new_sl}")
                    except Exception as e:
                        if log_slot:
                            log_slot.error(f"❌ Failed to update SL for {symbol}: {e}")
        
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
        engine.stop()  # Clean up
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
        
        # Use the new OpenAI client format if available
        if hasattr(openai, 'OpenAI'):
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
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
            return response.choices[0].message.content
        else:
            # Fallback to legacy format
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
        return f"Mariah AI error: {e}"

def listen_to_user():
    """Speech-to-text function for voice input."""
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
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
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

def send_email_with_attachment(subject, body, to_email, filename):
    """Send email with attachment."""
    try:
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
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

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
        
    # Ensure necessary columns exist
    df_trades = df_trades.copy()
    if "Realized PnL ($)" not in df_trades.columns:
        # Calculate PnL if it doesn't exist
        if all(col in df_trades.columns for col in ["take_profit", "entry_price", "qty"]):
            df_trades["Realized PnL ($)"] = (
                (df_trades["take_profit"] - df_trades["entry_price"]) * df_trades["qty"]
                - (df_trades["entry_price"] + df_trades["take_profit"]) * df_trades["qty"] * FEE_RATE
            )
        else:
            st.error("Missing required columns for PnL calculation")
            return
    
    # Ensure timestamp is datetime
    if "timestamp" in df_trades.columns:
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors='coerce')
    
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
        
        if "symbol" in df_trades.columns:
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
        if "timestamp" in df_trades.columns:
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
        
        if "timestamp" in df_trades.columns:
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
                text=df_trades.get("symbol", ""),
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
# ENHANCED SIGNAL SCANNER SYSTEM
# =======================

# Supporting Functions for the Enhanced Scanner
def calculate_atr(df, period=14):
    """Calculate Average True Range for volatility measurement"""
    if df.empty or len(df) < period:
        return 0
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean().iloc[-1] if len(true_range) >= period else 0


def get_simple_sentiment_score(symbol):
    """Simplified sentiment analysis - would integrate with real news API"""
    # This would normally query news APIs and analyze sentiment
    # For now, return a random score for demonstration
    import random
    return random.uniform(0.3, 0.8)


def get_simple_onchain_signal(symbol):
    """Simplified on-chain analysis - would integrate with Glassnode/Santiment"""
    # This would normally query on-chain data APIs
    # For now, return a mock signal
    import random
    
    signals = ['buy', 'sell', 'hold']
    confidences = [0.6, 0.7, 0.8, 0.9]
    
    mock_signal = random.choice(signals)
    mock_confidence = random.choice(confidences)
    
    reasons = {
        'buy': 'Large inflows to exchanges detected',
        'sell': 'Whale movements suggest distribution',
        'hold': 'On-chain metrics showing consolidation'
    }
    
    return {
        'signal': mock_signal,
        'confidence': mock_confidence,
        'reason': reasons[mock_signal]
    }


def calculate_weighted_consensus(scanner_results, weights, min_consensus):
    """Calculate weighted consensus from multiple scanners"""
    if not scanner_results:
        return {
            'final_signal': 'hold',
            'consensus_score': 0.0,
            'aligned_count': 0,
            'weighted_score': 0.0
        }
    
    # Separate signals by type
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    total_weighted_score = 0
    total_weight = 0
    
    for scanner_name, result in scanner_results.items():
        weight = weights.get(scanner_name, 1.0)
        weighted_confidence = result['confidence'] * weight
        
        if result['signal'] == 'buy':
            buy_signals.append(weighted_confidence)
        elif result['signal'] == 'sell':
            sell_signals.append(weighted_confidence)
        else:
            hold_signals.append(weighted_confidence)
        
        total_weighted_score += weighted_confidence
        total_weight += weight
    
    # Calculate scores
    buy_score = sum(buy_signals)
    sell_score = sum(sell_signals)
    hold_score = sum(hold_signals)
    
    # Determine final signal
    if buy_score > sell_score and buy_score > hold_score:
        final_signal = 'buy'
        consensus_score = buy_score / total_weighted_score if total_weighted_score > 0 else 0
        aligned_count = len(buy_signals)
    elif sell_score > hold_score:
        final_signal = 'sell'
        consensus_score = sell_score / total_weighted_score if total_weighted_score > 0 else 0
        aligned_count = len(sell_signals)
    else:
        final_signal = 'hold'
        consensus_score = hold_score / total_weighted_score if total_weighted_score > 0 else 0
        aligned_count = len(hold_signals)
    
    # Calculate final weighted score
    avg_weighted_score = total_weighted_score / total_weight if total_weight > 0 else 0
    
    return {
        'final_signal': final_signal,
        'consensus_score': consensus_score,
        'aligned_count': aligned_count,
        'weighted_score': avg_weighted_score
    }

# =======================
# MORE TOOLS TAB FUNCTIONS
# =======================
def render_daily_pnl():
    """Render Daily PnL view"""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📆 Daily PnL Summary")
    
    try:
        # Load daily PnL data
        if os.path.exists(DAILY_PNL_SPLIT_FILE):
            df_daily = pd.read_csv(DAILY_PNL_SPLIT_FILE)
            df_daily["date"] = pd.to_datetime(df_daily["date"])
            
            # Calculate rolling averages
            df_daily["bot_pnl_7d"] = df_daily["bot_pnl"].rolling(7).mean()
            df_daily["manual_pnl_7d"] = df_daily["manual_pnl"].rolling(7).mean()
            df_daily["total_pnl_7d"] = df_daily["total_pnl"].rolling(7).mean()
            
            # Calculate win rates
            df_daily["bot_win_rate"] = np.where(
                df_daily["bot_trades"] > 0,
                df_daily["bot_wins"] / df_daily["bot_trades"] * 100,
                0
            )
            df_daily["manual_win_rate"] = np.where(
                df_daily["manual_trades"] > 0,
                df_daily["manual_wins"] / df_daily["manual_trades"] * 100,
                0
            )
            
            # Display recent data
            st.subheader("📊 Last 10 Days")
            recent_data = df_daily.tail(10)
            st.dataframe(recent_data[[
                "date", "bot_pnl", "manual_pnl", "total_pnl",
                "bot_trades", "manual_trades", "bot_win_rate", "manual_win_rate"
            ]])
            
            # Create daily PnL chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_daily["date"],
                y=df_daily["bot_pnl"],
                name="Bot PnL",
                marker_color="blue",
                opacity=0.7
            ))
            
            fig.add_trace(go.Bar(
                x=df_daily["date"],
                y=df_daily["manual_pnl"],
                name="Manual PnL",
                marker_color="green",
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Daily PnL - Bot vs Manual",
                xaxis_title="Date",
                yaxis_title="PnL ($)",
                barmode="stack"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Win rate trends
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=df_daily["date"],
                y=df_daily["bot_win_rate"],
                mode="lines+markers",
                name="Bot Win Rate (%)",
                line=dict(color="blue")
            ))
            
            fig2.add_trace(go.Scatter(
                x=df_daily["date"],
                y=df_daily["manual_win_rate"],
                mode="lines+markers",
                name="Manual Win Rate (%)",
                line=dict(color="green")
            ))
            
            fig2.update_layout(
                title="Win Rate Trends",
                xaxis_title="Date",
                yaxis_title="Win Rate (%)"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.info("No daily PnL data available yet.")
            
    except Exception as e:
        st.error(f"❌ Error loading daily PnL data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_performance_trends():
    """Render Performance Trends view"""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📈 Performance Trends Analysis")
    
    try:
        # Load trades data
        df_trades = load_trades()
        
        if df_trades.empty:
            st.info("No trade data available for trend analysis.")
            return
        
        # Ensure required columns
        df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors='coerce')
        
        # Split bot trades
        df_bot_open, df_bot_closed = split_bot_trades(df_trades)
        
        if df_bot_closed.empty:
            st.info("No closed trades available for analysis.")
            return
        
        # Monthly trends
        st.subheader("📅 Monthly Performance")
        
        df_bot_closed["month"] = df_bot_closed["timestamp"].dt.to_period("M")
        monthly_stats = df_bot_closed.groupby("month").agg({
            "Realized PnL ($)": ["sum", "mean", "count"],
            "symbol": "count"
        }).round(2)
        
        # Flatten column names
        monthly_stats.columns = ["Total PnL", "Avg PnL", "Trade Count", "Symbols"]
        monthly_stats = monthly_stats.reset_index()
        monthly_stats["month"] = monthly_stats["month"].astype(str)
        
        # Calculate win rates
        win_rates = []
        for period in monthly_stats["month"]:
            period_data = df_bot_closed[df_bot_closed["month"].astype(str) == period]
            wins = len(period_data[period_data["Realized PnL ($)"] > 0])
            total = len(period_data)
            win_rate = (wins / total * 100) if total > 0 else 0
            win_rates.append(win_rate)
        
        monthly_stats["Win Rate (%)"] = win_rates
        st.dataframe(monthly_stats)
        
        # Performance by symbol
        st.subheader("🎯 Performance by Symbol")
        
        if "symbol" in df_bot_closed.columns:
            symbol_performance = df_bot_closed.groupby("symbol").agg({
                "Realized PnL ($)": ["sum", "mean", "count"]
            }).round(2)
            
            symbol_performance.columns = ["Total PnL", "Avg PnL", "Trade Count"]
            symbol_performance = symbol_performance.reset_index()
            
            # Add win rates by symbol
            symbol_win_rates = []
            for symbol in symbol_performance["symbol"]:
                symbol_data = df_bot_closed[df_bot_closed["symbol"] == symbol]
                wins = len(symbol_data[symbol_data["Realized PnL ($)"] > 0])
                total = len(symbol_data)
                win_rate = (wins / total * 100) if total > 0 else 0
                symbol_win_rates.append(win_rate)
            
            symbol_performance["Win Rate (%)"] = symbol_win_rates
            symbol_performance = symbol_performance.sort_values("Total PnL", ascending=False)
            
            st.dataframe(symbol_performance)
        
        # Strategy mode performance
        if "mode" in df_bot_closed.columns:
            st.subheader("⚙️ Performance by Strategy Mode")
            
            mode_performance = df_bot_closed.groupby("mode").agg({
                "Realized PnL ($)": ["sum", "mean", "count"]
            }).round(2)
            
            mode_performance.columns = ["Total PnL", "Avg PnL", "Trade Count"]
            mode_performance = mode_performance.reset_index()
            
            st.dataframe(mode_performance)
        
        # Time of day analysis
        st.subheader("🕐 Performance by Hour of Day")
        
        df_bot_closed["hour"] = df_bot_closed["timestamp"].dt.hour
        hourly_pnl = df_bot_closed.groupby("hour")["Realized PnL ($)"].agg(["sum", "count"]).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_pnl["hour"],
            y=hourly_pnl["sum"],
            name="PnL by Hour"
        ))
        
        fig.update_layout(
            title="Total PnL by Hour of Day",
            xaxis_title="Hour (24h format)",
            yaxis_title="Total PnL ($)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error analyzing performance trends: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_filter_by_date():
    """Render Filter by Date view"""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📆 Filter Trades by Date Range")
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.Timestamp.now().date() - pd.Timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.Timestamp.now().date()
        )
    
    if start_date > end_date:
        st.error("❌ Start date must be before end date!")
        return
    
    # Trade type selection
    trade_types = st.multiselect(
        "Select Trade Types",
        ["Bot Trades", "Manual Trades"],
        default=["Bot Trades", "Manual Trades"]
    )
    
    # Apply filters
    if st.button("🔍 Apply Filters"):
        try:
            all_trades = []
            
            # Filter bot trades
            if "Bot Trades" in trade_types:
                df_trades = load_trades()
                if not df_trades.empty:
                    df_trades["timestamp"] = pd.to_datetime(df_trades["timestamp"], errors='coerce')
                    df_trades["trade_type"] = "Bot"
                    # Ensure required columns exist
                    if "Realized PnL ($)" not in df_trades.columns:
                        # Calculate if needed
                        if all(col in df_trades.columns for col in ["take_profit", "entry_price", "qty"]):
                            df_trades = split_bot_trades(df_trades)[1]  # Get closed trades with PnL
                        else:
                            df_trades["Realized PnL ($)"] = 0
                    all_trades.append(df_trades)
            
            # Filter manual trades
            if "Manual Trades" in trade_types:
                df_manual = load_closed_manual_trades(session)
                if not df_manual.empty:
                    # Standardize manual trades format
                    df_manual_std = pd.DataFrame({
                        "timestamp": pd.to_datetime(df_manual.get("timestamp", pd.Timestamp.now())),
                        "symbol": df_manual.get("Symbol", df_manual.get("symbol", "")),
                        "side": df_manual.get("Side", df_manual.get("side", "")),
                        "qty": df_manual.get("Size", df_manual.get("qty", 0)),
                        "entry_price": df_manual.get("Entry Price", df_manual.get("entry_price", 0)),
                        "stop_loss": 0,
                        "take_profit": df_manual.get("Exit Price", df_manual.get("take_profit", 0)),
                        "note": "manual",
                        "Realized PnL ($)": df_manual["Realized PnL ($)"],
                        "Realized PnL (%)": df_manual.get("Realized PnL (%)", 0),
                        "trade_type": "Manual"
                    })
                    all_trades.append(df_manual_std)
            
            if not all_trades:
                st.warning("No trades found with selected filters.")
                return
            
            # Combine all trades
            df_combined = pd.concat(all_trades, ignore_index=True)
            
            # Apply date filter
            df_filtered = df_combined[
                (df_combined["timestamp"].dt.date >= start_date) &
                (df_combined["timestamp"].dt.date <= end_date)
            ]
            
            if df_filtered.empty:
                st.warning("No trades found in the selected date range.")
                return
            
            # Display summary
            st.subheader("📊 Filtered Results Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(df_filtered))
            
            with col2:
                total_pnl = df_filtered["Realized PnL ($)"].sum()
                st.metric("Total PnL", f"${total_pnl:,.2f}")
            
            with col3:
                wins = len(df_filtered[df_filtered["Realized PnL ($)"] > 0])
                win_rate = (wins / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.2f}%")
            
            with col4:
                avg_pnl = df_filtered["Realized PnL ($)"].mean()
                st.metric("Avg PnL", f"${avg_pnl:.2f}")
            
            # Display trades
            st.subheader("📋 Filtered Trades")
            
            # Create display dataframe
            display_columns = ["timestamp", "symbol", "side", "qty", "entry_price",
                             "stop_loss", "take_profit", "note", "Realized PnL ($)",
                             "Realized PnL (%)", "trade_type"]
            
            # Ensure all columns exist
            for col in display_columns:
                if col not in df_filtered.columns:
                    df_filtered[col] = 0 if "PnL" in col else ""
            
            display_df = df_filtered[display_columns].copy()
            
            # Sort by timestamp
            display_df = display_df.sort_values("timestamp", ascending=False)
            
            st.dataframe(display_df)
            
            # Create visualization
            st.subheader("📈 PnL Over Time")
            
            # Daily PnL aggregation
            df_filtered["date"] = df_filtered["timestamp"].dt.date
            daily_pnl = df_filtered.groupby(["date", "trade_type"])["Realized PnL ($)"].sum().reset_index()
            
            fig = px.bar(
                daily_pnl,
                x="date",
                y="Realized PnL ($)",
                color="trade_type",
                title="Daily PnL by Trade Type"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error filtering trades: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_crypto_news():
    """Render the Crypto News tab."""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📰 Real-Time Crypto News")
    
    try:
        # Get crypto news from external API
        news_items = get_crypto_news()
        
        if not news_items:
            st.info("No news available right now.")
        else:
            # Create tabs for different news categories
            tabs = st.tabs(["📰 All News", "🚨 Alerts", "📈 Market Analysis"])
            
            with tabs[0]:
                st.subheader("Latest Crypto News")
                
                for item in news_items:
                    title = item.get("title", "No title")
                    url = item.get("url", "#")
                    published = item.get("published", "Unknown")
                    source = item.get("source", "Unknown")
                    description = item.get("description", "")
                    
                    # Determine alert level based on keywords
                    alert = ""
                    alert_color = "white"
                    
                    title_lower = title.lower()
                    if any(keyword in title_lower for keyword in ["hack", "exploit", "attack", "stolen"]):
                        alert = "🚨 SECURITY ALERT "
                        alert_color = "red"
                    elif any(keyword in title_lower for keyword in ["bullish", "moon", "rally", "surge"]):
                        alert = "📈 BULLISH "
                        alert_color = "green"
                    elif any(keyword in title_lower for keyword in ["bearish", "crash", "dump", "fall"]):
                        alert = "📉 BEARISH "
                        alert_color = "orange"
                    
                    # Create news card
                    st.markdown(f"""
                    <div style="border: 1px solid rgba(255,255,255,0.1); 
                                border-radius: 10px; 
                                padding: 15px; 
                                margin-bottom: 15px;
                                background-color: rgba(255,255,255,0.02);">
                        <h4>
                            <span style="color: {alert_color};">{alert}</span>
                            <a href="{url}" target="_blank" style="color: #00fff5; text-decoration: none;">
                                {title}
                            </a>
                        </h4>
                        <p style="color: #cccccc; font-size: 0.9rem;">
                            {description[:200]}{'...' if len(description) > 200 else ''}
                        </p>
                        <p style="color: #888; font-size: 0.8rem;">
                            📅 {published} | 📰 {source}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tabs[1]:
                st.subheader("Security & High-Impact Alerts")
                
                # Filter for security-related news
                security_alerts = [
                    item for item in news_items
                    if any(keyword in item.get("title", "").lower() 
                          for keyword in ["hack", "exploit", "attack", "stolen", "breach", "scam"])
                ]
                
                if security_alerts:
                    for alert in security_alerts:
                        st.error(f"🚨 **{alert.get('source', 'Unknown')}**: {alert.get('title', 'No title')}")
                        st.write(f"🔗 [Read more]({alert.get('url', '#')})")
                        st.markdown("---")
                else:
                    st.success("✅ No security alerts at this time.")
            
            with tabs[2]:
                st.subheader("Market Analysis & Insights")
                
                # Filter for market analysis
                analysis_news = [
                    item for item in news_items
                    if any(keyword in item.get("title", "").lower() 
                          for keyword in ["analysis", "prediction", "forecast", "outlook", "technical"])
                ]
                
                if analysis_news:
                    for item in analysis_news:
                        title = item.get("title", "No title")
                        url = item.get("url", "#")
                        source = item.get("source", "Unknown")
                        
                        st.markdown(f"""
                        <div style="border-left: 4px solid #00fff5; 
                                    padding-left: 15px; 
                                    margin-bottom: 15px;">
                            <h4><a href="{url}" target="_blank" style="color: #00fff5; text-decoration: none;">{title}</a></h4>
                            <p style="color: #888;">📰 {source}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No market analysis articles available.")
                    
    except Exception as e:
        st.error(f"❌ Failed to fetch crypto news: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_onchain_data():
    """Render On-Chain Data tab."""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("📡 ETH Gas + Block Info (Etherscan)")
    
    try:
        # Get on-chain data
        gas = get_eth_gas()
        block = get_block_info()
        
        # Gas prices section
        st.subheader("⛽ Current Gas Prices")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🐌 Safe Gas", f"{gas['low']} Gwei", help="Recommended for non-urgent transactions")
        
        with col2:
            st.metric("⚡ Standard Gas", f"{gas['avg']} Gwei", help="Recommended for normal transactions")
        
        with col3:
            st.metric("🚀 Fast Gas", f"{gas['high']} Gwei", help="Recommended for urgent transactions")
        
        with col4:
            st.metric("📦 Latest Block", f"#{block}", help="Most recent block number")
        
        # Gas price history chart
        if 'gas_history' in st.session_state:
            st.subheader("📈 Gas Price History")
            
            # Add current data to history
            current_time = pd.Timestamp.now()
            new_data = pd.DataFrame({
                'timestamp': [current_time],
                'low': [gas['low']],
                'avg': [gas['avg']],
                'high': [gas['high']]
            })
            
            # Keep last 100 data points
            st.session_state.gas_history = pd.concat([st.session_state.gas_history, new_data]).tail(100)
        else:
            # Initialize with current data
            st.session_state.gas_history = pd.DataFrame({
                'timestamp': [pd.Timestamp.now()],
                'low': [gas['low']],
                'avg': [gas['avg']],
                'high': [gas['high']]
            })
        
        # Create gas price chart
        fig = go.Figure()
        
        gas_history = st.session_state.gas_history
        
        fig.add_trace(go.Scatter(
            x=gas_history['timestamp'],
            y=gas_history['low'],
            mode='lines+markers',
            name='Safe Gas',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=gas_history['timestamp'],
            y=gas_history['avg'],
            mode='lines+markers',
            name='Standard Gas',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=gas_history['timestamp'],
            y=gas_history['high'],
            mode='lines+markers',
            name='Fast Gas',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Gas Price Trends",
            xaxis_title="Time",
            yaxis_title="Gas Price (Gwei)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Network stats
        st.subheader("🌐 Network Statistics")
        
        # Get additional network data (if available)
        try:
            # Placeholder for additional network stats
            # In a real implementation, you would fetch:
            # - Network congestion
            # - Average block time
            # - Pending transactions
            # - MEV data
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("⏱️ Avg Block Time", "12-15 seconds", help="Current average block confirmation time")
                st.metric("🏗️ Network Congestion", "Medium", help="Based on pending transactions")
            
            with col2:
                st.metric("💸 Average Transaction Fee", f"${(gas['avg'] * 21000 * 2000 / 1e9):.2f}", help="Estimated for standard transfer")
                st.metric("⏳ Pending Txs", "~50,000", help="Approximate pending transactions")
            
        except Exception as e:
            st.warning(f"Additional network stats unavailable: {e}")
        
        # Gas optimization tips
        st.subheader("💡 Gas Optimization Tips")
        
        st.markdown("""
        **Tips to save on gas fees:**
        
        - 🕐 **Optimal timing**: Gas prices are typically lower during weekends and late night hours (UTC)
        - 📊 **Monitor trends**: Use the chart above to identify low-gas periods
        - 🔄 **Batch transactions**: Combine multiple operations when possible
        - ⚙️ **Optimize smart contracts**: Use more efficient code patterns
        - 🎯 **Set appropriate gas limits**: Avoid over-estimating gas requirements
        """)
        
    except Exception as e:
        st.error(f"❌ Failed to fetch on-chain data: {e}")
        st.info("Please check your internet connection and API keys.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_signal_scanner(mode, account_balance, df_bot_closed, df_manual_closed):
    """Enhanced Multi-Signal Scanner with weighted consensus logic"""
    with st.container():
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📡 Multi-Signal Scanner System")
        
        # Initialize Mariah Level 2
        if "mariah_level2" not in st.session_state:
            st.session_state.mariah_level2 = MariahLevel2()
        
        mariah2 = st.session_state.mariah_level2
        
        # Signal Scanner Selection
        st.markdown("### 🔍 Available Signal Scanners")
        
        # Create columns for scanner selection
        scanner_cols = st.columns(3)
        
        with scanner_cols[0]:
            use_rsi_macd = st.checkbox("📊 RSI/MACD Scanner", value=True, help="Classic oversold/overbought entries and trend confirmations")
            use_ml = st.checkbox("🤖 ML Signal Scanner", value=True, help="Probabilistic predictions from trained models")
        
        with scanner_cols[1]:
            use_sentiment = st.checkbox("📰 News Sentiment Scanner", value=True, help="Detects keywords and adjusts risk/direction")
            use_onchain = st.checkbox("⛓️ On-Chain Flow Scanner", value=True, help="Monitors whale flows, gas spikes, CEX flows")
        
        with scanner_cols[2]:
            use_enhanced = st.checkbox("🔧 Enhanced Multi-Indicator", value=True, help="Advanced technical analysis combination")
        
        # Consensus Logic Settings
        st.markdown("---")
        st.markdown("### ⚖️ Consensus Logic Settings")
        
        logic_cols = st.columns(3)
        
        with logic_cols[0]:
            min_consensus = st.slider("Minimum Scanners Aligned", min_value=2, max_value=5, value=3, 
                                    help="Minimum number of scanners that must agree before executing")
        
        with logic_cols[1]:
            volatility_threshold = st.slider("High Volatility Threshold (%)", min_value=1.0, max_value=10.0, value=5.0, step=0.5,
                                           help="ATR percentage threshold for high volatility period")
        
        with logic_cols[2]:
            high_vol_weight_ml = st.slider("ML Weight (High Vol)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
            high_vol_weight_onchain = st.slider("On-Chain Weight (High Vol)", min_value=1.0, max_value=3.0, value=1.3, step=0.1)
        
        # Scanner Weights Configuration
        st.markdown("### 🏋️ Scanner Weights Configuration")
        weight_cols = st.columns(5)
        
        weights = {}
        with weight_cols[0]:
            weights['rsi_macd'] = st.number_input("RSI/MACD Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        with weight_cols[1]:
            weights['ml'] = st.number_input("ML Weight", min_value=0.1, max_value=2.0, value=1.2, step=0.1)
        with weight_cols[2]:
            weights['sentiment'] = st.number_input("Sentiment Weight", min_value=0.1, max_value=2.0, value=0.8, step=0.1)
        with weight_cols[3]:
            weights['onchain'] = st.number_input("On-Chain Weight", min_value=0.1, max_value=2.0, value=1.1, step=0.1)
        with weight_cols[4]:
            weights['enhanced'] = st.number_input("Enhanced Weight", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        
        # Trading Parameters
        st.markdown("---")
        st.markdown("### 📈 Trading Parameters")
        
        param_cols = st.columns(2)
        with param_cols[0]:
            interval = st.selectbox("Candle Interval", ["5", "15", "30", "60", "240"], index=1)
        with param_cols[1]:
            symbols = st.multiselect("Symbols to Scan", 
                                   ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT"], 
                                   default=["BTCUSDT", "ETHUSDT"])
        
        # Get risk percent from sidebar or default
        risk_percent = st.session_state.get("risk_percent", 2.0)
        
        st.markdown("---")
        
        # Enhanced Analysis for each symbol
        for symbol in symbols:
            with st.expander(f"📈 {symbol} Multi-Signal Analysis", expanded=True):
                # Daily loss guard
                if check_max_daily_loss(df_bot_closed, df_manual_closed):
                    st.warning(f"🛑 Mariah skipped {symbol} — Daily loss limit reached.")
                    continue
                
                try:
                    # Initialize signal results
                    scanner_results = {}
                    signal_signals = []
                    
                    # Get current market data for volatility calculation
                    historical_data = get_historical_data(session, symbol, interval, limit=100)
                    if not historical_data.empty:
                        # Calculate ATR for volatility
                        atr = calculate_atr(historical_data)
                        current_price = historical_data['close'].iloc[-1]
                        volatility_pct = (atr / current_price) * 100
                        is_high_volatility = volatility_pct > volatility_threshold
                        
                        # Adjust weights for high volatility
                        current_weights = weights.copy()
                        if is_high_volatility:
                            current_weights['ml'] *= high_vol_weight_ml
                            current_weights['onchain'] *= high_vol_weight_onchain
                            st.info(f"🔥 High volatility detected ({volatility_pct:.2f}%) - ML and On-Chain weights increased")
                    else:
                        is_high_volatility = False
                        current_weights = weights.copy()
                    
                # 1. RSI/MACD Scanner
                    if use_rsi_macd:
                        rsi_value, rsi_trigger = check_rsi_signal(session, symbol=symbol, interval=interval, mode=mode)
                        
                        # Get MACD data
                        if not historical_data.empty:
                            macd_data = ta.macd(historical_data['close'])
                            if not macd_data['MACD_12_26_9'].empty and not macd_data['MACDs_12_26_9'].empty:
                                macd_line = macd_data['MACD_12_26_9'].iloc[-1]
                                signal_line = macd_data['MACDs_12_26_9'].iloc[-1]
                                macd_bullish = macd_line > signal_line
                            else:
                                macd_bullish = False
                        else:
                            macd_bullish = False
                        
                        # Format RSI value safely - this is the key fix
                        rsi_display = f"{rsi_value:.1f}" if rsi_value is not None else "N/A"
                        macd_display = "+" if macd_bullish else "-"
                        
                        # Combine RSI and MACD
                        if rsi_trigger and macd_bullish:
                            scanner_results['rsi_macd'] = {
                                'signal': 'buy',
                                'confidence': 0.85,
                                'reason': f'RSI oversold ({rsi_display}) + MACD bullish crossover'
                            }
                        elif rsi_value is not None and rsi_value > 70 and not macd_bullish:
                            scanner_results['rsi_macd'] = {
                                'signal': 'sell',
                                'confidence': 0.75,
                                'reason': f'RSI overbought ({rsi_display}) + MACD bearish'
                            }
                        else:
                            scanner_results['rsi_macd'] = {
                                'signal': 'hold',
                                'confidence': 0.5,
                                'reason': f'RSI: {rsi_display}, MACD: {macd_display}'
                            }
                    
                    # 2. ML Signal Scanner
                    if use_ml and not historical_data.empty:
                        ml_generator = MLSignalGenerator(model_path=f"models/{symbol}_{interval}_model.pkl")
                        ml_signal, ml_confidence = ml_generator.get_signal(historical_data)
                        
                        scanner_results['ml'] = {
                            'signal': ml_signal,
                            'confidence': ml_confidence,
                            'reason': f'ML model prediction: {ml_signal} ({ml_confidence:.1%})'
                        }
                    
                    # 3. News Sentiment Scanner
                    if use_sentiment:
                        # Simplified news sentiment (you would integrate with real news API)
                        sentiment_score = get_simple_sentiment_score(symbol)
                        
                        if sentiment_score > 0.6:
                            scanner_results['sentiment'] = {
                                'signal': 'buy',
                                'confidence': min(sentiment_score, 0.9),
                                'reason': f'Positive sentiment ({sentiment_score:.2f})'
                            }
                        elif sentiment_score < 0.4:
                            scanner_results['sentiment'] = {
                                'signal': 'sell',
                                'confidence': min(1 - sentiment_score, 0.9),
                                'reason': f'Negative sentiment ({sentiment_score:.2f})'
                            }
                        else:
                            scanner_results['sentiment'] = {
                                'signal': 'hold',
                                'confidence': 0.5,
                                'reason': f'Neutral sentiment ({sentiment_score:.2f})'
                            }
                    
                    # 4. On-Chain Flow Scanner
                    if use_onchain:
                        # Simplified on-chain analysis (you would integrate with Glassnode/Santiment)
                        onchain_signal = get_simple_onchain_signal(symbol)
                        
                        scanner_results['onchain'] = onchain_signal
                    
                    # 5. Enhanced Multi-Indicator Scanner
                    if use_enhanced:
                        analysis = mariah2.analyze_symbol(symbol, interval, session)
                        
                        if "error" not in analysis:
                            scanner_results['enhanced'] = {
                                'signal': analysis['action'],
                                'confidence': analysis['confidence'],
                                'reason': analysis['summary']
                            }
                    
                    # Calculate Weighted Consensus
                    consensus_result = calculate_weighted_consensus(scanner_results, current_weights, min_consensus)
                    
                    # Display Results
                    st.markdown(f"### 🎯 Consensus Result for {symbol}")
                    
                    result_cols = st.columns(4)
                    with result_cols[0]:
                        action_color = {
                            'buy': '🟢',
                            'sell': '🔴',
                            'hold': '⚪'
                        }[consensus_result['final_signal']]
                        st.metric("Final Signal", f"{action_color} {consensus_result['final_signal'].upper()}")
                    
                    with result_cols[1]:
                        st.metric("Consensus Strength", f"{consensus_result['consensus_score']:.1%}")
                    
                    with result_cols[2]:
                        st.metric("Scanners Aligned", f"{consensus_result['aligned_count']}/{len(scanner_results)}")
                    
                    with result_cols[3]:
                        st.metric("Weighted Score", f"{consensus_result['weighted_score']:.2f}")
                    
                    # Show individual scanner results
                    st.markdown("### 📊 Individual Scanner Results")
                    
                    for scanner_name, result in scanner_results.items():
                        signal_emoji = {
                            'buy': '🟢',
                            'sell': '🔴',
                            'hold': '⚪'
                        }[result['signal']]
                        
                        weight = current_weights.get(scanner_name, 1.0)
                        weighted_contribution = result['confidence'] * weight
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{scanner_name.replace('_', ' ').title()}**")
                        with col2:
                            st.write(f"{signal_emoji} {result['signal'].upper()}")
                        with col3:
                            st.write(f"Confidence: {result['confidence']:.1%}")
                        with col4:
                            st.write(f"Weighted: {weighted_contribution:.2f}")
                        
                        st.caption(f"💡 {result['reason']}")
                        st.markdown("---")
                    
                    # Execute Trade Button
                    if consensus_result['final_signal'] != 'hold' and consensus_result['aligned_count'] >= min_consensus:
                        if st.button(f"🚀 Execute {consensus_result['final_signal'].upper()} for {symbol}", 
                                   key=f"exec_consensus_{symbol}"):
                            
                            # Calculate position size
                            sl_pct, tp_pct, _, _ = get_strategy_params(mode)
                            current_price = float(session.get_tickers(category="linear", symbol=symbol)["result"]["list"][0]["lastPrice"])
                            
                            if consensus_result['final_signal'] == 'buy':
                                stop_loss = current_price * (1 - sl_pct / 100)
                            else:
                                stop_loss = current_price * (1 + sl_pct / 100)
                            
                            qty = position_size_from_risk(account_balance, risk_percent, current_price, stop_loss)
                            
                            # Reduce position size if consensus is weak
                            if consensus_result['consensus_score'] < 0.7:
                                qty *= 0.5  # Half position for weak consensus
                                st.info(f"⚠️ Reducing position size by 50% due to weak consensus")
                            
                            # Execute the order
                            try:
                                order = session.place_order(
                                    category="linear",
                                    symbol=symbol,
                                    side=consensus_result['final_signal'].title(),
                                    orderType="Market",
                                    qty=round(qty, 3),
                                    timeInForce="GoodTillCancel",
                                    reduceOnly=False,
                                    closeOnTrigger=False
                                )
                                
                                # Log with consensus details
                                log_rsi_trade_to_csv(
                                    symbol=symbol,
                                    side=consensus_result['final_signal'].title(),
                                    qty=qty,
                                    entry_price=current_price,
                                    mode=f"{mode}_Consensus_{consensus_result['aligned_count']}_scanners"
                                )
                                
                                st.success(f"✅ Consensus trade executed for {symbol}!")
                                st.json(order)
                                
                                # Mariah speaks about the consensus
                                mariah_speak(f"Consensus trade executed for {symbol}! "
                                           f"{consensus_result['aligned_count']} scanners aligned with "
                                           f"{consensus_result['consensus_score']:.1%} confidence.")
                            except Exception as e:
                                st.error(f"❌ Failed to execute order: {e}")
                    
                    elif consensus_result['aligned_count'] < min_consensus:
                        st.warning(f"⚠️ Insufficient consensus: Only {consensus_result['aligned_count']} scanners aligned "
                                 f"(need {min_consensus})")
                    
                except Exception as e:
                    st.error(f"❌ Error in multi-signal analysis for {symbol}: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Risk Controls
        st.markdown("---")
        st.markdown("### 🛑 Risk Controls")
        st.session_state["override_risk_lock"] = st.checkbox(
            "🚨 Manually override Mariah's risk lock (not recommended)",
            value=st.session_state.get("override_risk_lock", False)
        )
# =======================
# MAIN FUNCTION
# =======================
def main():
    """Enhanced main dashboard function with visual improvements."""
    
    # Load environment variables
    load_dotenv()
    
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # ADD THIS LINE HERE - RIGHT AFTER page config
    create_crypto_ticker(session)
    
    # Initial Mariah greeting
    if not st.session_state.get("mariah_greeted", False):
        mariah_speak("System online. Welcome to the Crypto Capital.")
        st.session_state["mariah_greeted"] = True
    
    # Load images with error handling
    logo_base64 = get_base64_image("IMG_7006.PNG")
    brain_base64 = get_base64_image("updatedbrain1.png")
    
    # Apply background image
    set_dashboard_background("Screenshot 2025.png")
    
    # =======================
    # ENHANCED HEADER
    # =======================
    # ... rest of your code
    # =======================
    # ENHANCED HEADER
    # =======================
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
    try:
        with open("ChatGPT Image May 4, 2025, 07_29_01 PM.png", "rb") as img_file:
            mariah_base64 = base64.b64encode(img_file.read()).decode()
        
        with header_col2:
            st.markdown(f"""
            <img src="data:image/png;base64,{mariah_base64}"
                class="mariah-avatar"
                width="130"
                style="margin-top: 0.5rem; border-radius: 12px;" />
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Could not load Mariah avatar: {e}")
    
    # =======================
    # ENHANCED SIDEBAR
    # =======================
    with st.sidebar:
        # AI Banner
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin: 1rem 0 0.5rem 0.5rem;">
            <span style="color: #00fff5; font-size: 1.1rem; font-weight: 600;">Powered by AI</span>
            <img src="data:image/png;base64,{brain_base64}" width="26" class="pulse-brain" />
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Actions Panel
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📡 Scanner", use_container_width=True, type="secondary"):
                st.session_state.current_tool = "Signal Scanner"
        with col2:
            if st.button("📰 News", use_container_width=True, type="secondary"):
                st.session_state.current_tool = "Crypto News"
        
        # Quick Execute Panel
        st.markdown("### 🚀 Quick Execute")
        quick_symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"], key="quick_symbol")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 BUY", type="primary", use_container_width=True):
                st.session_state.quick_action = "buy"
                st.session_state.quick_symbol = quick_symbol
                st.success(f"Quick BUY {quick_symbol} triggered!")
        with col2:
            if st.button("📉 SELL", use_container_width=True):
                st.session_state.quick_action = "sell"
                st.session_state.quick_symbol = quick_symbol
                st.error(f"Quick SELL {quick_symbol} triggered!")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Collapsible Tool Groups
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        with st.expander("📊 Analytics Tools", expanded=False):
            analytics_tool = st.radio(
                "Choose Analytics:",
                ["📆 Daily PnL", "📈 Performance Trends", "📊 Advanced Analytics"],
                key="analytics_selector"
            )
            if st.button("Open Analytics Tool", use_container_width=True):
                st.session_state.current_tool = analytics_tool
        
        with st.expander("📡 Data Sources", expanded=False):
            data_tool = st.radio(
                "Choose Data Source:",
                ["📡 On-Chain Data", "📆 Filter by Date"],
                key="data_selector"
            )
            if st.button("Open Data Tool", use_container_width=True):
                st.session_state.current_tool = data_tool
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Dashboard Controls
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.title("📊 Dashboard Controls")
        
        # Voice Settings
        st.markdown("### 🎤 Voice Settings")
        st.session_state["mute_mariah"] = st.checkbox(
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
        account_balance = st.number_input("Account Balance ($)", value=5000.0)
        risk_percent = st.slider("Risk % Per Trade", min_value=0.5, max_value=5.0, value=2.0)
        st.session_state["risk_percent"] = risk_percent  # Store in session state
        entry_price_sidebar = st.number_input("Entry Price", value=0.0, format="%.2f")
        stop_loss_sidebar = st.number_input("Stop Loss Price", value=0.0, format="%.2f")
        
        if entry_price_sidebar > 0 and stop_loss_sidebar > 0 and entry_price_sidebar != stop_loss_sidebar:
            qty_calc = position_size_from_risk(account_balance, risk_percent, entry_price_sidebar, stop_loss_sidebar)
            st.success(f"📊 Suggested Quantity: {qty_calc}")
        else:
            qty_calc = 0
            st.info("Enter valid entry and stop-loss.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    sl_log = st.empty()  # Log area for SL updates
    df_open_positions = load_open_positions(session)
    trailing_stop_loss(session, log_slot=sl_log)  # Update trailing stops
    df_manual_closed = load_closed_manual_trades(session)
    df_trades = load_trades()
    df_bot_open, df_bot_closed = split_bot_trades(df_trades)
    
    # Ensure columns exist
    if "Realized PnL ($)" not in df_manual_closed.columns:
        df_manual_closed["Realized PnL ($)"] = 0
    if "Realized PnL ($)" not in df_bot_closed.columns:
        df_bot_closed["Realized PnL ($)"] = 0
    
    # Calculate PnL
    open_pnl = df_open_positions["PnL ($)"].sum() if not df_open_positions.empty else 0
    closed_pnl = df_bot_closed["Realized PnL ($)"].sum() + df_manual_closed["Realized PnL ($)"].sum()
    total_pnl = open_pnl + closed_pnl
    
    # Calculate additional metrics
    total_positions = len(df_open_positions) if not df_open_positions.empty else 0
    risk_locked = check_max_daily_loss(df_bot_closed, df_manual_closed)
    
    # Mode colors
    mode_colors = {
        "Scalping": "#00ffcc",
        "Swing": "#ffaa00",
        "Momentum": "#ff4d4d"
    }
    
    # Check scanner status
    scanner_active = True  # You can set this based on your scanner settings
    
    # =======================
    # STATUS HEADER
    # =======================
    current_time = datetime.now()
    st.markdown(f"""
    <div class="blur-card" style="padding: 8px; margin-bottom: 1rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem;">
            <div style="color: #00fff5;">🕐 {current_time.strftime("%H:%M:%S")}</div>
            <div>📅 {current_time.strftime("%B %d, %Y")}</div>
            <div>🔄 {refresh_choice}</div>
            <div>💰 PnL: <span style="color: {'#00d87f' if total_pnl >= 0 else '#ff4d4d'};">${total_pnl:,.2f}</span></div>
            <div>⚙️ <span style="color: {mode_colors[mode]};">{mode}</span></div>
            <div>📡 Scanner: {'🟢' if scanner_active else '🔴'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =======================
    # CONTROL PANEL
    # =======================
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("🎛️ Control Panel")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        pnl_color = "#00d87f" if total_pnl >= 0 else "#ff4d4d"
        st.markdown(f"""
        <div style="text-align: center; background-color: rgba(0, 255, 245, 0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0; color: #aaa;">💰 Total PnL</h4>
            <h2 style="margin: 5px 0 0 0; color: {pnl_color};">${total_pnl:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; background-color: rgba(0, 255, 245, 0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0; color: #aaa;">⚙️ Trading Mode</h4>
            <h2 style="margin: 5px 0 0 0; color: {mode_colors[mode]};">{mode}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center; background-color: rgba(0, 255, 245, 0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0; color: #aaa;">📈 Open Positions</h4>
            <h2 style="margin: 5px 0 0 0;">{total_positions}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        risk_color = "#ff4d4d" if risk_locked else "#00d87f"
        risk_status = "LOCKED" if risk_locked else "OK"
        risk_icon = "🚫" if risk_locked else "✅"
        st.markdown(f"""
        <div style="text-align: center; background-color: rgba(0, 255, 245, 0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0; color: #aaa;">🛡️ Risk Status</h4>
            <h2 style="margin: 5px 0 0 0; color: {risk_color};">{risk_icon} {risk_status}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        scanner_color = "#00fff5" if scanner_active else "#888888"
        st.markdown(f"""
        <div style="text-align: center; background-color: rgba(0, 255, 245, 0.1); padding: 15px; border-radius: 8px;">
            <h4 style="margin: 0; color: #aaa;">📡 Scanner</h4>
            <h2 style="margin: 5px 0 0 0; color: {scanner_color};">{'🟢 ON' if scanner_active else '🔴 OFF'}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Banner
    if risk_locked and not st.session_state.get("override_risk_lock"):
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
    
    # Log daily stats
    log_daily_pnl_split(df_bot_closed, df_manual_closed)
    
    # =======================
    # ENHANCED MAIN TABS (5 instead of 8)
    # =======================
    main_tabs = [
        "📊 Trading Overview",
        "🤖 Bot Trading",
        "👤 Manual Trading",
        "📈 Analytics & Charts",
        "🚀 Execute & AI"
    ]
    
    # Handle selected tool from sidebar
    if st.session_state.current_tool:
        if st.session_state.current_tool in ["📆 Daily PnL", "📈 Performance Trends", "📊 Advanced Analytics"]:
            st.session_state.active_tab_index = 3  # Analytics tab
        elif st.session_state.current_tool in ["📡 On-Chain Data", "📆 Filter by Date"]:
            st.session_state.active_tab_index = 3  # Analytics tab
        elif st.session_state.current_tool == "Signal Scanner":
            st.session_state.active_tab_index = 3  # Analytics tab
        elif st.session_state.current_tool == "Crypto News":
            st.session_state.active_tab_index = 3  # Analytics tab
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(main_tabs)
    
    # =======================
    # TAB 1: TRADING OVERVIEW
    # =======================
    with tab1:
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📊 Trading Overview")
        
        # Quick Summary Stats
        today = pd.Timestamp.now().date()
        
        # Calculate today's stats with error handling
        try:
            # Ensure timestamp columns exist
            if not df_bot_closed.empty and "timestamp" in df_bot_closed.columns:
                df_bot_closed["timestamp"] = pd.to_datetime(df_bot_closed["timestamp"], errors='coerce')
                bot_today = df_bot_closed[df_bot_closed["timestamp"].dt.date == today]
            else:
                bot_today = pd.DataFrame()
            
            if not df_manual_closed.empty and "timestamp" in df_manual_closed.columns:
                df_manual_closed["timestamp"] = pd.to_datetime(df_manual_closed["timestamp"], errors='coerce')
                manual_today = df_manual_closed[df_manual_closed["timestamp"].dt.date == today]
            else:
                manual_today = pd.DataFrame()
            
            today_bot_trades = len(bot_today)
            today_manual_trades = len(manual_today)
            today_total_trades = today_bot_trades + today_manual_trades
            
            today_bot_pnl = bot_today["Realized PnL ($)"].sum() if not bot_today.empty else 0
            today_manual_pnl = manual_today["Realized PnL ($)"].sum() if not manual_today.empty else 0
            today_total_pnl = today_bot_pnl + today_manual_pnl
            
            today_bot_wins = len(bot_today[bot_today["Realized PnL ($)"] > 0]) if not bot_today.empty else 0
            today_manual_wins = len(manual_today[manual_today["Realized PnL ($)"] > 0]) if not manual_today.empty else 0
            today_total_wins = today_bot_wins + today_manual_wins
            
            today_winrate = (today_total_wins / today_total_trades * 100) if today_total_trades > 0 else 0
        except Exception as e:
            st.error(f"❌ Error calculating today's stats: {e}")
            today_total_trades = today_total_pnl = today_winrate = 0
        
        # Today's Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Today's Trades", today_total_trades)
        with col2:
            st.metric("Today's PnL", f"${today_total_pnl:.2f}", 
                     delta=f"${today_total_pnl:.2f}" if today_total_pnl != 0 else None)
        with col3:
            st.metric("Win Rate", f"{today_winrate:.1f}%")
        with col4:
            avg_trade = today_total_pnl / today_total_trades if today_total_trades > 0 else 0
            st.metric("Avg Trade", f"${avg_trade:.2f}")
        
        # Sub-tabs within Trading Overview
        overview_tabs = st.tabs(["📋 Recent Trades", "📊 Summary Stats", "📈 Quick Charts"])
        
        with overview_tabs[0]:
            st.subheader("🕐 Last 20 Trades")
            
            # Combine recent trades from both sources
            all_recent_trades = []
            
            # Add bot trades
            if not df_bot_closed.empty:
                bot_trades = df_bot_closed.tail(10).copy()
                bot_trades["Trade Type"] = "Bot"
                # Ensure all required columns exist
                required_cols = ["timestamp", "symbol", "side", "qty", "Realized PnL ($)", "Trade Type"]
                for col in required_cols:
                    if col not in bot_trades.columns:
                        bot_trades[col] = ""
                all_recent_trades.append(bot_trades[required_cols])
            
            # Add manual trades
            if not df_manual_closed.empty:
                manual_trades = df_manual_closed.tail(10).copy()
                manual_trades["Trade Type"] = "Manual"
                # Standardize column names
                if "Symbol" in manual_trades.columns:
                    manual_trades = manual_trades.rename(columns={"Symbol": "symbol", "Side": "side", "Size": "qty"})
                # Ensure all required columns exist
                for col in required_cols:
                    if col not in manual_trades.columns:
                        manual_trades[col] = ""
                all_recent_trades.append(manual_trades[required_cols])
            
            if all_recent_trades:
                recent_combined = pd.concat(all_recent_trades)
                # Sort by timestamp if column exists and has valid data
                if "timestamp" in recent_combined.columns:
                    recent_combined["timestamp"] = pd.to_datetime(recent_combined["timestamp"], errors='coerce')
                    recent_combined = recent_combined.sort_values("timestamp", ascending=False)
                recent_combined = recent_combined.head(20)
                st.dataframe(recent_combined, use_container_width=True)
            else:
                st.info("No recent trades to display")
        
        with overview_tabs[1]:
            st.subheader("📊 Today's Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Bot Performance")
                st.metric("Bot Trades", today_bot_trades)
                st.metric("Bot PnL", f"${today_bot_pnl:.2f}")
                bot_winrate = (today_bot_wins / today_bot_trades * 100) if today_bot_trades > 0 else 0
                st.metric("Bot Win Rate", f"{bot_winrate:.1f}%")
            
            with col2:
                st.markdown("### 👤 Manual Performance")
                st.metric("Manual Trades", today_manual_trades)
                st.metric("Manual PnL", f"${today_manual_pnl:.2f}")
                manual_winrate = (today_manual_wins / today_manual_trades * 100) if today_manual_trades > 0 else 0
                st.metric("Manual Win Rate", f"{manual_winrate:.1f}%")
        
        with overview_tabs[2]:
            st.subheader("📈 Today's Performance Chart")
            
            # Create a simple PnL progression chart for today
            if today_total_trades > 0:
                # Combine today's trades for progression chart
                today_trades = []
                if not bot_today.empty and "timestamp" in bot_today.columns and "Realized PnL ($)" in bot_today.columns:
                    today_trades.append(bot_today[["timestamp", "Realized PnL ($)"]])
                if not manual_today.empty and "timestamp" in manual_today.columns and "Realized PnL ($)" in manual_today.columns:
                    today_trades.append(manual_today[["timestamp", "Realized PnL ($)"]])
                
                if today_trades:
                    combined_today = pd.concat(today_trades)
                    combined_today = combined_today.sort_values("timestamp")
                    combined_today["Cumulative PnL"] = combined_today["Realized PnL ($)"].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=combined_today["timestamp"],
                        y=combined_today["Cumulative PnL"],
                        mode="lines+markers",
                        name="Cumulative PnL",
                        line=dict(color="#00fff5")
                    ))
                    fig.update_layout(
                        title="Today's PnL Progression",
                        xaxis_title="Time",
                        yaxis_title="Cumulative PnL ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trades today to chart")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================
    # TAB 2: BOT TRADING
    # =======================
    with tab2:
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("🤖 Bot Trading")
        
        bot_tabs = st.tabs(["📈 Open Positions", "✅ Closed Trades", "📋 Bot Settings"])
        
        with bot_tabs[0]:
            st.subheader("📈 Bot Open Trades")
            if df_bot_open.empty:
                st.info("No active bot trades.")
            else:
                # Enhanced display with more information
                display_df = df_bot_open.copy()
                # Ensure all required columns exist
                required_cols = ["timestamp", "symbol", "side", "qty", "entry_price", "note"]
                for col in required_cols:
                    if col not in display_df.columns:
                        display_df[col] = ""
                
                display_df["Unrealized PnL ($)"] = ""
                display_df["Unrealized PnL (%)"] = ""
                st.dataframe(display_df, use_container_width=True)
                
                # Quick stats for open positions
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Open Positions", len(df_bot_open))
                with col2:
                    total_qty = df_bot_open["qty"].sum() if "qty" in df_bot_open.columns else 0
                    st.metric("Total Quantity", f"{total_qty:.3f}")
                with col3:
                    avg_entry = df_bot_open["entry_price"].mean() if "entry_price" in df_bot_open.columns else 0
                    st.metric("Avg Entry Price", f"${avg_entry:.2f}")
        
        with bot_tabs[1]:
            st.subheader("✅ Bot Closed Trades")
            if df_bot_closed.empty:
                st.info("No closed bot trades yet.")
            else:
                # Show recent closed trades with enhanced info
                display_df = df_bot_closed.copy()
                if "timestamp" in display_df.columns:
                    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], errors='coerce')
                    display_df = display_df.sort_values("timestamp", ascending=False)
                st.dataframe(display_df.head(50), use_container_width=True)
                
                # Bot performance metrics
                st.subheader("📊 Bot Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_bot_trades = len(df_bot_closed)
                    st.metric("Total Bot Trades", total_bot_trades)
                
                with col2:
                    bot_wins = len(df_bot_closed[df_bot_closed["Realized PnL ($)"] > 0])
                    bot_win_rate = (bot_wins / total_bot_trades * 100) if total_bot_trades > 0 else 0
                    st.metric("Win Rate", f"{bot_win_rate:.2f}%")
                
                with col3:
                    avg_win = df_bot_closed[df_bot_closed["Realized PnL ($)"] > 0]["Realized PnL ($)"].mean()
                    st.metric("Avg Win", f"${avg_win:.2f}" if not pd.isna(avg_win) else "$0.00")
                
                with col4:
                    avg_loss = df_bot_closed[df_bot_closed["Realized PnL ($)"] < 0]["Realized PnL ($)"].mean()
                    st.metric("Avg Loss", f"${avg_loss:.2f}" if not pd.isna(avg_loss) else "$0.00")
        
        with bot_tabs[2]:
            st.subheader("📋 Bot Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Current Settings")
                st.write(f"**Strategy Mode**: {mode}")
                sl_pct, tp_pct, rsi_ob, rsi_os = get_strategy_params(mode)
                st.write(f"**Stop Loss**: {sl_pct}%")
                st.write(f"**Take Profit**: {tp_pct}%")
                st.write(f"**RSI Overbought**: {rsi_ob}")
                st.write(f"**RSI Oversold**: {rsi_os}")
                st.write(f"**Risk Per Trade**: {risk_percent}%")
            
            with col2:
                st.markdown("### Bot Status")
                st.write(f"**Account Balance**: ${account_balance:,.2f}")
                st.write(f"**Risk Locked**: {'Yes' if risk_locked else 'No'}")
                st.write(f"**Override Active**: {'Yes' if st.session_state.get('override_risk_lock') else 'No'}")
                st.write(f"**Daily PnL**: ${today_bot_pnl:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================
    # TAB 3: MANUAL TRADING
    # =======================
    with tab3:
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("👤 Manual Trading")
        
        manual_tabs = st.tabs(["🔥 Open Positions", "✅ Closed Trades", "📊 Performance"])
        
        with manual_tabs[0]:
            st.subheader("🔥 Manual Open Trades")
            
            # Load live manual positions
            try:
                res = session.get_positions(
                    category="linear",
                    settleCoin="USDT",
                    accountType="UNIFIED"
                )
                
                if res.get("retCode") == 0:
                    live_positions = res["result"]["list"]
                    manual_positions = []
                    
                    # Filter out bot positions
                    bot_open_keys = set()
                    if not df_bot_open.empty:
                        for _, row in df_bot_open.iterrows():
                            symbol = row.get("symbol", "")
                            qty = row.get("qty", 0)
                            entry_price = row.get("entry_price", 0)
                            key = f"{symbol}|{qty}|{entry_price}"
                            bot_open_keys.add(key)
                    
                    for pos in live_positions:
                        try:
                            size = float(pos.get("positionValue") or 0)
                            if size > 0:
                                symbol = pos.get("symbol", "")
                                entry_price = float(pos.get("avgPrice", 0))
                                qty = float(pos.get("size", 0))
                                key = f"{symbol}|{qty}|{entry_price}"
                                
                                if key not in bot_open_keys:
                                    manual_positions.append({
                                        "Symbol": symbol,
                                        "Side": pos.get("side", ""),
                                        "Size": size,
                                        "Entry Price": entry_price,
                                        "Mark Price": float(pos.get("markPrice", 0)),
                                        "PnL ($)": float(pos.get("unrealisedPnl", 0)),
                                        "PnL (%)": float(pos.get("unrealisedPnl", 0)) / (size * entry_price) * 100 if size * entry_price > 0 else 0
                                    })
                        except:
                            continue
                    
                    if manual_positions:
                        df_manual_pos = pd.DataFrame(manual_positions)
                        st.dataframe(df_manual_pos, use_container_width=True)
                        
                        # Manual position stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Manual Positions", len(manual_positions))
                        with col2:
                            total_manual_pnl = sum(pos["PnL ($)"] for pos in manual_positions)
                            st.metric("Total Unrealized PnL", f"${total_manual_pnl:.2f}")
                        with col3:
                            avg_manual_pnl = total_manual_pnl / len(manual_positions) if manual_positions else 0
                            st.metric("Avg PnL per Position", f"${avg_manual_pnl:.2f}")
                    else:
                        st.info("No open manual positions found.")
                else:
                    st.error(f"❌ API error: {res.get('retMsg', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error loading manual positions: {e}")
        
        with manual_tabs[1]:
            st.subheader("✅ Manual Closed Trades")
            
            if df_manual_closed.empty:
                st.info("No closed manual trades found.")
            else:
                # Standardize column names for display
                display_df = df_manual_closed.copy()
                # Rename columns if needed
                column_mapping = {
                    "Symbol": "symbol",
                    "Side": "side", 
                    "Size": "qty",
                    "Entry Price": "entry_price",
                    "Exit Price": "exit_price"
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in display_df.columns:
                        display_df = display_df.rename(columns={old_col: new_col})
                
                # Sort by timestamp if available
                if "timestamp" in display_df.columns:
                    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], errors='coerce')
                    display_df = display_df.sort_values("timestamp", ascending=False)
                
                st.dataframe(display_df.head(50), use_container_width=True)
                
                # Manual trading stats
                st.subheader("📊 Manual Trading Stats")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Manual Trades", len(df_manual_closed))
                
                with col2:
                    manual_wins = len(df_manual_closed[df_manual_closed["Realized PnL ($)"] > 0])
                    manual_win_rate = (manual_wins / len(df_manual_closed) * 100) if len(df_manual_closed) > 0 else 0
                    st.metric("Win Rate", f"{manual_win_rate:.2f}%")
                
                with col3:
                    manual_profit = df_manual_closed[df_manual_closed["Realized PnL ($)"] > 0]["Realized PnL ($)"].sum()
                    st.metric("Total Profit", f"${manual_profit:.2f}")
                
                with col4:
                    manual_loss = abs(df_manual_closed[df_manual_closed["Realized PnL ($)"] < 0]["Realized PnL ($)"].sum())
                    st.metric("Total Loss", f"${manual_loss:.2f}")
        
        with manual_tabs[2]:
            st.subheader("📊 Manual vs Bot Performance")
            
            # Create comparison chart
            bot_pnl = df_bot_closed["Realized PnL ($)"].sum() if not df_bot_closed.empty else 0
            manual_pnl = df_manual_closed["Realized PnL ($)"].sum() if not df_manual_closed.empty else 0
            
            bot_trades = len(df_bot_closed)
            manual_trades = len(df_manual_closed)
            
            bot_win_rate = (len(df_bot_closed[df_bot_closed["Realized PnL ($)"] > 0]) / bot_trades * 100) if bot_trades > 0 else 0
            manual_win_rate = (len(df_manual_closed[df_manual_closed["Realized PnL ($)"] > 0]) / manual_trades * 100) if manual_trades > 0 else 0
            
            bot_avg_trade = df_bot_closed["Realized PnL ($)"].mean() if bot_trades > 0 else 0
            manual_avg_trade = df_manual_closed["Realized PnL ($)"].mean() if manual_trades > 0 else 0
            
            comparison_data = {
                "Type": ["Bot", "Manual"],
                "Total PnL": [bot_pnl, manual_pnl],
                "Total Trades": [bot_trades, manual_trades],
                "Win Rate": [bot_win_rate, manual_win_rate],
                "Avg Trade": [bot_avg_trade, manual_avg_trade]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PnL comparison bar chart
                fig = px.bar(comparison_df, x="Type", y="Total PnL", 
                           title="Total PnL Comparison",
                           color="Type",
                           color_discrete_map={"Bot": "#00fff5", "Manual": "#ffaa00"})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Win rate comparison
                fig = px.bar(comparison_df, x="Type", y="Win Rate",
                           title="Win Rate Comparison (%)",
                           color="Type",
                           color_discrete_map={"Bot": "#00fff5", "Manual": "#ffaa00"})
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================
    # TAB 4: ANALYTICS & CHARTS
    # =======================
    with tab4:
        st.markdown('<div class="blur-card">', unsafe_allow_html=True)
        st.subheader("📈 Analytics & Charts")
        
        # Handle tool selection from sidebar
        if st.session_state.current_tool:
            if st.session_state.current_tool == "Signal Scanner":
                render_signal_scanner(mode, account_balance, df_bot_closed, df_manual_closed)
            elif st.session_state.current_tool == "📆 Daily PnL":
                render_daily_pnl()
            elif st.session_state.current_tool == "📈 Performance Trends":
                render_performance_trends()
            elif st.session_state.current_tool == "📊 Advanced Analytics":
                df_daily_pnl = pd.read_csv(DAILY_PNL_SPLIT_FILE) if os.path.exists(DAILY_PNL_SPLIT_FILE) else pd.DataFrame()
                render_advanced_analytics(df_trades, df_daily_pnl)
            elif st.session_state.current_tool == "📆 Filter by Date":
                render_filter_by_date()
            elif st.session_state.current_tool == "📰 Crypto News":
                render_crypto_news()
            elif st.session_state.current_tool == "📡 On-Chain Data":
                render_onchain_data()
            else:
                # Default analytics view
                analytics_tabs = st.tabs(["📊 Growth Curve", "🔍 Tool Selection", "📈 Key Metrics"])
                
                with analytics_tabs[0]:
                    # Growth curve
                    st.subheader("📊 Trading Growth Curve")
                    
                    if df_trades.empty or "take_profit" not in df_trades.columns:
                        st.warning("No bot trades available to plot.")
                    else:
                        try:
                            df_trades["timestamp"] = pd.to_datetime(df_trades.get("timestamp", pd.Timestamp.now()), errors='coerce')
                            df_closed = df_trades[df_trades["take_profit"] != 0].copy()
                            
                            if df_closed.empty:
                                st.info("No closed bot trades found to generate growth curve.")
                            else:
                                df_closed = df_closed.sort_values("timestamp")
                                
                                # Ensure we have the PnL calculation
                                if "Realized PnL ($)" not in df_closed.columns:
                                    df_closed["Realized PnL ($)"] = (
                                        (df_closed["take_profit"] - df_closed["entry_price"]) * df_closed["qty"]
                                        - (df_closed["entry_price"] + df_closed["take_profit"]) * df_closed["qty"] * FEE_RATE
                                    )
                                
                                df_closed["Cumulative PnL"] = df_closed["Realized PnL ($)"].cumsum()
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df_closed["timestamp"],
                                    y=df_closed["Cumulative PnL"],
                                    mode="lines+markers",
                                    name="Cumulative PnL",
                                    line=dict(color="#00fff5", width=2)
                                ))
                                fig.update_layout(
                                    title="Trading Growth Curve",
                                    xaxis_title="Date",
                                    yaxis_title="Cumulative PnL ($)",
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error creating growth curve: {e}")
                
                with analytics_tabs[1]:
                    st.subheader("🔍 Available Analytics Tools")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("### 📊 Analytics")
                        if st.button("📆 Daily PnL Analysis", use_container_width=True):
                            st.session_state.current_tool = "📆 Daily PnL"
                        if st.button("📈 Performance Trends", use_container_width=True):
                            st.session_state.current_tool = "📈 Performance Trends"
                        if st.button("📊 Advanced Analytics", use_container_width=True):
                            st.session_state.current_tool = "📊 Advanced Analytics"
                    
                    with col2:
                        st.markdown("### 📡 Data Sources")
                        if st.button("📡 On-Chain Data", use_container_width=True):
                            st.session_state.current_tool = "📡 On-Chain Data"
                        if st.button("📆 Filter by Date", use_container_width=True):
                            st.session_state.current_tool = "📆 Filter by Date"
                        if st.button("📰 Crypto News", use_container_width=True):
                            st.session_state.current_tool = "📰 Crypto News"
                    
                    with col3:
                        st.markdown("### 🔧 Tools")
                        if st.button("📡 Signal Scanner", use_container_width=True):
                            st.session_state.current_tool = "Signal Scanner"
                        if st.button("🔄 Clear Selection", use_container_width=True):
                            st.session_state.current_tool = None
                
                with analytics_tabs[2]:
                    st.subheader("📈 Key Performance Metrics")
                    
                    # Overall performance metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("#### 🎯 Overall Performance")
                        total_trades = len(df_bot_closed) + len(df_manual_closed)
                        total_pnl = df_bot_closed["Realized PnL ($)"].sum() + df_manual_closed["Realized PnL ($)"].sum()
                        st.metric("Total Trades", total_trades)
                        st.metric("Total Realized PnL", f"${total_pnl:.2f}")
                    
                    with col2:
                        st.markdown("#### 🤖 Bot Performance")
                        bot_total = len(df_bot_closed)
                        bot_pnl = df_bot_closed["Realized PnL ($)"].sum()
                        st.metric("Bot Trades", bot_total)
                        st.metric("Bot PnL", f"${bot_pnl:.2f}")
                    
                    with col3:
                        st.markdown("#### 👤 Manual Performance")
                        manual_total = len(df_manual_closed)
                        manual_pnl = df_manual_closed["Realized PnL ($)"].sum()
                        st.metric("Manual Trades", manual_total)
                        st.metric("Manual PnL", f"${manual_pnl:.2f}")
        else:
            # Default view when no tool is selected
            st.info("Select an analytics tool from the sidebar or tabs above.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================
    # TAB 5: EXECUTE & AI
    # =======================
    with tab5:
        execute_tabs = st.tabs(["🛒 Place Trade", "🧠 Mariah AI", "⚡ Quick Actions"])
        
        with execute_tabs[0]:
            # Place Trade
            st.markdown('<div class="blur-card">', unsafe_allow_html=True)
            st.subheader("🛒 Place Live Trade")
            
            col1, col2 = st.columns(2)
            
            with col1:
                symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT"])
                side = st.selectbox("Side", ["Buy", "Sell"])
                qty = st.number_input("Quantity", min_value=0.001, value=max(0.001, float(qty_calc)), step=0.001, format="%.3f")
            
            with col2:
                # Show current price and calculated values
                try:
                    ticker_res = session.get_tickers(category="linear", symbol=symbol)
                    if ticker_res.get("retCode") == 0:
                        current_price = float(ticker_res["result"]["list"][0]["lastPrice"])
                        st.metric("Current Price", f"${current_price:.2f}")
                        
                        order_value = qty * current_price
                        st.metric("Order Value", f"${order_value:.2f}")
                        
                        if side == "Buy":
                            potential_sl = current_price * (1 - get_strategy_params(mode)[0] / 100)
                        else:
                            potential_sl = current_price * (1 + get_strategy_params(mode)[0] / 100)
                        st.metric("Suggested Stop Loss", f"${potential_sl:.2f}")
                    else:
                        st.warning(f"Could not fetch current price: {ticker_res.get('retMsg')}")
                        current_price = 0
                    
                except Exception as e:
                    st.warning(f"Could not fetch current price: {e}")
                    current_price = 0
            
            if st.button("🚀 Place Market Order", type="primary"):
                try:
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
                    
                    mariah_speak(f"Order executed. {side} {qty} {symbol}.")
                    
                    if st.session_state.get("override_risk_lock"):
                        mariah_speak("Override active. Proceeding with caution.")
                    
                    log_rsi_trade_to_csv(symbol=symbol, side=side, qty=round(qty, 3), 
                                        entry_price=current_price if current_price > 0 else entry_price_sidebar, mode=mode)
                    
                    st.success(f"✅ Order placed: {side} {qty} {symbol}")
                    st.json(order)
                    
                except Exception as e:
                    mariah_speak("Order failed. Check trade parameters.")
                    st.error(f"❌ Order failed: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with execute_tabs[1]:
            # Mariah AI
            st.markdown('<div class="blur-card">', unsafe_allow_html=True)
            st.subheader("🧠 Talk to Mariah")
            
            # Mode display with colors
            st.markdown(f"""
            <div style="background-color: rgba(0, 255, 245, 0.1); padding: 10px; border-radius: 8px; margin-bottom: 1rem;">
                <span style='font-size: 1.1rem; font-weight: 600;'>🚦 Current Strategy Mode: 
                <span style='color: {mode_colors[mode]};'>{mode}</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Current status summary for Mariah
            st.markdown(f"""
            <div style="background-color: rgba(0, 255, 245, 0.05); padding: 10px; border-radius: 8px; margin-bottom: 1rem;">
                <strong>Current Status:</strong><br>
                • Total PnL: ${total_pnl:,.2f}<br>
                • Open Positions: {total_positions}<br>
                • Risk Status: {'🚫 LOCKED' if risk_locked else '✅ OK'}<br>
                • Today's Trades: {today_total_trades}
            </div>
            """, unsafe_allow_html=True)
            
            # Chat interface
            user_input = st.chat_input("Ask Mariah anything...", key="mariah_chat_input")
            
            if user_input:
                st.chat_message("user").markdown(user_input)
                override_on = st.session_state.get("override_risk_lock", False)
                response = get_mariah_reply(user_input, open_pnl, closed_pnl, override_on)
                st.chat_message("assistant").markdown(response)
                mariah_speak(response)
            
            st.markdown("---")
            st.markdown("🎙 Or press below to speak:")
            
            if st.button("🎙 Speak to Mariah"):
                voice_input = listen_to_user()
                
                if voice_input:
                    st.chat_message("user").markdown(voice_input)
                    override_on = st.session_state.get("override_risk_lock", False)
                    response = get_mariah_reply(voice_input, open_pnl, closed_pnl, override_on)
                    st.chat_message("assistant").markdown(response)
                    mariah_speak(response)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with execute_tabs[2]:
            # Quick Actions
            st.markdown('<div class="blur-card">', unsafe_allow_html=True)
            st.subheader("⚡ Quick Actions")
            
            # Handle quick actions from sidebar
            if st.session_state.get("quick_action"):
                action = st.session_state.quick_action
                symbol = st.session_state.get("quick_symbol", "BTCUSDT")
                
                st.success(f"Quick {action.upper()} triggered for {symbol}!")
                
                # Reset quick action
                st.session_state.quick_action = None
                
                # You can add actual execution logic here
                st.write(f"Execute {action} for {symbol} with current settings:")
                st.write(f"- Mode: {mode}")
                st.write(f"- Risk %: {risk_percent}%")
                st.write(f"- Account Balance: ${account_balance:,.2f}")
            
            # Quick preset configurations
            st.subheader("📋 Quick Preset Configs")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("⚡ Scalp Mode", use_container_width=True):
                    st.session_state.quick_mode = "Scalping"
                    st.success("Switched to Scalping mode!")
            
            with col2:
                if st.button("📈 Swing Mode", use_container_width=True):
                    st.session_state.quick_mode = "Swing"
                    st.success("Switched to Swing mode!")
            
            with col3:
                if st.button("🚀 Momentum Mode", use_container_width=True):
                    st.session_state.quick_mode = "Momentum"
                    st.success("Switched to Momentum mode!")
            
            # Emergency actions
            st.subheader("🚨 Emergency Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🛑 Close All Positions", type="secondary", use_container_width=True):
                    st.warning("This will close ALL open positions. Confirm to proceed.")
                    if st.button("⚠️ CONFIRM CLOSE ALL", use_container_width=True):
                        st.error("Close all positions function would execute here.")
            
            with col2:
                if st.button("⏸️ Pause All Bots", type="secondary", use_container_width=True):
                    st.warning("This will pause all automated trading.")
                    if st.button("⚠️ CONFIRM PAUSE", use_container_width=True):
                        st.error("Pause bots function would execute here.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # =======================
    # BOTTOM STATUS BAR
    # =======================
    st.markdown(f"""
    <div style="position: fixed; bottom: 0; left: 0; right: 0; 
                background-color: rgba(15, 15, 35, 0.95); 
                padding: 8px; border-top: 1px solid #00fff5;
                backdrop-filter: blur(10px);
                z-index: 999;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #ccc;">
            <div>🔄 Last Update: {current_time.strftime("%H:%M:%S")}</div>
            <div>📊 Active Trades: {total_positions}</div>
            <div>💰 P&L: <span style="color: {'#00d87f' if total_pnl >= 0 else '#ff4d4d'};">${total_pnl:,.2f}</span></div>
            <div>🛡️ Risk: {"LOCKED" if risk_locked else "OK"}</div>
            <div>⚙️ {mode}</div>
            <div>📡 Scanner: {'🟢' if scanner_active else '🔴'}</div>
        </div>
    </div>
    <div style="height: 40px;"></div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()