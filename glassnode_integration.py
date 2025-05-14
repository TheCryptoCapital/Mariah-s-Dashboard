"""
Advanced On-Chain Data Integration with Glassnode
"""

import requests
import os
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class GlassnodeAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GLASSNODE_API_KEY")
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
    def get_exchange_flows(self, asset="BTC", timeframe="7d"):
        """Get exchange inflow/outflow data"""
        
        # Get inflows
        inflow_url = f"{self.base_url}/transactions/transfers_to_exchanges_sum"
        outflow_url = f"{self.base_url}/transactions/transfers_from_exchanges_sum"
        
        params = {
            "a": asset,
            "api_key": self.api_key,
            "s": self._get_timestamp(timeframe),
            "i": "24h"
        }
        
        try:
            inflow_response = requests.get(inflow_url, params=params, timeout=10)
            outflow_response = requests.get(outflow_url, params=params, timeout=10)
            
            return {
                "inflows": inflow_response.json() if inflow_response.status_code == 200 else [],
                "outflows": outflow_response.json() if outflow_response.status_code == 200 else []
            }
        except Exception as e:
            return {"error": str(e), "inflows": [], "outflows": []}
    
    def get_whale_activity(self, asset="BTC", threshold="1m"):
        """Get whale transaction activity (transactions > threshold USD)"""
        url = f"{self.base_url}/transactions/count_greater_than_{threshold}_usd"
        params = {
            "a": asset,
            "api_key": self.api_key,
            "s": self._get_timestamp("7d"),
            "i": "24h"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return {"error": str(e)}
    
    def get_mvrv_ratio(self, asset="BTC"):
        """Get Market Value to Realized Value ratio"""
        url = f"{self.base_url}/market/mvrv"
        params = {
            "a": asset,
            "api_key": self.api_key,
            "s": self._get_timestamp("30d"),
            "i": "24h"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json() if response.status_code == 200 else []
            return data[-1]["v"] if data else None
        except Exception as e:
            return None
    
    def get_hodl_waves(self, asset="BTC"):
        """Get HODLer behavior data"""
        url = f"{self.base_url}/supply/hodl_waves"
        params = {
            "a": asset,
            "api_key": self.api_key,
            "i": "24h"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return {"error": str(e)}
    
    def get_exchange_balances(self, asset="BTC"):
        """Get total exchange balances"""
        url = f"{self.base_url}/distribution/balance_exchanges"
        params = {
            "a": asset,
            "api_key": self.api_key,
            "s": self._get_timestamp("30d"),
            "i": "24h"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json() if response.status_code == 200 else []
        except Exception as e:
            return {"error": str(e)}
    
    def get_network_activity(self, asset="BTC"):
        """Get network activity metrics"""
        # Active addresses
        active_url = f"{self.base_url}/addresses/active_count"
        # Transaction count
        tx_url = f"{self.base_url}/transactions/count"
        
        params = {
            "a": asset,
            "api_key": self.api_key,
            "s": self._get_timestamp("30d"),
            "i": "24h"
        }
        
        try:
            active_response = requests.get(active_url, params=params, timeout=10)
            tx_response = requests.get(tx_url, params=params, timeout=10)
            
            return {
                "active_addresses": active_response.json() if active_response.status_code == 200 else [],
                "transaction_count": tx_response.json() if tx_response.status_code == 200 else []
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_timestamp(self, timeframe):
        """Convert timeframe to Unix timestamp"""
        now = datetime.now()
        if timeframe == "24h":
            delta = timedelta(days=1)
        elif timeframe == "7d":
            delta = timedelta(days=7)
        elif timeframe == "30d":
            delta = timedelta(days=30)
        else:
            delta = timedelta(days=1)
        
        return int((now - delta).timestamp())

def get_enhanced_onchain_signal(symbol="BTCUSDT"):
    """Enhanced on-chain signal using Glassnode data"""
    
    # Remove 'USDT' suffix for Glassnode API
    asset = symbol.replace('USDT', '').replace('USD', '')
    
    # Check if we have API key
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        # Fallback to mock data if no API key
        import random
        signals = ['buy', 'sell', 'hold']
        confidences = [0.6, 0.7, 0.8, 0.9]
        
        mock_signal = random.choice(signals)
        mock_confidence = random.choice(confidences)
        
        reasons = {
            'buy': 'Mock: Large outflows from exchanges detected',
            'sell': 'Mock: Whale accumulation on exchanges',
            'hold': 'Mock: On-chain metrics showing consolidation'
        }
        
        return {
            'signal': mock_signal,
            'confidence': mock_confidence,
            'reason': reasons[mock_signal] + " (Demo Mode - No API Key)"
        }
    
    glassnode = GlassnodeAPI(api_key)
    
    try:
        # Get multiple on-chain metrics
        exchange_flows = glassnode.get_exchange_flows(asset)
        whale_activity = glassnode.get_whale_activity(asset, "1m")
        mvrv_ratio = glassnode.get_mvrv_ratio(asset)
        exchange_balances = glassnode.get_exchange_balances(asset)
        
        signals = []
        confidence_factors = []
        reasons = []
        
        # Analyze exchange flows
        if exchange_flows.get("inflows") and exchange_flows.get("outflows"):
            inflows = exchange_flows["inflows"]
            outflows = exchange_flows["outflows"]
            
            if inflows and outflows:
                recent_inflow = inflows[-1]["v"] if inflows else 0
                recent_outflow = outflows[-1]["v"] if outflows else 0
                
                # Calculate flow ratio (outflows/inflows)
                flow_ratio = recent_outflow / recent_inflow if recent_inflow > 0 else 1
                
                if flow_ratio > 1.5:  # More outflows than inflows
                    signals.append("buy")
                    confidence_factors.append(0.7)
                    reasons.append(f"Large exchange outflows (ratio: {flow_ratio:.2f})")
                elif flow_ratio < 0.5:  # More inflows than outflows
                    signals.append("sell")
                    confidence_factors.append(0.7)
                    reasons.append(f"Large exchange inflows (ratio: {flow_ratio:.2f})")
        
        # Analyze MVRV ratio
        if mvrv_ratio:
            if mvrv_ratio > 3.0:
                signals.append("sell")
                confidence_factors.append(0.8)
                reasons.append(f"MVRV ratio high ({mvrv_ratio:.2f}), potential top")
            elif mvrv_ratio < 1.0:
                signals.append("buy")
                confidence_factors.append(0.8)
                reasons.append(f"MVRV ratio low ({mvrv_ratio:.2f}), potential bottom")
        
        # Analyze exchange balances trend
        if exchange_balances and len(exchange_balances) > 7:
            current_balance = exchange_balances[-1]["v"]
            week_ago_balance = exchange_balances[-7]["v"]
            balance_change = (current_balance - week_ago_balance) / week_ago_balance
            
            if balance_change < -0.05:  # 5% decrease in exchange balances
                signals.append("buy")
                confidence_factors.append(0.6)
                reasons.append(f"Exchange balances decreasing ({balance_change:.1%})")
            elif balance_change > 0.05:  # 5% increase in exchange balances
                signals.append("sell")
                confidence_factors.append(0.6)
                reasons.append(f"Exchange balances increasing ({balance_change:.1%})")
        
        # Analyze whale activity
        if whale_activity and not whale_activity.get("error"):
            recent_whale_txs = whale_activity[-1]["v"] if whale_activity else 0
            avg_whale_txs = sum([d["v"] for d in whale_activity[-7:]]) / 7 if len(whale_activity) >= 7 else recent_whale_txs
            
            if recent_whale_txs > avg_whale_txs * 1.5:
                # High whale activity could be either direction
                signals.append("hold")
                confidence_factors.append(0.5)
                reasons.append(f"Increased whale activity ({recent_whale_txs} vs avg {avg_whale_txs:.0f})")
        
        # Determine final signal
        if not signals:
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'reason': 'On-chain metrics neutral'
            }
        
        # Count signal types
        buy_signals = signals.count("buy")
        sell_signals = signals.count("sell")
        hold_signals = signals.count("hold")
        
        if buy_signals > max(sell_signals, hold_signals):
            final_signal = "buy"
            buy_indices = [i for i, sig in enumerate(signals) if sig == "buy"]
            confidence = sum([confidence_factors[i] for i in buy_indices]) / len(buy_indices)
            reason = "; ".join([reasons[i] for i in buy_indices])
        elif sell_signals > max(buy_signals, hold_signals):
            final_signal = "sell"
            sell_indices = [i for i, sig in enumerate(signals) if sig == "sell"]
            confidence = sum([confidence_factors[i] for i in sell_indices]) / len(sell_indices)
            reason = "; ".join([reasons[i] for i in sell_indices])
        else:
            final_signal = "hold"
            confidence = 0.5
            reason = "Mixed on-chain signals"
        
        return {
            'signal': final_signal,
            'confidence': min(confidence, 0.9),
            'reason': reason
        }
        
    except Exception as e:
        return {
            'signal': 'hold',
            'confidence': 0.3,
            'reason': f'On-chain analysis error: {str(e)}'
        }

def render_enhanced_onchain_data():
    """Enhanced on-chain data display with Glassnode integration"""
    st.markdown('<div class="blur-card">', unsafe_allow_html=True)
    st.subheader("⛓️ Advanced On-Chain Analysis")
    
    # Check for API key
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        st.warning("⚠️ Glassnode API key not found. Add GLASSNODE_API_KEY to your .env file.")
        st.info("Get your free API key at: https://glassnode.com/")
        
        # Show basic ETH data instead
        from onchain_feed import get_eth_gas, get_block_info
        
        st.subheader("📡 ETH Gas & Block Info (Etherscan)")
        
        gas = get_eth_gas()
        block = get_block_info()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🐌 Safe Gas", f"{gas['low']} Gwei")
        with col2:
            st.metric("⚡ Standard Gas", f"{gas['avg']} Gwei")
        with col3:
            st.metric("🚀 Fast Gas", f"{gas['high']} Gwei") 
        with col4:
            st.metric("📦 Latest Block", f"#{block}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Initialize Glassnode API
    glassnode = GlassnodeAPI(api_key)
    
    # Create tabs for different on-chain metrics
    onchain_tabs = st.tabs([
        "📊 Exchange Flows", 
        "🐋 Whale Activity", 
        "📈 Market Metrics", 
        "💎 HODLer Analysis",
        "🌐 Network Activity"
    ])
    
    # Select asset
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_asset = st.selectbox(
            "Select Asset",
            ["BTC", "ETH"],
            key="onchain_asset_selector"
        )
    
    with onchain_tabs[0]:
        st.subheader("📊 Exchange Flow Analysis")
        
        # Get exchange flow data
        with st.spinner("Loading exchange flow data..."):
            flows = glassnode.get_exchange_flows(selected_asset)
            balances = glassnode.get_exchange_balances(selected_asset)
        
        if flows.get("inflows") and flows.get("outflows"):
            # Process data for visualization
            inflows_df = pd.DataFrame(flows["inflows"])
            outflows_df = pd.DataFrame(flows["outflows"])
            
            if not inflows_df.empty and not outflows_df.empty:
                # Convert timestamps
                inflows_df['date'] = pd.to_datetime(inflows_df['t'], unit='s')
                outflows_df['date'] = pd.to_datetime(outflows_df['t'], unit='s')
                
                # Create flow chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=inflows_df['date'],
                    y=inflows_df['v'],
                    mode='lines+markers',
                    name='Inflows',
                    line=dict(color='red'),
                    fill='tonexty'
                ))
                
                fig.add_trace(go.Scatter(
                    x=outflows_df['date'],
                    y=outflows_df['v'],
                    mode='lines+markers',
                    name='Outflows',
                    line=dict(color='green'),
                    fill='tozeroy'
                ))
                
                fig.update_layout(
                    title=f"{selected_asset} Exchange Flows (7 Days)",
                    xaxis_title="Date",
                    yaxis_title=f"Amount ({selected_asset})",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Current metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_inflow = inflows_df['v'].iloc[-1] if not inflows_df.empty else 0
                    st.metric("24h Inflows", f"{current_inflow:,.0f} {selected_asset}")
                
                with col2:
                    current_outflow = outflows_df['v'].iloc[-1] if not outflows_df.empty else 0
                    st.metric("24h Outflows", f"{current_outflow:,.0f} {selected_asset}")
                
                with col3:
                    net_flow = current_outflow - current_inflow
                    st.metric("Net Flow", f"{net_flow:,.0f} {selected_asset}", 
                             delta=f"{net_flow:,.0f}")
        
        else:
            st.warning("Unable to load exchange flow data. Check your Glassnode API key.")
    
    with onchain_tabs[1]:
        st.subheader("🐋 Whale Activity Monitor")
        
        # Get whale activity
        with st.spinner("Loading whale activity data..."):
            whale_1m = glassnode.get_whale_activity(selected_asset, "1m")
            whale_10m = glassnode.get_whale_activity(selected_asset, "10m")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if whale_1m and not whale_1m.get("error"):
                df_1m = pd.DataFrame(whale_1m)
                if not df_1m.empty:
                    df_1m['date'] = pd.to_datetime(df_1m['t'], unit='s')
                    
                    # Current whale activity
                    current_whale_txs = df_1m['v'].iloc[-1] if not df_1m.empty else 0
                    avg_whale_txs = df_1m['v'].mean()
                    
                    st.metric(
                        "Transactions >$1M (24h)", 
                        f"{current_whale_txs:,.0f}",
                        delta=f"{((current_whale_txs / avg_whale_txs - 1) * 100):+.1f}%" if avg_whale_txs > 0 else None
                    )
                    
                    # Whale activity chart
                    fig = px.bar(df_1m.tail(7), x='date', y='v', 
                               title=f"{selected_asset} Whale Transactions >$1M")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if whale_10m and not whale_10m.get("error"):
                df_10m = pd.DataFrame(whale_10m)
                if not df_10m.empty:
                    df_10m['date'] = pd.to_datetime(df_10m['t'], unit='s')
                    
                    current_mega_whale = df_10m['v'].iloc[-1] if not df_10m.empty else 0
                    
                    st.metric("Transactions >$10M (24h)", f"{current_mega_whale:,.0f}")
                    
                    # Mega whale activity chart
                    fig = px.bar(df_10m.tail(7), x='date', y='v',
                               title=f"{selected_asset} Mega Whale Transactions >$10M")
                    st.plotly_chart(fig, use_container_width=True)
    
    with onchain_tabs[2]:
        st.subheader("📈 Market Health Metrics")
        
        # Get market metrics
        with st.spinner("Loading market metrics..."):
            mvrv = glassnode.get_mvrv_ratio(selected_asset)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if mvrv is not None:
                st.metric("MVRV Ratio", f"{mvrv:.2f}")
                
                # MVRV interpretation
                if mvrv > 3.0:
                    st.error("⚠️ Potentially overbought")
                elif mvrv < 1.0:
                    st.success("✅ Potentially oversold")
                else:
                    st.info("➡️ Normal range")
        
        with col2:
            # Placeholder for realized cap
            st.metric("Realized Cap", "Loading...")
        
        with col3:
            # Placeholder for network value
            st.metric("Network Value", "Loading...")
        
        # MVRV trend chart would go here
        st.info("💡 MVRV Ratio: Market Value to Realized Value ratio. Values >3 suggest tops, <1 suggest bottoms.")
    
    with onchain_tabs[3]:
        st.subheader("💎 HODLer Behavior Analysis")
        
        # Get HODLer data
        with st.spinner("Loading HODLer data..."):
            hodl_data = glassnode.get_hodl_waves(selected_asset)
        
        if hodl_data and not hodl_data.get("error"):
            st.info("HODLer wave analysis shows how long coins have been held without moving.")
            # HODLer charts would be implemented here
        else:
            st.warning("HODLer data temporarily unavailable.")
    
    with onchain_tabs[4]:
        st.subheader("🌐 Network Activity")
        
        # Get network activity
        with st.spinner("Loading network activity..."):
            network_data = glassnode.get_network_activity(selected_asset)
        
        if network_data and not network_data.get("error"):
            active_addresses = network_data.get("active_addresses", [])
            tx_count = network_data.get("transaction_count", [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if active_addresses:
                    df_active = pd.DataFrame(active_addresses)
                    df_active['date'] = pd.to_datetime(df_active['t'], unit='s')
                    
                    current_active = df_active['v'].iloc[-1] if not df_active.empty else 0
                    st.metric("Active Addresses (24h)", f"{current_active:,.0f}")
                    
                    # Active addresses chart
                    fig = px.line(df_active.tail(30), x='date', y='v',
                                title=f"{selected_asset} Active Addresses Trend")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if tx_count:
                    df_tx = pd.DataFrame(tx_count)
                    df_tx['date'] = pd.to_datetime(df_tx['t'], unit='s')
                    
                    current_tx = df_tx['v'].iloc[-1] if not df_tx.empty else 0
                    st.metric("Transaction Count (24h)", f"{current_tx:,.0f}")
                    
                    # Transaction count chart
                    fig = px.line(df_tx.tail(30), x='date', y='v',
                                title=f"{selected_asset} Transaction Count Trend")
                    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)