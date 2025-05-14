# Multi-Agent Trading System
# Add this to a new file: multi_agent_system.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import json

# Add this import to fix the NameError
from mariah_rl import MariahRLAgent

class AgentType(Enum):
    """Different types of trading agents"""
    SCALPER = "scalper"
    SWING = "swing"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    NEWS_SENTIMENT = "news_sentiment"
    WHALE_TRACKER = "whale_tracker"
    VOLATILITY = "volatility"
    PAIRS_TRADER = "pairs_trader"
    GRID_TRADER = "grid_trader"
    TREND_FOLLOWER = "trend_follower"
    CONTRARIAN = "contrarian"
    MASTER_COORDINATOR = "master_coordinator"

@dataclass
class TradingSignal:
    """Standard trading signal format"""
    symbol: str
    action: str  # buy, sell, hold
    confidence: float
    position_size: float
    agent_id: str
    timestamp: pd.Timestamp
    reasoning: str
    metadata: Dict = None

@dataclass
class MarketContext:
    """Current market context for all agents"""
    timestamp: pd.Timestamp
    market_state: str  # trending, ranging, volatile, calm
    vix: float
    sentiment: float
    volume_profile: Dict
    active_symbols: List[str]

class BaseAgent:
    """Base class for all trading agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, config: Dict):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        
        # RL components
        self.rl_enabled = config.get('rl_enabled', True)
        if self.rl_enabled:
            self.rl_agent = MariahRLAgent(
                state_dim=config.get('state_dim', 50),
                action_dim=config.get('action_dim', 5),  # hold, buy_small, buy_large, sell_small, sell_large
                lr=config.get('learning_rate', 3e-4)
            )
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
    async def analyze(self, market_data: Dict, context: MarketContext) -> List[TradingSignal]:
        """Main analysis method - to be implemented by each agent"""
        raise NotImplementedError
    
    async def update_model(self, trade_result: Dict):
        """Update RL model based on trade results"""
        if self.rl_enabled and trade_result:
            # Extract learning from trade result
            reward = self._calculate_reward(trade_result)
            # Store experience and update model
            # Implementation depends on specific agent requirements
            pass
    
    def _calculate_reward(self, trade_result: Dict) -> float:
        """Calculate reward for RL training"""
        pnl = trade_result.get('pnl', 0)
        risk_adjusted_return = pnl / trade_result.get('risk', 1)
        
        # Add penalties/bonuses based on agent-specific goals
        if self.agent_type == AgentType.SCALPER:
            # Scalper gets bonus for quick profits
            time_held = trade_result.get('time_held_minutes', 0)
            if time_held < 5 and pnl > 0:
                reward = risk_adjusted_return * 1.5
            else:
                reward = risk_adjusted_return
        else:
            reward = risk_adjusted_return
            
        return reward

    def get_performance_summary(self) -> Dict:
        """Get agent performance summary"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'metrics': self.performance_metrics.copy(),
            'recent_signals': len(self.trade_history)
        }

class ScalperAgent(BaseAgent):
    """High-frequency scalping agent"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.SCALPER, config)
        self.min_profit_pips = config.get('min_profit_pips', 5)
        self.max_hold_time_minutes = config.get('max_hold_time_minutes', 2)
        
    async def analyze(self, market_data: Dict, context: MarketContext) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            if symbol not in context.active_symbols:
                continue
                
            # Look for micro-patterns
            price_data = pd.DataFrame(data[-100:])  # Last 100 candles
            
            # Scalping signals based on order book, spread, momentum
            spread = data[-1]['ask'] - data[-1]['bid']
            if spread < self.config.get('max_spread', 0.0001):
                
                # Quick momentum check
                recent_returns = price_data['close'].pct_change().tail(5)
                momentum = recent_returns.mean()
                
                if abs(momentum) > 0.0002:  # 0.02% momentum
                    action = 'buy' if momentum > 0 else 'sell'
                    confidence = min(abs(momentum) * 1000, 0.95)
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        position_size=self.config.get('position_size', 0.1),
                        agent_id=self.agent_id,
                        timestamp=context.timestamp,
                        reasoning=f"Scalp on {momentum:.4f} momentum",
                        metadata={'spread': spread, 'momentum': momentum}
                    )
                    signals.append(signal)
                    
        return signals

class SwingAgent(BaseAgent):
    """Medium-term swing trading agent"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.SWING, config)
        self.lookback_periods = config.get('lookback_periods', 50)
        
    async def analyze(self, market_data: Dict, context: MarketContext) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            df = pd.DataFrame(data[-self.lookback_periods:])
            
            # Swing signals based on support/resistance and trend
            # RSI divergence
            rsi = self._calculate_rsi(df['close'])
            rsi_divergence = self._check_rsi_divergence(df, rsi)
            
            # Support/Resistance levels
            support, resistance = self._find_support_resistance(df)
            
            # Current price position
            current_price = df['close'].iloc[-1]
            
            if rsi_divergence == 'bullish' and current_price < support * 1.02:
                signal = TradingSignal(
                    symbol=symbol,
                    action='buy',
                    confidence=0.75,
                    position_size=self.config.get('position_size', 0.25),
                    agent_id=self.agent_id,
                    timestamp=context.timestamp,
                    reasoning="Bullish RSI divergence near support",
                    metadata={'rsi': rsi.iloc[-1], 'support': support}
                )
                signals.append(signal)
                
            elif rsi_divergence == 'bearish' and current_price > resistance * 0.98:
                signal = TradingSignal(
                    symbol=symbol,
                    action='sell',
                    confidence=0.75,
                    position_size=self.config.get('position_size', 0.25),
                    agent_id=self.agent_id,
                    timestamp=context.timestamp,
                    reasoning="Bearish RSI divergence near resistance",
                    metadata={'rsi': rsi.iloc[-1], 'resistance': resistance}
                )
                signals.append(signal)
                
        return signals
    
    def _calculate_rsi(self, prices, periods=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _check_rsi_divergence(self, df, rsi):
        # Simplified divergence check
        price_trend = (df['close'].iloc[-1] - df['close'].iloc[-10]) > 0
        rsi_trend = (rsi.iloc[-1] - rsi.iloc[-10]) > 0
        
        if price_trend and not rsi_trend:
            return 'bearish'
        elif not price_trend and rsi_trend:
            return 'bullish'
        return 'none'
    
    def _find_support_resistance(self, df):
        # Simplified support/resistance
        recent_data = df.tail(20)
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        return support, resistance

class MomentumAgent(BaseAgent):
    """Momentum trading agent"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.MOMENTUM, config)
        
    async def analyze(self, market_data: Dict, context: MarketContext) -> List[TradingSignal]:
        signals = []
        
        for symbol, data in market_data.items():
            df = pd.DataFrame(data[-100:])
            
            # Multiple timeframe momentum
            short_momentum = self._calculate_momentum(df['close'], 5)
            medium_momentum = self._calculate_momentum(df['close'], 20)
            long_momentum = self._calculate_momentum(df['close'], 50)
            
            # Volume momentum
            volume_momentum = self._calculate_momentum(df['volume'], 10)
            
            # Combined momentum score
            momentum_score = (
                0.4 * short_momentum +
                0.4 * medium_momentum +
                0.2 * long_momentum
            ) * (1 + volume_momentum * 0.1)  # Volume confirmation
            
            if abs(momentum_score) > self.config.get('momentum_threshold', 0.02):
                action = 'buy' if momentum_score > 0 else 'sell'
                confidence = min(abs(momentum_score) * 25, 0.9)
                
                signal = TradingSignal(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    position_size=min(abs(momentum_score) * 2, 0.5),
                    agent_id=self.agent_id,
                    timestamp=context.timestamp,
                    reasoning=f"Momentum score: {momentum_score:.4f}",
                    metadata={'momentum_score': momentum_score}
                )
                signals.append(signal)
                
        return signals
    
    def _calculate_momentum(self, series, periods):
        """Calculate price momentum over periods"""
        return (series.iloc[-1] / series.iloc[-periods] - 1)

class WhaleTrackerAgent(BaseAgent):
    """Tracks large transactions and whale movements"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.WHALE_TRACKER, config)
        self.whale_threshold = config.get('whale_threshold', 1000000)  # $1M
        
    async def analyze(self, market_data: Dict, context: MarketContext) -> List[TradingSignal]:
        signals = []
        
        # This would integrate with on-chain data to track whale movements
        # For demonstration, using volume spikes as proxy
        
        for symbol, data in market_data.items():
            df = pd.DataFrame(data[-20:])
            
            # Detect unusual volume (whale activity proxy)
            avg_volume = df['volume'].rolling(10).mean().iloc[-2]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > self.config.get('volume_spike_threshold', 3.0):
                # Check if this is accumulation or distribution
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                
                if volume_ratio > 5.0:  # Very large volume
                    if price_change > 0:
                        # Large volume + price up = accumulation
                        signal = TradingSignal(
                            symbol=symbol,
                            action='buy',
                            confidence=0.8,
                            position_size=0.3,
                            agent_id=self.agent_id,
                            timestamp=context.timestamp,
                            reasoning=f"Whale accumulation detected (vol ratio: {volume_ratio:.2f})",
                            metadata={'volume_ratio': volume_ratio, 'price_change': price_change}
                        )
                        signals.append(signal)
                    elif price_change < -0.005:  # Significant price drop
                        # Large volume + price down = distribution
                        signal = TradingSignal(
                            symbol=symbol,
                            action='sell',
                            confidence=0.75,
                            position_size=0.25,
                            agent_id=self.agent_id,
                            timestamp=context.timestamp,
                            reasoning=f"Whale distribution detected (vol ratio: {volume_ratio:.2f})",
                            metadata={'volume_ratio': volume_ratio, 'price_change': price_change}
                        )
                        signals.append(signal)
                        
        return signals

class MasterCoordinator(BaseAgent):
    """Coordinates all agents and makes final decisions"""
    
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, AgentType.MASTER_COORDINATOR, config)
        self.agents: List[BaseAgent] = []
        self.signal_weights = config.get('signal_weights', {})
        
    def add_agent(self, agent: BaseAgent):
        """Add an agent to coordinate"""
        self.agents.append(agent)
        
    async def coordinate_signals(self, all_signals: List[TradingSignal]) -> List[TradingSignal]:
        """Coordinate signals from all agents"""
        # Group signals by symbol
        signals_by_symbol = {}
        for signal in all_signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)
        
        final_signals = []
        
        for symbol, signals in signals_by_symbol.items():
            # Resolve conflicts and combine signals
            combined_signal = self._combine_signals(signals)
            if combined_signal:
                final_signals.append(combined_signal)
                
        return final_signals
    
    def _combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """Combine multiple signals for the same symbol"""
        if not signals:
            return None
            
        # Weight signals by agent type and confidence
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        for signal in signals:
            weight = self.signal_weights.get(signal.agent_id, 1.0)
            weighted_confidence = signal.confidence * weight
            
            if signal.action == 'buy':
                buy_score += weighted_confidence
            elif signal.action == 'sell':
                sell_score += weighted_confidence
                
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight
        
        # Determine final action
        min_threshold = 0.3
        if buy_score > sell_score and buy_score > min_threshold:
            action = 'buy'
            confidence = buy_score
        elif sell_score > buy_score and sell_score > min_threshold:
            action = 'sell'
            confidence = sell_score
        else:
            return None  # No clear signal
        
        # Calculate position size based on consensus strength
        avg_position_size = np.mean([s.position_size for s in signals])
        consensus_strength = abs(buy_score - sell_score)
        final_position_size = avg_position_size * consensus_strength
        
        # Create reasoning summary
        agent_ids = [s.agent_id for s in signals if s.action == action]
        reasoning = f"Consensus from {len(agent_ids)} agents: {', '.join(agent_ids)}"
        
        return TradingSignal(
            symbol=signals[0].symbol,
            action=action,
            confidence=confidence,
            position_size=final_position_size,
            agent_id=self.agent_id,
            timestamp=signals[0].timestamp,
            reasoning=reasoning,
            metadata={
                'buy_score': buy_score,
                'sell_score': sell_score,
                'consensus_strength': consensus_strength,
                'contributing_agents': len(signals)
            }
        )

class MultiAgentTradingSystem:
    """Main system coordinating all 13 agents"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents: List[BaseAgent] = []
        self.coordinator = MasterCoordinator("master", config.get('coordinator', {}))
        self.logger = logging.getLogger("MultiAgentSystem")
        
        # Initialize all 13 agents
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all 13 specialized agents"""
        agent_configs = self.config.get('agents', {})
        
        # Create specialized agents
        agents_to_create = [
            (ScalperAgent, "scalper_1", AgentType.SCALPER),
            (ScalperAgent, "scalper_2", AgentType.SCALPER),
            (SwingAgent, "swing_1", AgentType.SWING),
            (SwingAgent, "swing_2", AgentType.SWING),
            (MomentumAgent, "momentum_1", AgentType.MOMENTUM),
            (MomentumAgent, "momentum_2", AgentType.MOMENTUM),
            (WhaleTrackerAgent, "whale_tracker", AgentType.WHALE_TRACKER),
            # Add more specialized agents here
            # (MeanReversionAgent, "mean_reversion", AgentType.MEAN_REVERSION),
            # (ArbitrageAgent, "arbitrage", AgentType.ARBITRAGE),
            # (NewsSentimentAgent, "news_sentiment", AgentType.NEWS_SENTIMENT),
            # (VolatilityAgent, "volatility", AgentType.VOLATILITY),
            # (PairsTraderAgent, "pairs_trader", AgentType.PAIRS_TRADER),
            # (GridTraderAgent, "grid_trader", AgentType.GRID_TRADER),
        ]
        
        for agent_class, agent_id, agent_type in agents_to_create:
            config = agent_configs.get(agent_id, {})
            agent = agent_class(agent_id, config)
            self.agents.append(agent)
            self.coordinator.add_agent(agent)
            
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    async def run_analysis_cycle(self, market_data: Dict) -> List[TradingSignal]:
        """Run one analysis cycle across all agents"""
        # Create market context
        context = self._create_market_context(market_data)
        
        # Collect signals from all agents in parallel
        tasks = []
        for agent in self.agents:
            task = agent.analyze(market_data, context)
            tasks.append(task)
        
        # Wait for all agents to complete
        agent_signals = await asyncio.gather(*tasks)
        
        # Flatten signal lists
        all_signals = []
        for signals in agent_signals:
            all_signals.extend(signals)
        
        # Coordinate signals
        final_signals = await self.coordinator.coordinate_signals(all_signals)
        
        self.logger.info(f"Collected {len(all_signals)} signals, finalized {len(final_signals)}")
        
        return final_signals
    
    def _create_market_context(self, market_data: Dict) -> MarketContext:
        """Create market context from current data"""
        # Analyze overall market state
        # This is simplified - you'd want more sophisticated analysis
        
        active_symbols = list(market_data.keys())
        
        # Calculate overall volatility
        volatilities = []
        for symbol, data in market_data.items():
            df = pd.DataFrame(data[-20:])
            if len(df) > 1:
                returns = df['close'].pct_change().dropna()
                vol = returns.std()
                volatilities.append(vol)
        
        avg_volatility = np.mean(volatilities) if volatilities else 0
        
        # Determine market state
        if avg_volatility > 0.02:
            market_state = "volatile"
        elif avg_volatility < 0.005:
            market_state = "calm"
        else:
            market_state = "normal"
        
        return MarketContext(
            timestamp=pd.Timestamp.now(),
            market_state=market_state,
            vix=avg_volatility * 100,  # Convert to percentage
            sentiment=0.5,  # Neutral for now
            volume_profile={},
            active_symbols=active_symbols
        )
    
    async def update_agents(self, trade_results: List[Dict]):
        """Update all agents based on trade results"""
        for agent in self.agents:
            for result in trade_results:
                if result.get('agent_id') == agent.agent_id:
                    await agent.update_model(result)
    
    def get_system_performance(self) -> Dict:
        """Get overall system performance"""
        agent_performances = [agent.get_performance_summary() for agent in self.agents]
        
        total_trades = sum(p['metrics']['total_trades'] for p in agent_performances)
        total_pnl = sum(p['metrics']['total_pnl'] for p in agent_performances)
        
        return {
            'total_agents': len(self.agents),
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'agent_performances': agent_performances,
            'coordinator_performance': self.coordinator.get_performance_summary()
        }
    
    def save_system_state(self, path: str):
        """Save all agent models and system state"""
        system_state = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'config': self.config,
            'agent_states': []
        }
        
        for agent in self.agents:
            if agent.rl_enabled:
                model_path = f"{path}_{agent.agent_id}_model.pt"
                agent.rl_agent.save_model(model_path)
                system_state['agent_states'].append({
                    'agent_id': agent.agent_id,
                    'agent_type': agent.agent_type.value,
                    'model_path': model_path,
                    'performance': agent.get_performance_summary()
                })
        
        # Save system state
        with open(f"{path}_system_state.json", 'w') as f:
            json.dump(system_state, f, indent=2)
        
        self.logger.info(f"System state saved to {path}")
    
    def load_system_state(self, path: str):
        """Load all agent models and system state"""
        with open(f"{path}_system_state.json", 'r') as f:
            system_state = json.load(f)
        
        for agent_state in system_state['agent_states']:
            agent_id = agent_state['agent_id']
            model_path = agent_state['model_path']
            
            # Find corresponding agent
            agent = next((a for a in self.agents if a.agent_id == agent_id), None)
            if agent and agent.rl_enabled and os.path.exists(model_path):
                agent.rl_agent.load_model(model_path)
        
        self.logger.info(f"System state loaded from {path}")

# Example usage configuration
MULTI_AGENT_CONFIG = {
    'coordinator': {
        'signal_weights': {
            'scalper_1': 0.8,
            'scalper_2': 0.8,
            'swing_1': 1.2,
            'swing_2': 1.2,
            'momentum_1': 1.0,
            'momentum_2': 1.0,
            'whale_tracker': 1.5
        }
    },
    'agents': {
        'scalper_1': {
            'rl_enabled': True,
            'position_size': 0.1,
            'max_spread': 0.0001,
            'min_profit_pips': 3
        },
        'swing_1': {
            'rl_enabled': True,
            'position_size': 0.25,
            'lookback_periods': 50
        },
        'momentum_1': {
            'rl_enabled': True,
            'momentum_threshold': 0.02
        },
        'whale_tracker': {
            'rl_enabled': True,
            'whale_threshold': 1000000,
            'volume_spike_threshold': 3.0
        }
    }
}