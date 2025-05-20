# Enhanced Mariah with Reinforcement Learning
# Add this to a new file: mariah_rl.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
import random
import joblib
from datetime import datetime
import logging

class MariahRLAgent:
    """
    PPO-based RL Agent for Mariah
    Learns from trading decisions and outcomes
    """
    
    def __init__(self, state_dim=50, action_dim=3, lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim  # [Hold, Buy, Sell]
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        
        # Networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
        
        # Logging
        self.logger = logging.getLogger("MariahRL")
        
    def get_action(self, state, deterministic=False):
        """Get action from current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.actor_critic(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
            return action.item(), action_probs.squeeze(), value.item()
    
    def store_experience(self, state, action, reward, next_state, done, action_prob, value):
        """Store experience in buffer"""
        self.buffer.store(state, action, reward, next_state, done, action_prob, value)
    
    def update_policy(self):
        """Update policy using PPO"""
        if len(self.buffer) < 1000:  # Wait for enough experiences
            return
            
        # Get batch of experiences
        states, actions, rewards, next_states, dones, old_probs, values = self.buffer.get_batch()
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get current policy outputs
            action_probs, new_values = self.actor_critic(states)
            new_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Compute losses
            actor_loss = self._compute_actor_loss(new_action_probs, old_probs, advantages)
            critic_loss = self._compute_critic_loss(new_values.squeeze(), returns)
            entropy_loss = self._compute_entropy_loss(action_probs)
            
            total_loss = actor_loss + self.value_coeff * critic_loss - self.entropy_coeff * entropy_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        self.logger.info(f"Policy updated - Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else values[i]
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return torch.tensor(advantages, dtype=torch.float32).to(self.device)
    
    def _compute_actor_loss(self, new_probs, old_probs, advantages):
        """Compute PPO actor loss with clipping"""
        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        return actor_loss
    
    def _compute_critic_loss(self, values, returns):
        """Compute critic loss (MSE)"""
        return ((returns - values) ** 2).mean()
    
    def _compute_entropy_loss(self, action_probs):
        """Compute entropy loss for exploration"""
        return -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
    
    def save_model(self, path="models/mariah_rl_agent.pt"):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path="models/mariah_rl_agent.pt"):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"Model loaded from {path}")


class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_probs, value


class ExperienceBuffer:
    """Experience buffer for PPO training"""
    
    def __init__(self, max_size=10000):
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)
        self.action_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
    
    def store(self, state, action, reward, next_state, done, action_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.action_probs.append(action_prob)
        self.values.append(value)
    
    def get_batch(self):
        """Get all experiences as tensors"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        states = torch.FloatTensor(list(self.states)).to(device)
        actions = torch.LongTensor(list(self.actions)).to(device)
        rewards = torch.FloatTensor(list(self.rewards)).to(device)
        next_states = torch.FloatTensor(list(self.next_states)).to(device)
        dones = torch.BoolTensor(list(self.dones)).to(device)
        action_probs = torch.FloatTensor(list(self.action_probs)).to(device)
        values = torch.FloatTensor(list(self.values)).to(device)
        
        return states, actions, rewards, next_states, dones, action_probs, values
    
    def clear(self):
        """Clear the buffer"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.action_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)


class TradingEnvironment:
    """Trading environment for Mariah RL training"""
    
    def __init__(self, historical_data, initial_balance=10000):
        self.data = historical_data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to start"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # Number of shares owned
        self.total_trades = 0
        self.win_trades = 0
        self.done = False
        
        return self._get_state()
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        if self.done:
            return self._get_state(), 0, True
        
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            
        return self._get_state(), reward, self.done
    
    def _execute_action(self, action):
        """Execute trading action"""
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close'] if self.current_step + 1 < len(self.data) else current_price
        
        reward = 0
        transaction_cost = 0.001  # 0.1% transaction cost
        
        if action == 1:  # Buy
            if self.balance > 0 and self.position == 0:
                shares_to_buy = self.balance / current_price
                self.position = shares_to_buy * (1 - transaction_cost)
                self.balance = 0
                self.total_trades += 1
                
        elif action == 2:  # Sell
            if self.position > 0:
                self.balance = self.position * current_price * (1 - transaction_cost)
                if self.balance > self.initial_balance:
                    self.win_trades += 1
                self.position = 0
                self.total_trades += 1
        
        # Calculate reward based on portfolio value change
        current_portfolio_value = self.balance + (self.position * current_price)
        next_portfolio_value = self.balance + (self.position * next_price)
        reward = (next_portfolio_value - current_portfolio_value) / current_portfolio_value
        
        # Add penalty for too many trades
        if self.total_trades > 100:
            reward -= 0.001
            
        return reward
    
    def _get_state(self):
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(50)
            
        # Get price and volume data
        current_data = self.data.iloc[max(0, self.current_step-10):self.current_step+1]
        
        # Technical indicators
        features = []
        
        # Price features
        features.extend([
            current_data['close'].iloc[-1] / current_data['close'].iloc[0] - 1,  # Price change
            current_data['high'].iloc[-1] / current_data['close'].iloc[-1] - 1,  # High vs close
            current_data['low'].iloc[-1] / current_data['close'].iloc[-1] - 1,   # Low vs close
        ])
        
        # Volume features
        features.append(current_data['volume'].iloc[-1] / current_data['volume'].mean() - 1)
        
        # Moving averages
        ma_5 = current_data['close'].rolling(5).mean().iloc[-1]
        ma_10 = current_data['close'].rolling(10).mean().iloc[-1]
        features.extend([
            current_data['close'].iloc[-1] / ma_5 - 1,
            current_data['close'].iloc[-1] / ma_10 - 1,
            ma_5 / ma_10 - 1
        ])
        
        # RSI
        rsi = self._calculate_rsi(current_data['close'])
        features.append(rsi / 100 - 0.5)
        
        # Portfolio state
        portfolio_value = self.balance + (self.position * current_data['close'].iloc[-1])
        features.extend([
            self.balance / self.initial_balance - 1,
            self.position,
            portfolio_value / self.initial_balance - 1,
            self.total_trades / 100,
            self.win_trades / max(self.total_trades, 1)
        ])
        
        # Pad to fixed size
        while len(features) < 50:
            features.append(0)
            
        return np.array(features[:50], dtype=np.float32)
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50


# Integration with existing Mariah system
class EnhancedMariahLevel2:
    """Enhanced Mariah with RL capabilities"""
    
    def __init__(self):
        self.rl_agent = MariahRLAgent()
        self.last_analysis = {}
        self.training_mode = False
        self.state_history = deque(maxlen=1000)
        
    def set_training_mode(self, enabled=True):
        """Enable/disable training mode"""
        self.training_mode = enabled
        
    def analyze_symbol_with_rl(self, symbol, interval, session):
        """Enhanced analysis using both traditional + RL"""
        # Get traditional analysis (existing code)
        traditional_analysis = self.analyze_symbol(symbol, interval, session)
        
        # Get current market state
        state = self._get_market_state(symbol, interval, session)
        
        # Get RL recommendation
        if state is not None:
            rl_action, action_probs, value = self.rl_agent.get_action(state)
            rl_confidence = torch.max(action_probs).item()
            
            # Convert action to signal
            rl_signal = ['hold', 'buy', 'sell'][rl_action]
            
            # Combine traditional and RL signals
            combined_analysis = self._combine_signals(traditional_analysis, {
                'signal': rl_signal,
                'confidence': rl_confidence,
                'value_estimate': value
            })
            
            # Store for training
            if self.training_mode and symbol in self.last_analysis:
                self._store_experience(symbol, state, rl_action, action_probs[rl_action].item(), value)
            
            self.last_analysis[symbol] = {
                'traditional': traditional_analysis,
                'rl': {'signal': rl_signal, 'confidence': rl_confidence, 'value': value},
                'combined': combined_analysis,
                'state': state,
                'timestamp': pd.Timestamp.now()
            }
            
            return combined_analysis
        
        return traditional_analysis
    
    def _get_market_state(self, symbol, interval, session):
        """Extract market state for RL agent"""
        try:
            # Get market data
            res = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=100
            )["result"]["list"]
            
            if not res:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(res, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            
            # Create trading environment temporarily to get state
            env = TradingEnvironment(df)
            return env._get_state()
            
        except Exception as e:
            print(f"Error getting market state: {e}")
            return None
    
    def _combine_signals(self, traditional, rl):
        """Combine traditional and RL signals"""
        # Weight the signals (you can adjust these)
        traditional_weight = 0.6
        rl_weight = 0.4
        
        # Convert signals to scores
        signal_scores = {'hold': 0, 'buy': 1, 'sell': -1}
        
        trad_score = signal_scores[traditional['action']] * traditional['confidence']
        rl_score = signal_scores[rl['signal']] * rl['confidence']
        
        # Weighted combination
        combined_score = traditional_weight * trad_score + rl_weight * rl_score
        
        # Convert back to signal
        if combined_score > 0.3:
            final_signal = 'buy'
        elif combined_score < -0.3:
            final_signal = 'sell'
        else:
            final_signal = 'hold'
        
        return {
            'action': final_signal,
            'confidence': min(abs(combined_score), 1.0),
            'traditional_score': trad_score,
            'rl_score': rl_score,
            'combined_score': combined_score,
            'rl_value_estimate': rl['value_estimate']
        }
    
    def _store_experience(self, symbol, state, action, action_prob, value):
        """Store experience for training"""
        # Get reward from actual trading result (you'll need to implement this)
        # For now, use a placeholder
        reward = 0  # Calculate based on actual trade outcome
        
        # Store in RL agent's buffer
        next_state = state  # Get next state
        done = False
        
        self.rl_agent.store_experience(state, action, reward, next_state, done, action_prob, value)
    
    def train_rl_agent(self, historical_data):
        """Train the RL agent on historical data"""
        self.rl_agent.logger.info("Starting RL training...")
        
        env = TradingEnvironment(historical_data)
        
        for episode in range(100):  # Number of training episodes
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action, action_probs, value = self.rl_agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                self.rl_agent.store_experience(
                    state, action, reward, next_state, done, 
                    action_probs[action].item(), value
                )
                
                state = next_state
                total_reward += reward
                
            # Update policy after each episode
            if episode % 10 == 0:
                self.rl_agent.update_policy()
                self.rl_agent.logger.info(f"Episode {episode}, Total Reward: {total_reward:.4f}")
        
        # Save trained model
        self.rl_agent.save_model()
        self.rl_agent.logger.info("RL training completed and model saved.")
    
    def load_trained_model(self):
        """Load a pre-trained RL model"""
        self.rl_agent.load_model()