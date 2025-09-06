"""
Strategy Agent Implementation for Autonomous Financial Risk Management
Uses TD3 (Twin Delayed Deep Deterministic Policy Gradient) for continuous action spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

from .base_agent import ReinforcementLearningAgent, MarketState, Decision, AgentState

logger = logging.getLogger(__name__)

@dataclass
class TradingAction:
    """Represents a trading action"""
    symbol: str
    action_type: str  # 'buy', 'sell', 'hold'
    quantity: float
    price_limit: Optional[float] = None
    confidence: float = 0.0

class ActorNetwork(nn.Module):
    """Actor network for TD3 algorithm"""
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_sizes: List[int] = [256, 256]):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        layers = []
        input_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.max_action * self.network(state)

class CriticNetwork(nn.Module):
    """Critic network for TD3 algorithm"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(CriticNetwork, self).__init__()
        
        # First critic network
        layers1 = []
        input_size = state_dim + action_dim
        
        for hidden_size in hidden_sizes:
            layers1.append(nn.Linear(input_size, hidden_size))
            layers1.append(nn.ReLU())
            input_size = hidden_size
        
        layers1.append(nn.Linear(input_size, 1))
        self.critic1 = nn.Sequential(*layers1)
        
        # Second critic network (for TD3)
        layers2 = []
        input_size = state_dim + action_dim
        
        for hidden_size in hidden_sizes:
            layers2.append(nn.Linear(input_size, hidden_size))
            layers2.append(nn.ReLU())
            input_size = hidden_size
        
        layers2.append(nn.Linear(input_size, 1))
        self.critic2 = nn.Sequential(*layers2)
    
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        return q1, q2
    
    def q1(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.critic1(state_action)

class StrategyAgent(ReinforcementLearningAgent):
    """
    Strategy Agent using TD3 algorithm for portfolio optimization
    Handles continuous action spaces for position sizing and asset allocation
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None):
        super().__init__(agent_id, config, model_path)
        
        # Strategy-specific parameters
        self.state_dim = config.get("state_dim", 100)
        self.action_dim = config.get("action_dim", 10)  # Number of assets
        self.max_action = config.get("max_action", 1.0)
        
        # TD3 specific parameters
        self.policy_noise = config.get("policy_noise", 0.2)
        self.noise_clip = config.get("noise_clip", 0.5)
        self.policy_delay = config.get("policy_delay", 2)
        
        # Portfolio constraints
        self.max_position_size = config.get("max_position_size", 0.1)
        self.max_leverage = config.get("max_leverage", 2.0)
        
        # Performance tracking
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        
        # Current portfolio state
        self.current_positions = {}
        self.cash_balance = config.get("initial_cash", 1000000.0)
        
        self.update_state(AgentState.ACTIVE)
        
    def _load_model(self):
        """Load TD3 actor and critic networks"""
        try:
            # Actor networks
            self.actor = ActorNetwork(self.state_dim, self.action_dim, self.max_action).to(self.device)
            self.actor_target = ActorNetwork(self.state_dim, self.action_dim, self.max_action).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            
            # Critic networks
            self.critic = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic_target = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            # Load pre-trained weights if available
            if self.model_path:
                try:
                    checkpoint = torch.load(f"{self.model_path}/{self.agent_id}.pth", map_location=self.device)
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.critic.load_state_dict(checkpoint['critic'])
                    logger.info(f"Loaded pre-trained model for {self.agent_id}")
                except FileNotFoundError:
                    logger.info(f"No pre-trained model found for {self.agent_id}, starting fresh")
            
            logger.info(f"Model loaded successfully for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error loading model for agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
    
    def _setup_optimizer(self):
        """Setup optimizers for actor and critic networks"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
    
    async def process_market_data(self, market_state: MarketState) -> Decision:
        """Process market data and generate trading decision"""
        try:
            # Convert market state to tensor
            state_vector = self._encode_market_state(market_state)
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            
            # Generate action using actor network
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy()[0]
            
            # Add exploration noise during training
            if self.state == AgentState.TRAINING:
                noise = np.random.normal(0, self.policy_noise, size=self.action_dim)
                noise = np.clip(noise, -self.noise_clip, self.noise_clip)
                action = np.clip(action + noise, -self.max_action, self.max_action)
            
            # Convert action to trading decisions
            trading_actions = self._action_to_trades(action, market_state)
            
            # Calculate confidence based on Q-values
            confidence = self._calculate_confidence(state_tensor, torch.FloatTensor(action).unsqueeze(0).to(self.device))
            
            # Create decision
            decision = Decision(
                agent_id=self.agent_id,
                decision_type="portfolio_allocation",
                action=str(trading_actions),
                confidence=confidence,
                reasoning=self._generate_reasoning(action, market_state),
                timestamp=datetime.now(timezone.utc),
                expected_outcome={"expected_return": float(np.mean(action) * 0.01)}
            )
            
            self.decisions_made += 1
            return decision
            
        except Exception as e:
            logger.error(f"Error processing market data in agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
            return self._create_hold_decision()
    
    def _encode_market_state(self, market_state: MarketState) -> np.ndarray:
        """Encode market state into feature vector"""
        features = []
        
        # Price features
        prices = list(market_state.prices.values())
        features.extend(prices[:self.action_dim])  # Limit to action dimension
        
        # Volume features
        volumes = list(market_state.volumes.values())
        features.extend(volumes[:self.action_dim])
        
        # Volatility features
        volatilities = list(market_state.volatility.values())
        features.extend(volatilities[:self.action_dim])
        
        # Market regime and sentiment
        regime_encoding = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}
        features.extend([
            regime_encoding.get(market_state.regime, 0.0),
            market_state.sentiment_score,
            market_state.risk_level
        ])
        
        # Portfolio state
        portfolio_features = self._encode_portfolio_state()
        features.extend(portfolio_features)
        
        # Pad or truncate to state_dim
        features = features[:self.state_dim]
        while len(features) < self.state_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_portfolio_state(self) -> List[float]:
        """Encode current portfolio state"""
        features = []
        
        # Current positions (normalized)
        for i in range(self.action_dim):
            symbol = f"asset_{i}"
            position = self.current_positions.get(symbol, 0.0)
            features.append(position / self.max_position_size)
        
        # Cash ratio
        total_value = self._calculate_portfolio_value()
        cash_ratio = self.cash_balance / total_value if total_value > 0 else 1.0
        features.append(cash_ratio)
        
        # Recent performance metrics
        features.extend([
            self.total_return,
            self.sharpe_ratio,
            self.max_drawdown,
            self.win_rate
        ])
        
        return features
    
    def _action_to_trades(self, action: np.ndarray, market_state: MarketState) -> List[TradingAction]:
        """Convert neural network action to trading actions"""
        trading_actions = []
        symbols = list(market_state.prices.keys())[:self.action_dim]
        
        for i, symbol in enumerate(symbols):
            if i < len(action):
                # Scale action to position size
                target_position = action[i] * self.max_position_size
                current_position = self.current_positions.get(symbol, 0.0)
                
                # Calculate required trade
                trade_quantity = target_position - current_position
                
                if abs(trade_quantity) > 0.001:  # Minimum trade threshold
                    action_type = "buy" if trade_quantity > 0 else "sell"
                    
                    trading_action = TradingAction(
                        symbol=symbol,
                        action_type=action_type,
                        quantity=abs(trade_quantity),
                        price_limit=market_state.prices[symbol] * 1.01 if trade_quantity > 0 else market_state.prices[symbol] * 0.99,
                        confidence=abs(action[i])
                    )
                    
                    trading_actions.append(trading_action)
        
        return trading_actions
    
    def _calculate_confidence(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Calculate decision confidence based on Q-values"""
        with torch.no_grad():
            q1, q2 = self.critic(state, action)
            # Use minimum Q-value for conservative estimate
            confidence = torch.min(q1, q2).item()
            # Normalize to [0, 1]
            confidence = torch.sigmoid(torch.tensor(confidence)).item()
        
        return confidence
    
    def _generate_reasoning(self, action: np.ndarray, market_state: MarketState) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []
        
        # Market regime analysis
        reasoning.append(f"Current market regime: {market_state.regime}")
        reasoning.append(f"Market sentiment score: {market_state.sentiment_score:.3f}")
        reasoning.append(f"Risk level: {market_state.risk_level:.3f}")
        
        # Action analysis
        avg_action = np.mean(action)
        if avg_action > 0.1:
            reasoning.append("Model suggests increasing overall exposure")
        elif avg_action < -0.1:
            reasoning.append("Model suggests reducing overall exposure")
        else:
            reasoning.append("Model suggests maintaining current positions")
        
        # Top positions
        top_actions = np.argsort(np.abs(action))[-3:]
        symbols = list(market_state.prices.keys())
        for idx in reversed(top_actions):
            if idx < len(symbols):
                symbol = symbols[idx]
                reasoning.append(f"Strong signal for {symbol}: {action[idx]:.3f}")
        
        return reasoning
    
    def _create_hold_decision(self) -> Decision:
        """Create a default hold decision when errors occur"""
        return Decision(
            agent_id=self.agent_id,
            decision_type="hold",
            action="hold_all_positions",
            confidence=0.0,
            reasoning=["Error occurred, maintaining current positions"],
            timestamp=datetime.now(timezone.utc),
            expected_outcome={"expected_return": 0.0}
        )
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash_balance
        for symbol, quantity in self.current_positions.items():
            # This would fetch current price in production
            price = 100.0  # Placeholder
            total_value += quantity * price
        return total_value
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Train the TD3 model"""
        try:
            self.update_state(AgentState.TRAINING)
            
            # Extract training data
            states = training_data["states"]
            actions = training_data["actions"]
            rewards = training_data["rewards"]
            next_states = training_data["next_states"]
            dones = training_data["dones"]
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            
            # Train critic
            critic_loss = self._train_critic(states, actions, rewards, next_states, dones)
            
            # Train actor (delayed)
            actor_loss = 0.0
            if self.training_step % self.policy_delay == 0:
                actor_loss = self._train_actor(states)
                self._soft_update_targets()
            
            self.training_step += 1
            
            # Update metrics
            training_metrics = {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "epsilon": self.epsilon,
                "training_step": self.training_step
            }
            
            # Decay epsilon
            self.decay_epsilon()
            
            self.update_state(AgentState.ACTIVE)
            logger.info(f"Training step completed for agent {self.agent_id}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error during training for agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
            return {"error": str(e)}
    
    def _train_critic(self, states, actions, rewards, next_states, dones) -> float:
        """Train the critic networks"""
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _train_actor(self, states) -> float:
        """Train the actor network"""
        # Actor loss
        actor_actions = self.actor(states)
        actor_loss = -self.critic.q1(states, actor_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        tau = 0.005  # Soft update parameter
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def update_positions(self, executed_trades: List[Dict[str, Any]]):
        """Update portfolio positions after trade execution"""
        for trade in executed_trades:
            symbol = trade["symbol"]
            quantity = trade["executed_quantity"]
            price = trade["executed_price"]
            action_type = trade["action_type"]
            
            if action_type == "buy":
                self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + quantity
                self.cash_balance -= quantity * price
            elif action_type == "sell":
                self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) - quantity
                self.cash_balance += quantity * price
    
    def calculate_performance_metrics(self, price_history: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            # Calculate returns
            portfolio_values = []
            for prices in zip(*price_history.values()):
                value = self.cash_balance
                for i, (symbol, price) in enumerate(zip(price_history.keys(), prices)):
                    position = self.current_positions.get(symbol, 0.0)
                    value += position * price
                portfolio_values.append(value)
            
            if len(portfolio_values) < 2:
                return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
            
            # Calculate metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            self.total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - peak) / peak
            self.max_drawdown = np.min(drawdowns) * 100
            
            # Win rate
            positive_returns = np.sum(returns > 0)
            self.win_rate = positive_returns / len(returns) * 100 if len(returns) > 0 else 0.0
            
            return {
                "total_return": self.total_return,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
    
    def save_model(self, path: str):
        """Save model state"""
        try:
            torch.save({
                'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'training_step': self.training_step,
                'epsilon': self.epsilon
            }, f"{path}/{self.agent_id}.pth")
            
            logger.info(f"Model saved for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error saving model for agent {self.agent_id}: {e}")

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "state_dim": 50,
        "action_dim": 5,
        "max_action": 1.0,
        "learning_rate": 0.001,
        "epsilon": 0.1,
        "gamma": 0.99,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_delay": 2,
        "max_position_size": 0.2,
        "initial_cash": 1000000.0
    }
    
    # Create strategy agent
    agent = StrategyAgent("strategy_001", config)
    
    print(f"Strategy agent {agent.agent_id} initialized successfully!")
    print(f"State: {agent.state}")
    print(f"Model loaded: {agent.model is not None}")
