"""
Base Agent Architecture for Autonomous Financial Risk Management Ecosystem
"""

import abc
import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    TRAINING = "training"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1 = low, 5 = high

@dataclass
class MarketState:
    """Current market state representation"""
    timestamp: datetime
    prices: Dict[str, float]
    volumes: Dict[str, float]
    volatility: Dict[str, float]
    sentiment_score: float
    regime: str  # bull, bear, sideways
    risk_level: float

@dataclass
class Decision:
    """Agent decision structure"""
    agent_id: str
    decision_type: str
    action: str
    confidence: float
    reasoning: List[str]
    timestamp: datetime
    expected_outcome: Dict[str, float]

class BaseAgent(abc.ABC):
    """Base class for all financial agents"""
    
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        model_path: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.config = config
        self.state = AgentState.IDLE
        self.model_path = model_path
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.subscribers = set()
        
        # Performance tracking
        self.decisions_made = 0
        self.accuracy_score = 0.0
        self.last_update = datetime.now(timezone.utc)
        
        # Model components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        
        # Initialize agent
        self._initialize()
        
    def _initialize(self):
        """Initialize agent-specific components"""
        logger.info(f"Initializing agent {self.agent_id}")
        self._load_model()
        self._setup_optimizer()
        
    @abc.abstractmethod
    def _load_model(self):
        """Load agent-specific model"""
        pass
    
    @abc.abstractmethod
    def _setup_optimizer(self):
        """Setup optimizer for model training"""
        pass
    
    @abc.abstractmethod
    async def process_market_data(self, market_state: MarketState) -> Decision:
        """Process market data and make decisions"""
        pass
    
    @abc.abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Train the agent's model"""
        pass
    
    async def send_message(self, message: AgentMessage):
        """Send message to another agent"""
        logger.info(f"Agent {self.agent_id} sending message to {message.receiver_id}")
        # In production, this would use a message broker like Redis or RabbitMQ
        await self._broadcast_message(message)
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive message from queue"""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def _broadcast_message(self, message: AgentMessage):
        """Broadcast message to subscribers"""
        # This is a simplified implementation
        # In production, use proper message broker
        pass
    
    def subscribe_to_agent(self, agent_id: str):
        """Subscribe to messages from another agent"""
        self.subscribers.add(agent_id)
    
    def update_state(self, new_state: AgentState):
        """Update agent state"""
        old_state = self.state
        self.state = new_state
        self.last_update = datetime.now(timezone.utc)
        logger.info(f"Agent {self.agent_id} state changed: {old_state} -> {new_state}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "decisions_made": self.decisions_made,
            "accuracy_score": self.accuracy_score,
            "last_update": self.last_update.isoformat(),
            "uptime": (datetime.now(timezone.utc) - self.last_update).total_seconds()
        }
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if model is loaded
            if self.model is None:
                return False
            
            # Check if agent is in valid state
            if self.state == AgentState.ERROR:
                return False
            
            # Check if recent activity
            time_since_update = datetime.now(timezone.utc) - self.last_update
            if time_since_update.total_seconds() > 300:  # 5 minutes
                return False
            
            return True
        except Exception as e:
            logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown agent"""
        logger.info(f"Shutting down agent {self.agent_id}")
        self.update_state(AgentState.DISABLED)
        
        # Save model state
        if self.model is not None and self.model_path:
            torch.save(self.model.state_dict(), f"{self.model_path}/{self.agent_id}_final.pth")
        
        # Clear queues
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

class ReinforcementLearningAgent(BaseAgent):
    """Base class for RL-based agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None):
        super().__init__(agent_id, config, model_path)
        
        # RL-specific parameters
        self.epsilon = config.get("epsilon", 0.1)
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 0.001)
        
        # Experience replay
        self.memory = []
        self.memory_size = config.get("memory_size", 10000)
        
        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.target_update_freq = config.get("target_update_freq", 100)
        self.training_step = 0
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
    
    def sample_batch(self) -> Optional[Tuple]:
        """Sample batch from experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = np.random.choice(self.memory, self.batch_size, replace=False)
        
        states = torch.tensor([e[0] for e in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([e[1] for e in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([e[3] for e in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([e[4] for e in batch], dtype=torch.bool).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        min_epsilon = self.config.get("min_epsilon", 0.01)
        epsilon_decay = self.config.get("epsilon_decay", 0.995)
        
        self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

class TransformerAgent(BaseAgent):
    """Base class for transformer-based agents"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None):
        super().__init__(agent_id, config, model_path)
        
        # Transformer parameters
        self.d_model = config.get("d_model", 512)
        self.n_heads = config.get("n_heads", 8)
        self.n_layers = config.get("n_layers", 6)
        self.seq_length = config.get("seq_length", 100)
        
        # Attention weights for explainability
        self.attention_weights = None
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get attention weights for explainability"""
        return self.attention_weights
    
    def encode_time_series(self, data: np.ndarray) -> torch.Tensor:
        """Encode time series data for transformer input"""
        # Add positional encoding
        seq_len, d_model = data.shape
        
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros(seq_len, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        encoded_data = torch.tensor(data, dtype=torch.float32) + pos_encoding
        return encoded_data.to(self.device)

class AgentCoordinator:
    """Coordinates multiple agents and manages their interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.message_broker = asyncio.Queue()
        self.is_running = False
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def start(self):
        """Start the coordination system"""
        self.is_running = True
        logger.info("Starting agent coordinator")
        
        # Start all agent tasks
        tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(self._run_agent(agent))
            tasks.append(task)
        
        # Start message routing task
        routing_task = asyncio.create_task(self._route_messages())
        tasks.append(routing_task)
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    async def _run_agent(self, agent: BaseAgent):
        """Run an individual agent"""
        while self.is_running:
            try:
                # Check for messages
                message = await agent.receive_message()
                if message:
                    await self._handle_agent_message(agent, message)
                
                # Health check
                if not await agent.health_check():
                    logger.warning(f"Agent {agent.agent_id} failed health check")
                    agent.update_state(AgentState.ERROR)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error running agent {agent.agent_id}: {e}")
                agent.update_state(AgentState.ERROR)
    
    async def _handle_agent_message(self, agent: BaseAgent, message: AgentMessage):
        """Handle message received by agent"""
        logger.info(f"Agent {agent.agent_id} received message: {message.message_type}")
        # Process message based on type
        # This is where agent-specific message handling would occur
    
    async def _route_messages(self):
        """Route messages between agents"""
        while self.is_running:
            try:
                # In a real implementation, this would use a proper message broker
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in message routing: {e}")
    
    async def shutdown(self):
        """Shutdown all agents and coordinator"""
        logger.info("Shutting down agent coordinator")
        self.is_running = False
        
        # Shutdown all agents
        shutdown_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.shutdown())
            shutdown_tasks.append(task)
        
        await asyncio.gather(*shutdown_tasks)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = agent.get_performance_metrics()
        
        return {
            "coordinator_running": self.is_running,
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() 
                               if a.state == AgentState.ACTIVE),
            "agent_details": agent_statuses
        }

# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Configuration
        config = {
            "epsilon": 0.1,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "memory_size": 10000,
            "batch_size": 32
        }
        
        # Create coordinator
        coordinator = AgentCoordinator(config)
        
        # This would be replaced with actual agent implementations
        # agent = StrategyAgent("strategy_001", config)
        # coordinator.register_agent(agent)
        
        # Start system
        # await coordinator.start()

    # Run the example
    # asyncio.run(main())
    print("Base agent architecture loaded successfully!")
