"""
Risk Management Agent Implementation
Uses advanced risk models including VaR, CVaR, and real-time anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from scipy import stats
import pandas as pd

from .base_agent import BaseAgent, MarketState, Decision, AgentState, AgentMessage

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics calculation results"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    timestamp: datetime

@dataclass
class RiskAlert:
    """Risk alert structure"""
    alert_id: str
    risk_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    recommended_action: str
    affected_assets: List[str]
    risk_value: float
    threshold: float
    timestamp: datetime

@dataclass
class PortfolioState:
    """Current portfolio state for risk analysis"""
    positions: Dict[str, float]
    market_values: Dict[str, float]
    weights: Dict[str, float]
    cash_balance: float
    total_value: float
    leverage: float
    beta: float

class AnomalyDetector(nn.Module):
    """Neural network for market anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(AnomalyDetector, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Encoder
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Bottleneck
        bottleneck_dim = hidden_dims[-1] // 2
        layers.append(nn.Linear(current_dim, bottleneck_dim))
        layers.append(nn.ReLU())
        
        # Decoder
        current_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, input_dim))
        
        self.autoencoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.autoencoder(x)
    
    def encode(self, x):
        """Extract encoded features"""
        layers = list(self.autoencoder.children())
        encoded = x
        
        # Forward through encoder part
        encoder_layers = len([l for l in layers if isinstance(l, nn.Linear)]) // 2 + 1
        for i, layer in enumerate(layers):
            encoded = layer(encoded)
            if isinstance(layer, nn.Linear) and i // 3 >= encoder_layers:
                break
        
        return encoded

class RiskAgent(BaseAgent):
    """
    Risk Management Agent for portfolio risk monitoring and control
    Implements VaR, CVaR, anomaly detection, and real-time risk alerts
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any], model_path: Optional[str] = None):
        super().__init__(agent_id, config, model_path)
        
        # Risk parameters
        self.var_confidence_levels = config.get("var_confidence_levels", [0.95, 0.99])
        self.lookback_period = config.get("lookback_period", 252)  # Trading days
        self.risk_limits = config.get("risk_limits", {
            "max_var_95": 0.05,  # 5% daily VaR
            "max_drawdown": 0.20,  # 20% maximum drawdown
            "max_concentration": 0.30,  # 30% max single position
            "max_leverage": 3.0,  # 3x maximum leverage
            "min_liquidity": 0.10  # 10% minimum liquidity buffer
        })
        
        # Anomaly detection parameters
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)  # Standard deviations
        self.feature_dim = config.get("feature_dim", 50)
        
        # Risk monitoring
        self.risk_history = []
        self.active_alerts = []
        self.last_risk_check = None
        
        # Market correlation matrix
        self.correlation_matrix = None
        self.correlation_update_freq = config.get("correlation_update_freq", 24)  # Hours
        self.last_correlation_update = None
        
        self.update_state(AgentState.ACTIVE)
        
    def _load_model(self):
        """Load anomaly detection model"""
        try:
            self.anomaly_model = AnomalyDetector(self.feature_dim).to(self.device)
            
            # Load pre-trained weights if available
            if self.model_path:
                try:
                    checkpoint = torch.load(f"{self.model_path}/{self.agent_id}_anomaly.pth", map_location=self.device)
                    self.anomaly_model.load_state_dict(checkpoint['model'])
                    logger.info(f"Loaded pre-trained anomaly model for {self.agent_id}")
                except FileNotFoundError:
                    logger.info(f"No pre-trained anomaly model found for {self.agent_id}")
            
            # Set to evaluation mode
            self.anomaly_model.eval()
            
            logger.info(f"Risk model loaded successfully for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error loading risk model for agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
    
    def _setup_optimizer(self):
        """Setup optimizer for anomaly detection model"""
        self.optimizer = torch.optim.Adam(self.anomaly_model.parameters(), lr=self.learning_rate)
    
    async def process_market_data(self, market_state: MarketState) -> Decision:
        """Process market data and generate risk assessment"""
        try:
            # Get current portfolio state (this would come from portfolio manager)
            portfolio_state = await self._get_portfolio_state(market_state)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_state, market_state)
            
            # Detect anomalies
            anomaly_score = self._detect_anomalies(market_state)
            
            # Check risk limits and generate alerts
            alerts = self._check_risk_limits(risk_metrics, portfolio_state, anomaly_score)
            
            # Update active alerts
            self._update_alerts(alerts)
            
            # Generate risk-based decision
            decision = self._generate_risk_decision(risk_metrics, alerts, anomaly_score)
            
            # Store risk metrics for history
            self.risk_history.append(risk_metrics)
            if len(self.risk_history) > 1000:  # Limit history size
                self.risk_history.pop(0)
            
            self.last_risk_check = datetime.now(timezone.utc)
            self.decisions_made += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing market data in risk agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
            return self._create_emergency_decision()
    
    async def _get_portfolio_state(self, market_state: MarketState) -> PortfolioState:
        """Get current portfolio state (mock implementation)"""
        # In production, this would query the portfolio management system
        positions = {"AAPL": 100, "GOOGL": 50, "MSFT": 75, "TSLA": 25, "AMZN": 40}
        
        market_values = {}
        total_value = 0
        cash_balance = 100000
        
        for symbol, quantity in positions.items():
            price = market_state.prices.get(symbol, 100.0)
            market_value = quantity * price
            market_values[symbol] = market_value
            total_value += market_value
        
        total_value += cash_balance
        
        # Calculate weights
        weights = {symbol: value / total_value for symbol, value in market_values.items()}
        
        # Calculate portfolio beta (simplified)
        beta = 1.2  # This would be calculated from historical data
        
        # Calculate leverage
        leverage = sum(market_values.values()) / total_value
        
        return PortfolioState(
            positions=positions,
            market_values=market_values,
            weights=weights,
            cash_balance=cash_balance,
            total_value=total_value,
            leverage=leverage,
            beta=beta
        )
    
    def _calculate_risk_metrics(self, portfolio_state: PortfolioState, market_state: MarketState) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Generate synthetic historical returns for demonstration
            # In production, this would use real historical data
            returns = self._generate_portfolio_returns(portfolio_state, 252)
            
            # Value at Risk (VaR) calculations
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= var_95].mean() if np.any(returns <= var_95) else var_95
            cvar_99 = returns[returns <= var_99].mean() if np.any(returns <= var_99) else var_99
            
            # Maximum drawdown calculation
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdowns)
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            excess_returns = np.mean(returns) - risk_free_rate / 252
            sharpe_ratio = excess_returns / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Correlation risk (portfolio diversification)
            correlation_risk = self._calculate_correlation_risk(portfolio_state)
            
            # Concentration risk
            concentration_risk = max(portfolio_state.weights.values()) if portfolio_state.weights else 0
            
            # Liquidity risk (simplified)
            liquidity_risk = self._calculate_liquidity_risk(portfolio_state, market_state)
            
            return RiskMetrics(
                var_95=abs(var_95),
                var_99=abs(var_99),
                cvar_95=abs(cvar_95),
                cvar_99=abs(cvar_99),
                max_drawdown=abs(max_drawdown),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                beta=portfolio_state.beta,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                liquidity_risk=liquidity_risk,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._create_default_risk_metrics()
    
    def _generate_portfolio_returns(self, portfolio_state: PortfolioState, periods: int) -> np.ndarray:
        """Generate synthetic portfolio returns for risk calculation"""
        # This is a simplified implementation for demonstration
        # In production, use real historical price data
        
        # Generate correlated returns for portfolio assets
        n_assets = len(portfolio_state.positions)
        mean_returns = np.random.normal(0.0005, 0.001, n_assets)  # Daily returns
        
        # Create correlation matrix
        correlation = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        # Generate multivariate normal returns
        asset_returns = np.random.multivariate_normal(mean_returns, correlation * 0.02**2, periods)
        
        # Calculate portfolio returns using weights
        weights = np.array(list(portfolio_state.weights.values()))
        portfolio_returns = np.dot(asset_returns, weights)
        
        return portfolio_returns
    
    def _calculate_correlation_risk(self, portfolio_state: PortfolioState) -> float:
        """Calculate portfolio correlation risk"""
        # Simplified correlation risk calculation
        # High correlation = high risk
        weights = np.array(list(portfolio_state.weights.values()))
        
        # Generate sample correlation matrix
        n_assets = len(weights)
        if n_assets <= 1:
            return 0.0
        
        # Mock correlation matrix (in production, use real correlations)
        avg_correlation = 0.6  # Assume 60% average correlation
        
        # Portfolio correlation risk
        correlation_risk = np.sum(np.outer(weights, weights)) * avg_correlation
        correlation_risk = max(0, correlation_risk - np.sum(weights**2))  # Remove self-correlation
        
        return correlation_risk
    
    def _calculate_liquidity_risk(self, portfolio_state: PortfolioState, market_state: MarketState) -> float:
        """Calculate portfolio liquidity risk"""
        # Simplified liquidity risk based on position sizes and volumes
        liquidity_risk = 0.0
        
        for symbol, position_value in portfolio_state.market_values.items():
            volume = market_state.volumes.get(symbol, 1000000)  # Default volume
            price = market_state.prices.get(symbol, 100)
            
            # Calculate position as percentage of daily volume
            daily_volume_value = volume * price
            position_pct_of_volume = position_value / daily_volume_value if daily_volume_value > 0 else 0
            
            # Higher percentage = higher liquidity risk
            liquidity_risk += position_pct_of_volume * portfolio_state.weights.get(symbol, 0)
        
        return min(liquidity_risk, 1.0)  # Cap at 100%
    
    def _detect_anomalies(self, market_state: MarketState) -> float:
        """Detect market anomalies using autoencoder"""
        try:
            # Encode market state
            features = self._encode_market_features(market_state)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get reconstruction from autoencoder
            with torch.no_grad():
                reconstruction = self.anomaly_model(features_tensor)
                
            # Calculate reconstruction error
            mse_error = F.mse_loss(reconstruction, features_tensor, reduction='mean')
            anomaly_score = mse_error.item()
            
            # Normalize score (higher score = more anomalous)
            normalized_score = min(anomaly_score / 0.1, 10.0)  # Normalize to reasonable range
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return 0.0
    
    def _encode_market_features(self, market_state: MarketState) -> np.ndarray:
        """Encode market state into feature vector for anomaly detection"""
        features = []
        
        # Price features (normalized)
        prices = list(market_state.prices.values())[:10]  # Limit to top 10 assets
        if prices:
            price_mean = np.mean(prices)
            price_std = np.std(prices) if len(prices) > 1 else 1.0
            normalized_prices = [(p - price_mean) / price_std for p in prices]
            features.extend(normalized_prices)
        
        # Volume features (log normalized)
        volumes = list(market_state.volumes.values())[:10]
        if volumes:
            log_volumes = [np.log(max(v, 1)) for v in volumes]
            vol_mean = np.mean(log_volumes)
            vol_std = np.std(log_volumes) if len(log_volumes) > 1 else 1.0
            normalized_volumes = [(v - vol_mean) / vol_std for v in log_volumes]
            features.extend(normalized_volumes)
        
        # Volatility features
        volatilities = list(market_state.volatility.values())[:10]
        features.extend(volatilities)
        
        # Market regime and sentiment
        regime_encoding = {"bull": 1.0, "bear": -1.0, "sideways": 0.0}
        features.extend([
            regime_encoding.get(market_state.regime, 0.0),
            market_state.sentiment_score,
            market_state.risk_level
        ])
        
        # Pad or truncate to feature_dim
        features = features[:self.feature_dim]
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _check_risk_limits(self, risk_metrics: RiskMetrics, portfolio_state: PortfolioState, anomaly_score: float) -> List[RiskAlert]:
        """Check risk metrics against predefined limits"""
        alerts = []
        
        # VaR limit check
        if risk_metrics.var_95 > self.risk_limits["max_var_95"]:
            alerts.append(RiskAlert(
                alert_id=f"var_95_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="VALUE_AT_RISK",
                severity="HIGH" if risk_metrics.var_95 > self.risk_limits["max_var_95"] * 1.5 else "MEDIUM",
                description=f"95% VaR ({risk_metrics.var_95:.3f}) exceeds limit ({self.risk_limits['max_var_95']:.3f})",
                recommended_action="Reduce portfolio exposure or hedge positions",
                affected_assets=list(portfolio_state.positions.keys()),
                risk_value=risk_metrics.var_95,
                threshold=self.risk_limits["max_var_95"],
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Maximum drawdown check
        if risk_metrics.max_drawdown > self.risk_limits["max_drawdown"]:
            alerts.append(RiskAlert(
                alert_id=f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="MAX_DRAWDOWN",
                severity="CRITICAL" if risk_metrics.max_drawdown > self.risk_limits["max_drawdown"] * 1.2 else "HIGH",
                description=f"Maximum drawdown ({risk_metrics.max_drawdown:.3f}) exceeds limit ({self.risk_limits['max_drawdown']:.3f})",
                recommended_action="Implement stop-loss or reduce position sizes",
                affected_assets=list(portfolio_state.positions.keys()),
                risk_value=risk_metrics.max_drawdown,
                threshold=self.risk_limits["max_drawdown"],
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Concentration risk check
        if risk_metrics.concentration_risk > self.risk_limits["max_concentration"]:
            # Find the most concentrated position
            max_weight_asset = max(portfolio_state.weights, key=portfolio_state.weights.get)
            
            alerts.append(RiskAlert(
                alert_id=f"concentration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="CONCENTRATION_RISK",
                severity="MEDIUM",
                description=f"Position concentration ({risk_metrics.concentration_risk:.3f}) exceeds limit ({self.risk_limits['max_concentration']:.3f})",
                recommended_action=f"Reduce position in {max_weight_asset} or diversify portfolio",
                affected_assets=[max_weight_asset],
                risk_value=risk_metrics.concentration_risk,
                threshold=self.risk_limits["max_concentration"],
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Leverage check
        if portfolio_state.leverage > self.risk_limits["max_leverage"]:
            alerts.append(RiskAlert(
                alert_id=f"leverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="LEVERAGE_RISK",
                severity="HIGH",
                description=f"Portfolio leverage ({portfolio_state.leverage:.2f}x) exceeds limit ({self.risk_limits['max_leverage']:.2f}x)",
                recommended_action="Reduce leverage by closing positions or adding capital",
                affected_assets=list(portfolio_state.positions.keys()),
                risk_value=portfolio_state.leverage,
                threshold=self.risk_limits["max_leverage"],
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Liquidity risk check
        cash_ratio = portfolio_state.cash_balance / portfolio_state.total_value
        if cash_ratio < self.risk_limits["min_liquidity"]:
            alerts.append(RiskAlert(
                alert_id=f"liquidity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="LIQUIDITY_RISK",
                severity="MEDIUM",
                description=f"Cash ratio ({cash_ratio:.3f}) below minimum ({self.risk_limits['min_liquidity']:.3f})",
                recommended_action="Increase cash position or improve portfolio liquidity",
                affected_assets=[],
                risk_value=cash_ratio,
                threshold=self.risk_limits["min_liquidity"],
                timestamp=datetime.now(timezone.utc)
            ))
        
        # Anomaly detection alert
        if anomaly_score > self.anomaly_threshold:
            alerts.append(RiskAlert(
                alert_id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_type="MARKET_ANOMALY",
                severity="HIGH" if anomaly_score > self.anomaly_threshold * 2 else "MEDIUM",
                description=f"Market anomaly detected (score: {anomaly_score:.2f}, threshold: {self.anomaly_threshold:.2f})",
                recommended_action="Monitor market conditions closely, consider reducing exposure",
                affected_assets=list(portfolio_state.positions.keys()),
                risk_value=anomaly_score,
                threshold=self.anomaly_threshold,
                timestamp=datetime.now(timezone.utc)
            ))
        
        return alerts
    
    def _update_alerts(self, new_alerts: List[RiskAlert]):
        """Update active alerts list"""
        # Add new alerts
        self.active_alerts.extend(new_alerts)
        
        # Remove expired alerts (older than 1 hour)
        current_time = datetime.now(timezone.utc)
        self.active_alerts = [
            alert for alert in self.active_alerts
            if (current_time - alert.timestamp).total_seconds() < 3600
        ]
        
        # Log new alerts
        for alert in new_alerts:
            logger.warning(f"Risk Alert [{alert.severity}]: {alert.description}")
    
    def _generate_risk_decision(self, risk_metrics: RiskMetrics, alerts: List[RiskAlert], anomaly_score: float) -> Decision:
        """Generate risk-based decision"""
        # Determine overall risk level
        risk_factors = [
            risk_metrics.var_95 / self.risk_limits["max_var_95"],
            risk_metrics.max_drawdown / self.risk_limits["max_drawdown"],
            risk_metrics.concentration_risk / self.risk_limits["max_concentration"],
            anomaly_score / self.anomaly_threshold
        ]
        
        overall_risk = np.mean(risk_factors)
        
        # Generate decision based on risk level
        if overall_risk > 1.5:
            action = "REDUCE_RISK_IMMEDIATELY"
            confidence = 0.9
        elif overall_risk > 1.0:
            action = "REDUCE_RISK_GRADUALLY"
            confidence = 0.7
        elif overall_risk > 0.8:
            action = "MONITOR_CLOSELY"
            confidence = 0.6
        else:
            action = "MAINTAIN_CURRENT_RISK"
            confidence = 0.5
        
        # Generate reasoning
        reasoning = [
            f"Overall risk score: {overall_risk:.2f}",
            f"95% VaR: {risk_metrics.var_95:.3f} (limit: {self.risk_limits['max_var_95']:.3f})",
            f"Max drawdown: {risk_metrics.max_drawdown:.3f} (limit: {self.risk_limits['max_drawdown']:.3f})",
            f"Concentration risk: {risk_metrics.concentration_risk:.3f}",
            f"Anomaly score: {anomaly_score:.2f}",
            f"Active alerts: {len(alerts)}"
        ]
        
        return Decision(
            agent_id=self.agent_id,
            decision_type="risk_management",
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc),
            expected_outcome={"risk_reduction": max(0, overall_risk - 1.0)}
        )
    
    def _create_emergency_decision(self) -> Decision:
        """Create emergency risk decision when errors occur"""
        return Decision(
            agent_id=self.agent_id,
            decision_type="emergency_risk",
            action="HALT_TRADING",
            confidence=1.0,
            reasoning=["Risk agent encountered error", "Emergency risk protocol activated"],
            timestamp=datetime.now(timezone.utc),
            expected_outcome={"risk_reduction": 1.0}
        )
    
    def _create_default_risk_metrics(self) -> RiskMetrics:
        """Create default risk metrics when calculation fails"""
        return RiskMetrics(
            var_95=0.02,
            var_99=0.04,
            cvar_95=0.03,
            cvar_99=0.05,
            max_drawdown=0.05,
            volatility=0.15,
            sharpe_ratio=0.5,
            beta=1.0,
            correlation_risk=0.5,
            concentration_risk=0.2,
            liquidity_risk=0.1,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Train the anomaly detection model"""
        try:
            self.update_state(AgentState.TRAINING)
            
            # Extract training data
            normal_data = training_data["normal_market_conditions"]
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(normal_data).to(self.device)
            
            # Training loop
            self.anomaly_model.train()
            total_loss = 0.0
            batch_size = 32
            num_batches = len(data_tensor) // batch_size
            
            for i in range(num_batches):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(data_tensor))
                batch_data = data_tensor[batch_start:batch_end]
                
                # Forward pass
                reconstructed = self.anomaly_model(batch_data)
                loss = F.mse_loss(reconstructed, batch_data)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Set back to evaluation mode
            self.anomaly_model.eval()
            self.update_state(AgentState.ACTIVE)
            
            logger.info(f"Anomaly model training completed for agent {self.agent_id}")
            
            return {
                "reconstruction_loss": avg_loss,
                "training_batches": num_batches
            }
            
        except Exception as e:
            logger.error(f"Error training anomaly model for agent {self.agent_id}: {e}")
            self.update_state(AgentState.ERROR)
            return {"error": str(e)}
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            if not self.risk_history:
                return {"error": "No risk history available"}
            
            latest_metrics = self.risk_history[-1]
            
            # Calculate risk trends
            if len(self.risk_history) >= 10:
                recent_var = [m.var_95 for m in self.risk_history[-10:]]
                var_trend = "INCREASING" if recent_var[-1] > recent_var[0] else "DECREASING"
            else:
                var_trend = "INSUFFICIENT_DATA"
            
            # Alert summary
            alert_summary = {}
            for alert in self.active_alerts:
                alert_summary[alert.risk_type] = alert_summary.get(alert.risk_type, 0) + 1
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "risk_metrics": {
                    "var_95": latest_metrics.var_95,
                    "var_99": latest_metrics.var_99,
                    "cvar_95": latest_metrics.cvar_95,
                    "max_drawdown": latest_metrics.max_drawdown,
                    "volatility": latest_metrics.volatility,
                    "sharpe_ratio": latest_metrics.sharpe_ratio,
                    "concentration_risk": latest_metrics.concentration_risk,
                    "liquidity_risk": latest_metrics.liquidity_risk
                },
                "risk_trends": {
                    "var_trend": var_trend,
                    "history_length": len(self.risk_history)
                },
                "active_alerts": {
                    "total_alerts": len(self.active_alerts),
                    "by_type": alert_summary,
                    "critical_alerts": len([a for a in self.active_alerts if a.severity == "CRITICAL"]),
                    "high_alerts": len([a for a in self.active_alerts if a.severity == "HIGH"])
                },
                "risk_limits": self.risk_limits,
                "last_update": self.last_risk_check.isoformat() if self.last_risk_check else None
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str):
        """Save anomaly detection model"""
        try:
            torch.save({
                'model': self.anomaly_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': {
                    'feature_dim': self.feature_dim,
                    'anomaly_threshold': self.anomaly_threshold
                }
            }, f"{path}/{self.agent_id}_anomaly.pth")
            
            logger.info(f"Risk model saved for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error saving risk model for agent {self.agent_id}: {e}")

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_risk_agent():
        # Configuration
        config = {
            "feature_dim": 50,
            "anomaly_threshold": 2.0,
            "var_confidence_levels": [0.95, 0.99],
            "lookback_period": 252,
            "risk_limits": {
                "max_var_95": 0.05,
                "max_drawdown": 0.20,
                "max_concentration": 0.30,
                "max_leverage": 3.0,
                "min_liquidity": 0.10
            },
            "learning_rate": 0.001
        }
        
        # Create risk agent
        risk_agent = RiskAgent("risk_001", config)
        
        # Create sample market state
        market_state = MarketState(
            timestamp=datetime.now(timezone.utc),
            prices={"AAPL": 150.0, "GOOGL": 2500.0, "MSFT": 300.0, "TSLA": 800.0, "AMZN": 3200.0},
            volumes={"AAPL": 1000000, "GOOGL": 500000, "MSFT": 800000, "TSLA": 600000, "AMZN": 400000},
            volatility={"AAPL": 0.25, "GOOGL": 0.30, "MSFT": 0.22, "TSLA": 0.45, "AMZN": 0.28},
            sentiment_score=0.2,
            regime="bull",
            risk_level=0.3
        )
        
        # Process market data
        decision = await risk_agent.process_market_data(market_state)
        
        print(f"Risk Agent Decision:")
        print(f"  Action: {decision.action}")
        print(f"  Confidence: {decision.confidence:.3f}")
        print(f"  Reasoning: {decision.reasoning}")
        
        # Get risk report
        risk_report = risk_agent.get_risk_report()
        print(f"\nRisk Report:")
        print(f"  VaR 95%: {risk_report['risk_metrics']['var_95']:.4f}")
        print(f"  Max Drawdown: {risk_report['risk_metrics']['max_drawdown']:.4f}")
        print(f"  Active Alerts: {risk_report['active_alerts']['total_alerts']}")
    
    # Run test
    asyncio.run(test_risk_agent())
