"""
FastAPI Main Application for Autonomous Financial Risk Management System
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import uvicorn

# Internal imports
from .routers import portfolio, risk, trading, monitoring
from .middleware.auth import verify_token, get_current_user
from .middleware.rate_limiting import RateLimiter
from .middleware.logging import LoggingMiddleware
from ..agents.coordinator import AgentCoordinator
from ..agents.strategy_agent import StrategyAgent
from ..agents.risk_agent import RiskAgent
from ..data.pipeline.data_ingestion import DataIngestionOrchestrator, DataIngestionConfig
from ..config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for managing system components
agent_coordinator: Optional[AgentCoordinator] = None
data_orchestrator: Optional[DataIngestionOrchestrator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global agent_coordinator, data_orchestrator
    
    logger.info("Starting Autonomous Financial Risk Management System")
    
    try:
        # Initialize settings
        settings = get_settings()
        
        # Initialize data ingestion
        data_config = DataIngestionConfig()
        data_orchestrator = DataIngestionOrchestrator(data_config)
        
        # Initialize agent coordinator
        agent_config = {
            "state_dim": 50,
            "action_dim": 10,
            "learning_rate": 0.001,
            "max_agents": 10
        }
        agent_coordinator = AgentCoordinator(agent_config)
        
        # Create and register agents
        strategy_agent = StrategyAgent("strategy_001", agent_config)
        risk_agent = RiskAgent("risk_001", agent_config)
        
        agent_coordinator.register_agent(strategy_agent)
        agent_coordinator.register_agent(risk_agent)
        
        # Start background tasks
        symbols = settings.TRADING_SYMBOLS
        
        # Start data ingestion in background
        asyncio.create_task(data_orchestrator.start(symbols))
        
        # Start agent coordinator in background
        asyncio.create_task(agent_coordinator.start())
        
        logger.info("System initialization completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Error during system initialization: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down system components")
        
        if agent_coordinator:
            await agent_coordinator.shutdown()
        
        if data_orchestrator:
            await data_orchestrator.stop()
        
        logger.info("System shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="Autonomous Financial Risk Management System",
    description="Advanced AI-powered financial risk management and trading system",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(LoggingMiddleware)

# Rate limiting
rate_limiter = RateLimiter()

# Include routers
app.include_router(
    portfolio.router,
    prefix="/api/v1/portfolio",
    tags=["Portfolio Management"],
    dependencies=[Depends(rate_limiter.limit_requests)]
)

app.include_router(
    risk.router,
    prefix="/api/v1/risk",
    tags=["Risk Management"],
    dependencies=[Depends(rate_limiter.limit_requests)]
)

app.include_router(
    trading.router,
    prefix="/api/v1/trading",
    tags=["Trading Operations"],
    dependencies=[Depends(rate_limiter.limit_requests)]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitoring",
    tags=["System Monitoring"],
    dependencies=[Depends(rate_limiter.limit_requests)]
)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Autonomous Financial Risk Management System",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs_url": "/api/docs"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        # Check agent coordinator
        if agent_coordinator:
            coordinator_status = agent_coordinator.get_system_status()
            health_status["components"]["agent_coordinator"] = {
                "status": "healthy" if coordinator_status["coordinator_running"] else "unhealthy",
                "active_agents": coordinator_status["active_agents"],
                "total_agents": coordinator_status["total_agents"]
            }
        else:
            health_status["components"]["agent_coordinator"] = {"status": "not_initialized"}
        
        # Check data orchestrator
        if data_orchestrator:
            data_status = data_orchestrator.get_status()
            health_status["components"]["data_orchestrator"] = {
                "status": "healthy" if data_status["is_running"] else "unhealthy",
                "symbols_tracked": len(data_status["symbols"])
            }
        else:
            health_status["components"]["data_orchestrator"] = {"status": "not_initialized"}
        
        # Overall status
        component_statuses = [comp.get("status") for comp in health_status["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "starting"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.get("/api/v1/system/status")
async def system_status(current_user: dict = Depends(get_current_user)):
    """Get detailed system status (authenticated endpoint)"""
    try:
        system_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": current_user["username"],
            "system": {
                "agents": {},
                "data_pipeline": {},
                "performance": {}
            }
        }
        
        # Agent information
        if agent_coordinator:
            agent_status = agent_coordinator.get_system_status()
            system_info["system"]["agents"] = agent_status
            
            # Individual agent performance
            for agent_id, agent_details in agent_status.get("agent_details", {}).items():
                system_info["system"]["agents"][agent_id] = agent_details
        
        # Data pipeline information
        if data_orchestrator:
            data_status = data_orchestrator.get_status()
            system_info["system"]["data_pipeline"] = data_status
        
        # Performance metrics (would be collected from monitoring system)
        system_info["system"]["performance"] = {
            "requests_per_second": 0.0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "uptime_seconds": 0
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@app.post("/api/v1/system/agents/{agent_id}/train")
async def train_agent(
    agent_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger agent training (authenticated endpoint)"""
    try:
        if not agent_coordinator:
            raise HTTPException(status_code=503, detail="Agent coordinator not initialized")
        
        # Check if agent exists
        if agent_id not in agent_coordinator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = agent_coordinator.agents[agent_id]
        
        # Generate training data (in production, this would be real historical data)
        training_data = {
            "states": [[0.1] * 50 for _ in range(1000)],
            "actions": [[0.05] * 10 for _ in range(1000)],
            "rewards": [0.01] * 1000,
            "next_states": [[0.1] * 50 for _ in range(1000)],
            "dones": [False] * 1000
        }
        
        # Start training in background
        background_tasks.add_task(agent.train, training_data)
        
        return {
            "message": f"Training started for agent {agent_id}",
            "agent_id": agent_id,
            "initiated_by": current_user["username"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting agent training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start agent training")

@app.post("/api/v1/system/emergency_stop")
async def emergency_stop(current_user: dict = Depends(get_current_user)):
    """Emergency stop all trading operations (authenticated endpoint)"""
    try:
        logger.warning(f"Emergency stop triggered by user: {current_user['username']}")
        
        # Stop all agents
        if agent_coordinator:
            for agent in agent_coordinator.agents.values():
                agent.update_state("DISABLED")
        
        # Stop data ingestion
        if data_orchestrator:
            await data_orchestrator.stop()
        
        return {
            "message": "Emergency stop executed successfully",
            "initiated_by": current_user["username"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "affected_components": ["agents", "data_ingestion", "trading_operations"]
        }
        
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute emergency stop")

@app.get("/api/v1/system/logs")
async def get_system_logs(
    limit: int = 100,
    level: str = "INFO",
    current_user: dict = Depends(get_current_user)
):
    """Get system logs (authenticated endpoint)"""
    try:
        # In production, this would read from a centralized logging system
        logs = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "component": "system",
                "message": "System operational",
                "details": {}
            }
        ]
        
        return {
            "logs": logs[:limit],
            "total_count": len(logs),
            "requested_by": current_user["username"],
            "filters": {"level": level, "limit": limit}
        }
        
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system logs")

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
import json

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                # Remove broken connections
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

@app.websocket("/api/v1/ws/live_updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time system updates"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic updates
            if agent_coordinator:
                system_status = agent_coordinator.get_system_status()
                await manager.send_personal_message(
                    json.dumps({
                        "type": "system_status",
                        "data": system_status,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }),
                    websocket
                )
            
            # Send market data updates (if available)
            # This would integrate with the real-time data pipeline
            
            await asyncio.sleep(5)  # Send updates every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code} error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url)
        }
    )

# Background tasks for system maintenance
async def system_maintenance_task():
    """Background task for system maintenance"""
    while True:
        try:
            # Perform maintenance tasks
            if agent_coordinator:
                # Health check all agents
                for agent_id, agent in agent_coordinator.agents.items():
                    if not await agent.health_check():
                        logger.warning(f"Agent {agent_id} failed health check")
            
            # Clean up old data, logs, etc.
            # This would be implemented based on specific requirements
            
            await asyncio.sleep(300)  # Run every 5 minutes
            
        except Exception as e:
            logger.error(f"Error in maintenance task: {e}")
            await asyncio.sleep(60)

# Start background maintenance task
@app.on_event("startup")
async def startup_event():
    """Startup event to initialize background tasks"""
    # Start maintenance task
    asyncio.create_task(system_maintenance_task())
    
    # Start real-time data broadcasting
    asyncio.create_task(broadcast_market_updates())

async def broadcast_market_updates():
    """Broadcast real-time market updates to WebSocket clients"""
    while True:
        try:
            if manager.active_connections:
                # Generate sample market update
                market_update = {
                    "type": "market_update",
                    "data": {
                        "AAPL": {"price": 150.0 + (datetime.now().second % 10), "change": 0.5},
                        "GOOGL": {"price": 2500.0 + (datetime.now().second % 20), "change": -0.3},
                        "MSFT": {"price": 300.0 + (datetime.now().second % 15), "change": 0.2},
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await manager.broadcast(json.dumps(market_update))
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            logger.error(f"Error broadcasting market updates: {e}")
            await asyncio.sleep(10)

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1  # Use 1 worker for development
    )
