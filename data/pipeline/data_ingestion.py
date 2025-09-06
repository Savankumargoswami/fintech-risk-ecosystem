"""
Real-time Data Ingestion Pipeline for Financial Risk Management System
Handles market data, alternative data sources, and streaming data processing
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import kafka
from kafka import KafkaProducer, KafkaConsumer
import redis
import websocket
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    high: float
    low: float
    open: float
    close: float
    source: str

@dataclass
class AlternativeDataPoint:
    """Alternative data point (sentiment, news, etc.)"""
    timestamp: datetime
    source: str
    data_type: str  # sentiment, news, social, satellite, etc.
    value: float
    metadata: Dict[str, Any]
    confidence: float

class DataIngestionConfig:
    """Configuration for data ingestion"""
    def __init__(self):
        # API Keys (should be in environment variables)
        self.alpha_vantage_key = "your_alpha_vantage_key"
        self.polygon_key = "your_polygon_key"
        self.twitter_bearer_token = "your_twitter_token"
        self.news_api_key = "your_news_api_key"
        
        # Kafka configuration
        self.kafka_bootstrap_servers = "localhost:9092"
        self.kafka_topics = {
            "market_data": "market_data_topic",
            "alternative_data": "alt_data_topic",
            "processed_data": "processed_data_topic"
        }
        
        # Redis configuration
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_db = 0
        
        # WebSocket endpoints
        self.websocket_endpoints = {
            "polygon": "wss://socket.polygon.io/stocks",
            "finnhub": "wss://ws.finnhub.io",
            "alpha_vantage": "wss://ws.alpha-vantage.com/quote"
        }
        
        # Data refresh intervals (seconds)
        self.refresh_intervals = {
            "market_data": 1,      # 1 second for real-time data
            "sentiment": 60,       # 1 minute for sentiment
            "news": 300,          # 5 minutes for news
            "economic": 3600,     # 1 hour for economic data
            "alternative": 900    # 15 minutes for alternative data
        }

class MarketDataProvider:
    """Market data provider interface"""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host, 
            port=config.redis_port, 
            db=config.redis_db
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # WebSocket connections
        self.websockets = {}
        self.is_running = False
        
    async def start_real_time_feed(self, symbols: List[str]):
        """Start real-time market data feed"""
        self.is_running = True
        logger.info(f"Starting real-time data feed for {len(symbols)} symbols")
        
        # Start multiple data sources concurrently
        tasks = [
            asyncio.create_task(self._polygon_websocket_feed(symbols)),
            asyncio.create_task(self._alpha_vantage_feed(symbols)),
            asyncio.create_task(self._rest_api_feed(symbols)),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _polygon_websocket_feed(self, symbols: List[str]):
        """Polygon WebSocket real-time data feed"""
        try:
            import websockets
            
            uri = f"{self.config.websocket_endpoints['polygon']}?auth_key={self.config.polygon_key}"
            
            async with websockets.connect(uri) as websocket:
                # Subscribe to symbols
                subscribe_msg = {
                    "action": "subscribe",
                    "params": ",".join([f"T.{symbol}" for symbol in symbols])
                }
                await websocket.send(json.dumps(subscribe_msg))
                
                logger.info("Connected to Polygon WebSocket")
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        if data.get("ev") == "T":  # Trade event
                            market_data_point = MarketDataPoint(
                                symbol=data.get("sym", ""),
                                timestamp=datetime.fromtimestamp(data.get("t", 0) / 1000, tz=timezone.utc),
                                price=float(data.get("p", 0)),
                                volume=float(data.get("s", 0)),
                                bid=0.0,  # Not available in trade data
                                ask=0.0,  # Not available in trade data
                                high=0.0,
                                low=0.0,
                                open=0.0,
                                close=float(data.get("p", 0)),
                                source="polygon"
                            )
                            
                            await self._publish_market_data(market_data_point)
                            
                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()
                        continue
                    except Exception as e:
                        logger.error(f"Error in Polygon WebSocket: {e}")
                        break
                        
        except Exception as e:
            logger.error(f"Failed to connect to Polygon WebSocket: {e}")
    
    async def _alpha_vantage_feed(self, symbols: List[str]):
        """Alpha Vantage API data feed"""
        while self.is_running:
            try:
                async with aiohttp.ClientSession() as session:
                    for symbol in symbols:
                        url = f"https://www.alphavantage.co/query"
                        params = {
                            "function": "GLOBAL_QUOTE",
                            "symbol": symbol,
                            "apikey": self.config.alpha_vantage_key
                        }
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                quote = data.get("Global Quote", {})
                                
                                if quote:
                                    market_data_point = MarketDataPoint(
                                        symbol=quote.get("01. symbol", symbol),
                                        timestamp=datetime.now(timezone.utc),
                                        price=float(quote.get("05. price", 0)),
                                        volume=float(quote.get("06. volume", 0)),
                                        bid=0.0,
                                        ask=0.0,
                                        high=float(quote.get("03. high", 0)),
                                        low=float(quote.get("04. low", 0)),
                                        open=float(quote.get("02. open", 0)),
                                        close=float(quote.get("08. previous close", 0)),
                                        source="alpha_vantage"
                                    )
                                    
                                    await self._publish_market_data(market_data_point)
                        
                        # Rate limiting
                        await asyncio.sleep(12)  # Alpha Vantage free tier: 5 calls/minute
                
                await asyncio.sleep(60)  # Wait 1 minute before next batch
                
            except Exception as e:
                logger.error(f"Error in Alpha Vantage feed: {e}")
                await asyncio.sleep(60)
    
    async def _rest_api_feed(self, symbols: List[str]):
        """Generic REST API data feed"""
        while self.is_running:
            try:
                # This would integrate with other REST APIs like Yahoo Finance, IEX, etc.
                await self._fetch_yahoo_finance_data(symbols)
                await asyncio.sleep(self.config.refresh_intervals["market_data"])
                
            except Exception as e:
                logger.error(f"Error in REST API feed: {e}")
                await asyncio.sleep(30)
    
    async def _fetch_yahoo_finance_data(self, symbols: List[str]):
        """Fetch data from Yahoo Finance (example implementation)"""
        try:
            # Using yfinance library (would need to be installed)
            import yfinance as yf
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                history = ticker.history(period="1d", interval="1m").tail(1)
                
                if not history.empty:
                    row = history.iloc[0]
                    market_data_point = MarketDataPoint(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        price=float(row['Close']),
                        volume=float(row['Volume']),
                        bid=0.0,
                        ask=0.0,
                        high=float(row['High']),
                        low=float(row['Low']),
                        open=float(row['Open']),
                        close=float(row['Close']),
                        source="yahoo_finance"
                    )
                    
                    await self._publish_market_data(market_data_point)
        
        except ImportError:
            logger.warning("yfinance not installed, skipping Yahoo Finance data")
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
    
    async def _publish_market_data(self, data_point: MarketDataPoint):
        """Publish market data to Kafka and Redis"""
        try:
            # Serialize data point
            data_dict = asdict(data_point)
            data_dict['timestamp'] = data_point.timestamp.isoformat()
            
            # Publish to Kafka
            self.kafka_producer.send(
                self.config.kafka_topics["market_data"],
                value=data_dict
            )
            
            # Store latest data in Redis
            redis_key = f"market_data:{data_point.symbol}"
            self.redis_client.setex(redis_key, 60, json.dumps(data_dict))
            
            # Store in time series for historical analysis
            ts_key = f"ts:{data_point.symbol}:{data_point.timestamp.strftime('%Y%m%d')}"
            self.redis_client.zadd(ts_key, {json.dumps(data_dict): data_point.timestamp.timestamp()})
            
        except Exception as e:
            logger.error(f"Error publishing market data: {e}")
    
    async def stop(self):
        """Stop data ingestion"""
        self.is_running = False
        self.kafka_producer.close()
        self.redis_client.close()
        logger.info("Data ingestion stopped")

class AlternativeDataProvider:
    """Alternative data provider for sentiment, news, etc."""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.redis_host, 
            port=config.redis_port, 
            db=config.redis_db
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.is_running = False
    
    async def start_alternative_feeds(self, symbols: List[str]):
        """Start all alternative data feeds"""
        self.is_running = True
        logger.info("Starting alternative data feeds")
        
        tasks = [
            asyncio.create_task(self._news_sentiment_feed(symbols)),
            asyncio.create_task(self._social_media_sentiment_feed(symbols)),
            asyncio.create_task(self._economic_indicators_feed()),
            asyncio.create_task(self._crypto_sentiment_feed()),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _news_sentiment_feed(self, symbols: List[str]):
        """News sentiment analysis feed"""
        while self.is_running:
            try:
                await self._fetch_news_data(symbols)
                await asyncio.sleep(self.config.refresh_intervals["news"])
            except Exception as e:
                logger.error(f"Error in news sentiment feed: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_news_data(self, symbols: List[str]):
        """Fetch and analyze news data"""
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    # News API example
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": f"{symbol} stock",
                        "apiKey": self.config.news_api_key,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 10
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get("articles", [])
                            
                            # Analyze sentiment of articles
                            sentiment_score = await self._analyze_news_sentiment(articles)
                            
                            alt_data_point = AlternativeDataPoint(
                                timestamp=datetime.now(timezone.utc),
                                source="news_api",
                                data_type="news_sentiment",
                                value=sentiment_score,
                                metadata={
                                    "symbol": symbol,
                                    "article_count": len(articles),
                                    "source": "newsapi"
                                },
                                confidence=0.7
                            )
                            
                            await self._publish_alternative_data(alt_data_point)
        
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
    
    async def _analyze_news_sentiment(self, articles: List[Dict]) -> float:
        """Analyze sentiment of news articles"""
        try:
            # Simple sentiment analysis (in production, use more sophisticated NLP)
            positive_words = ['gain', 'profit', 'growth', 'positive', 'bullish', 'upgrade', 'beat']
            negative_words = ['loss', 'decline', 'negative', 'bearish', 'downgrade', 'miss', 'fall']
            
            total_sentiment = 0.0
            article_count = 0
            
            for article in articles:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                content = f"{title} {description}"
                
                sentiment = 0.0
                for word in positive_words:
                    sentiment += content.count(word) * 0.1
                for word in negative_words:
                    sentiment -= content.count(word) * 0.1
                
                total_sentiment += sentiment
                article_count += 1
            
            return total_sentiment / article_count if article_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    async def _social_media_sentiment_feed(self, symbols: List[str]):
        """Social media sentiment feed (Twitter, Reddit, etc.)"""
        while self.is_running:
            try:
                await self._fetch_twitter_sentiment(symbols)
                await asyncio.sleep(self.config.refresh_intervals["sentiment"])
            except Exception as e:
                logger.error(f"Error in social media feed: {e}")
                await asyncio.sleep(300)
    
    async def _fetch_twitter_sentiment(self, symbols: List[str]):
        """Fetch Twitter sentiment data"""
        try:
            # Twitter API v2 example
            headers = {
                "Authorization": f"Bearer {self.config.twitter_bearer_token}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                for symbol in symbols:
                    url = "https://api.twitter.com/2/tweets/search/recent"
                    params = {
                        "query": f"${symbol} -is:retweet lang:en",
                        "max_results": 100,
                        "tweet.fields": "created_at,public_metrics"
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            tweets = data.get("data", [])
                            
                            sentiment_score = await self._analyze_twitter_sentiment(tweets)
                            
                            alt_data_point = AlternativeDataPoint(
                                timestamp=datetime.now(timezone.utc),
                                source="twitter",
                                data_type="social_sentiment",
                                value=sentiment_score,
                                metadata={
                                    "symbol": symbol,
                                    "tweet_count": len(tweets),
                                    "source": "twitter_api_v2"
                                },
                                confidence=0.6
                            )
                            
                            await self._publish_alternative_data(alt_data_point)
        
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment: {e}")
    
    async def _analyze_twitter_sentiment(self, tweets: List[Dict]) -> float:
        """Analyze sentiment of tweets"""
        try:
            # Simplified sentiment analysis
            positive_indicators = ['bullish', 'moon', '🚀', 'to the moon', 'buy', 'long']
            negative_indicators = ['bearish', 'crash', 'sell', 'short', 'dump', '📉']
            
            total_sentiment = 0.0
            tweet_count = 0
            
            for tweet in tweets:
                text = tweet.get('text', '').lower()
                
                sentiment = 0.0
                for indicator in positive_indicators:
                    sentiment += text.count(indicator) * 0.2
                for indicator in negative_indicators:
                    sentiment -= text.count(indicator) * 0.2
                
                # Weight by engagement metrics
                metrics = tweet.get('public_metrics', {})
                engagement_weight = 1 + (metrics.get('retweet_count', 0) + metrics.get('like_count', 0)) / 100
                sentiment *= engagement_weight
                
                total_sentiment += sentiment
                tweet_count += 1
            
            return total_sentiment / tweet_count if tweet_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            return 0.0
    
    async def _economic_indicators_feed(self):
        """Economic indicators feed (FRED, etc.)"""
        while self.is_running:
            try:
                await self._fetch_economic_data()
                await asyncio.sleep(self.config.refresh_intervals["economic"])
            except Exception as e:
                logger.error(f"Error in economic indicators feed: {e}")
                await asyncio.sleep(3600)
    
    async def _fetch_economic_data(self):
        """Fetch economic indicators from FRED API"""
        try:
            # FRED API example (would need API key)
            indicators = [
                "GDPC1",      # Real GDP
                "UNRATE",     # Unemployment Rate
                "CPIAUCSL",   # Consumer Price Index
                "FEDFUNDS",   # Federal Funds Rate
                "DGS10",      # 10-Year Treasury Rate
            ]
            
            fred_api_key = "your_fred_api_key"
            
            async with aiohttp.ClientSession() as session:
                for indicator in indicators:
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": indicator,
                        "api_key": fred_api_key,
                        "file_type": "json",
                        "limit": 1,
                        "sort_order": "desc"
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get("observations", [])
                            
                            if observations:
                                obs = observations[0]
                                value = float(obs.get("value", 0))
                                
                                alt_data_point = AlternativeDataPoint(
                                    timestamp=datetime.now(timezone.utc),
                                    source="fred",
                                    data_type="economic_indicator",
                                    value=value,
                                    metadata={
                                        "indicator": indicator,
                                        "date": obs.get("date"),
                                        "source": "fred_api"
                                    },
                                    confidence=0.9
                                )
                                
                                await self._publish_alternative_data(alt_data_point)
        
        except Exception as e:
            logger.error(f"Error fetching economic data: {e}")
    
    async def _crypto_sentiment_feed(self):
        """Cryptocurrency sentiment feed (Fear & Greed Index, etc.)"""
        while self.is_running:
            try:
                await self._fetch_crypto_sentiment()
                await asyncio.sleep(self.config.refresh_intervals["alternative"])
            except Exception as e:
                logger.error(f"Error in crypto sentiment feed: {e}")
                await asyncio.sleep(900)
    
    async def _fetch_crypto_sentiment(self):
        """Fetch cryptocurrency sentiment indicators"""
        try:
            # Fear & Greed Index API
            async with aiohttp.ClientSession() as session:
                url = "https://api.alternative.me/fng/"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        fng_data = data.get("data", [{}])[0]
                        
                        alt_data_point = AlternativeDataPoint(
                            timestamp=datetime.now(timezone.utc),
                            source="alternative_me",
                            data_type="crypto_sentiment",
                            value=float(fng_data.get("value", 50)) / 100.0,  # Normalize to 0-1
                            metadata={
                                "classification": fng_data.get("value_classification"),
                                "source": "fear_greed_index"
                            },
                            confidence=0.8
                        )
                        
                        await self._publish_alternative_data(alt_data_point)
        
        except Exception as e:
            logger.error(f"Error fetching crypto sentiment: {e}")
    
    async def _publish_alternative_data(self, data_point: AlternativeDataPoint):
        """Publish alternative data to Kafka and Redis"""
        try:
            # Serialize data point
            data_dict = asdict(data_point)
            data_dict['timestamp'] = data_point.timestamp.isoformat()
            
            # Publish to Kafka
            self.kafka_producer.send(
                self.config.kafka_topics["alternative_data"],
                value=data_dict
            )
            
            # Store in Redis
            redis_key = f"alt_data:{data_point.source}:{data_point.data_type}"
            self.redis_client.setex(redis_key, 3600, json.dumps(data_dict))
            
            logger.debug(f"Published alternative data: {data_point.source}:{data_point.data_type}")
            
        except Exception as e:
            logger.error(f"Error publishing alternative data: {e}")
    
    async def stop(self):
        """Stop alternative data ingestion"""
        self.is_running = False
        self.kafka_producer.close()
        self.redis_client.close()
        logger.info("Alternative data ingestion stopped")

class DataIngestionOrchestrator:
    """Orchestrates all data ingestion processes"""
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.market_provider = MarketDataProvider(config)
        self.alt_provider = AlternativeDataProvider(config)
        self.symbols = []
        self.is_running = False
    
    async def start(self, symbols: List[str]):
        """Start all data ingestion processes"""
        self.symbols = symbols
        self.is_running = True
        
        logger.info(f"Starting data ingestion orchestrator for {len(symbols)} symbols")
        
        # Start both providers concurrently
        tasks = [
            asyncio.create_task(self.market_provider.start_real_time_feed(symbols)),
            asyncio.create_task(self.alt_provider.start_alternative_feeds(symbols)),
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all data ingestion processes"""
        self.is_running = False
        
        await self.market_provider.stop()
        await self.alt_provider.stop()
        
        logger.info("Data ingestion orchestrator stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get ingestion status"""
        return {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "market_provider_running": self.market_provider.is_running,
            "alt_provider_running": self.alt_provider.is_running,
            "config": {
                "refresh_intervals": self.config.refresh_intervals,
                "kafka_topics": self.config.kafka_topics
            }
        }

# Example usage and testing
async def main():
    """Example usage of the data ingestion system"""
    config = DataIngestionConfig()
    orchestrator = DataIngestionOrchestrator(config)
    
    # List of symbols to track
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    try:
        # Start data ingestion
        await orchestrator.start(symbols)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        # Clean shutdown
        await orchestrator.stop()

if __name__ == "__main__":
    # Run the data ingestion system
    asyncio.run(main())
