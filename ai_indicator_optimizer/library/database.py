"""
PostgreSQL Database Schema und Connection Management für Trading Library
"""

import asyncio
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, LargeBinary
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.types import TypeDecorator, Text
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import uuid
import json
import pickle
import logging
from dataclasses import dataclass
import numpy as np
from PIL import Image
import io

from ..core.config import SystemConfig


@dataclass
class DatabaseConfig:
    """Datenbank-Konfiguration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_library"
    username: str = "postgres"
    password: str = "postgres"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600


# SQLAlchemy Base
# SQLAlchemy Base
Base = declarative_base()


class JSONType(TypeDecorator):
    """JSON Type für SQLite-Kompatibilität"""
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


class VisualPatternModel(Base):
    """SQLAlchemy Model für Visual Patterns"""
    __tablename__ = 'visual_patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_type = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(20), nullable=False, index=True)
    
    # Chart Image (komprimiert als BLOB)
    chart_image_data = Column(LargeBinary, nullable=False)
    chart_width = Column(Integer, nullable=False)
    chart_height = Column(Integer, nullable=False)
    
    # Pattern Eigenschaften
    confidence_score = Column(Float, nullable=False, index=True)
    market_context = Column(JSONType, nullable=False)
    
    # Performance Metriken
    performance_metrics = Column(JSONType, nullable=True)
    win_rate = Column(Float, nullable=True, index=True)
    profit_factor = Column(Float, nullable=True, index=True)
    max_drawdown = Column(Float, nullable=True)
    
    # Metadaten
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True, index=True)
    
    # Zusätzliche Indizes für Performance
    __table_args__ = (
        {'postgresql_partition_by': 'RANGE (created_at)'},
    )


class TradingStrategyModel(Base):
    """SQLAlchemy Model für Trading Strategien"""
    __tablename__ = 'trading_strategies'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_name = Column(String(200), nullable=False, index=True)
    strategy_type = Column(String(100), nullable=False, index=True)
    
    # Strategie-Definition
    indicators = Column(JSONType, nullable=False)
    entry_conditions = Column(JSONType, nullable=False)
    exit_conditions = Column(JSONType, nullable=False)
    risk_management = Column(JSONType, nullable=False)
    
    # Pine Script Code (falls generiert)
    pine_script_code = Column(Text, nullable=True)
    
    # Performance Metriken
    performance_metrics = Column(JSONType, nullable=True)
    backtest_results = Column(JSONType, nullable=True)
    total_return = Column(Float, nullable=True, index=True)
    sharpe_ratio = Column(Float, nullable=True, index=True)
    win_rate = Column(Float, nullable=True, index=True)
    profit_factor = Column(Float, nullable=True, index=True)
    max_drawdown = Column(Float, nullable=True, index=True)
    total_trades = Column(Integer, nullable=True)
    
    # Metadaten
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True, index=True)
    
    # AI-Training Metadaten
    training_accuracy = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)


class PatternPerformanceModel(Base):
    """SQLAlchemy Model für Pattern Performance Tracking"""
    __tablename__ = 'pattern_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Trade Details
    entry_time = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_time = Column(DateTime(timezone=True), nullable=True, index=True)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    
    # Trade Ergebnis
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True, index=True)
    trade_duration_minutes = Column(Integer, nullable=True)
    
    # Market Conditions
    market_volatility = Column(Float, nullable=True)
    market_trend = Column(String(20), nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))


class DatabaseManager:
    """
    Zentraler Database Manager für Trading Library
    Optimiert für 191GB RAM und High-Performance Queries
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self.async_pool = None
        self.logger = logging.getLogger(__name__)
        
        # In-Memory Cache (30GB für deine Hardware)
        self.cache_size_bytes = 30 * 1024 * 1024 * 1024  # 30GB
        self.pattern_cache = {}
        self.strategy_cache = {}
        self.performance_cache = {}
        
        self._setup_database()
    
    def _setup_database(self):
        """Initialisiert Datenbankverbindung"""
        try:
            # SQLAlchemy Engine mit Connection Pooling
            connection_string = (
                f"postgresql://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/{self.config.database}"
            )
            
            self.engine = create_engine(
                connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False  # Set to True for SQL debugging
            )
            
            # Session Factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            self.logger.info(f"Database connection established: {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            # Fallback zu SQLite für Development
            self._setup_sqlite_fallback()
    
    def _setup_sqlite_fallback(self):
        """SQLite Fallback für Development ohne PostgreSQL"""
        self.logger.warning("Using SQLite fallback for development")
        
        self.engine = create_engine(
            "sqlite:///trading_library.db",
            pool_pre_ping=True,
            echo=False
        )
        
        self.session_factory = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Erstellt alle Datenbank-Tabellen"""
        try:
            Base.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
            
            # Erstelle Indizes für Performance
            self._create_performance_indexes()
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {e}")
            raise
    
    def _create_performance_indexes(self):
        """Erstellt zusätzliche Performance-Indizes"""
        with self.get_session() as session:
            try:
                # Composite Indizes für häufige Queries
                session.execute("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_type_confidence 
                    ON visual_patterns(pattern_type, confidence_score DESC);
                """)
                
                session.execute("""
                    CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timeframe 
                    ON visual_patterns(symbol, timeframe, created_at DESC);
                """)
                
                session.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategies_performance 
                    ON trading_strategies(total_return DESC, sharpe_ratio DESC);
                """)
                
                session.execute("""
                    CREATE INDEX IF NOT EXISTS idx_performance_pnl 
                    ON pattern_performance(pnl_percent DESC, entry_time DESC);
                """)
                
                session.commit()
                self.logger.info("Performance indexes created")
                
            except Exception as e:
                self.logger.warning(f"Index creation failed: {e}")
                session.rollback()
    
    def get_session(self) -> Session:
        """Gibt neue Database Session zurück"""
        return self.session_factory()
    
    async def get_async_connection(self):
        """Async PostgreSQL Connection für High-Performance Bulk Operations"""
        if not self.async_pool:
            try:
                self.async_pool = await asyncpg.create_pool(
                    host=self.config.host,
                    port=self.config.port,
                    user=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                    min_size=5,
                    max_size=20
                )
            except Exception as e:
                self.logger.error(f"Async pool creation failed: {e}")
                return None
        
        return await self.async_pool.acquire()
    
    def compress_image(self, image: Image.Image, quality: int = 85) -> bytes:
        """Komprimiert PIL Image für Datenbank-Storage"""
        buffer = io.BytesIO()
        
        # Konvertiere zu RGB falls nötig
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        
        # JPEG Komprimierung für Charts
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        return buffer.getvalue()
    
    def decompress_image(self, image_data: bytes, width: int, height: int) -> Image.Image:
        """Dekomprimiert Image-Daten zu PIL Image"""
        buffer = io.BytesIO(image_data)
        image = Image.open(buffer)
        
        # Resize falls nötig
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        return image
    
    def cache_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """Cached Pattern in Memory (30GB Cache)"""
        # Einfache LRU-ähnliche Cache-Implementierung
        if len(self.pattern_cache) > 10000:  # Max 10k Patterns im Cache
            # Entferne älteste Einträge
            oldest_keys = list(self.pattern_cache.keys())[:1000]
            for key in oldest_keys:
                del self.pattern_cache[key]
        
        self.pattern_cache[pattern_id] = pattern_data
    
    def get_cached_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Gibt gecachtes Pattern zurück"""
        return self.pattern_cache.get(pattern_id)
    
    def cache_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]):
        """Cached Strategy in Memory"""
        if len(self.strategy_cache) > 5000:  # Max 5k Strategies im Cache
            oldest_keys = list(self.strategy_cache.keys())[:500]
            for key in oldest_keys:
                del self.strategy_cache[key]
        
        self.strategy_cache[strategy_id] = strategy_data
    
    def get_cached_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Gibt gecachte Strategy zurück"""
        return self.strategy_cache.get(strategy_id)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Gibt Cache-Statistiken zurück"""
        return {
            'pattern_cache_size': len(self.pattern_cache),
            'strategy_cache_size': len(self.strategy_cache),
            'performance_cache_size': len(self.performance_cache),
            'max_cache_size_gb': self.cache_size_bytes / (1024**3)
        }
    
    def optimize_database(self):
        """Führt Datenbank-Optimierungen durch"""
        with self.get_session() as session:
            try:
                # VACUUM und ANALYZE für PostgreSQL
                if 'postgresql' in str(self.engine.url):
                    session.execute("VACUUM ANALYZE visual_patterns;")
                    session.execute("VACUUM ANALYZE trading_strategies;")
                    session.execute("VACUUM ANALYZE pattern_performance;")
                
                # Update Statistiken
                session.execute("ANALYZE;")
                
                session.commit()
                self.logger.info("Database optimization completed")
                
            except Exception as e:
                self.logger.error(f"Database optimization failed: {e}")
                session.rollback()
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Bereinigt alte Daten (Pattern älter als X Tage)"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        with self.get_session() as session:
            try:
                # Lösche alte Patterns
                deleted_patterns = session.query(VisualPatternModel).filter(
                    VisualPatternModel.created_at < cutoff_date,
                    VisualPatternModel.confidence_score < 0.5  # Nur schlechte Patterns löschen
                ).delete()
                
                # Lösche alte Performance-Daten
                deleted_performance = session.query(PatternPerformanceModel).filter(
                    PatternPerformanceModel.entry_time < cutoff_date
                ).delete()
                
                session.commit()
                
                self.logger.info(f"Cleanup completed: {deleted_patterns} patterns, {deleted_performance} performance records deleted")
                
            except Exception as e:
                self.logger.error(f"Data cleanup failed: {e}")
                session.rollback()
    
    def close(self):
        """Schließt alle Datenbankverbindungen"""
        if self.engine:
            self.engine.dispose()
        
        if self.async_pool:
            asyncio.create_task(self.async_pool.close())
        
        # Cache leeren
        self.pattern_cache.clear()
        self.strategy_cache.clear()
        self.performance_cache.clear()
        
        self.logger.info("Database connections closed")


# Global Database Manager Instance
db_manager = None

def get_database_manager() -> DatabaseManager:
    """Singleton Database Manager"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def init_database():
    """Initialisiert Datenbank (für Startup)"""
    manager = get_database_manager()
    manager.create_tables()
    return manager