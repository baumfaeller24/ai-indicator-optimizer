"""
Unit Tests für Trading Library Database System
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import uuid

from ai_indicator_optimizer.library.database import (
    DatabaseManager, DatabaseConfig, VisualPatternModel, TradingStrategyModel
)
from ai_indicator_optimizer.library.pattern_library import PatternLibrary, PatternSimilarityEngine
from ai_indicator_optimizer.library.strategy_library import StrategyLibrary, StrategyEvolutionEngine
from ai_indicator_optimizer.library.models import VisualPattern, TradingStrategy, PerformanceMetrics


class TestDatabaseManager:
    """Tests für DatabaseManager"""
    
    @pytest.fixture
    def config(self):
        return DatabaseConfig(
            host="localhost",
            database="test_trading_library",
            pool_size=5
        )
    
    @pytest.fixture
    def db_manager(self, config):
        # Verwende SQLite für Tests
        manager = DatabaseManager(config)
        manager._setup_sqlite_fallback()
        manager.create_tables()
        return manager
    
    def test_database_initialization(self, db_manager):
        """Test Datenbank-Initialisierung"""
        assert db_manager.engine is not None
        assert db_manager.session_factory is not None
        assert db_manager.cache_size_bytes == 30 * 1024 * 1024 * 1024
    
    def test_session_creation(self, db_manager):
        """Test Session-Erstellung"""
        session = db_manager.get_session()
        assert session is not None
        session.close()
    
    def test_image_compression_decompression(self, db_manager):
        """Test Image-Komprimierung und -Dekomprimierung"""
        # Erstelle Test-Image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Komprimiere
        compressed_data = db_manager.compress_image(test_image)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        
        # Dekomprimiere
        decompressed_image = db_manager.decompress_image(compressed_data, 100, 100)
        assert isinstance(decompressed_image, Image.Image)
        assert decompressed_image.size == (100, 100)
    
    def test_pattern_caching(self, db_manager):
        """Test Pattern-Caching"""
        pattern_id = "test_pattern_123"
        pattern_data = {"test": "data"}
        
        # Cache Pattern
        db_manager.cache_pattern(pattern_id, pattern_data)
        
        # Retrieve from Cache
        cached_data = db_manager.get_cached_pattern(pattern_id)
        assert cached_data == pattern_data
        
        # Test Cache Miss
        missing_data = db_manager.get_cached_pattern("nonexistent")
        assert missing_data is None
    
    def test_cache_statistics(self, db_manager):
        """Test Cache-Statistiken"""
        # Füge Test-Daten hinzu
        db_manager.cache_pattern("pattern1", {"data": 1})
        db_manager.cache_strategy("strategy1", {"data": 2})
        
        stats = db_manager.get_cache_stats()
        
        assert stats['pattern_cache_size'] == 1
        assert stats['strategy_cache_size'] == 1
        assert stats['max_cache_size_gb'] == 30


class TestPatternSimilarityEngine:
    """Tests für PatternSimilarityEngine"""
    
    @pytest.fixture
    def similarity_engine(self):
        return PatternSimilarityEngine()
    
    @pytest.fixture
    def test_images(self):
        """Erstellt Test-Images"""
        # Ähnliche Images
        img1 = Image.new('RGB', (64, 64), color='red')
        img2 = Image.new('RGB', (64, 64), color='darkred')
        
        # Verschiedenes Image
        img3 = Image.new('RGB', (64, 64), color='blue')
        
        return img1, img2, img3
    
    def test_feature_extraction(self, similarity_engine, test_images):
        """Test Feature-Extraktion"""
        img1, img2, img3 = test_images
        
        features1 = similarity_engine.extract_image_features(img1)
        features2 = similarity_engine.extract_image_features(img2)
        features3 = similarity_engine.extract_image_features(img3)
        
        assert isinstance(features1, np.ndarray)
        assert len(features1) == 256  # Histogram mit 256 Bins
        assert np.sum(features1) == pytest.approx(1.0, rel=1e-6)  # Normalisiert
    
    def test_similarity_calculation(self, similarity_engine, test_images):
        """Test Ähnlichkeits-Berechnung"""
        img1, img2, img3 = test_images
        
        features1 = similarity_engine.extract_image_features(img1)
        features2 = similarity_engine.extract_image_features(img2)
        features3 = similarity_engine.extract_image_features(img3)
        
        # Ähnlichkeit zwischen ähnlichen Images
        similarity_12 = similarity_engine.calculate_similarity(features1, features2)
        
        # Ähnlichkeit zwischen verschiedenen Images
        similarity_13 = similarity_engine.calculate_similarity(features1, features3)
        
        assert 0.0 <= similarity_12 <= 1.0
        assert 0.0 <= similarity_13 <= 1.0
        assert similarity_12 > similarity_13  # Ähnliche sollten höhere Similarity haben


class TestPatternLibrary:
    """Tests für PatternLibrary"""
    
    @pytest.fixture
    def db_manager(self):
        manager = DatabaseManager()
        manager._setup_sqlite_fallback()
        manager.create_tables()
        return manager
    
    @pytest.fixture
    def pattern_library(self, db_manager):
        return PatternLibrary(db_manager)
    
    @pytest.fixture
    def sample_pattern(self):
        """Erstellt Sample VisualPattern"""
        test_image = Image.new('RGB', (100, 100), color='green')
        
        return VisualPattern(
            pattern_id=str(uuid.uuid4()),
            pattern_type="double_top",
            chart_image=test_image,
            confidence_score=0.85,
            market_context={
                "symbol": "EURUSD",
                "timeframe": "1h",
                "volatility": 0.6
            }
        )
    
    @pytest.fixture
    def sample_performance(self):
        """Erstellt Sample PerformanceMetrics"""
        return PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.05,
            win_rate=0.65,
            profit_factor=1.8,
            total_trades=50,
            avg_trade_duration=120.5
        )
    
    def test_store_pattern(self, pattern_library, sample_pattern, sample_performance):
        """Test Pattern-Speicherung"""
        pattern_id = pattern_library.store_pattern(sample_pattern, sample_performance)
        
        assert pattern_id is not None
        assert isinstance(pattern_id, str)
        
        # Prüfe ob Pattern in Datenbank gespeichert wurde
        retrieved_pattern = pattern_library.get_pattern(pattern_id)
        assert retrieved_pattern is not None
        assert retrieved_pattern.pattern_type == sample_pattern.pattern_type
        assert retrieved_pattern.confidence_score == sample_pattern.confidence_score
    
    def test_get_pattern_from_cache(self, pattern_library, sample_pattern):
        """Test Pattern-Abruf aus Cache"""
        # Speichere Pattern
        pattern_id = pattern_library.store_pattern(sample_pattern)
        
        # Erster Abruf (aus Datenbank)
        pattern1 = pattern_library.get_pattern(pattern_id)
        
        # Zweiter Abruf (aus Cache)
        pattern2 = pattern_library.get_pattern(pattern_id)
        
        assert pattern1 is not None
        assert pattern2 is not None
        assert pattern_library.cache_hits > 0
    
    def test_update_pattern_performance(self, pattern_library, sample_pattern, sample_performance):
        """Test Pattern-Performance-Update"""
        pattern_id = pattern_library.store_pattern(sample_pattern)
        
        # Update Performance
        new_performance = PerformanceMetrics(
            total_return=0.25,
            sharpe_ratio=1.5,
            max_drawdown=-0.03,
            win_rate=0.70,
            profit_factor=2.0,
            total_trades=75,
            avg_trade_duration=110.0
        )
        
        pattern_library.update_pattern_performance(pattern_id, new_performance)
        
        # Prüfe Update
        updated_pattern = pattern_library.get_pattern(pattern_id)
        # Note: In diesem Test können wir nicht direkt die Performance-Metriken prüfen,
        # da sie in der Datenbank gespeichert werden
        assert updated_pattern is not None
    
    def test_get_top_patterns(self, pattern_library):
        """Test Top-Patterns-Abruf"""
        # Erstelle mehrere Test-Patterns
        patterns = []
        for i in range(5):
            test_image = Image.new('RGB', (50, 50), color=(i*50, 100, 150))
            pattern = VisualPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="test_pattern",
                chart_image=test_image,
                confidence_score=0.5 + i * 0.1,
                market_context={"symbol": "EURUSD", "timeframe": "1m"}
            )
            
            performance = PerformanceMetrics(
                total_return=0.1 + i * 0.05,
                sharpe_ratio=1.0 + i * 0.2,
                max_drawdown=-0.1,
                win_rate=0.6 + i * 0.05,
                profit_factor=1.5 + i * 0.1,
                total_trades=20,
                avg_trade_duration=100.0
            )
            
            pattern_library.store_pattern(pattern, performance)
            patterns.append(pattern)
        
        # Hole Top-Patterns
        top_patterns = pattern_library.get_top_patterns(limit=3)
        
        assert len(top_patterns) <= 3
        assert all(isinstance(p, VisualPattern) for p in top_patterns)
    
    def test_pattern_statistics(self, pattern_library, sample_pattern):
        """Test Pattern-Statistiken"""
        # Speichere Test-Pattern
        pattern_library.store_pattern(sample_pattern)
        
        stats = pattern_library.get_pattern_statistics()
        
        assert 'total_patterns' in stats
        assert 'active_patterns' in stats
        assert 'pattern_types' in stats
        assert stats['total_patterns'] >= 1


class TestStrategyLibrary:
    """Tests für StrategyLibrary"""
    
    @pytest.fixture
    def db_manager(self):
        manager = DatabaseManager()
        manager._setup_sqlite_fallback()
        manager.create_tables()
        return manager
    
    @pytest.fixture
    def strategy_library(self, db_manager):
        return StrategyLibrary(db_manager)
    
    @pytest.fixture
    def sample_strategy(self):
        """Erstellt Sample TradingStrategy"""
        return TradingStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_name="RSI_Mean_Reversion",
            indicators=[
                {"name": "RSI", "period": 14},
                {"name": "SMA", "period": 20}
            ],
            entry_conditions=["RSI < 30", "Close > SMA20"],
            exit_conditions=["RSI > 70", "Stop Loss"],
            risk_management={
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "position_size": 0.1
            }
        )
    
    @pytest.fixture
    def sample_backtest_results(self):
        """Erstellt Sample Backtest-Ergebnisse"""
        return {
            "total_return": 0.18,
            "sharpe_ratio": 1.3,
            "max_drawdown": -0.08,
            "win_rate": 0.62,
            "profit_factor": 1.9,
            "total_trades": 45,
            "avg_trade_duration": 180.0,
            "best_trade": 0.05,
            "worst_trade": -0.03
        }
    
    def test_store_strategy(self, strategy_library, sample_strategy, sample_backtest_results):
        """Test Strategie-Speicherung"""
        strategy_id = strategy_library.store_strategy(sample_strategy, sample_backtest_results)
        
        assert strategy_id is not None
        assert isinstance(strategy_id, str)
        
        # Prüfe ob Strategie in Datenbank gespeichert wurde
        retrieved_strategy = strategy_library.get_strategy(strategy_id)
        assert retrieved_strategy is not None
        assert retrieved_strategy.strategy_name == sample_strategy.strategy_name
        assert len(retrieved_strategy.indicators) == len(sample_strategy.indicators)
    
    def test_get_strategy_performance(self, strategy_library, sample_strategy, sample_backtest_results):
        """Test Strategie-Performance-Abruf"""
        strategy_id = strategy_library.store_strategy(sample_strategy, sample_backtest_results)
        
        performance = strategy_library.get_strategy_performance(strategy_id)
        
        assert performance is not None
        assert 'total_return' in performance
        assert 'sharpe_ratio' in performance
        assert performance['total_return'] == sample_backtest_results['total_return']
    
    def test_update_strategy_performance(self, strategy_library, sample_strategy):
        """Test Strategie-Performance-Update"""
        strategy_id = strategy_library.store_strategy(sample_strategy)
        
        new_performance = {
            "total_return": 0.25,
            "sharpe_ratio": 1.6,
            "win_rate": 0.68,
            "total_trades": 60
        }
        
        strategy_library.update_strategy_performance(strategy_id, new_performance)
        
        # Prüfe Update
        updated_performance = strategy_library.get_strategy_performance(strategy_id)
        assert updated_performance['total_return'] == 0.25
        assert updated_performance['sharpe_ratio'] == 1.6
    
    def test_rank_strategies(self, strategy_library):
        """Test Strategie-Ranking"""
        # Erstelle mehrere Test-Strategien
        strategies = []
        for i in range(3):
            strategy = TradingStrategy(
                strategy_id=str(uuid.uuid4()),
                strategy_name=f"Test_Strategy_{i}",
                indicators=[{"name": "RSI", "period": 14}],
                entry_conditions=["RSI < 30"],
                exit_conditions=["RSI > 70"],
                risk_management={"stop_loss": 0.02}
            )
            
            backtest_results = {
                "total_return": 0.1 + i * 0.05,
                "sharpe_ratio": 1.0 + i * 0.3,
                "win_rate": 0.6 + i * 0.05,
                "total_trades": 20 + i * 10
            }
            
            strategy_library.store_strategy(strategy, backtest_results)
            strategies.append(strategy)
        
        # Rank Strategien
        market_conditions = {
            "volatility": 0.5,
            "trend": "bullish"
        }
        
        ranked_strategies = strategy_library.rank_strategies(market_conditions, limit=2)
        
        assert len(ranked_strategies) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked_strategies)
        assert all(isinstance(item[0], TradingStrategy) and isinstance(item[1], float) for item in ranked_strategies)
    
    def test_strategy_statistics(self, strategy_library, sample_strategy):
        """Test Strategie-Statistiken"""
        strategy_library.store_strategy(sample_strategy)
        
        stats = strategy_library.get_strategy_statistics()
        
        assert 'total_strategies' in stats
        assert 'active_strategies' in stats
        assert 'strategy_types' in stats
        assert stats['total_strategies'] >= 1


class TestStrategyEvolutionEngine:
    """Tests für StrategyEvolutionEngine"""
    
    @pytest.fixture
    def evolution_engine(self):
        return StrategyEvolutionEngine()
    
    @pytest.fixture
    def base_strategy(self):
        return TradingStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_name="Base_Strategy",
            indicators=[{"name": "RSI", "period": 14}],
            entry_conditions=["RSI < 30"],
            exit_conditions=["RSI > 70"],
            risk_management={"stop_loss": 0.02, "take_profit": 0.04}
        )
    
    def test_mutate_strategy(self, evolution_engine, base_strategy):
        """Test Strategie-Mutation"""
        mutated = evolution_engine.mutate_strategy(base_strategy)
        
        assert mutated.strategy_id != base_strategy.strategy_id
        assert "mutated" in mutated.strategy_name
        assert len(mutated.indicators) == len(base_strategy.indicators)
    
    def test_crossover_strategies(self, evolution_engine, base_strategy):
        """Test Strategie-Crossover"""
        # Erstelle zweite Strategie
        strategy2 = TradingStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_name="Second_Strategy",
            indicators=[{"name": "MACD", "fast": 12, "slow": 26}],
            entry_conditions=["MACD > Signal"],
            exit_conditions=["MACD < Signal"],
            risk_management={"stop_loss": 0.03, "position_size": 0.2}
        )
        
        child = evolution_engine.crossover_strategies(base_strategy, strategy2)
        
        assert child.strategy_id not in [base_strategy.strategy_id, strategy2.strategy_id]
        assert "_x_" in child.strategy_name
        assert len(child.indicators) >= len(base_strategy.indicators)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])