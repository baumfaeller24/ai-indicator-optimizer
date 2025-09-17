"""
Strategy Library - High-Performance CRUD Operations für Trading Strategien
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import logging
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session
import uuid
import json
import copy

from .models import TradingStrategy, PerformanceMetrics
from .database import (
    DatabaseManager, get_database_manager, 
    TradingStrategyModel
)


class StrategyEvolutionEngine:
    """
    Engine für Strategie-Evolution und Optimierung
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
    
    def mutate_strategy(self, strategy: TradingStrategy) -> TradingStrategy:
        """
        Mutiert eine Strategie für Evolution
        """
        mutated = copy.deepcopy(strategy)
        mutated.strategy_id = str(uuid.uuid4())
        mutated.strategy_name = f"{strategy.strategy_name}_mutated"
        
        # Mutiere Indikator-Parameter
        for indicator in mutated.indicators:
            if 'period' in indicator and np.random.random() < self.mutation_rate:
                current_period = indicator['period']
                # Variiere Period um ±20%
                variation = int(current_period * 0.2 * (np.random.random() - 0.5))
                indicator['period'] = max(5, current_period + variation)
        
        # Mutiere Risk Management
        if np.random.random() < self.mutation_rate:
            if 'stop_loss' in mutated.risk_management:
                current_sl = mutated.risk_management['stop_loss']
                variation = current_sl * 0.1 * (np.random.random() - 0.5)
                mutated.risk_management['stop_loss'] = max(0.005, current_sl + variation)
        
        return mutated
    
    def crossover_strategies(self, parent1: TradingStrategy, parent2: TradingStrategy) -> TradingStrategy:
        """
        Kreuzt zwei Strategien für Evolution
        """
        child = copy.deepcopy(parent1)
        child.strategy_id = str(uuid.uuid4())
        child.strategy_name = f"{parent1.strategy_name}_x_{parent2.strategy_name}"
        
        # Mische Indikatoren
        if len(parent2.indicators) > 0 and np.random.random() < self.crossover_rate:
            # Nehme zufälligen Indikator von Parent2
            random_indicator = np.random.choice(parent2.indicators)
            child.indicators.append(copy.deepcopy(random_indicator))
        
        # Mische Entry/Exit Conditions
        if len(parent2.entry_conditions) > 0 and np.random.random() < self.crossover_rate:
            random_condition = np.random.choice(parent2.entry_conditions)
            if random_condition not in child.entry_conditions:
                child.entry_conditions.append(random_condition)
        
        # Mische Risk Management
        for key, value in parent2.risk_management.items():
            if np.random.random() < self.crossover_rate:
                child.risk_management[key] = value
        
        return child


class StrategyRankingEngine:
    """
    Engine für intelligentes Strategie-Ranking basierend auf Marktbedingungen
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_strategy_score(self, strategy: TradingStrategy, 
                               market_conditions: Dict[str, Any]) -> float:
        """
        Berechnet Strategie-Score basierend auf Performance und Marktbedingungen
        """
        if not strategy.performance_metrics:
            return 0.0
        
        base_score = 0.0
        
        # Performance-Komponenten
        if 'total_return' in strategy.performance_metrics:
            base_score += strategy.performance_metrics['total_return'] * 0.3
        
        if 'sharpe_ratio' in strategy.performance_metrics:
            base_score += strategy.performance_metrics['sharpe_ratio'] * 0.25
        
        if 'win_rate' in strategy.performance_metrics:
            base_score += strategy.performance_metrics['win_rate'] * 0.2
        
        if 'profit_factor' in strategy.performance_metrics:
            base_score += (strategy.performance_metrics['profit_factor'] - 1) * 0.15
        
        if 'max_drawdown' in strategy.performance_metrics:
            # Niedrigerer Drawdown = höherer Score
            base_score += (1 - abs(strategy.performance_metrics['max_drawdown'])) * 0.1
        
        # Marktbedingungen-Anpassung
        market_volatility = market_conditions.get('volatility', 0.5)
        market_trend = market_conditions.get('trend', 'neutral')
        
        # Anpassung basierend auf Strategie-Typ
        if 'trend_following' in strategy.strategy_name.lower():
            if market_trend in ['bullish', 'bearish']:
                base_score *= 1.2  # Bonus für Trend-Strategien in Trending Markets
            else:
                base_score *= 0.8  # Malus in Sideways Markets
        
        elif 'mean_reversion' in strategy.strategy_name.lower():
            if market_trend == 'neutral':
                base_score *= 1.2  # Bonus für Mean Reversion in Sideways Markets
            else:
                base_score *= 0.8
        
        # Volatilitäts-Anpassung
        if market_volatility > 0.7:  # Hohe Volatilität
            if any('atr' in str(indicator).lower() for indicator in strategy.indicators):
                base_score *= 1.1  # Bonus für ATR-basierte Strategien
        
        return max(0.0, base_score)


class StrategyLibrary:
    """
    High-Performance Bibliothek für Trading-Strategien
    Optimiert für 191GB RAM und PostgreSQL
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        self.evolution_engine = StrategyEvolutionEngine()
        self.ranking_engine = StrategyRankingEngine()
        self.logger = logging.getLogger(__name__)
        
        # Performance-Metriken
        self.query_count = 0
        self.cache_hits = 0
    
    def store_strategy(self, strategy: TradingStrategy, backtest_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Speichert Strategie mit Backtest-Ergebnissen in Datenbank
        """
        try:
            with self.db_manager.get_session() as session:
                # Extrahiere Performance-Metriken aus Backtest-Ergebnissen
                performance_metrics = strategy.performance_metrics or {}
                if backtest_results:
                    performance_metrics.update(backtest_results)
                
                # Erstelle Database Model
                db_strategy = TradingStrategyModel(
                    id=uuid.UUID(strategy.strategy_id) if strategy.strategy_id else uuid.uuid4(),
                    strategy_name=strategy.strategy_name,
                    strategy_type=self._classify_strategy_type(strategy),
                    indicators=strategy.indicators,
                    entry_conditions=strategy.entry_conditions,
                    exit_conditions=strategy.exit_conditions,
                    risk_management=strategy.risk_management,
                    performance_metrics=performance_metrics,
                    backtest_results=backtest_results,
                    total_return=performance_metrics.get('total_return'),
                    sharpe_ratio=performance_metrics.get('sharpe_ratio'),
                    win_rate=performance_metrics.get('win_rate'),
                    profit_factor=performance_metrics.get('profit_factor'),
                    max_drawdown=performance_metrics.get('max_drawdown'),
                    total_trades=performance_metrics.get('total_trades')
                )
                
                session.add(db_strategy)
                session.commit()
                
                strategy_id = str(db_strategy.id)
                
                # Cache Strategy
                self.db_manager.cache_strategy(strategy_id, {
                    'strategy': strategy,
                    'backtest_results': backtest_results,
                    'db_model': db_strategy
                })
                
                self.logger.info(f"Stored strategy {strategy.strategy_name} with ID {strategy_id}")
                return strategy_id
                
        except Exception as e:
            self.logger.error(f"Failed to store strategy: {e}")
            raise
    
    def get_strategy(self, strategy_id: str) -> Optional[TradingStrategy]:
        """
        Lädt Strategie aus Datenbank oder Cache
        """
        # Prüfe Cache zuerst
        cached = self.db_manager.get_cached_strategy(strategy_id)
        if cached:
            self.cache_hits += 1
            return cached['strategy']
        
        try:
            with self.db_manager.get_session() as session:
                db_strategy = session.query(TradingStrategyModel).filter(
                    TradingStrategyModel.id == uuid.UUID(strategy_id)
                ).first()
                
                if not db_strategy:
                    return None
                
                # Erstelle TradingStrategy Object
                strategy = TradingStrategy(
                    strategy_id=str(db_strategy.id),
                    strategy_name=db_strategy.strategy_name,
                    indicators=db_strategy.indicators,
                    entry_conditions=db_strategy.entry_conditions,
                    exit_conditions=db_strategy.exit_conditions,
                    risk_management=db_strategy.risk_management,
                    performance_metrics=db_strategy.performance_metrics,
                    created_at=db_strategy.created_at
                )
                
                # Cache für zukünftige Zugriffe
                self.db_manager.cache_strategy(strategy_id, {
                    'strategy': strategy,
                    'backtest_results': db_strategy.backtest_results
                })
                
                self.query_count += 1
                return strategy
                
        except Exception as e:
            self.logger.error(f"Failed to get strategy {strategy_id}: {e}")
            return None
    
    def evolve_strategies(self, base_strategies: List[TradingStrategy], 
                         generations: int = 3,
                         population_size: int = 50) -> List[TradingStrategy]:
        """
        Entwickelt Strategien durch genetische Algorithmen weiter
        """
        try:
            if not base_strategies:
                return []
            
            current_population = base_strategies.copy()
            
            for generation in range(generations):
                self.logger.info(f"Evolution generation {generation + 1}/{generations}")
                
                new_population = []
                
                # Behalte beste Strategien (Elitism)
                elite_count = max(1, len(current_population) // 4)
                sorted_strategies = sorted(
                    current_population,
                    key=lambda s: s.performance_metrics.get('total_return', 0) if s.performance_metrics else 0,
                    reverse=True
                )
                new_population.extend(sorted_strategies[:elite_count])
                
                # Generiere neue Strategien durch Mutation und Crossover
                while len(new_population) < population_size:
                    if np.random.random() < self.evolution_engine.crossover_rate and len(current_population) >= 2:
                        # Crossover
                        parent1, parent2 = np.random.choice(current_population, 2, replace=False)
                        child = self.evolution_engine.crossover_strategies(parent1, parent2)
                        new_population.append(child)
                    else:
                        # Mutation
                        parent = np.random.choice(current_population)
                        mutated = self.evolution_engine.mutate_strategy(parent)
                        new_population.append(mutated)
                
                current_population = new_population[:population_size]
            
            self.logger.info(f"Evolution completed: {len(current_population)} strategies generated")
            return current_population
            
        except Exception as e:
            self.logger.error(f"Strategy evolution failed: {e}")
            return base_strategies
    
    def rank_strategies(self, market_conditions: Dict[str, Any], 
                       strategy_type: Optional[str] = None,
                       min_trades: int = 10,
                       limit: int = 10) -> List[Tuple[TradingStrategy, float]]:
        """
        Rankt Strategien nach Performance und Marktbedingungen
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(TradingStrategyModel).filter(
                    and_(
                        TradingStrategyModel.is_active == True,
                        TradingStrategyModel.total_trades >= min_trades
                    )
                )
                
                if strategy_type:
                    query = query.filter(TradingStrategyModel.strategy_type == strategy_type)
                
                db_strategies = query.all()
                
                ranked_strategies = []
                
                for db_strategy in db_strategies:
                    try:
                        # Erstelle TradingStrategy Object
                        strategy = TradingStrategy(
                            strategy_id=str(db_strategy.id),
                            strategy_name=db_strategy.strategy_name,
                            indicators=db_strategy.indicators,
                            entry_conditions=db_strategy.entry_conditions,
                            exit_conditions=db_strategy.exit_conditions,
                            risk_management=db_strategy.risk_management,
                            performance_metrics=db_strategy.performance_metrics,
                            created_at=db_strategy.created_at
                        )
                        
                        # Berechne Score basierend auf Marktbedingungen
                        score = self.ranking_engine.calculate_strategy_score(strategy, market_conditions)
                        
                        ranked_strategies.append((strategy, score))
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing strategy {db_strategy.id}: {e}")
                        continue
                
                # Sortiere nach Score
                ranked_strategies.sort(key=lambda x: x[1], reverse=True)
                
                self.logger.info(f"Ranked {len(ranked_strategies)} strategies")
                return ranked_strategies[:limit]
                
        except Exception as e:
            self.logger.error(f"Strategy ranking failed: {e}")
            return []
    
    def get_strategy_performance(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        Gibt detaillierte Performance-Daten für Strategie zurück
        """
        try:
            with self.db_manager.get_session() as session:
                db_strategy = session.query(TradingStrategyModel).filter(
                    TradingStrategyModel.id == uuid.UUID(strategy_id)
                ).first()
                
                if not db_strategy:
                    return None
                
                return {
                    'performance_metrics': db_strategy.performance_metrics,
                    'backtest_results': db_strategy.backtest_results,
                    'total_return': db_strategy.total_return,
                    'sharpe_ratio': db_strategy.sharpe_ratio,
                    'win_rate': db_strategy.win_rate,
                    'profit_factor': db_strategy.profit_factor,
                    'max_drawdown': db_strategy.max_drawdown,
                    'total_trades': db_strategy.total_trades,
                    'created_at': db_strategy.created_at,
                    'updated_at': db_strategy.updated_at
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get strategy performance: {e}")
            return None
    
    def update_strategy_performance(self, strategy_id: str, 
                                  performance_metrics: Dict[str, Any],
                                  backtest_results: Optional[Dict[str, Any]] = None):
        """
        Aktualisiert Strategie-Performance
        """
        try:
            with self.db_manager.get_session() as session:
                db_strategy = session.query(TradingStrategyModel).filter(
                    TradingStrategyModel.id == uuid.UUID(strategy_id)
                ).first()
                
                if db_strategy:
                    db_strategy.performance_metrics = performance_metrics
                    if backtest_results:
                        db_strategy.backtest_results = backtest_results
                    
                    # Update einzelne Performance-Felder für bessere Queries
                    db_strategy.total_return = performance_metrics.get('total_return')
                    db_strategy.sharpe_ratio = performance_metrics.get('sharpe_ratio')
                    db_strategy.win_rate = performance_metrics.get('win_rate')
                    db_strategy.profit_factor = performance_metrics.get('profit_factor')
                    db_strategy.max_drawdown = performance_metrics.get('max_drawdown')
                    db_strategy.total_trades = performance_metrics.get('total_trades')
                    db_strategy.updated_at = datetime.now(timezone.utc)
                    
                    session.commit()
                    
                    # Update Cache
                    cached = self.db_manager.get_cached_strategy(strategy_id)
                    if cached:
                        cached['strategy'].performance_metrics = performance_metrics
                        if backtest_results:
                            cached['backtest_results'] = backtest_results
                    
                    self.logger.info(f"Updated performance for strategy {strategy_id}")
                else:
                    self.logger.warning(f"Strategy {strategy_id} not found for performance update")
                    
        except Exception as e:
            self.logger.error(f"Failed to update strategy performance: {e}")
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """
        Gibt Strategy-Library Statistiken zurück
        """
        try:
            with self.db_manager.get_session() as session:
                # Gesamt-Statistiken
                total_strategies = session.query(func.count(TradingStrategyModel.id)).scalar()
                active_strategies = session.query(func.count(TradingStrategyModel.id)).filter(
                    TradingStrategyModel.is_active == True
                ).scalar()
                
                # Strategie-Typen
                strategy_types = session.query(
                    TradingStrategyModel.strategy_type,
                    func.count(TradingStrategyModel.id)
                ).group_by(TradingStrategyModel.strategy_type).all()
                
                # Performance-Statistiken
                avg_return = session.query(
                    func.avg(TradingStrategyModel.total_return)
                ).filter(
                    and_(
                        TradingStrategyModel.is_active == True,
                        TradingStrategyModel.total_return.isnot(None)
                    )
                ).scalar()
                
                avg_sharpe = session.query(
                    func.avg(TradingStrategyModel.sharpe_ratio)
                ).filter(
                    and_(
                        TradingStrategyModel.is_active == True,
                        TradingStrategyModel.sharpe_ratio.isnot(None)
                    )
                ).scalar()
                
                avg_win_rate = session.query(
                    func.avg(TradingStrategyModel.win_rate)
                ).filter(
                    and_(
                        TradingStrategyModel.is_active == True,
                        TradingStrategyModel.win_rate.isnot(None)
                    )
                ).scalar()
                
                return {
                    'total_strategies': total_strategies or 0,
                    'active_strategies': active_strategies or 0,
                    'strategy_types': dict(strategy_types) if strategy_types else {},
                    'avg_return': float(avg_return) if avg_return else 0.0,
                    'avg_sharpe_ratio': float(avg_sharpe) if avg_sharpe else 0.0,
                    'avg_win_rate': float(avg_win_rate) if avg_win_rate else 0.0,
                    'query_count': self.query_count,
                    'cache_hits': self.cache_hits,
                    'cache_hit_rate': self.cache_hits / max(1, self.query_count)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get strategy statistics: {e}")
            return {}
    
    def _classify_strategy_type(self, strategy: TradingStrategy) -> str:
        """
        Klassifiziert Strategie-Typ basierend auf Indikatoren und Conditions
        """
        strategy_name_lower = strategy.strategy_name.lower()
        
        # Trend Following
        if any(keyword in strategy_name_lower for keyword in ['trend', 'momentum', 'breakout']):
            return 'trend_following'
        
        # Mean Reversion
        if any(keyword in strategy_name_lower for keyword in ['reversion', 'rsi', 'oversold', 'overbought']):
            return 'mean_reversion'
        
        # Scalping
        if any(keyword in strategy_name_lower for keyword in ['scalp', 'quick', 'fast']):
            return 'scalping'
        
        # Swing Trading
        if any(keyword in strategy_name_lower for keyword in ['swing', 'daily', 'weekly']):
            return 'swing_trading'
        
        # Basierend auf Indikatoren
        indicator_names = [str(ind).lower() for ind in strategy.indicators]
        
        if any('macd' in ind or 'ema' in ind for ind in indicator_names):
            return 'trend_following'
        
        if any('rsi' in ind or 'stoch' in ind for ind in indicator_names):
            return 'mean_reversion'
        
        return 'mixed'
    
    async def bulk_import_strategies(self, strategies: List[Tuple[TradingStrategy, Optional[Dict[str, Any]]]]) -> int:
        """
        Bulk-Import von Strategien für bessere Performance
        """
        try:
            imported_count = 0
            
            # Verwende Async Connection für Bulk Operations
            async_conn = await self.db_manager.get_async_connection()
            if not async_conn:
                # Fallback zu synchronem Import
                for strategy, backtest_results in strategies:
                    try:
                        self.store_strategy(strategy, backtest_results)
                        imported_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to import strategy: {e}")
                return imported_count
            
            try:
                # Prepare bulk insert data
                insert_data = []
                for strategy, backtest_results in strategies:
                    performance_metrics = strategy.performance_metrics or {}
                    if backtest_results:
                        performance_metrics.update(backtest_results)
                    
                    insert_data.append({
                        'id': uuid.uuid4(),
                        'strategy_name': strategy.strategy_name,
                        'strategy_type': self._classify_strategy_type(strategy),
                        'indicators': strategy.indicators,
                        'entry_conditions': strategy.entry_conditions,
                        'exit_conditions': strategy.exit_conditions,
                        'risk_management': strategy.risk_management,
                        'performance_metrics': performance_metrics,
                        'backtest_results': backtest_results,
                        'total_return': performance_metrics.get('total_return'),
                        'sharpe_ratio': performance_metrics.get('sharpe_ratio'),
                        'win_rate': performance_metrics.get('win_rate'),
                        'profit_factor': performance_metrics.get('profit_factor'),
                        'max_drawdown': performance_metrics.get('max_drawdown'),
                        'total_trades': performance_metrics.get('total_trades'),
                        'created_at': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc),
                        'is_active': True
                    })
                
                # Bulk insert
                await async_conn.executemany("""
                    INSERT INTO trading_strategies (
                        id, strategy_name, strategy_type, indicators, entry_conditions,
                        exit_conditions, risk_management, performance_metrics, backtest_results,
                        total_return, sharpe_ratio, win_rate, profit_factor, max_drawdown,
                        total_trades, created_at, updated_at, is_active
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                    )
                """, [
                    (
                        data['id'], data['strategy_name'], data['strategy_type'],
                        json.dumps(data['indicators']), json.dumps(data['entry_conditions']),
                        json.dumps(data['exit_conditions']), json.dumps(data['risk_management']),
                        json.dumps(data['performance_metrics']), json.dumps(data['backtest_results']),
                        data['total_return'], data['sharpe_ratio'], data['win_rate'],
                        data['profit_factor'], data['max_drawdown'], data['total_trades'],
                        data['created_at'], data['updated_at'], data['is_active']
                    ) for data in insert_data
                ])
                
                imported_count = len(insert_data)
                self.logger.info(f"Bulk imported {imported_count} strategies")
                
            finally:
                await self.db_manager.async_pool.release(async_conn)
            
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Bulk import failed: {e}")
            return 0