#!/usr/bin/env python3
"""
Strategy Logic Generator für Entry/Exit-Conditions mit Confidence-Scoring
Phase 3 Implementation - Enhanced Pine Script Code Generator

Features:
- Automatische Entry/Exit-Logic-Generierung
- Confidence-basierte Position-Sizing
- Multi-Condition-Strategy-Building
- Risk-Management-Integration
- Enhanced Feature Integration
- Real-time Strategy-Anpassung
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import json
from pathlib import Path

# Local Imports
from .enhanced_feature_extractor import EnhancedFeatureExtractor
from .confidence_position_sizer import ConfidencePositionSizer
from .pine_script_generator import PineScriptGenerator, PineScriptConfig, IndicatorType


class StrategyType(Enum):
    """Unterstützte Strategy-Typen"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    AI_PATTERN = "ai_pattern"
    MULTI_TIMEFRAME = "multi_timeframe"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"


class ConditionType(Enum):
    """Typen von Trading-Conditions"""
    TECHNICAL_INDICATOR = "technical_indicator"
    PRICE_ACTION = "price_action"
    VOLUME_ANALYSIS = "volume_analysis"
    TIME_BASED = "time_based"
    VOLATILITY = "volatility"
    AI_CONFIDENCE = "ai_confidence"
    RISK_MANAGEMENT = "risk_management"


class LogicOperator(Enum):
    """Logische Operatoren für Condition-Verknüpfung"""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


@dataclass
class TradingCondition:
    """Einzelne Trading-Condition"""
    name: str
    condition_type: ConditionType
    indicator: str  # z.B. "rsi", "macd", "price"
    operator: str  # z.B. ">", "<", "crossover", "crossunder"
    value: Union[float, str]  # Threshold-Wert oder andere Condition
    weight: float = 1.0  # Gewichtung der Condition
    confidence_threshold: float = 0.0  # Mindest-Confidence
    enabled: bool = True


@dataclass
class StrategyLogic:
    """Komplette Strategy-Logic"""
    name: str
    strategy_type: StrategyType
    entry_conditions: List[TradingCondition]
    exit_conditions: List[TradingCondition]
    entry_logic_operator: LogicOperator = LogicOperator.AND
    exit_logic_operator: LogicOperator = LogicOperator.OR
    
    # Risk Management
    stop_loss_pct: float = 0.02  # 2% Stop Loss
    take_profit_pct: float = 0.04  # 4% Take Profit
    max_position_size: float = 0.1  # 10% des Portfolios
    
    # Confidence-basierte Anpassungen
    confidence_scaling: bool = True
    min_confidence: float = 0.6
    max_confidence: float = 0.95
    
    # Timing
    max_hold_time: Optional[int] = None  # Stunden
    min_hold_time: Optional[int] = None  # Stunden


@dataclass
class StrategyPerformance:
    """Performance-Metriken einer Strategy"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    
    # Confidence-spezifische Metriken
    avg_confidence: float
    confidence_accuracy: float  # Wie gut Confidence die Performance vorhersagt
    high_confidence_win_rate: float  # Win Rate bei hoher Confidence
    low_confidence_win_rate: float  # Win Rate bei niedriger Confidence


class StrategyLogicGenerator:
    """
    Haupt-Generator für Strategy-Logic
    
    Features:
    - Automatische Entry/Exit-Logic-Generierung
    - Confidence-Integration
    - Multi-Condition-Strategies
    - Performance-Optimierung
    - Risk-Management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.feature_extractor = EnhancedFeatureExtractor()
        self.position_sizer = ConfidencePositionSizer()
        self.pine_generator = PineScriptGenerator()
        
        # Strategy-Templates
        self.strategy_templates: Dict[StrategyType, StrategyLogic] = {}
        self._load_strategy_templates()
        
        # Performance-Tracking
        self.generated_strategies: List[StrategyLogic] = []
        self.performance_history: List[StrategyPerformance] = []
        
        # Statistiken
        self.stats = {
            "strategies_generated": 0,
            "strategies_tested": 0,
            "best_sharpe_ratio": -np.inf,
            "best_win_rate": 0.0,
            "avg_confidence_accuracy": 0.0,
            "total_generation_time": 0.0
        }
        
        self.logger.info("StrategyLogicGenerator initialized")
    
    def _load_strategy_templates(self):
        """Lade vordefinierte Strategy-Templates"""
        
        # Trend Following Template
        trend_following = StrategyLogic(
            name="Trend Following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            entry_conditions=[
                TradingCondition(
                    name="Price above MA",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="sma_20",
                    operator=">",
                    value="close",
                    weight=1.0
                ),
                TradingCondition(
                    name="RSI not overbought",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator="<",
                    value=70,
                    weight=0.8
                ),
                TradingCondition(
                    name="Volume confirmation",
                    condition_type=ConditionType.VOLUME_ANALYSIS,
                    indicator="volume_ratio",
                    operator=">",
                    value=1.2,
                    weight=0.6
                )
            ],
            exit_conditions=[
                TradingCondition(
                    name="Price below MA",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="sma_20",
                    operator="<",
                    value="close",
                    weight=1.0
                ),
                TradingCondition(
                    name="RSI overbought",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator=">",
                    value=75,
                    weight=0.9
                )
            ],
            stop_loss_pct=0.015,
            take_profit_pct=0.03
        )
        self.strategy_templates[StrategyType.TREND_FOLLOWING] = trend_following
        
        # Mean Reversion Template
        mean_reversion = StrategyLogic(
            name="Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            entry_conditions=[
                TradingCondition(
                    name="RSI oversold",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator="<",
                    value=30,
                    weight=1.0
                ),
                TradingCondition(
                    name="Price near BB lower",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="bb_position",
                    operator="<",
                    value=0.2,
                    weight=0.9
                ),
                TradingCondition(
                    name="Low volatility",
                    condition_type=ConditionType.VOLATILITY,
                    indicator="volatility_5",
                    operator="<",
                    value=0.002,
                    weight=0.7
                )
            ],
            exit_conditions=[
                TradingCondition(
                    name="RSI normalized",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator=">",
                    value=50,
                    weight=1.0
                ),
                TradingCondition(
                    name="Price at BB middle",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="bb_position",
                    operator=">",
                    value=0.5,
                    weight=0.8
                )
            ],
            entry_logic_operator=LogicOperator.AND,
            exit_logic_operator=LogicOperator.OR,
            stop_loss_pct=0.01,
            take_profit_pct=0.02
        )
        self.strategy_templates[StrategyType.MEAN_REVERSION] = mean_reversion
        
        # Breakout Template
        breakout = StrategyLogic(
            name="Breakout",
            strategy_type=StrategyType.BREAKOUT,
            entry_conditions=[
                TradingCondition(
                    name="Price breakout high",
                    condition_type=ConditionType.PRICE_ACTION,
                    indicator="close",
                    operator=">",
                    value="bb_upper",
                    weight=1.0
                ),
                TradingCondition(
                    name="High volume",
                    condition_type=ConditionType.VOLUME_ANALYSIS,
                    indicator="volume_ratio",
                    operator=">",
                    value=1.5,
                    weight=0.9
                ),
                TradingCondition(
                    name="Volatility expansion",
                    condition_type=ConditionType.VOLATILITY,
                    indicator="volatility_5",
                    operator=">",
                    value=0.003,
                    weight=0.7
                )
            ],
            exit_conditions=[
                TradingCondition(
                    name="Price back in range",
                    condition_type=ConditionType.PRICE_ACTION,
                    indicator="close",
                    operator="<",
                    value="bb_upper",
                    weight=1.0
                ),
                TradingCondition(
                    name="Volume decline",
                    condition_type=ConditionType.VOLUME_ANALYSIS,
                    indicator="volume_ratio",
                    operator="<",
                    value=0.8,
                    weight=0.6
                )
            ],
            stop_loss_pct=0.025,
            take_profit_pct=0.05
        )
        self.strategy_templates[StrategyType.BREAKOUT] = breakout
        
        # AI Pattern Template
        ai_pattern = StrategyLogic(
            name="AI Pattern Recognition",
            strategy_type=StrategyType.AI_PATTERN,
            entry_conditions=[
                TradingCondition(
                    name="AI Confidence High",
                    condition_type=ConditionType.AI_CONFIDENCE,
                    indicator="ai_confidence",
                    operator=">",
                    value=0.75,
                    weight=1.0
                ),
                TradingCondition(
                    name="Pattern Bullish",
                    condition_type=ConditionType.AI_CONFIDENCE,
                    indicator="pattern_direction",
                    operator="==",
                    value="bullish",
                    weight=0.9
                ),
                TradingCondition(
                    name="Market regime trending",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="regime_trending",
                    operator=">",
                    value=0.6,
                    weight=0.7
                )
            ],
            exit_conditions=[
                TradingCondition(
                    name="AI Confidence Low",
                    condition_type=ConditionType.AI_CONFIDENCE,
                    indicator="ai_confidence",
                    operator="<",
                    value=0.4,
                    weight=1.0
                ),
                TradingCondition(
                    name="Pattern Bearish",
                    condition_type=ConditionType.AI_CONFIDENCE,
                    indicator="pattern_direction",
                    operator="==",
                    value="bearish",
                    weight=0.8
                )
            ],
            confidence_scaling=True,
            min_confidence=0.7,
            max_confidence=0.95,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        self.strategy_templates[StrategyType.AI_PATTERN] = ai_pattern
    
    def generate_strategy(
        self,
        strategy_type: StrategyType,
        market_data: pd.DataFrame,
        optimization_target: str = "sharpe_ratio",
        custom_conditions: Optional[List[TradingCondition]] = None
    ) -> Dict[str, Any]:
        """
        Generiere optimierte Strategy-Logic
        
        Args:
            strategy_type: Typ der Strategy
            market_data: Historische Marktdaten
            optimization_target: Optimierungs-Ziel
            custom_conditions: Optionale custom Conditions
            
        Returns:
            Generierte Strategy mit Performance-Metriken
        """
        try:
            start_time = datetime.now()
            
            self.logger.info(f"Generating strategy: {strategy_type.value}")
            
            # Basis-Template laden
            if strategy_type in self.strategy_templates:
                base_strategy = self.strategy_templates[strategy_type]
            else:
                base_strategy = self._create_default_strategy(strategy_type)
            
            # Custom Conditions hinzufügen
            if custom_conditions:
                base_strategy.entry_conditions.extend(custom_conditions)
            
            # Market Data vorbereiten
            prepared_data = self._prepare_market_data(market_data)
            
            # Strategy optimieren
            optimized_strategy = self._optimize_strategy(base_strategy, prepared_data, optimization_target)
            
            # Performance testen
            performance = self._test_strategy_performance(optimized_strategy, prepared_data)
            
            # Pine Script generieren
            pine_script = self._generate_strategy_pine_script(optimized_strategy)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Statistiken updaten
            self.stats["strategies_generated"] += 1
            self.stats["strategies_tested"] += 1
            self.stats["total_generation_time"] += generation_time
            
            if performance.sharpe_ratio > self.stats["best_sharpe_ratio"]:
                self.stats["best_sharpe_ratio"] = performance.sharpe_ratio
            
            if performance.win_rate > self.stats["best_win_rate"]:
                self.stats["best_win_rate"] = performance.win_rate
            
            # Ergebnisse speichern
            self.generated_strategies.append(optimized_strategy)
            self.performance_history.append(performance)
            
            return {
                "success": True,
                "strategy": optimized_strategy,
                "performance": performance,
                "pine_script": pine_script,
                "generation_time": generation_time,
                "optimization_target": optimization_target,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "market_data_points": len(prepared_data),
                    "entry_conditions_count": len(optimized_strategy.entry_conditions),
                    "exit_conditions_count": len(optimized_strategy.exit_conditions),
                    "confidence_scaling_enabled": optimized_strategy.confidence_scaling
                }
            }
            
        except Exception as e:
            self.logger.error(f"Strategy generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_default_strategy(self, strategy_type: StrategyType) -> StrategyLogic:
        """Erstelle Default-Strategy für unbekannte Typen"""
        
        return StrategyLogic(
            name=f"Default {strategy_type.value}",
            strategy_type=strategy_type,
            entry_conditions=[
                TradingCondition(
                    name="RSI oversold",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator="<",
                    value=30,
                    weight=1.0
                )
            ],
            exit_conditions=[
                TradingCondition(
                    name="RSI overbought",
                    condition_type=ConditionType.TECHNICAL_INDICATOR,
                    indicator="rsi_14",
                    operator=">",
                    value=70,
                    weight=1.0
                )
            ]
        )
    
    def _prepare_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Bereite Market Data mit Enhanced Features vor"""
        
        # Kopie erstellen
        data = market_data.copy()
        
        # Enhanced Features hinzufügen
        try:
            # Für jeden Bar Enhanced Features extrahieren
            enhanced_features = []
            
            for idx, row in data.iterrows():
                bar_dict = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'timestamp': idx
                }
                
                # Enhanced Features extrahieren
                features = self.feature_extractor.extract_features_from_dict(bar_dict)
                enhanced_features.append(features)
            
            # Features zu DataFrame hinzufügen
            feature_names = self.feature_extractor.get_feature_names()
            
            for i, name in enumerate(feature_names):
                if i < len(enhanced_features[0]):
                    data[name] = [features[i] if len(features) > i else 0.0 for features in enhanced_features]
            
            # Zusätzliche technische Indikatoren
            data = self._add_technical_indicators(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {e}")
            return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Füge zusätzliche technische Indikatoren hinzu"""
        
        try:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            data['sma_10'] = data['close'].rolling(10).mean()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['ema_10'] = data['close'].ewm(span=10).mean()
            data['ema_20'] = data['close'].ewm(span=20).mean()
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_sma = data['close'].rolling(bb_period).mean()
            bb_std_dev = data['close'].rolling(bb_period).std()
            data['bb_upper'] = bb_sma + (bb_std_dev * bb_std)
            data['bb_lower'] = bb_sma - (bb_std_dev * bb_std)
            data['bb_middle'] = bb_sma
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # ATR
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data['atr_14'] = true_range.rolling(14).mean()
            
            # Volume Indicators
            data['volume_sma_10'] = data['volume'].rolling(10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma_10']
            
            # Volatility
            data['volatility_5'] = data['close'].pct_change().rolling(5).std()
            data['volatility_10'] = data['close'].pct_change().rolling(10).std()
            
            # Price Action
            data['price_change'] = data['close'].pct_change()
            data['price_range'] = (data['high'] - data['low']) / data['close']
            
            # Mock AI Features (in Produktion würde hier echte AI-Inference stattfinden)
            data['ai_confidence'] = np.random.uniform(0.5, 0.95, len(data))
            data['pattern_direction'] = np.random.choice(['bullish', 'bearish', 'neutral'], len(data))
            data['regime_trending'] = np.random.uniform(0.3, 0.9, len(data))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _optimize_strategy(
        self,
        strategy: StrategyLogic,
        market_data: pd.DataFrame,
        optimization_target: str
    ) -> StrategyLogic:
        """Optimiere Strategy-Parameter"""
        
        try:
            # Kopie der Strategy erstellen
            optimized_strategy = strategy
            
            # Parameter-Optimierung (vereinfacht)
            best_performance = -np.inf
            best_params = {}
            
            # Teste verschiedene Parameter-Kombinationen
            for stop_loss in [0.01, 0.015, 0.02, 0.025]:
                for take_profit in [0.02, 0.03, 0.04, 0.05]:
                    for min_confidence in [0.5, 0.6, 0.7, 0.8]:
                        
                        # Temporäre Strategy mit neuen Parametern
                        temp_strategy = strategy
                        temp_strategy.stop_loss_pct = stop_loss
                        temp_strategy.take_profit_pct = take_profit
                        temp_strategy.min_confidence = min_confidence
                        
                        # Performance testen
                        performance = self._test_strategy_performance(temp_strategy, market_data)
                        
                        # Bewertung basierend auf Optimierungs-Ziel
                        if optimization_target == "sharpe_ratio":
                            score = performance.sharpe_ratio
                        elif optimization_target == "win_rate":
                            score = performance.win_rate
                        elif optimization_target == "profit_factor":
                            score = performance.profit_factor
                        else:
                            score = performance.total_return
                        
                        if score > best_performance:
                            best_performance = score
                            best_params = {
                                "stop_loss_pct": stop_loss,
                                "take_profit_pct": take_profit,
                                "min_confidence": min_confidence
                            }
            
            # Beste Parameter anwenden
            if best_params:
                optimized_strategy.stop_loss_pct = best_params["stop_loss_pct"]
                optimized_strategy.take_profit_pct = best_params["take_profit_pct"]
                optimized_strategy.min_confidence = best_params["min_confidence"]
            
            return optimized_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy optimization error: {e}")
            return strategy
    
    def _test_strategy_performance(
        self,
        strategy: StrategyLogic,
        market_data: pd.DataFrame
    ) -> StrategyPerformance:
        """Teste Strategy-Performance"""
        
        try:
            # Entry/Exit-Signale generieren
            entry_signals = self._evaluate_conditions(strategy.entry_conditions, market_data, strategy.entry_logic_operator)
            exit_signals = self._evaluate_conditions(strategy.exit_conditions, market_data, strategy.exit_logic_operator)
            
            # Backtest durchführen
            backtest_result = self._run_strategy_backtest(
                entry_signals,
                exit_signals,
                market_data,
                strategy
            )
            
            return StrategyPerformance(
                total_return=backtest_result["total_return"],
                sharpe_ratio=backtest_result["sharpe_ratio"],
                max_drawdown=backtest_result["max_drawdown"],
                win_rate=backtest_result["win_rate"],
                profit_factor=backtest_result["profit_factor"],
                total_trades=backtest_result["total_trades"],
                avg_trade_duration=backtest_result["avg_trade_duration"],
                avg_confidence=backtest_result["avg_confidence"],
                confidence_accuracy=backtest_result["confidence_accuracy"],
                high_confidence_win_rate=backtest_result["high_confidence_win_rate"],
                low_confidence_win_rate=backtest_result["low_confidence_win_rate"]
            )
            
        except Exception as e:
            self.logger.error(f"Strategy performance test error: {e}")
            return StrategyPerformance(
                total_return=0.0, sharpe_ratio=0.0, max_drawdown=0.0,
                win_rate=0.0, profit_factor=1.0, total_trades=0,
                avg_trade_duration=0.0, avg_confidence=0.0,
                confidence_accuracy=0.0, high_confidence_win_rate=0.0,
                low_confidence_win_rate=0.0
            )    

    def _evaluate_conditions(
        self,
        conditions: List[TradingCondition],
        market_data: pd.DataFrame,
        logic_operator: LogicOperator
    ) -> pd.Series:
        """Evaluiere Trading-Conditions"""
        
        try:
            condition_results = []
            
            for condition in conditions:
                if not condition.enabled:
                    continue
                
                result = self._evaluate_single_condition(condition, market_data)
                condition_results.append(result * condition.weight)
            
            if not condition_results:
                return pd.Series(False, index=market_data.index)
            
            # Logische Verknüpfung
            if logic_operator == LogicOperator.AND:
                # Alle Conditions müssen erfüllt sein (gewichteter Durchschnitt > Threshold)
                combined = sum(condition_results) / len(condition_results)
                return combined > 0.7  # 70% Threshold
            
            elif logic_operator == LogicOperator.OR:
                # Mindestens eine Condition muss erfüllt sein
                combined = pd.concat(condition_results, axis=1).max(axis=1)
                return combined > 0.5  # 50% Threshold
            
            else:
                # Fallback: AND
                combined = sum(condition_results) / len(condition_results)
                return combined > 0.7
                
        except Exception as e:
            self.logger.error(f"Error evaluating conditions: {e}")
            return pd.Series(False, index=market_data.index)
    
    def _evaluate_single_condition(
        self,
        condition: TradingCondition,
        market_data: pd.DataFrame
    ) -> pd.Series:
        """Evaluiere einzelne Trading-Condition"""
        
        try:
            indicator_data = market_data.get(condition.indicator)
            
            if indicator_data is None:
                self.logger.warning(f"Indicator {condition.indicator} not found in market data")
                return pd.Series(0.0, index=market_data.index)
            
            # Operator-basierte Evaluation
            if condition.operator == ">":
                if isinstance(condition.value, str):
                    # Vergleich mit anderem Indikator
                    other_data = market_data.get(condition.value, 0)
                    result = (indicator_data > other_data).astype(float)
                else:
                    # Vergleich mit Wert
                    result = (indicator_data > condition.value).astype(float)
            
            elif condition.operator == "<":
                if isinstance(condition.value, str):
                    other_data = market_data.get(condition.value, 0)
                    result = (indicator_data < other_data).astype(float)
                else:
                    result = (indicator_data < condition.value).astype(float)
            
            elif condition.operator == "==":
                if isinstance(condition.value, str):
                    # String-Vergleich (z.B. für pattern_direction)
                    if condition.indicator in market_data.columns:
                        result = (market_data[condition.indicator] == condition.value).astype(float)
                    else:
                        result = pd.Series(0.0, index=market_data.index)
                else:
                    result = (indicator_data == condition.value).astype(float)
            
            elif condition.operator == "crossover":
                # Crossover: Indikator kreuzt Wert von unten nach oben
                if isinstance(condition.value, str):
                    other_data = market_data.get(condition.value, 0)
                    result = ((indicator_data > other_data) & (indicator_data.shift(1) <= other_data.shift(1))).astype(float)
                else:
                    result = ((indicator_data > condition.value) & (indicator_data.shift(1) <= condition.value)).astype(float)
            
            elif condition.operator == "crossunder":
                # Crossunder: Indikator kreuzt Wert von oben nach unten
                if isinstance(condition.value, str):
                    other_data = market_data.get(condition.value, 0)
                    result = ((indicator_data < other_data) & (indicator_data.shift(1) >= other_data.shift(1))).astype(float)
                else:
                    result = ((indicator_data < condition.value) & (indicator_data.shift(1) >= condition.value)).astype(float)
            
            else:
                # Fallback: Größer-Vergleich
                result = (indicator_data > condition.value).astype(float)
            
            # Confidence-Threshold anwenden
            if condition.confidence_threshold > 0 and 'ai_confidence' in market_data.columns:
                confidence_mask = market_data['ai_confidence'] >= condition.confidence_threshold
                result = result * confidence_mask.astype(float)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition.name}: {e}")
            return pd.Series(0.0, index=market_data.index)
    
    def _run_strategy_backtest(
        self,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        market_data: pd.DataFrame,
        strategy: StrategyLogic
    ) -> Dict[str, float]:
        """Führe Strategy-Backtest durch"""
        
        try:
            # Portfolio-Tracking
            position = 0  # 0=Neutral, 1=Long, -1=Short
            cash = 10000.0  # Startkapital
            portfolio_value = cash
            trades = []
            portfolio_values = []
            confidences = []
            
            entry_price = 0.0
            entry_time = None
            entry_confidence = 0.0
            
            for i, timestamp in enumerate(market_data.index):
                current_price = market_data.loc[timestamp, 'close']
                current_confidence = market_data.loc[timestamp, 'ai_confidence'] if 'ai_confidence' in market_data.columns else 0.7
                
                # Entry-Signal prüfen
                if entry_signals.loc[timestamp] and position == 0:
                    # Confidence-Check
                    if current_confidence >= strategy.min_confidence:
                        # Position-Size basierend auf Confidence berechnen
                        if strategy.confidence_scaling:
                            position_size = self._calculate_position_size(current_confidence, strategy, cash)
                        else:
                            position_size = strategy.max_position_size
                        
                        # Long-Position eröffnen
                        position = position_size
                        entry_price = current_price
                        entry_time = timestamp
                        entry_confidence = current_confidence
                
                # Exit-Signal prüfen
                elif exit_signals.loc[timestamp] and position != 0:
                    # Position schließen
                    pnl = (current_price - entry_price) * position * (cash / entry_price)
                    cash += pnl
                    
                    # Trade-Dauer berechnen
                    duration = (timestamp - entry_time).total_seconds() / 3600  # Stunden
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_size': position,
                        'pnl': pnl,
                        'duration': duration,
                        'entry_confidence': entry_confidence,
                        'exit_confidence': current_confidence
                    })
                    
                    confidences.append(entry_confidence)
                    position = 0
                
                # Stop Loss / Take Profit prüfen
                elif position != 0:
                    price_change = (current_price - entry_price) / entry_price
                    
                    # Stop Loss
                    if price_change <= -strategy.stop_loss_pct:
                        pnl = -strategy.stop_loss_pct * position * cash
                        cash += pnl
                        
                        duration = (timestamp - entry_time).total_seconds() / 3600
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position,
                            'pnl': pnl,
                            'duration': duration,
                            'entry_confidence': entry_confidence,
                            'exit_confidence': current_confidence,
                            'exit_reason': 'stop_loss'
                        })
                        
                        confidences.append(entry_confidence)
                        position = 0
                    
                    # Take Profit
                    elif price_change >= strategy.take_profit_pct:
                        pnl = strategy.take_profit_pct * position * cash
                        cash += pnl
                        
                        duration = (timestamp - entry_time).total_seconds() / 3600
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position,
                            'pnl': pnl,
                            'duration': duration,
                            'entry_confidence': entry_confidence,
                            'exit_confidence': current_confidence,
                            'exit_reason': 'take_profit'
                        })
                        
                        confidences.append(entry_confidence)
                        position = 0
                
                # Portfolio-Wert berechnen
                if position != 0:
                    unrealized_pnl = (current_price - entry_price) * position * (cash / entry_price)
                    portfolio_value = cash + unrealized_pnl
                else:
                    portfolio_value = cash
                
                portfolio_values.append(portfolio_value)
            
            # Performance-Metriken berechnen
            if not trades:
                return self._get_zero_strategy_result()
            
            # Basis-Metriken
            total_return = (cash - 10000.0) / 10000.0
            
            # Trade-Statistiken
            trade_pnls = [trade['pnl'] for trade in trades]
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
            
            # Sharpe Ratio
            portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
            sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0
            
            # Max Drawdown
            portfolio_series = pd.Series(portfolio_values)
            rolling_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Confidence-spezifische Metriken
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Confidence-Accuracy: Korrelation zwischen Confidence und Trade-Performance
            if len(trades) > 5:
                trade_confidences = [trade['entry_confidence'] for trade in trades]
                trade_returns = [trade['pnl'] / abs(trade['pnl']) if trade['pnl'] != 0 else 0 for trade in trades]
                confidence_accuracy = np.corrcoef(trade_confidences, trade_returns)[0, 1] if len(set(trade_confidences)) > 1 else 0
            else:
                confidence_accuracy = 0
            
            # High/Low Confidence Win Rates
            high_conf_trades = [trade for trade in trades if trade['entry_confidence'] > 0.8]
            low_conf_trades = [trade for trade in trades if trade['entry_confidence'] <= 0.6]
            
            high_confidence_win_rate = len([t for t in high_conf_trades if t['pnl'] > 0]) / len(high_conf_trades) if high_conf_trades else 0
            low_confidence_win_rate = len([t for t in low_conf_trades if t['pnl'] > 0]) / len(low_conf_trades) if low_conf_trades else 0
            
            # Durchschnittliche Trade-Dauer
            avg_trade_duration = np.mean([trade['duration'] for trade in trades]) if trades else 0
            
            return {
                "total_return": total_return,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": len(trades),
                "avg_trade_duration": avg_trade_duration,
                "avg_confidence": avg_confidence,
                "confidence_accuracy": confidence_accuracy,
                "high_confidence_win_rate": high_confidence_win_rate,
                "low_confidence_win_rate": low_confidence_win_rate
            }
            
        except Exception as e:
            self.logger.error(f"Strategy backtest error: {e}")
            return self._get_zero_strategy_result()
    
    def _calculate_position_size(
        self,
        confidence: float,
        strategy: StrategyLogic,
        available_cash: float
    ) -> float:
        """Berechne Position-Size basierend auf Confidence"""
        
        try:
            # Confidence-basierte Skalierung
            confidence_normalized = (confidence - strategy.min_confidence) / (strategy.max_confidence - strategy.min_confidence)
            confidence_normalized = max(0, min(1, confidence_normalized))
            
            # Position-Size zwischen 25% und 100% der max_position_size
            min_size = strategy.max_position_size * 0.25
            max_size = strategy.max_position_size
            
            position_size = min_size + (max_size - min_size) * confidence_normalized
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return strategy.max_position_size * 0.5  # Fallback
    
    def _get_zero_strategy_result(self) -> Dict[str, float]:
        """Erhalte Null-Strategy-Ergebnis für Fehlerbehandlung"""
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 1.0,
            "total_trades": 0,
            "avg_trade_duration": 0.0,
            "avg_confidence": 0.0,
            "confidence_accuracy": 0.0,
            "high_confidence_win_rate": 0.0,
            "low_confidence_win_rate": 0.0
        }
    
    def _generate_strategy_pine_script(self, strategy: StrategyLogic) -> Dict[str, Any]:
        """Generiere Pine Script für Strategy"""
        
        try:
            # Pine Script Config erstellen
            config = PineScriptConfig(
                script_name=f"AI {strategy.name}",
                indicator_type=IndicatorType.AI_PATTERN,
                ai_features=True,
                confidence_threshold=strategy.min_confidence,
                parameters={
                    "stop_loss_pct": strategy.stop_loss_pct,
                    "take_profit_pct": strategy.take_profit_pct,
                    "max_position_size": strategy.max_position_size,
                    "confidence_scaling": strategy.confidence_scaling
                }
            )
            
            # Pine Script generieren
            result = self.pine_generator.generate_script(config)
            
            if result["success"]:
                # Strategy-spezifische Anpassungen
                pine_script = result["script"]
                
                # Entry/Exit-Conditions hinzufügen
                conditions_code = self._generate_conditions_code(strategy)
                
                # Pine Script erweitern
                enhanced_script = pine_script.replace(
                    "// === AI Enhancement ===",
                    f"// === AI Enhancement ===\n{conditions_code}"
                )
                
                result["script"] = enhanced_script
                result["strategy_logic"] = strategy
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating strategy Pine script: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_conditions_code(self, strategy: StrategyLogic) -> str:
        """Generiere Pine Script Code für Conditions"""
        
        try:
            code_lines = []
            
            # Entry Conditions
            code_lines.append("// Entry Conditions")
            entry_conditions = []
            
            for i, condition in enumerate(strategy.entry_conditions):
                if not condition.enabled:
                    continue
                
                condition_var = f"entry_condition_{i}"
                
                if condition.operator == ">":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = {condition.indicator} > {condition.value}")
                    else:
                        code_lines.append(f"{condition_var} = {condition.indicator} > {condition.value}")
                elif condition.operator == "<":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = {condition.indicator} < {condition.value}")
                    else:
                        code_lines.append(f"{condition_var} = {condition.indicator} < {condition.value}")
                elif condition.operator == "crossover":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = ta.crossover({condition.indicator}, {condition.value})")
                    else:
                        code_lines.append(f"{condition_var} = ta.crossover({condition.indicator}, {condition.value})")
                elif condition.operator == "crossunder":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = ta.crossunder({condition.indicator}, {condition.value})")
                    else:
                        code_lines.append(f"{condition_var} = ta.crossunder({condition.indicator}, {condition.value})")
                
                entry_conditions.append(condition_var)
            
            # Entry Logic
            if strategy.entry_logic_operator == LogicOperator.AND:
                entry_logic = " and ".join(entry_conditions)
            else:
                entry_logic = " or ".join(entry_conditions)
            
            code_lines.append(f"entry_signal = {entry_logic}")
            
            # Exit Conditions
            code_lines.append("\n// Exit Conditions")
            exit_conditions = []
            
            for i, condition in enumerate(strategy.exit_conditions):
                if not condition.enabled:
                    continue
                
                condition_var = f"exit_condition_{i}"
                
                if condition.operator == ">":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = {condition.indicator} > {condition.value}")
                    else:
                        code_lines.append(f"{condition_var} = {condition.indicator} > {condition.value}")
                elif condition.operator == "<":
                    if isinstance(condition.value, str):
                        code_lines.append(f"{condition_var} = {condition.indicator} < {condition.value}")
                    else:
                        code_lines.append(f"{condition_var} = {condition.indicator} < {condition.value}")
                
                exit_conditions.append(condition_var)
            
            # Exit Logic
            if strategy.exit_logic_operator == LogicOperator.AND:
                exit_logic = " and ".join(exit_conditions)
            else:
                exit_logic = " or ".join(exit_conditions)
            
            code_lines.append(f"exit_signal = {exit_logic}")
            
            # Risk Management
            code_lines.append(f"\n// Risk Management")
            code_lines.append(f"stop_loss_pct = {strategy.stop_loss_pct}")
            code_lines.append(f"take_profit_pct = {strategy.take_profit_pct}")
            
            # Confidence-basierte Anpassungen
            if strategy.confidence_scaling:
                code_lines.append(f"\n// Confidence Scaling")
                code_lines.append(f"min_confidence = {strategy.min_confidence}")
                code_lines.append(f"confidence_adjusted_entry = entry_signal and ai_confidence > min_confidence")
                code_lines.append(f"position_size = (ai_confidence - min_confidence) / ({strategy.max_confidence} - min_confidence) * {strategy.max_position_size}")
            
            return "\n".join(code_lines)
            
        except Exception as e:
            self.logger.error(f"Error generating conditions code: {e}")
            return "// Error generating conditions code"
    
    def get_strategy_templates(self) -> List[Dict[str, Any]]:
        """Erhalte verfügbare Strategy-Templates"""
        
        templates = []
        
        for strategy_type, strategy in self.strategy_templates.items():
            templates.append({
                "type": strategy_type.value,
                "name": strategy.name,
                "entry_conditions_count": len(strategy.entry_conditions),
                "exit_conditions_count": len(strategy.exit_conditions),
                "confidence_scaling": strategy.confidence_scaling,
                "stop_loss_pct": strategy.stop_loss_pct,
                "take_profit_pct": strategy.take_profit_pct
            })
        
        return templates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Generator-Statistiken"""
        
        return {
            **self.stats,
            "generated_strategies_count": len(self.generated_strategies),
            "performance_history_count": len(self.performance_history),
            "available_templates": len(self.strategy_templates)
        }


# Factory Function
def create_strategy_logic_generator(config: Optional[Dict] = None) -> StrategyLogicGenerator:
    """
    Factory Function für Strategy Logic Generator
    
    Args:
        config: Generator-Konfiguration
        
    Returns:
        StrategyLogicGenerator Instance
    """
    return StrategyLogicGenerator(config=config)


# Demo/Test Function
def demo_strategy_logic_generator():
    """Demo für Strategy Logic Generator"""
    
    print("🧪 Testing Strategy Logic Generator...")
    
    # Generator erstellen
    generator = create_strategy_logic_generator()
    
    # Verfügbare Templates anzeigen
    templates = generator.get_strategy_templates()
    print(f"\n📋 Available Strategy Templates ({len(templates)}):")
    for template in templates:
        print(f"   - {template['name']}: {template['entry_conditions_count']} entry, {template['exit_conditions_count']} exit conditions")
    
    # Mock Market Data erstellen
    dates = pd.date_range(start='2023-01-01', end='2023-06-30', freq='1H')
    np.random.seed(42)
    
    # Simuliere realistische Preis-Bewegungen
    price_changes = np.random.normal(0, 0.001, len(dates))
    prices = 1.1000 + np.cumsum(price_changes)
    
    market_data = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.0001, len(dates)),
        'high': prices + np.abs(np.random.normal(0, 0.0005, len(dates))),
        'low': prices - np.abs(np.random.normal(0, 0.0005, len(dates))),
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    }, index=dates)
    
    # Sicherstelle dass High >= Low
    market_data['high'] = np.maximum(market_data['high'], market_data[['open', 'close']].max(axis=1))
    market_data['low'] = np.minimum(market_data['low'], market_data[['open', 'close']].min(axis=1))
    
    print(f"📊 Created market data: {len(market_data)} bars from {market_data.index[0]} to {market_data.index[-1]}")
    
    # Test verschiedene Strategy-Typen
    test_strategies = [
        StrategyType.TREND_FOLLOWING,
        StrategyType.MEAN_REVERSION,
        StrategyType.AI_PATTERN
    ]
    
    for strategy_type in test_strategies:
        print(f"\n--- Testing {strategy_type.value} Strategy ---")
        
        # Strategy generieren
        result = generator.generate_strategy(
            strategy_type=strategy_type,
            market_data=market_data,
            optimization_target="sharpe_ratio"
        )
        
        if result["success"]:
            performance = result["performance"]
            
            print(f"✅ Strategy generated successfully:")
            print(f"   Total Return: {performance.total_return:.3f}")
            print(f"   Sharpe Ratio: {performance.sharpe_ratio:.3f}")
            print(f"   Win Rate: {performance.win_rate:.3f}")
            print(f"   Profit Factor: {performance.profit_factor:.3f}")
            print(f"   Max Drawdown: {performance.max_drawdown:.3f}")
            print(f"   Total Trades: {performance.total_trades}")
            print(f"   Avg Confidence: {performance.avg_confidence:.3f}")
            print(f"   Confidence Accuracy: {performance.confidence_accuracy:.3f}")
            print(f"   Generation Time: {result['generation_time']:.2f}s")
            
            # Pine Script Info
            if result['pine_script']['success']:
                print(f"   Pine Script: {result['pine_script']['metadata']['script_length']} chars")
            
        else:
            print(f"❌ Strategy generation failed: {result['error']}")
    
    # Statistiken
    stats = generator.get_statistics()
    print(f"\n📈 Generator Statistics:")
    print(f"   Strategies Generated: {stats['strategies_generated']}")
    print(f"   Strategies Tested: {stats['strategies_tested']}")
    print(f"   Best Sharpe Ratio: {stats['best_sharpe_ratio']:.3f}")
    print(f"   Best Win Rate: {stats['best_win_rate']:.3f}")
    print(f"   Total Generation Time: {stats['total_generation_time']:.2f}s")


if __name__ == "__main__":
    demo_strategy_logic_generator()