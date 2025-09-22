#!/usr/bin/env python3
"""
🧩 BAUSTEIN B2: Multimodal Analysis Pipeline
End-to-End multimodale Analyse-Pipeline für Trading-Strategien

Features:
- Vollständige multimodale Analyse basierend auf Bausteinen A1-B1
- Strategien-Bewertung mit Vision+Indikatoren-Kombination
- Konfidenz-basierte Trading-Signale
- Integration aller bestehenden Komponenten
- Performance-optimierte Pipeline
"""

import numpy as np
import pandas as pd
import polars as pl
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# Import bestehender Komponenten (Bausteine A1-B1)
from ai_indicator_optimizer.ai.multimodal_fusion_engine import (
    MultimodalFusionEngine, MultimodalFeatures, FusionStrategy
)
from ai_indicator_optimizer.ai.enhanced_feature_extractor import EnhancedFeatureExtractor
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.data.enhanced_chart_processor import EnhancedChartProcessor
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector
from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType


class AnalysisMode(Enum):
    """Modi für multimodale Analyse"""
    COMPREHENSIVE = "comprehensive"
    FAST = "fast"
    DEEP = "deep"
    REAL_TIME = "real_time"


class TradingSignal(Enum):
    """Trading-Signal-Typen"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategyAnalysis:
    """Strategien-Analyse-Ergebnis"""
    # Basis-Informationen
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Multimodale Features
    multimodal_features: MultimodalFeatures
    
    # Strategien-Bewertung
    trading_signal: TradingSignal
    signal_confidence: float
    signal_reasoning: List[str]
    
    # Entry/Exit-Punkte
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_factor: float
    
    # Risk Assessment
    risk_score: float
    opportunity_score: float
    risk_reward_ratio: float
    
    # Performance-Metriken
    processing_time: float
    analysis_mode: AnalysisMode
    
    # Zusätzliche Insights
    key_insights: List[str]
    market_conditions: Dict[str, Any]
    confidence_breakdown: Dict[str, float]


class StrategyAnalyzer:
    """
    Strategien-Analyzer für multimodale Trading-Strategien
    
    Bewertet Trading-Strategien basierend auf:
    - Multimodalen Features (Vision + Technical)
    - Konfidenz-Scores
    - Risk-Reward-Verhältnissen
    - Marktbedingungen
    """
    
    def __init__(self):
        """Initialize Strategy Analyzer"""
        self.logger = logging.getLogger(__name__)
        
        # Trading-Parameter
        self.risk_free_rate = 0.02  # 2% Risk-free Rate
        self.max_position_size = 0.1  # 10% max Position Size
        self.min_confidence_threshold = 0.6  # Minimum Confidence für Trading
        
        # Signal-Schwellenwerte
        self.signal_thresholds = {
            TradingSignal.STRONG_BUY: 0.8,
            TradingSignal.BUY: 0.65,
            TradingSignal.HOLD: 0.4,
            TradingSignal.SELL: 0.35,
            TradingSignal.STRONG_SELL: 0.2
        }
        
        # Risk-Management-Parameter
        self.default_stop_loss_pct = 0.02  # 2% Stop Loss
        self.default_take_profit_pct = 0.04  # 4% Take Profit (2:1 R/R)
        
    def analyze_strategy(
        self,
        multimodal_features: MultimodalFeatures,
        current_price: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> StrategyAnalysis:
        """
        Analysiere Trading-Strategie basierend auf multimodalen Features
        
        Args:
            multimodal_features: Multimodale Features von Fusion Engine
            current_price: Aktueller Marktpreis
            market_data: Zusätzliche Marktdaten
            
        Returns:
            StrategyAnalysis mit Trading-Empfehlung
        """
        start_time = datetime.now()
        
        try:
            # 1. Trading-Signal generieren
            trading_signal, signal_confidence, reasoning = self._generate_trading_signal(
                multimodal_features
            )
            
            # 2. Entry/Exit-Punkte berechnen
            entry_price, stop_loss, take_profit = self._calculate_entry_exit_points(
                trading_signal, current_price, multimodal_features
            )
            
            # 3. Position-Sizing
            position_size_factor = self._calculate_position_size(
                signal_confidence, multimodal_features.fusion_confidence
            )
            
            # 4. Risk Assessment
            risk_score, opportunity_score, risk_reward_ratio = self._assess_risk_reward(
                multimodal_features, trading_signal, signal_confidence
            )
            
            # 5. Market Conditions Analysis
            market_conditions = self._analyze_market_conditions(
                multimodal_features, market_data
            )
            
            # 6. Key Insights generieren
            key_insights = self._generate_key_insights(
                multimodal_features, trading_signal, market_conditions
            )
            
            # 7. Confidence Breakdown
            confidence_breakdown = self._create_confidence_breakdown(
                multimodal_features, signal_confidence
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return StrategyAnalysis(
                symbol=multimodal_features.symbol,
                timeframe=multimodal_features.timeframe,
                timestamp=multimodal_features.timestamp,
                multimodal_features=multimodal_features,
                trading_signal=trading_signal,
                signal_confidence=signal_confidence,
                signal_reasoning=reasoning,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_factor=position_size_factor,
                risk_score=risk_score,
                opportunity_score=opportunity_score,
                risk_reward_ratio=risk_reward_ratio,
                processing_time=processing_time,
                analysis_mode=AnalysisMode.COMPREHENSIVE,
                key_insights=key_insights,
                market_conditions=market_conditions,
                confidence_breakdown=confidence_breakdown
            )
            
        except Exception as e:
            self.logger.error(f"Strategy analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Fallback-Analyse
            return StrategyAnalysis(
                symbol=multimodal_features.symbol,
                timeframe=multimodal_features.timeframe,
                timestamp=multimodal_features.timestamp,
                multimodal_features=multimodal_features,
                trading_signal=TradingSignal.HOLD,
                signal_confidence=0.0,
                signal_reasoning=[f"Analysis failed: {e}"],
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_factor=0.0,
                risk_score=1.0,
                opportunity_score=0.0,
                risk_reward_ratio=0.0,
                processing_time=processing_time,
                analysis_mode=AnalysisMode.COMPREHENSIVE,
                key_insights=[],
                market_conditions={},
                confidence_breakdown={}
            )
    
    def _generate_trading_signal(
        self, 
        multimodal_features: MultimodalFeatures
    ) -> Tuple[TradingSignal, float, List[str]]:
        """Generiere Trading-Signal basierend auf multimodalen Features"""
        
        reasoning = []
        signal_factors = []
        
        try:
            # 1. Multimodale Trend-Analyse
            trend_strength = multimodal_features.fused_features.get("multimodal_trend_strength", 0.0)
            if abs(trend_strength) > 0.7:
                signal_factors.append(trend_strength * 0.3)  # 30% Gewichtung
                reasoning.append(f"Strong trend detected: {trend_strength:.2f}")
            
            # 2. Momentum-Analyse
            momentum = multimodal_features.fused_features.get("multimodal_momentum", 0.5)
            momentum_signal = (momentum - 0.5) * 2  # Normalisiert zu [-1, 1]
            signal_factors.append(momentum_signal * 0.25)  # 25% Gewichtung
            reasoning.append(f"Momentum signal: {momentum_signal:.2f}")
            
            # 3. Pattern-Konfidenz
            pattern_confidence = multimodal_features.fused_features.get("multimodal_pattern_confidence", 0.0)
            if pattern_confidence > 0.6:
                signal_factors.append(pattern_confidence * 0.2)  # 20% Gewichtung
                reasoning.append(f"Strong pattern confidence: {pattern_confidence:.2f}")
            
            # 4. Reversal vs. Continuation
            reversal_prob = multimodal_features.fused_features.get("multimodal_reversal_probability", 0.0)
            breakout_prob = multimodal_features.fused_features.get("multimodal_breakout_probability", 0.0)
            
            if reversal_prob > breakout_prob and reversal_prob > 0.6:
                # Reversal-Signal
                reversal_signal = -trend_strength if trend_strength != 0 else 0.5
                signal_factors.append(reversal_signal * 0.15)  # 15% Gewichtung
                reasoning.append(f"Reversal pattern detected: {reversal_prob:.2f}")
            elif breakout_prob > 0.6:
                # Continuation-Signal
                continuation_signal = trend_strength if trend_strength != 0 else 0.0
                signal_factors.append(continuation_signal * 0.15)  # 15% Gewichtung
                reasoning.append(f"Breakout pattern detected: {breakout_prob:.2f}")
            
            # 5. Vision-Bestätigung
            vision_trend = multimodal_features.vision_features.get("vision_trend_numeric", 0.0)
            vision_confidence = multimodal_features.vision_confidence
            
            if vision_confidence > 0.7:
                signal_factors.append(vision_trend * vision_confidence * 0.1)  # 10% Gewichtung
                reasoning.append(f"Vision analysis confirms: {vision_trend:.2f} (conf: {vision_confidence:.2f})")
            
            # 6. Gesamtsignal berechnen
            total_signal = sum(signal_factors)
            
            # 7. Konfidenz basierend auf Fusion-Konfidenz und Konsistenz
            signal_confidence = multimodal_features.fusion_confidence
            
            # Konsistenz-Bonus
            consistency = multimodal_features.fused_features.get("multimodal_confidence_consistency", 0.5)
            signal_confidence *= (1.0 + consistency * 0.2)  # Bis zu 20% Bonus
            
            # 8. Trading-Signal bestimmen
            if total_signal > 0.8 and signal_confidence > self.signal_thresholds[TradingSignal.STRONG_BUY]:
                trading_signal = TradingSignal.STRONG_BUY
            elif total_signal > 0.3 and signal_confidence > self.signal_thresholds[TradingSignal.BUY]:
                trading_signal = TradingSignal.BUY
            elif total_signal < -0.8 and signal_confidence > self.signal_thresholds[TradingSignal.STRONG_SELL]:
                trading_signal = TradingSignal.STRONG_SELL
            elif total_signal < -0.3 and signal_confidence > self.signal_thresholds[TradingSignal.SELL]:
                trading_signal = TradingSignal.SELL
            else:
                trading_signal = TradingSignal.HOLD
            
            reasoning.append(f"Total signal: {total_signal:.2f}, Confidence: {signal_confidence:.2f}")
            
            return trading_signal, min(signal_confidence, 1.0), reasoning
            
        except Exception as e:
            self.logger.error(f"Trading signal generation failed: {e}")
            return TradingSignal.HOLD, 0.0, [f"Signal generation failed: {e}"]
    
    def _calculate_entry_exit_points(
        self,
        trading_signal: TradingSignal,
        current_price: float,
        multimodal_features: MultimodalFeatures
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Berechne Entry/Exit-Punkte"""
        
        if trading_signal == TradingSignal.HOLD:
            return None, None, None
        
        try:
            # Entry Price (aktueller Preis für Market Orders)
            entry_price = current_price
            
            # Volatility-basierte Stop Loss/Take Profit
            volatility = multimodal_features.fused_features.get("multimodal_volatility", 0.001)
            volatility_factor = max(volatility, 0.001)  # Minimum Volatility
            
            # Support/Resistance-Stärke berücksichtigen
            sr_strength = multimodal_features.fused_features.get("multimodal_support_resistance_strength", 1.0)
            
            # Angepasste Stop Loss/Take Profit basierend auf Volatility und S/R
            stop_loss_pct = self.default_stop_loss_pct * volatility_factor * (2.0 - sr_strength)
            take_profit_pct = self.default_take_profit_pct * volatility_factor * (2.0 - sr_strength)
            
            if trading_signal in [TradingSignal.BUY, TradingSignal.STRONG_BUY]:
                stop_loss = entry_price * (1.0 - stop_loss_pct)
                take_profit = entry_price * (1.0 + take_profit_pct)
            else:  # SELL or STRONG_SELL
                stop_loss = entry_price * (1.0 + stop_loss_pct)
                take_profit = entry_price * (1.0 - take_profit_pct)
            
            return entry_price, stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Entry/Exit calculation failed: {e}")
            return current_price, None, None
    
    def _calculate_position_size(
        self,
        signal_confidence: float,
        fusion_confidence: float
    ) -> float:
        """Berechne Position-Size-Faktor basierend auf Konfidenz"""
        
        try:
            # Basis-Position-Size basierend auf Konfidenz
            avg_confidence = (signal_confidence + fusion_confidence) / 2.0
            
            # Position Size zwischen 0% und max_position_size
            if avg_confidence < self.min_confidence_threshold:
                return 0.0  # Kein Trade bei niedriger Konfidenz
            
            # Lineare Skalierung von min_confidence_threshold bis 1.0
            confidence_range = 1.0 - self.min_confidence_threshold
            normalized_confidence = (avg_confidence - self.min_confidence_threshold) / confidence_range
            
            position_size_factor = normalized_confidence * self.max_position_size
            
            return min(position_size_factor, self.max_position_size)
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _assess_risk_reward(
        self,
        multimodal_features: MultimodalFeatures,
        trading_signal: TradingSignal,
        signal_confidence: float
    ) -> Tuple[float, float, float]:
        """Bewerte Risk-Reward-Verhältnis"""
        
        try:
            # Risk Score aus multimodalen Features
            base_risk = multimodal_features.fused_features.get("multimodal_risk_score", 0.5)
            
            # Volatility erhöht Risiko
            volatility = multimodal_features.fused_features.get("multimodal_volatility", 0.5)
            volatility_risk = min(volatility * 2.0, 1.0)  # Normalisiert
            
            # Konfidenz-Konsistenz reduziert Risiko
            consistency = multimodal_features.fused_features.get("multimodal_confidence_consistency", 0.5)
            consistency_risk_reduction = consistency * 0.3
            
            # Gesamtes Risiko
            risk_score = (base_risk + volatility_risk) / 2.0 - consistency_risk_reduction
            risk_score = np.clip(risk_score, 0.0, 1.0)
            
            # Opportunity Score
            opportunity_score = multimodal_features.fused_features.get("multimodal_opportunity_score", 0.5)
            
            # Signal-Stärke erhöht Opportunity
            signal_strength_bonus = 0.0
            if trading_signal in [TradingSignal.STRONG_BUY, TradingSignal.STRONG_SELL]:
                signal_strength_bonus = 0.2
            elif trading_signal in [TradingSignal.BUY, TradingSignal.SELL]:
                signal_strength_bonus = 0.1
            
            opportunity_score = min(opportunity_score + signal_strength_bonus, 1.0)
            
            # Risk-Reward-Ratio
            if risk_score > 0:
                risk_reward_ratio = opportunity_score / risk_score
            else:
                risk_reward_ratio = opportunity_score * 10  # Sehr niedriges Risiko
            
            return risk_score, opportunity_score, risk_reward_ratio
            
        except Exception as e:
            self.logger.error(f"Risk-reward assessment failed: {e}")
            return 0.5, 0.5, 1.0
    
    def _analyze_market_conditions(
        self,
        multimodal_features: MultimodalFeatures,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analysiere aktuelle Marktbedingungen"""
        
        conditions = {}
        
        try:
            # Trend-Bedingungen
            trend_strength = multimodal_features.fused_features.get("multimodal_trend_strength", 0.0)
            if abs(trend_strength) > 0.7:
                conditions["market_regime"] = "trending"
                conditions["trend_direction"] = "bullish" if trend_strength > 0 else "bearish"
            elif abs(trend_strength) < 0.3:
                conditions["market_regime"] = "ranging"
                conditions["trend_direction"] = "neutral"
            else:
                conditions["market_regime"] = "transitional"
                conditions["trend_direction"] = "mixed"
            
            # Volatility-Bedingungen
            volatility = multimodal_features.fused_features.get("multimodal_volatility", 0.5)
            if volatility > 0.8:
                conditions["volatility_regime"] = "high"
            elif volatility < 0.3:
                conditions["volatility_regime"] = "low"
            else:
                conditions["volatility_regime"] = "normal"
            
            # Pattern-Bedingungen
            pattern_confidence = multimodal_features.fused_features.get("multimodal_pattern_confidence", 0.0)
            reversal_prob = multimodal_features.fused_features.get("multimodal_reversal_probability", 0.0)
            breakout_prob = multimodal_features.fused_features.get("multimodal_breakout_probability", 0.0)
            
            if pattern_confidence > 0.7:
                if reversal_prob > breakout_prob:
                    conditions["pattern_expectation"] = "reversal"
                else:
                    conditions["pattern_expectation"] = "continuation"
            else:
                conditions["pattern_expectation"] = "unclear"
            
            # Multimodale Qualität
            fusion_confidence = multimodal_features.fusion_confidence
            if fusion_confidence > 0.8:
                conditions["analysis_quality"] = "excellent"
            elif fusion_confidence > 0.6:
                conditions["analysis_quality"] = "good"
            else:
                conditions["analysis_quality"] = "poor"
            
            # Zusätzliche Marktdaten einbeziehen
            if market_data:
                conditions.update(market_data)
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Market conditions analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_key_insights(
        self,
        multimodal_features: MultimodalFeatures,
        trading_signal: TradingSignal,
        market_conditions: Dict[str, Any]
    ) -> List[str]:
        """Generiere Key Insights für die Analyse"""
        
        insights = []
        
        try:
            # 1. Signal-Insight
            insights.append(f"Trading Signal: {trading_signal.value.upper()} with {multimodal_features.fusion_confidence:.1%} confidence")
            
            # 2. Multimodale Qualität
            tech_conf = multimodal_features.technical_confidence
            vision_conf = multimodal_features.vision_confidence
            
            if tech_conf > vision_conf:
                insights.append(f"Technical analysis dominates (Tech: {tech_conf:.1%}, Vision: {vision_conf:.1%})")
            elif vision_conf > tech_conf:
                insights.append(f"Vision analysis dominates (Vision: {vision_conf:.1%}, Tech: {tech_conf:.1%})")
            else:
                insights.append(f"Balanced multimodal analysis (Tech: {tech_conf:.1%}, Vision: {vision_conf:.1%})")
            
            # 3. Trend-Insight
            trend_strength = multimodal_features.fused_features.get("multimodal_trend_strength", 0.0)
            if abs(trend_strength) > 0.7:
                direction = "bullish" if trend_strength > 0 else "bearish"
                insights.append(f"Strong {direction} trend detected (strength: {abs(trend_strength):.2f})")
            
            # 4. Pattern-Insight
            reversal_prob = multimodal_features.fused_features.get("multimodal_reversal_probability", 0.0)
            breakout_prob = multimodal_features.fused_features.get("multimodal_breakout_probability", 0.0)
            
            if reversal_prob > 0.7:
                insights.append(f"High reversal probability detected ({reversal_prob:.1%})")
            elif breakout_prob > 0.7:
                insights.append(f"High breakout probability detected ({breakout_prob:.1%})")
            
            # 5. Risk-Insight
            risk_score = multimodal_features.fused_features.get("multimodal_risk_score", 0.5)
            if risk_score > 0.7:
                insights.append(f"High risk environment (risk score: {risk_score:.2f})")
            elif risk_score < 0.3:
                insights.append(f"Low risk environment (risk score: {risk_score:.2f})")
            
            # 6. Market Regime Insight
            regime = market_conditions.get("market_regime", "unknown")
            volatility_regime = market_conditions.get("volatility_regime", "unknown")
            insights.append(f"Market regime: {regime} with {volatility_regime} volatility")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Key insights generation failed: {e}")
            return [f"Insights generation failed: {e}"]
    
    def _create_confidence_breakdown(
        self,
        multimodal_features: MultimodalFeatures,
        signal_confidence: float
    ) -> Dict[str, float]:
        """Erstelle detailliertes Confidence Breakdown"""
        
        try:
            return {
                "technical_confidence": multimodal_features.technical_confidence,
                "vision_confidence": multimodal_features.vision_confidence,
                "fusion_confidence": multimodal_features.fusion_confidence,
                "signal_confidence": signal_confidence,
                "overall_confidence": (
                    multimodal_features.fusion_confidence + signal_confidence
                ) / 2.0,
                "confidence_consistency": multimodal_features.fused_features.get(
                    "multimodal_confidence_consistency", 0.5
                )
            }
            
        except Exception as e:
            self.logger.error(f"Confidence breakdown creation failed: {e}")
            return {}


class MultimodalAnalysisPipeline:
    """
    🧩 BAUSTEIN B2: Multimodal Analysis Pipeline
    
    End-to-End multimodale Analyse-Pipeline die alle Bausteine A1-B1 orchestriert:
    - Datensammlung und -aufbereitung
    - Multimodale Feature-Fusion (B1)
    - Strategien-Analyse und -Bewertung
    - Trading-Signal-Generierung
    - Performance-Tracking
    """
    
    def __init__(
        self,
        fusion_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_BASED,
        analysis_mode: AnalysisMode = AnalysisMode.COMPREHENSIVE,
        output_dir: str = "data/multimodal_analysis"
    ):
        """
        Initialize Multimodal Analysis Pipeline
        
        Args:
            fusion_strategy: Strategie für multimodale Fusion
            analysis_mode: Modus für Analyse-Tiefe
            output_dir: Output-Verzeichnis
        """
        self.fusion_strategy = fusion_strategy
        self.analysis_mode = analysis_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten initialisieren (Bausteine A1-B1)
        self.fusion_engine = MultimodalFusionEngine(
            fusion_strategy=fusion_strategy,
            output_dir=str(self.output_dir / "fusion")
        )
        self.strategy_analyzer = StrategyAnalyzer()
        self.data_connector = DukascopyConnector()
        self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        
        # Performance Tracking
        self.total_analyses = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.total_processing_time = 0.0
        
        # Threading für Performance
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Multimodal Analysis Pipeline initialized with {fusion_strategy.value} fusion and {analysis_mode.value} analysis")
    
    def analyze_multimodal_strategy(
        self,
        symbol: str = "EUR/USD",
        timeframe: str = "1h",
        lookback_periods: int = 100,
        current_price: Optional[float] = None
    ) -> StrategyAnalysis:
        """
        Hauptfunktion: Vollständige multimodale Strategien-Analyse
        
        Args:
            symbol: Trading-Symbol
            timeframe: Zeitrahmen für Analyse
            lookback_periods: Anzahl Perioden für historische Daten
            current_price: Aktueller Preis (optional, wird sonst aus Daten ermittelt)
            
        Returns:
            StrategyAnalysis mit vollständiger multimodaler Bewertung
        """
        start_time = datetime.now()
        processing_start = start_time.timestamp()
        
        try:
            self.logger.info(f"🔄 Starting multimodal strategy analysis for {symbol} {timeframe}")
            
            # 1. Marktdaten sammeln
            market_data = self._collect_market_data(symbol, timeframe, lookback_periods)
            
            if current_price is None:
                current_price = market_data["current_price"]
            
            # 2. Multimodale Feature-Fusion (Baustein B1)
            multimodal_features = self.fusion_engine.fuse_vision_and_indicators(
                ohlcv_data=market_data["ohlcv_data"],
                timeframe=timeframe,
                symbol=symbol,
                chart_analysis_type="comprehensive" if self.analysis_mode == AnalysisMode.COMPREHENSIVE else "fast"
            )
            
            # 3. Strategien-Analyse
            strategy_analysis = self.strategy_analyzer.analyze_strategy(
                multimodal_features=multimodal_features,
                current_price=current_price,
                market_data=market_data.get("additional_data")
            )
            
            # 4. Performance Tracking
            processing_time = datetime.now().timestamp() - processing_start
            self.total_analyses += 1
            self.successful_analyses += 1
            self.total_processing_time += processing_time
            
            # 5. Ergebnisse speichern
            self._save_analysis_results(strategy_analysis)
            
            self.logger.info(f"✅ Multimodal analysis completed in {processing_time:.3f}s - Signal: {strategy_analysis.trading_signal.value}")
            
            return strategy_analysis
            
        except Exception as e:
            processing_time = datetime.now().timestamp() - processing_start
            self.total_analyses += 1
            self.failed_analyses += 1
            self.total_processing_time += processing_time
            
            error_msg = f"Multimodal analysis failed: {e}"
            self.logger.error(error_msg)
            
            # Fallback-Analyse
            return StrategyAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=start_time,
                multimodal_features=MultimodalFeatures(
                    technical_features={},
                    technical_confidence=0.0,
                    vision_features={"error": str(e)},
                    vision_confidence=0.0,
                    fused_features={},
                    fusion_confidence=0.0,
                    fusion_strategy=self.fusion_strategy,
                    timestamp=start_time,
                    symbol=symbol,
                    timeframe=timeframe,
                    processing_time=processing_time
                ),
                trading_signal=TradingSignal.HOLD,
                signal_confidence=0.0,
                signal_reasoning=[error_msg],
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_factor=0.0,
                risk_score=1.0,
                opportunity_score=0.0,
                risk_reward_ratio=0.0,
                processing_time=processing_time,
                analysis_mode=self.analysis_mode,
                key_insights=[],
                market_conditions={},
                confidence_breakdown={}
            )
    
    def _collect_market_data(
        self,
        symbol: str,
        timeframe: str,
        lookback_periods: int
    ) -> Dict[str, Any]:
        """Sammle Marktdaten für Analyse"""
        
        try:
            # Zeitraum berechnen
            end_date = datetime.now()
            
            # Timeframe zu Timedelta mapping
            timeframe_mapping = {
                "1m": timedelta(minutes=1),
                "5m": timedelta(minutes=5),
                "15m": timedelta(minutes=15),
                "1h": timedelta(hours=1),
                "4h": timedelta(hours=4),
                "1d": timedelta(days=1)
            }
            
            period_delta = timeframe_mapping.get(timeframe, timedelta(hours=1))
            start_date = end_date - (period_delta * lookback_periods)
            
            # OHLCV-Daten abrufen
            ohlcv_data = self.data_connector.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if ohlcv_data.empty:
                raise ValueError(f"No OHLCV data available for {symbol} {timeframe}")
            
            # Aktueller Preis
            current_price = float(ohlcv_data.iloc[-1]["close"])
            
            return {
                "ohlcv_data": ohlcv_data,
                "current_price": current_price,
                "data_points": len(ohlcv_data),
                "start_date": start_date,
                "end_date": end_date,
                "additional_data": {
                    "volume_profile": self._calculate_volume_profile(ohlcv_data),
                    "price_levels": self._identify_key_levels(ohlcv_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market data collection failed: {e}")
            raise
    
    def _calculate_volume_profile(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """Berechne Volume Profile"""
        try:
            total_volume = ohlcv_data["volume"].sum()
            avg_volume = ohlcv_data["volume"].mean()
            volume_std = ohlcv_data["volume"].std()
            
            return {
                "total_volume": float(total_volume),
                "average_volume": float(avg_volume),
                "volume_volatility": float(volume_std / avg_volume) if avg_volume > 0 else 0.0,
                "high_volume_threshold": float(avg_volume + volume_std),
                "low_volume_threshold": float(max(0, avg_volume - volume_std))
            }
        except Exception:
            return {}
    
    def _identify_key_levels(self, ohlcv_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identifiziere wichtige Preis-Level"""
        try:
            # Einfache Support/Resistance basierend auf Highs/Lows
            highs = ohlcv_data["high"].nlargest(5).tolist()
            lows = ohlcv_data["low"].nsmallest(5).tolist()
            
            return {
                "resistance_levels": [float(h) for h in highs],
                "support_levels": [float(l) for l in lows],
                "pivot_point": float((ohlcv_data.iloc[-1]["high"] + ohlcv_data.iloc[-1]["low"] + ohlcv_data.iloc[-1]["close"]) / 3)
            }
        except Exception:
            return {"resistance_levels": [], "support_levels": [], "pivot_point": 0.0}
    
    def _save_analysis_results(self, strategy_analysis: StrategyAnalysis):
        """Speichere Analyse-Ergebnisse"""
        try:
            # Strategy Analysis als AI Prediction speichern
            prediction_data = {
                "timestamp": strategy_analysis.timestamp,
                "symbol": strategy_analysis.symbol,
                "timeframe": strategy_analysis.timeframe,
                "model_name": "MultimodalAnalysisPipeline",
                "prediction_class": strategy_analysis.trading_signal.value,
                "confidence_score": strategy_analysis.signal_confidence,
                "processing_time_ms": strategy_analysis.processing_time * 1000,
                "fusion_strategy": strategy_analysis.multimodal_features.fusion_strategy.value,
                "analysis_mode": strategy_analysis.analysis_mode.value,
                "risk_score": strategy_analysis.risk_score,
                "opportunity_score": strategy_analysis.opportunity_score,
                "risk_reward_ratio": strategy_analysis.risk_reward_ratio,
                "position_size_factor": strategy_analysis.position_size_factor
            }
            self.schema_manager.write_to_stream(prediction_data, DataStreamType.AI_PREDICTIONS)
            
            # Performance Metrics speichern
            performance_data = {
                "timestamp": strategy_analysis.timestamp,
                "component": "MultimodalAnalysisPipeline",
                "operation": "multimodal_strategy_analysis",
                "duration_ms": strategy_analysis.processing_time * 1000,
                "success_rate": 1.0,
                "analysis_mode": strategy_analysis.analysis_mode.value,
                "trading_signal": strategy_analysis.trading_signal.value
            }
            self.schema_manager.write_to_stream(performance_data, DataStreamType.PERFORMANCE_METRICS)
            
        except Exception as e:
            self.logger.warning(f"Failed to save analysis results: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        fusion_stats = self.fusion_engine.get_performance_stats()
        
        return {
            "pipeline_stats": {
                "total_analyses": self.total_analyses,
                "successful_analyses": self.successful_analyses,
                "failed_analyses": self.failed_analyses,
                "success_rate": self.successful_analyses / max(1, self.total_analyses),
                "total_processing_time": self.total_processing_time,
                "average_processing_time": self.total_processing_time / max(1, self.total_analyses),
                "analyses_per_minute": (self.total_analyses / self.total_processing_time * 60) if self.total_processing_time > 0 else 0,
                "fusion_strategy": self.fusion_strategy.value,
                "analysis_mode": self.analysis_mode.value
            },
            "fusion_engine_stats": fusion_stats
        }
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


def demo_multimodal_analysis_pipeline():
    """
    🧩 Demo für Multimodal Analysis Pipeline (Baustein B2)
    """
    print("🧩 BAUSTEIN B2: MULTIMODAL ANALYSIS PIPELINE DEMO")
    print("=" * 70)
    
    # Erstelle Analysis Pipeline
    pipeline = MultimodalAnalysisPipeline(
        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
        analysis_mode=AnalysisMode.COMPREHENSIVE
    )
    
    try:
        # Multimodale Strategien-Analyse
        print("🔄 Running multimodal strategy analysis...")
        
        analysis_result = pipeline.analyze_multimodal_strategy(
            symbol="EUR/USD",
            timeframe="1h",
            lookback_periods=50
        )
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"Symbol: {analysis_result.symbol}")
        print(f"Timeframe: {analysis_result.timeframe}")
        print(f"Trading Signal: {analysis_result.trading_signal.value.upper()}")
        print(f"Signal Confidence: {analysis_result.signal_confidence:.1%}")
        print(f"Risk Score: {analysis_result.risk_score:.2f}")
        print(f"Opportunity Score: {analysis_result.opportunity_score:.2f}")
        print(f"Risk/Reward Ratio: {analysis_result.risk_reward_ratio:.2f}")
        print(f"Position Size Factor: {analysis_result.position_size_factor:.1%}")
        print(f"Processing Time: {analysis_result.processing_time:.3f}s")
        
        print(f"\n💡 KEY INSIGHTS:")
        for insight in analysis_result.key_insights:
            print(f"  • {insight}")
        
        print(f"\n📈 CONFIDENCE BREAKDOWN:")
        for key, value in analysis_result.confidence_breakdown.items():
            print(f"  • {key}: {value:.1%}")
        
        print(f"\n🎯 ENTRY/EXIT POINTS:")
        if analysis_result.entry_price:
            print(f"  • Entry Price: {analysis_result.entry_price:.5f}")
            print(f"  • Stop Loss: {analysis_result.stop_loss:.5f}")
            print(f"  • Take Profit: {analysis_result.take_profit:.5f}")
        else:
            print(f"  • No entry points (HOLD signal)")
        
        # Performance Stats
        print(f"\n📊 PERFORMANCE STATS:")
        stats = pipeline.get_performance_stats()
        pipeline_stats = stats["pipeline_stats"]
        print(f"  • Total Analyses: {pipeline_stats['total_analyses']}")
        print(f"  • Success Rate: {pipeline_stats['success_rate']:.1%}")
        print(f"  • Avg Processing Time: {pipeline_stats['average_processing_time']:.3f}s")
        print(f"  • Analyses/min: {pipeline_stats['analyses_per_minute']:.1f}")
        
        print(f"\n✅ BAUSTEIN B2 DEMO COMPLETED SUCCESSFULLY!")
        
        return analysis_result
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run Demo
    demo_multimodal_analysis_pipeline()