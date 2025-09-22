"""
Multimodal Strategy Generator für kombinierte Vision+Text-Analyse.
Nutzt MiniCPM-4.1-8B für die Generierung von Trading-Strategien basierend auf 
visuellen Patterns und numerischen Indikatoren.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

from ..data.models import OHLCVData, IndicatorData
from .multimodal_ai import MultimodalAI
from .visual_pattern_analyzer import VisualPatternAnalyzer, VisualPattern, PatternAnalysisResult
from .numerical_indicator_optimizer import NumericalIndicatorOptimizer, OptimizationResult, IndicatorType

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Verschiedene Trading-Strategie-Typen"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"
    HYBRID = "hybrid"

class SignalStrength(Enum):
    """Signal-Stärke-Level"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TradingSignal:
    """Trading-Signal mit multimodalen Informationen"""
    direction: str  # "buy", "sell", "hold"
    strength: SignalStrength
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: Optional[str] = None
    reasoning: Optional[str] = None
    supporting_patterns: List[VisualPattern] = field(default_factory=list)
    supporting_indicators: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingStrategy:
    """Vollständige Trading-Strategie"""
    strategy_id: str
    strategy_type: StrategyType
    name: str
    description: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    indicators_used: List[IndicatorType]
    visual_patterns_used: List[str]
    performance_metrics: Dict[str, float]
    confidence_score: float
    multimodal_reasoning: str
    pine_script_code: Optional[str] = None

@dataclass
class MultimodalAnalysisInput:
    """Input für multimodale Analyse"""
    chart_image: Image.Image
    ohlcv_data: OHLCVData
    indicator_data: Optional[IndicatorData] = None
    timeframe: str = "1h"
    market_context: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[IndicatorType, OptimizationResult]] = None

@dataclass
class StrategyGenerationResult:
    """Ergebnis der Strategie-Generierung"""
    primary_strategy: TradingStrategy
    alternative_strategies: List[TradingStrategy]
    current_signal: TradingSignal
    market_analysis: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    generation_metadata: Dict[str, Any]

class MultimodalStrategyGenerator:
    """
    Generiert Trading-Strategien durch Kombination von visueller Pattern-Analyse
    und numerischer Indikator-Optimierung mit MiniCPM-4.1-8B.
    """
    
    def __init__(self, 
                 multimodal_ai: MultimodalAI,
                 visual_analyzer: VisualPatternAnalyzer,
                 indicator_optimizer: NumericalIndicatorOptimizer):
        self.multimodal_ai = multimodal_ai
        self.visual_analyzer = visual_analyzer
        self.indicator_optimizer = indicator_optimizer
        
        # Strategy Templates für verschiedene Typen
        self.strategy_templates = self._initialize_strategy_templates()
        
        # Gewichtungen für multimodale Fusion
        self.fusion_weights = {
            "visual_patterns": 0.4,
            "numerical_indicators": 0.35,
            "market_context": 0.15,
            "ai_reasoning": 0.1
        }
        
        logger.info("MultimodalStrategyGenerator initialisiert")
    
    def generate_strategy(self, analysis_input: MultimodalAnalysisInput) -> StrategyGenerationResult:
        """
        Generiert Trading-Strategien basierend auf multimodaler Analyse.
        
        Args:
            analysis_input: Multimodale Input-Daten
            
        Returns:
            StrategyGenerationResult mit generierten Strategien
        """
        try:
            logger.info("Starte multimodale Strategie-Generierung")
            
            # 1. Visuelle Pattern-Analyse
            visual_analysis = self._analyze_visual_patterns(analysis_input)
            
            # 2. Numerische Indikator-Optimierung
            indicator_analysis = self._analyze_indicators(analysis_input)
            
            # 3. Multimodale AI-Analyse
            ai_analysis = self._perform_ai_analysis(analysis_input, visual_analysis, indicator_analysis)
            
            # 4. Strategie-Synthese
            strategies = self._synthesize_strategies(visual_analysis, indicator_analysis, ai_analysis, analysis_input)
            
            # 5. Aktuelles Trading-Signal generieren
            current_signal = self._generate_current_signal(strategies[0], visual_analysis, indicator_analysis)
            
            # 6. Markt-Analyse zusammenfassen
            market_analysis = self._compile_market_analysis(visual_analysis, indicator_analysis, ai_analysis)
            
            # 7. Confidence-Breakdown berechnen
            confidence_breakdown = self._calculate_confidence_breakdown(visual_analysis, indicator_analysis, ai_analysis)
            
            result = StrategyGenerationResult(
                primary_strategy=strategies[0],
                alternative_strategies=strategies[1:],
                current_signal=current_signal,
                market_analysis=market_analysis,
                confidence_breakdown=confidence_breakdown,
                generation_metadata={
                    "timestamp": torch.tensor(0).item(),  # Placeholder
                    "model_version": "MiniCPM-4.1-8B",
                    "analysis_duration": 0.0,
                    "patterns_detected": len(visual_analysis.patterns),
                    "indicators_optimized": len(indicator_analysis) if indicator_analysis else 0
                }
            )
            
            logger.info(f"Strategie-Generierung abgeschlossen: {len(strategies)} Strategien erstellt")
            return result
            
        except Exception as e:
            logger.exception(f"Fehler bei Strategie-Generierung: {e}")
            return self._create_fallback_result(analysis_input, e)
    
    def _analyze_visual_patterns(self, analysis_input: MultimodalAnalysisInput) -> PatternAnalysisResult:
        """Führt visuelle Pattern-Analyse durch"""
        try:
            return self.visual_analyzer.analyze_chart_image(
                analysis_input.chart_image,
                analysis_input.ohlcv_data,
                analysis_input.indicator_data
            )
        except Exception as e:
            logger.exception(f"Visuelle Pattern-Analyse fehlgeschlagen: {e}")
            return PatternAnalysisResult(
                patterns=[],
                overall_sentiment="neutral",
                confidence_score=0.0,
                market_structure={},
                key_levels=[],
                analysis_metadata={"error": str(e)}
            )
    
    def _analyze_indicators(self, analysis_input: MultimodalAnalysisInput) -> Optional[Dict[IndicatorType, OptimizationResult]]:
        """Führt Indikator-Optimierung durch"""
        try:
            if analysis_input.optimization_results:
                return analysis_input.optimization_results
            
            # Standard-Indikatoren optimieren
            indicators_to_optimize = [
                IndicatorType.RSI,
                IndicatorType.MACD,
                IndicatorType.EMA,
                IndicatorType.BOLLINGER_BANDS
            ]
            
            return self.indicator_optimizer.optimize_multiple_indicators(
                indicators_to_optimize,
                analysis_input.ohlcv_data
            )
            
        except Exception as e:
            logger.exception(f"Indikator-Analyse fehlgeschlagen: {e}")
            return None
    
    def _perform_ai_analysis(self, 
                           analysis_input: MultimodalAnalysisInput,
                           visual_analysis: PatternAnalysisResult,
                           indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]]) -> Dict[str, Any]:
        """Führt AI-basierte multimodale Analyse durch"""
        try:
            # Kontext für AI-Analyse erstellen
            context = self._create_ai_context(analysis_input, visual_analysis, indicator_analysis)
            
            # Multimodale AI-Analyse
            ai_result = self.multimodal_ai.analyze_multimodal_strategy(
                analysis_input.chart_image,
                context
            )
            
            return ai_result
            
        except Exception as e:
            logger.exception(f"AI-Analyse fehlgeschlagen: {e}")
            return {
                "strategy_recommendation": "neutral",
                "confidence": 0.0,
                "reasoning": f"AI-Analyse fehlgeschlagen: {e}",
                "risk_assessment": "high"
            }
    
    def _synthesize_strategies(self,
                             visual_analysis: PatternAnalysisResult,
                             indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                             ai_analysis: Dict[str, Any],
                             analysis_input: MultimodalAnalysisInput) -> List[TradingStrategy]:
        """Synthetisiert Trading-Strategien aus allen Analysen"""
        try:
            strategies = []
            
            # 1. Primäre Strategie basierend auf stärksten Signalen
            primary_strategy = self._create_primary_strategy(
                visual_analysis, indicator_analysis, ai_analysis, analysis_input
            )
            strategies.append(primary_strategy)
            
            # 2. Alternative Strategien für verschiedene Szenarien
            alternative_strategies = self._create_alternative_strategies(
                visual_analysis, indicator_analysis, ai_analysis, analysis_input
            )
            strategies.extend(alternative_strategies)
            
            # Strategien nach Confidence sortieren
            strategies.sort(key=lambda s: s.confidence_score, reverse=True)
            
            return strategies[:5]  # Top 5 Strategien
            
        except Exception as e:
            logger.exception(f"Strategie-Synthese fehlgeschlagen: {e}")
            return [self._create_fallback_strategy(analysis_input)]
    
    def _create_primary_strategy(self,
                               visual_analysis: PatternAnalysisResult,
                               indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                               ai_analysis: Dict[str, Any],
                               analysis_input: MultimodalAnalysisInput) -> TradingStrategy:
        """Erstellt die primäre Trading-Strategie"""
        try:
            # Strategie-Typ basierend auf Analysen bestimmen
            strategy_type = self._determine_strategy_type(visual_analysis, indicator_analysis, ai_analysis)
            
            # Entry/Exit-Bedingungen erstellen
            entry_conditions = self._create_entry_conditions(visual_analysis, indicator_analysis, ai_analysis)
            exit_conditions = self._create_exit_conditions(visual_analysis, indicator_analysis, ai_analysis)
            
            # Risk Management definieren
            risk_management = self._create_risk_management(visual_analysis, indicator_analysis, ai_analysis)
            
            # Performance-Metriken schätzen
            performance_metrics = self._estimate_performance_metrics(indicator_analysis)
            
            # Confidence Score berechnen
            confidence_score = self._calculate_strategy_confidence(visual_analysis, indicator_analysis, ai_analysis)
            
            # Multimodale Begründung erstellen
            multimodal_reasoning = self._create_multimodal_reasoning(visual_analysis, indicator_analysis, ai_analysis)
            
            return TradingStrategy(
                strategy_id=f"multimodal_{strategy_type.value}_{torch.randint(1000, 9999, (1,)).item()}",
                strategy_type=strategy_type,
                name=f"Multimodal {strategy_type.value.replace('_', ' ').title()} Strategy",
                description=f"AI-generated {strategy_type.value} strategy combining visual patterns and optimized indicators",
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                indicators_used=list(indicator_analysis.keys()) if indicator_analysis else [],
                visual_patterns_used=[p.pattern_type.value for p in visual_analysis.patterns],
                performance_metrics=performance_metrics,
                confidence_score=confidence_score,
                multimodal_reasoning=multimodal_reasoning
            )
            
        except Exception as e:
            logger.exception(f"Primäre Strategie-Erstellung fehlgeschlagen: {e}")
            return self._create_fallback_strategy(analysis_input)
    
    def _create_alternative_strategies(self,
                                     visual_analysis: PatternAnalysisResult,
                                     indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                     ai_analysis: Dict[str, Any],
                                     analysis_input: MultimodalAnalysisInput) -> List[TradingStrategy]:
        """Erstellt alternative Trading-Strategien"""
        alternatives = []
        
        try:
            # Konservative Strategie
            conservative_strategy = self._create_conservative_strategy(visual_analysis, indicator_analysis, ai_analysis)
            alternatives.append(conservative_strategy)
            
            # Aggressive Strategie
            aggressive_strategy = self._create_aggressive_strategy(visual_analysis, indicator_analysis, ai_analysis)
            alternatives.append(aggressive_strategy)
            
            # Contrarian Strategie (falls Hauptstrategie Trend-Following ist)
            if visual_analysis.overall_sentiment != "neutral":
                contrarian_strategy = self._create_contrarian_strategy(visual_analysis, indicator_analysis, ai_analysis)
                alternatives.append(contrarian_strategy)
            
        except Exception as e:
            logger.warning(f"Alternative Strategien-Erstellung teilweise fehlgeschlagen: {e}")
        
        return alternatives
    
    def _generate_current_signal(self,
                               primary_strategy: TradingStrategy,
                               visual_analysis: PatternAnalysisResult,
                               indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]]) -> TradingSignal:
        """Generiert aktuelles Trading-Signal"""
        try:
            # Signal-Richtung basierend auf Analysen
            direction = self._determine_signal_direction(visual_analysis, indicator_analysis)
            
            # Signal-Stärke berechnen
            strength = self._calculate_signal_strength(visual_analysis, indicator_analysis)
            
            # Confidence berechnen
            confidence = primary_strategy.confidence_score
            
            # Entry/Exit-Preise schätzen
            entry_price, stop_loss, take_profit = self._calculate_signal_levels(visual_analysis, indicator_analysis)
            
            # Begründung erstellen
            reasoning = self._create_signal_reasoning(visual_analysis, indicator_analysis, direction, strength)
            
            return TradingSignal(
                direction=direction,
                strength=strength,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe="1h",  # Default
                reasoning=reasoning,
                supporting_patterns=visual_analysis.patterns[:3],  # Top 3 Patterns
                supporting_indicators={
                    indicator.value: result.optimal_parameters 
                    for indicator, result in (indicator_analysis or {}).items()
                }
            )
            
        except Exception as e:
            logger.exception(f"Signal-Generierung fehlgeschlagen: {e}")
            return TradingSignal(
                direction="hold",
                strength=SignalStrength.WEAK,
                confidence=0.0,
                reasoning=f"Signal-Generierung fehlgeschlagen: {e}"
            )
    
    def _determine_strategy_type(self,
                               visual_analysis: PatternAnalysisResult,
                               indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                               ai_analysis: Dict[str, Any]) -> StrategyType:
        """Bestimmt den optimalen Strategie-Typ"""
        try:
            # AI-Empfehlung priorisieren
            ai_recommendation = ai_analysis.get("strategy_type", "").lower()
            for strategy_type in StrategyType:
                if strategy_type.value in ai_recommendation:
                    return strategy_type
            
            # Fallback basierend auf visuellen Patterns
            if visual_analysis.overall_sentiment in ["bullish", "bearish"]:
                return StrategyType.TREND_FOLLOWING
            else:
                return StrategyType.MEAN_REVERSION
                
        except Exception as e:
            logger.warning(f"Strategie-Typ-Bestimmung fehlgeschlagen: {e}")
            return StrategyType.HYBRID
    
    def _create_entry_conditions(self,
                               visual_analysis: PatternAnalysisResult,
                               indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                               ai_analysis: Dict[str, Any]) -> List[str]:
        """Erstellt Entry-Bedingungen"""
        conditions = []
        
        try:
            # Pattern-basierte Bedingungen
            for pattern in visual_analysis.patterns[:3]:  # Top 3 Patterns
                if pattern.direction == "bullish":
                    conditions.append(f"{pattern.pattern_type.value} pattern detected with confidence > {pattern.confidence:.2f}")
                elif pattern.direction == "bearish":
                    conditions.append(f"{pattern.pattern_type.value} pattern detected with confidence > {pattern.confidence:.2f}")
            
            # Indikator-basierte Bedingungen
            if indicator_analysis:
                for indicator_type, result in indicator_analysis.items():
                    if result.performance_score > 0.1:  # Nur profitable Indikatoren
                        conditions.append(f"{indicator_type.value} signal with optimized parameters")
            
            # AI-basierte Bedingungen
            ai_conditions = ai_analysis.get("entry_conditions", [])
            if isinstance(ai_conditions, list):
                conditions.extend(ai_conditions[:2])  # Top 2 AI-Bedingungen
            
        except Exception as e:
            logger.warning(f"Entry-Bedingungen-Erstellung fehlgeschlagen: {e}")
        
        return conditions if conditions else ["Market conditions favorable for entry"]
    
    def _create_exit_conditions(self,
                              visual_analysis: PatternAnalysisResult,
                              indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                              ai_analysis: Dict[str, Any]) -> List[str]:
        """Erstellt Exit-Bedingungen"""
        conditions = []
        
        try:
            # Standard Exit-Bedingungen
            conditions.extend([
                "Stop loss triggered",
                "Take profit target reached",
                "Reversal pattern detected"
            ])
            
            # AI-basierte Exit-Bedingungen
            ai_exits = ai_analysis.get("exit_conditions", [])
            if isinstance(ai_exits, list):
                conditions.extend(ai_exits[:2])
            
        except Exception as e:
            logger.warning(f"Exit-Bedingungen-Erstellung fehlgeschlagen: {e}")
        
        return conditions if conditions else ["Standard exit conditions apply"]
    
    def _create_risk_management(self,
                              visual_analysis: PatternAnalysisResult,
                              indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                              ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Risk Management Regeln"""
        try:
            # Basis Risk Management
            risk_management = {
                "max_risk_per_trade": 0.02,  # 2% pro Trade
                "stop_loss_pct": 0.015,      # 1.5% Stop Loss
                "take_profit_pct": 0.03,     # 3% Take Profit (2:1 R/R)
                "max_concurrent_trades": 3,
                "max_daily_loss": 0.05       # 5% maximaler Tagesverlust
            }
            
            # Anpassung basierend auf Volatilität
            volatility = visual_analysis.market_structure.get("volatility", "medium")
            if volatility == "high":
                risk_management["stop_loss_pct"] = 0.025
                risk_management["max_risk_per_trade"] = 0.015
            elif volatility == "low":
                risk_management["stop_loss_pct"] = 0.01
                risk_management["take_profit_pct"] = 0.02
            
            # AI-basierte Anpassungen
            ai_risk = ai_analysis.get("risk_assessment", "medium")
            if ai_risk == "high":
                risk_management["max_risk_per_trade"] *= 0.5
                risk_management["max_concurrent_trades"] = 1
            
            return risk_management
            
        except Exception as e:
            logger.warning(f"Risk Management-Erstellung fehlgeschlagen: {e}")
            return {"max_risk_per_trade": 0.01, "stop_loss_pct": 0.02}
    
    def _estimate_performance_metrics(self, indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]]) -> Dict[str, float]:
        """Schätzt Performance-Metriken der Strategie"""
        try:
            if not indicator_analysis:
                return {"estimated_sharpe": 0.0, "estimated_return": 0.0}
            
            # Durchschnittliche Performance der Indikatoren
            avg_performance = np.mean([result.performance_score for result in indicator_analysis.values()])
            
            return {
                "estimated_sharpe": avg_performance,
                "estimated_return": avg_performance * 0.1,  # Grobe Schätzung
                "estimated_max_drawdown": 0.1,
                "estimated_win_rate": 0.55
            }
            
        except Exception as e:
            logger.warning(f"Performance-Schätzung fehlgeschlagen: {e}")
            return {"estimated_sharpe": 0.0}
    
    def _calculate_strategy_confidence(self,
                                     visual_analysis: PatternAnalysisResult,
                                     indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                     ai_analysis: Dict[str, Any]) -> float:
        """Berechnet Gesamt-Confidence der Strategie"""
        try:
            # Gewichtete Kombination aller Confidence-Scores
            visual_confidence = visual_analysis.confidence_score * self.fusion_weights["visual_patterns"]
            
            indicator_confidence = 0.0
            if indicator_analysis:
                avg_indicator_performance = np.mean([result.performance_score for result in indicator_analysis.values()])
                indicator_confidence = min(avg_indicator_performance, 1.0) * self.fusion_weights["numerical_indicators"]
            
            ai_confidence = ai_analysis.get("confidence", 0.0) * self.fusion_weights["ai_reasoning"]
            
            market_confidence = 0.5 * self.fusion_weights["market_context"]  # Default
            
            total_confidence = visual_confidence + indicator_confidence + ai_confidence + market_confidence
            
            return min(total_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.5
    
    def _create_multimodal_reasoning(self,
                                   visual_analysis: PatternAnalysisResult,
                                   indicator_analysis: Optional[Dict[IndicatorType, OptimizationResult]],
                                   ai_analysis: Dict[str, Any]) -> str:
        """Erstellt multimodale Begründung für die Strategie"""
        try:
            reasoning_parts = []
            
            # Visuelle Analyse
            if visual_analysis.patterns:
                top_pattern = visual_analysis.patterns[0]
                reasoning_parts.append(
                    f"Visual analysis identified {top_pattern.pattern_type.value} pattern "
                    f"with {top_pattern.confidence:.1%} confidence, suggesting {top_pattern.direction} bias."
                )
            
            # Indikator-Analyse
            if indicator_analysis:
                best_indicator = max(indicator_analysis.items(), key=lambda x: x[1].performance_score)
                reasoning_parts.append(
                    f"Optimized {best_indicator[0].value} shows strong performance "
                    f"(score: {best_indicator[1].performance_score:.3f}) with parameters {best_indicator[1].optimal_parameters}."
                )
            
            # AI-Analyse
            ai_reasoning = ai_analysis.get("reasoning", "")
            if ai_reasoning:
                reasoning_parts.append(f"AI analysis suggests: {ai_reasoning}")
            
            # Markt-Kontext
            market_sentiment = visual_analysis.overall_sentiment
            reasoning_parts.append(f"Overall market sentiment appears {market_sentiment}.")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            logger.warning(f"Reasoning-Erstellung fehlgeschlagen: {e}")
            return "Multimodal analysis completed with limited information."
    
    def _create_conservative_strategy(self, visual_analysis, indicator_analysis, ai_analysis) -> TradingStrategy:
        """Erstellt konservative Strategie"""
        # Implementierung ähnlich zu _create_primary_strategy, aber mit konservativeren Parametern
        return TradingStrategy(
            strategy_id="conservative_001",
            strategy_type=StrategyType.MEAN_REVERSION,
            name="Conservative Mean Reversion",
            description="Low-risk mean reversion strategy",
            entry_conditions=["Strong reversal signals only"],
            exit_conditions=["Quick profit taking", "Tight stop losses"],
            risk_management={"max_risk_per_trade": 0.01, "stop_loss_pct": 0.01},
            indicators_used=[],
            visual_patterns_used=[],
            performance_metrics={"estimated_sharpe": 0.8},
            confidence_score=0.6,
            multimodal_reasoning="Conservative approach based on risk management principles."
        )
    
    def _create_aggressive_strategy(self, visual_analysis, indicator_analysis, ai_analysis) -> TradingStrategy:
        """Erstellt aggressive Strategie"""
        return TradingStrategy(
            strategy_id="aggressive_001",
            strategy_type=StrategyType.MOMENTUM,
            name="Aggressive Momentum",
            description="High-risk momentum strategy",
            entry_conditions=["Strong momentum signals"],
            exit_conditions=["Momentum exhaustion"],
            risk_management={"max_risk_per_trade": 0.05, "stop_loss_pct": 0.03},
            indicators_used=[],
            visual_patterns_used=[],
            performance_metrics={"estimated_sharpe": 1.2},
            confidence_score=0.7,
            multimodal_reasoning="Aggressive approach targeting high returns with increased risk."
        )
    
    def _create_contrarian_strategy(self, visual_analysis, indicator_analysis, ai_analysis) -> TradingStrategy:
        """Erstellt Contrarian-Strategie"""
        return TradingStrategy(
            strategy_id="contrarian_001",
            strategy_type=StrategyType.CONTRARIAN,
            name="Contrarian Reversal",
            description="Counter-trend reversal strategy",
            entry_conditions=["Extreme sentiment readings", "Reversal patterns"],
            exit_conditions=["Trend resumption", "Target levels reached"],
            risk_management={"max_risk_per_trade": 0.02, "stop_loss_pct": 0.02},
            indicators_used=[],
            visual_patterns_used=[],
            performance_metrics={"estimated_sharpe": 0.9},
            confidence_score=0.5,
            multimodal_reasoning="Contrarian approach betting against prevailing sentiment."
        )
    
    def _determine_signal_direction(self, visual_analysis, indicator_analysis) -> str:
        """Bestimmt Signal-Richtung"""
        try:
            # Basierend auf Overall Sentiment
            if visual_analysis.overall_sentiment == "bullish":
                return "buy"
            elif visual_analysis.overall_sentiment == "bearish":
                return "sell"
            else:
                return "hold"
        except:
            return "hold"
    
    def _calculate_signal_strength(self, visual_analysis, indicator_analysis) -> SignalStrength:
        """Berechnet Signal-Stärke"""
        try:
            confidence = visual_analysis.confidence_score
            if confidence > 0.8:
                return SignalStrength.VERY_STRONG
            elif confidence > 0.6:
                return SignalStrength.STRONG
            elif confidence > 0.4:
                return SignalStrength.MODERATE
            else:
                return SignalStrength.WEAK
        except:
            return SignalStrength.WEAK
    
    def _calculate_signal_levels(self, visual_analysis, indicator_analysis) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Berechnet Entry/Stop/Target-Levels"""
        try:
            # Vereinfachte Implementierung
            key_levels = visual_analysis.key_levels
            if len(key_levels) >= 2:
                entry_price = key_levels[0]
                stop_loss = key_levels[1] * 0.99  # 1% unter Support
                take_profit = key_levels[0] * 1.02  # 2% über Entry
                return entry_price, stop_loss, take_profit
        except:
            pass
        return None, None, None
    
    def _create_signal_reasoning(self, visual_analysis, indicator_analysis, direction, strength) -> str:
        """Erstellt Signal-Begründung"""
        try:
            reasoning = f"{direction.upper()} signal with {strength.value} strength. "
            if visual_analysis.patterns:
                top_pattern = visual_analysis.patterns[0]
                reasoning += f"Based on {top_pattern.pattern_type.value} pattern. "
            reasoning += f"Market sentiment: {visual_analysis.overall_sentiment}."
            return reasoning
        except:
            return f"{direction.upper()} signal generated."
    
    def _compile_market_analysis(self, visual_analysis, indicator_analysis, ai_analysis) -> Dict[str, Any]:
        """Kompiliert Markt-Analyse"""
        return {
            "overall_sentiment": visual_analysis.overall_sentiment,
            "confidence_score": visual_analysis.confidence_score,
            "key_patterns": [p.pattern_type.value for p in visual_analysis.patterns[:3]],
            "market_structure": visual_analysis.market_structure,
            "ai_assessment": ai_analysis.get("market_assessment", "neutral")
        }
    
    def _calculate_confidence_breakdown(self, visual_analysis, indicator_analysis, ai_analysis) -> Dict[str, float]:
        """Berechnet Confidence-Breakdown"""
        return {
            "visual_patterns": visual_analysis.confidence_score,
            "numerical_indicators": np.mean([r.performance_score for r in (indicator_analysis or {}).values()]) if indicator_analysis else 0.0,
            "ai_reasoning": ai_analysis.get("confidence", 0.0),
            "market_context": 0.5  # Default
        }
    
    def _create_ai_context(self, analysis_input, visual_analysis, indicator_analysis) -> str:
        """Erstellt Kontext für AI-Analyse"""
        context_parts = [
            f"Timeframe: {analysis_input.timeframe}",
            f"Visual sentiment: {visual_analysis.overall_sentiment}",
            f"Patterns detected: {len(visual_analysis.patterns)}",
            f"Key levels: {len(visual_analysis.key_levels)}"
        ]
        
        if indicator_analysis:
            context_parts.append(f"Indicators optimized: {len(indicator_analysis)}")
        
        return ". ".join(context_parts)
    
    def _create_fallback_result(self, analysis_input: MultimodalAnalysisInput, error: Exception) -> StrategyGenerationResult:
        """Erstellt Fallback-Ergebnis bei Fehlern"""
        fallback_strategy = self._create_fallback_strategy(analysis_input)
        
        return StrategyGenerationResult(
            primary_strategy=fallback_strategy,
            alternative_strategies=[],
            current_signal=TradingSignal(
                direction="hold",
                strength=SignalStrength.WEAK,
                confidence=0.0,
                reasoning=f"Analysis failed: {error}"
            ),
            market_analysis={"error": str(error)},
            confidence_breakdown={"error": 1.0},
            generation_metadata={"error": str(error), "fallback": True}
        )
    
    def _create_fallback_strategy(self, analysis_input: MultimodalAnalysisInput) -> TradingStrategy:
        """Erstellt Fallback-Strategie"""
        return TradingStrategy(
            strategy_id="fallback_001",
            strategy_type=StrategyType.HYBRID,
            name="Fallback Strategy",
            description="Default strategy when analysis fails",
            entry_conditions=["Manual analysis required"],
            exit_conditions=["Manual exit required"],
            risk_management={"max_risk_per_trade": 0.01},
            indicators_used=[],
            visual_patterns_used=[],
            performance_metrics={"estimated_sharpe": 0.0},
            confidence_score=0.0,
            multimodal_reasoning="Fallback strategy due to analysis failure."
        )
    
    def _initialize_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Initialisiert Strategie-Templates"""
        return {
            StrategyType.TREND_FOLLOWING: {
                "risk_management": {"max_risk_per_trade": 0.02, "stop_loss_pct": 0.02},
                "typical_indicators": [IndicatorType.EMA, IndicatorType.MACD],
                "entry_style": "breakout",
                "exit_style": "trailing_stop"
            },
            StrategyType.MEAN_REVERSION: {
                "risk_management": {"max_risk_per_trade": 0.015, "stop_loss_pct": 0.015},
                "typical_indicators": [IndicatorType.RSI, IndicatorType.BOLLINGER_BANDS],
                "entry_style": "oversold_overbought",
                "exit_style": "mean_return"
            },
            StrategyType.BREAKOUT: {
                "risk_management": {"max_risk_per_trade": 0.025, "stop_loss_pct": 0.02},
                "typical_indicators": [IndicatorType.ATR, IndicatorType.BOLLINGER_BANDS],
                "entry_style": "volatility_breakout",
                "exit_style": "momentum_exhaustion"
            }
        }