"""
Multimodal Pattern Recognition Engine - Hauptintegration aller Komponenten.
Orchestriert Visual Pattern Analyzer, Numerical Indicator Optimizer, 
Multimodal Strategy Generator und Confidence Scoring.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time
from pathlib import Path

from ..core.hardware_detector import HardwareDetector
from ..data.models import OHLCVData, IndicatorData
from .multimodal_ai import MultimodalAI
from .visual_pattern_analyzer import VisualPatternAnalyzer, PatternAnalysisResult
from .numerical_indicator_optimizer import NumericalIndicatorOptimizer, OptimizationResult, IndicatorType, OptimizationConfig
from .multimodal_strategy_generator import MultimodalStrategyGenerator, StrategyGenerationResult, MultimodalAnalysisInput
from .confidence_scoring import ConfidenceScoring, ConfidenceMetrics

logger = logging.getLogger(__name__)

@dataclass
class PatternRecognitionConfig:
    """Konfiguration für Pattern Recognition Engine"""
    # Visual Analysis Config
    enable_visual_analysis: bool = True
    visual_confidence_threshold: float = 0.5
    max_patterns_to_analyze: int = 10
    
    # Numerical Optimization Config
    enable_numerical_optimization: bool = True
    optimization_method: str = "bayesian"  # "bayesian", "genetic", "grid", "random"
    max_optimization_iterations: int = 100
    parallel_optimization_jobs: int = 8
    
    # Strategy Generation Config
    enable_strategy_generation: bool = True
    max_strategies_to_generate: int = 5
    strategy_confidence_threshold: float = 0.4
    
    # Confidence Scoring Config
    enable_confidence_scoring: bool = True
    confidence_calibration: bool = True
    uncertainty_analysis: bool = True
    
    # Performance Config
    use_gpu_acceleration: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 1024

@dataclass
class PatternRecognitionResult:
    """Vollständiges Ergebnis der Pattern Recognition Engine"""
    # Analyse-Ergebnisse
    visual_analysis: PatternAnalysisResult
    indicator_optimization: Dict[IndicatorType, OptimizationResult]
    strategy_generation: StrategyGenerationResult
    confidence_metrics: ConfidenceMetrics
    
    # Meta-Informationen
    processing_time: float
    hardware_utilization: Dict[str, float]
    cache_hit_rate: float
    
    # Zusammenfassung
    executive_summary: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    
    # Debugging/Monitoring
    component_performance: Dict[str, Dict[str, Any]]
    error_log: List[str] = field(default_factory=list)

class MultimodalPatternRecognitionEngine:
    """
    Hauptklasse für multimodale Pattern-Erkennung.
    Orchestriert alle Komponenten und bietet einheitliche API.
    """
    
    def __init__(self, 
                 hardware_detector: HardwareDetector,
                 config: PatternRecognitionConfig = None):
        self.hardware_detector = hardware_detector
        self.config = config or PatternRecognitionConfig()
        
        # Komponenten initialisieren
        self._initialize_components()
        
        # Performance-Monitoring
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Cache für Ergebnisse
        self.result_cache = {} if self.config.enable_caching else None
        
        logger.info("Multimodal Pattern Recognition Engine initialisiert")
        logger.info(f"GPU verfügbar: {torch.cuda.is_available()}")
        logger.info(f"CPU Kerne: {self.hardware_detector.cpu_info.cores_logical if self.hardware_detector.cpu_info else 0}")
    
    def analyze_market_data(self,
                          chart_image: Image.Image,
                          ohlcv_data: OHLCVData,
                          indicator_data: Optional[IndicatorData] = None,
                          timeframe: str = "1h",
                          market_context: Optional[Dict[str, Any]] = None) -> PatternRecognitionResult:
        """
        Führt vollständige multimodale Pattern-Erkennung durch.
        
        Args:
            chart_image: Chart-Bild für visuelle Analyse
            ohlcv_data: OHLCV-Daten für numerische Analyse
            indicator_data: Optionale vorberechnete Indikatoren
            timeframe: Zeitrahmen der Analyse
            market_context: Zusätzlicher Marktkontext
            
        Returns:
            PatternRecognitionResult mit vollständigen Analyseergebnissen
        """
        start_time = time.time()
        self.performance_stats["total_analyses"] += 1
        
        try:
            logger.info(f"Starte multimodale Pattern-Erkennung für {timeframe} Timeframe")
            
            # Cache-Check
            cache_key = self._generate_cache_key(chart_image, ohlcv_data, timeframe)
            if self.result_cache and cache_key in self.result_cache:
                logger.info("Cache-Hit: Verwende gecachtes Ergebnis")
                self.performance_stats["cache_hits"] += 1
                cached_result = self.result_cache[cache_key]
                cached_result.processing_time = time.time() - start_time
                return cached_result
            
            self.performance_stats["cache_misses"] += 1
            
            # Hardware-Monitoring starten
            hardware_monitor = self._start_hardware_monitoring()
            
            # Komponenten-Performance-Tracking
            component_performance = {}
            error_log = []
            
            # 1. Visuelle Pattern-Analyse
            visual_analysis = None
            if self.config.enable_visual_analysis:
                try:
                    comp_start = time.time()
                    visual_analysis = self.visual_analyzer.analyze_chart_image(
                        chart_image, ohlcv_data, indicator_data
                    )
                    component_performance["visual_analysis"] = {
                        "duration": time.time() - comp_start,
                        "patterns_found": len(visual_analysis.patterns),
                        "confidence": visual_analysis.confidence_score,
                        "success": True
                    }
                    logger.info(f"Visuelle Analyse abgeschlossen: {len(visual_analysis.patterns)} Patterns")
                except Exception as e:
                    error_log.append(f"Visuelle Analyse fehlgeschlagen: {e}")
                    component_performance["visual_analysis"] = {"success": False, "error": str(e)}
                    visual_analysis = self._create_fallback_visual_analysis()
            
            # 2. Numerische Indikator-Optimierung
            indicator_optimization = {}
            if self.config.enable_numerical_optimization:
                try:
                    comp_start = time.time()
                    
                    # Standard-Indikatoren für Optimierung
                    indicators_to_optimize = [
                        IndicatorType.RSI,
                        IndicatorType.MACD,
                        IndicatorType.EMA,
                        IndicatorType.BOLLINGER_BANDS,
                        IndicatorType.STOCHASTIC
                    ]
                    
                    optimization_config = OptimizationConfig(
                        optimization_method=self.config.optimization_method,
                        max_iterations=self.config.max_optimization_iterations,
                        parallel_jobs=self.config.parallel_optimization_jobs
                    )
                    
                    indicator_optimization = self.indicator_optimizer.optimize_multiple_indicators(
                        indicators_to_optimize, ohlcv_data, optimization_config
                    )
                    
                    component_performance["indicator_optimization"] = {
                        "duration": time.time() - comp_start,
                        "indicators_optimized": len(indicator_optimization),
                        "avg_performance": np.mean([r.performance_score for r in indicator_optimization.values()]),
                        "success": True
                    }
                    logger.info(f"Indikator-Optimierung abgeschlossen: {len(indicator_optimization)} Indikatoren")
                except Exception as e:
                    error_log.append(f"Indikator-Optimierung fehlgeschlagen: {e}")
                    component_performance["indicator_optimization"] = {"success": False, "error": str(e)}
            
            # 3. Multimodale Strategie-Generierung
            strategy_generation = None
            if self.config.enable_strategy_generation and visual_analysis:
                try:
                    comp_start = time.time()
                    
                    analysis_input = MultimodalAnalysisInput(
                        chart_image=chart_image,
                        ohlcv_data=ohlcv_data,
                        indicator_data=indicator_data,
                        timeframe=timeframe,
                        market_context=market_context,
                        optimization_results=indicator_optimization
                    )
                    
                    strategy_generation = self.strategy_generator.generate_strategy(analysis_input)
                    
                    component_performance["strategy_generation"] = {
                        "duration": time.time() - comp_start,
                        "strategies_generated": 1 + len(strategy_generation.alternative_strategies),
                        "primary_confidence": strategy_generation.primary_strategy.confidence_score,
                        "success": True
                    }
                    logger.info(f"Strategie-Generierung abgeschlossen: {1 + len(strategy_generation.alternative_strategies)} Strategien")
                except Exception as e:
                    error_log.append(f"Strategie-Generierung fehlgeschlagen: {e}")
                    component_performance["strategy_generation"] = {"success": False, "error": str(e)}
                    strategy_generation = self._create_fallback_strategy_generation()
            
            # 4. Confidence Scoring
            confidence_metrics = None
            if self.config.enable_confidence_scoring and visual_analysis and strategy_generation:
                try:
                    comp_start = time.time()
                    
                    confidence_metrics = self.confidence_scorer.calculate_multimodal_confidence(
                        visual_analysis, indicator_optimization, strategy_generation, market_context
                    )
                    
                    component_performance["confidence_scoring"] = {
                        "duration": time.time() - comp_start,
                        "overall_confidence": confidence_metrics.overall_confidence,
                        "calibrated_confidence": confidence_metrics.calibrated_confidence,
                        "success": True
                    }
                    logger.info(f"Confidence Scoring abgeschlossen: {confidence_metrics.calibrated_confidence:.3f}")
                except Exception as e:
                    error_log.append(f"Confidence Scoring fehlgeschlagen: {e}")
                    component_performance["confidence_scoring"] = {"success": False, "error": str(e)}
                    confidence_metrics = self._create_fallback_confidence_metrics()
            
            # Hardware-Monitoring beenden
            hardware_utilization = self._stop_hardware_monitoring(hardware_monitor)
            
            # Ergebnis zusammenstellen
            processing_time = time.time() - start_time
            
            result = PatternRecognitionResult(
                visual_analysis=visual_analysis or self._create_fallback_visual_analysis(),
                indicator_optimization=indicator_optimization,
                strategy_generation=strategy_generation or self._create_fallback_strategy_generation(),
                confidence_metrics=confidence_metrics or self._create_fallback_confidence_metrics(),
                processing_time=processing_time,
                hardware_utilization=hardware_utilization,
                cache_hit_rate=self._calculate_cache_hit_rate(),
                executive_summary=self._create_executive_summary(visual_analysis, indicator_optimization, strategy_generation, confidence_metrics),
                recommendations=self._generate_recommendations(visual_analysis, indicator_optimization, strategy_generation, confidence_metrics),
                risk_assessment=self._assess_risks(visual_analysis, indicator_optimization, strategy_generation, confidence_metrics),
                component_performance=component_performance,
                error_log=error_log
            )
            
            # Cache speichern
            if self.result_cache and len(error_log) == 0:  # Nur erfolgreiche Ergebnisse cachen
                self.result_cache[cache_key] = result
                # Cache-Größe begrenzen
                if len(self.result_cache) > 100:  # Max 100 Einträge
                    oldest_key = next(iter(self.result_cache))
                    del self.result_cache[oldest_key]
            
            # Performance-Stats aktualisieren
            self.performance_stats["successful_analyses"] += 1
            self.performance_stats["average_processing_time"] = (
                (self.performance_stats["average_processing_time"] * (self.performance_stats["successful_analyses"] - 1) + processing_time) /
                self.performance_stats["successful_analyses"]
            )
            
            logger.info(f"Pattern-Erkennung abgeschlossen in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.exception(f"Kritischer Fehler bei Pattern-Erkennung: {e}")
            return self._create_fallback_result(e, time.time() - start_time)
    
    def _initialize_components(self):
        """Initialisiert alle Komponenten"""
        try:
            # MultimodalAI initialisieren
            self.multimodal_ai = MultimodalAI(self.hardware_detector)
            
            # Visual Pattern Analyzer
            self.visual_analyzer = VisualPatternAnalyzer(
                self.multimodal_ai, self.hardware_detector
            )
            
            # Numerical Indicator Optimizer
            self.indicator_optimizer = NumericalIndicatorOptimizer(self.hardware_detector)
            
            # Multimodal Strategy Generator
            self.strategy_generator = MultimodalStrategyGenerator(
                self.multimodal_ai, self.visual_analyzer, self.indicator_optimizer
            )
            
            # Confidence Scoring
            self.confidence_scorer = ConfidenceScoring()
            
            logger.info("Alle Komponenten erfolgreich initialisiert")
            
        except Exception as e:
            logger.exception(f"Fehler bei Komponenten-Initialisierung: {e}")
            raise
    
    def _generate_cache_key(self, chart_image: Image.Image, ohlcv_data: OHLCVData, timeframe: str) -> str:
        """Generiert Cache-Key für Ergebnis-Caching"""
        try:
            # Einfacher Hash basierend auf Bildgröße, Datenumfang und Timeframe
            image_hash = hash((chart_image.size, chart_image.mode))
            data_hash = hash((len(ohlcv_data.close), ohlcv_data.close[-1] if ohlcv_data.close else 0))
            return f"{image_hash}_{data_hash}_{timeframe}"
        except Exception:
            return f"fallback_{time.time()}"
    
    def _start_hardware_monitoring(self) -> Dict[str, Any]:
        """Startet Hardware-Monitoring"""
        try:
            return {
                "start_time": time.time(),
                "initial_gpu_memory": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        except Exception:
            return {"start_time": time.time()}
    
    def _stop_hardware_monitoring(self, monitor_data: Dict[str, Any]) -> Dict[str, float]:
        """Beendet Hardware-Monitoring und gibt Statistiken zurück"""
        try:
            duration = time.time() - monitor_data["start_time"]
            
            utilization = {
                "processing_duration": duration,
                "cpu_cores_used": self.hardware_detector.cpu_info.cores_logical if self.hardware_detector.cpu_info else 0,
                "gpu_available": torch.cuda.is_available()
            }
            
            if torch.cuda.is_available():
                utilization["gpu_memory_used"] = torch.cuda.memory_allocated() - monitor_data.get("initial_gpu_memory", 0)
                utilization["gpu_memory_cached"] = torch.cuda.memory_reserved()
            
            return utilization
            
        except Exception as e:
            logger.warning(f"Hardware-Monitoring fehlgeschlagen: {e}")
            return {"processing_duration": time.time() - monitor_data.get("start_time", time.time())}
    
    def _calculate_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate"""
        try:
            total_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
            if total_requests == 0:
                return 0.0
            return self.performance_stats["cache_hits"] / total_requests
        except Exception:
            return 0.0
    
    def _create_executive_summary(self, visual_analysis, indicator_optimization, strategy_generation, confidence_metrics) -> Dict[str, Any]:
        """Erstellt Executive Summary"""
        try:
            summary = {
                "overall_sentiment": "neutral",
                "confidence_level": "moderate",
                "primary_strategy": "hold",
                "key_patterns": [],
                "top_indicators": [],
                "risk_level": "medium"
            }
            
            if visual_analysis:
                summary["overall_sentiment"] = visual_analysis.overall_sentiment
                summary["key_patterns"] = [p.pattern_type.value for p in visual_analysis.patterns[:3]]
            
            if indicator_optimization:
                top_indicators = sorted(indicator_optimization.items(), 
                                      key=lambda x: x[1].performance_score, reverse=True)[:3]
                summary["top_indicators"] = [ind.value for ind, _ in top_indicators]
            
            if strategy_generation:
                summary["primary_strategy"] = strategy_generation.current_signal.direction
            
            if confidence_metrics:
                summary["confidence_level"] = confidence_metrics.confidence_level.value
                
            return summary
            
        except Exception as e:
            logger.warning(f"Executive Summary-Erstellung fehlgeschlagen: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, visual_analysis, indicator_optimization, strategy_generation, confidence_metrics) -> List[str]:
        """Generiert Handlungsempfehlungen"""
        try:
            recommendations = []
            
            if confidence_metrics and confidence_metrics.calibrated_confidence > 0.7:
                if strategy_generation and strategy_generation.current_signal.direction != "hold":
                    recommendations.append(f"Consider {strategy_generation.current_signal.direction} position based on high confidence analysis")
            
            if visual_analysis and len(visual_analysis.patterns) > 0:
                strongest_pattern = max(visual_analysis.patterns, key=lambda p: p.confidence)
                recommendations.append(f"Monitor {strongest_pattern.pattern_type.value} pattern development")
            
            if indicator_optimization:
                best_indicator = max(indicator_optimization.items(), key=lambda x: x[1].performance_score)
                if best_indicator[1].performance_score > 0.3:
                    recommendations.append(f"Focus on {best_indicator[0].value} signals with optimized parameters")
            
            if not recommendations:
                recommendations.append("Continue monitoring market conditions for clearer signals")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Empfehlungs-Generierung fehlgeschlagen: {e}")
            return ["Analysis incomplete - manual review recommended"]
    
    def _assess_risks(self, visual_analysis, indicator_optimization, strategy_generation, confidence_metrics) -> Dict[str, Any]:
        """Bewertet Risiken"""
        try:
            risk_assessment = {
                "overall_risk": "medium",
                "confidence_risk": "medium",
                "pattern_risk": "medium",
                "indicator_risk": "medium",
                "market_risk": "medium"
            }
            
            if confidence_metrics:
                if confidence_metrics.calibrated_confidence < 0.3:
                    risk_assessment["confidence_risk"] = "high"
                elif confidence_metrics.calibrated_confidence > 0.8:
                    risk_assessment["confidence_risk"] = "low"
            
            if visual_analysis:
                volatility = visual_analysis.market_structure.get("volatility", "medium")
                if volatility == "high":
                    risk_assessment["market_risk"] = "high"
                elif volatility == "low":
                    risk_assessment["market_risk"] = "low"
            
            # Gesamt-Risiko basierend auf Komponenten
            risk_scores = {"low": 1, "medium": 2, "high": 3}
            avg_risk_score = np.mean([risk_scores[risk] for risk in risk_assessment.values() if risk in risk_scores])
            
            if avg_risk_score <= 1.5:
                risk_assessment["overall_risk"] = "low"
            elif avg_risk_score >= 2.5:
                risk_assessment["overall_risk"] = "high"
            
            return risk_assessment
            
        except Exception as e:
            logger.warning(f"Risiko-Bewertung fehlgeschlagen: {e}")
            return {"overall_risk": "high", "error": str(e)}
    
    def _create_fallback_visual_analysis(self):
        """Erstellt Fallback Visual Analysis"""
        from .visual_pattern_analyzer import PatternAnalysisResult
        return PatternAnalysisResult(
            patterns=[],
            overall_sentiment="neutral",
            confidence_score=0.0,
            market_structure={"trend": "neutral"},
            key_levels=[],
            analysis_metadata={"fallback": True}
        )
    
    def _create_fallback_strategy_generation(self):
        """Erstellt Fallback Strategy Generation"""
        from .multimodal_strategy_generator import StrategyGenerationResult, TradingStrategy, TradingSignal, StrategyType, SignalStrength
        
        fallback_strategy = TradingStrategy(
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
        
        fallback_signal = TradingSignal(
            direction="hold",
            strength=SignalStrength.WEAK,
            confidence=0.0,
            reasoning="Analysis failed - manual review required"
        )
        
        return StrategyGenerationResult(
            primary_strategy=fallback_strategy,
            alternative_strategies=[],
            current_signal=fallback_signal,
            market_analysis={"fallback": True},
            confidence_breakdown={"error": 1.0},
            generation_metadata={"fallback": True}
        )
    
    def _create_fallback_confidence_metrics(self):
        """Erstellt Fallback Confidence Metrics"""
        from .confidence_scoring import ConfidenceMetrics, ConfidenceLevel, UncertaintySource
        
        return ConfidenceMetrics(
            overall_confidence=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            calibrated_confidence=0.0,
            uncertainty_sources={source: 1.0 for source in UncertaintySource},
            component_confidences={"fallback": 0.0},
            reliability_score=0.0,
            prediction_interval=(0.0, 0.0),
            confidence_metadata={"fallback": True}
        )
    
    def _create_fallback_result(self, error: Exception, processing_time: float) -> PatternRecognitionResult:
        """Erstellt vollständiges Fallback-Ergebnis"""
        return PatternRecognitionResult(
            visual_analysis=self._create_fallback_visual_analysis(),
            indicator_optimization={},
            strategy_generation=self._create_fallback_strategy_generation(),
            confidence_metrics=self._create_fallback_confidence_metrics(),
            processing_time=processing_time,
            hardware_utilization={"error": True},
            cache_hit_rate=0.0,
            executive_summary={"error": str(error)},
            recommendations=["System error - manual analysis required"],
            risk_assessment={"overall_risk": "high", "error": str(error)},
            component_performance={"error": str(error)},
            error_log=[str(error)]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        return self.performance_stats.copy()
    
    def clear_cache(self) -> None:
        """Leert den Ergebnis-Cache"""
        if self.result_cache:
            self.result_cache.clear()
            logger.info("Ergebnis-Cache geleert")
    
    def update_config(self, new_config: PatternRecognitionConfig) -> None:
        """Aktualisiert Konfiguration"""
        self.config = new_config
        logger.info("Konfiguration aktualisiert")