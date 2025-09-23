#!/usr/bin/env python3
"""
Dynamic Fusion Agent für adaptive Vision+Text-Prompts
U3 - Unified Multimodal Flow Integration - Day 1

Features:
- Context-aware Prompt Generation
- Adaptive Model Selection  
- Performance-based Optimization
- Cross-Modal Fusion Strategies
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum

import numpy as np
import pandas as pd


class FusionStrategy(Enum):
    """Fusion-Strategien für Vision+Text-Kombination"""
    VISION_DOMINANT = "vision_dominant"      # 70% Vision, 30% Text
    TEXT_DOMINANT = "text_dominant"          # 30% Vision, 70% Text
    BALANCED = "balanced"                    # 50% Vision, 50% Text
    ADAPTIVE = "adaptive"                    # Dynamische Gewichtung
    SEQUENTIAL = "sequential"                # Vision → Text Pipeline
    PARALLEL = "parallel"                    # Parallel Processing + Fusion


class ModelPreference(Enum):
    """Model-Präferenzen basierend auf Kontext"""
    OLLAMA_PREFERRED = "ollama_preferred"
    TORCHSERVE_PREFERRED = "torchserve_preferred"
    LOAD_BALANCED = "load_balanced"
    QUALITY_OPTIMIZED = "quality_optimized"


@dataclass
class MarketContext:
    """Marktkontext für adaptive Prompt-Generierung"""
    volatility: float
    trend_strength: float
    volume_profile: str
    session_time: str
    major_events: List[str]
    timeframe: str


@dataclass
class ChartData:
    """Chart-Daten für Vision-Analyse"""
    image_path: Path
    timeframe: str
    symbol: str
    bars_count: int
    resolution: tuple
    metadata: Dict[str, Any]


@dataclass
class AdaptivePrompt:
    """Adaptive Prompt-Struktur für Multimodal-Analyse"""
    vision_prompt: str
    text_prompt: str
    fusion_strategy: FusionStrategy
    confidence_threshold: float
    model_preference: ModelPreference
    context_score: float
    generation_timestamp: float
    prompt_id: str


@dataclass
class InferenceResult:
    """Ergebnis einer Multimodal-Inference"""
    strategy_signals: Dict[str, Any]
    confidence_score: float
    processing_time: float
    model_used: str
    fusion_quality: float
    error_info: Optional[str] = None


@dataclass
class FusionConfig:
    """Konfiguration für Dynamic Fusion Agent"""
    vision_weight_default: float = 0.6
    text_weight_default: float = 0.4
    adaptation_learning_rate: float = 0.1
    confidence_threshold_min: float = 0.5
    confidence_threshold_max: float = 0.9
    performance_history_size: int = 1000
    template_update_frequency: int = 100


class AdaptationEngine:
    """Engine für Performance-basierte Adaptation"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.strategy_performance = {}
        self.template_performance = {}
        self.context_patterns = {}
    
    def update_strategy(self, prompt: AdaptivePrompt, performance_score: float):
        """Aktualisiert Strategie basierend auf Performance"""
        strategy_key = f"{prompt.fusion_strategy.value}_{prompt.model_preference.value}"
        
        if strategy_key not in self.strategy_performance:
            self.strategy_performance[strategy_key] = []
        
        self.strategy_performance[strategy_key].append(performance_score)
        
        # Keep only recent performance data
        if len(self.strategy_performance[strategy_key]) > 100:
            self.strategy_performance[strategy_key] = self.strategy_performance[strategy_key][-100:]
    
    def get_best_strategy(self, context_score: float) -> tuple[FusionStrategy, ModelPreference]:
        """Ermittelt beste Strategie für gegebenen Kontext"""
        if not self.strategy_performance:
            return FusionStrategy.BALANCED, ModelPreference.LOAD_BALANCED
        
        best_score = 0
        best_strategy = FusionStrategy.BALANCED
        best_preference = ModelPreference.LOAD_BALANCED
        
        for strategy_key, performances in self.strategy_performance.items():
            if len(performances) >= 5:  # Mindestens 5 Samples
                avg_performance = np.mean(performances[-10:])  # Letzte 10 Performances
                
                if avg_performance > best_score:
                    best_score = avg_performance
                    strategy_parts = strategy_key.split('_', 1)
                    if len(strategy_parts) == 2:
                        strategy_name, preference_name = strategy_parts
                        try:
                            best_strategy = FusionStrategy(strategy_name)
                            best_preference = ModelPreference(preference_name)
                        except ValueError:
                            continue
        
        return best_strategy, best_preference


class PromptTemplateManager:
    """Manager für adaptive Prompt-Templates"""
    
    def __init__(self):
        self.vision_templates = self.load_vision_templates()
        self.text_templates = self.load_text_templates()
        self.template_performance = {}
    
    def load_vision_templates(self) -> Dict[str, str]:
        """Lädt Vision-Prompt-Templates"""
        return {
            "trend_analysis": """
            Analyze this trading chart for trend patterns:
            - Identify primary trend direction (bullish/bearish/sideways)
            - Detect support and resistance levels
            - Recognize chart patterns (triangles, flags, channels)
            - Assess momentum indicators visibility
            - Evaluate breakout potential
            
            Focus on actionable trading signals with confidence levels.
            """,
            
            "pattern_recognition": """
            Examine this chart for specific trading patterns:
            - Candlestick patterns (doji, hammer, engulfing, etc.)
            - Classical chart patterns (head & shoulders, double top/bottom)
            - Volume patterns and anomalies
            - Price action signals
            - Entry and exit points
            
            Provide pattern confidence and expected price targets.
            """,
            
            "multi_timeframe": """
            Analyze this chart considering multiple timeframe context:
            - Current timeframe trend and structure
            - Higher timeframe alignment
            - Key levels from different timeframes
            - Confluence areas for high-probability setups
            - Risk/reward assessment
            
            Emphasize multi-timeframe confluence for strategy generation.
            """,
            
            "volatility_analysis": """
            Focus on volatility and market structure analysis:
            - Current volatility regime (high/low/normal)
            - Price expansion and contraction phases
            - Breakout vs. mean reversion conditions
            - Market structure shifts
            - Optimal position sizing considerations
            
            Adapt strategy recommendations to volatility environment.
            """
        }
    
    def load_text_templates(self) -> Dict[str, str]:
        """Lädt Text-Prompt-Templates für numerische Features"""
        return {
            "technical_indicators": """
            Analyze these technical indicators for trading signals:
            
            Price Data: {price_summary}
            RSI: {rsi} (Overbought >70, Oversold <30)
            MACD: {macd_line} vs {macd_signal} (Histogram: {macd_histogram})
            Bollinger Bands: Price vs Upper/Lower bands
            Volume: {volume} vs Average Volume
            
            Generate trading signals based on indicator confluence.
            """,
            
            "market_context": """
            Consider this market context for strategy adaptation:
            
            Market Session: {session_time}
            Volatility Level: {volatility}
            Trend Strength: {trend_strength}
            Volume Profile: {volume_profile}
            Major Events: {major_events}
            
            Adapt strategy parameters to current market conditions.
            """,
            
            "risk_assessment": """
            Evaluate risk factors for position sizing:
            
            Current Drawdown: {drawdown}
            Volatility Percentile: {volatility_percentile}
            Correlation Risk: {correlation}
            Liquidity Conditions: {liquidity}
            News Impact: {news_sentiment}
            
            Recommend position size and risk management rules.
            """,
            
            "performance_context": """
            Consider recent performance for strategy adjustment:
            
            Recent Win Rate: {win_rate}
            Average Return: {avg_return}
            Sharpe Ratio: {sharpe_ratio}
            Maximum Drawdown: {max_drawdown}
            Strategy Consistency: {consistency}
            
            Suggest strategy modifications based on performance metrics.
            """
        }
    
    def select_optimal_template(self, 
                              context_score: float, 
                              market_context: MarketContext,
                              template_type: str = "vision") -> str:
        """Wählt optimales Template basierend auf Kontext"""
        templates = self.vision_templates if template_type == "vision" else self.text_templates
        
        # Context-based template selection
        if market_context.volatility > 0.8:
            return templates.get("volatility_analysis", list(templates.values())[0])
        elif market_context.trend_strength > 0.7:
            return templates.get("trend_analysis", list(templates.values())[0])
        elif context_score > 0.8:
            return templates.get("multi_timeframe", list(templates.values())[0])
        else:
            return templates.get("pattern_recognition", list(templates.values())[0])


class DynamicFusionAgent:
    """
    Hauptklasse für adaptive Vision+Text-Prompt-Generierung
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core Components
        self.template_manager = PromptTemplateManager()
        self.adaptation_engine = AdaptationEngine(self.config.adaptation_learning_rate)
        
        # Performance Tracking
        self.performance_history = []
        self.context_cache = {}
        
        # Statistics
        self.stats = {
            "total_prompts_generated": 0,
            "successful_inferences": 0,
            "adaptation_updates": 0,
            "average_performance": 0.0
        }
        
        self.logger.info("Dynamic Fusion Agent initialized")
    
    def analyze_context_complexity(self, 
                                 chart_data: ChartData, 
                                 numerical_features: Dict) -> float:
        """
        Analysiert Kontext-Komplexität für adaptive Prompt-Generierung
        
        Returns:
            float: Komplexitäts-Score zwischen 0.0 und 1.0
        """
        complexity_factors = []
        
        # Chart Complexity
        if chart_data.bars_count > 1000:
            complexity_factors.append(0.8)
        elif chart_data.bars_count > 500:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.4)
        
        # Timeframe Complexity
        timeframe_complexity = {
            "1m": 0.9, "5m": 0.8, "15m": 0.6,
            "1h": 0.5, "4h": 0.4, "1d": 0.3
        }
        complexity_factors.append(timeframe_complexity.get(chart_data.timeframe, 0.5))
        
        # Feature Complexity
        feature_count = len(numerical_features)
        if feature_count > 20:
            complexity_factors.append(0.8)
        elif feature_count > 10:
            complexity_factors.append(0.6)
        else:
            complexity_factors.append(0.4)
        
        # Volatility Complexity (if available)
        if "volatility" in numerical_features:
            vol = numerical_features["volatility"]
            if vol > 0.8:
                complexity_factors.append(0.9)
            elif vol > 0.5:
                complexity_factors.append(0.7)
            else:
                complexity_factors.append(0.5)
        
        return np.mean(complexity_factors)
    
    def determine_fusion_strategy(self, context_score: float) -> FusionStrategy:
        """Bestimmt optimale Fusion-Strategie basierend auf Kontext"""
        # Get best strategy from adaptation engine
        best_strategy, _ = self.adaptation_engine.get_best_strategy(context_score)
        
        # Fallback logic if no learned strategy available
        if best_strategy == FusionStrategy.BALANCED:
            if context_score > 0.8:
                return FusionStrategy.ADAPTIVE
            elif context_score > 0.6:
                return FusionStrategy.PARALLEL
            elif context_score > 0.4:
                return FusionStrategy.BALANCED
            else:
                return FusionStrategy.VISION_DOMINANT
        
        return best_strategy
    
    def select_model_preference(self, context_score: float) -> ModelPreference:
        """Wählt Model-Präferenz basierend auf Kontext"""
        _, best_preference = self.adaptation_engine.get_best_strategy(context_score)
        
        # Fallback logic
        if best_preference == ModelPreference.LOAD_BALANCED:
            if context_score > 0.7:
                return ModelPreference.QUALITY_OPTIMIZED
            else:
                return ModelPreference.LOAD_BALANCED
        
        return best_preference
    
    def calculate_confidence_threshold(self, context_score: float) -> float:
        """Berechnet adaptive Confidence-Threshold"""
        # Higher complexity requires higher confidence
        base_threshold = self.config.confidence_threshold_min
        max_threshold = self.config.confidence_threshold_max
        
        # Linear interpolation based on context complexity
        threshold = base_threshold + (max_threshold - base_threshold) * context_score
        
        return min(max_threshold, max(base_threshold, threshold))
    
    def generate_vision_prompt(self, chart_data: ChartData, template: str) -> str:
        """Generiert Vision-Prompt für Chart-Analyse"""
        # Format template with chart-specific information
        formatted_prompt = template.format(
            symbol=chart_data.symbol,
            timeframe=chart_data.timeframe,
            bars_count=chart_data.bars_count,
            resolution=f"{chart_data.resolution[0]}x{chart_data.resolution[1]}"
        )
        
        # Add chart-specific context
        context_addition = f"""
        
        Chart Context:
        - Symbol: {chart_data.symbol}
        - Timeframe: {chart_data.timeframe}
        - Data Points: {chart_data.bars_count} bars
        - Resolution: {chart_data.resolution[0]}x{chart_data.resolution[1]}
        
        Provide specific, actionable trading insights.
        """
        
        return formatted_prompt + context_addition
    
    def generate_text_prompt(self, numerical_features: Dict, template: str) -> str:
        """Generiert Text-Prompt für numerische Features"""
        # Extract key features for template formatting
        feature_summary = {
            "price_summary": f"O:{numerical_features.get('open', 'N/A')} H:{numerical_features.get('high', 'N/A')} L:{numerical_features.get('low', 'N/A')} C:{numerical_features.get('close', 'N/A')}",
            "rsi": numerical_features.get("rsi", "N/A"),
            "macd_line": numerical_features.get("macd", "N/A"),
            "macd_signal": numerical_features.get("macd_signal", "N/A"),
            "macd_histogram": numerical_features.get("macd_histogram", "N/A"),
            "volume": numerical_features.get("volume", "N/A"),
            "volatility": numerical_features.get("volatility", "N/A"),
            "trend_strength": numerical_features.get("trend_strength", "N/A"),
            "volume_profile": numerical_features.get("volume_profile", "Normal"),
            "session_time": numerical_features.get("session_time", "Unknown"),
            "major_events": numerical_features.get("major_events", [])
        }
        
        try:
            formatted_prompt = template.format(**feature_summary)
        except KeyError as e:
            # Fallback if template formatting fails
            self.logger.warning(f"Template formatting failed: {e}")
            formatted_prompt = template
        
        return formatted_prompt
    
    async def generate_adaptive_prompt(self, 
                                     chart_data: ChartData, 
                                     numerical_features: Dict,
                                     market_context: MarketContext) -> AdaptivePrompt:
        """
        Hauptmethode: Generiert adaptive Prompts basierend auf Kontext und Performance-Historie
        """
        start_time = time.time()
        
        try:
            # Context Analysis
            context_score = self.analyze_context_complexity(chart_data, numerical_features)
            
            # Template Selection
            vision_template = self.template_manager.select_optimal_template(
                context_score, market_context, "vision"
            )
            text_template = self.template_manager.select_optimal_template(
                context_score, market_context, "text"
            )
            
            # Adaptive Prompt Generation
            vision_prompt = self.generate_vision_prompt(chart_data, vision_template)
            text_prompt = self.generate_text_prompt(numerical_features, text_template)
            
            # Strategy Selection
            fusion_strategy = self.determine_fusion_strategy(context_score)
            model_preference = self.select_model_preference(context_score)
            confidence_threshold = self.calculate_confidence_threshold(context_score)
            
            # Create Adaptive Prompt
            prompt = AdaptivePrompt(
                vision_prompt=vision_prompt,
                text_prompt=text_prompt,
                fusion_strategy=fusion_strategy,
                confidence_threshold=confidence_threshold,
                model_preference=model_preference,
                context_score=context_score,
                generation_timestamp=time.time(),
                prompt_id=f"prompt_{int(time.time() * 1000)}"
            )
            
            # Update Statistics
            self.stats["total_prompts_generated"] += 1
            
            self.logger.info(f"Generated adaptive prompt {prompt.prompt_id} in {time.time() - start_time:.3f}s")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive prompt: {e}")
            raise
    
    def calculate_performance_score(self, result: InferenceResult) -> float:
        """Berechnet Performance-Score für Feedback-Loop"""
        if result.error_info:
            return 0.0
        
        # Combine multiple performance factors
        factors = []
        
        # Confidence Score (0.0 - 1.0)
        factors.append(result.confidence_score)
        
        # Processing Time (inverse relationship, faster = better)
        # Normalize to 0-1 range, assuming 5 seconds is maximum acceptable
        time_score = max(0, 1 - (result.processing_time / 5.0))
        factors.append(time_score)
        
        # Fusion Quality (0.0 - 1.0)
        factors.append(result.fusion_quality)
        
        # Strategy Signal Quality (if available)
        if "signal_strength" in result.strategy_signals:
            factors.append(result.strategy_signals["signal_strength"])
        
        return np.mean(factors)
    
    def adapt_based_on_feedback(self, prompt: AdaptivePrompt, result: InferenceResult):
        """Lernt aus Inference-Ergebnissen für zukünftige Optimierung"""
        try:
            performance_score = self.calculate_performance_score(result)
            
            # Update adaptation engine
            self.adaptation_engine.update_strategy(prompt, performance_score)
            
            # Update performance history
            self.performance_history.append({
                "prompt_id": prompt.prompt_id,
                "context_score": prompt.context_score,
                "fusion_strategy": prompt.fusion_strategy.value,
                "model_preference": prompt.model_preference.value,
                "performance_score": performance_score,
                "timestamp": time.time()
            })
            
            # Keep history size manageable
            if len(self.performance_history) > self.config.performance_history_size:
                self.performance_history = self.performance_history[-self.config.performance_history_size:]
            
            # Update statistics
            self.stats["adaptation_updates"] += 1
            if performance_score > 0.5:
                self.stats["successful_inferences"] += 1
            
            # Calculate running average performance
            recent_scores = [h["performance_score"] for h in self.performance_history[-100:]]
            self.stats["average_performance"] = np.mean(recent_scores) if recent_scores else 0.0
            
            self.logger.info(f"Updated adaptation for prompt {prompt.prompt_id}, performance: {performance_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in adaptation feedback: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Performance-Statistiken zurück"""
        return {
            **self.stats,
            "performance_history_size": len(self.performance_history),
            "adaptation_strategies": len(self.adaptation_engine.strategy_performance),
            "last_update": time.time()
        }
    
    def export_performance_data(self, filepath: Path):
        """Exportiert Performance-Daten für Analyse"""
        export_data = {
            "stats": self.stats,
            "performance_history": self.performance_history,
            "strategy_performance": self.adaptation_engine.strategy_performance,
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Performance data exported to {filepath}")


# Factory Function
def create_dynamic_fusion_agent(config: Optional[FusionConfig] = None) -> DynamicFusionAgent:
    """Factory function für Dynamic Fusion Agent"""
    return DynamicFusionAgent(config)


# Testing Function
async def test_dynamic_fusion_agent():
    """Test function für Dynamic Fusion Agent"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    chart_data = ChartData(
        image_path=Path("test_chart.png"),
        timeframe="1h",
        symbol="EUR/USD",
        bars_count=500,
        resolution=(1200, 800),
        metadata={"source": "test"}
    )
    
    numerical_features = {
        "open": 1.1000, "high": 1.1050, "low": 1.0980, "close": 1.1020,
        "volume": 15000, "rsi": 65.5, "macd": 0.0012, "volatility": 0.6
    }
    
    market_context = MarketContext(
        volatility=0.6,
        trend_strength=0.7,
        volume_profile="High",
        session_time="London",
        major_events=["ECB Meeting"],
        timeframe="1h"
    )
    
    # Test Dynamic Fusion Agent
    agent = create_dynamic_fusion_agent()
    
    prompt = await agent.generate_adaptive_prompt(chart_data, numerical_features, market_context)
    
    print(f"Generated Prompt ID: {prompt.prompt_id}")
    print(f"Fusion Strategy: {prompt.fusion_strategy.value}")
    print(f"Model Preference: {prompt.model_preference.value}")
    print(f"Context Score: {prompt.context_score:.3f}")
    print(f"Confidence Threshold: {prompt.confidence_threshold:.3f}")
    
    # Simulate feedback
    mock_result = InferenceResult(
        strategy_signals={"signal_strength": 0.8, "direction": "bullish"},
        confidence_score=0.85,
        processing_time=1.2,
        model_used="ollama",
        fusion_quality=0.9
    )
    
    agent.adapt_based_on_feedback(prompt, mock_result)
    
    stats = agent.get_performance_stats()
    print(f"Performance Stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_dynamic_fusion_agent())