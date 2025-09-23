#!/usr/bin/env python3
"""
Chart-to-Strategy Pipeline für End-to-End Processing
U3 - Unified Multimodal Flow Integration - Day 2

Features:
- Ollama Vision Client Integration
- Multi-Timeframe Chart Processing
- Pattern Recognition Engine
- Strategy Generation Pipeline
- Batch Processing Capabilities
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd
from PIL import Image

# Import existing components
from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI
from .dynamic_fusion_agent import (
    DynamicFusionAgent, ChartData, MarketContext, AdaptivePrompt, InferenceResult
)


class PatternType(Enum):
    """Erkannte Chart-Pattern-Typen"""
    TREND_CONTINUATION = "trend_continuation"
    TREND_REVERSAL = "trend_reversal"
    CONSOLIDATION = "consolidation"
    BREAKOUT = "breakout"
    SUPPORT_RESISTANCE = "support_resistance"
    CANDLESTICK_PATTERN = "candlestick_pattern"
    VOLUME_PATTERN = "volume_pattern"
    UNKNOWN = "unknown"


class StrategyType(Enum):
    """Generierte Strategy-Typen"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"
    HYBRID = "hybrid"


@dataclass
class ChartFeatures:
    """Extrahierte Chart-Features"""
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    volatility_regime: str
    volume_profile: str
    pattern_confidence: float
    key_levels: Dict[str, float]
    timeframe_alignment: bool


@dataclass
class RecognizedPattern:
    """Erkanntes Trading-Pattern"""
    pattern_type: PatternType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    pattern_description: str
    timeframe: str
    risk_reward_ratio: float
    pattern_metadata: Dict[str, Any]


@dataclass
class StrategySignal:
    """Generiertes Trading-Signal"""
    signal_type: str  # "BUY", "SELL", "HOLD"
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    confidence: float
    timeframe: str
    strategy_type: StrategyType
    risk_reward: float
    signal_metadata: Dict[str, Any]


@dataclass
class StrategyResult:
    """Vollständiges Strategy-Ergebnis"""
    strategy_id: str
    symbol: str
    timeframe: str
    signals: List[StrategySignal]
    patterns: List[RecognizedPattern]
    chart_features: ChartFeatures
    confidence: float
    processing_time: float
    vision_analysis: Dict[str, Any]
    fusion_quality: float
    generation_timestamp: float
    metadata: Dict[str, Any]


@dataclass
class PipelineConfig:
    """Konfiguration für Chart-to-Strategy Pipeline"""
    max_concurrent_charts: int = 10
    vision_timeout_seconds: int = 30
    pattern_confidence_threshold: float = 0.6
    strategy_confidence_threshold: float = 0.7
    enable_batch_processing: bool = True
    cache_vision_results: bool = True
    max_cache_size: int = 1000
    supported_timeframes: List[str] = None
    
    def __post_init__(self):
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]


class ChartProcessor:
    """Verarbeitet Chart-Images und extrahiert Features"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def extract_features(self, chart_path: Path) -> ChartFeatures:
        """
        Extrahiert technische Features aus Chart-Image
        
        Args:
            chart_path: Pfad zum Chart-Image
            
        Returns:
            ChartFeatures mit extrahierten Daten
        """
        try:
            # Simulate chart feature extraction
            # In real implementation, this would use computer vision
            # to analyze chart elements
            
            # Mock feature extraction for now
            features = ChartFeatures(
                trend_direction="bullish",  # bullish/bearish/sideways
                trend_strength=0.75,
                support_levels=[1.1000, 1.0980, 1.0950],
                resistance_levels=[1.1050, 1.1080, 1.1100],
                volatility_regime="normal",  # low/normal/high
                volume_profile="increasing",  # increasing/decreasing/stable
                pattern_confidence=0.8,
                key_levels={
                    "pivot": 1.1020,
                    "daily_high": 1.1055,
                    "daily_low": 1.0975,
                    "weekly_high": 1.1100,
                    "weekly_low": 1.0900
                },
                timeframe_alignment=True
            )
            
            self.logger.info(f"Extracted features from {chart_path}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting chart features: {e}")
            raise
    
    def validate_chart_image(self, chart_path: Path) -> bool:
        """Validiert Chart-Image Format und Qualität"""
        try:
            if not chart_path.exists():
                return False
            
            with Image.open(chart_path) as img:
                # Check image dimensions
                if img.size[0] < 800 or img.size[1] < 600:
                    return False
                
                # Check image format
                if img.format not in ['PNG', 'JPEG', 'JPG']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Chart validation failed: {e}")
            return False


class PatternRecognizer:
    """Erkennt Trading-Patterns in Chart-Features und Vision-Analyse"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pattern_templates = self.load_pattern_templates()
    
    def load_pattern_templates(self) -> Dict[str, Dict]:
        """Lädt Pattern-Recognition-Templates"""
        return {
            "bullish_engulfing": {
                "type": PatternType.TREND_REVERSAL,
                "description": "Bullish engulfing candlestick pattern",
                "confidence_base": 0.8,
                "risk_reward": 2.0
            },
            "support_bounce": {
                "type": PatternType.SUPPORT_RESISTANCE,
                "description": "Price bounce from support level",
                "confidence_base": 0.7,
                "risk_reward": 2.5
            },
            "breakout_pattern": {
                "type": PatternType.BREAKOUT,
                "description": "Price breakout from consolidation",
                "confidence_base": 0.75,
                "risk_reward": 3.0
            },
            "trend_continuation": {
                "type": PatternType.TREND_CONTINUATION,
                "description": "Trend continuation pattern",
                "confidence_base": 0.85,
                "risk_reward": 1.5
            }
        }
    
    def identify_patterns(self, 
                         chart_features: ChartFeatures, 
                         vision_analysis: Dict[str, Any]) -> List[RecognizedPattern]:
        """
        Identifiziert Trading-Patterns basierend auf Chart-Features und Vision-Analyse
        
        Args:
            chart_features: Extrahierte Chart-Features
            vision_analysis: Ollama Vision-Analyse-Ergebnis
            
        Returns:
            Liste erkannter Patterns
        """
        patterns = []
        
        try:
            # Extract pattern information from vision analysis
            vision_patterns = vision_analysis.get("patterns", [])
            vision_confidence = vision_analysis.get("confidence", 0.5)
            
            # Combine with chart features for pattern recognition
            if chart_features.trend_direction == "bullish" and chart_features.trend_strength > 0.7:
                # Strong bullish trend - look for continuation patterns
                pattern = self.create_pattern(
                    "trend_continuation",
                    chart_features,
                    vision_confidence * 0.9,
                    "Strong bullish trend continuation"
                )
                patterns.append(pattern)
            
            # Support/Resistance patterns
            if len(chart_features.support_levels) > 0:
                current_price = chart_features.key_levels.get("pivot", 1.0)
                nearest_support = min(chart_features.support_levels, 
                                    key=lambda x: abs(x - current_price))
                
                if abs(current_price - nearest_support) / current_price < 0.005:  # Within 0.5%
                    pattern = self.create_pattern(
                        "support_bounce",
                        chart_features,
                        vision_confidence * 0.8,
                        f"Price near support at {nearest_support}"
                    )
                    patterns.append(pattern)
            
            # Breakout patterns
            if chart_features.volatility_regime == "high" and chart_features.pattern_confidence > 0.8:
                pattern = self.create_pattern(
                    "breakout_pattern",
                    chart_features,
                    vision_confidence * chart_features.pattern_confidence,
                    "High volatility breakout setup"
                )
                patterns.append(pattern)
            
            # Process vision-detected patterns
            for vision_pattern in vision_patterns:
                if isinstance(vision_pattern, dict):
                    pattern_name = vision_pattern.get("name", "unknown")
                    pattern_conf = vision_pattern.get("confidence", 0.5)
                    
                    if pattern_name in self.pattern_templates:
                        pattern = self.create_pattern(
                            pattern_name,
                            chart_features,
                            pattern_conf,
                            vision_pattern.get("description", "Vision-detected pattern")
                        )
                        patterns.append(pattern)
            
            self.logger.info(f"Identified {len(patterns)} patterns")
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern recognition: {e}")
            return []
    
    def create_pattern(self, 
                      pattern_name: str, 
                      chart_features: ChartFeatures,
                      confidence: float,
                      description: str) -> RecognizedPattern:
        """Erstellt RecognizedPattern-Objekt"""
        template = self.pattern_templates.get(pattern_name, {})
        
        # Calculate entry, stop loss, and take profit levels
        current_price = chart_features.key_levels.get("pivot", 1.0)
        
        if chart_features.trend_direction == "bullish":
            entry_price = current_price * 1.001  # Slight premium for bullish entry
            stop_loss = min(chart_features.support_levels) if chart_features.support_levels else current_price * 0.995
            take_profit = [
                current_price * 1.01,   # TP1: 1%
                current_price * 1.02,   # TP2: 2%
                current_price * 1.03    # TP3: 3%
            ]
        else:
            entry_price = current_price * 0.999  # Slight discount for bearish entry
            stop_loss = max(chart_features.resistance_levels) if chart_features.resistance_levels else current_price * 1.005
            take_profit = [
                current_price * 0.99,   # TP1: -1%
                current_price * 0.98,   # TP2: -2%
                current_price * 0.97    # TP3: -3%
            ]
        
        risk_reward = abs(take_profit[0] - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 1.0
        
        return RecognizedPattern(
            pattern_type=template.get("type", PatternType.UNKNOWN),
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pattern_description=description,
            timeframe="1h",  # Default timeframe
            risk_reward_ratio=risk_reward,
            pattern_metadata={
                "template": pattern_name,
                "chart_features": asdict(chart_features),
                "creation_time": time.time()
            }
        )


class StrategyGenerator:
    """Generiert Trading-Strategien basierend auf erkannten Patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategy_templates = self.load_strategy_templates()
    
    def load_strategy_templates(self) -> Dict[str, Dict]:
        """Lädt Strategy-Generation-Templates"""
        return {
            "momentum_strategy": {
                "type": StrategyType.MOMENTUM,
                "description": "Momentum-based trading strategy",
                "position_size_factor": 0.02,
                "max_risk_per_trade": 0.01
            },
            "mean_reversion_strategy": {
                "type": StrategyType.MEAN_REVERSION,
                "description": "Mean reversion trading strategy",
                "position_size_factor": 0.015,
                "max_risk_per_trade": 0.008
            },
            "breakout_strategy": {
                "type": StrategyType.BREAKOUT,
                "description": "Breakout trading strategy",
                "position_size_factor": 0.025,
                "max_risk_per_trade": 0.012
            }
        }
    
    async def generate_strategy(self, 
                              patterns: List[RecognizedPattern],
                              timeframe: str,
                              market_context: pd.DataFrame) -> List[StrategySignal]:
        """
        Generiert Trading-Strategien basierend auf erkannten Patterns
        
        Args:
            patterns: Liste erkannter Patterns
            timeframe: Trading-Timeframe
            market_context: OHLCV-Marktdaten für Kontext
            
        Returns:
            Liste generierter Strategy-Signale
        """
        signals = []
        
        try:
            for pattern in patterns:
                if pattern.confidence < 0.6:  # Skip low-confidence patterns
                    continue
                
                # Determine strategy type based on pattern
                strategy_type = self.determine_strategy_type(pattern)
                
                # Generate signal based on pattern and strategy type
                signal = await self.create_strategy_signal(
                    pattern, strategy_type, timeframe, market_context
                )
                
                if signal:
                    signals.append(signal)
            
            # Sort signals by confidence
            signals.sort(key=lambda x: x.confidence, reverse=True)
            
            self.logger.info(f"Generated {len(signals)} strategy signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating strategies: {e}")
            return []
    
    def determine_strategy_type(self, pattern: RecognizedPattern) -> StrategyType:
        """Bestimmt Strategy-Typ basierend auf Pattern"""
        if pattern.pattern_type == PatternType.TREND_CONTINUATION:
            return StrategyType.MOMENTUM
        elif pattern.pattern_type == PatternType.TREND_REVERSAL:
            return StrategyType.MEAN_REVERSION
        elif pattern.pattern_type == PatternType.BREAKOUT:
            return StrategyType.BREAKOUT
        elif pattern.pattern_type == PatternType.SUPPORT_RESISTANCE:
            return StrategyType.SWING
        else:
            return StrategyType.HYBRID
    
    async def create_strategy_signal(self, 
                                   pattern: RecognizedPattern,
                                   strategy_type: StrategyType,
                                   timeframe: str,
                                   market_context: pd.DataFrame) -> Optional[StrategySignal]:
        """Erstellt Strategy-Signal basierend auf Pattern und Strategy-Typ"""
        try:
            template = self.strategy_templates.get(f"{strategy_type.value}_strategy", {})
            
            # Determine signal direction
            if pattern.entry_price > pattern.stop_loss:
                signal_type = "BUY"
            else:
                signal_type = "SELL"
            
            # Calculate position size based on risk
            risk_amount = abs(pattern.entry_price - pattern.stop_loss)
            max_risk = template.get("max_risk_per_trade", 0.01)
            position_size_factor = template.get("position_size_factor", 0.02)
            
            # Adjust position size based on risk
            if risk_amount > 0:
                position_size = min(position_size_factor, max_risk / risk_amount)
            else:
                position_size = position_size_factor
            
            # Adjust confidence based on market context
            adjusted_confidence = self.adjust_confidence_for_market(
                pattern.confidence, market_context, timeframe
            )
            
            signal = StrategySignal(
                signal_type=signal_type,
                entry_price=pattern.entry_price,
                stop_loss=pattern.stop_loss,
                take_profit=pattern.take_profit,
                position_size=position_size,
                confidence=adjusted_confidence,
                timeframe=timeframe,
                strategy_type=strategy_type,
                risk_reward=pattern.risk_reward_ratio,
                signal_metadata={
                    "pattern_id": id(pattern),
                    "pattern_type": pattern.pattern_type.value,
                    "template": template,
                    "generation_time": time.time()
                }
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating strategy signal: {e}")
            return None
    
    def adjust_confidence_for_market(self, 
                                   base_confidence: float,
                                   market_context: pd.DataFrame,
                                   timeframe: str) -> float:
        """Adjustiert Confidence basierend auf Marktkontext"""
        try:
            if market_context.empty:
                return base_confidence
            
            # Calculate market volatility
            returns = market_context['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate volume trend
            volume_trend = market_context['volume'].tail(10).mean() / market_context['volume'].tail(20).mean()
            
            # Adjust confidence based on market conditions
            confidence_adjustment = 1.0
            
            # High volatility reduces confidence
            if volatility > 0.02:  # 2% daily volatility
                confidence_adjustment *= 0.9
            elif volatility < 0.005:  # Very low volatility
                confidence_adjustment *= 0.95
            
            # Strong volume trend increases confidence
            if volume_trend > 1.2:
                confidence_adjustment *= 1.1
            elif volume_trend < 0.8:
                confidence_adjustment *= 0.9
            
            # Timeframe adjustment
            timeframe_multipliers = {
                "1m": 0.8, "5m": 0.9, "15m": 0.95,
                "1h": 1.0, "4h": 1.05, "1d": 1.1
            }
            confidence_adjustment *= timeframe_multipliers.get(timeframe, 1.0)
            
            adjusted_confidence = base_confidence * confidence_adjustment
            return min(1.0, max(0.0, adjusted_confidence))
            
        except Exception as e:
            self.logger.error(f"Error adjusting confidence: {e}")
            return base_confidence


class ChartToStrategyPipeline:
    """
    Hauptklasse für End-to-End Chart-to-Strategy Pipeline
    """
    
    def __init__(self, 
                 ollama_client: Optional[OllamaVisionClient] = None,
                 config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.ollama_client = ollama_client or OllamaVisionClient()
        self.chart_processor = ChartProcessor()
        self.pattern_recognizer = PatternRecognizer()
        self.strategy_generator = StrategyGenerator()
        
        # Initialize fusion agent for adaptive prompts
        self.fusion_agent = DynamicFusionAgent()
        
        # Cache for vision results
        self.vision_cache = {} if self.config.cache_vision_results else None
        
        # Statistics
        self.stats = {
            "total_charts_processed": 0,
            "successful_strategies": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("Chart-to-Strategy Pipeline initialized")
    
    def extract_market_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extrahiert Market-Features aus OHLCV-Daten"""
        try:
            if market_data.empty:
                return {}
            
            latest = market_data.iloc[-1]
            
            # Basic OHLCV features
            features = {
                "open": float(latest.get("open", 0)),
                "high": float(latest.get("high", 0)),
                "low": float(latest.get("low", 0)),
                "close": float(latest.get("close", 0)),
                "volume": float(latest.get("volume", 0))
            }
            
            # Calculate additional features
            if len(market_data) > 1:
                returns = market_data['close'].pct_change().dropna()
                features.update({
                    "volatility": float(returns.std()),
                    "trend_strength": float(abs(returns.mean())),
                    "volume_profile": "High" if latest.get("volume", 0) > market_data['volume'].mean() else "Normal"
                })
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting market features: {e}")
            return {}
    
    async def process_chart_to_strategy(self, 
                                      chart_path: Path,
                                      timeframe: str,
                                      market_data: pd.DataFrame,
                                      symbol: str = "EUR/USD") -> StrategyResult:
        """
        Hauptmethode: Vollständige Pipeline von Chart zu Trading-Strategy
        
        Args:
            chart_path: Pfad zum Chart-Image (1200x800 PNG)
            timeframe: Trading-Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            market_data: OHLCV-Daten für Kontext
            symbol: Trading-Symbol
            
        Returns:
            StrategyResult mit generierten Trading-Signalen
        """
        start_time = time.time()
        strategy_id = f"strategy_{symbol}_{timeframe}_{int(time.time())}"
        
        try:
            # Validate chart image
            if not self.chart_processor.validate_chart_image(chart_path):
                raise ValueError(f"Invalid chart image: {chart_path}")
            
            # Extract chart features
            chart_features = await self.chart_processor.extract_features(chart_path)
            
            # Check vision cache
            cache_key = f"{chart_path}_{timeframe}"
            vision_analysis = None
            
            if self.vision_cache and cache_key in self.vision_cache:
                vision_analysis = self.vision_cache[cache_key]
                self.stats["cache_hits"] += 1
                self.logger.info(f"Using cached vision analysis for {chart_path}")
            else:
                # Generate adaptive prompt using fusion agent
                chart_data = ChartData(
                    image_path=chart_path,
                    timeframe=timeframe,
                    symbol=symbol,
                    bars_count=len(market_data),
                    resolution=(1200, 800),
                    metadata={"source": "pipeline"}
                )
                
                market_features = self.extract_market_features(market_data)
                
                market_context = MarketContext(
                    volatility=market_features.get("volatility", 0.5),
                    trend_strength=market_features.get("trend_strength", 0.5),
                    volume_profile=market_features.get("volume_profile", "Normal"),
                    session_time="Unknown",
                    major_events=[],
                    timeframe=timeframe
                )
                
                # Generate adaptive prompt
                adaptive_prompt = await self.fusion_agent.generate_adaptive_prompt(
                    chart_data, market_features, market_context
                )
                
                # Vision Analysis via Ollama with adaptive prompt
                try:
                    vision_analysis = await asyncio.wait_for(
                        self.ollama_client.analyze_chart_pattern(
                            str(chart_path),
                            context={
                                "timeframe": timeframe,
                                "symbol": symbol,
                                "market_features": market_features,
                                "adaptive_prompt": adaptive_prompt.vision_prompt
                            }
                        ),
                        timeout=self.config.vision_timeout_seconds
                    )
                    
                    # Cache result if caching enabled
                    if self.vision_cache:
                        self.vision_cache[cache_key] = vision_analysis
                        # Limit cache size
                        if len(self.vision_cache) > self.config.max_cache_size:
                            # Remove oldest entry
                            oldest_key = next(iter(self.vision_cache))
                            del self.vision_cache[oldest_key]
                    
                    self.stats["cache_misses"] += 1
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Vision analysis timeout for {chart_path}")
                    vision_analysis = {"error": "timeout", "confidence": 0.0, "patterns": []}
                except Exception as e:
                    self.logger.error(f"Vision analysis failed: {e}")
                    vision_analysis = {"error": str(e), "confidence": 0.0, "patterns": []}
            
            # Pattern Recognition
            recognized_patterns = self.pattern_recognizer.identify_patterns(
                chart_features, vision_analysis
            )
            
            # Filter patterns by confidence threshold
            high_confidence_patterns = [
                p for p in recognized_patterns 
                if p.confidence >= self.config.pattern_confidence_threshold
            ]
            
            # Strategy Generation
            strategy_signals = await self.strategy_generator.generate_strategy(
                patterns=high_confidence_patterns,
                timeframe=timeframe,
                market_context=market_data
            )
            
            # Filter strategies by confidence threshold
            high_confidence_signals = [
                s for s in strategy_signals
                if s.confidence >= self.config.strategy_confidence_threshold
            ]
            
            # Calculate overall confidence and fusion quality
            if high_confidence_signals:
                overall_confidence = np.mean([s.confidence for s in high_confidence_signals])
            else:
                overall_confidence = 0.0
            
            fusion_quality = vision_analysis.get("confidence", 0.0) * chart_features.pattern_confidence
            
            processing_time = time.time() - start_time
            
            # Create result
            result = StrategyResult(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                signals=high_confidence_signals,
                patterns=high_confidence_patterns,
                chart_features=chart_features,
                confidence=overall_confidence,
                processing_time=processing_time,
                vision_analysis=vision_analysis,
                fusion_quality=fusion_quality,
                generation_timestamp=time.time(),
                metadata={
                    "chart_path": str(chart_path),
                    "market_data_points": len(market_data),
                    "cache_hit": cache_key in (self.vision_cache or {}),
                    "adaptive_prompt_used": True
                }
            )
            
            # Update statistics
            self.stats["total_charts_processed"] += 1
            if high_confidence_signals:
                self.stats["successful_strategies"] += 1
            
            # Update running average processing time
            total_processed = self.stats["total_charts_processed"]
            self.stats["average_processing_time"] = (
                (self.stats["average_processing_time"] * (total_processed - 1) + processing_time) / total_processed
            )
            
            # Provide feedback to fusion agent
            if hasattr(self, 'fusion_agent') and 'adaptive_prompt' in locals():
                inference_result = InferenceResult(
                    strategy_signals={s.signal_type: s.confidence for s in high_confidence_signals},
                    confidence_score=overall_confidence,
                    processing_time=processing_time,
                    model_used="ollama",
                    fusion_quality=fusion_quality
                )
                self.fusion_agent.adapt_based_on_feedback(adaptive_prompt, inference_result)
            
            self.logger.info(f"Processed chart {chart_path} in {processing_time:.2f}s, generated {len(high_confidence_signals)} signals")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed for {chart_path}: {e}")
            
            # Return error result
            return StrategyResult(
                strategy_id=strategy_id,
                symbol=symbol,
                timeframe=timeframe,
                signals=[],
                patterns=[],
                chart_features=ChartFeatures(
                    trend_direction="unknown", trend_strength=0.0,
                    support_levels=[], resistance_levels=[],
                    volatility_regime="unknown", volume_profile="unknown",
                    pattern_confidence=0.0, key_levels={},
                    timeframe_alignment=False
                ),
                confidence=0.0,
                processing_time=time.time() - start_time,
                vision_analysis={"error": str(e)},
                fusion_quality=0.0,
                generation_timestamp=time.time(),
                metadata={"error": str(e), "chart_path": str(chart_path)}
            )
    
    async def batch_process_charts(self, 
                                 chart_list: List[Tuple[Path, str, pd.DataFrame, str]]) -> List[StrategyResult]:
        """
        Batch-Processing für multiple Charts mit Parallelisierung
        
        Args:
            chart_list: Liste von (chart_path, timeframe, market_data, symbol) Tupeln
            
        Returns:
            Liste von StrategyResult-Objekten
        """
        if not self.config.enable_batch_processing:
            # Sequential processing
            results = []
            for chart_path, timeframe, market_data, symbol in chart_list:
                result = await self.process_chart_to_strategy(chart_path, timeframe, market_data, symbol)
                results.append(result)
            return results
        
        # Parallel processing with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_charts)
        
        async def process_with_semaphore(chart_path, timeframe, market_data, symbol):
            async with semaphore:
                return await self.process_chart_to_strategy(chart_path, timeframe, market_data, symbol)
        
        tasks = [
            process_with_semaphore(chart_path, timeframe, market_data, symbol)
            for chart_path, timeframe, market_data, symbol in chart_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing error for chart {i}: {result}")
            else:
                valid_results.append(result)
        
        self.logger.info(f"Batch processed {len(chart_list)} charts, {len(valid_results)} successful")
        
        return valid_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Gibt Pipeline-Statistiken zurück"""
        cache_stats = {}
        if self.vision_cache is not None:
            cache_stats = {
                "cache_size": len(self.vision_cache),
                "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            }
        
        return {
            **self.stats,
            **cache_stats,
            "fusion_agent_stats": self.fusion_agent.get_performance_stats() if hasattr(self, 'fusion_agent') else {}
        }
    
    def clear_cache(self):
        """Leert Vision-Cache"""
        if self.vision_cache:
            self.vision_cache.clear()
            self.logger.info("Vision cache cleared")


# Factory Function
def create_chart_to_strategy_pipeline(
    ollama_client: Optional[OllamaVisionClient] = None,
    config: Optional[PipelineConfig] = None
) -> ChartToStrategyPipeline:
    """Factory function für Chart-to-Strategy Pipeline"""
    return ChartToStrategyPipeline(ollama_client, config)


# Testing Function
async def test_chart_to_strategy_pipeline():
    """Test function für Chart-to-Strategy Pipeline"""
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_chart = Path("test_chart.png")
    test_timeframe = "1h"
    
    # Create mock market data
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    market_data = pd.DataFrame({
        'open': np.random.uniform(1.10, 1.11, 100),
        'high': np.random.uniform(1.105, 1.115, 100),
        'low': np.random.uniform(1.095, 1.105, 100),
        'close': np.random.uniform(1.10, 1.11, 100),
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Test pipeline
    pipeline = create_chart_to_strategy_pipeline()
    
    # Note: This would fail without actual chart file and Ollama setup
    # result = await pipeline.process_chart_to_strategy(test_chart, test_timeframe, market_data)
    
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline Stats: {stats}")
    
    print("Chart-to-Strategy Pipeline test completed")


if __name__ == "__main__":
    asyncio.run(test_chart_to_strategy_pipeline())