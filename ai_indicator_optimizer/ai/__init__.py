"""
AI Module: MiniCPM-4.1-8B Integration und multimodale Analyse
"""

from .multimodal_ai import MultimodalAI
from .fine_tuning import FineTuningManager
from .models import PatternAnalysis, OptimizedParameters, MultimodalInput
from .visual_pattern_analyzer import VisualPatternAnalyzer, VisualPattern, PatternAnalysisResult, PatternType
from .numerical_indicator_optimizer import NumericalIndicatorOptimizer, OptimizationResult, IndicatorType, OptimizationConfig
from .multimodal_strategy_generator import MultimodalStrategyGenerator, StrategyGenerationResult, TradingStrategy, TradingSignal
from .confidence_scoring import ConfidenceScoring, ConfidenceMetrics, ConfidenceLevel
from .pattern_recognition_engine import MultimodalPatternRecognitionEngine, PatternRecognitionConfig, PatternRecognitionResult

__all__ = [
    'MultimodalAI', 'FineTuningManager', 'PatternAnalysis', 'OptimizedParameters', 'MultimodalInput',
    'VisualPatternAnalyzer', 'VisualPattern', 'PatternAnalysisResult', 'PatternType',
    'NumericalIndicatorOptimizer', 'OptimizationResult', 'IndicatorType', 'OptimizationConfig',
    'MultimodalStrategyGenerator', 'StrategyGenerationResult', 'TradingStrategy', 'TradingSignal',
    'ConfidenceScoring', 'ConfidenceMetrics', 'ConfidenceLevel',
    'MultimodalPatternRecognitionEngine', 'PatternRecognitionConfig', 'PatternRecognitionResult'
]