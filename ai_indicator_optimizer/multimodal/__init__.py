#!/usr/bin/env python3
"""
Multimodal Integration Package
U3 - Unified Multimodal Flow Integration

Components:
- Dynamic Fusion Agent: Adaptive Vision+Text-Prompts
- Chart-to-Strategy Pipeline: End-to-End Processing
- Multimodal Confidence Scorer: Cross-Modal Validation
- Real-time Switcher: Load-based Model Selection
"""

from .dynamic_fusion_agent import (
    DynamicFusionAgent,
    FusionStrategy,
    ModelPreference,
    AdaptivePrompt,
    MarketContext,
    ChartData,
    InferenceResult,
    FusionConfig,
    create_dynamic_fusion_agent
)

__all__ = [
    "DynamicFusionAgent",
    "FusionStrategy", 
    "ModelPreference",
    "AdaptivePrompt",
    "MarketContext",
    "ChartData",
    "InferenceResult",
    "FusionConfig",
    "create_dynamic_fusion_agent"
]

__version__ = "1.0.0"
__author__ = "AI-Indicator-Optimizer Team"