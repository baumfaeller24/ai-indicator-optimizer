"""
AI Module: Enhanced Multimodal Pattern Recognition Engine
Phase 2 Implementation
"""

# Phase 2 Enhanced Components - Import nur was existiert
try:
    from .visual_pattern_analyzer import VisualPatternAnalyzer, CandlestickPattern, create_visual_pattern_analyzer
except ImportError:
    pass

try:
    from .enhanced_feature_extractor import EnhancedFeatureExtractor, create_enhanced_feature_extractor
except ImportError:
    pass

try:
    from .confidence_position_sizer import ConfidencePositionSizer, create_confidence_position_sizer
except ImportError:
    pass

try:
    from .live_control_system import LiveControlSystem, create_live_control_system
except ImportError:
    pass

try:
    from .environment_config import EnvironmentConfigManager, get_config_manager, get_config
except ImportError:
    pass

try:
    from .confidence_scoring import EnhancedConfidenceScorer, create_enhanced_confidence_scorer, ConfidenceLevel
except ImportError:
    pass

# Legacy Components (falls vorhanden)
try:
    from .multimodal_ai import MultimodalAI
except ImportError:
    pass

try:
    from .fine_tuning import FineTuningManager
except ImportError:
    pass

__all__ = [
    # Phase 2 Components
    'VisualPatternAnalyzer', 'CandlestickPattern', 'create_visual_pattern_analyzer',
    'EnhancedFeatureExtractor', 'create_enhanced_feature_extractor',
    'ConfidencePositionSizer', 'create_confidence_position_sizer',
    'LiveControlSystem', 'create_live_control_system',
    'EnvironmentConfigManager', 'get_config_manager', 'get_config',
    'EnhancedConfidenceScorer', 'create_enhanced_confidence_scorer', 'ConfidenceLevel',
    # Legacy Components
    'MultimodalAI', 'FineTuningManager'
]