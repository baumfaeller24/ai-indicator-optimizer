"""
AI Models und Datenstrukturen
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image


@dataclass
class PatternAnalysis:
    """Ergebnis der visuellen Pattern-Analyse"""
    pattern_type: str
    confidence_score: float
    bounding_box: Optional[tuple] = None
    features: Optional[Dict[str, float]] = None
    description: Optional[str] = None


@dataclass
class OptimizedParameters:
    """Optimierte Indikator-Parameter"""
    indicator_name: str
    parameters: Dict[str, Any]
    performance_score: float
    backtest_results: Optional[Dict[str, float]] = None


@dataclass
class MultimodalInput:
    """Multimodale Eingabe für KI-Model"""
    chart_images: List[Image.Image]
    numerical_data: np.ndarray
    indicator_data: Dict[str, Any]
    market_context: Dict[str, Any]
    
    def validate(self) -> bool:
        """Validiert Eingabedaten"""
        return (
            len(self.chart_images) > 0 and
            self.numerical_data.size > 0 and
            len(self.indicator_data) > 0
        )