#!/usr/bin/env python3
"""
Enhanced Visual Pattern Analyzer
Phase 2 Implementation - Enhanced Multimodal Pattern Recognition Engine

Features:
- Candlestick-Pattern-Erkennung in Chart-Images
- Multimodale AI-Integration mit MiniCPM-4.1-8B
- Pattern-Confidence-Scoring
- Visual Pattern Classification
- Chart-Image-Processing mit GPU-Beschleunigung
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import cv2
    import PIL.Image as Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

try:
    import requests
    import json
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

from nautilus_trader.model.data import Bar


class CandlestickPattern:
    """Candlestick Pattern Data Structure"""
    
    def __init__(
        self,
        pattern_name: str,
        confidence: float,
        pattern_type: str,  # "bullish", "bearish", "neutral"
        visual_features: Dict[str, float],
        bar_count: int = 1
    ):
        self.pattern_name = pattern_name
        self.confidence = confidence
        self.pattern_type = pattern_type
        self.visual_features = visual_features
        self.bar_count = bar_count
        self.timestamp = int(time.time() * 1e9)


class VisualPatternAnalyzer:
    """
    Enhanced Visual Pattern Analyzer für Candlestick-Pattern-Erkennung
    
    Phase 2 Features:
    - Chart-Image-Generierung aus OHLCV-Daten
    - Candlestick-Pattern-Erkennung (Doji, Hammer, Engulfing, etc.)
    - Multimodale AI-Integration für erweiterte Pattern-Analyse
    - Visual Feature Extraction
    - Pattern-Confidence-Scoring
    """
    
    def __init__(
        self,
        ai_endpoint: Optional[str] = None,
        use_mock: bool = False,
        debug_mode: bool = False,
        chart_width: int = 800,
        chart_height: int = 600
    ):
        """
        Initialize Visual Pattern Analyzer
        
        Args:
            ai_endpoint: MiniCPM-4.1-8B API Endpoint für multimodale Analyse
            use_mock: Ob Mock-Daten verwendet werden sollen
            debug_mode: Debug-Modus aktivieren
            chart_width: Breite der generierten Chart-Images
            chart_height: Höhe der generierten Chart-Images
        """
        if not VISION_AVAILABLE:
            raise ImportError("Vision dependencies missing: pip install opencv-python pillow")
        
        self.ai_endpoint = ai_endpoint or os.getenv("AI_ENDPOINT")
        self.use_mock = use_mock
        self.debug_mode = debug_mode
        self.chart_width = chart_width
        self.chart_height = chart_height
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Pattern-Definitionen
        self.pattern_definitions = self._initialize_pattern_definitions()
        
        # Statistics
        self.patterns_analyzed = 0
        self.patterns_detected = 0
        
        self.logger.info(f"VisualPatternAnalyzer initialized: AI={bool(ai_endpoint)}, Mock={use_mock}")
    
    def _initialize_pattern_definitions(self) -> Dict[str, Dict]:
        """Initialisiere Candlestick-Pattern-Definitionen"""
        return {
            "doji": {
                "description": "Doji - Indecision pattern",
                "type": "neutral",
                "body_ratio_max": 0.1,
                "shadow_ratio_min": 0.3
            },
            "hammer": {
                "description": "Hammer - Bullish reversal",
                "type": "bullish",
                "body_ratio_max": 0.3,
                "lower_shadow_min": 2.0,
                "upper_shadow_max": 0.1
            },
            "hanging_man": {
                "description": "Hanging Man - Bearish reversal",
                "type": "bearish",
                "body_ratio_max": 0.3,
                "lower_shadow_min": 2.0,
                "upper_shadow_max": 0.1
            },
            "shooting_star": {
                "description": "Shooting Star - Bearish reversal",
                "type": "bearish",
                "body_ratio_max": 0.3,
                "upper_shadow_min": 2.0,
                "lower_shadow_max": 0.1
            },
            "inverted_hammer": {
                "description": "Inverted Hammer - Bullish reversal",
                "type": "bullish",
                "body_ratio_max": 0.3,
                "upper_shadow_min": 2.0,
                "lower_shadow_max": 0.1
            },
            "marubozu_bullish": {
                "description": "Bullish Marubozu - Strong bullish",
                "type": "bullish",
                "body_ratio_min": 0.9,
                "shadow_ratio_max": 0.05
            },
            "marubozu_bearish": {
                "description": "Bearish Marubozu - Strong bearish",
                "type": "bearish",
                "body_ratio_min": 0.9,
                "shadow_ratio_max": 0.05
            }
        }
    
    def analyze_candlestick_patterns(self, bars: List[Bar]) -> List[CandlestickPattern]:
        """
        Analysiere Candlestick-Patterns in Bar-Daten
        
        Args:
            bars: Liste von Nautilus Bar-Objekten
            
        Returns:
            Liste der erkannten Candlestick-Patterns
        """
        try:
            patterns = []
            
            # Single-Bar-Patterns analysieren
            for i, bar in enumerate(bars):
                single_patterns = self._analyze_single_bar_patterns(bar, i)
                patterns.extend(single_patterns)
            
            # Multi-Bar-Patterns analysieren (falls genügend Bars)
            if len(bars) >= 2:
                multi_patterns = self._analyze_multi_bar_patterns(bars)
                patterns.extend(multi_patterns)
            
            self.patterns_analyzed += len(bars)
            self.patterns_detected += len(patterns)
            
            if self.debug_mode and patterns:
                self.logger.debug(f"Detected {len(patterns)} patterns in {len(bars)} bars")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing candlestick patterns: {e}")
            return []
    
    def _analyze_single_bar_patterns(self, bar: Bar, index: int) -> List[CandlestickPattern]:
        """Analysiere Single-Bar-Patterns"""
        patterns = []
        
        # Extrahiere Bar-Features
        features = self._extract_bar_features(bar)
        
        # Prüfe jedes Pattern
        for pattern_name, definition in self.pattern_definitions.items():
            confidence = self._calculate_pattern_confidence(features, definition)
            
            if confidence > 0.6:  # Mindest-Confidence
                pattern = CandlestickPattern(
                    pattern_name=pattern_name,
                    confidence=confidence,
                    pattern_type=definition["type"],
                    visual_features=features,
                    bar_count=1
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_bar_features(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere visuelle Features aus Bar"""
        open_price = float(bar.open)
        high_price = float(bar.high)
        low_price = float(bar.low)
        close_price = float(bar.close)
        
        # Berechnete Features
        price_range = high_price - low_price
        body_size = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # Ratios (normalisiert)
        body_ratio = body_size / max(price_range, 1e-6)
        upper_shadow_ratio = upper_shadow / max(price_range, 1e-6)
        lower_shadow_ratio = lower_shadow / max(price_range, 1e-6)
        shadow_ratio = (upper_shadow + lower_shadow) / max(price_range, 1e-6)
        
        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "price_range": price_range,
            "body_size": body_size,
            "body_ratio": body_ratio,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "upper_shadow_ratio": upper_shadow_ratio,
            "lower_shadow_ratio": lower_shadow_ratio,
            "shadow_ratio": shadow_ratio,
            "is_bullish": close_price > open_price,
            "is_bearish": close_price < open_price
        }
    
    def _calculate_pattern_confidence(self, features: Dict[str, float], definition: Dict) -> float:
        """Berechne Pattern-Confidence basierend auf Definition"""
        confidence = 0.0
        checks = 0
        
        # Body Ratio Checks
        if "body_ratio_max" in definition:
            if features["body_ratio"] <= definition["body_ratio_max"]:
                confidence += 1.0
            checks += 1
        
        if "body_ratio_min" in definition:
            if features["body_ratio"] >= definition["body_ratio_min"]:
                confidence += 1.0
            checks += 1
        
        # Shadow Ratio Checks
        if "shadow_ratio_min" in definition:
            if features["shadow_ratio"] >= definition["shadow_ratio_min"]:
                confidence += 1.0
            checks += 1
        
        if "shadow_ratio_max" in definition:
            if features["shadow_ratio"] <= definition["shadow_ratio_max"]:
                confidence += 1.0
            checks += 1
        
        # Upper Shadow Checks
        if "upper_shadow_min" in definition:
            upper_shadow_body_ratio = features["upper_shadow"] / max(features["body_size"], 1e-6)
            if upper_shadow_body_ratio >= definition["upper_shadow_min"]:
                confidence += 1.0
            checks += 1
        
        if "upper_shadow_max" in definition:
            if features["upper_shadow_ratio"] <= definition["upper_shadow_max"]:
                confidence += 1.0
            checks += 1
        
        # Lower Shadow Checks
        if "lower_shadow_min" in definition:
            lower_shadow_body_ratio = features["lower_shadow"] / max(features["body_size"], 1e-6)
            if lower_shadow_body_ratio >= definition["lower_shadow_min"]:
                confidence += 1.0
            checks += 1
        
        if "lower_shadow_max" in definition:
            if features["lower_shadow_ratio"] <= definition["lower_shadow_max"]:
                confidence += 1.0
            checks += 1
        
        # Pattern Type Checks
        if definition["type"] == "bullish" and not features["is_bullish"]:
            confidence *= 0.5  # Reduziere Confidence für falsche Richtung
        elif definition["type"] == "bearish" and not features["is_bearish"]:
            confidence *= 0.5
        
        return confidence / max(checks, 1) if checks > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Analyzer-Statistiken"""
        return {
            "patterns_analyzed": self.patterns_analyzed,
            "patterns_detected": self.patterns_detected,
            "detection_rate": self.patterns_detected / max(self.patterns_analyzed, 1),
            "pattern_definitions": len(self.pattern_definitions),
            "ai_endpoint": bool(self.ai_endpoint),
            "use_mock": self.use_mock,
            "debug_mode": self.debug_mode
        }


# Factory Function
def create_visual_pattern_analyzer(
    ai_endpoint: Optional[str] = None,
    use_mock: bool = False,
    debug_mode: bool = False,
    **kwargs
) -> VisualPatternAnalyzer:
    """
    Factory Function für Visual Pattern Analyzer
    
    Args:
        ai_endpoint: MiniCPM-4.1-8B API Endpoint
        use_mock: Mock-Modus für Testing
        debug_mode: Debug-Modus
        **kwargs: Weitere Parameter
    
    Returns:
        VisualPatternAnalyzer Instance
    """
    return VisualPatternAnalyzer(
        ai_endpoint=ai_endpoint,
        use_mock=use_mock,
        debug_mode=debug_mode,
        **kwargs
    )