"""
Visual Pattern Analyzer für Candlestick-Pattern-Erkennung in Chart-Images.
Nutzt MiniCPM-4.1-8B Vision Model für multimodale Chart-Analyse.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import cv2
import logging
from pathlib import Path

from ..core.hardware_detector import HardwareDetector
from ..data.models import OHLCVData, IndicatorData
from .multimodal_ai import MultimodalAI

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Erkannte Candlestick-Pattern-Typen"""
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    HARAMI = "harami"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD = "dark_cloud"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    SUPPORT_LEVEL = "support_level"
    RESISTANCE_LEVEL = "resistance_level"
    TREND_LINE = "trend_line"
    TRIANGLE = "triangle"
    HEAD_SHOULDERS = "head_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"

@dataclass
class VisualPattern:
    """Erkanntes visuelles Pattern mit Metadaten"""
    pattern_type: PatternType
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    timeframe_start: int
    timeframe_end: int
    price_level: Optional[float] = None
    direction: Optional[str] = None  # "bullish", "bearish", "neutral"
    strength: Optional[float] = None
    context_features: Optional[Dict[str, Any]] = None

@dataclass
class PatternAnalysisResult:
    """Ergebnis der visuellen Pattern-Analyse"""
    patterns: List[VisualPattern]
    overall_sentiment: str  # "bullish", "bearish", "neutral"
    confidence_score: float
    market_structure: Dict[str, Any]
    key_levels: List[float]
    analysis_metadata: Dict[str, Any]

class VisualPatternAnalyzer:
    """
    Analysiert Chart-Images zur Erkennung von Candlestick-Patterns und technischen Formationen.
    Nutzt MiniCPM-4.1-8B Vision Model für multimodale Analyse.
    """
    
    def __init__(self, multimodal_ai: MultimodalAI, hardware_detector: HardwareDetector):
        self.multimodal_ai = multimodal_ai
        self.hardware_detector = hardware_detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pattern-spezifische Prompts für Vision Model
        self.pattern_prompts = self._initialize_pattern_prompts()
        
        # Confidence-Schwellenwerte für verschiedene Pattern-Typen
        self.confidence_thresholds = {
            PatternType.DOJI: 0.7,
            PatternType.HAMMER: 0.75,
            PatternType.ENGULFING_BULLISH: 0.8,
            PatternType.ENGULFING_BEARISH: 0.8,
            PatternType.SUPPORT_LEVEL: 0.85,
            PatternType.RESISTANCE_LEVEL: 0.85,
            PatternType.HEAD_SHOULDERS: 0.9,
            PatternType.DOUBLE_TOP: 0.85,
            PatternType.DOUBLE_BOTTOM: 0.85
        }
        
        logger.info(f"VisualPatternAnalyzer initialisiert auf {self.device}")
    
    def analyze_chart_image(self, 
                          chart_image: Image.Image,
                          ohlcv_data: OHLCVData,
                          indicator_data: Optional[IndicatorData] = None) -> PatternAnalysisResult:
        """
        Analysiert ein Chart-Image zur Erkennung visueller Patterns.
        
        Args:
            chart_image: PIL Image des Charts
            ohlcv_data: Zugehörige OHLCV-Daten
            indicator_data: Optionale Indikator-Daten für Kontext
            
        Returns:
            PatternAnalysisResult mit erkannten Patterns
        """
        try:
            logger.info("Starte visuelle Pattern-Analyse")
            
            # Preprocessing des Chart-Images
            processed_image = self._preprocess_chart_image(chart_image)
            
            # Multimodale Analyse mit MiniCPM
            vision_analysis = self._analyze_with_vision_model(processed_image, ohlcv_data)
            
            # Pattern-Erkennung
            detected_patterns = self._detect_patterns(vision_analysis, processed_image, ohlcv_data)
            
            # Marktstruktur-Analyse
            market_structure = self._analyze_market_structure(vision_analysis, ohlcv_data)
            
            # Key-Level-Erkennung
            key_levels = self._identify_key_levels(vision_analysis, ohlcv_data)
            
            # Overall Sentiment bestimmen
            overall_sentiment = self._determine_overall_sentiment(detected_patterns, market_structure)
            
            # Confidence Score berechnen
            confidence_score = self._calculate_overall_confidence(detected_patterns)
            
            result = PatternAnalysisResult(
                patterns=detected_patterns,
                overall_sentiment=overall_sentiment,
                confidence_score=confidence_score,
                market_structure=market_structure,
                key_levels=key_levels,
                analysis_metadata={
                    "image_size": chart_image.size,
                    "data_points": len(ohlcv_data.close) if hasattr(ohlcv_data, 'close') else 0,
                    "analysis_timestamp": torch.tensor(0).item(),  # Placeholder
                    "model_version": "MiniCPM-4.1-8B"
                }
            )
            
            logger.info(f"Pattern-Analyse abgeschlossen: {len(detected_patterns)} Patterns erkannt")
            return result
            
        except Exception as e:
            logger.exception(f"Fehler bei visueller Pattern-Analyse: {e}")
            # Fallback: Leeres Ergebnis zurückgeben
            return PatternAnalysisResult(
                patterns=[],
                overall_sentiment="neutral",
                confidence_score=0.0,
                market_structure={},
                key_levels=[],
                analysis_metadata={"error": str(e)}
            )
    
    def _preprocess_chart_image(self, image: Image.Image) -> Image.Image:
        """Preprocessing des Chart-Images für optimale Pattern-Erkennung"""
        try:
            # Konvertiere zu RGB falls nötig
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Optimale Größe für MiniCPM Vision Model
            target_size = (1024, 768)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Kontrast-Enhancement für bessere Pattern-Erkennung
            image_array = np.array(image)
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Image-Preprocessing fehlgeschlagen: {e}, verwende Original")
            return image
    
    def _analyze_with_vision_model(self, image: Image.Image, ohlcv_data: OHLCVData) -> Dict[str, Any]:
        """Führt multimodale Analyse mit MiniCPM Vision Model durch"""
        try:
            # Erstelle kontextuellen Prompt
            context_prompt = self._create_analysis_prompt(ohlcv_data)
            
            # Vision Model Inference
            vision_result = self.multimodal_ai.analyze_chart_pattern(image, context_prompt)
            
            return vision_result
            
        except Exception as e:
            logger.exception(f"Vision Model Analyse fehlgeschlagen: {e}")
            return {"error": str(e), "patterns": [], "sentiment": "neutral"}
    
    def _detect_patterns(self, 
                        vision_analysis: Dict[str, Any], 
                        image: Image.Image, 
                        ohlcv_data: OHLCVData) -> List[VisualPattern]:
        """Extrahiert spezifische Patterns aus der Vision-Analyse"""
        patterns = []
        
        try:
            # Pattern-Erkennung basierend auf Vision Model Output
            if "patterns" in vision_analysis:
                for pattern_data in vision_analysis["patterns"]:
                    pattern = self._create_visual_pattern(pattern_data, ohlcv_data)
                    if pattern and pattern.confidence >= self.confidence_thresholds.get(pattern.pattern_type, 0.7):
                        patterns.append(pattern)
            
            # Zusätzliche regelbasierte Pattern-Erkennung als Backup
            rule_based_patterns = self._detect_rule_based_patterns(ohlcv_data)
            patterns.extend(rule_based_patterns)
            
            # Duplikate entfernen und nach Confidence sortieren
            patterns = self._deduplicate_patterns(patterns)
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
        except Exception as e:
            logger.exception(f"Pattern-Erkennung fehlgeschlagen: {e}")
        
        return patterns
    
    def _create_visual_pattern(self, pattern_data: Dict[str, Any], ohlcv_data: OHLCVData) -> Optional[VisualPattern]:
        """Erstellt VisualPattern-Objekt aus Vision Model Output"""
        try:
            pattern_type_str = pattern_data.get("type", "").lower()
            pattern_type = None
            
            # Mapping von String zu PatternType Enum
            for pt in PatternType:
                if pt.value in pattern_type_str or pattern_type_str in pt.value:
                    pattern_type = pt
                    break
            
            if not pattern_type:
                return None
            
            return VisualPattern(
                pattern_type=pattern_type,
                confidence=float(pattern_data.get("confidence", 0.0)),
                bounding_box=tuple(pattern_data.get("bounding_box", [0, 0, 100, 100])),
                timeframe_start=int(pattern_data.get("start_index", 0)),
                timeframe_end=int(pattern_data.get("end_index", 0)),
                price_level=pattern_data.get("price_level"),
                direction=pattern_data.get("direction"),
                strength=pattern_data.get("strength"),
                context_features=pattern_data.get("context", {})
            )
            
        except Exception as e:
            logger.warning(f"Fehler beim Erstellen von VisualPattern: {e}")
            return None
    
    def _detect_rule_based_patterns(self, ohlcv_data: OHLCVData) -> List[VisualPattern]:
        """Regelbasierte Pattern-Erkennung als Backup"""
        patterns = []
        
        try:
            if not hasattr(ohlcv_data, 'close') or len(ohlcv_data.close) < 3:
                return patterns
            
            closes = ohlcv_data.close
            opens = ohlcv_data.open
            highs = ohlcv_data.high
            lows = ohlcv_data.low
            
            # Einfache Doji-Erkennung
            for i in range(1, len(closes) - 1):
                body_size = abs(closes[i] - opens[i])
                candle_range = highs[i] - lows[i]
                
                if candle_range > 0 and body_size / candle_range < 0.1:  # Doji-Kriterium
                    patterns.append(VisualPattern(
                        pattern_type=PatternType.DOJI,
                        confidence=0.75,
                        bounding_box=(i-1, 0, 3, 100),
                        timeframe_start=i-1,
                        timeframe_end=i+1,
                        price_level=closes[i],
                        direction="neutral",
                        strength=0.7
                    ))
            
        except Exception as e:
            logger.warning(f"Regelbasierte Pattern-Erkennung fehlgeschlagen: {e}")
        
        return patterns
    
    def _analyze_market_structure(self, vision_analysis: Dict[str, Any], ohlcv_data: OHLCVData) -> Dict[str, Any]:
        """Analysiert die Marktstruktur"""
        try:
            return {
                "trend": vision_analysis.get("trend", "neutral"),
                "volatility": vision_analysis.get("volatility", "medium"),
                "support_resistance": vision_analysis.get("support_resistance", []),
                "market_phase": vision_analysis.get("market_phase", "consolidation")
            }
        except Exception as e:
            logger.warning(f"Marktstruktur-Analyse fehlgeschlagen: {e}")
            return {"trend": "neutral", "volatility": "medium"}
    
    def _identify_key_levels(self, vision_analysis: Dict[str, Any], ohlcv_data: OHLCVData) -> List[float]:
        """Identifiziert wichtige Preis-Level"""
        try:
            key_levels = vision_analysis.get("key_levels", [])
            
            # Zusätzliche Level aus OHLCV-Daten
            if hasattr(ohlcv_data, 'high') and hasattr(ohlcv_data, 'low'):
                recent_high = max(ohlcv_data.high[-20:]) if len(ohlcv_data.high) >= 20 else max(ohlcv_data.high)
                recent_low = min(ohlcv_data.low[-20:]) if len(ohlcv_data.low) >= 20 else min(ohlcv_data.low)
                
                key_levels.extend([recent_high, recent_low])
            
            return sorted(list(set(key_levels)))
            
        except Exception as e:
            logger.warning(f"Key-Level-Identifikation fehlgeschlagen: {e}")
            return []
    
    def _determine_overall_sentiment(self, patterns: List[VisualPattern], market_structure: Dict[str, Any]) -> str:
        """Bestimmt das Gesamt-Sentiment basierend auf erkannten Patterns"""
        try:
            bullish_score = 0
            bearish_score = 0
            
            # Pattern-basierte Bewertung
            for pattern in patterns:
                if pattern.direction == "bullish":
                    bullish_score += pattern.confidence
                elif pattern.direction == "bearish":
                    bearish_score += pattern.confidence
            
            # Marktstruktur-Einfluss
            trend = market_structure.get("trend", "neutral")
            if trend == "bullish":
                bullish_score += 0.3
            elif trend == "bearish":
                bearish_score += 0.3
            
            # Sentiment bestimmen
            if bullish_score > bearish_score + 0.2:
                return "bullish"
            elif bearish_score > bullish_score + 0.2:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.warning(f"Sentiment-Bestimmung fehlgeschlagen: {e}")
            return "neutral"
    
    def _calculate_overall_confidence(self, patterns: List[VisualPattern]) -> float:
        """Berechnet Gesamt-Confidence-Score"""
        try:
            if not patterns:
                return 0.0
            
            # Gewichteter Durchschnitt der Pattern-Confidences
            total_weight = sum(p.confidence for p in patterns)
            if total_weight == 0:
                return 0.0
            
            weighted_confidence = sum(p.confidence ** 2 for p in patterns) / total_weight
            return min(weighted_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Confidence-Berechnung fehlgeschlagen: {e}")
            return 0.0
    
    def _deduplicate_patterns(self, patterns: List[VisualPattern]) -> List[VisualPattern]:
        """Entfernt doppelte Patterns"""
        try:
            unique_patterns = []
            seen_patterns = set()
            
            for pattern in patterns:
                pattern_key = (pattern.pattern_type, pattern.timeframe_start, pattern.timeframe_end)
                if pattern_key not in seen_patterns:
                    unique_patterns.append(pattern)
                    seen_patterns.add(pattern_key)
            
            return unique_patterns
            
        except Exception as e:
            logger.warning(f"Pattern-Deduplizierung fehlgeschlagen: {e}")
            return patterns
    
    def _create_analysis_prompt(self, ohlcv_data: OHLCVData) -> str:
        """Erstellt kontextuellen Prompt für Vision Model"""
        try:
            prompt = """Analyze this EUR/USD forex chart image for trading patterns. 
            
            Focus on identifying:
            1. Candlestick patterns (doji, hammer, engulfing, etc.)
            2. Support and resistance levels
            3. Trend lines and chart formations
            4. Overall market sentiment and direction
            
            Provide confidence scores for each identified pattern.
            Consider the current market context and timeframe."""
            
            # Zusätzlicher Kontext aus OHLCV-Daten
            if hasattr(ohlcv_data, 'close') and len(ohlcv_data.close) > 0:
                current_price = ohlcv_data.close[-1]
                prompt += f"\n\nCurrent price level: {current_price:.5f}"
            
            return prompt
            
        except Exception as e:
            logger.warning(f"Prompt-Erstellung fehlgeschlagen: {e}")
            return "Analyze this forex chart for trading patterns."
    
    def _initialize_pattern_prompts(self) -> Dict[PatternType, str]:
        """Initialisiert Pattern-spezifische Prompts"""
        return {
            PatternType.DOJI: "Look for doji candlestick patterns with small bodies and long wicks",
            PatternType.HAMMER: "Identify hammer patterns with small bodies and long lower wicks",
            PatternType.ENGULFING_BULLISH: "Find bullish engulfing patterns where a large green candle engulfs the previous red candle",
            PatternType.ENGULFING_BEARISH: "Find bearish engulfing patterns where a large red candle engulfs the previous green candle",
            PatternType.SUPPORT_LEVEL: "Identify horizontal support levels where price has bounced multiple times",
            PatternType.RESISTANCE_LEVEL: "Identify horizontal resistance levels where price has been rejected multiple times",
            PatternType.HEAD_SHOULDERS: "Look for head and shoulders reversal patterns",
            PatternType.DOUBLE_TOP: "Identify double top reversal patterns",
            PatternType.DOUBLE_BOTTOM: "Identify double bottom reversal patterns"
        }