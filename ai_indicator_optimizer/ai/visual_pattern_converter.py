#!/usr/bin/env python3
"""
Visual Pattern to Pine Script Converter für Pattern-Logic-Translation
Phase 3 Implementation - Task 10

Features:
- Konvertierung visueller Candlestick-Pattern zu Pine Script Code
- AI-basierte Pattern-Erkennung und Code-Generierung
- Template-basierte Pine Script Generierung für Pattern
- Integration mit Enhanced Feature Extractor
- Confidence-basierte Pattern-Validation
"""

import re
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum

# Local Imports
from .enhanced_feature_extractor import EnhancedFeatureExtractor
from .pine_script_generator import PineScriptGenerator, PineScriptConfig, IndicatorType


class PatternType(Enum):
    """Unterstützte Candlestick-Pattern-Typen"""
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    SPINNING_TOP = "spinning_top"
    MARUBOZU_BULLISH = "marubozu_bullish"
    MARUBOZU_BEARISH = "marubozu_bearish"


class PatternSignal(Enum):
    """Pattern-Signal-Typen"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


@dataclass
class PatternDefinition:
    """Definition eines Candlestick-Patterns"""
    pattern_type: PatternType
    signal: PatternSignal
    candle_count: int  # Anzahl Kerzen für das Pattern
    conditions: List[str]  # Pine Script Bedingungen
    description: str
    reliability: float = 0.7  # Zuverlässigkeit des Patterns (0-1)
    
    # Pattern-spezifische Parameter
    body_ratio_threshold: float = 0.1
    shadow_ratio_threshold: float = 2.0
    size_comparison_threshold: float = 1.5


@dataclass
class ConversionResult:
    """Ergebnis einer Pattern-zu-Pine-Script Konvertierung"""
    success: bool
    pattern_type: PatternType
    pine_script: str
    pattern_conditions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    conversion_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualPatternConverter:
    """
    Converter für visuelle Pattern zu Pine Script Code
    
    Features:
    - Pattern-Definition und -Erkennung
    - Pine Script Code-Generierung
    - AI-basierte Pattern-Analyse
    - Template-basierte Code-Erstellung
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.feature_extractor = EnhancedFeatureExtractor()
        self.pine_generator = PineScriptGenerator()
        
        # Pattern-Definitionen laden
        self.pattern_definitions = self._load_pattern_definitions()
        
        # Pine Script Templates für Pattern
        self.pattern_templates = self._load_pattern_templates()
        
        # Statistiken
        self.stats = {
            "patterns_converted": 0,
            "successful_conversions": 0,
            "avg_conversion_time": 0.0,
            "patterns_by_type": {pattern.value: 0 for pattern in PatternType},
            "avg_confidence": 0.0
        }
        
        self.logger.info("VisualPatternConverter initialized")
    
    def convert_pattern_to_pine_script(
        self,
        pattern_type: PatternType,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> ConversionResult:
        """
        Konvertiere ein visuelles Pattern zu Pine Script Code
        
        Args:
            pattern_type: Typ des zu konvertierenden Patterns
            custom_parameters: Optionale custom Parameter
            
        Returns:
            ConversionResult mit generiertem Pine Script
        """
        try:
            start_time = datetime.now()
            
            # Pattern-Definition laden
            if pattern_type not in self.pattern_definitions:
                raise ValueError(f"Pattern type {pattern_type.value} not supported")
            
            pattern_def = self.pattern_definitions[pattern_type]
            
            # Pine Script generieren
            pine_script = self._generate_pattern_pine_script(pattern_def, custom_parameters)
            
            # Pattern-Bedingungen extrahieren
            pattern_conditions = self._extract_pattern_conditions(pattern_def)
            
            # Confidence berechnen
            confidence = self._calculate_pattern_confidence(pattern_def, custom_parameters)
            
            conversion_time = (datetime.now() - start_time).total_seconds()
            
            result = ConversionResult(
                success=True,
                pattern_type=pattern_type,
                pine_script=pine_script,
                pattern_conditions=pattern_conditions,
                confidence=confidence,
                conversion_time=conversion_time,
                metadata={
                    "candle_count": pattern_def.candle_count,
                    "signal": pattern_def.signal.value,
                    "reliability": pattern_def.reliability,
                    "description": pattern_def.description
                }
            )
            
            # Statistiken updaten
            self.stats["patterns_converted"] += 1
            self.stats["successful_conversions"] += 1
            self.stats["patterns_by_type"][pattern_type.value] += 1
            self.stats["avg_conversion_time"] = (
                (self.stats["avg_conversion_time"] * (self.stats["patterns_converted"] - 1) + conversion_time) 
                / self.stats["patterns_converted"]
            )
            self.stats["avg_confidence"] = (
                (self.stats["avg_confidence"] * (self.stats["patterns_converted"] - 1) + confidence) 
                / self.stats["patterns_converted"]
            )
            
            self.logger.info(f"Pattern {pattern_type.value} converted successfully (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern conversion error: {e}")
            
            self.stats["patterns_converted"] += 1
            
            return ConversionResult(
                success=False,
                pattern_type=pattern_type,
                pine_script="",
                confidence=0.0,
                conversion_time=0.0,
                metadata={"error": str(e)}
            )
    
    def convert_multiple_patterns(
        self,
        pattern_types: List[PatternType],
        combine_in_single_script: bool = True
    ) -> Dict[str, Any]:
        """
        Konvertiere mehrere Pattern zu Pine Script
        
        Args:
            pattern_types: Liste der zu konvertierenden Pattern
            combine_in_single_script: Ob alle Pattern in einem Script kombiniert werden sollen
            
        Returns:
            Dictionary mit Konvertierungs-Ergebnissen
        """
        try:
            start_time = datetime.now()
            
            individual_results = []
            
            # Konvertiere jedes Pattern einzeln
            for pattern_type in pattern_types:
                result = self.convert_pattern_to_pine_script(pattern_type)
                individual_results.append(result)
            
            if combine_in_single_script:
                # Kombiniere alle Pattern in einem Script
                combined_script = self._combine_patterns_in_script(individual_results)
            else:
                combined_script = None
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": all(r.success for r in individual_results),
                "individual_results": individual_results,
                "combined_script": combined_script,
                "total_patterns": len(pattern_types),
                "successful_conversions": len([r for r in individual_results if r.success]),
                "total_time": total_time,
                "avg_confidence": np.mean([r.confidence for r in individual_results if r.success])
            }
            
        except Exception as e:
            self.logger.error(f"Multiple pattern conversion error: {e}")
            return {
                "success": False,
                "error": str(e),
                "individual_results": [],
                "combined_script": None
            }
    
    def analyze_pattern_from_data(
        self,
        market_data: Dict[str, Any],
        detect_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Analysiere Pattern aus Marktdaten und generiere entsprechenden Pine Script
        
        Args:
            market_data: OHLCV-Daten
            detect_patterns: Ob Pattern automatisch erkannt werden sollen
            
        Returns:
            Analyse-Ergebnisse mit erkannten Pattern und Pine Script
        """
        try:
            # Enhanced Features extrahieren
            features = self.feature_extractor.extract_features_from_dict(market_data)
            
            detected_patterns = []
            
            if detect_patterns:
                # Pattern-Erkennung basierend auf Features
                detected_patterns = self._detect_patterns_from_features(features, market_data)
            
            # Pine Scripts für erkannte Pattern generieren
            pattern_scripts = []
            for pattern_info in detected_patterns:
                conversion_result = self.convert_pattern_to_pine_script(
                    pattern_info["pattern_type"],
                    pattern_info.get("parameters")
                )
                
                if conversion_result.success:
                    pattern_scripts.append({
                        "pattern_type": pattern_info["pattern_type"].value,
                        "confidence": pattern_info["confidence"],
                        "pine_script": conversion_result.pine_script,
                        "description": conversion_result.metadata.get("description", "")
                    })
            
            return {
                "success": True,
                "market_data_analyzed": True,
                "detected_patterns": detected_patterns,
                "pattern_scripts": pattern_scripts,
                "features_extracted": len(features),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Pattern analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "detected_patterns": [],
                "pattern_scripts": []
            }
    
    def _generate_pattern_pine_script(
        self,
        pattern_def: PatternDefinition,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generiere Pine Script für ein Pattern"""
        
        try:
            # Template für das Pattern laden
            template = self.pattern_templates.get(pattern_def.pattern_type, self.pattern_templates["default"])
            
            # Parameter vorbereiten
            parameters = {
                "pattern_name": pattern_def.pattern_type.value.replace("_", " ").title(),
                "pattern_type": pattern_def.pattern_type.value,
                "signal_type": pattern_def.signal.value,
                "candle_count": pattern_def.candle_count,
                "body_ratio_threshold": pattern_def.body_ratio_threshold,
                "shadow_ratio_threshold": pattern_def.shadow_ratio_threshold,
                "size_comparison_threshold": pattern_def.size_comparison_threshold,
                "reliability": pattern_def.reliability,
                "description": pattern_def.description,
                "conditions": pattern_def.conditions
            }
            
            # Custom Parameter hinzufügen
            if custom_parameters:
                parameters.update(custom_parameters)
            
            # Pine Script aus Template generieren
            pine_script = template.format(**parameters)
            
            return pine_script
            
        except Exception as e:
            self.logger.error(f"Pine script generation error: {e}")
            return f"// Error generating Pine script for {pattern_def.pattern_type.value}: {str(e)}"
    
    def _extract_pattern_conditions(self, pattern_def: PatternDefinition) -> List[str]:
        """Extrahiere Pattern-Bedingungen"""
        
        conditions = []
        
        # Basis-Bedingungen für alle Pattern
        conditions.append(f"// {pattern_def.description}")
        conditions.append(f"// Signal: {pattern_def.signal.value}")
        conditions.append(f"// Candles required: {pattern_def.candle_count}")
        conditions.append(f"// Reliability: {pattern_def.reliability:.1%}")
        
        # Pattern-spezifische Bedingungen
        conditions.extend(pattern_def.conditions)
        
        return conditions
    
    def _calculate_pattern_confidence(
        self,
        pattern_def: PatternDefinition,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> float:
        """Berechne Confidence für Pattern-Konvertierung"""
        
        try:
            # Basis-Confidence basierend auf Pattern-Reliability
            confidence = pattern_def.reliability
            
            # Anpassungen basierend auf Pattern-Komplexität
            if pattern_def.candle_count == 1:
                confidence *= 0.9  # Einzelne Kerzen sind weniger zuverlässig
            elif pattern_def.candle_count == 2:
                confidence *= 1.0  # Optimal
            elif pattern_def.candle_count >= 3:
                confidence *= 1.1  # Mehr Kerzen = mehr Kontext
            
            # Anpassungen basierend auf Signal-Typ
            if pattern_def.signal == PatternSignal.REVERSAL:
                confidence *= 1.1  # Reversal-Pattern sind wichtiger
            elif pattern_def.signal == PatternSignal.CONTINUATION:
                confidence *= 0.9  # Continuation-Pattern sind weniger eindeutig
            
            # Custom Parameter-Anpassungen
            if custom_parameters:
                if "confidence_boost" in custom_parameters:
                    confidence *= custom_parameters["confidence_boost"]
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5  # Fallback
    
    def _detect_patterns_from_features(
        self,
        features: np.ndarray,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Erkenne Pattern aus Enhanced Features"""
        
        detected_patterns = []
        
        try:
            # Feature-Namen für bessere Lesbarkeit
            feature_names = self.feature_extractor.get_feature_names()
            
            # OHLCV-Daten extrahieren
            open_price = market_data.get("open", 0)
            high_price = market_data.get("high", 0)
            low_price = market_data.get("low", 0)
            close_price = market_data.get("close", 0)
            
            # Basis-Berechnungen
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            body_ratio = body_size / total_range if total_range > 0 else 0
            
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Pattern-Erkennung basierend auf Candlestick-Eigenschaften
            
            # Doji Pattern
            if body_ratio < 0.1:  # Sehr kleiner Body
                detected_patterns.append({
                    "pattern_type": PatternType.DOJI,
                    "confidence": 0.8 - body_ratio * 2,  # Je kleiner der Body, desto höher die Confidence
                    "parameters": {"body_ratio": body_ratio}
                })
            
            # Hammer Pattern
            if (body_ratio < 0.3 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5):
                detected_patterns.append({
                    "pattern_type": PatternType.HAMMER,
                    "confidence": 0.7,
                    "parameters": {
                        "body_ratio": body_ratio,
                        "lower_shadow_ratio": lower_shadow / body_size if body_size > 0 else 0
                    }
                })
            
            # Shooting Star Pattern
            if (body_ratio < 0.3 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5):
                detected_patterns.append({
                    "pattern_type": PatternType.SHOOTING_STAR,
                    "confidence": 0.7,
                    "parameters": {
                        "body_ratio": body_ratio,
                        "upper_shadow_ratio": upper_shadow / body_size if body_size > 0 else 0
                    }
                })
            
            # Marubozu Pattern (sehr großer Body, kleine Schatten)
            if body_ratio > 0.8:
                if close_price > open_price:
                    detected_patterns.append({
                        "pattern_type": PatternType.MARUBOZU_BULLISH,
                        "confidence": 0.8,
                        "parameters": {"body_ratio": body_ratio}
                    })
                else:
                    detected_patterns.append({
                        "pattern_type": PatternType.MARUBOZU_BEARISH,
                        "confidence": 0.8,
                        "parameters": {"body_ratio": body_ratio}
                    })
            
            # Spinning Top Pattern
            if (0.1 < body_ratio < 0.3 and 
                upper_shadow > body_size and 
                lower_shadow > body_size):
                detected_patterns.append({
                    "pattern_type": PatternType.SPINNING_TOP,
                    "confidence": 0.6,
                    "parameters": {"body_ratio": body_ratio}
                })
            
            # Zusätzliche Pattern-Erkennung basierend auf Enhanced Features
            if len(features) >= 57:  # Vollständige Feature-Set
                # Verwende AI-Features für erweiterte Pattern-Erkennung
                pattern_confidence = features[26] if len(features) > 26 else 0.5  # pattern_confidence_max
                pattern_bullish = features[27] if len(features) > 27 else 0.0     # pattern_bullish
                pattern_bearish = features[28] if len(features) > 28 else 0.0     # pattern_bearish
                
                if pattern_confidence > 0.7:
                    if pattern_bullish > pattern_bearish:
                        # Bullish Pattern erkannt
                        detected_patterns.append({
                            "pattern_type": PatternType.ENGULFING_BULLISH,
                            "confidence": pattern_confidence,
                            "parameters": {"ai_confidence": pattern_confidence}
                        })
                    elif pattern_bearish > pattern_bullish:
                        # Bearish Pattern erkannt
                        detected_patterns.append({
                            "pattern_type": PatternType.ENGULFING_BEARISH,
                            "confidence": pattern_confidence,
                            "parameters": {"ai_confidence": pattern_confidence}
                        })
            
        except Exception as e:
            self.logger.error(f"Pattern detection error: {e}")
        
        return detected_patterns
    
    def _combine_patterns_in_script(self, results: List[ConversionResult]) -> str:
        """Kombiniere mehrere Pattern in einem Pine Script"""
        
        try:
            successful_results = [r for r in results if r.success]
            
            if not successful_results:
                return "// No successful pattern conversions to combine"
            
            # Header für kombiniertes Script
            combined_script = """
//@version=5
indicator("Multi-Pattern Detector", shorttitle="Patterns", overlay=true)

// === Parameters ===
show_pattern_names = input.bool(true, title="Show Pattern Names")
show_signals = input.bool(true, title="Show Buy/Sell Signals")
min_confidence = input.float(0.7, title="Minimum Confidence", minval=0.0, maxval=1.0)

// === Pattern Detection Functions ===
"""
            
            # Füge jedes Pattern hinzu
            pattern_conditions = []
            
            for i, result in enumerate(successful_results):
                pattern_name = result.pattern_type.value
                
                # Extrahiere Pattern-Logic aus dem generierten Script
                pattern_logic = self._extract_pattern_logic(result.pine_script)
                
                combined_script += f"\n// {pattern_name.replace('_', ' ').title()} Pattern\n"
                combined_script += pattern_logic + "\n"
                
                # Sammle Pattern-Bedingungen
                condition_var = f"{pattern_name}_detected"
                pattern_conditions.append(condition_var)
            
            # Kombinierte Signal-Logic
            combined_script += """
// === Combined Signals ===
any_bullish_pattern = """ + " or ".join([f"{p}_bullish" for p in [r.pattern_type.value for r in successful_results]]) + """
any_bearish_pattern = """ + " or ".join([f"{p}_bearish" for p in [r.pattern_type.value for r in successful_results]]) + """

// === Plots ===
plotshape(show_signals and any_bullish_pattern, style=shape.triangleup, location=location.belowbar, color=color.green, size=size.normal, title="Bullish Pattern")
plotshape(show_signals and any_bearish_pattern, style=shape.triangledown, location=location.abovebar, color=color.red, size=size.normal, title="Bearish Pattern")

// === Alerts ===
alertcondition(any_bullish_pattern, title="Bullish Pattern Alert", message="Bullish candlestick pattern detected")
alertcondition(any_bearish_pattern, title="Bearish Pattern Alert", message="Bearish candlestick pattern detected")
"""
            
            return combined_script
            
        except Exception as e:
            self.logger.error(f"Script combination error: {e}")
            return f"// Error combining scripts: {str(e)}"
    
    def _extract_pattern_logic(self, pine_script: str) -> str:
        """Extrahiere Pattern-Logic aus Pine Script"""
        
        try:
            lines = pine_script.split('\n')
            logic_lines = []
            
            in_logic_section = False
            
            for line in lines:
                # Suche nach Pattern-Logic-Sektion
                if "// Pattern Detection" in line or "// Calculations" in line:
                    in_logic_section = True
                    continue
                elif "// Plots" in line or "// Alerts" in line:
                    in_logic_section = False
                    continue
                
                if in_logic_section and line.strip():
                    logic_lines.append(line)
            
            return '\n'.join(logic_lines)
            
        except Exception as e:
            self.logger.error(f"Logic extraction error: {e}")
            return "// Error extracting pattern logic"    

    def _load_pattern_definitions(self) -> Dict[PatternType, PatternDefinition]:
        """Lade Pattern-Definitionen"""
        
        definitions = {}
        
        # Doji Pattern
        definitions[PatternType.DOJI] = PatternDefinition(
            pattern_type=PatternType.DOJI,
            signal=PatternSignal.NEUTRAL,
            candle_count=1,
            conditions=[
                "body_size = math.abs(close - open)",
                "total_range = high - low",
                "body_ratio = body_size / total_range",
                "doji_condition = body_ratio < 0.1"
            ],
            description="Doji indicates indecision in the market",
            reliability=0.6,
            body_ratio_threshold=0.1
        )
        
        # Hammer Pattern
        definitions[PatternType.HAMMER] = PatternDefinition(
            pattern_type=PatternType.HAMMER,
            signal=PatternSignal.BULLISH,
            candle_count=1,
            conditions=[
                "body_size = math.abs(close - open)",
                "lower_shadow = math.min(open, close) - low",
                "upper_shadow = high - math.max(open, close)",
                "hammer_condition = body_size < (high - low) * 0.3 and lower_shadow > body_size * 2 and upper_shadow < body_size * 0.5"
            ],
            description="Hammer pattern indicates potential bullish reversal",
            reliability=0.7,
            body_ratio_threshold=0.3,
            shadow_ratio_threshold=2.0
        )
        
        # Shooting Star Pattern
        definitions[PatternType.SHOOTING_STAR] = PatternDefinition(
            pattern_type=PatternType.SHOOTING_STAR,
            signal=PatternSignal.BEARISH,
            candle_count=1,
            conditions=[
                "body_size = math.abs(close - open)",
                "lower_shadow = math.min(open, close) - low",
                "upper_shadow = high - math.max(open, close)",
                "shooting_star_condition = body_size < (high - low) * 0.3 and upper_shadow > body_size * 2 and lower_shadow < body_size * 0.5"
            ],
            description="Shooting Star pattern indicates potential bearish reversal",
            reliability=0.7,
            body_ratio_threshold=0.3,
            shadow_ratio_threshold=2.0
        )
        
        # Engulfing Bullish Pattern
        definitions[PatternType.ENGULFING_BULLISH] = PatternDefinition(
            pattern_type=PatternType.ENGULFING_BULLISH,
            signal=PatternSignal.BULLISH,
            candle_count=2,
            conditions=[
                "prev_body_size = math.abs(close[1] - open[1])",
                "curr_body_size = math.abs(close - open)",
                "engulfing_bullish = close[1] < open[1] and close > open and open < close[1] and close > open[1] and curr_body_size > prev_body_size"
            ],
            description="Bullish Engulfing pattern indicates strong bullish reversal",
            reliability=0.8,
            size_comparison_threshold=1.2
        )
        
        # Engulfing Bearish Pattern
        definitions[PatternType.ENGULFING_BEARISH] = PatternDefinition(
            pattern_type=PatternType.ENGULFING_BEARISH,
            signal=PatternSignal.BEARISH,
            candle_count=2,
            conditions=[
                "prev_body_size = math.abs(close[1] - open[1])",
                "curr_body_size = math.abs(close - open)",
                "engulfing_bearish = close[1] > open[1] and close < open and open > close[1] and close < open[1] and curr_body_size > prev_body_size"
            ],
            description="Bearish Engulfing pattern indicates strong bearish reversal",
            reliability=0.8,
            size_comparison_threshold=1.2
        )
        
        # Morning Star Pattern
        definitions[PatternType.MORNING_STAR] = PatternDefinition(
            pattern_type=PatternType.MORNING_STAR,
            signal=PatternSignal.BULLISH,
            candle_count=3,
            conditions=[
                "first_bearish = close[2] < open[2]",
                "middle_small = math.abs(close[1] - open[1]) < math.abs(close[2] - open[2]) * 0.3",
                "third_bullish = close > open and close > (open[2] + close[2]) / 2",
                "morning_star = first_bearish and middle_small and third_bullish"
            ],
            description="Morning Star pattern indicates bullish reversal after downtrend",
            reliability=0.8,
            body_ratio_threshold=0.3
        )
        
        # Evening Star Pattern
        definitions[PatternType.EVENING_STAR] = PatternDefinition(
            pattern_type=PatternType.EVENING_STAR,
            signal=PatternSignal.BEARISH,
            candle_count=3,
            conditions=[
                "first_bullish = close[2] > open[2]",
                "middle_small = math.abs(close[1] - open[1]) < math.abs(close[2] - open[2]) * 0.3",
                "third_bearish = close < open and close < (open[2] + close[2]) / 2",
                "evening_star = first_bullish and middle_small and third_bearish"
            ],
            description="Evening Star pattern indicates bearish reversal after uptrend",
            reliability=0.8,
            body_ratio_threshold=0.3
        )
        
        # Weitere Pattern können hier hinzugefügt werden...
        
        return definitions
    
    def _load_pattern_templates(self) -> Dict[PatternType, str]:
        """Lade Pine Script Templates für Pattern"""
        
        templates = {}
        
        # Default Template
        default_template = """
//@version=5
indicator("{pattern_name} Detector", shorttitle="{pattern_type}", overlay=true)

// === Parameters ===
show_signals = input.bool(true, title="Show Signals")
show_pattern_names = input.bool(true, title="Show Pattern Names")
min_confidence = input.float({reliability}, title="Minimum Confidence", minval=0.0, maxval=1.0)

// === Pattern Detection ===
{conditions}

// Pattern detected with confidence
pattern_detected = {pattern_type}_condition
pattern_confidence = {reliability}

// Signal based on pattern type
{signal_type}_signal = pattern_detected and pattern_confidence >= min_confidence

// === Plots ===
plotshape(show_signals and {signal_type}_signal, 
         style=shape.triangleup if "{signal_type}" == "bullish" else shape.triangledown, 
         location=location.belowbar if "{signal_type}" == "bullish" else location.abovebar, 
         color=color.green if "{signal_type}" == "bullish" else color.red, 
         size=size.small, 
         title="{pattern_name} Signal")

// Pattern name label
if show_pattern_names and pattern_detected
    label.new(bar_index, high + (high - low) * 0.1, 
             text="{pattern_name}\\nConf: " + str.tostring(pattern_confidence, "#.##"), 
             style=label.style_label_down, 
             color=color.blue, 
             textcolor=color.white, 
             size=size.small)

// === Alerts ===
alertcondition({signal_type}_signal, title="{pattern_name} Alert", 
              message="{pattern_name} pattern detected with confidence " + str.tostring(pattern_confidence, "#.##"))

// === Background Highlight ===
bgcolor(pattern_detected ? color.new(color.yellow, 90) : na, title="Pattern Background")
"""
        
        # Spezifische Templates für komplexere Pattern
        
        # Doji Template
        templates[PatternType.DOJI] = """
//@version=5
indicator("Doji Pattern Detector", shorttitle="DOJI", overlay=true)

// === Parameters ===
show_signals = input.bool(true, title="Show Doji Signals")
body_threshold = input.float(0.1, title="Body Size Threshold", minval=0.01, maxval=0.5)
show_pattern_names = input.bool(true, title="Show Pattern Names")

// === Doji Detection ===
body_size = math.abs(close - open)
total_range = high - low
body_ratio = total_range > 0 ? body_size / total_range : 0

// Doji condition
doji_condition = body_ratio < body_threshold

// Doji types
dragonfly_doji = doji_condition and (low == math.min(open, close)) and (high - math.max(open, close)) > body_size * 2
gravestone_doji = doji_condition and (high == math.max(open, close)) and (math.min(open, close) - low) > body_size * 2
long_legged_doji = doji_condition and (high - math.max(open, close)) > body_size and (math.min(open, close) - low) > body_size

// === Plots ===
plotshape(show_signals and doji_condition, style=shape.diamond, location=location.absolute, 
         color=color.yellow, size=size.small, title="Doji")

plotshape(show_signals and dragonfly_doji, style=shape.triangleup, location=location.belowbar, 
         color=color.green, size=size.tiny, title="Dragonfly Doji")

plotshape(show_signals and gravestone_doji, style=shape.triangledown, location=location.abovebar, 
         color=color.red, size=size.tiny, title="Gravestone Doji")

// Pattern names
if show_pattern_names and doji_condition
    doji_type = dragonfly_doji ? "Dragonfly" : gravestone_doji ? "Gravestone" : long_legged_doji ? "Long-Legged" : "Standard"
    label.new(bar_index, high + (high - low) * 0.1, 
             text=doji_type + " Doji\\nRatio: " + str.tostring(body_ratio, "#.###"), 
             style=label.style_label_down, color=color.yellow, textcolor=color.black, size=size.small)

// === Alerts ===
alertcondition(doji_condition, title="Doji Pattern", message="Doji pattern detected - market indecision")
alertcondition(dragonfly_doji, title="Dragonfly Doji", message="Dragonfly Doji - potential bullish reversal")
alertcondition(gravestone_doji, title="Gravestone Doji", message="Gravestone Doji - potential bearish reversal")
"""
        
        # Engulfing Template
        templates[PatternType.ENGULFING_BULLISH] = templates[PatternType.ENGULFING_BEARISH] = """
//@version=5
indicator("Engulfing Pattern Detector", shorttitle="ENGULF", overlay=true)

// === Parameters ===
show_signals = input.bool(true, title="Show Engulfing Signals")
size_threshold = input.float(1.2, title="Size Comparison Threshold", minval=1.0, maxval=3.0)
show_pattern_names = input.bool(true, title="Show Pattern Names")

// === Engulfing Detection ===
prev_body_size = math.abs(close[1] - open[1])
curr_body_size = math.abs(close - open)
size_ratio = prev_body_size > 0 ? curr_body_size / prev_body_size : 0

// Bullish Engulfing
bullish_engulfing = close[1] < open[1] and close > open and 
                   open < close[1] and close > open[1] and 
                   size_ratio >= size_threshold

// Bearish Engulfing  
bearish_engulfing = close[1] > open[1] and close < open and 
                   open > close[1] and close < open[1] and 
                   size_ratio >= size_threshold

// === Plots ===
plotshape(show_signals and bullish_engulfing, style=shape.triangleup, location=location.belowbar, 
         color=color.green, size=size.normal, title="Bullish Engulfing")

plotshape(show_signals and bearish_engulfing, style=shape.triangledown, location=location.abovebar, 
         color=color.red, size=size.normal, title="Bearish Engulfing")

// Pattern names
if show_pattern_names and (bullish_engulfing or bearish_engulfing)
    pattern_name = bullish_engulfing ? "Bullish Engulfing" : "Bearish Engulfing"
    label.new(bar_index, bullish_engulfing ? low - (high - low) * 0.1 : high + (high - low) * 0.1, 
             text=pattern_name + "\\nRatio: " + str.tostring(size_ratio, "#.##"), 
             style=bullish_engulfing ? label.style_label_up : label.style_label_down, 
             color=bullish_engulfing ? color.green : color.red, 
             textcolor=color.white, size=size.small)

// === Alerts ===
alertcondition(bullish_engulfing, title="Bullish Engulfing", message="Bullish Engulfing pattern - strong reversal signal")
alertcondition(bearish_engulfing, title="Bearish Engulfing", message="Bearish Engulfing pattern - strong reversal signal")

// === Background ===
bgcolor(bullish_engulfing ? color.new(color.green, 95) : bearish_engulfing ? color.new(color.red, 95) : na)
"""
        
        templates["default"] = default_template
        
        return templates
    
    def get_supported_patterns(self) -> List[Dict[str, Any]]:
        """Erhalte Liste unterstützter Pattern"""
        
        patterns = []
        
        for pattern_type, definition in self.pattern_definitions.items():
            patterns.append({
                "pattern_type": pattern_type.value,
                "name": pattern_type.value.replace("_", " ").title(),
                "signal": definition.signal.value,
                "candle_count": definition.candle_count,
                "reliability": definition.reliability,
                "description": definition.description
            })
        
        return patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Converter-Statistiken"""
        
        return {
            **self.stats,
            "supported_patterns": len(self.pattern_definitions),
            "available_templates": len(self.pattern_templates),
            "success_rate": (
                self.stats["successful_conversions"] / self.stats["patterns_converted"] 
                if self.stats["patterns_converted"] > 0 else 0.0
            )
        }


# Factory Function
def create_visual_pattern_converter(config: Optional[Dict] = None) -> VisualPatternConverter:
    """Factory Function für Visual Pattern Converter"""
    return VisualPatternConverter(config=config)


# Demo/Test Function
def demo_visual_pattern_converter():
    """Demo für Visual Pattern Converter"""
    
    print("🧪 Testing Visual Pattern Converter...")
    
    # Converter erstellen
    converter = create_visual_pattern_converter()
    
    # Unterstützte Pattern anzeigen
    supported_patterns = converter.get_supported_patterns()
    print(f"\n📋 Supported Patterns ({len(supported_patterns)}):")
    for pattern in supported_patterns[:5]:  # Zeige nur erste 5
        print(f"   - {pattern['name']}: {pattern['signal']} signal, {pattern['candle_count']} candle(s)")
        print(f"     Reliability: {pattern['reliability']:.1%}, Description: {pattern['description']}")
    
    # Test einzelne Pattern-Konvertierung
    test_patterns = [PatternType.DOJI, PatternType.HAMMER, PatternType.ENGULFING_BULLISH]
    
    print(f"\n🔄 Converting individual patterns...")
    
    individual_results = []
    for pattern_type in test_patterns:
        print(f"\n--- Converting {pattern_type.value} ---")
        
        result = converter.convert_pattern_to_pine_script(pattern_type)
        individual_results.append(result)
        
        if result.success:
            print(f"✅ Conversion successful:")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Conversion Time: {result.conversion_time:.3f}s")
            print(f"   Signal Type: {result.metadata['signal']}")
            print(f"   Candle Count: {result.metadata['candle_count']}")
            print(f"   Script Length: {len(result.pine_script)} chars")
            
            # Zeige ersten Teil des Scripts
            script_preview = result.pine_script[:300] + "..." if len(result.pine_script) > 300 else result.pine_script
            print(f"   Script Preview: {script_preview}")
        else:
            print(f"❌ Conversion failed: {result.metadata.get('error', 'Unknown error')}")
    
    # Test Multi-Pattern-Konvertierung
    print(f"\n🔄 Converting multiple patterns...")
    
    multi_result = converter.convert_multiple_patterns(test_patterns, combine_in_single_script=True)
    
    if multi_result["success"]:
        print(f"✅ Multi-pattern conversion successful:")
        print(f"   Total Patterns: {multi_result['total_patterns']}")
        print(f"   Successful Conversions: {multi_result['successful_conversions']}")
        print(f"   Average Confidence: {multi_result['avg_confidence']:.3f}")
        print(f"   Total Time: {multi_result['total_time']:.3f}s")
        
        if multi_result["combined_script"]:
            combined_length = len(multi_result["combined_script"])
            print(f"   Combined Script Length: {combined_length} chars")
    else:
        print(f"❌ Multi-pattern conversion failed: {multi_result.get('error', 'Unknown error')}")
    
    # Test Pattern-Analyse aus Marktdaten
    print(f"\n📊 Analyzing patterns from market data...")
    
    # Mock Market Data
    mock_market_data = {
        "open": 1.1000,
        "high": 1.1005,
        "low": 1.0995,
        "close": 1.0996,  # Kleiner Body für Doji
        "volume": 1500,
        "timestamp": datetime.now().isoformat()
    }
    
    analysis_result = converter.analyze_pattern_from_data(mock_market_data, detect_patterns=True)
    
    if analysis_result["success"]:
        print(f"✅ Pattern analysis successful:")
        print(f"   Features Extracted: {analysis_result['features_extracted']}")
        print(f"   Patterns Detected: {len(analysis_result['detected_patterns'])}")
        print(f"   Pattern Scripts Generated: {len(analysis_result['pattern_scripts'])}")
        
        for pattern_info in analysis_result['detected_patterns']:
            print(f"   - {pattern_info['pattern_type'].value}: confidence {pattern_info['confidence']:.3f}")
    else:
        print(f"❌ Pattern analysis failed: {analysis_result.get('error', 'Unknown error')}")
    
    # Converter-Statistiken
    stats = converter.get_statistics()
    print(f"\n📈 Converter Statistics:")
    print(f"   Patterns Converted: {stats['patterns_converted']}")
    print(f"   Successful Conversions: {stats['successful_conversions']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Average Confidence: {stats['avg_confidence']:.3f}")
    print(f"   Average Conversion Time: {stats['avg_conversion_time']:.3f}s")
    print(f"   Supported Patterns: {stats['supported_patterns']}")
    
    print(f"\n🔧 Patterns by Type:")
    for pattern_type, count in stats['patterns_by_type'].items():
        if count > 0:
            print(f"   {pattern_type}: {count}")


if __name__ == "__main__":
    demo_visual_pattern_converter()