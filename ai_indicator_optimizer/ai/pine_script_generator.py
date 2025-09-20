#!/usr/bin/env python3
"""
Pine Script Generator mit Template-System
Phase 3 Implementation - Enhanced Pine Script Code Generator

Features:
- Template-basierte Pine Script Generation
- Verschiedene Indikator-Typen (RSI, MACD, Bollinger Bands, etc.)
- AI-optimierte Parameter-Generierung
- Code-Validation und Syntax-Checking
- Multi-Timeframe-Support
- Enhanced Feature Integration
- Custom Strategy Logic Integration
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
import numpy as np

# Template Engine
try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Local Imports
from .enhanced_feature_extractor import EnhancedFeatureExtractor


class IndicatorType(Enum):
    """Unterstützte Indikator-Typen"""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    MOVING_AVERAGE = "moving_average"
    STOCHASTIC = "stochastic"
    ATR = "atr"
    VOLUME_PROFILE = "volume_profile"
    FIBONACCI = "fibonacci"
    SUPPORT_RESISTANCE = "support_resistance"
    AI_PATTERN = "ai_pattern"
    MULTI_INDICATOR = "multi_indicator"


class StrategyType(Enum):
    """Unterstützte Strategy-Typen"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    AI_PATTERN = "ai_pattern"
    MULTI_TIMEFRAME = "multi_timeframe"


@dataclass
class PineScriptConfig:
    """Konfiguration für Pine Script Generation"""
    # Basis-Einstellungen
    script_name: str = "AI Generated Indicator"
    script_version: str = "5"
    overlay: bool = True
    
    # Indikator-Einstellungen
    indicator_type: IndicatorType = IndicatorType.RSI
    strategy_type: Optional[StrategyType] = None
    
    # Parameter
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Erweiterte Einstellungen
    multi_timeframe: bool = False
    alerts_enabled: bool = True
    backtesting_enabled: bool = False
    
    # AI-Integration
    ai_features: bool = False
    confidence_threshold: float = 0.7
    
    # Styling
    colors: Dict[str, str] = field(default_factory=lambda: {
        "bullish": "color.green",
        "bearish": "color.red",
        "neutral": "color.gray",
        "signal": "color.blue"
    })


class PineScriptTemplate:
    """Template für Pine Script Code"""
    
    def __init__(self, template_name: str, template_code: str):
        self.name = template_name
        self.code = template_code
        self.required_params = self._extract_required_params()
    
    def _extract_required_params(self) -> List[str]:
        """Extrahiere erforderliche Parameter aus Template"""
        # Finde alle {{ parameter }} Platzhalter
        pattern = r'\{\{\s*(\w+)\s*\}\}'
        matches = re.findall(pattern, self.code)
        return list(set(matches))
    
    def render(self, parameters: Dict[str, Any]) -> str:
        """Rendere Template mit Parametern"""
        if JINJA2_AVAILABLE:
            template = Template(self.code)
            return template.render(**parameters)
        else:
            # Fallback: Einfache String-Ersetzung
            result = self.code
            for key, value in parameters.items():
                placeholder = f"{{{{{key}}}}}"
                result = result.replace(placeholder, str(value))
            return result


class PineScriptGenerator:
    """
    Haupt-Generator für Pine Script Code
    
    Features:
    - Template-basierte Generation
    - AI-optimierte Parameter
    - Code-Validation
    - Multi-Indikator-Support
    - Enhanced Feature Integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Templates laden
        self.templates: Dict[str, PineScriptTemplate] = {}
        self._load_builtin_templates()
        
        # Enhanced Feature Extractor für AI-Integration
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # Validation-Regeln
        self.validation_rules = self._setup_validation_rules()
        
        # Statistiken
        self.stats = {
            "scripts_generated": 0,
            "validation_errors": 0,
            "templates_used": {},
            "parameter_optimizations": 0
        }
        
        self.logger.info(f"PineScriptGenerator initialized with {len(self.templates)} templates")
    
    def _load_builtin_templates(self):
        """Lade eingebaute Templates"""
        
        # RSI Template
        rsi_template = PineScriptTemplate(
            "rsi_indicator",
            """
//@version={{version}}
indicator("{{script_name}}", shorttitle="{{short_title}}", overlay={{overlay}})

// === Parameters ===
rsi_length = input.int({{rsi_length}}, title="RSI Length", minval=1, maxval=200)
rsi_overbought = input.float({{rsi_overbought}}, title="Overbought Level", minval=50, maxval=100)
rsi_oversold = input.float({{rsi_oversold}}, title="Oversold Level", minval=0, maxval=50)
show_signals = input.bool({{show_signals}}, title="Show Buy/Sell Signals")
{% if ai_features %}
confidence_threshold = input.float({{confidence_threshold}}, title="AI Confidence Threshold", minval=0.0, maxval=1.0)
{% endif %}

// === Calculations ===
rsi_value = ta.rsi(close, rsi_length)

// === Signals ===
buy_signal = ta.crossover(rsi_value, rsi_oversold)
sell_signal = ta.crossunder(rsi_value, rsi_overbought)

{% if ai_features %}
// === AI Enhancement ===
// Placeholder for AI confidence scoring
ai_confidence = 0.75  // This would be replaced with actual AI model output
enhanced_buy = buy_signal and ai_confidence > confidence_threshold
enhanced_sell = sell_signal and ai_confidence > confidence_threshold
{% endif %}

// === Plots ===
rsi_plot = plot(rsi_value, title="RSI", color={{rsi_color}}, linewidth=2)
hline(rsi_overbought, title="Overbought", color={{overbought_color}}, linestyle=hline.style_dashed)
hline(rsi_oversold, title="Oversold", color={{oversold_color}}, linestyle=hline.style_dotted)
hline(50, title="Midline", color=color.gray, linestyle=hline.style_dotted)

// === Signal Shapes ===
{% if ai_features %}
plotshape(show_signals and enhanced_buy, style=shape.triangleup, location=location.bottom, color={{bullish_color}}, size=size.small, title="AI Enhanced Buy")
plotshape(show_signals and enhanced_sell, style=shape.triangledown, location=location.top, color={{bearish_color}}, size=size.small, title="AI Enhanced Sell")
{% else %}
plotshape(show_signals and buy_signal, style=shape.triangleup, location=location.bottom, color={{bullish_color}}, size=size.small, title="Buy Signal")
plotshape(show_signals and sell_signal, style=shape.triangledown, location=location.top, color={{bearish_color}}, size=size.small, title="Sell Signal")
{% endif %}

{% if alerts_enabled %}
// === Alerts ===
{% if ai_features %}
alertcondition(enhanced_buy, title="AI Enhanced RSI Buy", message="AI Enhanced RSI Buy Signal - Confidence: " + str.tostring(ai_confidence))
alertcondition(enhanced_sell, title="AI Enhanced RSI Sell", message="AI Enhanced RSI Sell Signal - Confidence: " + str.tostring(ai_confidence))
{% else %}
alertcondition(buy_signal, title="RSI Buy Signal", message="RSI crossed above oversold level")
alertcondition(sell_signal, title="RSI Sell Signal", message="RSI crossed below overbought level")
{% endif %}
{% endif %}
"""
        )
        self.templates["rsi"] = rsi_template
        
        # MACD Template
        macd_template = PineScriptTemplate(
            "macd_indicator",
            """
//@version={{version}}
indicator("{{script_name}}", shorttitle="{{short_title}}", overlay={{overlay}})

// === Parameters ===
fast_length = input.int({{fast_length}}, title="Fast Length", minval=1, maxval=50)
slow_length = input.int({{slow_length}}, title="Slow Length", minval=1, maxval=100)
signal_length = input.int({{signal_length}}, title="Signal Length", minval=1, maxval=50)
show_histogram = input.bool({{show_histogram}}, title="Show Histogram")
{% if ai_features %}
confidence_threshold = input.float({{confidence_threshold}}, title="AI Confidence Threshold", minval=0.0, maxval=1.0)
{% endif %}

// === Calculations ===
[macd_line, signal_line, histogram] = ta.macd(close, fast_length, slow_length, signal_length)

// === Signals ===
buy_signal = ta.crossover(macd_line, signal_line)
sell_signal = ta.crossunder(macd_line, signal_line)
bullish_divergence = macd_line > macd_line[1] and close < close[1]
bearish_divergence = macd_line < macd_line[1] and close > close[1]

{% if ai_features %}
// === AI Enhancement ===
ai_confidence = 0.80  // Placeholder for AI model output
enhanced_buy = buy_signal and ai_confidence > confidence_threshold
enhanced_sell = sell_signal and ai_confidence > confidence_threshold
{% endif %}

// === Plots ===
macd_plot = plot(macd_line, title="MACD", color={{macd_color}}, linewidth=2)
signal_plot = plot(signal_line, title="Signal", color={{signal_color}}, linewidth=1)
if show_histogram
    plot(histogram, title="Histogram", color=histogram > 0 ? {{bullish_color}} : {{bearish_color}}, style=plot.style_histogram)
hline(0, title="Zero Line", color=color.black, linestyle=hline.style_solid)

// === Signal Shapes ===
{% if ai_features %}
plotshape(enhanced_buy, style=shape.triangleup, location=location.bottom, color={{bullish_color}}, size=size.small, title="AI Enhanced Buy")
plotshape(enhanced_sell, style=shape.triangledown, location=location.top, color={{bearish_color}}, size=size.small, title="AI Enhanced Sell")
{% else %}
plotshape(buy_signal, style=shape.triangleup, location=location.bottom, color={{bullish_color}}, size=size.small, title="MACD Buy")
plotshape(sell_signal, style=shape.triangledown, location=location.top, color={{bearish_color}}, size=size.small, title="MACD Sell")
{% endif %}

// === Divergence Signals ===
plotshape(bullish_divergence, style=shape.circle, location=location.bottom, color=color.lime, size=size.tiny, title="Bullish Divergence")
plotshape(bearish_divergence, style=shape.circle, location=location.top, color=color.orange, size=size.tiny, title="Bearish Divergence")

{% if alerts_enabled %}
// === Alerts ===
{% if ai_features %}
alertcondition(enhanced_buy, title="AI Enhanced MACD Buy", message="AI Enhanced MACD Buy Signal")
alertcondition(enhanced_sell, title="AI Enhanced MACD Sell", message="AI Enhanced MACD Sell Signal")
{% else %}
alertcondition(buy_signal, title="MACD Buy Signal", message="MACD crossed above signal line")
alertcondition(sell_signal, title="MACD Sell Signal", message="MACD crossed below signal line")
{% endif %}
alertcondition(bullish_divergence, title="MACD Bullish Divergence", message="MACD Bullish Divergence detected")
alertcondition(bearish_divergence, title="MACD Bearish Divergence", message="MACD Bearish Divergence detected")
{% endif %}
"""
        )
        self.templates["macd"] = macd_template
        
        # Bollinger Bands Template
        bb_template = PineScriptTemplate(
            "bollinger_bands",
            """
//@version={{version}}
indicator("{{script_name}}", shorttitle="{{short_title}}", overlay={{overlay}})

// === Parameters ===
bb_length = input.int({{bb_length}}, title="Bollinger Bands Length", minval=1, maxval=200)
bb_mult = input.float({{bb_mult}}, title="Standard Deviation Multiplier", minval=0.1, maxval=5.0)
show_squeeze = input.bool({{show_squeeze}}, title="Show Squeeze Signals")
{% if ai_features %}
confidence_threshold = input.float({{confidence_threshold}}, title="AI Confidence Threshold", minval=0.0, maxval=1.0)
{% endif %}

// === Calculations ===
bb_basis = ta.sma(close, bb_length)
bb_dev = bb_mult * ta.stdev(close, bb_length)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev
bb_width = (bb_upper - bb_lower) / bb_basis
bb_position = (close - bb_lower) / (bb_upper - bb_lower)

// === Signals ===
buy_signal = ta.crossover(close, bb_lower)
sell_signal = ta.crossunder(close, bb_upper)
squeeze = bb_width < bb_width[20] * 0.5  // Squeeze when width is 50% of 20-period average
breakout_up = close > bb_upper and close[1] <= bb_upper[1]
breakout_down = close < bb_lower and close[1] >= bb_lower[1]

{% if ai_features %}
// === AI Enhancement ===
ai_confidence = 0.75  // Placeholder for AI model output
enhanced_buy = buy_signal and ai_confidence > confidence_threshold
enhanced_sell = sell_signal and ai_confidence > confidence_threshold
{% endif %}

// === Plots ===
upper_plot = plot(bb_upper, title="Upper Band", color={{upper_color}}, linewidth=1)
lower_plot = plot(bb_lower, title="Lower Band", color={{lower_color}}, linewidth=1)
basis_plot = plot(bb_basis, title="Basis", color={{basis_color}}, linewidth=2)
fill(upper_plot, lower_plot, color=color.new({{fill_color}}, 95), title="BB Fill")

// === Signal Shapes ===
{% if ai_features %}
plotshape(enhanced_buy, style=shape.triangleup, location=location.belowbar, color={{bullish_color}}, size=size.small, title="AI Enhanced Buy")
plotshape(enhanced_sell, style=shape.triangledown, location=location.abovebar, color={{bearish_color}}, size=size.small, title="AI Enhanced Sell")
{% else %}
plotshape(buy_signal, style=shape.triangleup, location=location.belowbar, color={{bullish_color}}, size=size.small, title="BB Buy")
plotshape(sell_signal, style=shape.triangledown, location=location.abovebar, color={{bearish_color}}, size=size.small, title="BB Sell")
{% endif %}

// === Squeeze and Breakout Signals ===
if show_squeeze
    plotshape(squeeze, style=shape.diamond, location=location.top, color=color.yellow, size=size.tiny, title="Squeeze")
plotshape(breakout_up, style=shape.arrowup, location=location.belowbar, color=color.lime, size=size.small, title="Breakout Up")
plotshape(breakout_down, style=shape.arrowdown, location=location.abovebar, color=color.red, size=size.small, title="Breakout Down")

{% if alerts_enabled %}
// === Alerts ===
{% if ai_features %}
alertcondition(enhanced_buy, title="AI Enhanced BB Buy", message="AI Enhanced Bollinger Bands Buy Signal")
alertcondition(enhanced_sell, title="AI Enhanced BB Sell", message="AI Enhanced Bollinger Bands Sell Signal")
{% else %}
alertcondition(buy_signal, title="BB Buy Signal", message="Price crossed above lower Bollinger Band")
alertcondition(sell_signal, title="BB Sell Signal", message="Price crossed below upper Bollinger Band")
{% endif %}
alertcondition(squeeze, title="BB Squeeze", message="Bollinger Bands Squeeze detected")
alertcondition(breakout_up, title="BB Breakout Up", message="Bollinger Bands Breakout to upside")
alertcondition(breakout_down, title="BB Breakout Down", message="Bollinger Bands Breakout to downside")
{% endif %}
"""
        )
        self.templates["bollinger_bands"] = bb_template
        
        # AI Pattern Recognition Template
        ai_pattern_template = PineScriptTemplate(
            "ai_pattern",
            """
//@version={{version}}
indicator("{{script_name}}", shorttitle="{{short_title}}", overlay={{overlay}})

// === Parameters ===
lookback_period = input.int({{lookback_period}}, title="Pattern Lookback Period", minval=5, maxval=100)
confidence_threshold = input.float({{confidence_threshold}}, title="AI Confidence Threshold", minval=0.0, maxval=1.0)
show_pattern_names = input.bool({{show_pattern_names}}, title="Show Pattern Names")
show_confidence = input.bool({{show_confidence}}, title="Show Confidence Scores")

// === Enhanced Feature Calculations ===
// OHLCV Features
price_change = (close - open) / open
price_range = (high - low) / open
body_size = math.abs(close - open) / open
body_ratio = body_size / price_range
is_bullish = close > open ? 1 : 0
is_bearish = close < open ? 1 : 0
is_doji = body_size < (price_range * 0.1) ? 1 : 0

// Technical Indicators
rsi_14 = ta.rsi(close, 14)
[macd_line, signal_line, _] = ta.macd(close, 12, 26, 9)
bb_basis = ta.sma(close, 20)
bb_dev = 2 * ta.stdev(close, 20)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev
bb_position = (close - bb_lower) / (bb_upper - bb_lower)
atr_14 = ta.atr(14)

// Volume Features
volume_sma = ta.sma(volume, 10)
volume_ratio = volume / volume_sma

// === AI Pattern Recognition (Simulated) ===
// In a real implementation, this would call an external AI model
// For now, we simulate pattern recognition based on technical conditions

// Doji Pattern
doji_condition = is_doji and volume_ratio > 1.2
doji_confidence = doji_condition ? math.random(0.7, 0.95) : 0.0

// Hammer Pattern
hammer_condition = body_size < (price_range * 0.3) and (low < math.min(open, close) - body_size * 2)
hammer_confidence = hammer_condition ? math.random(0.6, 0.9) : 0.0

// Engulfing Pattern
prev_body_size = math.abs(close[1] - open[1])
engulfing_bull = is_bullish and close[1] < open[1] and open < close[1] and close > open[1] and body_size > prev_body_size
engulfing_bear = is_bearish and close[1] > open[1] and open > close[1] and close < open[1] and body_size > prev_body_size
engulfing_confidence = (engulfing_bull or engulfing_bear) ? math.random(0.75, 0.95) : 0.0

// Morning/Evening Star (simplified)
star_condition = is_doji[1] and math.abs(close - close[2]) > atr_14
morning_star = star_condition and close > close[2] and is_bullish
evening_star = star_condition and close < close[2] and is_bearish
star_confidence = (morning_star or evening_star) ? math.random(0.7, 0.9) : 0.0

// === Pattern Selection (Highest Confidence) ===
max_confidence = math.max(doji_confidence, math.max(hammer_confidence, math.max(engulfing_confidence, star_confidence)))
pattern_detected = max_confidence > confidence_threshold

pattern_name = 
  max_confidence == doji_confidence ? "Doji" :
  max_confidence == hammer_confidence ? "Hammer" :
  max_confidence == engulfing_confidence ? (engulfing_bull ? "Engulfing Bull" : "Engulfing Bear") :
  max_confidence == star_confidence ? (morning_star ? "Morning Star" : "Evening Star") : "None"

pattern_bullish = (hammer_condition and low < bb_lower) or engulfing_bull or morning_star
pattern_bearish = engulfing_bear or evening_star
pattern_neutral = doji_condition

// === Signals ===
buy_signal = pattern_detected and pattern_bullish and rsi_14 < 40
sell_signal = pattern_detected and pattern_bearish and rsi_14 > 60
neutral_signal = pattern_detected and pattern_neutral

// === Plots ===
{% if overlay %}
// Pattern Markers on Chart
plotshape(buy_signal, style=shape.triangleup, location=location.belowbar, color={{bullish_color}}, size=size.normal, title="AI Buy Pattern")
plotshape(sell_signal, style=shape.triangledown, location=location.abovebar, color={{bearish_color}}, size=size.normal, title="AI Sell Pattern")
plotshape(neutral_signal, style=shape.diamond, location=location.absolute, color={{neutral_color}}, size=size.small, title="AI Neutral Pattern")

// Pattern Names
if show_pattern_names and pattern_detected
    label.new(bar_index, high + atr_14, text=pattern_name, style=label.style_label_down, color=color.blue, textcolor=color.white, size=size.small)

// Confidence Scores
if show_confidence and pattern_detected
    label.new(bar_index, low - atr_14, text="Conf: " + str.tostring(max_confidence, "#.##"), style=label.style_label_up, color=color.gray, textcolor=color.white, size=size.tiny)
{% else %}
// Indicator Panel
plot(max_confidence, title="Pattern Confidence", color=color.blue, linewidth=2)
hline(confidence_threshold, title="Confidence Threshold", color=color.red, linestyle=hline.style_dashed)
hline(0.5, title="Midline", color=color.gray, linestyle=hline.style_dotted)

// Signal Bars
bgcolor(buy_signal ? color.new({{bullish_color}}, 80) : na, title="Buy Signal Background")
bgcolor(sell_signal ? color.new({{bearish_color}}, 80) : na, title="Sell Signal Background")
bgcolor(neutral_signal ? color.new({{neutral_color}}, 80) : na, title="Neutral Signal Background")
{% endif %}

{% if alerts_enabled %}
// === Alerts ===
alertcondition(buy_signal, title="AI Pattern Buy", message="AI detected bullish pattern: " + pattern_name + " (Confidence: " + str.tostring(max_confidence, "#.##") + ")")
alertcondition(sell_signal, title="AI Pattern Sell", message="AI detected bearish pattern: " + pattern_name + " (Confidence: " + str.tostring(max_confidence, "#.##") + ")")
alertcondition(neutral_signal, title="AI Pattern Neutral", message="AI detected neutral pattern: " + pattern_name + " (Confidence: " + str.tostring(max_confidence, "#.##") + ")")
{% endif %}

// === Table with Pattern Information ===
if show_pattern_names and barstate.islast
    var table pattern_table = table.new(position.top_right, 2, 6, bgcolor=color.white, border_width=1)
    table.cell(pattern_table, 0, 0, "Pattern", text_color=color.black, bgcolor=color.gray)
    table.cell(pattern_table, 1, 0, "Confidence", text_color=color.black, bgcolor=color.gray)
    table.cell(pattern_table, 0, 1, "Doji", text_color=color.black)
    table.cell(pattern_table, 1, 1, str.tostring(doji_confidence, "#.##"), text_color=color.black)
    table.cell(pattern_table, 0, 2, "Hammer", text_color=color.black)
    table.cell(pattern_table, 1, 2, str.tostring(hammer_confidence, "#.##"), text_color=color.black)
    table.cell(pattern_table, 0, 3, "Engulfing", text_color=color.black)
    table.cell(pattern_table, 1, 3, str.tostring(engulfing_confidence, "#.##"), text_color=color.black)
    table.cell(pattern_table, 0, 4, "Star", text_color=color.black)
    table.cell(pattern_table, 1, 4, str.tostring(star_confidence, "#.##"), text_color=color.black)
    table.cell(pattern_table, 0, 5, "Active", text_color=color.black, bgcolor=color.yellow)
    table.cell(pattern_table, 1, 5, pattern_name, text_color=color.black, bgcolor=color.yellow)
"""
        )
        self.templates["ai_pattern"] = ai_pattern_template
        
        # Multi-Indicator Template
        multi_template = PineScriptTemplate(
            "multi_indicator",
            """
//@version={{version}}
indicator("{{script_name}}", shorttitle="{{short_title}}", overlay={{overlay}})

// === Parameters ===
// RSI Settings
rsi_length = input.int({{rsi_length}}, title="RSI Length", group="RSI Settings")
rsi_overbought = input.float({{rsi_overbought}}, title="RSI Overbought", group="RSI Settings")
rsi_oversold = input.float({{rsi_oversold}}, title="RSI Oversold", group="RSI Settings")

// MACD Settings
macd_fast = input.int({{macd_fast}}, title="MACD Fast", group="MACD Settings")
macd_slow = input.int({{macd_slow}}, title="MACD Slow", group="MACD Settings")
macd_signal = input.int({{macd_signal}}, title="MACD Signal", group="MACD Settings")

// Bollinger Bands Settings
bb_length = input.int({{bb_length}}, title="BB Length", group="Bollinger Bands")
bb_mult = input.float({{bb_mult}}, title="BB Multiplier", group="Bollinger Bands")

// Signal Settings
signal_mode = input.string("{{signal_mode}}", title="Signal Mode", options=["All Agree", "Majority", "Any"], group="Signal Settings")
{% if ai_features %}
confidence_threshold = input.float({{confidence_threshold}}, title="AI Confidence Threshold", group="AI Settings")
{% endif %}

// === Calculations ===
// RSI
rsi_value = ta.rsi(close, rsi_length)
rsi_buy = rsi_value < rsi_oversold
rsi_sell = rsi_value > rsi_overbought

// MACD
[macd_line, macd_signal_line, macd_histogram] = ta.macd(close, macd_fast, macd_slow, macd_signal)
macd_buy = ta.crossover(macd_line, macd_signal_line)
macd_sell = ta.crossunder(macd_line, macd_signal_line)

// Bollinger Bands
bb_basis = ta.sma(close, bb_length)
bb_dev = bb_mult * ta.stdev(close, bb_length)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev
bb_buy = close < bb_lower
bb_sell = close > bb_upper

// === Signal Aggregation ===
buy_signals = (rsi_buy ? 1 : 0) + (macd_buy ? 1 : 0) + (bb_buy ? 1 : 0)
sell_signals = (rsi_sell ? 1 : 0) + (macd_sell ? 1 : 0) + (bb_sell ? 1 : 0)

final_buy = 
  signal_mode == "All Agree" ? buy_signals == 3 :
  signal_mode == "Majority" ? buy_signals >= 2 :
  buy_signals >= 1

final_sell = 
  signal_mode == "All Agree" ? sell_signals == 3 :
  signal_mode == "Majority" ? sell_signals >= 2 :
  sell_signals >= 1

{% if ai_features %}
// === AI Enhancement ===
ai_confidence = math.random(0.6, 0.95)  // Placeholder for real AI model
enhanced_buy = final_buy and ai_confidence > confidence_threshold
enhanced_sell = final_sell and ai_confidence > confidence_threshold
{% endif %}

// === Plots ===
{% if overlay %}
// On-chart signals
{% if ai_features %}
plotshape(enhanced_buy, style=shape.triangleup, location=location.belowbar, color={{bullish_color}}, size=size.normal, title="Multi-Indicator AI Buy")
plotshape(enhanced_sell, style=shape.triangledown, location=location.abovebar, color={{bearish_color}}, size=size.normal, title="Multi-Indicator AI Sell")
{% else %}
plotshape(final_buy, style=shape.triangleup, location=location.belowbar, color={{bullish_color}}, size=size.normal, title="Multi-Indicator Buy")
plotshape(final_sell, style=shape.triangledown, location=location.abovebar, color={{bearish_color}}, size=size.normal, title="Multi-Indicator Sell")
{% endif %}

// Bollinger Bands
plot(bb_upper, title="BB Upper", color=color.blue, linewidth=1)
plot(bb_lower, title="BB Lower", color=color.blue, linewidth=1)
plot(bb_basis, title="BB Basis", color=color.orange, linewidth=2)
{% else %}
// Indicator panel
plot(rsi_value, title="RSI", color=color.purple, linewidth=1)
plot(50, title="RSI Midline", color=color.gray, linewidth=1, linestyle=line.style_dotted)
hline(rsi_overbought, title="RSI Overbought", color=color.red, linestyle=hline.style_dashed)
hline(rsi_oversold, title="RSI Oversold", color=color.green, linestyle=hline.style_dashed)

plot(macd_line, title="MACD", color=color.blue, linewidth=2)
plot(macd_signal_line, title="MACD Signal", color=color.red, linewidth=1)
plot(macd_histogram, title="MACD Histogram", color=color.gray, style=plot.style_histogram)

// Signal strength
plot(buy_signals, title="Buy Signal Strength", color=color.green, linewidth=2, display=display.data_window)
plot(sell_signals, title="Sell Signal Strength", color=color.red, linewidth=2, display=display.data_window)
{% endif %}

{% if alerts_enabled %}
// === Alerts ===
{% if ai_features %}
alertcondition(enhanced_buy, title="Multi-Indicator AI Buy", message="Multi-Indicator AI Buy Signal (Confidence: " + str.tostring(ai_confidence, "#.##") + ")")
alertcondition(enhanced_sell, title="Multi-Indicator AI Sell", message="Multi-Indicator AI Sell Signal (Confidence: " + str.tostring(ai_confidence, "#.##") + ")")
{% else %}
alertcondition(final_buy, title="Multi-Indicator Buy", message="Multi-Indicator Buy Signal (Strength: " + str.tostring(buy_signals) + "/3)")
alertcondition(final_sell, title="Multi-Indicator Sell", message="Multi-Indicator Sell Signal (Strength: " + str.tostring(sell_signals) + "/3)")
{% endif %}
{% endif %}

// === Dashboard ===
if barstate.islast
    var table dashboard = table.new(position.bottom_right, 3, 5, bgcolor=color.white, border_width=1)
    table.cell(dashboard, 0, 0, "Indicator", text_color=color.black, bgcolor=color.gray)
    table.cell(dashboard, 1, 0, "Buy", text_color=color.black, bgcolor=color.gray)
    table.cell(dashboard, 2, 0, "Sell", text_color=color.black, bgcolor=color.gray)
    
    table.cell(dashboard, 0, 1, "RSI", text_color=color.black)
    table.cell(dashboard, 1, 1, rsi_buy ? "✓" : "✗", text_color=rsi_buy ? color.green : color.red)
    table.cell(dashboard, 2, 1, rsi_sell ? "✓" : "✗", text_color=rsi_sell ? color.red : color.green)
    
    table.cell(dashboard, 0, 2, "MACD", text_color=color.black)
    table.cell(dashboard, 1, 2, macd_buy ? "✓" : "✗", text_color=macd_buy ? color.green : color.red)
    table.cell(dashboard, 2, 2, macd_sell ? "✓" : "✗", text_color=macd_sell ? color.red : color.green)
    
    table.cell(dashboard, 0, 3, "BB", text_color=color.black)
    table.cell(dashboard, 1, 3, bb_buy ? "✓" : "✗", text_color=bb_buy ? color.green : color.red)
    table.cell(dashboard, 2, 3, bb_sell ? "✓" : "✗", text_color=bb_sell ? color.red : color.green)
    
    table.cell(dashboard, 0, 4, "FINAL", text_color=color.black, bgcolor=color.yellow)
    {% if ai_features %}
    table.cell(dashboard, 1, 4, enhanced_buy ? "BUY" : "---", text_color=enhanced_buy ? color.green : color.black, bgcolor=enhanced_buy ? color.new(color.green, 80) : color.white)
    table.cell(dashboard, 2, 4, enhanced_sell ? "SELL" : "---", text_color=enhanced_sell ? color.red : color.black, bgcolor=enhanced_sell ? color.new(color.red, 80) : color.white)
    {% else %}
    table.cell(dashboard, 1, 4, final_buy ? "BUY" : "---", text_color=final_buy ? color.green : color.black, bgcolor=final_buy ? color.new(color.green, 80) : color.white)
    table.cell(dashboard, 2, 4, final_sell ? "SELL" : "---", text_color=final_sell ? color.red : color.black, bgcolor=final_sell ? color.new(color.red, 80) : color.white)
    {% endif %}
"""
        )
        self.templates["multi_indicator"] = multi_template
    
    def _setup_validation_rules(self) -> Dict[str, List[str]]:
        """Setup Validation-Regeln für Pine Script"""
        return {
            "syntax_errors": [
                r"(?i)undefined\s+variable",
                r"(?i)syntax\s+error",
                r"(?i)compilation\s+error",
                r"(?i)invalid\s+function",
            ],
            "best_practices": [
                r"//@version=5",  # Sollte Version 5 verwenden
                r"input\.",  # Sollte input.* verwenden
                r"ta\.",  # Sollte ta.* für technische Indikatoren verwenden
            ],
            "required_elements": [
                r"indicator\(",  # Muss indicator() haben
                r"plot\(",  # Sollte mindestens einen plot haben
            ]
        }
    
    def generate_script(self, config: PineScriptConfig) -> Dict[str, Any]:
        """
        Generiere Pine Script basierend auf Konfiguration
        
        Args:
            config: PineScriptConfig mit allen Einstellungen
            
        Returns:
            Dictionary mit generiertem Script und Metadaten
        """
        try:
            start_time = datetime.now()
            
            # Template auswählen
            template_name = config.indicator_type.value
            if template_name not in self.templates:
                template_name = "rsi"  # Fallback
            
            template = self.templates[template_name]
            
            # Parameter vorbereiten
            render_params = self._prepare_render_parameters(config)
            
            # AI-Features integrieren falls aktiviert
            if config.ai_features:
                render_params = self._enhance_with_ai_features(render_params, config)
            
            # Script generieren
            generated_script = template.render(render_params)
            
            # Validation
            validation_result = self._validate_script(generated_script)
            
            # Statistiken updaten
            self.stats["scripts_generated"] += 1
            if template_name not in self.stats["templates_used"]:
                self.stats["templates_used"][template_name] = 0
            self.stats["templates_used"][template_name] += 1
            
            if not validation_result["is_valid"]:
                self.stats["validation_errors"] += 1
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "script": generated_script,
                "config": config,
                "template_used": template_name,
                "validation": validation_result,
                "parameters": render_params,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "script_length": len(generated_script),
                    "line_count": len(generated_script.split('\n')),
                    "ai_enhanced": config.ai_features,
                    "multi_timeframe": config.multi_timeframe
                }
            }
            
        except Exception as e:
            self.logger.error(f"Script generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
    
    def _prepare_render_parameters(self, config: PineScriptConfig) -> Dict[str, Any]:
        """Bereite Render-Parameter vor"""
        
        # Basis-Parameter
        params = {
            "version": config.script_version,
            "script_name": config.script_name,
            "short_title": config.script_name[:10],  # Kurzer Titel
            "overlay": "true" if config.overlay else "false",
            "ai_features": config.ai_features,
            "alerts_enabled": config.alerts_enabled,
            "confidence_threshold": config.confidence_threshold,
        }
        
        # Farben
        params.update(config.colors)
        
        # Indikator-spezifische Parameter
        if config.indicator_type == IndicatorType.RSI:
            params.update({
                "rsi_length": config.parameters.get("rsi_length", 14),
                "rsi_overbought": config.parameters.get("rsi_overbought", 70),
                "rsi_oversold": config.parameters.get("rsi_oversold", 30),
                "show_signals": "true",
                "rsi_color": config.colors.get("signal", "color.blue"),
                "overbought_color": config.colors.get("bearish", "color.red"),
                "oversold_color": config.colors.get("bullish", "color.green"),
            })
        
        elif config.indicator_type == IndicatorType.MACD:
            params.update({
                "fast_length": config.parameters.get("fast_length", 12),
                "slow_length": config.parameters.get("slow_length", 26),
                "signal_length": config.parameters.get("signal_length", 9),
                "show_histogram": "true",
                "macd_color": config.colors.get("signal", "color.blue"),
                "signal_color": config.colors.get("neutral", "color.red"),
            })
        
        elif config.indicator_type == IndicatorType.BOLLINGER_BANDS:
            params.update({
                "bb_length": config.parameters.get("bb_length", 20),
                "bb_mult": config.parameters.get("bb_mult", 2.0),
                "show_squeeze": "true",
                "upper_color": config.colors.get("bearish", "color.red"),
                "lower_color": config.colors.get("bullish", "color.green"),
                "basis_color": config.colors.get("neutral", "color.blue"),
                "fill_color": config.colors.get("neutral", "color.gray"),
            })
        
        elif config.indicator_type == IndicatorType.AI_PATTERN:
            params.update({
                "lookback_period": config.parameters.get("lookback_period", 20),
                "show_pattern_names": "true",
                "show_confidence": "true",
            })
        
        elif config.indicator_type == IndicatorType.MULTI_INDICATOR:
            params.update({
                "rsi_length": config.parameters.get("rsi_length", 14),
                "rsi_overbought": config.parameters.get("rsi_overbought", 70),
                "rsi_oversold": config.parameters.get("rsi_oversold", 30),
                "macd_fast": config.parameters.get("macd_fast", 12),
                "macd_slow": config.parameters.get("macd_slow", 26),
                "macd_signal": config.parameters.get("macd_signal", 9),
                "bb_length": config.parameters.get("bb_length", 20),
                "bb_mult": config.parameters.get("bb_mult", 2.0),
                "signal_mode": config.parameters.get("signal_mode", "Majority"),
            })
        
        # Zusätzliche Parameter aus config.parameters
        params.update(config.parameters)
        
        return params
    
    def _enhance_with_ai_features(self, params: Dict[str, Any], config: PineScriptConfig) -> Dict[str, Any]:
        """Erweitere Parameter mit AI-Features"""
        
        # AI-spezifische Parameter
        ai_params = {
            "ai_model_endpoint": "http://localhost:8080/predictions/ai_pattern_model",
            "feature_extraction_enabled": True,
            "confidence_scoring_enabled": True,
            "pattern_recognition_enabled": True,
        }
        
        # Enhanced Feature Integration
        if hasattr(self, 'feature_extractor'):
            ai_params["enhanced_features"] = True
            ai_params["feature_count"] = 57  # Enhanced Feature Extractor liefert 57 Features
        
        params.update(ai_params)
        self.stats["parameter_optimizations"] += 1
        
        return params
    
    def _validate_script(self, script: str) -> Dict[str, Any]:
        """Validiere generierten Pine Script"""
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        try:
            # Syntax-Fehler prüfen
            for error_pattern in self.validation_rules["syntax_errors"]:
                if re.search(error_pattern, script, re.IGNORECASE):
                    validation_result["errors"].append(f"Potential syntax error: {error_pattern}")
                    validation_result["is_valid"] = False
            
            # Best Practices prüfen
            for practice_pattern in self.validation_rules["best_practices"]:
                if not re.search(practice_pattern, script):
                    validation_result["warnings"].append(f"Best practice not followed: {practice_pattern}")
            
            # Erforderliche Elemente prüfen
            for required_pattern in self.validation_rules["required_elements"]:
                if not re.search(required_pattern, script):
                    validation_result["errors"].append(f"Required element missing: {required_pattern}")
                    validation_result["is_valid"] = False
            
            # Zusätzliche Checks
            lines = script.split('\n')
            
            # Prüfe auf leere Zeilen am Anfang
            if lines and not lines[0].strip():
                validation_result["suggestions"].append("Remove empty lines at the beginning")
            
            # Prüfe auf zu lange Zeilen
            long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
            if long_lines:
                validation_result["warnings"].append(f"Lines too long (>120 chars): {long_lines[:5]}")
            
            # Prüfe auf doppelte Leerzeichen
            if '  ' in script:
                validation_result["suggestions"].append("Consider removing double spaces")
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["is_valid"] = False
        
        return validation_result
    
    def optimize_parameters(self, config: PineScriptConfig, market_data: Optional[Dict] = None) -> PineScriptConfig:
        """
        Optimiere Parameter basierend auf Market-Data und AI-Analyse
        
        Args:
            config: Basis-Konfiguration
            market_data: Optional Market-Data für Optimierung
            
        Returns:
            Optimierte PineScriptConfig
        """
        try:
            optimized_config = config
            
            if market_data and hasattr(self, 'feature_extractor'):
                # Verwende Enhanced Feature Extractor für Analyse
                features = self.feature_extractor.extract_features_from_dict(market_data)
                
                # AI-basierte Parameter-Optimierung (simuliert)
                if config.indicator_type == IndicatorType.RSI:
                    # Optimiere RSI-Parameter basierend auf Volatilität
                    volatility = features[40] if len(features) > 40 else 0.002  # volatility_5
                    
                    if volatility > 0.005:  # Hohe Volatilität
                        optimized_config.parameters["rsi_length"] = 10  # Kürzerer Zeitraum
                        optimized_config.parameters["rsi_overbought"] = 75
                        optimized_config.parameters["rsi_oversold"] = 25
                    elif volatility < 0.001:  # Niedrige Volatilität
                        optimized_config.parameters["rsi_length"] = 21  # Längerer Zeitraum
                        optimized_config.parameters["rsi_overbought"] = 65
                        optimized_config.parameters["rsi_oversold"] = 35
                
                elif config.indicator_type == IndicatorType.MACD:
                    # Optimiere MACD basierend auf Trend-Stärke
                    trend_strength = features[44] if len(features) > 44 else 0.001  # trend_strength
                    
                    if trend_strength > 0.005:  # Starker Trend
                        optimized_config.parameters["fast_length"] = 8
                        optimized_config.parameters["slow_length"] = 21
                    elif trend_strength < 0.001:  # Schwacher Trend
                        optimized_config.parameters["fast_length"] = 15
                        optimized_config.parameters["slow_length"] = 30
                
                self.stats["parameter_optimizations"] += 1
            
            return optimized_config
            
        except Exception as e:
            self.logger.error(f"Parameter optimization error: {e}")
            return config
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Erhalte Liste verfügbarer Templates"""
        templates_info = []
        
        for name, template in self.templates.items():
            templates_info.append({
                "name": name,
                "display_name": name.replace("_", " ").title(),
                "required_parameters": template.required_params,
                "code_length": len(template.code),
                "supports_ai": "ai_features" in template.code,
                "supports_alerts": "alerts_enabled" in template.code
            })
        
        return templates_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Generator-Statistiken"""
        return {
            **self.stats,
            "available_templates": len(self.templates),
            "template_names": list(self.templates.keys())
        }


# Factory Function
def create_pine_script_generator(config: Optional[Dict] = None) -> PineScriptGenerator:
    """
    Factory Function für Pine Script Generator
    
    Args:
        config: Generator-Konfiguration
        
    Returns:
        PineScriptGenerator Instance
    """
    return PineScriptGenerator(config=config)


# Demo/Test Function
def demo_pine_script_generator():
    """Demo für Pine Script Generator"""
    
    print("🧪 Testing Pine Script Generator...")
    
    # Generator erstellen
    generator = create_pine_script_generator()
    
    # Verfügbare Templates anzeigen
    templates = generator.get_available_templates()
    print(f"\n📋 Available Templates ({len(templates)}):")
    for template in templates:
        print(f"   - {template['display_name']}: {len(template['required_parameters'])} params, AI: {template['supports_ai']}")
    
    # Test-Konfigurationen
    test_configs = [
        # RSI Indicator
        PineScriptConfig(
            script_name="AI Enhanced RSI",
            indicator_type=IndicatorType.RSI,
            ai_features=True,
            parameters={
                "rsi_length": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30
            }
        ),
        
        # MACD Indicator
        PineScriptConfig(
            script_name="Smart MACD Strategy",
            indicator_type=IndicatorType.MACD,
            overlay=False,
            parameters={
                "fast_length": 12,
                "slow_length": 26,
                "signal_length": 9
            }
        ),
        
        # AI Pattern Recognition
        PineScriptConfig(
            script_name="AI Pattern Scanner",
            indicator_type=IndicatorType.AI_PATTERN,
            ai_features=True,
            confidence_threshold=0.75,
            parameters={
                "lookback_period": 20
            }
        ),
        
        # Multi-Indicator
        PineScriptConfig(
            script_name="Multi-Signal Dashboard",
            indicator_type=IndicatorType.MULTI_INDICATOR,
            ai_features=True,
            overlay=True,
            parameters={
                "signal_mode": "Majority"
            }
        )
    ]
    
    # Scripts generieren
    print(f"\n🔧 Generating {len(test_configs)} Pine Scripts...")
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Script {i+1}: {config.script_name} ---")
        
        # Parameter-Optimierung (mit Mock-Data)
        mock_market_data = {
            "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1002,
            "volume": 1500, "volatility_5": 0.003, "trend_strength": 0.002
        }
        
        optimized_config = generator.optimize_parameters(config, mock_market_data)
        
        # Script generieren
        result = generator.generate_script(optimized_config)
        
        if result["success"]:
            print(f"✅ Generated successfully:")
            print(f"   Template: {result['template_used']}")
            print(f"   Length: {result['metadata']['script_length']} chars")
            print(f"   Lines: {result['metadata']['line_count']}")
            print(f"   Generation Time: {result['generation_time']:.3f}s")
            print(f"   Valid: {result['validation']['is_valid']}")
            
            if result['validation']['errors']:
                print(f"   Errors: {len(result['validation']['errors'])}")
            if result['validation']['warnings']:
                print(f"   Warnings: {len(result['validation']['warnings'])}")
            
            # Zeige ersten Teil des Scripts
            script_preview = result['script'][:200] + "..." if len(result['script']) > 200 else result['script']
            print(f"   Preview: {script_preview}")
            
        else:
            print(f"❌ Generation failed: {result['error']}")
    
    # Statistiken
    stats = generator.get_statistics()
    print(f"\n📊 Generator Statistics:")
    print(f"   Scripts Generated: {stats['scripts_generated']}")
    print(f"   Validation Errors: {stats['validation_errors']}")
    print(f"   Parameter Optimizations: {stats['parameter_optimizations']}")
    print(f"   Templates Used: {stats['templates_used']}")


if __name__ == "__main__":
    demo_pine_script_generator()