#!/usr/bin/env python3
"""
🧩 BAUSTEIN C1: KI-Enhanced Pine Script Generator
Automatische Pine Script Code-Generierung basierend auf KI-Strategien-Bewertung

Features:
- Integration mit Baustein B3 (AI Strategy Evaluator)
- Automatische Pine Script v5 Code-Generierung
- KI-basierte Entry/Exit-Logik
- Multimodale Konfidenz-Integration
- Risk-Management-Implementierung
- Top-5-Strategien Pine Script Export
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
import textwrap
from collections import defaultdict

# Import Baustein B3 Komponenten
try:
    from ai_indicator_optimizer.ai.ai_strategy_evaluator import (
        AIStrategyEvaluator, StrategyScore, Top5StrategiesResult, StrategyRankingCriteria
    )
    from ai_indicator_optimizer.logging.unified_schema_manager import UnifiedSchemaManager, DataStreamType
except ImportError as e:
    print(f"Warning: Could not import dependencies: {e}")


class PineScriptVersion(Enum):
    """Pine Script Versionen"""
    V4 = "4"
    V5 = "5"


class StrategyType(Enum):
    """Pine Script Strategien-Typen"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"


class RiskManagementType(Enum):
    """Risk-Management-Typen"""
    FIXED_STOP_LOSS = "fixed_stop_loss"
    ATR_BASED = "atr_based"
    PERCENTAGE_BASED = "percentage_based"
    DYNAMIC_TRAILING = "dynamic_trailing"


@dataclass
class PineScriptConfig:
    """Konfiguration für Pine Script Generierung"""
    version: PineScriptVersion = PineScriptVersion.V5
    strategy_type: StrategyType = StrategyType.SWING_TRADING
    risk_management: RiskManagementType = RiskManagementType.ATR_BASED
    
    # Trading-Parameter
    initial_capital: float = 10000.0
    default_qty_type: str = "strategy.percent_of_equity"
    default_qty_value: float = 10.0  # 10% of equity
    
    # Risk-Management-Parameter
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_risk_per_trade: float = 0.02  # 2%
    
    # Technische Parameter
    include_alerts: bool = True
    include_plotting: bool = True
    include_backtesting: bool = True
    
    # KI-Integration
    use_confidence_filtering: bool = True
    min_confidence_threshold: float = 0.6
    use_multimodal_confirmation: bool = True


@dataclass
class GeneratedPineScript:
    """Generierter Pine Script Code"""
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    
    # Pine Script Code
    pine_script_code: str
    
    # Metadaten
    generation_timestamp: datetime
    strategy_score: StrategyScore
    config: PineScriptConfig
    
    # Validierung
    syntax_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Performance-Schätzungen
    estimated_performance: Dict[str, float] = field(default_factory=dict)
    
    # Code-Statistiken
    code_lines: int = 0
    code_complexity: str = "medium"
    
    # Export-Informationen
    file_path: Optional[str] = None
    export_timestamp: Optional[datetime] = None


class PineScriptTemplateEngine:
    """
    Template-Engine für Pine Script Code-Generierung
    
    Generiert strukturierte Pine Script Templates basierend auf:
    - Strategien-Typ
    - KI-Bewertungen
    - Risk-Management-Einstellungen
    - Multimodale Konfidenz-Scores
    """
    
    def __init__(self):
        """Initialize Template Engine"""
        self.logger = logging.getLogger(__name__)
        
        # Template-Bibliothek
        self.templates = {
            "header": self._get_header_template(),
            "inputs": self._get_inputs_template(),
            "indicators": self._get_indicators_template(),
            "strategy_logic": self._get_strategy_logic_template(),
            "risk_management": self._get_risk_management_template(),
            "alerts": self._get_alerts_template(),
            "plotting": self._get_plotting_template()
        }
    
    def generate_pine_script(
        self,
        strategy_score: StrategyScore,
        config: PineScriptConfig
    ) -> str:
        """
        Generiere vollständigen Pine Script Code
        
        Args:
            strategy_score: Strategien-Score von Baustein B3
            config: Pine Script Konfiguration
            
        Returns:
            Vollständiger Pine Script Code
        """
        try:
            # Template-Variablen vorbereiten
            template_vars = self._prepare_template_variables(strategy_score, config)
            
            # Code-Abschnitte generieren
            code_sections = []
            
            # 1. Header
            code_sections.append(self._generate_header(template_vars, config))
            
            # 2. Inputs
            code_sections.append(self._generate_inputs(template_vars, config))
            
            # 3. Indikatoren
            code_sections.append(self._generate_indicators(template_vars, config))
            
            # 4. Strategien-Logik
            code_sections.append(self._generate_strategy_logic(template_vars, config))
            
            # 5. Risk-Management
            code_sections.append(self._generate_risk_management(template_vars, config))
            
            # 6. Alerts (optional)
            if config.include_alerts:
                code_sections.append(self._generate_alerts(template_vars, config))
            
            # 7. Plotting (optional)
            if config.include_plotting:
                code_sections.append(self._generate_plotting(template_vars, config))
            
            # Code zusammenfügen
            pine_script_code = "\n\n".join(code_sections)
            
            # Code formatieren
            pine_script_code = self._format_code(pine_script_code)
            
            return pine_script_code
            
        except Exception as e:
            self.logger.error(f"Pine Script generation failed: {e}")
            return self._get_fallback_pine_script(strategy_score, config)
    
    def _prepare_template_variables(self, strategy_score: StrategyScore, config: PineScriptConfig) -> Dict[str, Any]:
        """Bereite Template-Variablen vor"""
        
        return {
            # Basis-Informationen
            "strategy_name": f"AI_Strategy_{strategy_score.symbol.replace('/', '_')}_{strategy_score.timeframe}",
            "strategy_id": strategy_score.strategy_id,
            "symbol": strategy_score.symbol,
            "timeframe": strategy_score.timeframe,
            "timestamp": strategy_score.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            
            # KI-Scores
            "signal_confidence": strategy_score.signal_confidence_score,
            "risk_reward_ratio": strategy_score.risk_reward_score,
            "opportunity_score": strategy_score.opportunity_score,
            "fusion_confidence": strategy_score.fusion_confidence_score,
            "composite_score": strategy_score.composite_score,
            "weighted_score": strategy_score.weighted_score,
            
            # Performance-Schätzungen
            "expected_return": strategy_score.expected_return,
            "expected_risk": strategy_score.expected_risk,
            "expected_sharpe": strategy_score.expected_sharpe,
            
            # Trading-Parameter
            "initial_capital": config.initial_capital,
            "qty_type": config.default_qty_type,
            "qty_value": config.default_qty_value,
            
            # Risk-Management
            "stop_loss_atr": config.stop_loss_atr_multiplier,
            "take_profit_atr": config.take_profit_atr_multiplier,
            "max_risk": config.max_risk_per_trade,
            
            # KI-Parameter
            "min_confidence": config.min_confidence_threshold,
            "use_multimodal": config.use_multimodal_confirmation,
            
            # Technische Parameter
            "pine_version": config.version.value
        }
    
    def _generate_header(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Pine Script Header"""
        
        header = f'''//@version={vars["pine_version"]}
strategy("{vars["strategy_name"]}", 
         shorttitle="{vars["symbol"]}_{vars["timeframe"]}", 
         overlay=true,
         initial_capital={vars["initial_capital"]},
         default_qty_type={vars["qty_type"]},
         default_qty_value={vars["qty_value"]},
         commission_type=strategy.commission.percent,
         commission_value=0.1)

// 🧩 AI-Enhanced Trading Strategy
// Generated by: KI-Enhanced Pine Script Generator (Baustein C1)
// Strategy ID: {vars["strategy_id"]}
// Symbol: {vars["symbol"]}
// Timeframe: {vars["timeframe"]}
// Generated: {vars["timestamp"]}
//
// 📊 AI Performance Metrics:
// Signal Confidence: {vars["signal_confidence"]:.1%}
// Risk/Reward Score: {vars["risk_reward_ratio"]:.3f}
// Opportunity Score: {vars["opportunity_score"]:.1%}
// Fusion Confidence: {vars["fusion_confidence"]:.1%}
// Composite Score: {vars["composite_score"]:.3f}
// Weighted Score: {vars["weighted_score"]:.3f}
//
// 📈 Expected Performance:
// Expected Return: {vars["expected_return"]:.1%}
// Expected Risk: {vars["expected_risk"]:.1%}
// Expected Sharpe: {vars["expected_sharpe"]:.2f}'''
        
        return header
    
    def _generate_inputs(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Input-Parameter"""
        
        inputs = f'''// 🎛️ Input Parameters
// AI Configuration
ai_confidence_threshold = input.float({vars["min_confidence"]}, "AI Confidence Threshold", minval=0.0, maxval=1.0, step=0.01, group="AI Settings")
use_multimodal_confirmation = input.bool({str(vars["use_multimodal"]).lower()}, "Use Multimodal Confirmation", group="AI Settings")

// Risk Management
stop_loss_atr_mult = input.float({vars["stop_loss_atr"]}, "Stop Loss ATR Multiplier", minval=0.5, maxval=5.0, step=0.1, group="Risk Management")
take_profit_atr_mult = input.float({vars["take_profit_atr"]}, "Take Profit ATR Multiplier", minval=1.0, maxval=10.0, step=0.1, group="Risk Management")
max_risk_per_trade = input.float({vars["max_risk"] * 100}, "Max Risk Per Trade (%)", minval=0.5, maxval=10.0, step=0.1, group="Risk Management") / 100

// Technical Indicators
rsi_length = input.int(14, "RSI Length", minval=5, maxval=50, group="Technical Indicators")
ema_fast = input.int(12, "EMA Fast", minval=5, maxval=50, group="Technical Indicators")
ema_slow = input.int(26, "EMA Slow", minval=20, maxval=100, group="Technical Indicators")
atr_length = input.int(14, "ATR Length", minval=5, maxval=50, group="Technical Indicators")

// Strategy Settings
enable_long = input.bool(true, "Enable Long Trades", group="Strategy Settings")
enable_short = input.bool(true, "Enable Short Trades", group="Strategy Settings")'''
        
        return inputs
    
    def _generate_indicators(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere technische Indikatoren"""
        
        indicators = '''// 📊 Technical Indicators
// Moving Averages
ema_fast_val = ta.ema(close, ema_fast)
ema_slow_val = ta.ema(close, ema_slow)

// RSI
rsi_val = ta.rsi(close, rsi_length)

// ATR for Risk Management
atr_val = ta.atr(atr_length)

// Bollinger Bands
bb_length = 20
bb_mult = 2.0
bb_basis = ta.sma(close, bb_length)
bb_dev = bb_mult * ta.stdev(close, bb_length)
bb_upper = bb_basis + bb_dev
bb_lower = bb_basis - bb_dev

// MACD
[macd_line, signal_line, hist] = ta.macd(close, 12, 26, 9)

// 🧠 AI-Enhanced Indicators
// Multimodal Confidence Score (simulated based on technical confluence)
technical_confluence = 0.0
technical_confluence := (rsi_val > 30 and rsi_val < 70) ? technical_confluence + 0.2 : technical_confluence
technical_confluence := (close > ema_fast_val and ema_fast_val > ema_slow_val) ? technical_confluence + 0.3 : technical_confluence
technical_confluence := (macd_line > signal_line) ? technical_confluence + 0.2 : technical_confluence
technical_confluence := (close > bb_basis) ? technical_confluence + 0.15 : technical_confluence
technical_confluence := (hist > hist[1]) ? technical_confluence + 0.15 : technical_confluence

// AI Confidence Score (based on strategy performance metrics)
ai_confidence_score = ''' + f'{vars["signal_confidence"]:.3f}' + '''
multimodal_confidence = ''' + f'{vars["fusion_confidence"]:.3f}' + '''

// Combined AI Signal Strength
combined_ai_signal = (ai_confidence_score + multimodal_confidence + technical_confluence) / 3.0'''
        
        return indicators
    
    def _generate_strategy_logic(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Strategien-Logik"""
        
        strategy_logic = f'''// 🎯 AI-Enhanced Strategy Logic
// Entry Conditions
long_condition = false
short_condition = false

// AI-Based Entry Signals
if use_multimodal_confirmation
    // Multimodal AI Entry Logic
    long_condition := enable_long and 
                     combined_ai_signal > ai_confidence_threshold and
                     close > ema_fast_val and 
                     ema_fast_val > ema_slow_val and
                     rsi_val > 40 and rsi_val < 80 and
                     macd_line > signal_line and
                     close > bb_basis
    
    short_condition := enable_short and 
                      combined_ai_signal > ai_confidence_threshold and
                      close < ema_fast_val and 
                      ema_fast_val < ema_slow_val and
                      rsi_val < 60 and rsi_val > 20 and
                      macd_line < signal_line and
                      close < bb_basis
else
    // Standard AI Entry Logic
    long_condition := enable_long and 
                     ai_confidence_score > ai_confidence_threshold and
                     close > ema_fast_val and 
                     ema_fast_val > ema_slow_val and
                     rsi_val > 50
    
    short_condition := enable_short and 
                      ai_confidence_score > ai_confidence_threshold and
                      close < ema_fast_val and 
                      ema_fast_val < ema_slow_val and
                      rsi_val < 50

// 📈 Position Sizing based on AI Confidence
// Higher confidence = larger position size (within risk limits)
confidence_multiplier = math.min(combined_ai_signal * 1.5, 2.0)  // Max 2x multiplier
position_size = strategy.equity * max_risk_per_trade * confidence_multiplier / (stop_loss_atr_mult * atr_val)

// Execute Trades
if long_condition and strategy.position_size == 0
    strategy.entry("AI_Long", strategy.long, qty=position_size)
    
if short_condition and strategy.position_size == 0
    strategy.entry("AI_Short", strategy.short, qty=position_size)'''
        
        return strategy_logic
    
    def _generate_risk_management(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Risk-Management"""
        
        risk_management = '''// 🛡️ AI-Enhanced Risk Management
// Dynamic Stop Loss and Take Profit based on ATR and AI Confidence
if strategy.position_size > 0  // Long Position
    stop_loss_long = strategy.position_avg_price - (stop_loss_atr_mult * atr_val)
    take_profit_long = strategy.position_avg_price + (take_profit_atr_mult * atr_val)
    
    // AI-Enhanced Exit: Reduce confidence threshold for exits
    ai_exit_threshold = ai_confidence_threshold * 0.7
    
    strategy.exit("AI_Long_Exit", "AI_Long", 
                  stop=stop_loss_long, 
                  limit=take_profit_long)
    
    // Early exit if AI confidence drops significantly
    if combined_ai_signal < ai_exit_threshold
        strategy.close("AI_Long", comment="AI Confidence Drop")

if strategy.position_size < 0  // Short Position
    stop_loss_short = strategy.position_avg_price + (stop_loss_atr_mult * atr_val)
    take_profit_short = strategy.position_avg_price - (take_profit_atr_mult * atr_val)
    
    // AI-Enhanced Exit: Reduce confidence threshold for exits
    ai_exit_threshold = ai_confidence_threshold * 0.7
    
    strategy.exit("AI_Short_Exit", "AI_Short", 
                  stop=stop_loss_short, 
                  limit=take_profit_short)
    
    // Early exit if AI confidence drops significantly
    if combined_ai_signal < ai_exit_threshold
        strategy.close("AI_Short", comment="AI Confidence Drop")

// Emergency Risk Management
// Close all positions if drawdown exceeds threshold
max_drawdown_threshold = 0.10  // 10%
current_drawdown = (strategy.max_equity - strategy.equity) / strategy.max_equity

if current_drawdown > max_drawdown_threshold
    strategy.close_all(comment="Emergency Drawdown Protection")'''
        
        return risk_management
    
    def _generate_alerts(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Alert-System"""
        
        alerts = '''// 🔔 AI-Enhanced Alerts
// Entry Alerts
if long_condition and strategy.position_size == 0
    alert("AI Long Entry Signal\\n" + 
          "Symbol: ''' + vars["symbol"] + '''\\n" +
          "Confidence: " + str.tostring(combined_ai_signal, "#.###") + "\\n" +
          "Price: " + str.tostring(close, "#.#####"), 
          alert.freq_once_per_bar)

if short_condition and strategy.position_size == 0
    alert("AI Short Entry Signal\\n" + 
          "Symbol: ''' + vars["symbol"] + '''\\n" +
          "Confidence: " + str.tostring(combined_ai_signal, "#.###") + "\\n" +
          "Price: " + str.tostring(close, "#.#####"), 
          alert.freq_once_per_bar)

// Risk Management Alerts
if current_drawdown > max_drawdown_threshold * 0.8  // 80% of max drawdown
    alert("Risk Warning: High Drawdown\\n" + 
          "Current Drawdown: " + str.tostring(current_drawdown * 100, "#.##") + "%\\n" +
          "Threshold: " + str.tostring(max_drawdown_threshold * 100, "#.##") + "%", 
          alert.freq_once_per_bar)'''
        
        return alerts
    
    def _generate_plotting(self, vars: Dict[str, Any], config: PineScriptConfig) -> str:
        """Generiere Plotting-Code"""
        
        plotting = '''// 📊 AI-Enhanced Plotting
// Moving Averages
plot(ema_fast_val, "EMA Fast", color=color.blue, linewidth=1)
plot(ema_slow_val, "EMA Slow", color=color.red, linewidth=1)

// Bollinger Bands
p1 = plot(bb_upper, "BB Upper", color=color.gray, linewidth=1)
p2 = plot(bb_lower, "BB Lower", color=color.gray, linewidth=1)
fill(p1, p2, color=color.new(color.gray, 95), title="BB Background")

// AI Confidence Indicator
hline(0.5, "AI Confidence Mid", color=color.gray, linestyle=hline.style_dashed)
plot(combined_ai_signal, "AI Signal Strength", color=color.purple, linewidth=2, display=display.data_window)
plot(ai_confidence_threshold, "AI Threshold", color=color.orange, linewidth=1, display=display.data_window)

// Entry/Exit Markers
plotshape(long_condition and strategy.position_size == 0, "Long Entry", 
          shape.triangleup, location.belowbar, color.green, size=size.small)
plotshape(short_condition and strategy.position_size == 0, "Short Entry", 
          shape.triangledown, location.abovebar, color.red, size=size.small)

// AI Confidence Background
bgcolor(combined_ai_signal > ai_confidence_threshold ? color.new(color.green, 95) : 
        combined_ai_signal < ai_confidence_threshold * 0.7 ? color.new(color.red, 95) : na, 
        title="AI Confidence Background")

// Performance Info Table
if barstate.islast
    var table info_table = table.new(position.top_right, 2, 8, bgcolor=color.white, border_width=1)
    table.cell(info_table, 0, 0, "AI Strategy Info", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 0, "''' + vars["strategy_name"] + '''", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 1, "Signal Confidence", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 1, "''' + f'{vars["signal_confidence"]:.1%}' + '''", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 2, "Expected Return", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 2, "''' + f'{vars["expected_return"]:.1%}' + '''", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 3, "Expected Risk", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 3, "''' + f'{vars["expected_risk"]:.1%}' + '''", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 4, "Composite Score", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 4, "''' + f'{vars["composite_score"]:.3f}' + '''", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 5, "Current AI Signal", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 5, str.tostring(combined_ai_signal, "#.###"), text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 6, "Position Size", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 6, str.tostring(strategy.position_size, "#.##"), text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 7, "P&L", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 7, str.tostring(strategy.netprofit, "#.##"), 
               text_color=strategy.netprofit >= 0 ? color.green : color.red, text_size=size.small)'''
        
        return plotting
    
    def _format_code(self, code: str) -> str:
        """Formatiere Pine Script Code"""
        
        # Entferne überschüssige Leerzeilen
        lines = code.split('\n')
        formatted_lines = []
        
        prev_empty = False
        for line in lines:
            is_empty = line.strip() == ''
            
            if is_empty and prev_empty:
                continue  # Skip multiple empty lines
            
            formatted_lines.append(line)
            prev_empty = is_empty
        
        return '\n'.join(formatted_lines)
    
    def _get_fallback_pine_script(self, strategy_score: StrategyScore, config: PineScriptConfig) -> str:
        """Fallback Pine Script bei Fehlern"""
        
        return f'''//@version={config.version.value}
strategy("AI_Fallback_{strategy_score.symbol.replace('/', '_')}", 
         shorttitle="AI_FB", 
         overlay=true)

// Fallback AI Strategy
// Generated due to error in main generation process

// Simple Moving Average Strategy
sma_fast = ta.sma(close, 10)
sma_slow = ta.sma(close, 20)

long_condition = ta.crossover(sma_fast, sma_slow)
short_condition = ta.crossunder(sma_fast, sma_slow)

if long_condition
    strategy.entry("Long", strategy.long)
if short_condition
    strategy.entry("Short", strategy.short)

plot(sma_fast, "SMA Fast", color=color.blue)
plot(sma_slow, "SMA Slow", color=color.red)'''
    
    def _get_header_template(self) -> str:
        return "// Pine Script Header Template"
    
    def _get_inputs_template(self) -> str:
        return "// Pine Script Inputs Template"
    
    def _get_indicators_template(self) -> str:
        return "// Pine Script Indicators Template"
    
    def _get_strategy_logic_template(self) -> str:
        return "// Pine Script Strategy Logic Template"
    
    def _get_risk_management_template(self) -> str:
        return "// Pine Script Risk Management Template"
    
    def _get_alerts_template(self) -> str:
        return "// Pine Script Alerts Template"
    
    def _get_plotting_template(self) -> str:
        return "// Pine Script Plotting Template"


class PineScriptValidator:
    """
    Pine Script Syntax-Validator
    
    Validiert generierten Pine Script Code auf:
    - Syntax-Korrektheit
    - Pine Script v5 Kompatibilität
    - Logische Konsistenz
    - Performance-Optimierungen
    """
    
    def __init__(self):
        """Initialize Validator"""
        self.logger = logging.getLogger(__name__)
        
        # Validation Rules
        self.syntax_rules = {
            "version_declaration": r"^//@version=[45]$",
            "strategy_declaration": r"strategy\s*\(",
            "valid_functions": [
                "ta.sma", "ta.ema", "ta.rsi", "ta.macd", "ta.atr", "ta.stdev",
                "strategy.entry", "strategy.exit", "strategy.close", "strategy.close_all",
                "plot", "plotshape", "hline", "bgcolor", "fill", "alert"
            ],
            "forbidden_patterns": [
                r"security\s*\(",  # Avoid repainting
                r"request\.security",  # Avoid repainting
                r"varip\s+",  # Avoid varip in strategies
            ]
        }
    
    def validate_pine_script(self, pine_script_code: str) -> Tuple[bool, List[str]]:
        """
        Validiere Pine Script Code
        
        Args:
            pine_script_code: Pine Script Code zum Validieren
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        
        errors = []
        
        try:
            # 1. Version Check
            if not self._validate_version(pine_script_code):
                errors.append("Missing or invalid @version declaration")
            
            # 2. Strategy Declaration Check
            if not self._validate_strategy_declaration(pine_script_code):
                errors.append("Missing or invalid strategy() declaration")
            
            # 3. Syntax Check
            syntax_errors = self._validate_syntax(pine_script_code)
            errors.extend(syntax_errors)
            
            # 4. Function Usage Check
            function_errors = self._validate_functions(pine_script_code)
            errors.extend(function_errors)
            
            # 5. Forbidden Patterns Check
            forbidden_errors = self._validate_forbidden_patterns(pine_script_code)
            errors.extend(forbidden_errors)
            
            # 6. Logic Consistency Check
            logic_errors = self._validate_logic_consistency(pine_script_code)
            errors.extend(logic_errors)
            
            is_valid = len(errors) == 0
            
            if is_valid:
                self.logger.info("Pine Script validation successful")
            else:
                self.logger.warning(f"Pine Script validation failed with {len(errors)} errors")
            
            return is_valid, errors
            
        except Exception as e:
            self.logger.error(f"Pine Script validation error: {e}")
            return False, [f"Validation error: {e}"]
    
    def _validate_version(self, code: str) -> bool:
        """Validiere Version-Deklaration"""
        lines = code.split('\n')
        if not lines:
            return False
        
        first_line = lines[0].strip()
        return re.match(self.syntax_rules["version_declaration"], first_line) is not None
    
    def _validate_strategy_declaration(self, code: str) -> bool:
        """Validiere Strategy-Deklaration"""
        return re.search(self.syntax_rules["strategy_declaration"], code) is not None
    
    def _validate_syntax(self, code: str) -> List[str]:
        """Validiere grundlegende Syntax"""
        errors = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line or line.startswith('//'):
                continue
            
            # Check for common syntax errors
            if line.count('(') != line.count(')'):
                errors.append(f"Line {i}: Unmatched parentheses")
            
            if line.count('[') != line.count(']'):
                errors.append(f"Line {i}: Unmatched brackets")
            
            if line.count('{') != line.count('}'):
                errors.append(f"Line {i}: Unmatched braces")
        
        return errors
    
    def _validate_functions(self, code: str) -> List[str]:
        """Validiere Funktions-Verwendung"""
        errors = []
        
        # Check for deprecated functions (Pine Script v4 -> v5)
        deprecated_functions = {
            "study": "indicator",
            "security": "request.security",
            "rsi": "ta.rsi",
            "sma": "ta.sma",
            "ema": "ta.ema",
            "macd": "ta.macd",
            "atr": "ta.atr",
            "stdev": "ta.stdev",
            "crossover": "ta.crossover",
            "crossunder": "ta.crossunder"
        }
        
        for old_func, new_func in deprecated_functions.items():
            if re.search(rf'\b{old_func}\s*\(', code):
                errors.append(f"Deprecated function '{old_func}' found. Use '{new_func}' instead.")
        
        return errors
    
    def _validate_forbidden_patterns(self, code: str) -> List[str]:
        """Validiere verbotene Patterns"""
        errors = []
        
        for pattern in self.syntax_rules["forbidden_patterns"]:
            if re.search(pattern, code):
                errors.append(f"Forbidden pattern found: {pattern}")
        
        return errors
    
    def _validate_logic_consistency(self, code: str) -> List[str]:
        """Validiere logische Konsistenz"""
        errors = []
        
        # Check for strategy.entry without strategy.exit
        has_entry = re.search(r'strategy\.entry\s*\(', code)
        has_exit = re.search(r'strategy\.exit\s*\(', code)
        
        if has_entry and not has_exit:
            errors.append("strategy.entry found without corresponding strategy.exit")
        
        # Check for plot statements in strategy (should be minimal)
        plot_count = len(re.findall(r'plot\s*\(', code))
        if plot_count > 10:
            errors.append(f"Too many plot statements ({plot_count}). Consider reducing for performance.")
        
        return errors


class KIEnhancedPineScriptGenerator:
    """
    🧩 BAUSTEIN C1: KI-Enhanced Pine Script Generator
    
    Hauptklasse für KI-basierte Pine Script Code-Generierung:
    - Integration mit Baustein B3 (AI Strategy Evaluator)
    - Automatische Pine Script v5 Code-Generierung
    - KI-basierte Entry/Exit-Logik
    - Multimodale Konfidenz-Integration
    - Top-5-Strategien Pine Script Export
    """
    
    def __init__(
        self,
        ai_strategy_evaluator: Optional[AIStrategyEvaluator] = None,
        output_dir: str = "data/pine_scripts"
    ):
        """
        Initialize KI-Enhanced Pine Script Generator
        
        Args:
            ai_strategy_evaluator: AI Strategy Evaluator (Baustein B3)
            output_dir: Output-Verzeichnis für Pine Scripts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Komponenten initialisieren
        self.ai_evaluator = ai_strategy_evaluator or AIStrategyEvaluator()
        self.template_engine = PineScriptTemplateEngine()
        self.validator = PineScriptValidator()
        
        try:
            self.schema_manager = UnifiedSchemaManager(str(self.output_dir / "unified"))
        except:
            self.schema_manager = None
        
        # Performance Tracking
        self.total_generations = 0
        self.successful_generations = 0
        self.total_generation_time = 0.0
        
        # Generated Scripts Cache
        self.generated_scripts: List[GeneratedPineScript] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"KI-Enhanced Pine Script Generator initialized")
    
    def generate_top5_pine_scripts(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: PineScriptConfig = None
    ) -> List[GeneratedPineScript]:
        """
        Hauptfunktion: Generiere Pine Scripts für Top-5-Strategien
        
        Args:
            symbols: Liste der zu analysierenden Symbole
            timeframes: Liste der Zeitrahmen
            config: Pine Script Konfiguration
            
        Returns:
            Liste der generierten Pine Scripts
        """
        start_time = datetime.now()
        generation_start = start_time.timestamp()
        
        try:
            self.logger.info(f"🔄 Starting Top-5 Pine Script generation")
            
            # Default-Parameter
            symbols = symbols or ["EUR/USD", "GBP/USD", "USD/JPY"]
            timeframes = timeframes or ["1h", "4h"]
            config = config or PineScriptConfig()
            
            # 1. Top-5-Strategien von Baustein B3 abrufen
            top5_result = self.ai_evaluator.evaluate_and_rank_strategies(
                symbols=symbols,
                timeframes=timeframes,
                max_strategies=5,
                evaluation_mode="comprehensive"
            )
            
            # 2. Pine Scripts für jede Top-Strategie generieren
            generated_scripts = []
            
            for i, strategy_score in enumerate(top5_result.top_strategies):
                try:
                    # Pine Script generieren
                    pine_script = self._generate_single_pine_script(strategy_score, config)
                    generated_scripts.append(pine_script)
                    
                    self.logger.info(f"Generated Pine Script {i+1}/5: {pine_script.strategy_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate Pine Script {i+1}: {e}")
                    continue
            
            # 3. Performance Tracking
            generation_time = datetime.now().timestamp() - generation_start
            self.total_generations += 1
            self.successful_generations += 1 if generated_scripts else 0
            self.total_generation_time += generation_time
            
            # 4. Scripts speichern
            self._save_generated_scripts(generated_scripts)
            
            # 5. Cache aktualisieren
            self.generated_scripts.extend(generated_scripts)
            
            self.logger.info(f"✅ Pine Script generation completed in {generation_time:.3f}s - {len(generated_scripts)} scripts generated")
            
            return generated_scripts
            
        except Exception as e:
            generation_time = datetime.now().timestamp() - generation_start
            self.total_generations += 1
            self.total_generation_time += generation_time
            
            error_msg = f"Pine Script generation failed: {e}"
            self.logger.error(error_msg)
            
            return []
    
    def _generate_single_pine_script(
        self,
        strategy_score: StrategyScore,
        config: PineScriptConfig
    ) -> GeneratedPineScript:
        """Generiere einzelnen Pine Script"""
        
        try:
            # 1. Pine Script Code generieren
            pine_script_code = self.template_engine.generate_pine_script(strategy_score, config)
            
            # 2. Code validieren
            is_valid, validation_errors = self.validator.validate_pine_script(pine_script_code)
            
            # 3. Performance-Schätzungen
            estimated_performance = {
                "expected_return": strategy_score.expected_return,
                "expected_risk": strategy_score.expected_risk,
                "expected_sharpe": strategy_score.expected_sharpe,
                "composite_score": strategy_score.composite_score
            }
            
            # 4. Code-Statistiken
            code_lines = len(pine_script_code.split('\n'))
            code_complexity = self._assess_code_complexity(pine_script_code)
            
            # 5. GeneratedPineScript erstellen
            generated_script = GeneratedPineScript(
                strategy_id=strategy_score.strategy_id,
                strategy_name=f"AI_Strategy_{strategy_score.symbol.replace('/', '_')}_{strategy_score.timeframe}",
                symbol=strategy_score.symbol,
                timeframe=strategy_score.timeframe,
                pine_script_code=pine_script_code,
                generation_timestamp=datetime.now(),
                strategy_score=strategy_score,
                config=config,
                syntax_valid=is_valid,
                validation_errors=validation_errors,
                estimated_performance=estimated_performance,
                code_lines=code_lines,
                code_complexity=code_complexity
            )
            
            return generated_script
            
        except Exception as e:
            self.logger.error(f"Single Pine Script generation failed: {e}")
            
            # Fallback-Script
            return GeneratedPineScript(
                strategy_id=strategy_score.strategy_id,
                strategy_name=f"AI_Fallback_{strategy_score.symbol.replace('/', '_')}",
                symbol=strategy_score.symbol,
                timeframe=strategy_score.timeframe,
                pine_script_code=self.template_engine._get_fallback_pine_script(strategy_score, config),
                generation_timestamp=datetime.now(),
                strategy_score=strategy_score,
                config=config,
                syntax_valid=False,
                validation_errors=[f"Generation error: {e}"],
                estimated_performance={},
                code_lines=0,
                code_complexity="simple"
            )
    
    def _assess_code_complexity(self, pine_script_code: str) -> str:
        """Bewerte Code-Komplexität"""
        
        lines = len(pine_script_code.split('\n'))
        functions = len(re.findall(r'\w+\s*\(', pine_script_code))
        conditions = len(re.findall(r'\bif\b|\belse\b|\belseif\b', pine_script_code))
        
        complexity_score = lines * 0.1 + functions * 0.5 + conditions * 1.0
        
        if complexity_score < 20:
            return "simple"
        elif complexity_score < 50:
            return "medium"
        else:
            return "complex"
    
    def _save_generated_scripts(self, generated_scripts: List[GeneratedPineScript]):
        """Speichere generierte Pine Scripts"""
        
        try:
            for script in generated_scripts:
                # 1. Pine Script Datei speichern
                filename = f"{script.strategy_name}_{script.generation_timestamp.strftime('%Y%m%d_%H%M%S')}.pine"
                file_path = self.output_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(script.pine_script_code)
                
                script.file_path = str(file_path)
                script.export_timestamp = datetime.now()
                
                # 2. Metadaten speichern
                metadata = {
                    "strategy_id": script.strategy_id,
                    "strategy_name": script.strategy_name,
                    "symbol": script.symbol,
                    "timeframe": script.timeframe,
                    "generation_timestamp": script.generation_timestamp.isoformat(),
                    "syntax_valid": script.syntax_valid,
                    "validation_errors": script.validation_errors,
                    "estimated_performance": script.estimated_performance,
                    "code_lines": script.code_lines,
                    "code_complexity": script.code_complexity,
                    "file_path": script.file_path
                }
                
                metadata_file = self.output_dir / f"{script.strategy_name}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                # 3. Schema Manager (falls verfügbar)
                if self.schema_manager:
                    generation_data = {
                        "timestamp": script.generation_timestamp,
                        "component": "KIEnhancedPineScriptGenerator",
                        "operation": "pine_script_generation",
                        "strategy_id": script.strategy_id,
                        "symbol": script.symbol,
                        "timeframe": script.timeframe,
                        "syntax_valid": script.syntax_valid,
                        "code_lines": script.code_lines,
                        "code_complexity": script.code_complexity,
                        "composite_score": script.strategy_score.composite_score,
                        "file_path": script.file_path
                    }
                    self.schema_manager.write_to_stream(generation_data, DataStreamType.AI_PREDICTIONS)
            
            self.logger.info(f"Saved {len(generated_scripts)} Pine Scripts to {self.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save Pine Scripts: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        
        ai_evaluator_stats = self.ai_evaluator.get_performance_stats()
        
        return {
            "generator_stats": {
                "total_generations": self.total_generations,
                "successful_generations": self.successful_generations,
                "success_rate": self.successful_generations / max(1, self.total_generations),
                "total_generation_time": self.total_generation_time,
                "average_generation_time": self.total_generation_time / max(1, self.total_generations),
                "generations_per_minute": (self.total_generations / self.total_generation_time * 60) if self.total_generation_time > 0 else 0,
                "scripts_in_cache": len(self.generated_scripts)
            },
            "ai_evaluator_stats": ai_evaluator_stats
        }
    
    def export_all_scripts_summary(self) -> str:
        """Exportiere Zusammenfassung aller generierten Scripts"""
        
        if not self.generated_scripts:
            return "No Pine Scripts generated yet."
        
        summary_lines = [
            "🧩 KI-Enhanced Pine Script Generator - Summary Report",
            "=" * 70,
            f"Generated Scripts: {len(self.generated_scripts)}",
            f"Total Generations: {self.total_generations}",
            f"Success Rate: {self.successful_generations / max(1, self.total_generations):.1%}",
            "",
            "📊 Generated Scripts Overview:",
            ""
        ]
        
        for i, script in enumerate(self.generated_scripts, 1):
            summary_lines.extend([
                f"{i}. {script.strategy_name}",
                f"   Symbol: {script.symbol} | Timeframe: {script.timeframe}",
                f"   Syntax Valid: {'✅' if script.syntax_valid else '❌'}",
                f"   Code Lines: {script.code_lines} | Complexity: {script.code_complexity}",
                f"   Expected Return: {script.estimated_performance.get('expected_return', 0):.1%}",
                f"   Composite Score: {script.estimated_performance.get('composite_score', 0):.3f}",
                f"   File: {script.file_path or 'Not saved'}",
                ""
            ])
        
        return "\n".join(summary_lines)


def demo_ki_enhanced_pine_script_generator():
    """
    🧩 Demo für KI-Enhanced Pine Script Generator (Baustein C1)
    """
    print("🧩 BAUSTEIN C1: KI-ENHANCED PINE SCRIPT GENERATOR DEMO")
    print("=" * 70)
    
    # Erstelle KI-Enhanced Pine Script Generator
    generator = KIEnhancedPineScriptGenerator()
    
    try:
        # Pine Script Konfiguration
        config = PineScriptConfig(
            version=PineScriptVersion.V5,
            strategy_type=StrategyType.SWING_TRADING,
            risk_management=RiskManagementType.ATR_BASED,
            initial_capital=10000.0,
            use_confidence_filtering=True,
            min_confidence_threshold=0.6,
            use_multimodal_confirmation=True,
            include_alerts=True,
            include_plotting=True
        )
        
        print("🔄 Generating Top-5 Pine Scripts...")
        
        # Top-5 Pine Scripts generieren
        generated_scripts = generator.generate_top5_pine_scripts(
            symbols=["EUR/USD", "GBP/USD"],
            timeframes=["1h", "4h"],
            config=config
        )
        
        print(f"\n📊 PINE SCRIPT GENERATION RESULTS:")
        print(f"Scripts Generated: {len(generated_scripts)}")
        
        for i, script in enumerate(generated_scripts, 1):
            print(f"\n🏆 SCRIPT #{i}: {script.strategy_name}")
            print(f"   Symbol: {script.symbol} | Timeframe: {script.timeframe}")
            print(f"   Syntax Valid: {'✅' if script.syntax_valid else '❌'}")
            print(f"   Code Lines: {script.code_lines}")
            print(f"   Complexity: {script.code_complexity}")
            print(f"   Expected Return: {script.estimated_performance.get('expected_return', 0):.1%}")
            print(f"   Expected Risk: {script.estimated_performance.get('expected_risk', 0):.1%}")
            print(f"   Composite Score: {script.estimated_performance.get('composite_score', 0):.3f}")
            
            if script.validation_errors:
                print(f"   ⚠️  Validation Issues: {len(script.validation_errors)}")
                for error in script.validation_errors[:2]:  # Show first 2 errors
                    print(f"      • {error}")
            
            if script.file_path:
                print(f"   📄 Saved to: {script.file_path}")
        
        # Performance Stats
        print(f"\n📊 GENERATOR PERFORMANCE STATS:")
        stats = generator.get_performance_stats()
        generator_stats = stats["generator_stats"]
        print(f"   Total Generations: {generator_stats['total_generations']}")
        print(f"   Success Rate: {generator_stats['success_rate']:.1%}")
        print(f"   Avg Generation Time: {generator_stats['average_generation_time']:.3f}s")
        print(f"   Generations/min: {generator_stats['generations_per_minute']:.1f}")
        
        # Zeige Beispiel Pine Script Code (erste 30 Zeilen)
        if generated_scripts:
            print(f"\n📜 EXAMPLE PINE SCRIPT CODE (First 30 lines):")
            print("=" * 50)
            example_code = generated_scripts[0].pine_script_code
            example_lines = example_code.split('\n')[:30]
            for line_num, line in enumerate(example_lines, 1):
                print(f"{line_num:2d}: {line}")
            
            if len(example_code.split('\n')) > 30:
                print(f"... ({len(example_code.split('\n')) - 30} more lines)")
        
        # Summary Export
        summary = generator.export_all_scripts_summary()
        summary_file = generator.output_dir / "generation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n📄 Summary saved to: {summary_file}")
        print(f"\n✅ BAUSTEIN C1 DEMO COMPLETED SUCCESSFULLY!")
        
        return generated_scripts
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run Demo
    demo_ki_enhanced_pine_script_generator()