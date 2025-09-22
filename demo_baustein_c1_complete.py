#!/usr/bin/env python3
"""
🧩 BAUSTEIN C1 DEMO: KI-Enhanced Pine Script Generator
Vollständige Demo für automatische Pine Script Code-Generierung

Demonstriert:
- Integration mit Baustein B3 (AI Strategy Evaluator)
- Automatische Pine Script v5 Code-Generierung
- KI-basierte Entry/Exit-Logik
- Multimodale Konfidenz-Integration
- Top-5-Strategien Pine Script Export
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

print("🧩 BAUSTEIN C1: KI-ENHANCED PINE SCRIPT GENERATOR DEMO")
print("=" * 70)
print(f"Start Time: {datetime.now()}")
print()

# Mock-Klassen für Demo
class PineScriptVersion(Enum):
    V4 = "4"
    V5 = "5"

class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"

class RiskManagementType(Enum):
    FIXED_STOP_LOSS = "fixed_stop_loss"
    ATR_BASED = "atr_based"
    PERCENTAGE_BASED = "percentage_based"
    DYNAMIC_TRAILING = "dynamic_trailing"

@dataclass
class MockStrategyScore:
    strategy_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    signal_confidence_score: float
    risk_reward_score: float
    opportunity_score: float
    fusion_confidence_score: float
    consistency_score: float
    profit_potential_score: float
    drawdown_risk_score: float
    composite_score: float
    weighted_score: float
    rank_position: int = 0
    expected_return: float = 0.0
    expected_risk: float = 0.0
    expected_sharpe: float = 0.0

@dataclass
class PineScriptConfig:
    version: PineScriptVersion = PineScriptVersion.V5
    strategy_type: StrategyType = StrategyType.SWING_TRADING
    risk_management: RiskManagementType = RiskManagementType.ATR_BASED
    initial_capital: float = 10000.0
    default_qty_type: str = "strategy.percent_of_equity"
    default_qty_value: float = 10.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    max_risk_per_trade: float = 0.02
    include_alerts: bool = True
    include_plotting: bool = True
    include_backtesting: bool = True
    use_confidence_filtering: bool = True
    min_confidence_threshold: float = 0.6
    use_multimodal_confirmation: bool = True

@dataclass
class GeneratedPineScript:
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    pine_script_code: str
    generation_timestamp: datetime
    strategy_score: MockStrategyScore
    config: PineScriptConfig
    syntax_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    estimated_performance: Dict[str, float] = field(default_factory=dict)
    code_lines: int = 0
    code_complexity: str = "medium"
    file_path: Optional[str] = None
    export_timestamp: Optional[datetime] = None

class PineScriptTemplateEngine:
    """Template-Engine für Pine Script Code-Generierung"""
    
    def __init__(self):
        self.logger = None
    
    def generate_pine_script(self, strategy_score: MockStrategyScore, config: PineScriptConfig) -> str:
        """Generiere vollständigen Pine Script Code"""
        
        # Template-Variablen vorbereiten
        template_vars = {
            "strategy_name": f"AI_Strategy_{strategy_score.symbol.replace('/', '_')}_{strategy_score.timeframe}",
            "strategy_id": strategy_score.strategy_id,
            "symbol": strategy_score.symbol,
            "timeframe": strategy_score.timeframe,
            "timestamp": strategy_score.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "signal_confidence": strategy_score.signal_confidence_score,
            "risk_reward_ratio": strategy_score.risk_reward_score,
            "opportunity_score": strategy_score.opportunity_score,
            "fusion_confidence": strategy_score.fusion_confidence_score,
            "composite_score": strategy_score.composite_score,
            "weighted_score": strategy_score.weighted_score,
            "expected_return": strategy_score.expected_return,
            "expected_risk": strategy_score.expected_risk,
            "expected_sharpe": strategy_score.expected_sharpe,
            "initial_capital": config.initial_capital,
            "qty_type": config.default_qty_type,
            "qty_value": config.default_qty_value,
            "stop_loss_atr": config.stop_loss_atr_multiplier,
            "take_profit_atr": config.take_profit_atr_multiplier,
            "max_risk": config.max_risk_per_trade,
            "min_confidence": config.min_confidence_threshold,
            "use_multimodal": config.use_multimodal_confirmation,
            "pine_version": config.version.value
        }
        
        # Pine Script Code generieren
        pine_script_code = f'''//@version={template_vars["pine_version"]}
strategy("{template_vars["strategy_name"]}", 
         shorttitle="{template_vars["symbol"]}_{template_vars["timeframe"]}", 
         overlay=true,
         initial_capital={template_vars["initial_capital"]},
         default_qty_type={template_vars["qty_type"]},
         default_qty_value={template_vars["qty_value"]},
         commission_type=strategy.commission.percent,
         commission_value=0.1)

// 🧩 AI-Enhanced Trading Strategy
// Generated by: KI-Enhanced Pine Script Generator (Baustein C1)
// Strategy ID: {template_vars["strategy_id"]}
// Symbol: {template_vars["symbol"]}
// Timeframe: {template_vars["timeframe"]}
// Generated: {template_vars["timestamp"]}
//
// 📊 AI Performance Metrics:
// Signal Confidence: {template_vars["signal_confidence"]:.1%}
// Risk/Reward Score: {template_vars["risk_reward_ratio"]:.3f}
// Opportunity Score: {template_vars["opportunity_score"]:.1%}
// Fusion Confidence: {template_vars["fusion_confidence"]:.1%}
// Composite Score: {template_vars["composite_score"]:.3f}
// Weighted Score: {template_vars["weighted_score"]:.3f}
//
// 📈 Expected Performance:
// Expected Return: {template_vars["expected_return"]:.1%}
// Expected Risk: {template_vars["expected_risk"]:.1%}
// Expected Sharpe: {template_vars["expected_sharpe"]:.2f}

// 🎛️ Input Parameters
// AI Configuration
ai_confidence_threshold = input.float({template_vars["min_confidence"]}, "AI Confidence Threshold", minval=0.0, maxval=1.0, step=0.01, group="AI Settings")
use_multimodal_confirmation = input.bool({str(template_vars["use_multimodal"]).lower()}, "Use Multimodal Confirmation", group="AI Settings")

// Risk Management
stop_loss_atr_mult = input.float({template_vars["stop_loss_atr"]}, "Stop Loss ATR Multiplier", minval=0.5, maxval=5.0, step=0.1, group="Risk Management")
take_profit_atr_mult = input.float({template_vars["take_profit_atr"]}, "Take Profit ATR Multiplier", minval=1.0, maxval=10.0, step=0.1, group="Risk Management")
max_risk_per_trade = input.float({template_vars["max_risk"] * 100}, "Max Risk Per Trade (%)", minval=0.5, maxval=10.0, step=0.1, group="Risk Management") / 100

// Technical Indicators
rsi_length = input.int(14, "RSI Length", minval=5, maxval=50, group="Technical Indicators")
ema_fast = input.int(12, "EMA Fast", minval=5, maxval=50, group="Technical Indicators")
ema_slow = input.int(26, "EMA Slow", minval=20, maxval=100, group="Technical Indicators")
atr_length = input.int(14, "ATR Length", minval=5, maxval=50, group="Technical Indicators")

// Strategy Settings
enable_long = input.bool(true, "Enable Long Trades", group="Strategy Settings")
enable_short = input.bool(true, "Enable Short Trades", group="Strategy Settings")

// 📊 Technical Indicators
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
ai_confidence_score = {template_vars["signal_confidence"]:.3f}
multimodal_confidence = {template_vars["fusion_confidence"]:.3f}

// Combined AI Signal Strength
combined_ai_signal = (ai_confidence_score + multimodal_confidence + technical_confluence) / 3.0

// 🎯 AI-Enhanced Strategy Logic
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
    strategy.entry("AI_Short", strategy.short, qty=position_size)

// 🛡️ AI-Enhanced Risk Management
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
    strategy.close_all(comment="Emergency Drawdown Protection")

// 🔔 AI-Enhanced Alerts
// Entry Alerts
if long_condition and strategy.position_size == 0
    alert("AI Long Entry Signal\\n" + 
          "Symbol: {template_vars["symbol"]}\\n" +
          "Confidence: " + str.tostring(combined_ai_signal, "#.###") + "\\n" +
          "Price: " + str.tostring(close, "#.#####"), 
          alert.freq_once_per_bar)

if short_condition and strategy.position_size == 0
    alert("AI Short Entry Signal\\n" + 
          "Symbol: {template_vars["symbol"]}\\n" +
          "Confidence: " + str.tostring(combined_ai_signal, "#.###") + "\\n" +
          "Price: " + str.tostring(close, "#.#####"), 
          alert.freq_once_per_bar)

// Risk Management Alerts
if current_drawdown > max_drawdown_threshold * 0.8  // 80% of max drawdown
    alert("Risk Warning: High Drawdown\\n" + 
          "Current Drawdown: " + str.tostring(current_drawdown * 100, "#.##") + "%\\n" +
          "Threshold: " + str.tostring(max_drawdown_threshold * 100, "#.##") + "%", 
          alert.freq_once_per_bar)

// 📊 AI-Enhanced Plotting
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
    table.cell(info_table, 1, 0, "{template_vars["strategy_name"]}", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 1, "Signal Confidence", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 1, "{template_vars["signal_confidence"]:.1%}", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 2, "Expected Return", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 2, "{template_vars["expected_return"]:.1%}", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 3, "Expected Risk", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 3, "{template_vars["expected_risk"]:.1%}", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 4, "Composite Score", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 4, "{template_vars["composite_score"]:.3f}", text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 5, "Current AI Signal", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 5, str.tostring(combined_ai_signal, "#.###"), text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 6, "Position Size", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 6, str.tostring(strategy.position_size, "#.##"), text_color=color.black, text_size=size.small)
    table.cell(info_table, 0, 7, "P&L", text_color=color.black, text_size=size.small)
    table.cell(info_table, 1, 7, str.tostring(strategy.netprofit, "#.##"), 
               text_color=strategy.netprofit >= 0 ? color.green : color.red, text_size=size.small)'''
        
        return pine_script_code

class PineScriptValidator:
    """Pine Script Syntax-Validator"""
    
    def __init__(self):
        pass
    
    def validate_pine_script(self, pine_script_code: str) -> Tuple[bool, List[str]]:
        """Validiere Pine Script Code"""
        
        errors = []
        
        # Basis-Validierungen
        if not pine_script_code.strip():
            errors.append("Empty Pine Script code")
            return False, errors
        
        lines = pine_script_code.split('\n')
        
        # Version Check
        if not lines[0].strip().startswith('//@version='):
            errors.append("Missing @version declaration")
        
        # Strategy Declaration Check
        if 'strategy(' not in pine_script_code:
            errors.append("Missing strategy() declaration")
        
        # Syntax Check (vereinfacht)
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Check for unmatched parentheses
            if line.count('(') != line.count(')'):
                errors.append(f"Line {i}: Unmatched parentheses")
        
        # Check for deprecated functions
        deprecated_functions = ['security(', 'study(', 'rsi(', 'sma(', 'ema(']
        for func in deprecated_functions:
            if func in pine_script_code:
                errors.append(f"Deprecated function found: {func}")
        
        is_valid = len(errors) == 0
        return is_valid, errors

class MockKIEnhancedPineScriptGenerator:
    """Mock KI-Enhanced Pine Script Generator für Demo"""
    
    def __init__(self, output_dir: str = "data/pine_scripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.template_engine = PineScriptTemplateEngine()
        self.validator = PineScriptValidator()
        
        self.total_generations = 0
        self.successful_generations = 0
        self.total_generation_time = 0.0
        self.generated_scripts = []
    
    def create_mock_strategy_scores(self, symbols: List[str], timeframes: List[str]) -> List[MockStrategyScore]:
        """Erstelle Mock-Strategien-Scores"""
        
        strategy_scores = []
        
        for i, symbol in enumerate(symbols):
            for j, timeframe in enumerate(timeframes):
                base_score = 0.6 + (i * 0.1) + (j * 0.05)
                
                strategy_score = MockStrategyScore(
                    strategy_id=f"strategy_{i+1}_{j+1}_{symbol}_{timeframe}",
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    signal_confidence_score=base_score + 0.1,
                    risk_reward_score=base_score + 0.05,
                    opportunity_score=base_score,
                    fusion_confidence_score=base_score + 0.08,
                    consistency_score=base_score + 0.03,
                    profit_potential_score=base_score + 0.07,
                    drawdown_risk_score=1.0 - base_score,  # Inverted
                    composite_score=base_score,
                    weighted_score=base_score + 0.02,
                    rank_position=i * len(timeframes) + j + 1,
                    expected_return=base_score * 0.15,  # Up to 15%
                    expected_risk=0.08 + (1.0 - base_score) * 0.07,  # 8-15%
                    expected_sharpe=base_score * 2.5  # Up to 2.5
                )
                
                strategy_scores.append(strategy_score)
        
        # Sort by weighted_score (descending) and take top 5
        strategy_scores.sort(key=lambda x: x.weighted_score, reverse=True)
        return strategy_scores[:5]
    
    def generate_top5_pine_scripts(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: PineScriptConfig = None
    ) -> List[GeneratedPineScript]:
        """Generiere Pine Scripts für Top-5-Strategien"""
        
        start_time = time.time()
        
        try:
            symbols = symbols or ["EUR/USD", "GBP/USD", "USD/JPY"]
            timeframes = timeframes or ["1h", "4h"]
            config = config or PineScriptConfig()
            
            # Mock Top-5-Strategien erstellen
            top5_strategies = self.create_mock_strategy_scores(symbols, timeframes)
            
            generated_scripts = []
            
            for strategy_score in top5_strategies:
                # Pine Script generieren
                pine_script_code = self.template_engine.generate_pine_script(strategy_score, config)
                
                # Code validieren
                is_valid, validation_errors = self.validator.validate_pine_script(pine_script_code)
                
                # Performance-Schätzungen
                estimated_performance = {
                    "expected_return": strategy_score.expected_return,
                    "expected_risk": strategy_score.expected_risk,
                    "expected_sharpe": strategy_score.expected_sharpe,
                    "composite_score": strategy_score.composite_score
                }
                
                # Code-Statistiken
                code_lines = len(pine_script_code.split('\n'))
                code_complexity = "complex" if code_lines > 200 else "medium" if code_lines > 100 else "simple"
                
                # GeneratedPineScript erstellen
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
                
                generated_scripts.append(generated_script)
            
            # Performance Tracking
            generation_time = time.time() - start_time
            self.total_generations += 1
            self.successful_generations += 1 if generated_scripts else 0
            self.total_generation_time += generation_time
            
            # Scripts speichern
            self._save_generated_scripts(generated_scripts)
            
            self.generated_scripts.extend(generated_scripts)
            
            return generated_scripts
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.total_generations += 1
            self.total_generation_time += generation_time
            print(f"Generation failed: {e}")
            return []
    
    def _save_generated_scripts(self, generated_scripts: List[GeneratedPineScript]):
        """Speichere generierte Pine Scripts"""
        
        try:
            for script in generated_scripts:
                # Pine Script Datei speichern
                filename = f"{script.strategy_name}_{script.generation_timestamp.strftime('%Y%m%d_%H%M%S')}.pine"
                file_path = self.output_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(script.pine_script_code)
                
                script.file_path = str(file_path)
                script.export_timestamp = datetime.now()
                
                # Metadaten speichern
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
            
        except Exception as e:
            print(f"Failed to save Pine Scripts: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        
        return {
            "generator_stats": {
                "total_generations": self.total_generations,
                "successful_generations": self.successful_generations,
                "success_rate": self.successful_generations / max(1, self.total_generations),
                "total_generation_time": self.total_generation_time,
                "average_generation_time": self.total_generation_time / max(1, self.total_generations),
                "generations_per_minute": (self.total_generations / self.total_generation_time * 60) if self.total_generation_time > 0 else 0,
                "scripts_in_cache": len(self.generated_scripts)
            }
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
    """Hauptdemo für Baustein C1"""
    
    print("🔄 Initializing KI-Enhanced Pine Script Generator...")
    
    # Erstelle Generator
    generator = MockKIEnhancedPineScriptGenerator()
    
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

if __name__ == "__main__":
    try:
        start_time = time.time()
        
        # Run Demo
        generated_scripts = demo_ki_enhanced_pine_script_generator()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️  Total Processing Time: {processing_time:.3f} seconds")
        print(f"📊 Scripts per Second: {len(generated_scripts)/processing_time:.1f}")
        
        print(f"\n🎯 BAUSTEIN C1 READY FOR INTEGRATION!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()