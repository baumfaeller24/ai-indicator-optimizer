"""
Community Strategy Importer für externe Trading-Strategien
Importiert Strategien aus verschiedenen Quellen und Formaten
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
import re
from urllib.parse import urlparse
import io

from .historical_pattern_miner import MinedPattern


@dataclass
class StrategySource:
    """Quelle für Trading-Strategien"""
    name: str
    url: str
    format: str  # "json", "xml", "csv", "pine_script", "mql4"
    auth_required: bool = False
    api_key: Optional[str] = None
    rate_limit: int = 60  # Requests per minute
    
    
@dataclass
class ImportedStrategy:
    """Importierte Trading-Strategie"""
    strategy_id: str
    name: str
    description: str
    author: str
    source: str
    strategy_type: str  # "indicator", "signal", "complete_system"
    
    # Strategy Logic
    entry_conditions: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Data
    backtest_results: Optional[Dict[str, Any]] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    # Metadata
    timeframes: List[str] = field(default_factory=lambda: ["1H"])
    symbols: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Raw Data
    raw_code: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return asdict(self)


class PineScriptParser:
    """Parser für Pine Script Strategien"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_pine_script(self, pine_code: str) -> ImportedStrategy:
        """Parsed Pine Script Code"""
        
        try:
            # Extract Strategy Info
            strategy_info = self._extract_strategy_info(pine_code)
            
            # Extract Conditions
            entry_conditions = self._extract_entry_conditions(pine_code)
            exit_conditions = self._extract_exit_conditions(pine_code)
            
            # Extract Risk Management
            risk_management = self._extract_risk_management(pine_code)
            
            return ImportedStrategy(
                strategy_id=f"pine_{hash(pine_code) % 1000000}",
                name=strategy_info.get("title", "Unknown Pine Strategy"),
                description=strategy_info.get("description", ""),
                author=strategy_info.get("author", "Unknown"),
                source="pine_script",
                strategy_type="complete_system",
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                timeframes=strategy_info.get("timeframes", ["1H"]),
                symbols=strategy_info.get("symbols", ["EURUSD"]),
                created_at=datetime.now(),
                raw_code=pine_code
            )
            
        except Exception as e:
            self.logger.exception(f"Pine Script parsing failed: {e}")
            raise
    
    def _extract_strategy_info(self, pine_code: str) -> Dict[str, Any]:
        """Extrahiert Strategy-Informationen"""
        
        info = {}
        
        # Title
        title_match = re.search(r'strategy\s*\(\s*["\']([^"\']+)["\']', pine_code)
        if title_match:
            info["title"] = title_match.group(1)
        
        # Comments für Description
        comment_lines = re.findall(r'//\s*(.+)', pine_code)
        if comment_lines:
            info["description"] = " ".join(comment_lines[:3])  # Erste 3 Kommentare
        
        return info
    
    def _extract_entry_conditions(self, pine_code: str) -> List[str]:
        """Extrahiert Entry-Conditions"""
        
        conditions = []
        
        # Strategy Entry Patterns
        entry_patterns = [
            r'strategy\.entry\s*\([^)]+when\s*=\s*([^,)]+)',
            r'if\s+([^:]+):\s*strategy\.entry',
            r'longCondition\s*=\s*([^\n]+)',
            r'shortCondition\s*=\s*([^\n]+)'
        ]
        
        for pattern in entry_patterns:
            matches = re.findall(pattern, pine_code, re.MULTILINE)
            conditions.extend(matches)
        
        # Clean up conditions
        cleaned_conditions = []
        for condition in conditions:
            cleaned = condition.strip().replace('\n', ' ')
            if cleaned and len(cleaned) > 5:
                cleaned_conditions.append(cleaned)
        
        return cleaned_conditions[:10]  # Limit to 10 conditions
    
    def _extract_exit_conditions(self, pine_code: str) -> List[str]:
        """Extrahiert Exit-Conditions"""
        
        conditions = []
        
        # Strategy Exit Patterns
        exit_patterns = [
            r'strategy\.exit\s*\([^)]+when\s*=\s*([^,)]+)',
            r'strategy\.close\s*\([^)]+when\s*=\s*([^,)]+)',
            r'exitCondition\s*=\s*([^\n]+)'
        ]
        
        for pattern in exit_patterns:
            matches = re.findall(pattern, pine_code, re.MULTILINE)
            conditions.extend(matches)
        
        # Clean up
        cleaned_conditions = []
        for condition in conditions:
            cleaned = condition.strip().replace('\n', ' ')
            if cleaned and len(cleaned) > 5:
                cleaned_conditions.append(cleaned)
        
        return cleaned_conditions[:10]
    
    def _extract_risk_management(self, pine_code: str) -> Dict[str, Any]:
        """Extrahiert Risk Management Parameter"""
        
        risk_mgmt = {}
        
        # Stop Loss
        sl_patterns = [
            r'stop_loss\s*=\s*([0-9.]+)',
            r'stopLoss\s*=\s*([0-9.]+)',
            r'sl\s*=\s*([0-9.]+)'
        ]
        
        for pattern in sl_patterns:
            match = re.search(pattern, pine_code)
            if match:
                risk_mgmt["stop_loss"] = float(match.group(1))
                break
        
        # Take Profit
        tp_patterns = [
            r'take_profit\s*=\s*([0-9.]+)',
            r'takeProfit\s*=\s*([0-9.]+)',
            r'tp\s*=\s*([0-9.]+)'
        ]
        
        for pattern in tp_patterns:
            match = re.search(pattern, pine_code)
            if match:
                risk_mgmt["take_profit"] = float(match.group(1))
                break
        
        # Position Size
        size_patterns = [
            r'qty\s*=\s*([0-9.]+)',
            r'quantity\s*=\s*([0-9.]+)',
            r'size\s*=\s*([0-9.]+)'
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, pine_code)
            if match:
                risk_mgmt["position_size"] = float(match.group(1))
                break
        
        return risk_mgmt


class CommunityStrategyImporter:
    """
    Hauptklasse für Community Strategy Import
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Parsers
        self.pine_parser = PineScriptParser()
        
        # Strategy Sources
        self.strategy_sources = self._setup_strategy_sources()
        
        # Import Statistics
        self.import_stats = {
            "total_imported": 0,
            "successful_imports": 0,
            "failed_imports": 0,
            "sources_processed": 0
        }
        
        self.logger.info("CommunityStrategyImporter initialized")
    
    def _setup_strategy_sources(self) -> List[StrategySource]:
        """Setup verfügbarer Strategy-Quellen"""
        
        sources = [
            # TradingView Public Library (simuliert)
            StrategySource(
                name="TradingView Public",
                url="https://www.tradingview.com/scripts/",
                format="pine_script",
                auth_required=False,
                rate_limit=30
            ),
            
            # GitHub Trading Repositories
            StrategySource(
                name="GitHub Trading",
                url="https://api.github.com/search/repositories",
                format="json",
                auth_required=False,
                rate_limit=60
            ),
            
            # QuantConnect Community
            StrategySource(
                name="QuantConnect",
                url="https://www.quantconnect.com/api/",
                format="json",
                auth_required=True,
                rate_limit=100
            ),
            
            # Local File Sources
            StrategySource(
                name="Local Files",
                url="./community_strategies/",
                format="mixed",
                auth_required=False,
                rate_limit=1000
            )
        ]
        
        return sources
    
    def import_strategies_from_source(self, source_name: str) -> List[ImportedStrategy]:
        """Importiert Strategien aus spezifischer Quelle"""
        
        source = next((s for s in self.strategy_sources if s.name == source_name), None)
        if not source:
            self.logger.error(f"Unknown source: {source_name}")
            return []
        
        try:
            if source.format == "pine_script":
                return self._import_pine_scripts(source)
            elif source.format == "json":
                return self._import_json_strategies(source)
            elif source.format == "mixed":
                return self._import_local_files(source)
            else:
                self.logger.warning(f"Unsupported format: {source.format}")
                return []
                
        except Exception as e:
            self.logger.exception(f"Import from {source_name} failed: {e}")
            return []
    
    def _import_pine_scripts(self, source: StrategySource) -> List[ImportedStrategy]:
        """Importiert Pine Script Strategien"""
        
        strategies = []
        
        try:
            # Simuliere Pine Script Import (echte Implementation würde Web Scraping nutzen)
            sample_pine_scripts = self._get_sample_pine_scripts()
            
            for pine_code in sample_pine_scripts:
                try:
                    strategy = self.pine_parser.parse_pine_script(pine_code)
                    strategy.source = source.name
                    strategies.append(strategy)
                    
                except Exception as e:
                    self.logger.exception(f"Pine script parsing failed: {e}")
                    self.import_stats["failed_imports"] += 1
            
            self.import_stats["successful_imports"] += len(strategies)
            self.logger.info(f"Imported {len(strategies)} Pine Script strategies")
            
        except Exception as e:
            self.logger.exception(f"Pine script import failed: {e}")
        
        return strategies
    
    def _get_sample_pine_scripts(self) -> List[str]:
        """Gibt Sample Pine Scripts zurück (für Demo)"""
        
        return [
            '''
//@version=5
strategy("RSI Strategy", overlay=true)

// Parameters
rsi_length = input(14, "RSI Length")
rsi_overbought = input(70, "RSI Overbought")
rsi_oversold = input(30, "RSI Oversold")

// Calculate RSI
rsi = ta.rsi(close, rsi_length)

// Entry Conditions
longCondition = rsi < rsi_oversold
shortCondition = rsi > rsi_overbought

// Strategy Entries
if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.entry("Short", strategy.short)

// Exit Conditions
if (rsi > 50)
    strategy.close("Long")

if (rsi < 50)
    strategy.close("Short")
            ''',
            
            '''
//@version=5
strategy("MACD Crossover", overlay=false)

// MACD Parameters
fast_length = input(12, "Fast Length")
slow_length = input(26, "Slow Length")
signal_length = input(9, "Signal Length")

// Calculate MACD
[macd, signal, hist] = ta.macd(close, fast_length, slow_length, signal_length)

// Entry Conditions
longCondition = ta.crossover(macd, signal)
shortCondition = ta.crossunder(macd, signal)

// Risk Management
stop_loss = input(0.02, "Stop Loss %") / 100
take_profit = input(0.04, "Take Profit %") / 100

// Strategy Entries
if (longCondition)
    strategy.entry("Long", strategy.long)
    strategy.exit("Long Exit", "Long", stop=close*(1-stop_loss), limit=close*(1+take_profit))

if (shortCondition)
    strategy.entry("Short", strategy.short)
    strategy.exit("Short Exit", "Short", stop=close*(1+stop_loss), limit=close*(1-take_profit))
            ''',
            
            '''
//@version=5
strategy("Bollinger Bands Mean Reversion", overlay=true)

// BB Parameters
bb_length = input(20, "BB Length")
bb_mult = input(2.0, "BB Multiplier")

// Calculate Bollinger Bands
[middle, upper, lower] = ta.bb(close, bb_length, bb_mult)

// Entry Conditions
longCondition = close < lower and close[1] >= lower[1]
shortCondition = close > upper and close[1] <= upper[1]

// Exit Conditions
longExit = close > middle
shortExit = close < middle

// Strategy Logic
if (longCondition)
    strategy.entry("Long", strategy.long)

if (shortCondition)
    strategy.entry("Short", strategy.short)

if (longExit)
    strategy.close("Long")

if (shortExit)
    strategy.close("Short")
            '''
        ]
    
    def _import_json_strategies(self, source: StrategySource) -> List[ImportedStrategy]:
        """Importiert JSON-basierte Strategien"""
        
        strategies = []
        
        try:
            # Simuliere JSON Strategy Import
            sample_json_strategies = self._get_sample_json_strategies()
            
            for strategy_data in sample_json_strategies:
                try:
                    strategy = self._parse_json_strategy(strategy_data, source)
                    strategies.append(strategy)
                    
                except Exception as e:
                    self.logger.error(f"JSON strategy parsing failed: {e}")
                    self.import_stats["failed_imports"] += 1
            
            self.import_stats["successful_imports"] += len(strategies)
            self.logger.info(f"Imported {len(strategies)} JSON strategies")
            
        except Exception as e:
            self.logger.error(f"JSON import failed: {e}")
        
        return strategies
    
    def _get_sample_json_strategies(self) -> List[Dict[str, Any]]:
        """Sample JSON Strategien für Demo"""
        
        return [
            {
                "name": "Moving Average Crossover",
                "description": "Simple MA crossover strategy with risk management",
                "author": "Community User 1",
                "type": "trend_following",
                "entry_conditions": [
                    "SMA(10) crosses above SMA(20)",
                    "Volume > Average Volume * 1.2"
                ],
                "exit_conditions": [
                    "SMA(10) crosses below SMA(20)",
                    "Stop Loss: 2%",
                    "Take Profit: 4%"
                ],
                "risk_management": {
                    "stop_loss": 0.02,
                    "take_profit": 0.04,
                    "position_size": 0.1
                },
                "backtest_results": {
                    "win_rate": 0.65,
                    "profit_factor": 1.8,
                    "max_drawdown": 0.12
                },
                "timeframes": ["1H", "4H"],
                "symbols": ["EURUSD", "GBPUSD"]
            },
            
            {
                "name": "Momentum Breakout",
                "description": "Breakout strategy based on momentum indicators",
                "author": "Community User 2", 
                "type": "breakout",
                "entry_conditions": [
                    "Price breaks above 20-day high",
                    "RSI > 60",
                    "Volume spike > 150% average"
                ],
                "exit_conditions": [
                    "Price falls below 10-day low",
                    "RSI < 40",
                    "Time stop: 5 days"
                ],
                "risk_management": {
                    "stop_loss": 0.03,
                    "take_profit": 0.06,
                    "position_size": 0.05,
                    "max_positions": 3
                },
                "backtest_results": {
                    "win_rate": 0.58,
                    "profit_factor": 2.1,
                    "max_drawdown": 0.18
                },
                "timeframes": ["1D"],
                "symbols": ["EURUSD", "USDJPY", "GBPUSD"]
            }
        ]
    
    def _parse_json_strategy(self, strategy_data: Dict[str, Any], source: StrategySource) -> ImportedStrategy:
        """Parsed JSON Strategy Data"""
        
        return ImportedStrategy(
            strategy_id=f"json_{hash(str(strategy_data)) % 1000000}",
            name=strategy_data.get("name", "Unknown Strategy"),
            description=strategy_data.get("description", ""),
            author=strategy_data.get("author", "Unknown"),
            source=source.name,
            strategy_type=strategy_data.get("type", "unknown"),
            entry_conditions=strategy_data.get("entry_conditions", []),
            exit_conditions=strategy_data.get("exit_conditions", []),
            risk_management=strategy_data.get("risk_management", {}),
            backtest_results=strategy_data.get("backtest_results"),
            win_rate=strategy_data.get("backtest_results", {}).get("win_rate"),
            profit_factor=strategy_data.get("backtest_results", {}).get("profit_factor"),
            max_drawdown=strategy_data.get("backtest_results", {}).get("max_drawdown"),
            timeframes=strategy_data.get("timeframes", ["1H"]),
            symbols=strategy_data.get("symbols", ["EURUSD"]),
            created_at=datetime.now(),
            raw_data=strategy_data
        )
    
    def _import_local_files(self, source: StrategySource) -> List[ImportedStrategy]:
        """Importiert Strategien aus lokalen Dateien"""
        
        strategies = []
        
        try:
            local_path = Path(source.url)
            
            if not local_path.exists():
                self.logger.warning(f"Local path does not exist: {local_path}")
                return []
            
            # Durchsuche Dateien
            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    try:
                        strategy = self._import_single_file(file_path, source)
                        if strategy:
                            strategies.append(strategy)
                    except Exception as e:
                        self.logger.error(f"File import failed {file_path}: {e}")
                        self.import_stats["failed_imports"] += 1
            
            self.import_stats["successful_imports"] += len(strategies)
            self.logger.info(f"Imported {len(strategies)} strategies from local files")
            
        except Exception as e:
            self.logger.error(f"Local file import failed: {e}")
        
        return strategies
    
    def _import_single_file(self, file_path: Path, source: StrategySource) -> Optional[ImportedStrategy]:
        """Importiert einzelne Datei"""
        
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == ".pine":
                # Pine Script File
                with open(file_path, 'r', encoding='utf-8') as f:
                    pine_code = f.read()
                
                strategy = self.pine_parser.parse_pine_script(pine_code)
                strategy.source = f"{source.name}:{file_path.name}"
                return strategy
            
            elif file_extension == ".json":
                # JSON Strategy File
                with open(file_path, 'r', encoding='utf-8') as f:
                    strategy_data = json.load(f)
                
                return self._parse_json_strategy(strategy_data, source)
            
            elif file_extension in [".csv", ".xlsx"]:
                # Spreadsheet Strategy (vereinfacht)
                return self._import_spreadsheet_strategy(file_path, source)
            
            else:
                self.logger.debug(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            self.logger.error(f"Single file import failed: {e}")
            return None
    
    def _import_spreadsheet_strategy(self, file_path: Path, source: StrategySource) -> ImportedStrategy:
        """Importiert Strategie aus Spreadsheet"""
        
        try:
            # Lade Spreadsheet
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Extrahiere Strategy Info (vereinfacht)
            strategy_name = file_path.stem
            
            # Suche nach Entry/Exit Conditions in Spalten
            entry_conditions = []
            exit_conditions = []
            
            for column in df.columns:
                if "entry" in column.lower():
                    conditions = df[column].dropna().unique().tolist()
                    entry_conditions.extend([str(c) for c in conditions if str(c) != 'nan'])
                
                elif "exit" in column.lower():
                    conditions = df[column].dropna().unique().tolist()
                    exit_conditions.extend([str(c) for c in conditions if str(c) != 'nan'])
            
            return ImportedStrategy(
                strategy_id=f"spreadsheet_{hash(str(file_path)) % 1000000}",
                name=strategy_name,
                description=f"Strategy imported from {file_path.name}",
                author="Spreadsheet Import",
                source=f"{source.name}:{file_path.name}",
                strategy_type="spreadsheet_import",
                entry_conditions=entry_conditions[:5],  # Limit
                exit_conditions=exit_conditions[:5],
                risk_management={},
                timeframes=["1H"],
                symbols=["EURUSD"],
                created_at=datetime.now(),
                raw_data={"file_path": str(file_path), "columns": df.columns.tolist()}
            )
            
        except Exception as e:
            self.logger.error(f"Spreadsheet import failed: {e}")
            raise
    
    def import_all_sources(self) -> List[ImportedStrategy]:
        """Importiert Strategien aus allen verfügbaren Quellen"""
        
        all_strategies = []
        
        for source in self.strategy_sources:
            self.logger.info(f"Importing from source: {source.name}")
            
            try:
                strategies = self.import_strategies_from_source(source.name)
                all_strategies.extend(strategies)
                
                self.import_stats["sources_processed"] += 1
                
            except Exception as e:
                self.logger.error(f"Source import failed {source.name}: {e}")
        
        self.import_stats["total_imported"] = len(all_strategies)
        
        self.logger.info(f"Total imported strategies: {len(all_strategies)}")
        return all_strategies
    
    def filter_strategies(self, 
                         strategies: List[ImportedStrategy],
                         min_win_rate: Optional[float] = None,
                         min_profit_factor: Optional[float] = None,
                         max_drawdown: Optional[float] = None,
                         strategy_types: Optional[List[str]] = None) -> List[ImportedStrategy]:
        """Filtert Strategien nach Kriterien"""
        
        filtered = strategies.copy()
        
        # Win Rate Filter
        if min_win_rate is not None:
            filtered = [s for s in filtered if s.win_rate is not None and s.win_rate >= min_win_rate]
        
        # Profit Factor Filter
        if min_profit_factor is not None:
            filtered = [s for s in filtered if s.profit_factor is not None and s.profit_factor >= min_profit_factor]
        
        # Max Drawdown Filter
        if max_drawdown is not None:
            filtered = [s for s in filtered if s.max_drawdown is not None and s.max_drawdown <= max_drawdown]
        
        # Strategy Type Filter
        if strategy_types:
            filtered = [s for s in filtered if s.strategy_type in strategy_types]
        
        self.logger.info(f"Filtered {len(strategies)} → {len(filtered)} strategies")
        return filtered
    
    def export_strategies(self, strategies: List[ImportedStrategy], output_dir: str):
        """Exportiert importierte Strategien"""
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # JSON Export
            strategies_data = [strategy.to_dict() for strategy in strategies]
            
            json_file = output_path / f"imported_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w') as f:
                json.dump(strategies_data, f, indent=2, default=str)
            
            # Statistics Export
            stats_file = output_path / f"import_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.import_stats, f, indent=2)
            
            self.logger.info(f"Strategies exported to {json_file}")
            
        except Exception as e:
            self.logger.error(f"Strategy export failed: {e}")
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Gibt Import-Statistiken zurück"""
        
        stats = self.import_stats.copy()
        
        # Success Rate
        total = stats["successful_imports"] + stats["failed_imports"]
        if total > 0:
            stats["success_rate"] = stats["successful_imports"] / total
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def convert_to_mined_patterns(self, strategies: List[ImportedStrategy]) -> List[MinedPattern]:
        """Konvertiert importierte Strategien zu MinedPattern Format"""
        
        mined_patterns = []
        
        for strategy in strategies:
            try:
                # Konvertiere Strategy zu Pattern
                pattern = MinedPattern(
                    pattern_id=f"strategy_{strategy.strategy_id}",
                    symbol="IMPORTED",
                    timeframe=strategy.timeframes[0] if strategy.timeframes else "1H",
                    pattern_type="imported_strategy",
                    confidence=min(1.0, 0.5 + (strategy.win_rate / 2)) if strategy.win_rate else 0.8,
                    start_time=datetime.now() - timedelta(days=30),
                    end_time=datetime.now(),
                    price_data={"strategy_based": True},
                    indicators={},
                    pattern_features={
                        "strategy_name": strategy.name,
                        "strategy_type": strategy.strategy_type,
                        "entry_conditions": strategy.entry_conditions,
                        "exit_conditions": strategy.exit_conditions,
                        "risk_management": strategy.risk_management,
                        "performance": {
                            "win_rate": strategy.win_rate,
                            "profit_factor": strategy.profit_factor,
                            "max_drawdown": strategy.max_drawdown
                        }
                    },
                    market_context={
                        "source": strategy.source,
                        "author": strategy.author,
                        "imported_strategy": True
                    },
                    mining_timestamp=datetime.now()
                )
                
                mined_patterns.append(pattern)
                
            except Exception as e:
                self.logger.error(f"Strategy conversion failed: {e}")
        
        self.logger.info(f"Converted {len(mined_patterns)} strategies to patterns")
        return mined_patterns


# Convenience Functions
def quick_strategy_import(source_name: str = "Local Files") -> List[ImportedStrategy]:
    """Schneller Strategy Import für Testing"""
    
    importer = CommunityStrategyImporter()
    return importer.import_strategies_from_source(source_name)


def import_all_community_strategies() -> List[ImportedStrategy]:
    """Importiert alle verfügbaren Community Strategien"""
    
    importer = CommunityStrategyImporter()
    return importer.import_all_sources()