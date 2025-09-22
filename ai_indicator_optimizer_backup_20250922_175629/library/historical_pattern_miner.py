"""
Historical Pattern Miner für automatische Pattern-Extraktion
Nutzt DukascopyConnector + PatternDetector für 14-Tage-Datensammlung
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
import time
from tqdm import tqdm
import pickle

# Globales Logging-Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from ..data.dukascopy_connector import DukascopyConnector, create_trading_data_from_ohlcv
except ImportError:
    DukascopyConnector = None
    create_trading_data_from_ohlcv = None

try:
    from ..training.training_dataset_builder import PatternDetector
except ImportError:
    PatternDetector = None

try:
    from ..indicators.indicator_calculator import IndicatorCalculator
except ImportError:
    IndicatorCalculator = None

try:
    from ..library.pattern_library import PatternLibrary
except ImportError:
    PatternLibrary = None

try:
    from ..core.resource_manager import ResourceManager
except ImportError:
    ResourceManager = None


@dataclass
class MiningConfig:
    """Konfiguration für Pattern Mining"""
    
    # Data Collection
    symbols: List[str] = field(default_factory=lambda: ["EUR/USD"])
    timeframes: List[str] = field(default_factory=lambda: ["1H", "4H", "1D"])
    mining_days: int = 14
    lookback_window: int = 100
    
    # Pattern Detection
    min_confidence: float = 0.6
    max_patterns_per_symbol: int = 50
    pattern_types: List[str] = field(default_factory=lambda: [
        "double_top", "double_bottom", "head_shoulders",
        "triangle", "support_resistance", "breakout"
    ])
    
    # Performance
    max_workers: int = 16  # Für 32 CPU-Kerne optimal
    batch_size: int = 10
    use_multiprocessing: bool = True
    
    # Storage
    output_dir: str = "./library/mined_patterns"
    cache_enabled: bool = True
    
    # __post_init__ nicht mehr nötig - field(default_factory) löst das Problem


@dataclass
class MinedPattern:
    """Container für geminte Patterns"""
    pattern_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    price_data: Dict[str, Any]
    indicators: Dict[str, Any]
    pattern_features: Dict[str, Any]
    market_context: Dict[str, Any]
    mining_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary für Speicherung"""
        return {
            "pattern_id": self.pattern_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "price_data": self.price_data,
            "indicators": self.indicators,
            "pattern_features": self.pattern_features,
            "market_context": self.market_context,
            "mining_timestamp": self.mining_timestamp.isoformat()
        }


class HistoricalPatternMiner:
    """
    Automatische Pattern-Extraktion aus historischen Forex-Daten
    Nutzt 32 CPU-Kerne für parallele Verarbeitung
    """
    
    def __init__(self, 
                 config: Optional[MiningConfig] = None,
                 resource_manager: Optional[ResourceManager] = None):
        
        self.config = config or MiningConfig()
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.dukascopy_connector = DukascopyConnector()
        self.pattern_detector = PatternDetector()
        self.indicator_calculator = IndicatorCalculator()
        
        # Pattern Library Integration
        self.pattern_library = None
        
        # Mining State
        self.mined_patterns: List[MinedPattern] = []
        self.mining_statistics = {
            "total_processed": 0,
            "patterns_found": 0,
            "processing_time": 0.0,
            "symbols_processed": 0,
            "timeframes_processed": 0
        }
        
        # Setup Output Directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"HistoricalPatternMiner initialized with {self.config.max_workers} workers")
    
    def mine_patterns_comprehensive(self) -> List[MinedPattern]:
        """
        Comprehensive Pattern Mining über alle Symbole und Timeframes
        """
        
        start_time = time.time()
        self.logger.info("Starting comprehensive pattern mining...")
        
        try:
            # Erstelle Mining-Tasks
            mining_tasks = []
            
            for symbol in self.config.symbols:
                for timeframe in self.config.timeframes:
                    mining_tasks.append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "days": self.config.mining_days
                    })
            
            self.logger.info(f"Created {len(mining_tasks)} mining tasks")
            
            # Parallel Processing
            all_patterns = []
            
            if self.config.use_multiprocessing:
                all_patterns = self._mine_patterns_multiprocessing(mining_tasks)
            else:
                all_patterns = self._mine_patterns_threading(mining_tasks)
            
            # Prüfe ob Patterns gefunden wurden
            if not all_patterns:
                self.logger.warning("No patterns found during mining process")
                return []
            
            # Filter und Deduplizierung
            filtered_patterns = self._filter_and_deduplicate_patterns(all_patterns)
            
            # Update Statistics
            processing_time = time.time() - start_time
            self.mining_statistics.update({
                "total_processed": len(mining_tasks),
                "patterns_found": len(filtered_patterns),
                "processing_time": processing_time,
                "symbols_processed": len(self.config.symbols),
                "timeframes_processed": len(self.config.timeframes)
            })
            
            # Speichere Patterns
            self._save_mined_patterns(filtered_patterns)
            
            self.logger.info(f"Pattern mining completed: {len(filtered_patterns)} patterns in {processing_time:.2f}s")
            
            return filtered_patterns
            
        except Exception as e:
            self.logger.error(f"Comprehensive pattern mining failed: {e}")
            return []
    
    def _mine_patterns_multiprocessing(self, mining_tasks: List[Dict[str, Any]]) -> List[MinedPattern]:
        """Mining mit Multiprocessing für CPU-intensive Tasks"""
        
        all_patterns = []
        
        # Batch Processing für bessere Performance
        batch_size = self.config.batch_size
        task_batches = [mining_tasks[i:i + batch_size] for i in range(0, len(mining_tasks), batch_size)]
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            
            # Submit Batches
            future_to_batch = {
                executor.submit(self._process_mining_batch, batch): batch 
                for batch in task_batches
            }
            
            # Collect Results mit Progress Bar
            with tqdm(total=len(task_batches), desc="Mining Patterns") as pbar:
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    
                    try:
                        batch_patterns = future.result()
                        all_patterns.extend(batch_patterns)
                        
                        pbar.set_postfix({
                            "Patterns": len(all_patterns),
                            "Batch": f"{len(batch)} tasks"
                        })
                        
                    except Exception as e:
                        self.logger.exception(f"Batch processing failed: {e}")
                        pbar.update(1)  # Update auch bei Fehlern
                    else:
                        pbar.update(1)
        
        return all_patterns
    
    def _mine_patterns_threading(self, mining_tasks: List[Dict[str, Any]]) -> List[MinedPattern]:
        """Mining mit Threading für I/O-intensive Tasks"""
        
        all_patterns = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            
            # Submit Tasks
            future_to_task = {
                executor.submit(self._mine_patterns_for_symbol_timeframe, 
                               task["symbol"], task["timeframe"], task["days"]): task 
                for task in mining_tasks
            }
            
            # Collect Results
            with tqdm(total=len(mining_tasks), desc="Mining Patterns") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    
                    try:
                        patterns = future.result()
                        all_patterns.extend(patterns)
                        
                        pbar.set_postfix({
                            "Patterns": len(all_patterns),
                            "Symbol": task["symbol"],
                            "TF": task["timeframe"]
                        })
                        
                    except Exception as e:
                        self.logger.exception(f"Task {task} failed: {e}")
                        pbar.update(1)  # Update auch bei Fehlern
                    else:
                        pbar.update(1)
        
        return all_patterns
    
    def _process_mining_batch(self, batch: List[Dict[str, Any]]) -> List[MinedPattern]:
        """Verarbeitet Batch von Mining-Tasks (für Multiprocessing)"""
        
        batch_patterns = []
        
        # Neue Instanzen für Multiprocessing
        dukascopy_connector = DukascopyConnector()
        pattern_detector = PatternDetector()
        indicator_calculator = IndicatorCalculator()
        
        for task in batch:
            try:
                patterns = self._mine_patterns_for_symbol_timeframe_static(
                    task["symbol"], 
                    task["timeframe"], 
                    task["days"],
                    dukascopy_connector,
                    pattern_detector,
                    indicator_calculator,
                    self.config
                )
                batch_patterns.extend(patterns)
                
            except Exception as e:
                logging.exception(f"Batch task {task} failed: {e}")
        
        return batch_patterns
    
    @staticmethod
    def _mine_patterns_for_symbol_timeframe_static(symbol: str,
                                                  timeframe: str,
                                                  days: int,
                                                  dukascopy_connector: DukascopyConnector,
                                                  pattern_detector: PatternDetector,
                                                  indicator_calculator: IndicatorCalculator,
                                                  config: MiningConfig) -> List[MinedPattern]:
        """Statische Methode für Multiprocessing"""
        
        try:
            # Lade historische Daten
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ohlcv_data = dukascopy_connector.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if ohlcv_data.empty:
                return []
            
            # Berechne Indikatoren
            trading_data = create_trading_data_from_ohlcv(symbol, timeframe, ohlcv_data)
            indicators = indicator_calculator.calculate_all_indicators(trading_data)
            
            # Detektiere Patterns
            detected_patterns = pattern_detector.detect_patterns(
                ohlcv_data, 
                indicators, 
                lookback_window=config.lookback_window
            )
            
            # Konvertiere zu MinedPattern
            mined_patterns = []
            
            for pattern_info in detected_patterns:
                if pattern_info["confidence"] >= config.min_confidence:
                    
                    # Pattern ID generieren (UUID für Eindeutigkeit)
                    import uuid
                    pattern_id = f"{symbol}_{timeframe}_{pattern_info['pattern_type']}_{uuid.uuid4().hex[:8]}"
                    
                    # Zeitbereich bestimmen
                    start_idx = pattern_info.get("start_index", 0)
                    end_idx = pattern_info.get("end_index", len(ohlcv_data) - 1)
                    
                    start_time = ohlcv_data.iloc[start_idx]["timestamp"]
                    end_time = ohlcv_data.iloc[end_idx]["timestamp"]
                    
                    # Price Data extrahieren
                    price_data = {
                        "ohlcv": ohlcv_data.iloc[start_idx:end_idx+1].to_dict("records"),
                        "price_range": {
                            "high": float(ohlcv_data.iloc[start_idx:end_idx+1]["high"].max()),
                            "low": float(ohlcv_data.iloc[start_idx:end_idx+1]["low"].min())
                        }
                    }
                    
                    # Market Context
                    market_context = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "volatility": float(ohlcv_data["close"].pct_change().std()),
                        "trend": "bullish" if ohlcv_data["close"].iloc[-1] > ohlcv_data["close"].iloc[0] else "bearish",
                        "volume_avg": float(ohlcv_data["volume"].mean())
                    }
                    
                    mined_pattern = MinedPattern(
                        pattern_id=pattern_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        pattern_type=pattern_info["pattern_type"],
                        confidence=pattern_info["confidence"],
                        start_time=start_time,
                        end_time=end_time,
                        price_data=price_data,
                        indicators=indicators,
                        pattern_features=pattern_info,
                        market_context=market_context,
                        mining_timestamp=datetime.now()
                    )
                    
                    mined_patterns.append(mined_pattern)
            
            return mined_patterns
            
        except Exception as e:
            logging.exception(f"Pattern mining for {symbol} {timeframe} failed: {e}")
            return []
    
    def _mine_patterns_for_symbol_timeframe(self, 
                                          symbol: str,
                                          timeframe: str,
                                          days: int) -> List[MinedPattern]:
        """Mining für ein Symbol/Timeframe Paar"""
        
        return self._mine_patterns_for_symbol_timeframe_static(
            symbol, timeframe, days,
            self.dukascopy_connector,
            self.pattern_detector,
            self.indicator_calculator,
            self.config
        )
    
    def _filter_and_deduplicate_patterns(self, patterns: List[MinedPattern]) -> List[MinedPattern]:
        """Filtert und dedupliziert Patterns"""
        
        try:
            # Nach Confidence sortieren
            patterns.sort(key=lambda p: p.confidence, reverse=True)
            
            # Deduplizierung basierend auf Ähnlichkeit
            unique_patterns = []
            
            for pattern in patterns:
                is_duplicate = False
                
                for existing in unique_patterns:
                    if self._are_patterns_similar(pattern, existing):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Prüfe Symbol-Limit vor dem Hinzufügen
                    symbol_count = len([p for p in unique_patterns if p.symbol == pattern.symbol])
                    if symbol_count < self.config.max_patterns_per_symbol:
                        unique_patterns.append(pattern)
            
            self.logger.info(f"Filtered {len(patterns)} → {len(unique_patterns)} unique patterns")
            return unique_patterns
            
        except Exception as e:
            self.logger.error(f"Pattern filtering failed: {e}")
            return patterns
    
    def _are_patterns_similar(self, pattern1: MinedPattern, pattern2: MinedPattern) -> bool:
        """Prüft ob zwei Patterns ähnlich sind"""
        
        try:
            # Gleicher Typ und Symbol
            if (pattern1.pattern_type != pattern2.pattern_type or 
                pattern1.symbol != pattern2.symbol or
                pattern1.timeframe != pattern2.timeframe):
                return False
            
            # Zeitliche Überlappung
            time_overlap = (
                pattern1.start_time <= pattern2.end_time and 
                pattern2.start_time <= pattern1.end_time
            )
            
            if time_overlap:
                return True
            
            # Preis-Ähnlichkeit
            price1_range = pattern1.price_data["price_range"]
            price2_range = pattern2.price_data["price_range"]
            
            price_similarity = (
                abs(price1_range["high"] - price2_range["high"]) < 0.001 and
                abs(price1_range["low"] - price2_range["low"]) < 0.001
            )
            
            return price_similarity
            
        except Exception:
            return False
    
    def _save_mined_patterns(self, patterns: List[MinedPattern]):
        """Speichert geminte Patterns"""
        
        try:
            # Timestamp einmal generieren (DRY Prinzip)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON Export
            patterns_data = [pattern.to_dict() for pattern in patterns]
            
            json_file = self.output_dir / f"mined_patterns_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
            
            # Pickle Export für Python Objects
            pickle_file = self.output_dir / f"mined_patterns_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(patterns, f)
            
            # Statistics Export
            stats_file = self.output_dir / f"mining_statistics_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.mining_statistics, f, indent=2)
            
            self.logger.info(f"Patterns saved to {json_file}")
            
        except Exception as e:
            self.logger.error(f"Pattern saving failed: {e}")
    
    def mine_patterns_for_symbol(self, 
                                symbol: str,
                                timeframes: Optional[List[str]] = None,
                                days: Optional[int] = None) -> List[MinedPattern]:
        """Mining für spezifisches Symbol"""
        
        timeframes = timeframes or self.config.timeframes
        days = days or self.config.mining_days
        
        all_patterns = []
        
        for timeframe in timeframes:
            patterns = self._mine_patterns_for_symbol_timeframe(symbol, timeframe, days)
            all_patterns.extend(patterns)
        
        return all_patterns
    
    def get_mining_statistics(self) -> Dict[str, Any]:
        """Gibt Mining-Statistiken zurück"""
        
        stats = self.mining_statistics.copy()
        
        # Pattern-Typ Verteilung
        if self.mined_patterns:
            pattern_types = {}
            for pattern in self.mined_patterns:
                pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
            
            stats["pattern_type_distribution"] = pattern_types
            
            # Confidence-Statistiken
            confidences = [p.confidence for p in self.mined_patterns]
            stats["confidence_stats"] = {
                "mean": np.mean(confidences),
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences)
            }
        
        return stats
    
    def load_mined_patterns(self, file_path: str) -> List[MinedPattern]:
        """Lädt gespeicherte Patterns"""
        
        try:
            file_path = Path(file_path)
            
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    patterns_data = json.load(f)
                
                patterns = []
                for data in patterns_data:
                    # Konvertiere zurück zu MinedPattern
                    pattern = MinedPattern(
                        pattern_id=data["pattern_id"],
                        symbol=data["symbol"],
                        timeframe=data["timeframe"],
                        pattern_type=data["pattern_type"],
                        confidence=data["confidence"],
                        start_time=datetime.fromisoformat(data["start_time"]),
                        end_time=datetime.fromisoformat(data["end_time"]),
                        price_data=data["price_data"],
                        indicators=data["indicators"],
                        pattern_features=data["pattern_features"],
                        market_context=data["market_context"],
                        mining_timestamp=datetime.fromisoformat(data["mining_timestamp"])
                    )
                    patterns.append(pattern)
                
                return patterns
            
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Loading patterns failed: {e}")
            return []
    
    def export_patterns_to_library(self, patterns: List[MinedPattern]) -> bool:
        """Exportiert Patterns zur Pattern Library"""
        
        try:
            if self.pattern_library is None:
                self.pattern_library = PatternLibrary()
            
            exported_count = 0
            
            for pattern in patterns:
                # Konvertiere zu Library-Format
                library_pattern = {
                    "name": f"{pattern.pattern_type}_{pattern.symbol}_{pattern.timeframe}",
                    "type": pattern.pattern_type,
                    "symbol": pattern.symbol,
                    "timeframe": pattern.timeframe,
                    "confidence": pattern.confidence,
                    "price_data": pattern.price_data,
                    "indicators": pattern.indicators,
                    "features": pattern.pattern_features,
                    "market_context": pattern.market_context,
                    "source": "historical_mining",
                    "created_at": pattern.mining_timestamp.isoformat()
                }
                
                # Füge zur Library hinzu
                success = self.pattern_library.add_pattern(library_pattern)
                if success:
                    exported_count += 1
            
            self.logger.info(f"Exported {exported_count}/{len(patterns)} patterns to library")
            return exported_count > 0
            
        except Exception as e:
            self.logger.error(f"Pattern export to library failed: {e}")
            return False
    
    def cleanup(self):
        """Bereinigt Ressourcen"""
        try:
            if self.dukascopy_connector:
                self.dukascopy_connector.cleanup()
            
            self.logger.info("HistoricalPatternMiner cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


# Convenience Functions
def quick_pattern_mining(symbol: str = "EUR/USD", 
                        days: int = 7,
                        timeframe: str = "1H") -> List[MinedPattern]:
    """Schnelles Pattern Mining für Testing"""
    
    config = MiningConfig(
        symbols=[symbol],
        timeframes=[timeframe],
        mining_days=days,
        max_workers=4
    )
    
    miner = HistoricalPatternMiner(config)
    return miner.mine_patterns_comprehensive()


def batch_pattern_mining(symbols: List[str],
                        timeframes: List[str],
                        days: int = 14) -> List[MinedPattern]:
    """Batch Pattern Mining für mehrere Symbole"""
    
    config = MiningConfig(
        symbols=symbols,
        timeframes=timeframes,
        mining_days=days,
        max_workers=16
    )
    
    miner = HistoricalPatternMiner(config)
    return miner.mine_patterns_comprehensive()