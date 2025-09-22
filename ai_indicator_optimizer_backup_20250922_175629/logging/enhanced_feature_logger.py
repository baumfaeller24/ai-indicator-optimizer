#!/usr/bin/env python3
"""
Enhanced Feature Prediction Logger für AI Trading System
Phase 1 Implementation - Foundation Enhancement

Features:
- Strukturiertes AI-Prediction-Logging mit Parquet-Export
- Buffer-System mit konfigurierbarer Größe
- Automatische Parquet-Flush-Funktionalität mit Kompression
- Timestamp-basierte Logging mit Instrument-ID-Tracking
- Integration mit BarDatasetBuilder
- Polars-basierte Performance-Optimierungen
"""

import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class EnhancedFeaturePredictionLogger:
    """
    Enhanced Feature Prediction Logger für AI Trading System
    
    Phase 1 Foundation Enhancement:
    - Production-ready Parquet-basiertes Logging
    - Multi-Process-Safety mit PID-Integration
    - Schema-Drift-Protection
    - Memory-Monitoring
    - Automatische Rotation (täglich/stündlich)
    """
    
    def __init__(
        self,
        base_path: str = "logs/ai_features",
        buffer_size: int = 1000,
        rotation: str = "daily",  # "daily", "hourly", "none"
        compression: str = "zstd",
        include_pid: bool = True,
        mem_monitoring: bool = True,
        fixed_fields: Optional[List[str]] = None
    ):
        """
        Initialize Enhanced Feature Prediction Logger
        
        Args:
            base_path: Basis-Pfad für Log-Dateien
            buffer_size: Buffer-Größe vor automatischem Flush
            rotation: Rotation-Modus ("daily", "hourly", "none")
            compression: Parquet-Kompression ("zstd", "snappy", "gzip")
            include_pid: Ob Process-ID in Dateinamen
            mem_monitoring: Ob Memory-Usage geloggt werden soll
        """
        if not PYARROW_AVAILABLE:
            raise ImportError("PyArrow ist erforderlich: pip install 'pyarrow>=14.0.0'")
        
        self.base_path = Path(base_path)
        self.buffer_size = buffer_size
        self.rotation = rotation
        self.compression = compression
        self.include_pid = include_pid
        self.mem_monitoring = mem_monitoring and PSUTIL_AVAILABLE
        
        # ChatGPT Enhancement: Fixed Fields für stabile Schema
        self.fixed_fields = fixed_fields
        
        # State
        self.buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self._current_path: Optional[Path] = None
        self._pid = os.getpid()
        
        # Statistics
        self.total_entries_logged = 0
        self.total_flushes = 0
        self.total_files_created = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Erstelle Output-Verzeichnis
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"EnhancedFeaturePredictionLogger initialized: {base_path}")
        self.logger.info(f"Config: rotation={rotation}, buffer_size={buffer_size}, pid={include_pid}")
    
    def _get_period_suffix(self, ts_ns: int) -> str:
        """Generiere Zeitraum-Suffix für Rotation"""
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        
        if self.rotation == "daily":
            return dt.strftime("%Y%m%d")
        elif self.rotation == "hourly":
            return dt.strftime("%Y%m%d_%H")
        else:  # "none"
            return "all"
    
    def _get_target_path(self, ts_ns: int) -> Path:
        """Bestimme Ziel-Pfad basierend auf Timestamp"""
        suffix = self._get_period_suffix(ts_ns)
        name = f"{self.base_path.name}_{suffix}"
        
        if self.include_pid:
            name += f"_pid{self._pid}"
        
        return self.base_path.with_name(name).with_suffix(".parquet")
    
    def _rotate_if_needed(self, ts_ns: int) -> None:
        """Rotiere Datei falls nötig"""
        target_path = self._get_target_path(ts_ns)
        
        if target_path != self._current_path:
            # Flush aktueller Buffer
            if self.buffer:
                self._flush_buffer()
            
            # Schließe aktuellen Writer
            self._close_writer()
            
            # Setup neue Datei
            self._current_path = target_path
            self._current_path.parent.mkdir(parents=True, exist_ok=True)
            self.total_files_created += 1
            
            self.logger.info(f"Rotated to: {self._current_path.name}")
    
    def _close_writer(self) -> None:
        """Schließe ParquetWriter sicher"""
        if self._writer:
            try:
                self._writer.close()
                self.logger.debug("ParquetWriter closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing ParquetWriter: {e}")
                self.logger.debug(traceback.format_exc())
            finally:
                self._writer = None
                self._schema = None
    
    def _flush_buffer(self) -> None:
        """Flush Buffer zu Parquet mit PyArrow"""
        if not self.buffer:
            return
        
        try:
            # Konvertiere zu PyArrow Table
            table = pa.Table.from_pylist(self.buffer)
            
            # Setup Writer falls nötig
            if self._writer is None:
                # ChatGPT Enhancement: Kompressions-Fallback
                comp = self.compression
                try:
                    # Test ob Kompression verfügbar ist
                    dummy_schema = pa.schema([("test", pa.int64())])
                    test_path = self._current_path.with_suffix(".test")
                    test_writer = pq.ParquetWriter(test_path, dummy_schema, compression=comp)
                    test_writer.close()
                    test_path.unlink(missing_ok=True)  # Cleanup
                except Exception:
                    comp = "snappy"
                    self.logger.warning(f"Compression '{self.compression}' not available → falling back to 'snappy'")
                
                self._schema = table.schema
                self._writer = pq.ParquetWriter(
                    self._current_path, 
                    self._schema, 
                    compression=comp
                )
                self.logger.debug(f"Created new ParquetWriter: {self._current_path.name} (comp={comp})")
            else:
                # ChatGPT Enhancement: Schema-Drift-Check mit check_metadata=False
                if not table.schema.equals(self._schema, check_metadata=False):
                    raise ValueError(
                        f"Schema drift detected in {self._current_path.name}! "
                        f"Expected: {self._schema}, Got: {table.schema} (metadata ignored)"
                    )
            
            # Schreibe Table
            self._writer.write_table(table)
            
            # Statistics
            entries_count = len(self.buffer)
            self.total_entries_logged += entries_count
            self.total_flushes += 1
            
            # Memory-Monitoring
            if self.mem_monitoring:
                mem_gb = psutil.Process().memory_info().rss / (1024**3)
                self.logger.info(
                    f"Flushed {entries_count} entries to {self._current_path.name} | "
                    f"RSS={mem_gb:.2f} GB | Total: {self.total_entries_logged}"
                )
            else:
                self.logger.info(
                    f"Flushed {entries_count} entries to {self._current_path.name} | "
                    f"Total: {self.total_entries_logged}"
                )
            
            # Buffer leeren
            self.buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer: {e}")
            self.logger.debug(traceback.format_exc())
            raise e
    
    def log_prediction(
        self,
        *,
        ts_ns: int,
        instrument: str,
        features: Dict[str, Any],
        prediction: Dict[str, Any],
        confidence_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        market_regime: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log AI Prediction mit Enhanced Features
        
        Args:
            ts_ns: Timestamp in Nanosekunden
            instrument: Trading-Instrument (z.B. "EUR/USD")
            features: Feature-Dictionary für ML-Training
            prediction: AI-Prediction-Dictionary
            confidence_score: Enhanced Confidence Score
            risk_score: Risk Assessment Score
            market_regime: Market Regime Classification
            additional_data: Zusätzliche Daten
        """
        try:
            # Rotation-Check
            self._rotate_if_needed(ts_ns)
            
            # Erstelle strukturierten Entry
            entry = {
                # Core Timestamp & Instrument
                "ts_ns": ts_ns,
                "timestamp": datetime.utcfromtimestamp(ts_ns / 1e9).isoformat(),
                "instrument": instrument,
                
                # Features (flach für bessere Parquet-Performance)
                **{f"feat_{k}": v for k, v in features.items()},
                
                # AI Predictions
                "pred_action": prediction.get("action"),
                "pred_confidence": prediction.get("confidence"),
                "pred_reasoning": prediction.get("reasoning"),
                
                # Enhanced Scores
                "enhanced_confidence": confidence_score,
                "risk_score": risk_score,
                "market_regime": market_regime,
                
                # Metadata
                "pid": self._pid,
                "log_entry_id": len(self.buffer),
                "session_time": int(time.time() * 1e9)
            }
            
            # Zusätzliche Daten hinzufügen
            if additional_data:
                entry.update({f"extra_{k}": v for k, v in additional_data.items()})
            
            # ChatGPT Enhancement: Row-Normalisierung für stabile Keys
            if self.fixed_fields:
                normalized_entry = {}
                for field in self.fixed_fields:
                    normalized_entry[field] = entry.get(field, "")  # Default empty string
                entry = normalized_entry
            
            # Füge zu Buffer hinzu
            self.buffer.append(entry)
            
            # Auto-Flush bei Buffer-Überlauf
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
                
        except Exception as e:
            self.logger.error(f"Error logging prediction: {e}")
            self.logger.debug(traceback.format_exc())
            # Nicht re-raisen - Logging-Fehler sollen Trading nicht stoppen
    
    def flush(self) -> None:
        """Manueller Flush des Buffers"""
        if self.buffer:
            self._flush_buffer()
    
    def close(self) -> None:
        """Schließe Logger sicher"""
        try:
            self.flush()
        except Exception as e:
            self.logger.error(f"Error during final flush: {e}")
        finally:
            try:
                self._close_writer()
            except Exception as e:
                self.logger.error(f"Error closing writer: {e}")
        
        self.logger.info("EnhancedFeaturePredictionLogger closed")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Logger-Statistiken"""
        file_size = 0
        if self._current_path and self._current_path.exists():
            file_size = self._current_path.stat().st_size
        
        stats = {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "buffer_usage_pct": (len(self.buffer) / self.buffer_size) * 100,
            "current_file": str(self._current_path) if self._current_path else None,
            "file_size_bytes": file_size,
            "total_entries_logged": self.total_entries_logged,
            "total_flushes": self.total_flushes,
            "total_files_created": self.total_files_created,
            "rotation": self.rotation,
            "compression": self.compression,
            "pid": self._pid,
            "schema_columns": len(self._schema.names) if self._schema else 0
        }
        
        if self.mem_monitoring:
            stats["memory_rss_gb"] = psutil.Process().memory_info().rss / (1024**3)
        
        return stats
    
    def __enter__(self):
        """Context Manager Support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support"""
        self.close()


# Factory Functions für einfache Nutzung
def create_enhanced_feature_logger(
    base_path: str = "logs/ai_features",
    buffer_size: int = 1000,
    rotation: str = "daily",
    fixed_fields: Optional[List[str]] = None,
    **kwargs
) -> EnhancedFeaturePredictionLogger:
    """
    Factory Function für Enhanced Feature Logger
    
    Args:
        base_path: Basis-Pfad für Log-Dateien
        buffer_size: Buffer-Größe vor automatischem Flush
        rotation: Rotation-Modus ("daily", "hourly", "none")
        **kwargs: Weitere Parameter für EnhancedFeaturePredictionLogger
    
    Returns:
        EnhancedFeaturePredictionLogger Instance
    """
    return EnhancedFeaturePredictionLogger(
        base_path=base_path,
        buffer_size=buffer_size,
        rotation=rotation,
        fixed_fields=fixed_fields,
        **kwargs
    )


if __name__ == "__main__":
    # Test des Enhanced Feature Loggers
    print("🧪 Testing EnhancedFeaturePredictionLogger...")
    
    # Test mit verschiedenen Rotation-Modi
    for rotation in ["none", "daily", "hourly"]:
        print(f"\\n📋 Testing rotation: {rotation}")
        
        with create_enhanced_feature_logger(
            f"test_logs/enhanced_features_{rotation}",
            buffer_size=3,
            rotation=rotation,
            mem_monitoring=True
        ) as logger:
            
            # Simuliere Trading-Daten über mehrere Zeiträume
            base_time = int(time.time() * 1e9)
            
            for period in range(2):
                for entry in range(4):
                    # Verschiedene Zeitstempel für Rotation-Test
                    if rotation == "hourly":
                        ts = base_time + period * 3600 * 1e9 + entry * 60 * 1e9
                    elif rotation == "daily":
                        ts = base_time + period * 24 * 3600 * 1e9 + entry * 3600 * 1e9
                    else:
                        ts = base_time + (period * 4 + entry) * 60 * 1e9
                    
                    logger.log_prediction(
                        ts_ns=int(ts),
                        instrument="EUR/USD",
                        features={
                            "open": 1.1000 + period * 0.001 + entry * 0.0001,
                            "close": 1.1005 + period * 0.001 + entry * 0.0001,
                            "rsi_14": 50 + entry * 5,
                            "volatility": 0.001 + entry * 0.0001,
                            "period": period,
                            "entry": entry
                        },
                        prediction={
                            "action": ["BUY", "SELL", "HOLD"][entry % 3],
                            "confidence": 0.6 + entry * 0.1,
                            "reasoning": f"enhanced_test_p{period}_e{entry}"
                        },
                        confidence_score=0.7 + entry * 0.05,
                        risk_score=0.1 + entry * 0.02,
                        market_regime=["trending", "ranging", "volatile"][entry % 3]
                    )
            
            # Zeige Stats
            stats = logger.get_statistics()
            print(f"📊 {rotation} Stats: {stats}")
    
    print("\\n✅ EnhancedFeaturePredictionLogger Test abgeschlossen!")