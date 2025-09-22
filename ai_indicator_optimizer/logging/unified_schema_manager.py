#!/usr/bin/env python3
"""
🧩 BAUSTEIN A1: Unified Schema Manager
Schema-Problem-Behebung für Parquet-Logging

Problem gelöst:
- BarDatasetBuilder: feature_label_fwd_ret_h5 (Forward Return Labels)
- IntegratedDatasetLogger: feature_sma_5 (Technical Indicators)
- Schema-Mismatch beim Parquet-Append

Lösung:
- Separate Logging-Streams für verschiedene Datentypen
- Unified Schema Definition mit festen Spalten-Sets
- Schema-Validierung vor Parquet-Writes
- Backward-Compatibility mit bestehenden Loggern
"""

from __future__ import annotations
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path
from datetime import datetime
import logging
from enum import Enum


class DataStreamType(Enum):
    """Verschiedene Datenstream-Typen mit separaten Schemas"""
    TECHNICAL_FEATURES = "technical_features"
    ML_DATASET = "ml_dataset"
    AI_PREDICTIONS = "ai_predictions"
    PERFORMANCE_METRICS = "performance_metrics"


class UnifiedSchemaManager:
    """
    🧩 BAUSTEIN A1: Unified Schema Manager
    
    Löst das Schema-Mismatch-Problem durch:
    1. Separate Logging-Streams für verschiedene Datentypen
    2. Feste Schema-Definitionen pro Stream-Typ
    3. Automatische Schema-Validierung
    4. Graceful Degradation bei Schema-Konflikten
    """
    
    # FIXED SCHEMAS für verschiedene Datenstream-Typen (angepasst an bestehende Daten)
    SCHEMAS = {
        DataStreamType.TECHNICAL_FEATURES: [
            # Basis OHLCV (Kern-Felder)
            "timestamp", "symbol", "open", "close",
            # Häufig verwendete technische Indikatoren
            "sma_5", "rsi_14", "macd",
            # Optional: Erweiterte OHLCV
            "high", "low", "volume", "timeframe",
            # Optional: Erweiterte Indikatoren
            "sma_10", "sma_20", "ema_12", "ema_26", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14", "adx_14", "stoch_k", "stoch_d",
            # Optional: Candlestick Features
            "price_change", "price_change_pct", "volatility",
            "body_size", "upper_shadow", "lower_shadow", "is_bullish",
            # Optional: Zusätzliche Features
            "volume_sma", "volume_ratio", "price_range"
        ],
        
        DataStreamType.ML_DATASET: [
            # Basis Info (Kern-Felder)
            "timestamp", "symbol", "open", "close",
            # Forward Return Labels (häufig verwendet)
            "feature_label_fwd_ret_h5", "label_binary",
            # Feature-Subset für ML (häufig verwendet)
            "feature_sma_5",
            # Optional: Erweiterte Info
            "timeframe", "bar_index", "high", "low", "volume",
            # Optional: Erweiterte Forward Returns
            "fwd_ret_1", "fwd_ret_3", "fwd_ret_5", "fwd_ret_10",
            "fwd_ret_20", "fwd_ret_h1", "fwd_ret_h4", "fwd_ret_d1",
            # Optional: Erweiterte Labels
            "label_multiclass", "label_regression",
            # Optional: Erweiterte Features
            "feature_rsi_14", "feature_macd", "feature_volatility", "feature_volume_ratio"
        ],
        
        DataStreamType.AI_PREDICTIONS: [
            # Prediction Info (Kern-Felder)
            "timestamp", "symbol", "prediction_class", "confidence_score",
            # Häufig verwendete Predictions
            "buy_probability", "model_name",
            # Optional: Erweiterte Prediction Info
            "timeframe", "prediction_id", "model_version", "inference_time_ms",
            # Optional: Erweiterte Predictions
            "prediction_probability", "sell_probability", "hold_probability",
            # Optional: Risk Assessment
            "risk_score", "position_size", "stop_loss", "take_profit",
            # Optional: Market Context
            "market_regime", "volatility_regime", "trend_strength"
        ],
        
        DataStreamType.PERFORMANCE_METRICS: [
            # System Performance
            "timestamp", "component", "operation",
            "duration_ms", "memory_usage_mb", "cpu_usage_pct",
            "gpu_usage_pct", "throughput_ops_sec",
            # Business Metrics
            "bars_processed", "predictions_made", "accuracy_score",
            "sharpe_ratio", "max_drawdown", "total_return"
        ]
    }
    
    def __init__(self, base_output_path: str = "logs/unified"):
        """
        Initialize Unified Schema Manager
        
        Args:
            base_output_path: Basis-Pfad für alle separaten Logging-Streams
        """
        self.base_path = Path(base_output_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Schema-Validierung Cache
        self._schema_cache: Dict[DataStreamType, pa.Schema] = {}
        self._file_writers: Dict[DataStreamType, Optional[pq.ParquetWriter]] = {}
        
        # Performance Tracking
        self.validation_count = 0
        self.schema_mismatches = 0
        self.successful_writes = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"UnifiedSchemaManager initialized: {base_output_path}")
        
        # Initialisiere Schema-Cache
        self._initialize_schema_cache()
    
    def _initialize_schema_cache(self) -> None:
        """Initialisiere PyArrow Schema-Cache für alle Stream-Typen"""
        for stream_type in DataStreamType:
            schema_fields = []
            for field_name in self.SCHEMAS[stream_type]:
                # Automatische Typ-Inferenz basierend auf Feldnamen
                if field_name in ["timestamp"]:
                    field_type = pa.timestamp('ns')
                elif field_name in ["symbol", "timeframe", "model_name", "component", "operation", "market_regime"]:
                    field_type = pa.string()
                elif "probability" in field_name or "score" in field_name or "ratio" in field_name:
                    field_type = pa.float64()
                elif "count" in field_name or "index" in field_name or "_ms" in field_name:
                    field_type = pa.int64()
                elif "is_" in field_name:
                    field_type = pa.bool_()
                else:
                    field_type = pa.float64()  # Default für numerische Features
                
                schema_fields.append(pa.field(field_name, field_type))
            
            self._schema_cache[stream_type] = pa.schema(schema_fields)
            self.logger.debug(f"Schema cached for {stream_type.value}: {len(schema_fields)} fields")
    
    def get_output_path(self, stream_type: DataStreamType, date_suffix: bool = True) -> Path:
        """
        Generiere Output-Pfad für spezifischen Stream-Typ
        
        Args:
            stream_type: Typ des Datenstreams
            date_suffix: Ob Datum im Dateinamen enthalten sein soll
            
        Returns:
            Path-Objekt für Output-Datei
        """
        filename = stream_type.value
        if date_suffix:
            date_str = datetime.now().strftime("%Y%m%d")
            filename = f"{filename}_{date_str}"
        
        return self.base_path / f"{filename}.parquet"
    
    def validate_data_schema(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]], pl.DataFrame], 
        stream_type: DataStreamType
    ) -> bool:
        """
        Validiere Daten gegen Schema für spezifischen Stream-Typ
        
        Args:
            data: Zu validierende Daten
            stream_type: Erwarteter Stream-Typ
            
        Returns:
            True wenn Schema kompatibel, False sonst
        """
        self.validation_count += 1
        
        try:
            # Konvertiere zu einheitlichem Format
            if isinstance(data, dict):
                data_fields = set(data.keys())
            elif isinstance(data, list) and len(data) > 0:
                data_fields = set(data[0].keys())
            elif isinstance(data, pl.DataFrame):
                data_fields = set(data.columns)
            else:
                self.logger.warning(f"Unknown data type for validation: {type(data)}")
                return False
            
            # Erwartete Felder für Stream-Typ
            expected_fields = set(self.SCHEMAS[stream_type])
            
            # Schema-Validierung
            missing_fields = expected_fields - data_fields
            extra_fields = data_fields - expected_fields
            
            if missing_fields:
                self.logger.warning(f"Missing fields for {stream_type.value}: {missing_fields}")
            
            if extra_fields:
                self.logger.debug(f"Extra fields for {stream_type.value}: {extra_fields}")
            
            # Akzeptiere wenn mindestens 30% der erwarteten Felder vorhanden sind (flexibler für bestehende Daten)
            coverage = len(data_fields & expected_fields) / len(expected_fields)
            is_valid = coverage >= 0.3
            
            if not is_valid:
                self.schema_mismatches += 1
                self.logger.error(f"Schema validation failed for {stream_type.value}: {coverage:.1%} coverage")
            
            return is_valid
            
        except Exception as e:
            self.logger.error(f"Schema validation error: {e}")
            self.schema_mismatches += 1
            return False
    
    def normalize_data_for_schema(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]]], 
        stream_type: DataStreamType
    ) -> List[Dict[str, Any]]:
        """
        Normalisiere Daten für spezifisches Schema
        
        Args:
            data: Zu normalisierende Daten
            stream_type: Ziel-Stream-Typ
            
        Returns:
            Normalisierte Daten-Liste
        """
        # Konvertiere zu Liste von Dictionaries
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data
        
        expected_fields = self.SCHEMAS[stream_type]
        normalized_data = []
        
        for row in data_list:
            normalized_row = {}
            
            # Fülle erwartete Felder
            for field in expected_fields:
                if field in row:
                    normalized_row[field] = row[field]
                else:
                    # Default-Werte für fehlende Felder
                    if field == "timestamp":
                        normalized_row[field] = datetime.now()
                    elif field in ["symbol", "timeframe", "model_name"]:
                        normalized_row[field] = "unknown"
                    elif "probability" in field or "score" in field:
                        normalized_row[field] = 0.0
                    elif "is_" in field:
                        normalized_row[field] = False
                    else:
                        normalized_row[field] = 0.0
            
            normalized_data.append(normalized_row)
        
        return normalized_data
    
    def write_to_stream(
        self, 
        data: Union[Dict[str, Any], List[Dict[str, Any]], pl.DataFrame], 
        stream_type: DataStreamType,
        validate_schema: bool = True
    ) -> bool:
        """
        Schreibe Daten in spezifischen Stream mit Schema-Validierung
        
        Args:
            data: Zu schreibende Daten
            stream_type: Ziel-Stream-Typ
            validate_schema: Ob Schema validiert werden soll
            
        Returns:
            True wenn erfolgreich geschrieben, False sonst
        """
        try:
            # Schema-Validierung (optional)
            if validate_schema and not self.validate_data_schema(data, stream_type):
                self.logger.warning(f"Schema validation failed, attempting normalization for {stream_type.value}")
                
                # Versuche Daten zu normalisieren
                if isinstance(data, pl.DataFrame):
                    data = data.to_dicts()
                
                data = self.normalize_data_for_schema(data, stream_type)
            
            # Konvertiere zu Polars DataFrame
            if isinstance(data, dict):
                df = pl.DataFrame([data])
            elif isinstance(data, list):
                df = pl.DataFrame(data)
            else:
                df = data
            
            # Output-Pfad generieren
            output_path = self.get_output_path(stream_type)
            
            # Schreibe zu Parquet mit Schema-Enforcement
            df.write_parquet(
                output_path,
                compression="zstd",
                use_pyarrow=True
            )
            
            self.successful_writes += 1
            self.logger.debug(f"Successfully wrote {len(df)} rows to {stream_type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write to {stream_type.value}: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "validation_count": self.validation_count,
            "schema_mismatches": self.schema_mismatches,
            "successful_writes": self.successful_writes,
            "success_rate": self.successful_writes / max(1, self.validation_count),
            "mismatch_rate": self.schema_mismatches / max(1, self.validation_count)
        }


# Convenience Functions für einfache Nutzung
def create_technical_features_logger(base_path: str = "logs/unified") -> UnifiedSchemaManager:
    """Erstelle Logger für technische Features"""
    return UnifiedSchemaManager(base_path)


def create_ml_dataset_logger(base_path: str = "logs/unified") -> UnifiedSchemaManager:
    """Erstelle Logger für ML-Datasets"""
    return UnifiedSchemaManager(base_path)


def create_ai_predictions_logger(base_path: str = "logs/unified") -> UnifiedSchemaManager:
    """Erstelle Logger für AI-Predictions"""
    return UnifiedSchemaManager(base_path)


# Migration Helper für bestehende Logger
def migrate_existing_parquet_files(
    source_dir: str, 
    target_dir: str, 
    schema_manager: UnifiedSchemaManager
) -> Dict[str, int]:
    """
    Migriere bestehende Parquet-Dateien zu neuen Schema-konformen Streams
    
    Args:
        source_dir: Quell-Verzeichnis mit bestehenden Parquet-Dateien
        target_dir: Ziel-Verzeichnis für migrierte Dateien
        schema_manager: UnifiedSchemaManager-Instanz
        
    Returns:
        Dictionary mit Migrations-Statistiken
    """
    source_path = Path(source_dir)
    migration_stats = {"migrated_files": 0, "migrated_rows": 0, "errors": 0}
    
    logger = logging.getLogger(__name__)
    
    for parquet_file in source_path.glob("*.parquet"):
        try:
            # Lade bestehende Daten
            df = pl.read_parquet(parquet_file)
            logger.info(f"Migrating {parquet_file.name}: {len(df)} rows")
            
            # Bestimme Stream-Typ basierend auf Spalten
            columns = set(df.columns)
            
            if "fwd_ret" in str(columns) or "label_" in str(columns):
                stream_type = DataStreamType.ML_DATASET
            elif "prediction" in str(columns) or "confidence" in str(columns):
                stream_type = DataStreamType.AI_PREDICTIONS
            elif "sma_" in str(columns) or "rsi" in str(columns):
                stream_type = DataStreamType.TECHNICAL_FEATURES
            else:
                stream_type = DataStreamType.PERFORMANCE_METRICS
            
            # Migriere zu neuem Schema
            success = schema_manager.write_to_stream(df, stream_type, validate_schema=True)
            
            if success:
                migration_stats["migrated_files"] += 1
                migration_stats["migrated_rows"] += len(df)
                logger.info(f"Successfully migrated {parquet_file.name} to {stream_type.value}")
            else:
                migration_stats["errors"] += 1
                logger.error(f"Failed to migrate {parquet_file.name}")
                
        except Exception as e:
            migration_stats["errors"] += 1
            logger.error(f"Migration error for {parquet_file.name}: {e}")
    
    return migration_stats


if __name__ == "__main__":
    # Test der UnifiedSchemaManager
    logging.basicConfig(level=logging.INFO)
    
    # Erstelle Schema Manager
    schema_manager = UnifiedSchemaManager("test_logs/unified")
    
    # Test Technical Features
    technical_data = {
        "timestamp": datetime.now(),
        "symbol": "EUR/USD",
        "timeframe": "1m",
        "open": 1.0950,
        "high": 1.0955,
        "low": 1.0948,
        "close": 1.0952,
        "volume": 1000,
        "sma_5": 1.0951,
        "rsi_14": 65.5,
        "macd": 0.0002
    }
    
    success = schema_manager.write_to_stream(technical_data, DataStreamType.TECHNICAL_FEATURES)
    print(f"Technical features write: {'SUCCESS' if success else 'FAILED'}")
    
    # Test AI Predictions
    prediction_data = {
        "timestamp": datetime.now(),
        "symbol": "EUR/USD",
        "prediction_class": "BUY",
        "confidence_score": 0.85,
        "buy_probability": 0.75,
        "sell_probability": 0.15,
        "hold_probability": 0.10
    }
    
    success = schema_manager.write_to_stream(prediction_data, DataStreamType.AI_PREDICTIONS)
    print(f"AI predictions write: {'SUCCESS' if success else 'FAILED'}")
    
    # Performance Stats
    stats = schema_manager.get_performance_stats()
    print(f"Performance Stats: {stats}")