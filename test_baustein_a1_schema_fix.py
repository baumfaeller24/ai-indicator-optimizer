#!/usr/bin/env python3
"""
🧩 BAUSTEIN A1 TEST: Schema-Problem-Behebung
Test-Suite für UnifiedSchemaManager und Schema-Migration

Testet:
1. Schema-Validierung für verschiedene Stream-Typen
2. Separate Logging-Streams
3. Migration bestehender Parquet-Dateien
4. Performance und Kompatibilität
"""

import pytest
import polars as pl
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Import der neuen Komponenten
from ai_indicator_optimizer.logging.unified_schema_manager import (
    UnifiedSchemaManager, 
    DataStreamType,
    create_technical_features_logger,
    migrate_existing_parquet_files
)
from ai_indicator_optimizer.logging.schema_migration_tool import (
    SchemaMigrationTool,
    run_schema_migration
)


class TestBausteinA1SchemaFix:
    """
    🧩 BAUSTEIN A1 TEST: Comprehensive Schema Fix Testing
    """
    
    def setup_method(self):
        """Setup für jeden Test"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Erstelle Test-Verzeichnisse
        self.source_dir = self.temp_path / "source"
        self.target_dir = self.temp_path / "target"
        self.backup_dir = self.temp_path / "backup"
        
        for dir_path in [self.source_dir, self.target_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Schema Manager
        self.schema_manager = UnifiedSchemaManager(str(self.target_dir))
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def teardown_method(self):
        """Cleanup nach jedem Test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_parquet_files(self):
        """Erstelle Test-Parquet-Dateien mit verschiedenen Schemas"""
        
        # 1. Technical Features File (wie IntegratedDatasetLogger)
        technical_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 100,
            "symbol": ["EUR/USD"] * 100,
            "open": [1.095 + i * 0.0001 for i in range(100)],
            "close": [1.096 + i * 0.0001 for i in range(100)],
            "sma_5": [1.0955 + i * 0.0001 for i in range(100)],
            "rsi_14": [50 + i * 0.1 for i in range(100)],
            "macd": [0.0001 + i * 0.00001 for i in range(100)]
        })
        technical_file = self.source_dir / "technical_features.parquet"
        technical_data.write_parquet(technical_file)
        
        # 2. ML Dataset File (wie BarDatasetBuilder)
        ml_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 50,
            "symbol": ["EUR/USD"] * 50,
            "open": [1.095 + i * 0.0001 for i in range(50)],
            "close": [1.096 + i * 0.0001 for i in range(50)],
            "feature_label_fwd_ret_h5": [0.001 + i * 0.0001 for i in range(50)],
            "label_binary": [1 if i % 2 == 0 else 0 for i in range(50)],
            "feature_sma_5": [1.0955 + i * 0.0001 for i in range(50)]
        })
        ml_file = self.source_dir / "ml_dataset.parquet"
        ml_data.write_parquet(ml_file)
        
        # 3. AI Predictions File
        predictions_data = pl.DataFrame({
            "timestamp": [datetime.now()] * 30,
            "symbol": ["EUR/USD"] * 30,
            "prediction_class": ["BUY", "SELL", "HOLD"] * 10,
            "confidence_score": [0.8 + i * 0.01 for i in range(30)],
            "buy_probability": [0.6 + i * 0.01 for i in range(30)],
            "model_name": ["MiniCPM"] * 30
        })
        predictions_file = self.source_dir / "predictions.parquet"
        predictions_data.write_parquet(predictions_file)
        
        return [technical_file, ml_file, predictions_file]
    
    def test_unified_schema_manager_initialization(self):
        """Test: UnifiedSchemaManager Initialisierung"""
        assert self.schema_manager is not None
        assert len(self.schema_manager.SCHEMAS) == 4  # 4 Stream-Typen
        assert DataStreamType.TECHNICAL_FEATURES in self.schema_manager.SCHEMAS
        assert DataStreamType.ML_DATASET in self.schema_manager.SCHEMAS
        assert DataStreamType.AI_PREDICTIONS in self.schema_manager.SCHEMAS
        assert DataStreamType.PERFORMANCE_METRICS in self.schema_manager.SCHEMAS
        
        self.logger.info("✅ UnifiedSchemaManager initialization test passed")
    
    def test_schema_validation(self):
        """Test: Schema-Validierung für verschiedene Stream-Typen"""
        
        # Test Technical Features
        technical_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "open": 1.095,
            "close": 1.096,
            "sma_5": 1.0955,
            "rsi_14": 65.5
        }
        
        is_valid = self.schema_manager.validate_data_schema(
            technical_data, DataStreamType.TECHNICAL_FEATURES
        )
        assert is_valid, "Technical features validation should pass"
        
        # Test AI Predictions
        prediction_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "prediction_class": "BUY",
            "confidence_score": 0.85,
            "buy_probability": 0.75
        }
        
        is_valid = self.schema_manager.validate_data_schema(
            prediction_data, DataStreamType.AI_PREDICTIONS
        )
        assert is_valid, "AI predictions validation should pass"
        
        # Test Invalid Data
        invalid_data = {"random_field": "random_value"}
        is_valid = self.schema_manager.validate_data_schema(
            invalid_data, DataStreamType.TECHNICAL_FEATURES
        )
        assert not is_valid, "Invalid data validation should fail"
        
        self.logger.info("✅ Schema validation tests passed")
    
    def test_separate_logging_streams(self):
        """Test: Separate Logging-Streams für verschiedene Datentypen"""
        
        # Technical Features Stream
        technical_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "sma_5": 1.0955,
            "rsi_14": 65.5,
            "open": 1.095,
            "close": 1.096
        }
        
        success = self.schema_manager.write_to_stream(
            technical_data, DataStreamType.TECHNICAL_FEATURES
        )
        assert success, "Technical features write should succeed"
        
        # AI Predictions Stream
        prediction_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "prediction_class": "BUY",
            "confidence_score": 0.85
        }
        
        success = self.schema_manager.write_to_stream(
            prediction_data, DataStreamType.AI_PREDICTIONS
        )
        assert success, "AI predictions write should succeed"
        
        # Verify separate files created
        tech_file = self.schema_manager.get_output_path(DataStreamType.TECHNICAL_FEATURES)
        pred_file = self.schema_manager.get_output_path(DataStreamType.AI_PREDICTIONS)
        
        assert tech_file.exists(), "Technical features file should exist"
        assert pred_file.exists(), "AI predictions file should exist"
        assert tech_file != pred_file, "Files should be separate"
        
        self.logger.info("✅ Separate logging streams test passed")
    
    def test_data_normalization(self):
        """Test: Daten-Normalisierung für Schema-Kompatibilität"""
        
        # Incomplete data
        incomplete_data = {
            "symbol": "EUR/USD",
            "sma_5": 1.0955
            # Missing required fields like timestamp, open, close
        }
        
        normalized = self.schema_manager.normalize_data_for_schema(
            incomplete_data, DataStreamType.TECHNICAL_FEATURES
        )
        
        assert len(normalized) == 1, "Should return one normalized row"
        assert "timestamp" in normalized[0], "Should add missing timestamp"
        assert "open" in normalized[0], "Should add missing open"
        assert normalized[0]["symbol"] == "EUR/USD", "Should preserve existing data"
        
        self.logger.info("✅ Data normalization test passed")
    
    def test_schema_migration_tool(self):
        """Test: Schema Migration Tool"""
        
        # Erstelle Test-Dateien
        test_files = self.create_test_parquet_files()
        
        # Initialisiere Migration Tool
        migration_tool = SchemaMigrationTool(
            str(self.source_dir),
            str(self.target_dir),
            str(self.backup_dir)
        )
        
        # Analysiere Schemas
        schema_analysis = migration_tool.analyze_existing_schemas()
        
        assert len(schema_analysis) == 3, "Should analyze 3 test files"
        assert "technical_features.parquet" in schema_analysis
        assert "ml_dataset.parquet" in schema_analysis
        assert "predictions.parquet" in schema_analysis
        
        # Test Stream-Typ-Vorschläge
        tech_analysis = schema_analysis["technical_features.parquet"]
        assert tech_analysis["suggested_stream_type"] == DataStreamType.TECHNICAL_FEATURES
        
        ml_analysis = schema_analysis["ml_dataset.parquet"]
        assert ml_analysis["suggested_stream_type"] == DataStreamType.ML_DATASET
        
        pred_analysis = schema_analysis["predictions.parquet"]
        assert pred_analysis["suggested_stream_type"] == DataStreamType.AI_PREDICTIONS
        
        self.logger.info("✅ Schema migration tool test passed")
    
    def test_full_migration_process(self):
        """Test: Vollständiger Migrations-Prozess"""
        
        # Erstelle Test-Dateien
        test_files = self.create_test_parquet_files()
        
        # Führe Migration durch
        migration_results = run_schema_migration(
            source_dir=str(self.source_dir),
            target_dir=str(self.target_dir),
            backup_dir=str(self.backup_dir),
            create_backup=True
        )
        
        # Validiere Ergebnisse
        assert migration_results["files_migrated"] == 3, "Should migrate 3 files"
        assert migration_results["files_failed"] == 0, "Should have no failures"
        assert migration_results["total_rows_migrated"] > 0, "Should migrate rows"
        
        # Validiere migrierte Dateien
        for stream_type in [DataStreamType.TECHNICAL_FEATURES, DataStreamType.ML_DATASET, DataStreamType.AI_PREDICTIONS]:
            output_file = self.target_dir / f"{stream_type.value}_{datetime.now().strftime('%Y%m%d')}.parquet"
            if output_file.exists():
                df = pl.read_parquet(output_file)
                assert len(df) > 0, f"Migrated file {stream_type.value} should have data"
        
        # Validiere Backup
        backup_files = list(self.backup_dir.glob("backup_*/backup_manifest.json"))
        assert len(backup_files) > 0, "Should create backup manifest"
        
        self.logger.info("✅ Full migration process test passed")
    
    def test_performance_stats(self):
        """Test: Performance-Statistiken"""
        
        # Schreibe Test-Daten
        for i in range(10):
            test_data = {
                "timestamp": datetime.now(),
                "symbol": f"TEST{i}",
                "sma_5": 1.0 + i * 0.01,
                "open": 1.0 + i * 0.01,
                "close": 1.0 + i * 0.01
            }
            
            self.schema_manager.write_to_stream(
                test_data, DataStreamType.TECHNICAL_FEATURES
            )
        
        # Hole Performance-Stats
        stats = self.schema_manager.get_performance_stats()
        
        assert stats["validation_count"] >= 10, "Should have validation count"
        assert stats["successful_writes"] >= 10, "Should have successful writes"
        assert "success_rate" in stats, "Should have success rate"
        assert "mismatch_rate" in stats, "Should have mismatch rate"
        
        self.logger.info("✅ Performance stats test passed")
    
    def test_schema_problem_resolution(self):
        """Test: Lösung des ursprünglichen Schema-Problems"""
        
        # Simuliere das ursprüngliche Problem:
        # BarDatasetBuilder-Daten (feature_label_fwd_ret_h5)
        bar_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "feature_label_fwd_ret_h5": 0.001,
            "label_binary": 1,
            "open": 1.095,
            "close": 1.096
        }
        
        # IntegratedDatasetLogger-Daten (feature_sma_5)
        integrated_data = {
            "timestamp": datetime.now(),
            "symbol": "EUR/USD",
            "feature_sma_5": 1.0955,
            "rsi_14": 65.5,
            "open": 1.095,
            "close": 1.096
        }
        
        # Schreibe zu separaten Streams (sollte funktionieren)
        bar_success = self.schema_manager.write_to_stream(
            bar_data, DataStreamType.ML_DATASET
        )
        
        integrated_success = self.schema_manager.write_to_stream(
            integrated_data, DataStreamType.TECHNICAL_FEATURES
        )
        
        assert bar_success, "BarDatasetBuilder data should write successfully"
        assert integrated_success, "IntegratedDatasetLogger data should write successfully"
        
        # Validiere separate Dateien
        ml_file = self.schema_manager.get_output_path(DataStreamType.ML_DATASET)
        tech_file = self.schema_manager.get_output_path(DataStreamType.TECHNICAL_FEATURES)
        
        assert ml_file.exists(), "ML dataset file should exist"
        assert tech_file.exists(), "Technical features file should exist"
        assert ml_file != tech_file, "Files should be separate"
        
        # Validiere Inhalte
        ml_df = pl.read_parquet(ml_file)
        tech_df = pl.read_parquet(tech_file)
        
        assert len(ml_df) > 0, "ML dataset should have data"
        assert len(tech_df) > 0, "Technical features should have data"
        
        self.logger.info("✅ Schema problem resolution test passed")


def run_baustein_a1_tests():
    """Führe alle Baustein A1 Tests durch"""
    
    print("🧩 BAUSTEIN A1 TESTS: Schema-Problem-Behebung")
    print("=" * 60)
    
    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Erstelle Test-Instanz
    test_instance = TestBausteinA1SchemaFix()
    
    try:
        # Führe alle Tests durch
        test_methods = [
            "test_unified_schema_manager_initialization",
            "test_schema_validation",
            "test_separate_logging_streams",
            "test_data_normalization",
            "test_schema_migration_tool",
            "test_full_migration_process",
            "test_performance_stats",
            "test_schema_problem_resolution"
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                test_instance.setup_method()
                getattr(test_instance, test_method)()
                test_instance.teardown_method()
                passed_tests += 1
                print(f"✅ {test_method}")
            except Exception as e:
                print(f"❌ {test_method}: {e}")
                test_instance.teardown_method()
        
        print("=" * 60)
        print(f"🧩 BAUSTEIN A1 TESTS COMPLETED: {passed_tests}/{total_tests} passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - BAUSTEIN A1 READY FOR DEPLOYMENT!")
            return True
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = run_baustein_a1_tests()
    exit(0 if success else 1)