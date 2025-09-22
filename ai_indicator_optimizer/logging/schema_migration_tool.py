#!/usr/bin/env python3
"""
🧩 BAUSTEIN A1: Schema Migration Tool
Migrations-Tool für bestehende Parquet-Dateien zum neuen Schema-System

Funktionen:
- Automatische Migration bestehender Parquet-Dateien
- Schema-Analyse und -Mapping
- Backup-Erstellung vor Migration
- Validierung nach Migration
- Rollback-Funktionalität
"""

import polars as pl
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import json

from .unified_schema_manager import UnifiedSchemaManager, DataStreamType


class SchemaMigrationTool:
    """
    🧩 BAUSTEIN A1: Schema Migration Tool
    
    Migriert bestehende Parquet-Dateien zum neuen Unified Schema System:
    1. Analysiert bestehende Schemas
    2. Mappt zu neuen Stream-Typen
    3. Erstellt Backups
    4. Führt Migration durch
    5. Validiert Ergebnisse
    """
    
    def __init__(self, source_dir: str, target_dir: str, backup_dir: Optional[str] = None):
        """
        Initialize Schema Migration Tool
        
        Args:
            source_dir: Verzeichnis mit bestehenden Parquet-Dateien
            target_dir: Ziel-Verzeichnis für migrierte Dateien
            backup_dir: Backup-Verzeichnis (optional)
        """
        self.source_path = Path(source_dir)
        self.target_path = Path(target_dir)
        self.backup_path = Path(backup_dir) if backup_dir else self.source_path / "backup"
        
        # Erstelle Verzeichnisse
        self.target_path.mkdir(parents=True, exist_ok=True)
        self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Schema Manager
        self.schema_manager = UnifiedSchemaManager(str(self.target_path))
        
        # Migration Stats
        self.migration_stats = {
            "files_analyzed": 0,
            "files_migrated": 0,
            "files_failed": 0,
            "total_rows_migrated": 0,
            "schema_mappings": {},
            "errors": []
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SchemaMigrationTool initialized: {source_dir} -> {target_dir}")
    
    def analyze_existing_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Analysiere bestehende Parquet-Dateien und ihre Schemas
        
        Returns:
            Dictionary mit Schema-Analyse pro Datei
        """
        schema_analysis = {}
        
        for parquet_file in self.source_path.glob("*.parquet"):
            try:
                # Lade Datei-Metadaten
                df = pl.read_parquet(parquet_file)
                
                analysis = {
                    "file_path": str(parquet_file),
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                    "file_size_mb": parquet_file.stat().st_size / (1024 * 1024),
                    "suggested_stream_type": self._suggest_stream_type(df.columns),
                    "schema_compatibility": self._check_schema_compatibility(df.columns)
                }
                
                schema_analysis[parquet_file.name] = analysis
                self.migration_stats["files_analyzed"] += 1
                
                self.logger.info(f"Analyzed {parquet_file.name}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                error_msg = f"Failed to analyze {parquet_file.name}: {e}"
                self.logger.error(error_msg)
                self.migration_stats["errors"].append(error_msg)
        
        return schema_analysis
    
    def _suggest_stream_type(self, columns: List[str]) -> DataStreamType:
        """
        Schlage Stream-Typ basierend auf Spalten vor
        
        Args:
            columns: Liste der Spalten-Namen
            
        Returns:
            Vorgeschlagener DataStreamType
        """
        columns_set = set(columns)
        
        # ML Dataset Indicators
        ml_indicators = {"fwd_ret", "label_", "feature_label", "forward_return"}
        if any(indicator in str(columns_set).lower() for indicator in ml_indicators):
            return DataStreamType.ML_DATASET
        
        # AI Predictions Indicators
        ai_indicators = {"prediction", "confidence", "probability", "model_"}
        if any(indicator in str(columns_set).lower() for indicator in ai_indicators):
            return DataStreamType.AI_PREDICTIONS
        
        # Performance Metrics Indicators
        perf_indicators = {"duration_ms", "throughput", "cpu_usage", "memory_usage"}
        if any(indicator in str(columns_set).lower() for indicator in perf_indicators):
            return DataStreamType.PERFORMANCE_METRICS
        
        # Default: Technical Features
        return DataStreamType.TECHNICAL_FEATURES
    
    def _check_schema_compatibility(self, columns: List[str]) -> Dict[str, Any]:
        """
        Prüfe Schema-Kompatibilität mit allen Stream-Typen
        
        Args:
            columns: Liste der Spalten-Namen
            
        Returns:
            Kompatibilitäts-Analyse
        """
        columns_set = set(columns)
        compatibility = {}
        
        for stream_type in DataStreamType:
            expected_fields = set(self.schema_manager.SCHEMAS[stream_type])
            
            # Berechne Überschneidung
            intersection = columns_set & expected_fields
            coverage = len(intersection) / len(expected_fields) if expected_fields else 0
            missing = expected_fields - columns_set
            extra = columns_set - expected_fields
            
            compatibility[stream_type.value] = {
                "coverage": coverage,
                "matching_fields": len(intersection),
                "missing_fields": list(missing),
                "extra_fields": list(extra),
                "is_compatible": coverage >= 0.5  # 50% Mindest-Überschneidung
            }
        
        return compatibility
    
    def create_backup(self) -> bool:
        """
        Erstelle Backup aller Parquet-Dateien
        
        Returns:
            True wenn Backup erfolgreich, False sonst
        """
        try:
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_backup = self.backup_path / f"backup_{backup_timestamp}"
            timestamped_backup.mkdir(parents=True, exist_ok=True)
            
            backup_count = 0
            for parquet_file in self.source_path.glob("*.parquet"):
                backup_file = timestamped_backup / parquet_file.name
                shutil.copy2(parquet_file, backup_file)
                backup_count += 1
            
            # Erstelle Backup-Manifest
            manifest = {
                "backup_timestamp": backup_timestamp,
                "source_directory": str(self.source_path),
                "files_backed_up": backup_count,
                "backup_files": [f.name for f in timestamped_backup.glob("*.parquet")]
            }
            
            manifest_file = timestamped_backup / "backup_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Backup created: {backup_count} files in {timestamped_backup}")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return False
    
    def migrate_file(
        self, 
        source_file: Path, 
        target_stream_type: DataStreamType,
        validate_after_migration: bool = True
    ) -> bool:
        """
        Migriere einzelne Parquet-Datei zu neuem Schema
        
        Args:
            source_file: Quell-Parquet-Datei
            target_stream_type: Ziel-Stream-Typ
            validate_after_migration: Ob nach Migration validiert werden soll
            
        Returns:
            True wenn Migration erfolgreich, False sonst
        """
        try:
            # Lade Quell-Daten
            df = pl.read_parquet(source_file)
            original_row_count = len(df)
            
            self.logger.info(f"Migrating {source_file.name} to {target_stream_type.value}: {original_row_count} rows")
            
            # Schreibe zu neuem Schema-Stream
            success = self.schema_manager.write_to_stream(
                df, 
                target_stream_type, 
                validate_schema=True
            )
            
            if success:
                # Validierung nach Migration (optional)
                if validate_after_migration:
                    validation_success = self._validate_migrated_data(
                        df, target_stream_type, original_row_count
                    )
                    if not validation_success:
                        self.logger.warning(f"Post-migration validation failed for {source_file.name}")
                
                self.migration_stats["files_migrated"] += 1
                self.migration_stats["total_rows_migrated"] += original_row_count
                self.migration_stats["schema_mappings"][source_file.name] = target_stream_type.value
                
                self.logger.info(f"Successfully migrated {source_file.name}")
                return True
            else:
                self.migration_stats["files_failed"] += 1
                error_msg = f"Failed to migrate {source_file.name} to {target_stream_type.value}"
                self.logger.error(error_msg)
                self.migration_stats["errors"].append(error_msg)
                return False
                
        except Exception as e:
            self.migration_stats["files_failed"] += 1
            error_msg = f"Migration error for {source_file.name}: {e}"
            self.logger.error(error_msg)
            self.migration_stats["errors"].append(error_msg)
            return False
    
    def _validate_migrated_data(
        self, 
        original_df: pl.DataFrame, 
        stream_type: DataStreamType, 
        expected_row_count: int
    ) -> bool:
        """
        Validiere migrierte Daten
        
        Args:
            original_df: Original DataFrame
            stream_type: Stream-Typ der Migration
            expected_row_count: Erwartete Anzahl Zeilen
            
        Returns:
            True wenn Validierung erfolgreich, False sonst
        """
        try:
            # Lade migrierte Daten
            migrated_file = self.schema_manager.get_output_path(stream_type)
            if not migrated_file.exists():
                self.logger.error(f"Migrated file not found: {migrated_file}")
                return False
            
            migrated_df = pl.read_parquet(migrated_file)
            
            # Validiere Zeilen-Anzahl
            if len(migrated_df) < expected_row_count:
                self.logger.warning(f"Row count mismatch: expected {expected_row_count}, got {len(migrated_df)}")
                return False
            
            # Validiere Schema-Konformität
            is_valid = self.schema_manager.validate_data_schema(migrated_df, stream_type)
            if not is_valid:
                self.logger.warning(f"Schema validation failed for migrated data")
                return False
            
            self.logger.debug(f"Migration validation successful: {len(migrated_df)} rows")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration validation error: {e}")
            return False
    
    def run_full_migration(
        self, 
        create_backup: bool = True,
        auto_suggest_stream_types: bool = True
    ) -> Dict[str, Any]:
        """
        Führe vollständige Migration aller Parquet-Dateien durch
        
        Args:
            create_backup: Ob Backup erstellt werden soll
            auto_suggest_stream_types: Ob Stream-Typen automatisch vorgeschlagen werden sollen
            
        Returns:
            Migrations-Ergebnisse
        """
        self.logger.info("Starting full migration process...")
        
        # 1. Backup erstellen (optional)
        if create_backup:
            backup_success = self.create_backup()
            if not backup_success:
                self.logger.error("Backup creation failed, aborting migration")
                return self.migration_stats
        
        # 2. Schema-Analyse
        schema_analysis = self.analyze_existing_schemas()
        
        # 3. Migration durchführen
        for filename, analysis in schema_analysis.items():
            source_file = Path(analysis["file_path"])
            
            if auto_suggest_stream_types:
                target_stream_type = analysis["suggested_stream_type"]
            else:
                # Verwende besten kompatiblen Stream-Typ
                best_compatibility = max(
                    analysis["schema_compatibility"].items(),
                    key=lambda x: x[1]["coverage"]
                )
                target_stream_type = DataStreamType(best_compatibility[0])
            
            # Migriere Datei
            self.migrate_file(source_file, target_stream_type)
        
        # 4. Erstelle Migrations-Report
        self._create_migration_report(schema_analysis)
        
        self.logger.info(f"Migration completed: {self.migration_stats['files_migrated']} files migrated")
        return self.migration_stats
    
    def _create_migration_report(self, schema_analysis: Dict[str, Dict[str, Any]]) -> None:
        """Erstelle detaillierten Migrations-Report"""
        report = {
            "migration_timestamp": datetime.now().isoformat(),
            "source_directory": str(self.source_path),
            "target_directory": str(self.target_path),
            "backup_directory": str(self.backup_path),
            "migration_stats": self.migration_stats,
            "schema_analysis": schema_analysis,
            "schema_manager_stats": self.schema_manager.get_performance_stats()
        }
        
        report_file = self.target_path / "migration_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Migration report created: {report_file}")


def run_schema_migration(
    source_dir: str = "logs",
    target_dir: str = "logs/unified",
    backup_dir: Optional[str] = None,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Convenience-Funktion für Schema-Migration
    
    Args:
        source_dir: Quell-Verzeichnis mit bestehenden Parquet-Dateien
        target_dir: Ziel-Verzeichnis für migrierte Dateien
        backup_dir: Backup-Verzeichnis (optional)
        create_backup: Ob Backup erstellt werden soll
        
    Returns:
        Migrations-Ergebnisse
    """
    migration_tool = SchemaMigrationTool(source_dir, target_dir, backup_dir)
    return migration_tool.run_full_migration(create_backup=create_backup)


if __name__ == "__main__":
    # Test der Schema-Migration
    logging.basicConfig(level=logging.INFO)
    
    # Führe Migration durch
    results = run_schema_migration(
        source_dir="logs",
        target_dir="logs/unified_test",
        create_backup=True
    )
    
    print(f"Migration Results: {results}")