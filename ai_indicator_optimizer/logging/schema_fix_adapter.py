#!/usr/bin/env python3
"""
Schema-Problem-Fix Adapter
Temporärer Adapter für bestehende Logger bis vollständige Migration
"""

from pathlib import Path
from datetime import datetime
import polars as pl

class SchemaFixAdapter:
    """Adapter für bestehende Logger mit Schema-Problem-Fix"""
    
    def __init__(self, base_path="logs/unified"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_technical_features(self, data):
        """Schreibe technische Features zu separatem Stream"""
        date_suffix = datetime.now().strftime("%Y%m%d")
        file_path = self.base_path / f"technical_features_{date_suffix}.parquet"
        
        if isinstance(data, dict):
            df = pl.DataFrame([data])
        else:
            df = data
        
        # Append oder Create
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_parquet(file_path, compression="zstd")
        else:
            df.write_parquet(file_path, compression="zstd")
    
    def write_ml_dataset(self, data):
        """Schreibe ML-Dataset zu separatem Stream"""
        date_suffix = datetime.now().strftime("%Y%m%d")
        file_path = self.base_path / f"ml_dataset_{date_suffix}.parquet"
        
        if isinstance(data, dict):
            df = pl.DataFrame([data])
        else:
            df = data
        
        # Append oder Create
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_parquet(file_path, compression="zstd")
        else:
            df.write_parquet(file_path, compression="zstd")

# Global Adapter Instance
schema_fix_adapter = SchemaFixAdapter()
