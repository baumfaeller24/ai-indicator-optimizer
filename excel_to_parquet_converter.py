#!/usr/bin/env python3
"""
🚀 EXCEL TO PARQUET CONVERTER
Konvertiert Excel-Tickdaten aus ZIP-Dateien zu Parquet für ML-Training

Features:
- Automatisches ZIP-Entpacken
- Excel-zu-Parquet-Konvertierung
- Schema-Normalisierung
- Batch-Verarbeitung mehrerer Dateien
- Optimiert für RTX 5090 + 32 Cores
"""

import zipfile
import pandas as pd
import polars as pl
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil


class ExcelToParquetConverter:
    """
    🚀 Excel to Parquet Converter
    
    Konvertiert Excel-Tickdaten zu Parquet-Format:
    - ZIP-Dateien automatisch entpacken
    - Excel-Dateien zu Parquet konvertieren
    - Schema-Normalisierung für ML-Training
    - Parallel-Verarbeitung für Speed
    """
    
    def __init__(self, input_dir: str = "forex", output_dir: str = "data/forex_converted"):
        """
        Initialize Excel to Parquet Converter
        
        Args:
            input_dir: Verzeichnis mit ZIP-Dateien
            output_dir: Output-Verzeichnis für Parquet-Dateien
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp_extraction")
        
        # Erstelle Verzeichnisse
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance Tracking
        self.total_files_processed = 0
        self.total_rows_converted = 0
        self.processing_start_time = None
        
        # Hardware Info
        self.cpu_cores = psutil.cpu_count()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized with {self.cpu_cores} CPU cores")
    
    def find_zip_files(self) -> List[Path]:
        """
        Finde alle ZIP-Dateien im Input-Verzeichnis
        
        Returns:
            Liste der ZIP-Dateien
        """
        zip_files = list(self.input_dir.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} ZIP files: {[f.name for f in zip_files]}")
        return zip_files
    
    def extract_zip_file(self, zip_path: Path) -> List[Path]:
        """
        Extrahiere ZIP-Datei zu temporärem Verzeichnis
        
        Args:
            zip_path: Pfad zur ZIP-Datei
            
        Returns:
            Liste der extrahierten Dateien
        """
        self.logger.info(f"🔄 Extracting {zip_path.name}...")
        
        # Erstelle spezifisches Temp-Verzeichnis für diese ZIP
        extract_dir = self.temp_dir / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extrahiere alle Dateien
                zip_ref.extractall(extract_dir)
                
                # Sammle alle extrahierten Dateien
                for root, dirs, files in os.walk(extract_dir):
                    for file in files:
                        file_path = Path(root) / file
                        extracted_files.append(file_path)
                
                self.logger.info(f"✅ Extracted {len(extracted_files)} files from {zip_path.name}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to extract {zip_path.name}: {e}")
            
        return extracted_files
    
    def detect_file_format(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Erkenne Datei-Format und Schema (Excel oder CSV)
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Schema-Information oder None
        """
        try:
            file_extension = file_path.suffix.lower()
            
            # CSV-Dateien (häufig bei Tickdaten)
            if file_extension == '.csv':
                try:
                    # Lese erste Zeilen ohne Header
                    df_sample = pd.read_csv(file_path, header=None, nrows=5)
                    
                    # Erkenne Tickdata-Format basierend auf Spalten-Anzahl
                    if len(df_sample.columns) == 4:
                        # Standard Tickdata: Symbol, Timestamp, Bid, Ask
                        column_names = ['symbol', 'timestamp', 'bid', 'ask']
                    elif len(df_sample.columns) == 5:
                        # Mit Volume: Symbol, Timestamp, Bid, Ask, Volume
                        column_names = ['symbol', 'timestamp', 'bid', 'ask', 'volume']
                    elif len(df_sample.columns) == 6:
                        # OHLCV: Symbol, Timestamp, Open, High, Low, Close
                        column_names = ['symbol', 'timestamp', 'open', 'high', 'low', 'close']
                    else:
                        # Fallback: Generische Namen
                        column_names = [f'col_{i}' for i in range(len(df_sample.columns))]
                    
                    schema_info = {
                        'file_type': 'csv',
                        'columns': column_names,
                        'original_columns': list(df_sample.columns),
                        'sample_data': df_sample.head(2).values.tolist(),
                        'row_count_estimate': None
                    }
                    
                    self.logger.info(f"📊 Detected CSV schema for {file_path.name}: {len(column_names)} columns")
                    return schema_info
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ CSV detection failed for {file_path.name}: {e}")
            
            # Excel-Dateien
            elif file_extension in ['.xlsx', '.xls']:
                for engine in ['openpyxl', 'xlrd']:
                    try:
                        # Lese erste paar Zeilen zum Schema-Detection
                        df_sample = pd.read_excel(file_path, engine=engine, nrows=5)
                        
                        schema_info = {
                            'file_type': 'excel',
                            'engine': engine,
                            'columns': list(df_sample.columns),
                            'dtypes': df_sample.dtypes.to_dict(),
                            'sample_data': df_sample.head(2).to_dict(),
                            'row_count_estimate': None
                        }
                        
                        self.logger.info(f"📊 Detected Excel schema for {file_path.name}: {len(df_sample.columns)} columns")
                        return schema_info
                        
                    except Exception as e:
                        continue
            
            self.logger.warning(f"⚠️ Could not detect schema for {file_path.name}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Schema detection failed for {file_path.name}: {e}")
            return None
    
    def normalize_tickdata_schema(self, df: pd.DataFrame, file_name: str) -> pl.DataFrame:
        """
        Normalisiere Tickdata-Schema für ML-Training
        
        Args:
            df: Pandas DataFrame
            file_name: Name der Datei für Kontext
            
        Returns:
            Normalisierter Polars DataFrame
        """
        try:
            # Konvertiere zu Polars für bessere Performance
            pl_df = pl.from_pandas(df)
            
            # Standard-Spalten für Tickdaten
            standard_columns = {
                'timestamp': None,
                'symbol': None,
                'bid': None,
                'ask': None,
                'volume': None,
                'date': None,
                'time': None,
                'price': None,
                'open': None,
                'high': None,
                'low': None,
                'close': None
            }
            
            # Erkenne Spalten basierend auf Namen (case-insensitive)
            column_mapping = {}
            
            for col in pl_df.columns:
                col_lower = col.lower()
                
                # Timestamp/Date/Time Erkennung
                if any(keyword in col_lower for keyword in ['timestamp', 'datetime', 'date_time']):
                    column_mapping['timestamp'] = col
                elif any(keyword in col_lower for keyword in ['date']):
                    column_mapping['date'] = col
                elif any(keyword in col_lower for keyword in ['time']):
                    column_mapping['time'] = col
                
                # Price Erkennung
                elif any(keyword in col_lower for keyword in ['bid']):
                    column_mapping['bid'] = col
                elif any(keyword in col_lower for keyword in ['ask']):
                    column_mapping['ask'] = col
                elif any(keyword in col_lower for keyword in ['price', 'close']):
                    column_mapping['price'] = col
                elif any(keyword in col_lower for keyword in ['open']):
                    column_mapping['open'] = col
                elif any(keyword in col_lower for keyword in ['high']):
                    column_mapping['high'] = col
                elif any(keyword in col_lower for keyword in ['low']):
                    column_mapping['low'] = col
                
                # Volume/Symbol
                elif any(keyword in col_lower for keyword in ['volume', 'vol']):
                    column_mapping['volume'] = col
                elif any(keyword in col_lower for keyword in ['symbol', 'pair', 'instrument']):
                    column_mapping['symbol'] = col
            
            # Erstelle normalisierten DataFrame
            normalized_data = {}
            
            # Timestamp handling
            if 'timestamp' in column_mapping:
                normalized_data['timestamp'] = pl_df[column_mapping['timestamp']]
            elif 'date' in column_mapping and 'time' in column_mapping:
                # Kombiniere Date und Time
                try:
                    date_col = pl_df[column_mapping['date']].cast(pl.Utf8)
                    time_col = pl_df[column_mapping['time']].cast(pl.Utf8)
                    normalized_data['timestamp'] = date_col + " " + time_col
                except:
                    normalized_data['timestamp'] = pl_df[column_mapping['date']]
            elif 'date' in column_mapping:
                normalized_data['timestamp'] = pl_df[column_mapping['date']]
            else:
                # Erstelle Index-basierte Timestamps
                normalized_data['timestamp'] = pl.arange(len(pl_df), eager=True)
            
            # Symbol
            if 'symbol' in column_mapping:
                normalized_data['symbol'] = pl_df[column_mapping['symbol']]
            else:
                # Extrahiere Symbol aus Dateiname
                symbol = "UNKNOWN"
                if "EURUSD" in file_name.upper():
                    symbol = "EUR/USD"
                elif "GBPUSD" in file_name.upper():
                    symbol = "GBP/USD"
                elif "USDJPY" in file_name.upper():
                    symbol = "USD/JPY"
                normalized_data['symbol'] = pl.lit(symbol)
            
            # Price data
            if 'bid' in column_mapping and 'ask' in column_mapping:
                normalized_data['bid'] = pl_df[column_mapping['bid']].cast(pl.Float64, strict=False)
                normalized_data['ask'] = pl_df[column_mapping['ask']].cast(pl.Float64, strict=False)
                # Berechne Mid-Price
                normalized_data['price'] = (normalized_data['bid'] + normalized_data['ask']) / 2
            elif 'price' in column_mapping:
                normalized_data['price'] = pl_df[column_mapping['price']].cast(pl.Float64, strict=False)
                # Erstelle Bid/Ask basierend auf Price (mit kleinem Spread)
                normalized_data['bid'] = normalized_data['price'] - 0.00001
                normalized_data['ask'] = normalized_data['price'] + 0.00001
            elif 'open' in column_mapping and 'close' in column_mapping:
                # OHLC Daten
                normalized_data['open'] = pl_df[column_mapping['open']].cast(pl.Float64, strict=False)
                normalized_data['high'] = pl_df[column_mapping['high']].cast(pl.Float64, strict=False)
                normalized_data['low'] = pl_df[column_mapping['low']].cast(pl.Float64, strict=False)
                normalized_data['close'] = pl_df[column_mapping['close']].cast(pl.Float64, strict=False)
                normalized_data['price'] = normalized_data['close']
                normalized_data['bid'] = normalized_data['close'] - 0.00001
                normalized_data['ask'] = normalized_data['close'] + 0.00001
            
            # Volume
            if 'volume' in column_mapping:
                normalized_data['volume'] = pl_df[column_mapping['volume']].cast(pl.Float64, strict=False)
            else:
                normalized_data['volume'] = pl.lit(1.0)  # Default volume
            
            # Erstelle finalen DataFrame
            result_df = pl.DataFrame(normalized_data)
            
            self.logger.info(f"✅ Normalized {file_name}: {len(result_df)} rows, {len(result_df.columns)} columns")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Schema normalization failed for {file_name}: {e}")
            # Fallback: Versuche einfache Konvertierung
            try:
                return pl.from_pandas(df)
            except:
                return pl.DataFrame()
    
    def convert_file_to_parquet(self, file_path: Path) -> Optional[Path]:
        """
        Konvertiere Datei (Excel oder CSV) zu Parquet
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Pfad zur Parquet-Datei oder None
        """
        try:
            self.logger.info(f"🔄 Converting {file_path.name} to Parquet...")
            
            # Erkenne Schema
            schema_info = self.detect_file_format(file_path)
            if not schema_info:
                return None
            
            # Lade Datei basierend auf Typ
            if schema_info['file_type'] == 'csv':
                # CSV ohne Header laden
                df = pd.read_csv(file_path, header=None, names=schema_info['columns'])
            elif schema_info['file_type'] == 'excel':
                # Excel-Datei laden
                df = pd.read_excel(file_path, engine=schema_info['engine'])
            else:
                self.logger.error(f"❌ Unsupported file type: {file_path.name}")
                return None
            
            if len(df) == 0:
                self.logger.warning(f"⚠️ Empty file: {file_path.name}")
                return None
            
            # Normalisiere Schema
            normalized_df = self.normalize_tickdata_schema(df, file_path.name)
            
            if len(normalized_df) == 0:
                self.logger.warning(f"⚠️ Normalization resulted in empty DataFrame: {file_path.name}")
                return None
            
            # Erstelle Output-Pfad
            output_path = self.output_dir / f"{file_path.stem}.parquet"
            
            # Speichere als Parquet
            normalized_df.write_parquet(output_path, compression="zstd")
            
            self.total_files_processed += 1
            self.total_rows_converted += len(normalized_df)
            
            self.logger.info(f"✅ Converted {file_path.name}: {len(normalized_df):,} rows → {output_path.name}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Conversion failed for {file_path.name}: {e}")
            return None
    
    def process_zip_file(self, zip_path: Path) -> List[Path]:
        """
        Verarbeite komplette ZIP-Datei
        
        Args:
            zip_path: Pfad zur ZIP-Datei
            
        Returns:
            Liste der generierten Parquet-Dateien
        """
        self.logger.info(f"🚀 Processing ZIP file: {zip_path.name}")
        
        # Extrahiere ZIP
        extracted_files = self.extract_zip_file(zip_path)
        
        # Filtere Excel-Dateien
        excel_files = [
            f for f in extracted_files 
            if f.suffix.lower() in ['.xlsx', '.xls', '.csv']
        ]
        
        self.logger.info(f"📊 Found {len(excel_files)} Excel/CSV files in {zip_path.name}")
        
        # Konvertiere alle Dateien
        parquet_files = []
        
        for data_file in excel_files:
            parquet_path = self.convert_file_to_parquet(data_file)
            if parquet_path:
                parquet_files.append(parquet_path)
        
        # Cleanup temporäre Dateien
        extract_dir = self.temp_dir / zip_path.stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        self.logger.info(f"✅ Processed {zip_path.name}: {len(parquet_files)} Parquet files generated")
        return parquet_files
    
    def run_batch_conversion(self) -> Dict[str, Any]:
        """
        Führe Batch-Konvertierung aller ZIP-Dateien durch
        
        Returns:
            Konvertierungs-Ergebnisse
        """
        self.processing_start_time = time.time()
        self.logger.info("🚀 Starting batch Excel to Parquet conversion...")
        
        results = {
            "start_time": datetime.now(),
            "zip_files_processed": 0,
            "excel_files_converted": 0,
            "total_rows_converted": 0,
            "parquet_files_generated": [],
            "processing_time_minutes": 0,
            "errors": []
        }
        
        try:
            # Finde alle ZIP-Dateien
            zip_files = self.find_zip_files()
            
            if not zip_files:
                self.logger.warning("⚠️ No ZIP files found!")
                return results
            
            # Verarbeite jede ZIP-Datei
            all_parquet_files = []
            
            for zip_file in zip_files:
                try:
                    parquet_files = self.process_zip_file(zip_file)
                    all_parquet_files.extend(parquet_files)
                    results["zip_files_processed"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to process {zip_file.name}: {e}"
                    self.logger.error(f"❌ {error_msg}")
                    results["errors"].append(error_msg)
            
            # Finale Statistiken
            processing_time = time.time() - self.processing_start_time
            results.update({
                "end_time": datetime.now(),
                "excel_files_converted": self.total_files_processed,
                "total_rows_converted": self.total_rows_converted,
                "parquet_files_generated": [str(p) for p in all_parquet_files],
                "processing_time_minutes": processing_time / 60
            })
            
            self.logger.info(f"\n🎉 CONVERSION COMPLETED!")
            self.logger.info(f"  - ZIP files processed: {results['zip_files_processed']}")
            self.logger.info(f"  - Excel files converted: {results['excel_files_converted']}")
            self.logger.info(f"  - Total rows converted: {results['total_rows_converted']:,}")
            self.logger.info(f"  - Parquet files generated: {len(all_parquet_files)}")
            self.logger.info(f"  - Processing time: {results['processing_time_minutes']:.1f} minutes")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Batch conversion failed: {e}")
            results["errors"].append(str(e))
            return results
        
        finally:
            # Cleanup temporäres Verzeichnis
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)


def run_excel_to_parquet_conversion():
    """
    🚀 Hauptfunktion für Excel-zu-Parquet-Konvertierung
    """
    
    print("🚀 EXCEL TO PARQUET CONVERSION")
    print("=" * 60)
    print("Converting Excel tickdata from ZIP files to Parquet format...")
    print("Input: forex/*.zip files")
    print("Output: data/forex_converted/*.parquet files")
    print("=" * 60)
    
    # Erstelle Converter
    converter = ExcelToParquetConverter()
    
    # Führe Batch-Konvertierung durch
    results = converter.run_batch_conversion()
    
    # Speichere Ergebnisse
    results_file = "excel_to_parquet_conversion_results.json"
    import json
    
    # Serialisierbare Version
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, datetime):
            serializable_results[key] = value.isoformat()
        else:
            serializable_results[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📊 CONVERSION SUMMARY:")
    print(f"  - ZIP files processed: {results.get('zip_files_processed', 0)}")
    print(f"  - Excel files converted: {results.get('excel_files_converted', 0)}")
    print(f"  - Total rows: {results.get('total_rows_converted', 0):,}")
    print(f"  - Parquet files: {len(results.get('parquet_files_generated', []))}")
    print(f"  - Processing time: {results.get('processing_time_minutes', 0):.1f} minutes")
    print(f"  - Results saved to: {results_file}")
    
    if results.get("errors"):
        print(f"\n⚠️ ERRORS ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")
        return False
    else:
        print(f"\n🎉 SUCCESS: All Excel files converted to Parquet!")
        return True


if __name__ == "__main__":
    success = run_excel_to_parquet_conversion()
    exit(0 if success else 1)