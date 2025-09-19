#!/usr/bin/env python3
"""
Enhanced Feature Prediction Logger
ChatGPT-Verbesserung: Strukturiertes Logging für AI-Predictions mit Parquet-Export
Basierend auf tradingbeispiele.md
"""

import polars as pl
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path
import logging


class FeaturePredictionLogger:
    """
    ChatGPT Enhancement: Feature & Prediction Logger mit Parquet-Export
    
    Features:
    - Buffer-System für Performance-Optimierung
    - Automatische Parquet-Flush-Funktionalität
    - Timestamp-basierte Logging mit Instrument-ID-Tracking
    - Kompression (zstd) für optimale Speichernutzung
    """
    
    def __init__(
        self, 
        output_path: str = "logs/ai_features.parquet", 
        buffer_size: int = 5000,
        compression: str = "zstd",
        auto_flush: bool = True
    ):
        """
        Initialize Feature Prediction Logger
        
        Args:
            output_path: Path für Parquet-Output
            buffer_size: Anzahl Einträge vor automatischem Flush
            compression: Kompression für Parquet (zstd, snappy, gzip)
            auto_flush: Automatisches Flushen bei Buffer-Überlauf
        """
        self.output_path = Path(output_path)
        self.buffer_size = buffer_size
        self.compression = compression
        self.auto_flush = auto_flush
        
        # Buffer für Performance-Optimierung
        self.buffer: List[Dict[str, Any]] = []
        self.total_entries = 0
        
        # Logging Setup
        self.logger = logging.getLogger(__name__)
        
        # Erstelle Output-Verzeichnis falls nicht vorhanden
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"FeaturePredictionLogger initialized: {output_path}")
        self.logger.info(f"Buffer size: {buffer_size}, Compression: {compression}")
    
    def log(
        self, 
        *, 
        ts_ns: int, 
        instrument: str, 
        features: Dict[str, Any], 
        prediction: Dict[str, Any],
        confidence_score: Optional[float] = None,
        risk_score: Optional[float] = None,
        market_regime: Optional[str] = None
    ) -> None:
        """
        Log Feature und Prediction Entry
        
        Args:
            ts_ns: Timestamp in Nanosekunden
            instrument: Trading-Instrument (z.B. "EUR/USD")
            features: Feature-Dictionary mit allen extrahierten Features
            prediction: AI-Prediction mit action, confidence, reasoning
            confidence_score: Enhanced Confidence Score (optional)
            risk_score: Risk Assessment Score (optional)
            market_regime: Detected Market Regime (optional)
        """
        try:
            # Erstelle strukturierten Log-Entry
            entry = {
                # Timestamp & Instrument
                "ts_ns": ts_ns,
                "timestamp": datetime.utcfromtimestamp(ts_ns / 1e9).isoformat(),
                "instrument": instrument,
                
                # Features (mit Präfix für bessere Organisation)
                **{f"feature_{k}": v for k, v in features.items()},
                
                # Predictions
                "pred_action": prediction.get("action"),
                "pred_confidence": prediction.get("confidence"),
                "pred_reasoning": prediction.get("reasoning"),
                
                # Enhanced Scores (ChatGPT-Verbesserung)
                "enhanced_confidence": confidence_score,
                "risk_score": risk_score,
                "market_regime": market_regime,
                
                # Metadata
                "log_entry_id": self.total_entries
            }
            
            # Füge zu Buffer hinzu
            self.buffer.append(entry)
            self.total_entries += 1
            
            # Auto-Flush bei Buffer-Überlauf
            if self.auto_flush and len(self.buffer) >= self.buffer_size:
                self.flush()
                
        except Exception as e:
            self.logger.error(f"Error logging feature prediction: {e}")
    
    def flush(self) -> bool:
        """
        Flush Buffer zu Parquet-Datei
        
        Returns:
            bool: True wenn erfolgreich, False bei Fehler
        """
        if not self.buffer:
            self.logger.debug("Buffer ist leer, nichts zu flushen")
            return True
            
        try:
            # Konvertiere Buffer zu Polars DataFrame
            df = pl.DataFrame(self.buffer)
            
            # Bestimme Write-Mode (append oder overwrite)
            if self.output_path.exists():
                # Lade existierende Daten und füge neue hinzu
                existing_df = pl.read_parquet(self.output_path)
                combined_df = pl.concat([existing_df, df])
                
                # Schreibe kombinierte Daten
                combined_df.write_parquet(
                    self.output_path, 
                    compression=self.compression
                )
                
                self.logger.info(f"Appended {len(df)} entries to existing Parquet file")
            else:
                # Erstelle neue Datei
                df.write_parquet(
                    self.output_path, 
                    compression=self.compression
                )
                
                self.logger.info(f"Created new Parquet file with {len(df)} entries")
            
            # Buffer leeren
            entries_flushed = len(self.buffer)
            self.buffer.clear()
            
            self.logger.info(f"Successfully flushed {entries_flushed} entries to {self.output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error flushing buffer to Parquet: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Erhalte Logger-Statistiken
        
        Returns:
            Dict mit aktuellen Statistiken
        """
        file_size = 0
        file_entries = 0
        
        if self.output_path.exists():
            file_size = self.output_path.stat().st_size
            try:
                df = pl.read_parquet(self.output_path)
                file_entries = len(df)
            except Exception:
                file_entries = -1
        
        return {
            "total_entries_logged": self.total_entries,
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "buffer_usage_pct": (len(self.buffer) / self.buffer_size) * 100,
            "file_size_bytes": file_size,
            "file_entries": file_entries,
            "output_path": str(self.output_path),
            "compression": self.compression
        }
    
    def close(self) -> None:
        """
        Schließe Logger und flush alle verbleibenden Daten
        """
        if self.buffer:
            self.logger.info("Closing logger, flushing remaining buffer...")
            self.flush()
        
        self.logger.info(f"FeaturePredictionLogger closed. Total entries: {self.total_entries}")
    
    def __enter__(self):
        """Context Manager Support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support"""
        self.close()


class RotatingFeaturePredictionLogger(FeaturePredictionLogger):
    """
    ChatGPT Enhancement: Rotating Logger für tägliche Parquet-Dateien
    Erstellt separate Dateien pro Tag für bessere Organisation
    """
    
    def __init__(
        self, 
        base_path: str = "logs/ai_features", 
        buffer_size: int = 5000,
        compression: str = "zstd"
    ):
        """
        Initialize Rotating Feature Prediction Logger
        
        Args:
            base_path: Basis-Pfad ohne Dateiendung
            buffer_size: Buffer-Größe
            compression: Kompression
        """
        self.base_path = Path(base_path)
        self.current_date = None
        
        # Initialisiere mit aktuellem Datum
        super().__init__(
            output_path=self._get_daily_path(),
            buffer_size=buffer_size,
            compression=compression
        )
    
    def _get_daily_path(self) -> str:
        """Generiere täglichen Datei-Pfad"""
        today = datetime.now().strftime("%Y%m%d")
        return f"{self.base_path}_{today}.parquet"
    
    def log(self, *, ts_ns: int, **kwargs) -> None:
        """
        Log mit automatischer Datei-Rotation
        """
        # Prüfe ob neuer Tag begonnen hat
        log_date = datetime.utcfromtimestamp(ts_ns / 1e9).strftime("%Y%m%d")
        
        if self.current_date != log_date:
            # Flush aktuellen Buffer
            if self.buffer:
                self.flush()
            
            # Update auf neuen Tag
            self.current_date = log_date
            self.output_path = Path(self._get_daily_path())
            
            self.logger.info(f"Rotated to new daily file: {self.output_path}")
        
        # Standard Logging
        super().log(ts_ns=ts_ns, **kwargs)


# Convenience Functions für einfache Nutzung
def create_feature_logger(
    output_path: str = "logs/ai_features.parquet",
    buffer_size: int = 5000,
    rotating: bool = False
) -> FeaturePredictionLogger:
    """
    Factory Function für Feature Logger
    
    Args:
        output_path: Output-Pfad
        buffer_size: Buffer-Größe
        rotating: Ob tägliche Rotation verwendet werden soll
    
    Returns:
        FeaturePredictionLogger Instance
    """
    if rotating:
        return RotatingFeaturePredictionLogger(
            base_path=output_path.replace('.parquet', ''),
            buffer_size=buffer_size
        )
    else:
        return FeaturePredictionLogger(
            output_path=output_path,
            buffer_size=buffer_size
        )


if __name__ == "__main__":
    # Test des Feature Prediction Loggers
    import time
    
    print("🧪 Testing FeaturePredictionLogger...")
    
    with create_feature_logger("test_logs/test_features.parquet", buffer_size=3) as logger:
        # Test-Daten
        for i in range(5):
            logger.log(
                ts_ns=int(time.time() * 1e9),
                instrument="EUR/USD",
                features={
                    "open": 1.1000 + i * 0.0001,
                    "close": 1.1005 + i * 0.0001,
                    "rsi": 50 + i,
                    "macd": 0.001 * i
                },
                prediction={
                    "action": "BUY" if i % 2 == 0 else "SELL",
                    "confidence": 0.7 + i * 0.05,
                    "reasoning": f"test_prediction_{i}"
                },
                confidence_score=0.8 + i * 0.02,
                risk_score=0.1 + i * 0.01,
                market_regime="trending"
            )
            
            time.sleep(0.1)
    
    print("✅ FeaturePredictionLogger Test abgeschlossen!")
    
    # Zeige Statistiken
    stats = logger.get_stats()
    print(f"📊 Stats: {stats}")