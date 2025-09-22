#!/usr/bin/env python3
"""
Production-Ready Rotating Parquet Logger
ChatGPT-Enhanced Version mit PyArrow für echtes Append

Features:
- ECHTES Append via ParquetWriter (kein Read-Modify-Write)
- Eine Datei pro Zeitraum (day/hour), optional mit PID-Suffix für Multi-Process-Safety
- Schema-Drift: fail-fast
- Robust: Fehler im Flush/Close werden geloggt, brechen Hot-Path nicht
- Optionales Memory-Logging für Langläufer
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
import os
import sys
import traceback

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import psutil
except Exception:
    psutil = None


class RotatingParquetLogger:
    """
    Production-Ready Rotating Parquet Logger
    
    Features:
    - ECHTES Append via ParquetWriter (kein Read-Modify-Write)
    - Eine Datei pro Zeitraum (day/hour), optional mit PID-Suffix für Multi-Process-Safety
    - Schema-Drift: fail-fast mit check_metadata=False
    - Robust: Fehler im Flush/Close werden geloggt, brechen Hot-Path nicht
    - Optionales Memory-Logging für Langläufer
    - Row-Normalisierung für stabile Keys
    - Kompressions-Fallback (zstd → snappy)
    
    Thread-Safety:
    - NICHT thread-safe! Pro Thread/Prozess eigenen Logger nutzen
    - Multi-Process: PID-Suffix aktiviert lassen
    - In Kubernetes/PM2: zusätzlich pro Instanz base_path variieren
    
    ChatGPT-Enhanced Version mit Production-Ready Fixes
    """

    def __init__(
        self,
        base_path: str,
        rotation: str = "day",  # "day" | "hour"
        compression: str = "zstd",
        buffer_size: int = 2000,
        include_pid: bool = True,
        mem_log_every_flush: bool = False,
        logger: Optional[Callable[[str], None]] = None,  # z.B. self.log.warning
        fixed_fields: Optional[List[str]] = None,
        default_value: Optional[Dict[str, object]] = None,
    ):
        self.base = Path(base_path)
        self.rot = rotation
        self.comp = compression
        self.buf: List[Dict] = []
        self.buf_size = int(buffer_size)
        self.include_pid = bool(include_pid)
        self.mem_log = bool(mem_log_every_flush)
        self._pid = os.getpid()
        self._writer: Optional[pq.ParquetWriter] = None
        self._cur_path: Optional[Path] = None
        self._schema: Optional[pa.Schema] = None
        self._log = logger or (lambda msg: print(f"[RotatingParquetLogger] {msg}", file=sys.stderr))
        
        # ChatGPT Enhancement: Row-Normalisierung
        self._fixed_fields = fixed_fields  # feste Feldreihenfolge/-menge
        self._defaults = default_value or {}
        
        # Stats
        self.total_rows_logged = 0
        self.total_flushes = 0
        self.total_files_created = 0

    # ---------- helpers ----------
    def _period_suffix(self, ts_ns: int) -> str:
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y%m%d_%H" if self.rot == "hour" else "%Y%m%d")

    def _target_path(self, ts_ns: int) -> Path:
        name = f"{self.base.name}_{self._period_suffix(ts_ns)}"
        if self.include_pid:
            name += f"_pid{self._pid}"
        return self.base.with_name(name).with_suffix(".parquet")

    def _rotate_if_needed(self, ts_ns: int):
        path = self._target_path(ts_ns)
        if path != self._cur_path:
            self._close_writer()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._cur_path = path
            self._schema = None
            self._writer = None
            # ChatGPT Enhancement: Keine File-Zählung ohne Writer (verschoben zu _write_table)
            self._log(f"Rotated to {path.name}")

    def _write_table(self, table: pa.Table):
        if self._writer is None:
            # ChatGPT Enhancement: Kompressions-Fallback
            comp = self.comp
            try:
                # Test ob Kompression verfügbar ist durch Dummy-Schema-Test
                dummy_schema = pa.schema([("test", pa.int64())])
                test_path = self._cur_path.with_suffix(".test")
                test_writer = pq.ParquetWriter(test_path, dummy_schema, compression=comp)
                test_writer.close()
                test_path.unlink(missing_ok=True)  # Cleanup
            except Exception:
                comp = "snappy"
                self._log(f"Compression '{self.comp}' not available → falling back to 'snappy'")
            
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self._cur_path, self._schema, compression=comp)
            self.total_files_created += 1  # ChatGPT Enhancement: Zähle erst beim Writer-Create
            self._log(f"Created ParquetWriter for {self._cur_path.name} with {len(self._schema)} cols (comp={comp})")
        else:
            # ChatGPT Enhancement: Robust gegen Metadaten-Differenzen
            if not table.schema.equals(self._schema, check_metadata=False):
                raise ValueError(
                    f"Schema drift: expected={self._schema} got={table.schema} (metadata ignored)"
                )
        
        self._writer.write_table(table)

    # ---------- api ----------
    def log(self, *, ts_ns: int, row: Dict):
        """
        Log a single row with timestamp
        
        ChatGPT Enhancement: Row-Normalisierung für stabile Keys
        """
        if self._fixed_fields:
            # Sorge für vollständige & geordnete Rows
            norm = {k: row.get(k, self._defaults.get(k, "")) for k in self._fixed_fields}
            row = norm
        
        self.buf.append(row)
        self.total_rows_logged += 1
        
        if len(self.buf) >= self.buf_size:
            self.flush()

    def flush(self):
        """Flush buffer to Parquet file"""
        if not self.buf:
            return
        
        try:
            ts_ns = int(self.buf[-1].get("ts_ns", 0) or self.buf[0]["ts_ns"])
            self._rotate_if_needed(ts_ns)
            table = pa.Table.from_pylist(self.buf)
            self._write_table(table)
            
            rows_flushed = len(self.buf)
            self.buf.clear()
            self.total_flushes += 1
            
            if self.mem_log and psutil:
                mem = psutil.Process().memory_info().rss / (1024**3)
                self._log(f"Flush #{self.total_flushes} OK → {self._cur_path.name} | {rows_flushed} rows | RSS={mem:.2f} GB")
            else:
                self._log(f"Flush #{self.total_flushes} OK → {self._cur_path.name} | {rows_flushed} rows")
                
        except Exception:
            self._log(f"Flush error:\n{traceback.format_exc()}")
            # Buffer bei Fehler NICHT verwerfen – für Recovery

    def close(self):
        """Close logger and flush remaining data"""
        try:
            self.flush()
        except Exception:
            self._log(f"Close.flush error:\n{traceback.format_exc()}")
        finally:
            try:
                self._close_writer()
            except Exception:
                self._log(f"Close.writer error:\n{traceback.format_exc()}")
        
        self._log(f"Logger closed. Stats: {self.get_stats()}")

    def get_stats(self) -> Dict:
        """Get logger statistics"""
        return {
            "total_rows_logged": self.total_rows_logged,
            "total_flushes": self.total_flushes,
            "total_files_created": self.total_files_created,
            "buffer_size": len(self.buf),
            "buffer_capacity": self.buf_size,
            "current_file": str(self._cur_path) if self._cur_path else None,
            "rotation": self.rot,
            "compression": self.comp,
            "include_pid": self.include_pid,
            "pid": self._pid
        }

    # ---------- intern ----------
    def _close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.close()


# Factory function for easy creation
def create_rotating_logger(
    base_path: str,
    rotation: str = "day",
    buffer_size: int = 2000,
    include_pid: bool = True,
    compression: str = "zstd",
    mem_log_every_flush: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    fixed_fields: Optional[List[str]] = None,
    default_value: Optional[Dict[str, object]] = None,
) -> RotatingParquetLogger:
    """
    Factory function for creating RotatingParquetLogger
    
    Args:
        base_path: Base path for log files
        rotation: "day" or "hour"
        buffer_size: Number of rows to buffer before flush
        include_pid: Include process ID in filename for multi-process safety
        logger: Optional logger function
    
    Returns:
        RotatingParquetLogger instance
    """
    return RotatingParquetLogger(
        base_path=base_path,
        rotation=rotation,
        buffer_size=buffer_size,
        include_pid=include_pid,
        compression=compression,
        mem_log_every_flush=mem_log_every_flush,
        logger=logger,
        fixed_fields=fixed_fields,
        default_value=default_value,
    )


if __name__ == "__main__":
    # Test the logger
    import time
    
    print("🧪 Testing RotatingParquetLogger...")
    
    with create_rotating_logger("test_logs/test_parquet", rotation="day", buffer_size=5) as logger:
        # Test data
        for i in range(12):
            row = {
                "ts_ns": int(time.time() * 1e9) + i * 1000000000,
                "instrument": "EUR/USD",
                "f_open": 1.1000 + i * 0.0001,
                "f_close": 1.1005 + i * 0.0001,
                "f_volume": 1000.0 + i * 100,
                "pred_action": ["BUY", "SELL", "HOLD"][i % 3],
                "pred_confidence": 0.7 + i * 0.02,
                "pred_reason": f"test_prediction_{i}",
                "pred_risk": 0.1 + i * 0.01
            }
            
            logger.log(ts_ns=row["ts_ns"], row=row)
            
            if i == 4:  # Test auto-flush
                print(f"   Auto-flush triggered at row {i+1}")
            
            time.sleep(0.1)
    
    # Show stats
    stats = logger.get_stats()
    print(f"📊 Final Stats: {stats}")
    
    print("✅ RotatingParquetLogger Test completed!")




    """Chatgpt: hier ist mein kurzer, konkreter Review mit Fix-Empfehlungen (safe-by-default), plus Mini-Patches:

🔎 Befunde

Schema-Drift: guter Fail-Fast, aber nutze check_metadata=False (sonst drift durch unwichtige Metadaten).

Row-Normalisierung: from_pylist bricht bei fehlenden/zusätzlichen Keys (oder ändert Schema). Einheitliche Keys + Default-Werte absichern.

total_files_created: wird beim bloßen Rotieren gezählt, auch wenn noch keine Daten ins neue File geschrieben wurden → zähle erst beim Writer-Create.

Kompression: zstd evtl. nicht verfügbar → eleganter Fallback auf snappy.

Logger-Typ: Optional[callable] → präziser Optional[Callable[[str], None]].

Thread-Safety: Klasse ist nicht threadsafe → klar dokumentieren; pro Thread/Prozess eigenen Logger nutzen (PID-Suffix hilft).

Factory: keine Parameter für compression / mem_log_every_flush → ergänzen.

Minor: get_stats["buffer_size"] ist die aktuelle Füllung, Name ist ok, nur zur Klarheit erwähnen.

✅ Minimal-Patches
1) Schema-Vergleich & Writer-Zählung & Kompressions-Fallback
# oben
from typing import Callable

# __init__ (...)
logger: Optional[Callable[[str], None]] = None,

# _write_table
def _write_table(self, table: pa.Table):
    if self._writer is None:
        # Kompressions-Fallback
        comp = self.comp
        try:
            pq.ParquetWriter._ensure_compression(comp)  # intern – wenn nicht vorhanden, Exception
        except Exception:
            comp = "snappy"
            self._log(f"Compression '{self.comp}' not available → falling back to 'snappy'")
        self._schema = table.schema
        self._writer = pq.ParquetWriter(self._cur_path, self._schema, compression=comp)
        self.total_files_created += 1
        self._log(f"Created ParquetWriter for {self._cur_path.name} with {len(self._schema)} cols (comp={comp})")
    else:
        # robust gegen Metadaten-Differenzen
        if not table.schema.equals(self._schema, check_metadata=False):
            raise ValueError(
                f"Schema drift: expected={self._schema} got={table.schema} (metadata ignored)"
            )
    self._writer.write_table(table)

2) Row-Normalisierung (Key-Set stabil halten)

Füge eine optionale Key-Liste hinzu, um fehlende Keys aufzufüllen (z. B. leere Strings/0.0):

# __init__
def __init__(..., fixed_fields: Optional[List[str]] = None, default_value: Optional[Dict[str, object]] = None, ...):
    ...
    self._fixed_fields = fixed_fields  # feste Feldreihenfolge/ -menge
    self._defaults = default_value or {}

# log(...)
def log(self, *, ts_ns: int, row: Dict):
    if self._fixed_fields:
        # sorge für vollständige & geordnete Rows
        norm = {k: row.get(k, self._defaults.get(k, "")) for k in self._fixed_fields}
        row = norm
    self.buf.append(row)
    self.total_rows_logged += 1
    if len(self.buf) >= self.buf_size:
        self.flush()


Tipp: setze fixed_fields beim Erzeugen (gleiche Keys wie in deinem Beispiel-Row), z. B.:

fixed = ["ts_ns","instrument","f_open","f_high","f_low","f_close","f_volume","pred_action","pred_confidence","pred_reason","pred_risk"]
logger = create_rotating_logger(..., fixed_fields=fixed, default_value={"pred_reason": ""})

3) Rotations-Logging (keine File-Zählung ohne Writer)
def _rotate_if_needed(self, ts_ns: int):
    path = self._target_path(ts_ns)
    if path != self._cur_path:
        self._close_writer()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._cur_path = path
        self._schema = None
        self._writer = None
        self._log(f"Rotated to {path.name}")


(Zähler wurde in _write_table verschoben.)

4) Factory mit mehr Parametern
def create_rotating_logger(
    base_path: str,
    rotation: str = "day",
    buffer_size: int = 2000,
    include_pid: bool = True,
    compression: str = "zstd",
    mem_log_every_flush: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    fixed_fields: Optional[List[str]] = None,
    default_value: Optional[Dict[str, object]] = None,
) -> RotatingParquetLogger:
    return RotatingParquetLogger(
        base_path=base_path,
        rotation=rotation,
        buffer_size=buffer_size,
        include_pid=include_pid,
        compression=compression,
        mem_log_every_flush=mem_log_every_flush,
        logger=logger,
        fixed_fields=fixed_fields,
        default_value=default_value,
    )

📌 Hinweise (Prod)

Multi-Process: PID-Suffix aktiv lassen; in Kubernetes/PM2 zusätzlich pro Instanz base_path variieren.

Multi-Thread: pro Thread eigener Logger oder Aufrufe serialisieren.

Backpressure: bei Flush-Fehlern werden Rows behalten; optional: Notfall-Dump als CSV in .../failed/.

Monitoring: get_stats() regelmäßig loggen; Memory-Log (mem_log_every_flush=True) nur bei Bedarf."""