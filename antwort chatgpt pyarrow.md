top — hier ist die **finale, production-ready PyArrow-Variante** mit den gewünschten Verbesserungen:

- flexiblere Pins

- robustes Error-Handling (Flush/Close mit Log)

- Multi-Process-Safety (PID im Dateinamen, optional)

- optionales Memory-Monitoring (leichter Health-Log)

## Installation (empfohlen)

`pip install "pyarrow>=14.0.0" "polars>=0.20" # optional für Memory-Log: pip install psutil`

## `rotating_parquet_logger.py`

`from __future__ import annotations from pathlib import Path from datetime import datetime, timezone from typing import Dict, List, Optional import os, sys, traceback  import pyarrow as pa import pyarrow.parquet as pq  try:    import psutil except Exception:     psutil = None  class RotatingParquetLogger:     """     - ECHTES Append via ParquetWriter (kein Read-Modify-Write)     - Eine Datei pro Zeitraum (day/hour), optional mit PID-Suffix für Multi-Process-Safety     - Schema-Drift: fail-fast     - Robust: Fehler im Flush/Close werden geloggt, brechen Hot-Path nicht     - Optionales Memory-Logging für Langläufer     """      def __init__(         self,         base_path: str,         rotation: str = "day",         # "day" | "hour"         compression: str = "zstd",         buffer_size: int = 2000,         include_pid: bool = True,         mem_log_every_flush: bool = False,         logger: Optional[callable] = None,  # z.B. self.log.warning     ):         self.base = Path(base_path)         self.rot = rotation         self.comp = compression         self.buf: List[Dict] = []         self.buf_size = int(buffer_size)         self.include_pid = bool(include_pid)         self.mem_log = bool(mem_log_every_flush)         self._pid = os.getpid()         self._writer: Optional[pq.ParquetWriter] = None         self._cur_path: Optional[Path] = None         self._schema: Optional[pa.Schema] = None         self._log = logger or (lambda msg: print(msg, file=sys.stderr))    # ---------- helpers ----------     def _period_suffix(self, ts_ns: int) -> str:         dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)        return dt.strftime("%Y%m%d_%H" if self.rot == "hour" else "%Y%m%d")    def _target_path(self, ts_ns: int) -> Path:         name = f"{self.base.name}_{self._period_suffix(ts_ns)}"        if self.include_pid:             name += f"_pid{self._pid}"        return self.base.with_name(name).with_suffix(".parquet")    def _rotate_if_needed(self, ts_ns: int):         path = self._target_path(ts_ns)        if path != self._cur_path:             self._close_writer()             path.parent.mkdir(parents=True, exist_ok=True)             self._cur_path = path             self._schema = None             self._writer = None      def _write_table(self, table: pa.Table):        if self._writer is None:             self._schema = table.schema             self._writer = pq.ParquetWriter(self._cur_path, self._schema, compression=self.comp)        else:            if table.schema != self._schema:                raise ValueError("Schema drift detected — ensure consistent keys & dtypes.")         self._writer.write_table(table)    # ---------- api ----------     def log(self, *, ts_ns: int, row: Dict):         self.buf.append(row)        if len(self.buf) >= self.buf_size:             self.flush()    def flush(self):        if not self.buf:            return         try:             ts_ns = int(self.buf[-1].get("ts_ns", 0) or self.buf[0]["ts_ns"])             self._rotate_if_needed(ts_ns)             table = pa.Table.from_pylist(self.buf)             self._write_table(table)             self.buf.clear()            if self.mem_log and psutil:                 mem = psutil.Process().memory_info().rss / (1024**3)                 self._log(f"[RotatingParquetLogger] flush ok → {self._cur_path.name} | RSS={mem:.2f} GB")        except Exception:             self._log("[RotatingParquetLogger] flush error:\n" + traceback.format_exc())            # Buffer bei Fehler NICHT verwerfen – du kannst alternativ hier auf Disk-Dump wechseln.      def close(self):        try:             self.flush()        except Exception:             self._log("[RotatingParquetLogger] close.flush error:\n" + traceback.format_exc())        finally:            try:                 self._close_writer()            except Exception:                 self._log("[RotatingParquetLogger] close.writer error:\n" + traceback.format_exc())    # ---------- intern ----------     def _close_writer(self):        if self._writer is not None:             self._writer.close()             self._writer = None`

## Verwendung in deiner Strategy (kurz)

`# on_start from rotating_parquet_logger import RotatingParquetLogger self.flog = RotatingParquetLogger(     base_path="logs/ai/eurusd",     rotation="day",          # oder "hour"     buffer_size=2000,     include_pid=True,        # Multi-Process sicher     mem_log_every_flush=True,     logger=self.log.warning, # Nautilus-Logger verwenden )  # on_bar row = {  "ts_ns": int(bar.ts_init),  "instrument": str(bar.bar_type.instrument_id),  "f_open": float(bar.open), "f_high": float(bar.high),  "f_low": float(bar.low), "f_close": float(bar.close), "f_volume": float(bar.volume),  "pred_action": pred.get("action",""), "pred_confidence": float(pred.get("confidence",0.0)),  "pred_reason": pred.get("reasoning",""), "pred_risk": float(pred.get("risk_score",0.0)), } self.flog.log(ts_ns=row["ts_ns"], row=row)  # on_stop self.flog.close()`

Damit hast du:

- elegantes, schnelles **Append**

- **Schema-Sicherheit** und Rotation

- **Multi-Process-Safe** (PID-Suffix)

- produktionsreifes **Fehler-Logging** & optionales Memory-Monitoring.
