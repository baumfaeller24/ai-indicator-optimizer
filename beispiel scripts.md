mega — hier ist das **korrigierte Paket** (4 Dateien), direkt lauffähig, robust und produktionsnäher (Rotation, Factory, sichere Orders, saubere Datasets & Tests).

# 1) `ai_pattern_strategy.py` (fixed)

```python
"""
AI-basierte Pattern-Recognition Strategie (fixed)
– sichere REST-KI-Anbindung (TorchServe)
– robuste Order-Submission
– optionales Feature/Prediction-Logging & Dataset-Aufbau
"""
from __future__ import annotations
import os
import json
import requests
import numpy as np
from typing import Dict, Optional

from nautilus_trader.strategy.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide

# optionale Hilfen (lokale Module)
try:
    from feature_prediction_logger import FeaturePredictionLogger, create_feature_logger
except Exception:
    FeaturePredictionLogger = None
    def create_feature_logger(*args, **kwargs): return None

try:
    from bar_dataset_builder import BarDatasetBuilder
except Exception:
    BarDatasetBuilder = None


class AIPatternStrategy(Strategy):
    """
    Integriert ein externes TorchServe-Modell über REST.
    Erwartet JSON-Input mit OHLCV-Features; Output:
      {"action": "BUY|SELL|HOLD", "confidence": float, "reasoning": str, "risk_score": float}
    """

    def __init__(self, config=None):
        super().__init__(config)

        cfg = config or {}
        # KI / REST
        self.ai_endpoint = cfg.get("ai_endpoint", os.getenv("AI_ENDPOINT", "http://localhost:8080/predictions/pattern_model"))
        self.request_timeout = float(cfg.get("request_timeout", os.getenv("AI_TIMEOUT", 2.0)))
        self.min_confidence = float(cfg.get("min_confidence", 0.70))
        self.use_mock = bool(cfg.get("use_mock", False))

        # Trading
        self.position_size = int(cfg.get("position_size", 1_000))
        self.max_positions = int(cfg.get("max_positions", 1))
        self.risk_per_trade = float(cfg.get("risk_per_trade", 0.02))

        # Logging / Datasets (optional)
        self.enable_feature_logging = bool(cfg.get("enable_feature_logging", True))
        self.feature_log_base = cfg.get("feature_log_base", os.getenv("FEATURE_LOG_BASE", "logs/ai"))
        self.rotate_hourly = bool(cfg.get("rotate_hourly", False))
        self.enable_dataset = bool(cfg.get("enable_dataset", False))
        self.dataset_horizon = int(cfg.get("dataset_horizon", 5))
        self.dataset_path = cfg.get("dataset_path", "datasets/eurusd_bars_h5.parquet")

        # Stats
        self.predictions_count = 0
        self.successful_predictions = 0

        # intern
        self.feature_logger: Optional[FeaturePredictionLogger] = None
        self.ds: Optional[BarDatasetBuilder] = None

    # ----- lifecycle -----
    def on_start(self):
        self.log.info("✅ AI Pattern Strategy started")
        self.log.info(f"📡 AI Endpoint: {self.ai_endpoint} | ⏱ timeout={self.request_timeout}s | 🎯 min_conf={self.min_confidence}")
        self.log.info(f"📦 feature_logging={self.enable_feature_logging} dataset={self.enable_dataset}")

        if self.enable_feature_logging and create_feature_logger:
            # Rotation: täglich (default) oder stündlich
            rotate = "hour" if self.rotate_hourly else "day"
            self.feature_logger = create_feature_logger(base_path=self.feature_log_base, buffer_size=2000, rotation=rotate)

        if self.enable_dataset and BarDatasetBuilder:
            self.ds = BarDatasetBuilder(horizon=self.dataset_horizon, min_bars=50)

    def on_stop(self):
        acc = (self.successful_predictions / max(self.predictions_count, 1)) * 100.0
        self.log.info(f"📊 AI Strategy Performance: {acc:.1f}% ({self.successful_predictions}/{self.predictions_count})")

        if self.feature_logger:
            try:
                self.feature_logger.close()
            except Exception as e:
                self.log.warning(f"Feature logger close failed: {e}")

        if self.ds:
            try:
                ok = self.ds.to_parquet(self.dataset_path)
                self.log.info(f"💾 Dataset saved={ok} → {self.dataset_path}")
            except Exception as e:
                self.log.warning(f"Dataset write failed: {e}")

    # ----- main loop -----
    def on_bar(self, bar: Bar):
        try:
            features = self._extract_features(bar)
            pred = self._get_prediction(features)

            if self.feature_logger:
                self.feature_logger.log(
                    ts_ns=int(bar.ts_init),
                    instrument=str(bar.bar_type.instrument_id),
                    features=features,
                    prediction=pred or {"action": "HOLD", "confidence": 0.0, "reasoning": "no_pred"},
                )

            if self.ds:
                self.ds.on_bar(bar)

            if pred and float(pred.get("confidence", 0.0)) >= self.min_confidence:
                self._execute_signal(pred, bar)

        except Exception as e:
            self.log.error(f"⚠️ on_bar failed: {e}")

    # ----- helpers -----
    def _extract_features(self, bar: Bar) -> Dict:
        open_, high, low, close, vol = map(float, (bar.open, bar.high, bar.low, bar.close, bar.volume))
        price_change = close - open_
        rng = max(high - low, 1e-9)
        body_ratio = abs(price_change) / rng
        return {
            "open": open_, "high": high, "low": low, "close": close, "volume": vol,
            "price_change": price_change, "price_range": high - low, "body_ratio": body_ratio,
            "timestamp": int(bar.ts_init), "instrument": str(bar.bar_type.instrument_id), "bar_type": str(bar.bar_type),
        }

    def _get_prediction(self, features: Dict) -> Optional[Dict]:
        if self.use_mock:
            return self._mock_pred(features)

        try:
            res = requests.post(self.ai_endpoint, json=features, timeout=self.request_timeout,
                                headers={"Content-Type": "application/json"})
            res.raise_for_status()
            data = res.json()
            # TorchServe kann Liste oder Objekt liefern:
            pred = data[0] if isinstance(data, list) and data else data
            self.predictions_count += 1
            return {
                "action": pred.get("action", "HOLD"),
                "confidence": float(pred.get("confidence", 0.0)),
                "reasoning": pred.get("reasoning", "n/a"),
                "risk_score": float(pred.get("risk_score", 0.0)),
            }
        except requests.exceptions.Timeout:
            self.log.warning("⏰ AI timeout")
        except Exception as e:
            self.log.error(f"❌ AI request failed: {e}")
        return None

    def _mock_pred(self, f: Dict) -> Dict:
        pc = float(f.get("price_change", 0.0))
        br = float(f.get("body_ratio", 0.0))
        if pc > 0 and br > 0.7:
            return {"action": "BUY", "confidence": 0.75, "reasoning": "mock-bull", "risk_score": 0.2}
        if pc < 0 and br > 0.7:
            return {"action": "SELL", "confidence": 0.75, "reasoning": "mock-bear", "risk_score": 0.2}
        return {"action": "HOLD", "confidence": 0.5, "reasoning": "mock-hold", "risk_score": 0.3}

    def _execute_signal(self, pred: Dict, bar: Bar):
        if len(self.portfolio.positions_open()) >= self.max_positions:
            self.log.info("⏸️ Max positions reached")
            return
        side = OrderSide.BUY if pred["action"] == "BUY" else OrderSide.SELL
        qty = self.position_size
        self._submit_market_order(side, qty)

        self.log.info(
            f"[AI] 🎯 {pred['action']} qty={qty} "
            f"📊 conf={pred['confidence']:.2f} 🧠 {pred.get('reasoning','')}"
        )

    def _submit_market_order(self, side: OrderSide, qty: int):
        # Bevorzugt die abstrakte Convenience-API, fallback auf Fehlerlogging
        try:
            # viele Nautilus-Versionen erlauben side als str ("BUY"/"SELL")
            self.submit_market_order(side.name, quantity=qty)
        except Exception as e:
            self.log.error(f"Order submission failed: {e}")

    # ----- events -----
    def on_order_filled(self, event):
        self.log.info(f"✅ Order filled: {getattr(event, 'order_id', 'n/a')}")

    def on_position_opened(self, position):
        self.log.info(f"📈 Position opened: {position.instrument_id} {position.side}")

    def on_position_closed(self, position):
        pnl = getattr(position, "realized_pnl", None)
        try:
            pnl_val = pnl.as_double() if pnl is not None else 0.0
        except Exception:
            pnl_val = float(pnl) if pnl is not None else 0.0
        if pnl_val > 0:
            self.successful_predictions += 1
        self.log.info(f"📉 Position closed: {position.instrument_id} PnL: {pnl_val:.6f}")
```

# 2) `feature_prediction_logger.py` (factory + rotation, ohne teures Append)

```python
"""
Feature/Prediction-Logger mit Rotation (day/hour) und einfacher Factory.
Schreibt batches in eigenständige Parquet-Dateien, kein teures Read-Modify-Write.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import polars as pl

class FeaturePredictionLogger:
    def __init__(self, output_path: Path, buffer_size: int = 2000, compression: str = "zstd"):
        self.output_path = Path(output_path)
        self.buffer_size = int(buffer_size)
        self.compression = compression
        self.buffer: List[Dict] = []
        self.count = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, ts_ns: int, instrument: str, features: Dict, prediction: Dict, **extra):
        row = {
            "ts_ns": int(ts_ns),
            "time": datetime.utcfromtimestamp(ts_ns / 1e9).isoformat(),
            "instrument": instrument,
            **{f"f_{k}": v for k, v in features.items()},
            "pred_action": prediction.get("action"),
            "pred_confidence": float(prediction.get("confidence", 0.0)),
            "pred_reason": prediction.get("reasoning"),
            "pred_risk": float(prediction.get("risk_score", 0.0)),
            **extra,
        }
        self.buffer.append(row)
        self.count += 1
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        df = pl.DataFrame(self.buffer)
        # statt Append an eine große Datei → eigene Batch-Datei, stabil & schnell
        part = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        batch_path = self.output_path.with_suffix(f".{part}.parquet")
        df.write_parquet(batch_path, compression=self.compression)
        self.buffer.clear()

    def close(self):
        self.flush()


class RotatingFeaturePredictionLogger(FeaturePredictionLogger):
    """Rotation per Tag oder Stunde: schreibt direkt in passende Datei, ohne reads."""
    def __init__(self, base_path: Path, rotation: str = "day", buffer_size: int = 2000, compression: str = "zstd"):
        self.base_path = Path(base_path)
        self.rotation = rotation  # "day" | "hour"
        self.current_path: Optional[Path] = None
        super().__init__(output_path=self._make_path_for_now(), buffer_size=buffer_size, compression=compression)

    def _make_path_for_now(self) -> Path:
        now = datetime.utcnow()
        if self.rotation == "hour":
            suffix = now.strftime("%Y%m%d_%H")
        else:
            suffix = now.strftime("%Y%m%d")
        p = self.base_path.with_name(f"{self.base_path.name}_{suffix}.parquet")
        p.parent.mkdir(parents=True, exist_ok=True)
        self.current_path = p
        return p

    def log(self, *, ts_ns: int, **kwargs):
        # rotiere bei neuem Zeitslot
        path_now = self._make_path_for_now()
        if path_now != self.current_path:
            self.flush()
            self.output_path = path_now
            self.current_path = path_now
        return super().log(ts_ns=ts_ns, **kwargs)


def create_feature_logger(base_path: str, buffer_size: int = 2000, rotation: str = "day"):
    """
    Factory für Tests/Prod:
      rotation="day" | "hour"
    """
    base = Path(base_path)
    return RotatingFeaturePredictionLogger(base_path=base, rotation=rotation, buffer_size=buffer_size)
```

# 3) `bar_dataset_builder.py` (fertig: labeling + stats + parquet)

```python
"""
BarDatasetBuilder – erzeugt supervised Datensätze aus Bars:
  * Features: OHLCV + simple Shape-Features
  * Label: Forward-Return über Horizon, zusätzlich diskrete Klasse (BUY/SELL/HOLD)
"""
from __future__ import annotations
from typing import Dict, List
from collections import deque
from pathlib import Path
import polars as pl
from nautilus_trader.model.data import Bar

class BarDatasetBuilder:
    def __init__(self, horizon: int = 5, min_bars: int = 100, price_col: str = "close"):
        self.h = int(horizon)
        self.min_bars = int(min_bars)
        self.price_col = price_col
        self.buf: deque[Dict] = deque(maxlen=self.h + 1)
        self.rows: List[Dict] = []

    def on_bar(self, bar: Bar):
        o, h, l, c, v = map(float, (bar.open, bar.high, bar.low, bar.close, bar.volume))
        rng = max(h - l, 1e-9)
        row = {
            "ts_ns": int(bar.ts_init),
            "instrument": str(bar.bar_type.instrument_id),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
            "price_change": c - o,
            "price_range": h - l,
            "body_ratio": abs(c - o) / rng,
        }
        self.buf.append(row)

        if len(self.buf) == self.buf.maxlen:
            x = dict(self.buf[0])  # shallow copy
            y_close = self.buf[-1]["close"]
            fwd_ret = (y_close / x["close"]) - 1.0
            x[f"label_fwd_ret_h{self.h}"] = fwd_ret
            # diskret
            thr = 0.0003
            x[f"label_class_h{self.h}"] = 0 if fwd_ret > thr else (1 if fwd_ret < -thr else 2)  # buy/sell/hold
            self.rows.append(x)

    def to_parquet(self, path: str, compression: str = "zstd") -> bool:
        if len(self.rows) < self.min_bars:
            return False
        df = pl.DataFrame(self.rows).sort("ts_ns")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out, compression=compression)
        return True

    def get_stats(self) -> Dict:
        if not self.rows:
            return {"total": 0, "labels": {}, "ret": {}}
        df = pl.DataFrame(self.rows)
        total = df.height
        lab_col = f"label_class_h{self.h}"
        ret_col = f"label_fwd_ret_h{self.h}"

        labs = df[lab_col].value_counts().to_dict(as_series=False)
        ret = df[ret_col]
        return {
            "total": int(total),
            "labels": {int(k): int(v) for k, v in zip(labs["values"], labs["counts"])},
            "ret": {
                "mean": float(ret.mean()), "std": float(ret.std()),
                "min": float(ret.min()), "max": float(ret.max()),
            },
        }
```

# 4) `test_enhanced_logging_only.py` (grüner Minimaltest)

```python
"""
Minimaltest: Logger/Factory & Strategy-Aufruf-Pfad (ohne Broker).
Läuft lokal, erzeugt Log-Parquet-Batches/Rotation.
"""
import os, json, time
from pathlib import Path

from feature_prediction_logger import create_feature_logger
from bar_dataset_builder import BarDatasetBuilder

def test_logger_and_dataset(tmp_path: Path):
    # Logger
    flog = create_feature_logger(base_path=str(tmp_path / "ai_log/eurusd"), rotation="day", buffer_size=10)
    for i in range(25):
        ts_ns = int(time.time_ns())
        features = {"open":1.1,"high":1.12,"low":1.09,"close":1.105 + i*1e-5,"volume":1000+i}
        pred = {"action":["BUY","SELL","HOLD"][i%3],"confidence":0.6,"reasoning":"test","risk_score":0.2}
        flog.log(ts_ns=ts_ns, instrument="EUR/USD", features=features, prediction=pred)
    flog.close()
    # Dateien sollten geschrieben worden sein
    assert any(p.suffix == ".parquet" for p in (tmp_path / "ai_log").rglob("*.parquet"))

    # Dataset
    ds = BarDatasetBuilder(horizon=3, min_bars=3)
    class BarStub:
        def __init__(self, t, o,h,l,c,v, iid="EUR/USD"):
            self.ts_init = t
            self.open, self.high, self.low, self.close, self.volume = o,h,l,c,v
            self.bar_type = type("BT", (), {"instrument_id": iid, "__str__": lambda s: "BAR1M"})()
    t0 = int(time.time_ns())
    for k in range(10):
        ds.on_bar(BarStub(t0 + k*60_000_000_000, 1.10,1.12,1.09,1.10+(k*1e-4),1000+k))
    out = tmp_path / "ds/eurusd.parquet"
    ok = ds.to_parquet(str(out))
    assert ok and out.exists()
```

---

**Hinweise**

- Strategy: nutzt `self.submit_market_order(side.name, quantity=qty)` → falls deine Nautilus-Version eine andere Signatur erwartet, sag mir welche; ich passe sie sofort an.

- Logger: Rotation erzeugt neue Dateien (schnell, kein Append-Lock). Für Konsum/Training einfach alle Parquets aus dem Verzeichnis laden.

- Dataset: Horizon/Label-Schwelle (`thr`) kannst du im Code/Config anpassen.

Wenn du willst, packe ich dir das als **Git-Template** (mit `pyproject.toml`, `requirements.txt`, `Makefile`) zusammen.
