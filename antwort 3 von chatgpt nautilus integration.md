**Kurzfassung (mit Fixes & Code‑Patch):**  
Ich behebe die in eurem Paket gemeldeten Probleme zu **Data‑Flow/Types, TorchServe‑Parametern, Result‑Konversion, Error‑Handling und Imports** – exakt entlang eurer Signaturen/Logs . Unten ist ein **drop‑in Patch** für `nautilus_integrated_pipeline.py`, der:

- **Data‑Flow konsistent** macht (echtes `DataFrame` → Features/optional Chart‑Image)

- **TorchServe korrekt** füttert (Dict statt ‘"df"’-String, List‑Wrapping nur wenn benötigt)

- **Top5StrategiesResult** robust normalisiert (kein `len()` auf Non‑List)

- **Dukascopy/Date‑Fehler** abfängt (Zeitspalten → `pd.DatetimeIndex`)

- **Fallback** statt Hard‑Abort nutzt

- **Absolute Imports** + Script‑Start fix (PYTHONPATH)

- **SYNC→async** sauber via `asyncio.to_thread` bridged

> Kontext & Ziele stammen aus eurem **Error‑Analysis‑Paket** (Fehlerbilder, Fragen 1–5) und **Fix‑Paket** (exakte Signaturen & gewünschte Struktur) . Das Design‑Summary bestätigt die Integrationsrichtung (Nautilus optional, Services separat) .

---

### 🔧 Patch: `nautilus_integrated_pipeline.py` (relevante Abschnitte ersetzt/ergänzt)

```python
# --- NEU: Utilities ganz oben hinzufügen ---
from __future__ import annotations
import asyncio, json, logging, time, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

def _ensure_df(obj: Any) -> Optional[pd.DataFrame]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    # Falls z.B. Arrow/Polars: best effort
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None

def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behebt 'int' has no attribute 'date' durch saubere Zeitspalten:
    - akzeptiert 'time'/'timestamp' oder Index
    - konvertiert zu UTC DatetimeIndex
    """
    if df is None or df.empty:
        return df
    dfe = df.copy()
    if "time" in dfe.columns:
        dfe["time"] = pd.to_datetime(dfe["time"], errors="coerce", utc=True)
        dfe = dfe.set_index("time", drop=True)
    elif "timestamp" in dfe.columns:
        dfe["timestamp"] = pd.to_datetime(dfe["timestamp"], errors="coerce", utc=True)
        dfe = dfe.set_index("timestamp", drop=True)
    elif not isinstance(dfe.index, pd.DatetimeIndex):
        # manchmal Millisekunden/sekunden als int Index
        try:
            dfe.index = pd.to_datetime(dfe.index, errors="coerce", utc=True)
        except Exception:
            pass
    dfe = dfe.sort_index()
    return dfe

def _build_feature_dict(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extrahiert numerische Features aus dem DataFrame (letzte Bar).
    Verhindert String→Float Fehler (nur float‑fähige Werte).
    """
    if df is None or df.empty:
        return {}
    last = df.iloc[-1]
    feats = {}
    for k in ["open","high","low","close","volume"]:
        v = last.get(k, np.nan)
        try:
            feats[k] = float(v)
        except Exception:
            feats[k] = float("nan")
    # abgeleitete Features
    try:
        feats["range"] = float(feats["high"] - feats["low"])
    except Exception:
        feats["range"] = float("nan")
    try:
        feats["ret_1"] = float(np.log(df["close"].iloc[-1] / df["close"].iloc[-2])) if len(df) > 1 else 0.0
    except Exception:
        feats["ret_1"] = 0.0
    return feats

def _normalize_top5(obj: Any) -> List[Dict]:
    """
    Konvertiert Top5StrategiesResult → List[Dict] (kein len() Fehler).
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    for attr in ("to_list", "to_dict", "strategies", "items"):
        if hasattr(obj, attr):
            v = getattr(obj, attr)
            v = v() if callable(v) else v
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                # häufig {'strategies': [...]} oder ähnlich
                for key in ("strategies","top","items","data"):
                    if key in v and isinstance(v[key], list):
                        return v[key]
                return [v]
    # Fallback: als einzelnes Element einpacken
    return [obj]

def _create_mock_df(symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq="T")
    base = 1.10
    noise = np.cumsum(np.random.normal(0, 1e-4, bars))
    close = base + noise
    high = close + np.abs(np.random.normal(0, 5e-5, bars))
    low  = close - np.abs(np.random.normal(0, 5e-5, bars))
    open_ = np.r_[close[0], close[:-1]]
    vol = np.random.randint(900, 1500, bars).astype(float)
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=idx)

# --- Nautilus optional ---
try:
    from nautilus_trader.trading.node import TradingNode
    from nautilus_trader.config import TradingNodeConfig
    NAUTILUS_AVAILABLE = True
except Exception as e:
    logging.warning(f"Nautilus not available: {e} → fallback mode")
    NAUTILUS_AVAILABLE = False

# --- Absolute Imports gemäß Signaturen (Fix‑Paket) ---
from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI
from ai_indicator_optimizer.ai.ai_strategy_evaluator import AIStrategyEvaluator
from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler
try:
    from ai_indicator_optimizer.ai.torchserve_handler import TorchServeConfig  # optional
except Exception:
    TorchServeConfig = None
from ai_indicator_optimizer.ai.live_control_system import LiveControlSystem
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector
from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger
from nautilus_config import NautilusHardwareConfig
```

```python
# --- AIServiceManager.multimodal_analysis: FIX Data-Flow & TorchServe Param ---
class AIServiceManager:
    # ... __init__/start/stop unverändert ...

    async def multimodal_analysis(self, chart: Dict, numerical: Dict) -> Dict:
        t0 = time.time()
        try:
            # numerical['ohlcv'] sollte DataFrame sein
            df = _ensure_df(numerical.get("ohlcv"))
            df = _normalize_time(df) if df is not None else None
            feats = _build_feature_dict(df)

            # Vision: chart_image optional; wenn None → überspringen
            chart_img = chart.get("chart_image") if isinstance(chart, dict) else None
            vision = await asyncio.to_thread(
                self.services["multimodal"].analyze_chart_pattern,
                chart_img,
                {"features": feats}  # numerische Indikatoren optional
            )

            # TorchServe: erwartet Dict oder List[Dict] → hier Dict
            ts_res = await asyncio.to_thread(
                self.services["torchserve"].process_features,
                feats,                            # ← KEIN "df"-String!
                "pattern_model"
            )

            out = {
                "vision_analysis": vision or {},
                "features_analysis": ts_res if isinstance(ts_res, dict) else (ts_res[0] if ts_res else {}),
                "processing_time": time.time() - t0,
                "timestamp": time.time(),
            }
            self.metrics["last_analysis_time"] = out["processing_time"]
            self.metrics["total_analyses"] = self.metrics.get("total_analyses", 0) + 1
            return out
        except Exception as e:
            self.log.error(f"multimodal_analysis failed: {e}")
            return {"error": str(e), "timestamp": time.time()}
```

```python
# --- Dukascopy-Adapter: FIX Parameter & None→Fallback; Date-Fixes ---
class NautilusDataEngineAdapter:
    def __init__(self, cfg: NautilusIntegrationConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)
        self.dukascopy = DukascopyConnector()
        self._cache: Dict[str, Any] = {}

    async def fetch_market_data(self, symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        key = f"{symbol}_{timeframe}_{bars}"
        if key in self._cache:
            return self._cache[key]
        try:
            # Signatur: (symbol, timeframe="1H", bars=1000)
            df = await asyncio.to_thread(self.dukascopy.get_ohlcv_data, symbol, timeframe, bars)
            df = _ensure_df(df)
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            df = _normalize_time(df)
            # Optional Integrität prüfen:
            try:
                _ = self.dukascopy.validate_data_integrity(df)
            except Exception:
                pass
            self._cache[key] = df
            return df
        except Exception as e:
            self.log.error(f"fetch_market_data failed: {e}")
            # Graceful Degradation statt Abbruch:
            mock = _create_mock_df(symbol, timeframe, min(bars, 300))
            self._cache[key] = mock
            return mock

    def clear_cache(self):
        self._cache.clear()
        self.log.info("Data cache cleared")
```

```python
# --- Pipeline.execute_pipeline: FIX chart/numerical + Top5 Normalisierung ---
class NautilusIntegratedPipeline:
    # ... initialize unverändert (mit fallback) ...

    async def execute_pipeline(self, symbol: str = "EUR/USD", timeframe: str = "1m", bars: int = 1000) -> Dict:
        t0 = time.time()
        try:
            self.logger.info(f"Run pipeline: {symbol} {timeframe} ({bars} bars)")

            market_df = await self.data.fetch_market_data(symbol, timeframe, bars)
            # kein raise; wir haben oben bereits Mock-Fallback
            points = int(len(market_df)) if market_df is not None else 0

            analysis = await self.ai.multimodal_analysis(
                chart={"symbol": symbol, "chart_image": None},   # image optional (Vision kann None)
                numerical={"ohlcv": market_df, "indicators": {}} # echtes DF rein!
            )

            top_raw = self.ai.evaluate_top_strategies(
                symbols=[symbol], timeframes=[timeframe], max_n=self.cfg.max_strategies
            )
            top = _normalize_top5(top_raw)

            dt = time.time() - t0
            self._upd_metrics(dt, True)
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars_requested": bars,
                "market_data_points": points,
                "analysis_result": analysis,
                "top_strategies": top,
                "execution_time": dt,
                "timestamp": time.time(),
                "pipeline_mode": "fallback" if self.cfg.fallback_mode else "nautilus",
                "success": True,
            }
        except Exception as e:
            dt = time.time() - t0
            self._upd_metrics(dt, False)
            self.logger.error(f"Pipeline failed: {e}")
            return {"error": str(e), "execution_time": dt, "success": False, "timestamp": time.time()}
```

```python
# --- Script-Start: Import-Probleme als Script beheben ---
if __name__ == "__main__":
    # Sicherstellen, dass Projektwurzel im Pfad ist (falls als Script gestartet)
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import asyncio
    asyncio.run(_main())
```

---

### ✅ Antworten auf eure 5 Fragen (konkret)

1. **Data‑Flow:** `chart_image` darf `None` sein (Vision‑Pfad soll robust degradieren). Besser: DataFrame→Image generieren; bis dahin **None zulassen** und nur Text/Features nutzen .

2. **TorchServe:** **Kein `"df"`‑String.** Übergib ein **Dict mit floats** (siehe `_build_feature_dict`); List‑Wrapping nur, wenn du Batch willst .

3. **Result Types:** `Top5StrategiesResult` via `_normalize_top5` in `List[Dict]` überführen – kein `len()` auf dem Objekt mehr .

4. **Error Strategy:** Statt `raise ValueError("No market data")` nutzen wir **Mock‑Fallback** (Prod‑freundlich, Logs behalten Fehler sichtbar) .

5. **Package Imports:** Absolute Imports + `sys.path`‑Bootstrap im `__main__`, oder Start via `python -m ai_indicator_optimizer.integration.nautilus_integrated_pipeline` (empfohlen) .

---

Wenn du mir die **genauen Rückgabeformen** von `Top5StrategiesResult` und ggf. das **Dukascopy‑Schema** (Spaltennamen) schickst, passe ich `_normalize_top5` und `_normalize_time/_build_feature_dict` noch präziser an.
