Alles klar — hier sind kompakte, lauffähige Beispiele zu den drei Punkten:

# 1) Dataset-Erzeugung aus Bar-Features (inkl. Labeling)

```python
# dataset_builder.py
from typing import List, Dict
from collections import deque
import polars as pl
from nautilus_trader.model.data import Bar

class BarDatasetBuilder:
    """Sammelt Features pro Bar und erzeugt Labels per Forward-Return."""
    def __init__(self, horizon=5, min_bars=10):
        self.rows: List[Dict] = []
        self.buffer = deque(maxlen=horizon+1)
        self.horizon = horizon
        self.min_bars = min_bars

    def on_bar(self, bar: Bar):
        feat = {
            "ts": int(bar.ts_init),  # ns
            "instrument": str(bar.bar_type.instrument_id),
            "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close),
            "volume": float(bar.volume),
        }
        feat["price_change"] = feat["close"] - feat["open"]
        rng = feat["high"] - feat["low"]
        feat["body_ratio"] = abs(feat["price_change"]) / max(rng, 1e-6)

        self.buffer.append(feat)
        if len(self.buffer) == self.buffer.maxlen:
            x = self.buffer[0]
            y_close = self.buffer[-1]["close"]
            fwd_ret = (y_close / x["close"]) - 1.0  # Label: prozentuale Vorwärtsrendite
            x["label_fwd_ret@%d" % self.horizon] = fwd_ret
            # optional: diskrete Klassen
            x["label_class"] = 0 if fwd_ret > 0.0003 else (1 if fwd_ret < -0.0003 else 2)  # buy/sell/hold
            self.rows.append(x.copy())

    def to_parquet(self, path: str):
        if len(self.rows) < self.min_bars:
            print("Not enough bars to write dataset.")
            return
        pl.DataFrame(self.rows).write_parquet(path)
```

**Nutzung in deiner Strategy (kurz):**

```python
# in AIPatternStrategy.on_start()
from dataset_builder import BarDatasetBuilder
self.ds = BarDatasetBuilder(horizon=5)

# in AIPatternStrategy.on_bar(bar)
self.ds.on_bar(bar)

# in AIPatternStrategy.on_stop()
self.ds.to_parquet("datasets/eurusd_bars_h5.parquet")
```

# 2) TorchServe-Handler, der dein Feature-JSON verarbeitet

```python
# tick_handler.py (TorchServe)
from ts.torch_handler.base_handler import BaseHandler
import torch, json
import torch.nn as nn

class SimpleBarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 3)  # buy/sell/hold
        )
    def forward(self, x):
        return self.net(x)

def extract_vector(d: dict):
    # Gleiche Features wie in deiner Strategy:
    open_, high, low, close, vol = map(float, (d["open"], d["high"], d["low"], d["close"], d["volume"]))
    price_change = close - open_
    body_ratio = abs(price_change) / max(high - low, 1e-6)
    return [price_change, body_ratio, open_, close, vol, high - low]

class BarFeatureHandler(BaseHandler):
    def initialize(self, ctx):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimpleBarModel().to(self.device)
        # Optional: Gewichte laden
        # state = torch.load("model.pt", map_location=self.device)
        # self.model.load_state_dict(state)
        self.model.eval()

    def handle(self, data, ctx):
        # Supports batch: either list of dicts or single dict
        payload = data[0].get("body")
        if isinstance(payload, (bytes, bytearray)): payload = payload.decode("utf-8")
        payload = json.loads(payload) if isinstance(payload, str) else payload
        batch = payload if isinstance(payload, list) else [payload]

        X = torch.tensor([extract_vector(x) for x in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X)
            prob = torch.softmax(logits, dim=1)
            conf, pred = torch.max(prob, dim=1)

        out = []
        for p, c in zip(pred.tolist(), conf.tolist()):
            action = ["BUY", "SELL", "HOLD"][p]
            out.append({"action": action, "confidence": float(c), "reasoning": "simple_bar_model"})
        return out
```

**Request-Beispiel aus der Strategy:**

```python
import requests
features = {"open":1.1,"high":1.12,"low":1.095,"close":1.115,"volume":12345}
r = requests.post("http://localhost:8080/predictions/barfeature", json=features, timeout=2.0)
print(r.json())  # [{"action":"BUY","confidence":0.78,"reasoning":"simple_bar_model"}]
```

# 3) Logging-Template für Features & Predictions (CSV/Parquet)

```python
# feature_logger.py
import polars as pl
from typing import List, Dict
from datetime import datetime

class FeaturePredictionLogger:
    def __init__(self, out_parquet="logs/ai_features.parquet", buffer_size=5000):
        self.out = out_parquet
        self.buf: List[Dict] = []
        self.n = 0
        self.buffer_size = buffer_size

    def log(self, *, ts_ns: int, instrument: str, features: Dict, prediction: Dict):
        row = {
            "ts": ts_ns,
            "time": datetime.utcfromtimestamp(ts_ns/1e9).isoformat(),
            "instrument": instrument,
            **{f"f_{k}": v for k, v in features.items()},
            "pred_action": prediction.get("action"),
            "pred_confidence": prediction.get("confidence"),
            "pred_reason": prediction.get("reasoning"),
        }
        self.buf.append(row)
        self.n += 1
        if len(self.buf) >= self.buffer_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        df = pl.DataFrame(self.buf)
        mode = "wb" if self.n == len(self.buf) else "ab"
        df.write_parquet(self.out, compression="zstd")  # idempotent append via file rotation empfiehlt sich
        self.buf.clear()

    def close(self):
        self.flush()
```

**Einbindung in deine Strategy:**

```python
# on_start
from feature_logger import FeaturePredictionLogger
self.flog = FeaturePredictionLogger("logs/eurusd_ai.parquet", buffer_size=2000)

# on_bar
features = self._extract_features(bar)
pred = self._get_ai_prediction(features) or {"action":"HOLD","confidence":0.0,"reasoning":"no_pred"}
self.flog.log(
    ts_ns=int(bar.ts_init),
    instrument=str(bar.bar_type.instrument_id),
    features=features,
    prediction=pred,
)

# on_stop
self.flog.close()
```

**Optional: Rotation pro Tag**

- Schreibe nach `logs/ai_{YYYYMMDD}.parquet` basierend auf `ts` → einfaches Re-Training & Auditing.

Fertig.
