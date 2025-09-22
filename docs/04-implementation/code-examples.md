# 💻 Code-Beispiele - AI-Indicator-Optimizer
## Praktische Implementierungsbeispiele

**Datum:** 22. September 2025 (Reorganisiert)  
**Quelle:** tradingbeispiele.md und bestehende Implementierung  
**Zweck:** Praktische Beispiele für Entwickler  

---

## 1️⃣ **Dataset-Erzeugung aus Bar-Features (inkl. Labeling)**

### **BarDatasetBuilder - Forward-Return-Labeling**

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

### **Nutzung in AI-Strategy:**

```python
# in AIPatternStrategy.on_start()
from dataset_builder import BarDatasetBuilder
self.ds = BarDatasetBuilder(horizon=5)

# in AIPatternStrategy.on_bar(bar)
self.ds.on_bar(bar)

# in AIPatternStrategy.on_stop()
self.ds.to_parquet("datasets/eurusd_bars_h5.parquet")
```

---

## 2️⃣ **TorchServe-Handler für Feature-JSON-Processing**

### **Production-Ready TorchServe Handler**

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

### **Request-Beispiel aus der Strategy:**

```python
import requests
features = {"open":1.1,"high":1.12,"low":1.095,"close":1.115,"volume":12345}
r = requests.post("http://localhost:8080/predictions/barfeature", json=features, timeout=2.0)
print(r.json())  # [{"action":"BUY","confidence":0.78,"reasoning":"simple_bar_model"}]
```

---

## 3️⃣ **Feature & Prediction Logging (Parquet)**

### **FeaturePredictionLogger - Production-Ready**

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

### **Einbindung in Strategy:**

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

---

## 4️⃣ **Multimodale KI-Integration (Geplant)**

### **OllamaVisionClient - Für Baustein A2**

```python
# Zu implementieren in Baustein A2:
class OllamaVisionClient:
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.model = "minicpm-v:latest"  # Vision-fähiges Model
    
    def analyze_chart_image(self, chart_bytes: bytes, prompt: str) -> Dict:
        """Chart-Bild an Ollama Vision senden für Pattern-Erkennung"""
        # POST request to Ollama with image + prompt
        # Return structured analysis: trends, support/resistance, patterns
        pass
    
    def extract_visual_features(self, chart_bytes: bytes) -> Dict:
        """Strukturierte visuelle Features extrahieren"""
        # Candlestick patterns, trend direction, volatility
        pass
```

### **MultimodalFusionEngine - Für Baustein B1**

```python
# Zu implementieren in Baustein B1:
class MultimodalFusionEngine:
    def __init__(self):
        self.feature_extractor = EnhancedFeatureExtractor()
        self.vision_client = OllamaVisionClient()
    
    def fuse_vision_and_indicators(self, chart_bytes: bytes, ohlcv_data: List) -> Dict:
        # 1. Technische Indikatoren (bestehend)
        technical_features = self.feature_extractor.extract_features(ohlcv_data)
        
        # 2. Vision-Features (neu)
        visual_features = self.vision_client.extract_visual_features(chart_bytes)
        
        # 3. Multimodale Fusion
        return self.combine_multimodal_features(technical_features, visual_features)
    
    def combine_multimodal_features(self, technical: Dict, visual: Dict) -> Dict:
        """Intelligente Kombination beider Feature-Sets"""
        # Weighted combination, confidence scoring, conflict resolution
        pass
```

---

## 5️⃣ **Schema-Problem Lösung (Baustein A1)**

### **UnifiedSchemaManager - Quick Win**

```python
# Zu implementieren in Baustein A1:
class UnifiedSchemaManager:
    def __init__(self):
        self.streams = {
            'technical_features': 'logs/unified/technical_features_{timestamp}.parquet',
            'ml_dataset': 'logs/unified/ml_dataset_{timestamp}.parquet',
            'ai_predictions': 'logs/unified/ai_predictions_{timestamp}.parquet',
            'performance_metrics': 'logs/unified/performance_metrics_{timestamp}.parquet'
        }
    
    def create_separate_logging_streams(self):
        """Separate Logging-Streams für verschiedene Datentypen"""
        # Verhindert Schema-Mismatch durch getrennte Dateien
        pass
    
    def validate_schema_compatibility(self, stream_type: str, data: Dict):
        """Schema-Validierung vor Parquet-Writes"""
        # Prüft Schema-Konsistenz vor dem Schreiben
        pass
```

---

## 6️⃣ **CLI Interface - Aktuell Implementiert**

### **Main Application CLI**

```python
# Bereits implementiert in main_application.py:
def main():
    parser = argparse.ArgumentParser(description="AI Indicator Optimizer")
    parser.add_argument("--mode", choices=["demo", "live", "backtest"], default="demo")
    parser.add_argument("--symbol", default="EUR/USD")
    parser.add_argument("--timeframe", default="1m")
    
    args = parser.parse_args()
    
    # Hardware Detection
    hardware_info = detect_hardware()
    
    # AI Integration (Ollama)
    ai_client = OllamaClient()
    
    # Run selected mode
    if args.mode == "demo":
        run_demo_mode(args.symbol, args.timeframe)
    elif args.mode == "live":
        run_live_mode(args.symbol, args.timeframe)
    elif args.mode == "backtest":
        run_backtest_mode(args.symbol, args.timeframe)
```

---

## 🎯 **Nächste Implementierungsschritte**

### **Sofort umsetzbar:**
1. **Schema-Problem beheben** (Baustein A1) - 2-3 Tage
2. **Ollama Vision-Client** erweitern (Baustein A2) - 3-4 Tage
3. **Chart-Vision-Integration** (Baustein A3) - 4-5 Tage

### **Mittelfristig:**
1. **Multimodale Fusion-Engine** (Baustein B1)
2. **KI-Enhanced Pine Script Generator** (Baustein C1)
3. **Top-5-Strategien-Ranking** (Baustein C2)

**Detaillierte Roadmap:** Siehe [multimodal-roadmap.md](../05-roadmaps/multimodal-roadmap.md)

---

**Erstellt:** 22. September 2025  
**Basierend auf:** tradingbeispiele.md + bestehende Implementierung  
**Zweck:** Praktische Entwicklungsrichtlinien