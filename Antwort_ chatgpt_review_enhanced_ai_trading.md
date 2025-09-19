kurz & knackig: ich hab deine Dateien gründlich geprüft. Hier sind die **konkreten Fehler**, **Ursachen** und **sofort umsetzbare Fixes/Optimierungen** – plus Mini-Patches.

# Fix-Liste (direkt einbaubar)

## 1) Strategy: kaputte Aufrufe & Order-API härten (64:0)

- **Import-Pfade korrigieren** (oder echtes Paket anlegen). Schnelllösung: relative Imports nutzen.

- **Fehlende Methoden implementieren** oder Aufrufe entfernen.

- **Order-Platzierung vereinfachen**: nutze `self.submit_market_order(...)` statt manuell `MarketOrder(...)`.

**Patch (Kernausschnitte):**

```python
# oben: lokale Imports statt ai_indicator_optimizer...
from feature_prediction_logger import FeaturePredictionLogger
from bar_dataset_builder import BarDatasetBuilder

def on_bar(self, bar):
    ...
    prediction = self._get_ai_prediction(features)
    # fallback: wenn du (noch) kein Enhanced-Scoring hast
    conf = prediction.get("confidence", 0.0) if prediction else 0.0

    if self.feature_logger and prediction:
        self.feature_logger.log(
            ts_ns=int(bar.ts_init),
            instrument=str(bar.bar_type.instrument_id),
            features=features,
            prediction=prediction,
            confidence_score=conf,
            risk_score=prediction.get("risk_score", 0.0),
            market_regime="unknown",
        )

    if prediction and conf > self.min_confidence:
        self._execute_signal(prediction, bar)  # benutze die vorhandene Methode

def _submit_market_order(self, side: OrderSide, bar):
    try:
        qty = self.base_position_size  # feste Größe; optional: conf-basiert skalieren
        self.submit_market_order(side.name, quantity=qty)  # einfache, robuste API
        self.log.info(f"🟢 Submitted {side.name} {qty}")
    except Exception as e:
        self.log.error(f"❌ Order submission failed: {e}")
```

> Wenn du **Enhanced-Confidence** willst, füge eine einfache Version ein:

```python
def _calculate_enhanced_confidence(self, pred: dict, features: dict) -> float:
    if not pred: return 0.0
    c = float(pred.get("confidence", 0.0))
    risk = float(pred.get("risk_score", 0.0))
    return max(0.0, min(1.0, c * (1.0 - 0.5*risk)))
```

und rufe diese statt der nicht existierenden Methoden.

## 2) Logger: Factory + Rotation + O(1)-Flush (67:0)

- **Fehlende Factory ergänzen** (für Tests).

- **Rotating-Logger fertigstellen** (täglicher Pfad + `super().log`).

- **Flush beschleunigen**: nicht bestehende Datei einlesen/konkattenieren; stattdessen **append by partition** → tägliche Datei pro Datum (Rotation löst das).

**Patch (Kernausschnitte):**

```python
# am Ende von feature_prediction_logger.py
def create_feature_logger(base_path:str, buffer_size:int=5000, rotating:bool=False):
    if rotating:
        return RotatingFeaturePredictionLogger(base_path=base_path, buffer_size=buffer_size)
    return FeaturePredictionLogger(output_path=f"{base_path}.parquet", buffer_size=buffer_size)

class RotatingFeaturePredictionLogger(FeaturePredictionLogger):
    ...
    def _get_daily_path(self, ts_ns: int=None) -> str:
        d = datetime.utcfromtimestamp((ts_ns or int(datetime.utcnow().timestamp()*1e9)) / 1e9).strftime("%Y%m%d")
        return f"{self.base_path}_{d}.parquet"

    def log(self, *, ts_ns: int, **kwargs) -> None:
        new_path = Path(self._get_daily_path(ts_ns))
        if self.output_path != new_path:
            if self.buffer: self.flush()
            self.output_path = new_path
        return super().log(ts_ns=ts_ns, **kwargs)

def _safe_write_parquet(path: Path, df: pl.DataFrame, compression: str):
    # direkter Write ohne vorheriges Lesen – Rotation vermeidet Re-Write großer Dateien
    df.write_parquet(path, compression=compression)

# in FeaturePredictionLogger.flush():
# ersetze den bestehenden Lese+Concat-Block durch:
df = pl.DataFrame(self.buffer)
_safe_write_parquet(self.output_path, df, self.compression) if not self.output_path.exists() \
    else df.write_parquet(self.output_path, compression=self.compression, mode="ab")  # Polars 1.5+: falls verfügbar
# Falls append mode nicht verfügbar: behalte Rotation bei und schreibe immer neue Tagesdatei.
```

> Falls dein Polars **kein `mode="ab"`** unterstützt: belasse Rotation und **überschreibe Tagesdatei in Batches** (oder sharde zusätzlich per Stunde).

## 3) Dataset-Builder: `to_parquet` & `get_stats` fertigstellen (65:0)

- **Implementierung vervollständigen**, **Metadaten speichern**, **Stats anbieten**.

**Patch (Kernausschnitte):**

```python
def to_parquet(self, path: str, compression: str="zstd", include_metadata: bool=True) -> bool:
    if len(self.rows) < self.min_bars:
        self.logger.warning(f"Not enough bars ({len(self.rows)}) ...")
        return False
    try:
        df = pl.DataFrame(self.rows).sort("ts_ns")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression=compression)
        if include_metadata:
            meta = self.get_stats()
            Path(Path(path).with_suffix(".json")).write_text(pl.Series([meta]).to_list()[0].__repr__())
        return True
    except Exception as e:
        self.logger.error(f"Error writing parquet: {e}")
        return False

def get_stats(self) -> Dict[str, any]:
    if not self.rows: 
        return {"total_entries": 0, "label_distribution": {}, "return_statistics": {}}
    df = pl.DataFrame(self.rows)
    total = len(df)
    labels = df.select(pl.col(f"label_name_h{self.horizon}")).to_series().value_counts()
    rets = df.select(pl.col(f"label_fwd_ret_h{self.horizon}")).to_series()
    return {
        "total_entries": total,
        "label_distribution": dict(zip(labels[""], labels["counts"])) if hasattr(labels, "__getitem__") else {},
        "return_statistics": {
            "mean": float(rets.mean()), "std": float(rets.std()),
            "min": float(rets.min()), "max": float(rets.max())
        }
    }
```

## 4) Tests reparieren (71:0)

- **Imports** an reale Module anpassen.

- **Factory verwenden**.

- **Abgeschnittene Zeile** fixen.

**Patch (Kernausschnitte):**

```python
from feature_prediction_logger import FeaturePredictionLogger, create_feature_logger
from bar_dataset_builder import BarDatasetBuilder
...
prediction = {
    "action": ["BUY","SELL","HOLD"][i % 3],
    "confidence": 0.6 + i * 0.02
}
```

# Weitere Optimierungen

- **Timeout/Retry** für AI-Requests (Strategy): `timeout=2.0`, `backoff` bei Fehlern.

- **Konfigurierbare Pfade** via ENV (Logs/Datasets).

- **Thread-Safety**: Logger in Multi-Thread/Process nur per Rotation + pro-Prozess Datei verwenden.

- **Quantities/Position-Sizing** sauber zentralisieren (z. B. `size = int(self.base_position_size * conf)` mit Cap).

Wenn du willst, liefere ich dir die **fertigen, korrigierten Dateien** als Paket (Strategy, Logger mit Factory & Rotation, Dataset-Builder, Tests) direkt einsatzbereit.
