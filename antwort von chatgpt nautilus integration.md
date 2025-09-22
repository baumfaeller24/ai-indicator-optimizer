Hier ist ein **präziser Review** mit **Showstoppern + Fixes** (inkl. Patch‑Snippets) zu deiner Integrationsdatei. Ich beziehe mich direkt auf den hochgeladenen Inhalt .

---

## ❌ Showstopper / Inkonsistenzen

1. **Dateiformat & Einrückung**  
   Die Datei ist ein Mischmasch aus Markdown + Python; zahlreiche Blöcke sind **falsch eingerückt** (z. B. nach `try:`/`except:`), was sofort zu `IndentationError`/`SyntaxError` führt. → Als **.py** oder strikt per Codefence in **.md**.

2. **Nautilus-API-Nutzung**
- `TradingNode.add_actor(self.ai_actor)` ist höchstwahrscheinlich **nicht** öffentliche API. Üblich: **Gateways/Strategies** zum Node hinzufügen; “Actor-System” ist intern/instabil.

- `await self.trading_node.start_async()` ist fraglich; in vielen Versionen heißt es eher `start()`/`run()`.  
  → **Empfehlung:** AI‑Services **außerhalb** des Nodes managen; Node nur für Market‑Data/Orders/Strategien nutzen.
3. **Import-/Paketstruktur**  
   `from ..ai.*` und `from ..data.*` setzen **Package‑Kontext** voraus. Als Script brechen diese **relative Imports**. → Absolut importieren oder `__package__` sauber setzen.

4. **Wrong Factory**

```python
if config:
    integration_config = NautilusIntegratedPipeline(**config)   # ❌
else:
    integration_config = NautilusIntegrationConfig()
return NautilusIntegratedPipeline(integration_config)
```

→ Muss `NautilusIntegrationConfig(**config)` bauen, **nicht** die Pipeline.

5. **Falsche/unsichere Referenzen**
- `from nautilus_config import NautilusHardwareConfig` (nicht vorhandener Name – zuvor hieß das bei dir `HardwareOptimizedConfig`).

- `DataEngine`/`Actor`‑Imports können je nach Nautilus‑Version abweichen/entfallen.

- `AIStrategyEvaluator.evaluate_and_rank_strategies(...)` wird aufgerufen, **aber das Ergebnis nicht genutzt**; danach wird eine eigene Sortierung über die Input‑Strategien gemacht.
6. **Async/Threading Mix**
- `TorchServeHandler.handle_batch` wird `await`et – ist deine Implementierung wirklich `async`? Wenn **sync**, per `asyncio.to_thread(...)` aufrufen.

- `Ollama analyze_chart_pattern` wird korrekt via `to_thread` gewrapped.

---

## ✅ Korrigierende Patch‑Snippets (kompakt)

### A) Import/Fallback & saubere Struktur

```python
# imports (oben, sauber eingerückt)
import asyncio, time, json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from nautilus_trader.trading.node import TradingNode
    from nautilus_trader.config import TradingNodeConfig
    NAUTILUS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Nautilus not available: {e}. Using fallback mode.")
    NAUTILUS_AVAILABLE = False

# AI-Komponenten: absolute Imports bevorzugen
from ai.multimodal_ai import MultimodalAI
from ai.ai_strategy_evaluator import AIStrategyEvaluator
from ai.torchserve_handler import TorchServeHandler
from ai.live_control_system import LiveControlSystem
from data.dukascopy_connector import DukascopyConnector
```

### B) Config/Factory fixen

```python
@dataclass
class NautilusIntegrationConfig:
    trader_id: str = "AI-OPTIMIZER-001"
    instance_id: str = "001"
    use_nautilus: bool = True
    fallback_mode: bool = False
    torchserve_endpoint: str = "http://localhost:8080/predictions/pattern_model"
    ollama_endpoint: str = "http://localhost:11434"
    redis_host: str = "localhost"
    redis_port: int = 6379
    max_workers: int = 32
    batch_size: int = 1000
    timeout_seconds: int = 30
    min_confidence: float = 0.5
    max_strategies: int = 5

def create_nautilus_pipeline(config: Optional[Dict] = None):
    cfg = NautilusIntegrationConfig(**config) if config else NautilusIntegrationConfig()
    return NautilusIntegratedPipeline(cfg)
```

### C) **Kein add_actor** – AI‑Services separat managen

```python
class AIServiceManager:
    def __init__(self, cfg: NautilusIntegrationConfig):
        self.cfg = cfg
        self.log = logging.getLogger(__name__)
        self.svc = {}
        self.metrics = {}

    async def start(self):
        # TorchServe
        self.svc["torchserve"] = TorchServeHandler(base_url=self.cfg.torchserve_endpoint.rsplit("/predictions",1)[0],
                                                   timeout=self.cfg.timeout_seconds)
        # Ollama
        self.svc["multimodal"] = MultimodalAI({"ai_endpoint": self.cfg.ollama_endpoint, "use_mock": False})
        # Live Control
        self.svc["live_control"] = LiveControlSystem(strategy_id=self.cfg.trader_id,
                                                     config={"redis_host": self.cfg.redis_host,"redis_port": self.cfg.redis_port},
                                                     use_redis=True)
        # Evaluator
        self.svc["evaluator"] = AIStrategyEvaluator()

    async def stop(self):
        for s in self.svc.values():
            if hasattr(s, "cleanup"):
                await asyncio.to_thread(s.cleanup)
```

### D) TradingNode starten (ohne Actors)

```python
class NautilusIntegratedPipeline:
    def __init__(self, cfg: NautilusIntegrationConfig):
        self.cfg = cfg
        self.node: Optional[TradingNode] = None
        self.ai: Optional[AIServiceManager] = None
        self.data = NautilusDataEngineAdapter(cfg)
        self.logger = logging.getLogger(__name__)
        self.metrics = {"total_executions":0,"successful_executions":0,"average_execution_time":0.0,"last_execution_time":None}

    async def initialize(self) -> bool:
        self.logger.info("Init pipeline…")
        self.ai = AIServiceManager(self.cfg)
        await self.ai.start()

        if not (NAUTILUS_AVAILABLE and self.cfg.use_nautilus):
            self.cfg.fallback_mode = True
            return True

        try:
            # Hardware config import anpassen:
            from nautilus_hardware_config import HardwareOptimizedConfig  # ← korrigierter Modulname
            hw = HardwareOptimizedConfig()
            tn_cfg: TradingNodeConfig = hw.create_trading_node_config()
            self.node = TradingNode(config=tn_cfg)

            # ⚠️ Starte Node synchron, wenn keine async-API
            await asyncio.to_thread(self.node.start)
            self.logger.info("TradingNode started.")
            return True
        except Exception as e:
            self.logger.error(f"Nautilus init failed: {e}")
            self.cfg.fallback_mode = True
            return True
```

### E) Multimodal/TS‑Aufrufe robust

```python
    async def process_multimodal(self, chart_data: Dict, numerical_data: Dict) -> Dict:
        t0 = time.time()
        try:
            vision = await asyncio.to_thread(self.ai.svc["multimodal"].analyze_chart_pattern,
                                             chart_data.get("chart_image", chart_data))
            # TorchServe: sync? → to_thread
            feats = await asyncio.to_thread(self.ai.svc["torchserve"].handle_batch, [numerical_data])
            out = {
                "vision_analysis": vision or {},
                "features_analysis": (feats[0] if feats else {}),
                "processing_time": time.time()-t0,
            }
            self.ai.metrics["last_analysis_time"] = out["processing_time"]
            self.ai.metrics["total_analyses"] = self.ai.metrics.get("total_analyses",0)+1
            return out
        except Exception as e:
            self.logger.error(f"Multimodal failed: {e}")
            return {"error": str(e)}
```

### F) Strategy‑Bewertung wirklich nutzen

```python
    async def evaluate_strategies(self, strategies: List[Dict]) -> List[Dict]:
        try:
            ev = self.ai.svc["evaluator"]
            ranked = ev.evaluate_and_rank_strategies(candidates=strategies, max_strategies=min(len(strategies), self.cfg.max_strategies))
            return ranked
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return []
```

### G) Pipeline‑Factory/Main fix

```python
async def main():
    pipe = create_nautilus_pipeline()
    ok = await pipe.initialize()
    if not ok:
        print("Init failed"); return
    status = await pipe.get_system_status()
    print(json.dumps(status, indent=2))
    res = await pipe.execute_pipeline(symbol="EUR/USD", timeframe="5m", bars=500)
    print(json.dumps(res, indent=2, default=str))
    await pipe.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ✅ Dependency‑Check (präzise Angaben bitte)

1. **NautilusTrader Version** (Commit/Tag)? (API-Namen variieren → Start/Config/Imports).

2. **TorchServeHandler**: ist `handle_batch` **sync** oder **async**?

3. **Module‑Pfade** deiner AI/Connector‑Pakete (z. B. `ai.multimodal_ai`, `data.dukascopy_connector`).

4. **HardwareConfig‑Modulname** (heißt es bei dir `nautilus_hardware_config.py` mit `HardwareOptimizedConfig`?).

Schick mir gern diese 3–4 Dateien (oder die genauen Signaturen), dann setze ich die Patches **passgenau** für deine Struktur um.
