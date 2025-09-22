# Exakte Signaturen für ChatGPT Patches

## 1. TorchServeHandler - TATSÄCHLICHE Signatur

```python
# File: ai_indicator_optimizer/ai/torchserve_handler.py
class TorchServeHandler:
    def __init__(self, config: Optional[TorchServeConfig] = None):
        pass
    
    def process_features(self, features: Union[Dict[str, Any], List[Dict[str, Any]]], model_type: str = "pattern_model") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """SYNC method - Main processing method (not handle_batch!)"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        pass
```

## 2. LiveControlSystem - TATSÄCHLICHE Signatur

```python
# File: ai_indicator_optimizer/ai/live_control_system.py
# ACHTUNG: Datei existiert, aber Interface unbekannt
# Muss geprüft werden!
```

## 3. Nautilus Config - TATSÄCHLICHE Signatur

```python
# File: nautilus_config.py
class NautilusHardwareConfig:
    def __init__(self):
        pass
    
    def create_trading_node_config(self) -> Union[TradingNodeConfig, Dict]:
        """Returns TradingNodeConfig or Dict if mock mode"""
        pass
```

## 4. Import-Probleme - TATSÄCHLICHE Struktur

```python
# PROBLEM: Relative Imports in nautilus_integrated_pipeline.py
from ..ai.multimodal_ai import MultimodalAI                    # ❌ Bricht als Script
from ..ai.ai_strategy_evaluator import AIStrategyEvaluator     # ❌ Bricht als Script
from ..ai.torchserve_handler import TorchServeHandler         # ❌ Bricht als Script
from ..ai.live_control_system import LiveControlSystem        # ❌ Bricht als Script
from ..data.dukascopy_connector import DukascopyConnector     # ❌ Bricht als Script
from ..logging.feature_prediction_logger import FeaturePredictionLogger  # ❌ Bricht als Script

# LÖSUNG: Absolute Imports
from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI
from ai_indicator_optimizer.ai.ai_strategy_evaluator import AIStrategyEvaluator
from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler
from ai_indicator_optimizer.ai.live_control_system import LiveControlSystem
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector
from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger
```

## 5. Factory Pattern - TATSÄCHLICHER Fehler

```python
# AKTUELLER FEHLER in nautilus_integrated_pipeline.py:
def create_nautilus_pipeline(config: Optional[Dict] = None) -> NautilusIntegratedPipeline:
    if config:
        integration_config = NautilusIntegratedPipeline(**config)  # ❌ FALSCH!
    else:
        integration_config = NautilusIntegrationConfig()
    
    return NautilusIntegratedPipeline(integration_config)

# KORREKTE VERSION:
def create_nautilus_pipeline(config: Optional[Dict] = None) -> NautilusIntegratedPipeline:
    if config:
        integration_config = NautilusIntegrationConfig(**config)  # ✅ RICHTIG!
    else:
        integration_config = NautilusIntegrationConfig()
    
    return NautilusIntegratedPipeline(integration_config)
```

## 6. Async/Sync Probleme - TATSÄCHLICHE Methoden

```python
# PROBLEM: Diese Methoden sind SYNC, werden aber als ASYNC aufgerufen:

# TorchServeHandler.process_features() - SYNC
# MultimodalAI.analyze_chart_pattern() - SYNC  
# AIStrategyEvaluator.evaluate_and_rank_strategies() - SYNC
# DukascopyConnector.get_ohlcv_data() - SYNC

# LÖSUNG: asyncio.to_thread() wrapper für alle SYNC-Methoden
```

## 7. Nautilus API Problem - TATSÄCHLICHER Code

```python
# AKTUELLER PROBLEMATISCHER CODE:
self.ai_actor = AIServiceActor(self.config)
self.trading_node.add_actor(self.ai_actor)  # ❌ Nicht öffentliche API!

# CHATGPT'S EMPFEHLUNG:
# AI-Services SEPARAT managen, nicht über Nautilus Actor-System
```

## 8. Benötigte Dependency-Checks

**ChatGPT braucht diese Informationen:**

1. **LiveControlSystem Interface** - Muss geprüft werden
2. **Nautilus Version** - Welche API-Version verwenden wir?
3. **Alle SYNC vs ASYNC Methoden** - Für korrekte Wrapper

## 9. Gewünschtes Ergebnis

ChatGPT soll **diese spezifischen Probleme** fixen:

1. ✅ Relative → Absolute Imports
2. ✅ Factory Pattern Korrektur  
3. ✅ SYNC-Methoden mit asyncio.to_thread() wrappen
4. ✅ AI-Services separat managen (nicht über Nautilus Actor)
5. ✅ Robuste Fallback-Modi

**Alle bestehenden Komponenten funktionieren - nur die Integration braucht Fixes!**