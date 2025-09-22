# 🚀 ChatGPT Integration Fix Package
**Komplette Signaturen für passgenau Patches**

## 📋 **AUFTRAG FÜR CHATGPT:**

Bitte **passgenau patchen** der `nautilus_integrated_pipeline.py` basierend auf diesen **exakten Signaturen** unserer funktionierenden Komponenten.

---

## 1. 🔧 **TorchServeHandler - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/ai/torchserve_handler.py
class TorchServeHandler:
    def __init__(self, config: Optional[TorchServeConfig] = None):
        pass
    
    def process_features(self, 
                        features: Union[Dict[str, Any], List[Dict[str, Any]]], 
                        model_type: str = "pattern_model") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """SYNC method - Main processing (nicht handle_batch!)"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """SYNC method"""
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """SYNC method"""
        pass
```

## 2. 🧠 **MultimodalAI - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/ai/multimodal_ai.py
class MultimodalAI:
    def __init__(self, config: Dict):
        # config keys: 'ai_endpoint', 'use_mock', 'debug_mode'
        pass
    
    def analyze_chart_pattern(self, 
                            chart_image, 
                            numerical_indicators: Optional[Dict] = None) -> PatternAnalysis:
        """SYNC method - nicht async!"""
        pass
```

## 3. 📊 **AIStrategyEvaluator - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/ai/ai_strategy_evaluator.py
class AIStrategyEvaluator:
    def __init__(self, 
                 ranking_criteria: List = None, 
                 output_dir: str = "data/strategy_evaluation"):
        pass
    
    def evaluate_and_rank_strategies(self, 
                                   symbols: List[str] = None, 
                                   timeframes: List[str] = None, 
                                   max_strategies: int = 5, 
                                   evaluation_mode: str = "comprehensive") -> Top5StrategiesResult:
        """SYNC method - Main evaluation method"""
        pass
```

## 4. 🎛️ **LiveControlSystem - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/ai/live_control_system.py
class LiveControlSystem:
    def __init__(self, 
                 strategy_id: str, 
                 config: Optional[Dict] = None, 
                 use_redis: bool = True, 
                 use_kafka: bool = False):
        pass
    
    def start(self):
        """SYNC method"""
        pass
    
    def stop(self):
        """SYNC method"""
        pass
    
    def send_command(self, command_type: str, parameters: Dict[str, Any] = None) -> bool:
        """SYNC method"""
        pass
    
    def get_current_status(self) -> LiveControlStatus:
        """SYNC method"""
        pass
    
    def is_trading_allowed(self) -> bool:
        """SYNC method"""
        pass
```

## 5. 📈 **DukascopyConnector - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/data/dukascopy_connector.py
class DukascopyConnector:
    def __init__(self, config: Optional[DukascopyConfig] = None):
        pass
    
    def get_ohlcv_data(self, 
                      symbol: str, 
                      timeframe: str = "1H", 
                      bars: int = 1000) -> pd.DataFrame:
        """SYNC method - returns DataFrame"""
        pass
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """SYNC method"""
        pass
    
    def cleanup(self):
        """SYNC method"""
        pass
```

## 6. 📝 **FeaturePredictionLogger - EXAKTE Signatur**

```python
# File: ai_indicator_optimizer/logging/feature_prediction_logger.py
class FeaturePredictionLogger:
    def __init__(self, buffer_size: int = 1000, output_dir: str = "logs"):
        pass
    
    def log_prediction(self, entry: Dict):
        """SYNC method"""
        pass
    
    def flush(self):
        """SYNC method"""
        pass
```

## 7. ⚙️ **NautilusHardwareConfig - EXAKTE Signatur**

```python
# File: nautilus_config.py
class NautilusHardwareConfig:
    def __init__(self):
        pass
    
    def create_trading_node_config(self) -> Union[TradingNodeConfig, Dict]:
        """SYNC method - Returns TradingNodeConfig or Dict if mock mode"""
        pass
```

---

## 🚨 **KRITISCHE PROBLEME ZU FIXEN:**

### **Problem 1: Import-Struktur**
```python
# ❌ AKTUELL (bricht als Script):
from ..ai.multimodal_ai import MultimodalAI
from ..ai.ai_strategy_evaluator import AIStrategyEvaluator
from ..ai.torchserve_handler import TorchServeHandler
from ..ai.live_control_system import LiveControlSystem
from ..data.dukascopy_connector import DukascopyConnector
from ..logging.feature_prediction_logger import FeaturePredictionLogger

# ✅ GEWÜNSCHT (absolute Imports):
from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI
from ai_indicator_optimizer.ai.ai_strategy_evaluator import AIStrategyEvaluator
from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler
from ai_indicator_optimizer.ai.live_control_system import LiveControlSystem
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector
from ai_indicator_optimizer.logging.feature_prediction_logger import FeaturePredictionLogger
```

### **Problem 2: Factory Pattern**
```python
# ❌ AKTUELL (falsche Klasse):
def create_nautilus_pipeline(config: Optional[Dict] = None) -> NautilusIntegratedPipeline:
    if config:
        integration_config = NautilusIntegratedPipeline(**config)  # ❌ FALSCH!
    else:
        integration_config = NautilusIntegrationConfig()
    return NautilusIntegratedPipeline(integration_config)

# ✅ GEWÜNSCHT (korrekte Klasse):
def create_nautilus_pipeline(config: Optional[Dict] = None) -> NautilusIntegratedPipeline:
    if config:
        integration_config = NautilusIntegrationConfig(**config)  # ✅ RICHTIG!
    else:
        integration_config = NautilusIntegrationConfig()
    return NautilusIntegratedPipeline(integration_config)
```

### **Problem 3: Async/Sync Vermischung**
```python
# ❌ AKTUELL (SYNC-Methoden als async aufgerufen):
vision_result = await self._ai_services['multimodal'].analyze_chart_pattern(chart_data)
features_result = await self._ai_services['torchserve'].handle_batch([numerical_data])

# ✅ GEWÜNSCHT (SYNC-Methoden mit asyncio.to_thread):
vision_result = await asyncio.to_thread(
    self._ai_services['multimodal'].analyze_chart_pattern,
    chart_data.get('chart_image', chart_data)
)
features_result = await asyncio.to_thread(
    self._ai_services['torchserve'].process_features,  # Korrekte Methode!
    [numerical_data]
)
```

### **Problem 4: Nautilus Actor API Missbrauch**
```python
# ❌ AKTUELL (nicht öffentliche API):
self.ai_actor = AIServiceActor(self.config)
self.trading_node.add_actor(self.ai_actor)  # ❌ Nicht öffentliche API!

# ✅ GEWÜNSCHT (AI-Services separat managen):
class AIServiceManager:
    def __init__(self, config):
        self.services = {}
    
    async def start(self):
        self.services['torchserve'] = TorchServeHandler()
        self.services['multimodal'] = MultimodalAI(config)
        # etc.
```

---

## 🎯 **GEWÜNSCHTES ERGEBNIS:**

1. ✅ **Saubere Import-Struktur** (absolute Imports)
2. ✅ **Korrekte Factory-Pattern** Implementation
3. ✅ **Robuste Async-Wrapper** für alle SYNC-Methoden
4. ✅ **AI-Service-Manager** (ohne Nautilus Actor-System)
5. ✅ **Fallback-Mode** für Nautilus-freien Betrieb
6. ✅ **Korrekte Methoden-Namen** (process_features statt handle_batch)

## 📁 **DATEIEN-STRUKTUR:**
```
ai_indicator_optimizer/
├── ai/
│   ├── multimodal_ai.py
│   ├── ai_strategy_evaluator.py
│   ├── torchserve_handler.py
│   └── live_control_system.py
├── data/
│   └── dukascopy_connector.py
├── logging/
│   └── feature_prediction_logger.py
└── integration/
    └── nautilus_integrated_pipeline.py  # ← Diese Datei fixen!

nautilus_config.py (root level)
```

**Alle Komponenten funktionieren bereits - nur die Integration braucht passgenau Patches!**

---

## 🚀 **AUFTRAG:**
Bitte erstelle **passgenau Patches** für `nautilus_integrated_pipeline.py` basierend auf diesen exakten Signaturen. Fokus auf **robuste Integration** ohne Änderung der funktionierenden Komponenten.