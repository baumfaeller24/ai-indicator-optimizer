# ChatGPT Integration Analysis Package
**Für passgenau Patches der Nautilus Integration**

## 1. TorchServeHandler - Signatur & Interface

```python
# File: ai_indicator_optimizer/ai/torchserve_handler.py
class TorchServeHandler:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        pass
    
    def is_connected(self) -> bool:
        """Check if TorchServe is available"""
        pass
    
    def handle_batch(self, features_list: List[Dict]) -> List[Dict]:
        """SYNC method - not async!"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        pass
```

## 2. MultimodalAI - Signatur & Interface

```python
# File: ai_indicator_optimizer/ai/multimodal_ai.py
class MultimodalAI:
    def __init__(self, config: Dict):
        # config keys: 'ai_endpoint', 'use_mock', 'debug_mode'
        pass
    
    def analyze_chart_pattern(self, chart_image, numerical_indicators: Optional[Dict] = None) -> PatternAnalysis:
        """SYNC method - not async!"""
        pass
```

## 3. AIStrategyEvaluator - Signatur & Interface

```python
# File: ai_indicator_optimizer/ai/ai_strategy_evaluator.py
class AIStrategyEvaluator:
    def __init__(self, ranking_criteria: List = None, output_dir: str = "data/strategy_evaluation"):
        pass
    
    def evaluate_and_rank_strategies(self, 
                                   symbols: List[str] = None, 
                                   timeframes: List[str] = None, 
                                   max_strategies: int = 5, 
                                   evaluation_mode: str = "comprehensive") -> Top5StrategiesResult:
        """Main evaluation method - SYNC"""
        pass
```

## 4. LiveControlSystem - Signatur & Interface

```python
# File: ai_indicator_optimizer/ai/live_control_system.py
class LiveControlSystem:
    def __init__(self, strategy_id: str, config: Dict, use_redis: bool = True):
        pass
    
    def pause_strategy(self, strategy_id: str):
        pass
    
    def resume_strategy(self, strategy_id: str):
        pass
    
    def update_parameters(self, strategy_id: str, params: Dict):
        pass
    
    def get_system_status(self) -> Dict:
        pass
```

## 5. DukascopyConnector - Signatur & Interface

```python
# File: ai_indicator_optimizer/data/dukascopy_connector.py
class DukascopyConnector:
    def __init__(self, config: Optional[DukascopyConfig] = None):
        pass
    
    def get_ohlcv_data(self, symbol: str, timeframe: str = "1H", bars: int = 1000) -> pd.DataFrame:
        """SYNC method - returns DataFrame"""
        pass
    
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass
    
    def cleanup(self):
        pass
```

## 6. FeaturePredictionLogger - Signatur & Interface

```python
# File: ai_indicator_optimizer/logging/feature_prediction_logger.py
class FeaturePredictionLogger:
    def __init__(self, buffer_size: int = 1000, output_dir: str = "logs"):
        pass
    
    def log_prediction(self, entry: Dict):
        pass
    
    def flush(self):
        pass
```

## 7. Nautilus Hardware Config - Signatur & Interface

```python
# File: nautilus_config.py
class NautilusHardwareConfig:
    def __init__(self):
        pass
    
    def create_trading_node_config(self) -> Union[TradingNodeConfig, Dict]:
        """Returns TradingNodeConfig or Dict if Nautilus not available"""
        pass
```

## 8. Import-Struktur & Paket-Layout

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
    └── nautilus_integrated_pipeline.py

nautilus_config.py (root level)
```

## 9. Aktuelle Probleme (ChatGPT's Analyse)

1. **Factory Pattern Fehler:**
```python
# ❌ FALSCH:
integration_config = NautilusIntegratedPipeline(**config)

# ✅ RICHTIG:
integration_config = NautilusIntegrationConfig(**config)
```

2. **Async/Sync Vermischung:**
- TorchServeHandler.handle_batch ist SYNC
- MultimodalAI.analyze_chart_pattern ist SYNC
- Beide brauchen asyncio.to_thread() wrapper

3. **Nautilus API Missbrauch:**
- add_actor() ist nicht öffentliche API
- AI-Services sollten separat gemanagt werden

4. **Import-Probleme:**
- Relative Imports (..ai.*, ..data.*) brechen als Script
- Braucht absolute Imports oder Package-Kontext

## 10. Gewünschte Lösung

ChatGPT soll **passgenau** patchen für:
- Korrekte Async-Wrapper für SYNC-Methoden
- Saubere AI-Service-Manager (ohne Nautilus Actor-System)
- Robuste Import-Struktur
- Korrekte Factory-Pattern Implementation
- Fallback-Mode für Nautilus-freien Betrieb

**Alle Komponenten sind bereits funktionsfähig - nur die Integration braucht Fixes!**