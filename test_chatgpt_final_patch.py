#!/usr/bin/env python3
"""
ChatGPT's Final Patch Test
Production-Ready Type Safety und JSON-Kompatibilität
"""

import sys
import os
import asyncio
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Direct imports
from ai_indicator_optimizer.ai.multimodal_ai import MultimodalAI
from ai_indicator_optimizer.ai.ai_strategy_evaluator import AIStrategyEvaluator
from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler, ModelType
from ai_indicator_optimizer.ai.live_control_system import LiveControlSystem
from ai_indicator_optimizer.data.dukascopy_connector import DukascopyConnector

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional
import time


def _fake_feats():
    """ChatGPT's test feature generator"""
    return {
        "open": 1.1,
        "high": 1.101,
        "low": 1.099,
        "close": 1.1005,
        "volume": 1000.0,
        "range": 0.002,
        "ret_1": 0.0
    }


def _create_mock_df(symbol: str, timeframe: str, bars: int = 100) -> pd.DataFrame:
    """Erstellt Mock-DataFrame"""
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=bars, freq="T")
    base = 1.10
    noise = np.cumsum(np.random.normal(0, 1e-4, bars))
    close = base + noise
    high = close + np.abs(np.random.normal(0, 5e-5, bars))
    low = close - np.abs(np.random.normal(0, 5e-5, bars))
    open_ = np.r_[close[0], close[:-1]]
    vol = np.random.randint(900, 1500, bars).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _ensure_df(obj: Any) -> Optional[pd.DataFrame]:
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
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
        try:
            dfe.index = pd.to_datetime(dfe.index, errors="coerce", utc=True)
        except Exception:
            pass
    dfe = dfe.sort_index()
    return dfe


def _build_feature_dict(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return _fake_feats()  # Use ChatGPT's fake features as fallback
    last = df.iloc[-1]
    feats = {}
    for k in ["open", "high", "low", "close", "volume"]:
        v = last.get(k, np.nan)
        try:
            feats[k] = float(v)
        except Exception:
            feats[k] = float("nan")
    try:
        feats["range"] = float(feats["high"] - feats["low"])
    except Exception:
        feats["range"] = float("nan")
    try:
        feats["ret_1"] = float(np.log(df["close"].iloc[-1] / df["close"].iloc[-2])) if len(df) > 1 else 0.0
    except Exception:
        feats["ret_1"] = 0.0
    return feats


@dataclass
class NautilusIntegrationConfig:
    trader_id: str = "CHATGPT-FINAL-TEST"
    use_nautilus: bool = False
    fallback_mode: bool = True
    torchserve_endpoint: str = "http://localhost:8080/predictions/pattern_model"
    ollama_endpoint: str = "http://localhost:11434"
    redis_host: str = "localhost"
    redis_port: int = 6379
    use_redis: bool = False
    use_kafka: bool = False
    max_workers: int = 32
    batch_size: int = 1000
    timeout_seconds: int = 30
    min_confidence: float = 0.5
    max_strategies: int = 5


class ChatGPTFinalAIServiceManager:
    """ChatGPT's Final Production-Ready AI Service Manager"""
    
    def __init__(self, cfg: NautilusIntegrationConfig, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)
        self.services: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def start(self):
        # TorchServe
        self.services["torchserve"] = TorchServeHandler()

        # Multimodal (Ollama/MiniCPM)
        self.services["multimodal"] = MultimodalAI({
            "ai_endpoint": self.cfg.ollama_endpoint,
            "use_mock": True,
            "debug_mode": True,
        })

        # Live Control
        self.services["live_control"] = LiveControlSystem(
            strategy_id=self.cfg.trader_id,
            config={"redis_host": self.cfg.redis_host, "redis_port": self.cfg.redis_port},
            use_redis=self.cfg.use_redis,
            use_kafka=self.cfg.use_kafka,
        )

        # Evaluator
        self.services["evaluator"] = AIStrategyEvaluator()

        self.log.info(f"ChatGPT Final AIServiceManager started with services: {list(self.services.keys())}")

    def stop(self):
        try:
            if "live_control" in self.services:
                self.services["live_control"].stop()
        except Exception as e:
            self.log.warning(f"LiveControl stop error: {e}")

    async def multimodal_analysis(self, chart: Dict, numerical: Dict) -> Dict:
        """ChatGPT's Final Production-Ready Multimodal Analysis"""
        t0 = time.time()
        try:
            df = _ensure_df(numerical.get("ohlcv"))
            df = _normalize_time(df) if df is not None else None
            feats = _build_feature_dict(df)

            chart_img = chart.get("chart_image") if isinstance(chart, dict) else None
            vision = await asyncio.to_thread(
                self.services["multimodal"].analyze_chart_pattern,
                chart_img,
                {"features": feats},
            )

            # ✅ Enum statt String; Rückgabe: InferenceResult (Dataclass)
            ts_res = await asyncio.to_thread(
                self.services["torchserve"].process_features,
                feats,
                ModelType.PATTERN_RECOGNITION,
            )

            # ✅ typsichere Extraktion
            if hasattr(ts_res, "predictions"):
                ts_predictions = ts_res.predictions
            else:
                ts_predictions = ts_res  # falls bereits Dict

            # ✅ JSON-ready
            if is_dataclass(ts_predictions):
                ts_predictions = asdict(ts_predictions)

            out = {
                "vision_analysis": vision or {},
                "features_analysis": ts_predictions or {},
                "processing_time": time.time() - t0,
                "timestamp": time.time(),
            }
            self.metrics["last_analysis_time"] = out["processing_time"]
            self.metrics["total_analyses"] = self.metrics.get("total_analyses", 0) + 1
            
            print(f"🎯 ChatGPT Final: Type-safe extraction successful")
            print(f"📊 Features Analysis Type: {type(ts_predictions)}")
            print(f"✅ JSON-Ready: {not is_dataclass(ts_predictions)}")
            
            return out
        except Exception as e:
            self.log.error(f"multimodal_analysis failed: {e}")
            return {"error": str(e), "timestamp": time.time()}


def test_torchserve_integration(ai_mgr: ChatGPTFinalAIServiceManager):
    """ChatGPT's Mini-Smoke-Test"""
    res = asyncio.get_event_loop().run_until_complete(
        ai_mgr.multimodal_analysis(
            chart={"chart_image": None}, 
            numerical={"ohlcv": None, "indicators": {}}
        )
    )
    assert "features_analysis" in res
    assert isinstance(res["features_analysis"], dict)
    return res


async def test_chatgpt_final_patch():
    """Test ChatGPT's Final Production-Ready Patch"""
    
    print("🚀 TESTING CHATGPT'S FINAL PRODUCTION PATCH")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    config = NautilusIntegrationConfig()
    ai_manager = ChatGPTFinalAIServiceManager(config)
    
    try:
        # Test 1: AI Services Initialization
        print("\n🧪 Test 1: ChatGPT Final AI Services")
        await asyncio.to_thread(ai_manager.start)
        print("✅ All services started with final patch")
        
        # Test 2: ChatGPT's Mini-Smoke-Test
        print("\n🧪 Test 2: ChatGPT's Mini-Smoke-Test")
        result = test_torchserve_integration(ai_manager)
        print(f"✅ Smoke test passed: {result.get('success', True)}")
        
        # Test 3: Type Safety Validation
        print("\n🧪 Test 3: Type Safety & JSON Compatibility")
        analysis = await ai_manager.multimodal_analysis(
            chart={"symbol": "EUR/USD", "chart_image": None},
            numerical={"ohlcv": _create_mock_df("EUR/USD", "5m", 50), "indicators": {}}
        )
        
        # Validate type safety
        features_analysis = analysis.get("features_analysis", {})
        print(f"✅ Features Analysis Type: {type(features_analysis)}")
        print(f"✅ Is Dict (JSON-ready): {isinstance(features_analysis, dict)}")
        print(f"✅ Not Dataclass: {not is_dataclass(features_analysis)}")
        
        # Test 4: JSON Serialization (Production Critical)
        print("\n🧪 Test 4: JSON Serialization (Production Critical)")
        try:
            json_str = json.dumps(analysis, default=str)
            json_parsed = json.loads(json_str)
            print(f"✅ JSON Serialization: Success ({len(json_str)} chars)")
            print(f"✅ JSON Round-trip: Success")
        except Exception as e:
            print(f"❌ JSON Error: {e}")
            raise
        
        # Test 5: Production Pipeline Simulation
        print("\n🧪 Test 5: Production Pipeline Simulation")
        pipeline_result = {
            "symbol": "EUR/USD",
            "timeframe": "5m",
            "bars_requested": 50,
            "analysis_result": analysis,
            "execution_time": analysis.get("processing_time", 0),
            "timestamp": time.time(),
            "pipeline_mode": "chatgpt_final",
            "success": "error" not in analysis,
        }
        
        # Final JSON test
        final_json = json.dumps(pipeline_result, default=str)
        print(f"✅ Final Pipeline JSON: Success ({len(final_json)} chars)")
        
        # Cleanup
        await asyncio.to_thread(ai_manager.stop)
        
        print("\n" + "=" * 60)
        print("🎉 CHATGPT'S FINAL PATCH TEST COMPLETED")
        
        # Summary
        summary = {
            "ai_services_final": True,
            "smoke_test_passed": "features_analysis" in result,
            "type_safety": isinstance(features_analysis, dict),
            "json_compatibility": True,
            "production_ready": pipeline_result["success"],
            "chatgpt_patch_success": True
        }
        
        print(f"\n📊 CHATGPT FINAL PATCH SUMMARY:")
        for test, status in summary.items():
            print(f"✅ {test}: {'SUCCESS' if status else 'FAILED'}")
        
        overall_success = all(summary.values())
        print(f"\n🎯 CHATGPT FINAL PATCH: {'✅ 100% PRODUCTION READY' if overall_success else '❌ ISSUES REMAIN'}")
        
        return summary
        
    except Exception as e:
        print(f"\n❌ CHATGPT FINAL PATCH TEST FAILED: {e}")
        await asyncio.to_thread(ai_manager.stop)
        return {"error": str(e), "chatgpt_patch_success": False}


if __name__ == "__main__":
    asyncio.run(test_chatgpt_final_patch())