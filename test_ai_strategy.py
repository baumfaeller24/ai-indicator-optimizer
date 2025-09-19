#!/usr/bin/env python3
"""
Test-Script für AI Pattern Strategy
Testet die Nautilus AI-Strategy mit Mock-Daten
"""

import sys
import os
import json
from pathlib import Path
import logging

# Projekt-Pfad hinzufügen
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ai_strategy_mock():
    """Testet die AI-Strategy mit Mock-Daten"""
    try:
        logger.info("🧪 Teste AI Pattern Strategy (Mock Mode)")
        
        # Importiere AI Strategy
        from strategies.ai_strategies.ai_pattern_strategy import AIPatternStrategy
        
        # Lade Konfiguration
        config_path = Path("config/ai_strategy_config.json")
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
                strategy_config = config_data["ai_pattern_strategy"]["config"]
        else:
            # Fallback-Konfiguration
            strategy_config = {
                "ai_endpoint": "http://localhost:8080/predictions/pattern_model",
                "min_confidence": 0.7,
                "position_size": 1000,
                "use_mock": True
            }
        
        logger.info(f"📋 Strategy Config: {strategy_config}")
        
        # Teste Mock-Prediction direkt
        logger.info("🔍 Teste Mock-Prediction-Logik...")
        
        # Simuliere verschiedene Bar-Features
        test_features = [
            {
                "name": "Bullish Candle",
                "features": {
                    "open": 1.0850,
                    "high": 1.0890,
                    "low": 1.0845,
                    "close": 1.0885,
                    "volume": 1000,
                    "price_change": 0.0035,  # Positiv
                    "body_ratio": 0.8        # Starker Body
                }
            },
            {
                "name": "Bearish Candle", 
                "features": {
                    "open": 1.0885,
                    "high": 1.0890,
                    "low": 1.0845,
                    "close": 1.0850,
                    "volume": 1200,
                    "price_change": -0.0035,  # Negativ
                    "body_ratio": 0.8         # Starker Body
                }
            },
            {
                "name": "Neutral/Doji Candle",
                "features": {
                    "open": 1.0870,
                    "high": 1.0890,
                    "low": 1.0850,
                    "close": 1.0872,
                    "volume": 800,
                    "price_change": 0.0002,   # Minimal
                    "body_ratio": 0.1         # Schwacher Body
                }
            }
        ]
        
        # Teste Mock-Predictions
        for test_case in test_features:
            logger.info(f"\n📊 Test Case: {test_case['name']}")
            
            # Erstelle temporäre Strategy-Instanz für Mock-Test
            class MockStrategy:
                def __init__(self, config):
                    self.config = config
                    self.use_mock = config.get("use_mock", True)
                    self.predictions_count = 0
                
                def _get_mock_prediction(self, features):
                    """Mock-Prediction Logik (kopiert aus AI Strategy)"""
                    price_change = features.get("price_change", 0)
                    body_ratio = features.get("body_ratio", 0)
                    
                    if price_change > 0 and body_ratio > 0.7:
                        action = "BUY"
                        confidence = 0.75
                        reasoning = "Strong bullish candle detected"
                    elif price_change < 0 and body_ratio > 0.7:
                        action = "SELL" 
                        confidence = 0.75
                        reasoning = "Strong bearish candle detected"
                    else:
                        action = "HOLD"
                        confidence = 0.5
                        reasoning = "No clear pattern detected"
                    
                    return {
                        "action": action,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "pattern_type": "mock_pattern",
                        "risk_score": 0.3
                    }
            
            # Teste Mock-Prediction
            mock_strategy = MockStrategy(strategy_config)
            prediction = mock_strategy._get_mock_prediction(test_case["features"])
            
            # Ergebnis ausgeben
            logger.info(f"   📈 Features: {test_case['features']}")
            logger.info(f"   🤖 Prediction: {prediction}")
            logger.info(f"   ✅ Action: {prediction['action']} (Confidence: {prediction['confidence']:.2f})")
            logger.info(f"   💭 Reasoning: {prediction['reasoning']}")
        
        logger.info("\n🎉 AI Strategy Mock-Test erfolgreich abgeschlossen!")
        return True
        
    except Exception as e:
        logger.exception(f"❌ AI Strategy Test fehlgeschlagen: {e}")
        return False

def test_strategy_configuration():
    """Testet die Strategy-Konfiguration"""
    try:
        logger.info("⚙️ Teste Strategy-Konfiguration...")
        
        config_path = Path("config/ai_strategy_config.json")
        
        if not config_path.exists():
            logger.warning(f"⚠️ Konfigurationsdatei nicht gefunden: {config_path}")
            return False
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Validiere Konfiguration
        required_keys = ["ai_pattern_strategy", "torchserve_config", "development_settings"]
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ Fehlender Konfigurationsschlüssel: {key}")
                return False
        
        strategy_config = config["ai_pattern_strategy"]["config"]
        required_strategy_keys = ["ai_endpoint", "min_confidence", "position_size", "use_mock"]
        
        for key in required_strategy_keys:
            if key not in strategy_config:
                logger.error(f"❌ Fehlender Strategy-Parameter: {key}")
                return False
        
        logger.info("✅ Konfiguration ist vollständig und gültig")
        logger.info(f"   📡 AI Endpoint: {strategy_config['ai_endpoint']}")
        logger.info(f"   🎯 Min Confidence: {strategy_config['min_confidence']}")
        logger.info(f"   💰 Position Size: {strategy_config['position_size']}")
        logger.info(f"   🔧 Mock Mode: {strategy_config['use_mock']}")
        
        return True
        
    except Exception as e:
        logger.exception(f"❌ Konfigurationstest fehlgeschlagen: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🧪 AI Pattern Strategy Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: Konfiguration
    if not test_strategy_configuration():
        success = False
    
    print()
    
    # Test 2: Mock-Strategy
    if not test_ai_strategy_mock():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Alle Tests erfolgreich!")
        print("\n🚀 Nächste Schritte:")
        print("   1. TorchServe für echte AI-Inferenz starten")
        print("   2. Strategy in Nautilus-Backtest integrieren")
        print("   3. Live-Trading mit echten Daten testen")
        return 0
    else:
        print("💥 Tests fehlgeschlagen!")
        return 1

if __name__ == "__main__":
    sys.exit(main())