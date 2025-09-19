"""
Enhanced AI Pattern Strategy - Integriert ChatGPT-Verbesserungen
Basiert auf den Hinweisen aus "ai pattern strategy.md" und "tradingbeispiele.md"
"""
import os
import json
import requests
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
from collections import deque
import polars as pl

from nautilus_trader.strategy.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.orders import MarketOrder


class FeaturePredictionLogger:
    """
    Feature & Prediction Logger basierend auf ChatGPT-Vorschlag
    Speichert alle AI-Features und Predictions für spätere Analyse
    """
    
    def __init__(self, out_parquet="logs/ai_features.parquet", buffer_size=1000):
        self.out = out_parquet
        self.buf: List[Dict] = []
        self.n = 0
        self.buffer_size = buffer_size
        
        # Erstelle logs Verzeichnis falls nicht vorhanden
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)

    def log(self, *, ts_ns: int, instrument: str, features: Dict, prediction: Dict):
        """Logge Features und Prediction"""
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
        """Schreibe Buffer zu Parquet"""
        if not self.buf:
            return
        
        try:
            df = pl.DataFrame(self.buf)
            # Append-Mode für kontinuierliches Logging
            if os.path.exists(self.out):
                existing_df = pl.read_parquet(self.out)
                df = pl.concat([existing_df, df])
            
            df.write_parquet(self.out, compression="zstd")
            self.buf.clear()
        except Exception as e:
            print(f"Warning: Could not write feature log: {e}")

    def close(self):
        """Schließe Logger und schreibe finalen Buffer"""
        self.flush()


class BarDatasetBuilder:
    """
    Dataset Builder basierend auf ChatGPT-Vorschlag
    Sammelt Features und erstellt Labels für ML-Training
    """
    
    def __init__(self, horizon=5, min_bars=10):
        self.rows: List[Dict] = []
        self.buffer = deque(maxlen=horizon+1)
        self.horizon = horizon
        self.min_bars = min_bars

    def on_bar(self, bar: Bar):
        """Verarbeite Bar und erstelle Features mit Forward-Return Labels"""
        feat = {
            "ts": int(bar.ts_init),
            "instrument": str(bar.bar_type.instrument_id),
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low), 
            "close": float(bar.close),
            "volume": float(bar.volume),
        }
        
        # Technische Features
        feat["price_change"] = feat["close"] - feat["open"]
        rng = feat["high"] - feat["low"]
        feat["body_ratio"] = abs(feat["price_change"]) / max(rng, 1e-6)
        
        self.buffer.append(feat)
        
        # Erstelle Label wenn Buffer voll
        if len(self.buffer) == self.buffer.maxlen:
            x = self.buffer[0]
            y_close = self.buffer[-1]["close"]
            fwd_ret = (y_close / x["close"]) - 1.0  # Forward Return
            
            x[f"label_fwd_ret@{self.horizon}"] = fwd_ret
            # Diskrete Klassen: buy/sell/hold
            x["label_class"] = 0 if fwd_ret > 0.0003 else (1 if fwd_ret < -0.0003 else 2)
            
            self.rows.append(x.copy())

    def to_parquet(self, path: str):
        """Exportiere Dataset zu Parquet"""
        if len(self.rows) < self.min_bars:
            print(f"Not enough bars ({len(self.rows)}) to write dataset.")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pl.DataFrame(self.rows).write_parquet(path)
        print(f"✅ Dataset exported: {path} ({len(self.rows)} samples)")


class EnhancedAIPatternStrategy(Strategy):
    """
    Enhanced AI Pattern Strategy mit ChatGPT-Verbesserungen:
    - Erweiterte Konfigurierbarkeit (ENV-Support)
    - Feature Logging für ML-Training
    - Dataset Builder für Forward-Return Labels
    - Erweiterte technische Indikatoren
    - Live-Control via Pause-Mechanismus
    - Verbessertes Confidence Handling
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Erweiterte Konfiguration (ChatGPT-Verbesserung #1)
        self.ai_endpoint = config.get("ai_endpoint", 
                                    os.getenv("AI_ENDPOINT", "http://localhost:8080/predictions/pattern_model"))
        self.min_confidence = config.get("min_confidence", 0.7)
        self.position_size = config.get("position_size", 1000)
        self.use_mock = config.get("use_mock", True)
        self.debug_mode = config.get("debug_mode", False)
        self.request_timeout = config.get("request_timeout", 2.0)
        
        # Live-Control (ChatGPT-Verbesserung #4)
        self.paused = config.get("paused", False)
        
        # Feature Logging (ChatGPT-Verbesserung #3)
        self.enable_logging = config.get("enable_logging", True)
        self.feature_logger = None
        
        # Dataset Building für Training
        self.enable_dataset_building = config.get("enable_dataset_building", True)
        self.dataset_builder = None
        
        # Trading-Parameter
        self.max_positions = config.get("max_positions", 1)
        self.risk_per_trade = config.get("risk_per_trade", 0.02)
        
        # Performance-Tracking
        self.predictions_count = 0
        self.successful_predictions = 0
        
        # Erweiterte Features (ChatGPT-Verbesserung #2)
        self.use_extended_features = config.get("use_extended_features", True)
        self.rsi_period = config.get("rsi_period", 14)
        
        # RSI Calculation Buffer
        self.price_buffer = deque(maxlen=self.rsi_period + 1)
        
    def on_start(self):
        """Strategy startup mit erweiterten Features"""
        self.log.info("🚀 Enhanced AI Pattern Strategy started")
        self.log.info(f"📡 AI Endpoint: {self.ai_endpoint}")
        self.log.info(f"🎯 Min Confidence: {self.min_confidence}")
        self.log.info(f"🔧 Mock Mode: {self.use_mock}")
        self.log.info(f"📊 Extended Features: {self.use_extended_features}")
        self.log.info(f"📝 Feature Logging: {self.enable_logging}")
        
        # Initialisiere Feature Logger
        if self.enable_logging:
            log_path = f"logs/ai_features_{datetime.now().strftime('%Y%m%d')}.parquet"
            self.feature_logger = FeaturePredictionLogger(log_path)
            self.log.info(f"📝 Feature Logger initialized: {log_path}")
        
        # Initialisiere Dataset Builder
        if self.enable_dataset_building:
            self.dataset_builder = BarDatasetBuilder(horizon=5)
            self.log.info("📊 Dataset Builder initialized")
        
    def on_stop(self):
        """Strategy shutdown mit Cleanup"""
        accuracy = (self.successful_predictions / max(self.predictions_count, 1)) * 100
        self.log.info(f"📊 AI Strategy Performance: {accuracy:.1f}% accuracy ({self.successful_predictions}/{self.predictions_count})")
        
        # Cleanup Logger
        if self.feature_logger:
            self.feature_logger.close()
            self.log.info("📝 Feature Logger closed")
        
        # Export Dataset
        if self.dataset_builder:
            dataset_path = f"datasets/training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            self.dataset_builder.to_parquet(dataset_path)
            self.log.info(f"📊 Training dataset exported: {dataset_path}")
        
    def on_bar(self, bar: Bar):
        """Hauptlogik mit Pause-Support"""
        try:
            # Live-Control: Pause-Check (ChatGPT-Verbesserung #4)
            if self.paused:
                if self.debug_mode:
                    self.log.debug("⏸️ Strategy paused via command channel")
                return
            
            # Dataset Building
            if self.dataset_builder:
                self.dataset_builder.on_bar(bar)
            
            # Features extrahieren (erweitert)
            features = self._extract_enhanced_features(bar)
            
            # AI-Prediction abrufen
            prediction = self._get_ai_prediction(features)
            
            # Feature Logging
            if self.feature_logger and prediction:
                self.feature_logger.log(
                    ts_ns=int(bar.ts_init),
                    instrument=str(bar.bar_type.instrument_id),
                    features=features,
                    prediction=prediction
                )
            
            # Verbessertes Confidence Handling (ChatGPT-Verbesserung #5)
            if prediction and self._should_execute_signal(prediction):
                self._execute_signal(prediction, bar)
                
        except Exception as e:
            self.log.error(f"⚠️ AI analysis failed: {e}")
    
    def _extract_enhanced_features(self, bar: Bar) -> Dict:
        """Erweiterte Feature-Extraktion (ChatGPT-Verbesserung #2)"""
        
        # Basis-Features
        features = {
            # OHLCV-Daten
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            
            # Zeitstempel für Kontext
            "timestamp": bar.ts_init,
            "instrument": str(bar.bar_type.instrument_id),
            
            # Basis technische Indikatoren
            "price_change": float(bar.close - bar.open),
            "price_range": float(bar.high - bar.low),
            "body_ratio": abs(float(bar.close - bar.open)) / max(float(bar.high - bar.low), 0.0001),
        }
        
        # Erweiterte Features (ChatGPT-Verbesserung #2)
        if self.use_extended_features:
            # Zeitnormierung
            dt = datetime.fromtimestamp(bar.ts_init / 1e9)
            features.update({
                "hour": dt.hour,
                "minute": dt.minute,
                "day_of_week": dt.weekday(),
            })
            
            # RSI Berechnung
            self.price_buffer.append(float(bar.close))
            if len(self.price_buffer) >= self.rsi_period:
                features["rsi_14"] = self._calculate_rsi()
            
            # Zusätzliche technische Features
            features.update({
                "upper_shadow": float(bar.high - max(bar.open, bar.close)),
                "lower_shadow": float(min(bar.open, bar.close) - bar.low),
                "is_green": float(bar.close > bar.open),
                "volume_ratio": float(bar.volume) / max(float(bar.volume), 1.0),  # Normalisiert
            })
        
        return features
    
    def _calculate_rsi(self) -> float:
        """Einfache RSI-Berechnung"""
        if len(self.price_buffer) < self.rsi_period:
            return 50.0  # Neutral
        
        prices = list(self.price_buffer)
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]
        
        avg_gain = sum(gains) / len(deltas) if gains else 0
        avg_loss = sum(losses) / len(deltas) if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _should_execute_signal(self, prediction: Dict) -> bool:
        """Verbessertes Confidence Handling (ChatGPT-Verbesserung #5)"""
        confidence = prediction.get("confidence", 0.0)
        risk_score = prediction.get("risk_score", 0.0)
        
        # Gewichteter Score: Confidence minus Risk
        weighted_score = confidence * (1 - risk_score)
        
        # Dynamischer Threshold basierend auf Market Conditions
        dynamic_threshold = self.min_confidence
        
        # Erhöhe Threshold bei hohem Risk
        if risk_score > 0.5:
            dynamic_threshold += 0.1
        
        return weighted_score > dynamic_threshold
    
    def _get_ai_prediction(self, features: Dict) -> Optional[Dict]:
        """AI-Prediction mit verbessertem Error Handling"""
        
        if self.use_mock:
            return self._get_enhanced_mock_prediction(features)
        
        try:
            response = requests.post(
                self.ai_endpoint, 
                json=features,
                timeout=self.request_timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            prediction = response.json()
            self.predictions_count += 1
            
            if self.debug_mode:
                self.log.debug(f"🤖 AI Prediction: {prediction}")
            
            return prediction
            
        except requests.exceptions.Timeout:
            self.log.warning("⏰ AI prediction timeout - using fallback")
            return None
        except requests.exceptions.RequestException as e:
            self.log.error(f"❌ AI prediction request failed: {e}")
            return None
        except Exception as e:
            self.log.error(f"❌ AI prediction failed: {e}")
            return None
    
    def _get_enhanced_mock_prediction(self, features: Dict) -> Dict:
        """Erweiterte Mock-Prediction mit mehr Features"""
        
        price_change = features.get("price_change", 0)
        body_ratio = features.get("body_ratio", 0)
        rsi = features.get("rsi_14", 50)
        
        # Erweiterte Mock-Logik
        if price_change > 0 and body_ratio > 0.7 and rsi < 70:
            action = "BUY"
            confidence = min(0.75 + (body_ratio - 0.7) * 0.5, 0.95)
            reasoning = f"Strong bullish candle (RSI: {rsi:.1f})"
            risk_score = 0.2
        elif price_change < 0 and body_ratio > 0.7 and rsi > 30:
            action = "SELL" 
            confidence = min(0.75 + (body_ratio - 0.7) * 0.5, 0.95)
            reasoning = f"Strong bearish candle (RSI: {rsi:.1f})"
            risk_score = 0.2
        else:
            action = "HOLD"
            confidence = 0.5
            reasoning = f"No clear pattern (RSI: {rsi:.1f})"
            risk_score = 0.1
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "pattern_type": "enhanced_mock_pattern",
            "risk_score": risk_score,
            "features_used": len(features)
        }
    
    def _execute_signal(self, prediction: Dict, bar: Bar):
        """Signal-Ausführung mit erweiterten Logs"""
        
        action = prediction.get("action", "HOLD")
        confidence = prediction.get("confidence", 0.0)
        reasoning = prediction.get("reasoning", "N/A")
        risk_score = prediction.get("risk_score", 0.0)
        
        # Prüfe Positions-Limit
        if len(self.portfolio.positions_open()) >= self.max_positions:
            if self.debug_mode:
                self.log.info(f"⏸️ Max positions reached, skipping {action} signal")
            return
        
        # Führe Trading-Action aus
        if action == "BUY":
            self._submit_market_order(OrderSide.BUY, bar)
        elif action == "SELL":
            self._submit_market_order(OrderSide.SELL, bar)
        
        # Erweiterte Logs
        self.log.info(
            f"[AI] 🎯 Action: {action} | "
            f"📊 Confidence: {confidence:.2f} | "
            f"⚠️ Risk: {risk_score:.2f} | "
            f"💭 Reason: {reasoning}"
        )
    
    def _submit_market_order(self, side: OrderSide, bar: Bar):
        """Market Order mit Error Handling"""
        try:
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=bar.bar_type.instrument_id,
                order_side=side,
                quantity=self.instrument.make_qty(self.position_size),
                time_in_force=self.time_in_force,
                order_id=self.generate_order_id(),
                ts_init=self.clock.timestamp_ns(),
            )
            
            self.submit_order(order)
            
            if self.debug_mode:
                self.log.info(f"🟢 Submitted {side.name} order for {self.position_size} units")
            
        except Exception as e:
            self.log.error(f"❌ Order submission failed: {e}")
    
    def on_position_closed(self, position):
        """Position-Tracking für Performance-Messung"""
        pnl = position.realized_pnl
        if pnl and pnl.as_double() > 0:
            self.successful_predictions += 1
            
        self.log.info(f"📉 Position closed: {position.instrument_id} PnL: {pnl}")
    
    def pause_strategy(self):
        """Pausiere Strategie (Live-Control)"""
        self.paused = True
        self.log.info("⏸️ Strategy paused")
    
    def resume_strategy(self):
        """Setze Strategie fort (Live-Control)"""
        self.paused = False
        self.log.info("▶️ Strategy resumed")
    
    def reset(self):
        """Reset Strategy State"""
        super().reset()
        self.predictions_count = 0
        self.successful_predictions = 0
        self.price_buffer.clear()
        
        if self.feature_logger:
            self.feature_logger.close()
        if self.dataset_builder:
            self.dataset_builder = BarDatasetBuilder(horizon=5)