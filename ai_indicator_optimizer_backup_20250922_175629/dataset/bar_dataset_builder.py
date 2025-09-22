#!/usr/bin/env python3
"""
Enhanced Bar Dataset Builder
ChatGPT-Verbesserung: Automatische Forward-Return-Label-Generierung
Basierend auf tradingbeispiele.md
"""

import polars as pl
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
import numpy as np
from pathlib import Path
import logging

from nautilus_trader.model.data import Bar


class BarDatasetBuilder:
    """
    ChatGPT Enhancement: Automatisches Dataset-Building mit Forward-Return-Labeling
    
    Features:
    - Forward-Return-Label-Generierung für verschiedene Horizonte
    - OHLCV-Feature-Extraktion mit technischen Indikatoren
    - Diskrete Klassen-Labels (BUY/SELL/HOLD)
    - Polars-basierte Performance-Optimierung
    - Parquet-Export mit Kompression
    """
    
    def __init__(
        self, 
        horizon: int = 5, 
        min_bars: int = 10,
        return_thresholds: Dict[str, float] = None,
        include_technical_indicators: bool = True
    ):
        """
        Initialize Bar Dataset Builder
        
        Args:
            horizon: Forward-Return-Horizont in Bars
            min_bars: Minimum Bars vor Export
            return_thresholds: Schwellenwerte für BUY/SELL/HOLD Klassifikation
            include_technical_indicators: Ob technische Indikatoren berechnet werden sollen
        """
        self.horizon = horizon
        self.min_bars = min_bars
        self.include_technical_indicators = include_technical_indicators
        
        # Default Return Thresholds für Klassifikation
        self.return_thresholds = return_thresholds or {
            "buy_threshold": 0.0003,    # 0.03% für BUY
            "sell_threshold": -0.0003   # -0.03% für SELL
        }
        
        # Buffer für Forward-Return-Berechnung
        self.buffer = deque(maxlen=horizon + 1)
        self.rows: List[Dict[str, Any]] = []
        
        # Technische Indikatoren Buffer
        self.price_history = deque(maxlen=50)  # Für Moving Averages etc.
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BarDatasetBuilder initialized: horizon={horizon}, min_bars={min_bars}")
    
    def on_bar(self, bar: Bar) -> None:
        """
        Process Bar und generiere Features + Forward-Return-Labels
        
        Args:
            bar: Nautilus Bar-Objekt
        """
        try:
            # Extrahiere Basis-Features aus Bar
            features = self._extract_bar_features(bar)
            
            # Füge technische Indikatoren hinzu (falls aktiviert)
            if self.include_technical_indicators:
                tech_features = self._calculate_technical_indicators(bar)
                features.update(tech_features)
            
            # Füge zu Buffer hinzu
            self.buffer.append(features)
            self.price_history.append(float(bar.close))
            
            # Forward-Return-Label-Generierung wenn Buffer voll
            if len(self.buffer) == self.buffer.maxlen:
                labeled_entry = self._create_labeled_entry()
                if labeled_entry:
                    self.rows.append(labeled_entry)
                    
                    if len(self.rows) % 1000 == 0:
                        self.logger.info(f"Processed {len(self.rows)} labeled entries")
        
        except Exception as e:
            self.logger.error(f"Error processing bar: {e}")
    
    def _extract_bar_features(self, bar: Bar) -> Dict[str, Any]:
        """
        Extrahiere Basis-Features aus Bar
        
        Args:
            bar: Nautilus Bar
            
        Returns:
            Dict mit extrahierten Features
        """
        # OHLCV-Daten
        open_price = float(bar.open)
        high_price = float(bar.high)
        low_price = float(bar.low)
        close_price = float(bar.close)
        volume = float(bar.volume)
        
        # Berechnete Features
        price_change = close_price - open_price
        price_range = high_price - low_price
        body_ratio = abs(price_change) / max(price_range, 1e-6)
        
        # Upper/Lower Shadows
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        return {
            # Timestamp & Instrument
            "ts_ns": int(bar.ts_init),
            "timestamp": datetime.utcfromtimestamp(bar.ts_init / 1e9).isoformat(),
            "instrument": str(bar.bar_type.instrument_id),
            
            # OHLCV
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            
            # Berechnete Features
            "price_change": price_change,
            "price_change_pct": price_change / max(open_price, 1e-6),
            "price_range": price_range,
            "body_ratio": body_ratio,
            "upper_shadow": upper_shadow,
            "lower_shadow": lower_shadow,
            "upper_shadow_ratio": upper_shadow / max(price_range, 1e-6),
            "lower_shadow_ratio": lower_shadow / max(price_range, 1e-6),
            
            # Candlestick Pattern Features
            "is_doji": body_ratio < 0.1,
            "is_hammer": (lower_shadow > 2 * abs(price_change)) and (upper_shadow < abs(price_change)),
            "is_shooting_star": (upper_shadow > 2 * abs(price_change)) and (lower_shadow < abs(price_change)),
            "is_bullish": price_change > 0,
            "is_bearish": price_change < 0,
        }
    
    def _calculate_technical_indicators(self, bar: Bar) -> Dict[str, Any]:
        """
        Berechne technische Indikatoren
        
        Args:
            bar: Nautilus Bar
            
        Returns:
            Dict mit technischen Indikatoren
        """
        indicators = {}
        
        if len(self.price_history) < 2:
            return indicators
        
        prices = np.array(list(self.price_history))
        current_price = float(bar.close)
        
        try:
            # Simple Moving Averages
            if len(prices) >= 5:
                indicators["sma_5"] = np.mean(prices[-5:])
            if len(prices) >= 10:
                indicators["sma_10"] = np.mean(prices[-10:])
            if len(prices) >= 20:
                indicators["sma_20"] = np.mean(prices[-20:])
            
            # Exponential Moving Averages (vereinfacht)
            if len(prices) >= 12:
                indicators["ema_12"] = self._calculate_ema(prices, 12)
            if len(prices) >= 26:
                indicators["ema_26"] = self._calculate_ema(prices, 26)
            
            # RSI (vereinfacht)
            if len(prices) >= 14:
                indicators["rsi_14"] = self._calculate_rsi(prices, 14)
            
            # Volatilität (Standard Deviation)
            if len(prices) >= 10:
                indicators["volatility_10"] = np.std(prices[-10:])
            
            # Price Position in Range
            if len(prices) >= 20:
                recent_high = np.max(prices[-20:])
                recent_low = np.min(prices[-20:])
                if recent_high > recent_low:
                    indicators["price_position"] = (current_price - recent_low) / (recent_high - recent_low)
                else:
                    indicators["price_position"] = 0.5
            
            # Momentum
            if len(prices) >= 5:
                indicators["momentum_5"] = (current_price - prices[-5]) / prices[-5]
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Vereinfachte EMA-Berechnung"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Vereinfachte RSI-Berechnung"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_labeled_entry(self) -> Optional[Dict[str, Any]]:
        """
        Erstelle gelabelten Eintrag mit Forward-Return
        
        Returns:
            Dict mit Features und Labels oder None
        """
        if len(self.buffer) < self.buffer.maxlen:
            return None
        
        try:
            # Aktueller Eintrag (erstes Element im Buffer)
            current_entry = self.buffer[0].copy()
            
            # Future Price (letztes Element im Buffer)
            future_entry = self.buffer[-1]
            
            current_price = current_entry["close"]
            future_price = future_entry["close"]
            
            # Forward Return berechnen
            forward_return = (future_price / current_price) - 1.0
            
            # Diskrete Klassen-Labels
            if forward_return > self.return_thresholds["buy_threshold"]:
                label_class = 0  # BUY
                label_name = "BUY"
            elif forward_return < self.return_thresholds["sell_threshold"]:
                label_class = 1  # SELL
                label_name = "SELL"
            else:
                label_class = 2  # HOLD
                label_name = "HOLD"
            
            # Confidence basierend auf Return-Magnitude
            confidence = min(abs(forward_return) / 0.001, 1.0)  # Normalisiert auf 0.1% Return
            
            # Füge Labels hinzu
            current_entry.update({
                f"label_fwd_ret_h{self.horizon}": forward_return,
                f"label_class_h{self.horizon}": label_class,
                f"label_name_h{self.horizon}": label_name,
                f"label_confidence_h{self.horizon}": confidence,
                "future_price": future_price,
                "horizon": self.horizon
            })
            
            return current_entry
            
        except Exception as e:
            self.logger.error(f"Error creating labeled entry: {e}")
            return None
    
    def to_parquet(
        self, 
        path: str, 
        compression: str = "zstd",
        include_metadata: bool = True
    ) -> bool:
        """
        Export Dataset zu Parquet-Format
        
        Args:
            path: Output-Pfad
            compression: Kompression (zstd, snappy, gzip)
            include_metadata: Ob Metadata hinzugefügt werden soll
            
        Returns:
            bool: True wenn erfolgreich
        """
        if len(self.rows) < self.min_bars:
            self.logger.warning(f"Not enough bars ({len(self.rows)}) to write dataset. Minimum: {self.min_bars}")
            return False
        
        try:
            # Erstelle Polars DataFrame
            df = pl.DataFrame(self.rows)
            
            # Sortiere nach Timestamp
            df = df.sort("ts_ns")
            
            # Füge Metadata hinzu
            if include_metadata:
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "total_entries": len(df),
                    "horizon": self.horizon,
                    "return_thresholds": self.return_thresholds,
                    "technical_indicators": self.include_technical_indicators,
                    "instruments": df["instrument"].unique().to_list(),
                    "date_range": {
                        "start": df["timestamp"].min(),
                        "end": df["timestamp"].max()
                    }
                }
                
                # Schreibe Metadata als separate JSON
                metadata_path = Path(path).with_suffix('.json')
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Erstelle Output-Verzeichnis
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Schreibe Parquet
            df.write_parquet(path, compression=compression)
            
            self.logger.info(f"Successfully exported {len(df)} entries to {path}")
            self.logger.info(f"Compression: {compression}, File size: {Path(path).stat().st_size} bytes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to Parquet: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Erhalte Dataset-Statistiken
        
        Returns:
            Dict mit Statistiken
        """
        if not self.rows:
            return {"total_entries": 0, "buffer_size": len(self.buffer)}
        
        df = pl.DataFrame(self.rows)
        
        # Label-Verteilung
        label_col = f"label_name_h{self.horizon}"
        if label_col in df.columns:
            label_distribution = df[label_col].value_counts().to_dict()
        else:
            label_distribution = {}
        
        # Return-Statistiken
        return_col = f"label_fwd_ret_h{self.horizon}"
        if return_col in df.columns:
            returns = df[return_col]
            return_stats = {
                "mean": returns.mean(),
                "std": returns.std(),
                "min": returns.min(),
                "max": returns.max(),
                "positive_returns": (returns > 0).sum(),
                "negative_returns": (returns < 0).sum()
            }
        else:
            return_stats = {}
        
        return {
            "total_entries": len(self.rows),
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer.maxlen,
            "horizon": self.horizon,
            "label_distribution": label_distribution,
            "return_statistics": return_stats,
            "instruments": df["instrument"].unique().to_list() if "instrument" in df.columns else [],
            "technical_indicators_enabled": self.include_technical_indicators
        }
    
    def reset(self) -> None:
        """Reset Dataset Builder"""
        self.rows.clear()
        self.buffer.clear()
        self.price_history.clear()
        self.logger.info("BarDatasetBuilder reset")


if __name__ == "__main__":
    # Test des Bar Dataset Builders
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.enums import BarType
    from nautilus_trader.model.data import Bar
    import time
    
    print("🧪 Testing BarDatasetBuilder...")
    
    # Erstelle Test-Builder
    builder = BarDatasetBuilder(horizon=3, min_bars=5)
    
    # Generiere Test-Bars
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    
    for i in range(10):
        # Simuliere Preis-Bewegung
        base_price = 1.1000
        price_change = (i - 5) * 0.0001
        
        bar = Bar(
            bar_type=BarType.from_str("EUR/USD.SIM-1-MINUTE-BID-EXTERNAL"),
            open=base_price + price_change,
            high=base_price + price_change + 0.0002,
            low=base_price + price_change - 0.0001,
            close=base_price + price_change + 0.0001,
            volume=1000 + i * 100,
            ts_event=int(time.time() * 1e9) + i * 60 * 1e9,
            ts_init=int(time.time() * 1e9) + i * 60 * 1e9
        )
        
        builder.on_bar(bar)
    
    # Zeige Statistiken
    stats = builder.get_stats()
    print(f"📊 Stats: {stats}")
    
    # Export Test
    if builder.to_parquet("test_logs/test_dataset.parquet"):
        print("✅ BarDatasetBuilder Test erfolgreich!")
    else:
        print("❌ BarDatasetBuilder Test fehlgeschlagen!")