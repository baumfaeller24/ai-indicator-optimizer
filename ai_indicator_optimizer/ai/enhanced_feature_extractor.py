#!/usr/bin/env python3
"""
Enhanced Feature Extractor mit Zeitnormierung
Phase 2 Implementation - Core AI Enhancement

Features:
- Zeitnormierung (hour, minute, day_of_week)
- Erweiterte technische Indikatoren
- Market-Regime-Detection
- Volatility-Features
- Pattern-Features Integration
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import logging
import math

# Use flexible Bar type (can be Nautilus Bar or Mock Bar)
from typing import Protocol

class BarProtocol(Protocol):
    """Protocol for Bar objects (Nautilus or Mock)"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    ts_init: int

# Type alias for flexibility  
Bar = BarProtocol


class EnhancedFeatureExtractor:
    """
    Enhanced Feature Extractor für AI Trading System
    
    Phase 2 Core AI Enhancement:
    - Zeitnormierung für bessere ML-Performance
    - Erweiterte technische Indikatoren
    - Market-Regime-Detection
    - Volatility-Features
    - Pattern-Features Integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Enhanced Feature Extractor
        
        Args:
            config: Konfiguration für Feature-Extraktion
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Feature-Extraction-Parameter
        self.include_time_features = self.config.get("include_time_features", True)
        self.include_technical_indicators = self.config.get("include_technical_indicators", True)
        self.include_pattern_features = self.config.get("include_pattern_features", True)
        self.include_volatility_features = self.config.get("include_volatility_features", True)
        
        # Pattern Analyzer für Visual Features
        self.pattern_analyzer = None
        if self.include_pattern_features:
            try:
                from .visual_pattern_analyzer import VisualPatternAnalyzer
                self.pattern_analyzer = VisualPatternAnalyzer(use_mock=True, debug_mode=False)
            except Exception as e:
                self.logger.warning(f"Pattern analyzer not available: {e}")
        
        # Price History für technische Indikatoren
        self.price_history = []
        self.volume_history = []
        self.max_history = self.config.get("max_history", 50)
        
        # Statistics
        self.features_extracted = 0
        
        self.logger.info(f"EnhancedFeatureExtractor initialized: time={self.include_time_features}, tech={self.include_technical_indicators}")
    
    def extract_enhanced_features(self, bar: Bar) -> Dict[str, Any]:
        """
        Extrahiere Enhanced Features aus Bar mit Zeitnormierung
        
        Args:
            bar: Nautilus Bar-Objekt
            
        Returns:
            Dictionary mit Enhanced Features
        """
        try:
            features = {}
            
            # 1. Basis OHLCV Features
            features.update(self._extract_ohlcv_features(bar))
            
            # 2. Zeitnormierung Features
            if self.include_time_features:
                features.update(self._extract_time_features(bar))
            
            # 3. Technische Indikatoren
            if self.include_technical_indicators:
                features.update(self._extract_technical_indicators(bar))
            
            # 4. Volatility Features
            if self.include_volatility_features:
                features.update(self._extract_volatility_features(bar))
            
            # 5. Pattern Features
            if self.include_pattern_features and self.pattern_analyzer:
                features.update(self._extract_pattern_features(bar))
            
            # 6. Market Regime Features
            features.update(self._extract_market_regime_features(bar))
            
            # Update History
            self._update_history(bar)
            
            self.features_extracted += 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting enhanced features: {e}")
            return self._get_fallback_features(bar)
    
    def _extract_ohlcv_features(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere Basis OHLCV Features"""
        open_price = float(bar.open)
        high_price = float(bar.high)
        low_price = float(bar.low)
        close_price = float(bar.close)
        volume = float(bar.volume)
        
        # Berechnete Features
        price_change = close_price - open_price
        price_range = high_price - low_price
        body_size = abs(price_change)
        
        return {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "price_change": price_change,
            "price_change_pct": price_change / max(open_price, 1e-6),
            "price_range": price_range,
            "body_size": body_size,
            "body_ratio": body_size / max(price_range, 1e-6),
            "is_bullish": float(close_price > open_price),
            "is_bearish": float(close_price < open_price),
            "is_doji": float(body_size / max(price_range, 1e-6) < 0.1)
        }
    
    def _extract_time_features(self, bar: Bar) -> Dict[str, float]:
        """
        Phase 2 Enhancement: Zeitnormierung Features
        Extrahiere zeitbasierte Features für bessere ML-Performance
        """
        # Konvertiere Timestamp zu datetime
        dt = datetime.fromtimestamp(bar.ts_init / 1e9, tz=timezone.utc)
        
        # Zeitnormierung Features
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        day_of_month = dt.day
        month = dt.month
        
        # Zyklische Kodierung für bessere ML-Performance
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        minute_sin = math.sin(2 * math.pi * minute / 60)
        minute_cos = math.cos(2 * math.pi * minute / 60)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        
        # Trading Session Features
        is_london_session = float(7 <= hour <= 16)  # London: 8-17 UTC
        is_ny_session = float(13 <= hour <= 22)     # New York: 14-23 UTC
        is_tokyo_session = float(23 <= hour or hour <= 8)  # Tokyo: 0-9 UTC
        is_overlap_london_ny = float(13 <= hour <= 16)  # Overlap
        
        # Market Activity Features
        is_weekend = float(day_of_week >= 5)  # Saturday/Sunday
        is_month_end = float(day_of_month >= 28)
        is_quarter_end = float(month in [3, 6, 9, 12] and day_of_month >= 28)
        
        return {
            # Raw Time Features
            "hour": float(hour),
            "minute": float(minute),
            "day_of_week": float(day_of_week),
            "day_of_month": float(day_of_month),
            "month": float(month),
            
            # Zyklische Features
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "minute_sin": minute_sin,
            "minute_cos": minute_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            
            # Trading Session Features
            "is_london_session": is_london_session,
            "is_ny_session": is_ny_session,
            "is_tokyo_session": is_tokyo_session,
            "is_overlap_london_ny": is_overlap_london_ny,
            
            # Market Activity Features
            "is_weekend": is_weekend,
            "is_month_end": is_month_end,
            "is_quarter_end": is_quarter_end
        }
    
    def _extract_technical_indicators(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere erweiterte technische Indikatoren"""
        features = {}
        
        if len(self.price_history) < 2:
            # Nicht genügend History - Return Defaults
            return {
                "sma_5": float(bar.close),
                "sma_10": float(bar.close),
                "sma_20": float(bar.close),
                "ema_5": float(bar.close),
                "ema_10": float(bar.close),
                "rsi_14": 50.0,
                "bb_upper": float(bar.high),
                "bb_lower": float(bar.low),
                "bb_position": 0.5,
                "atr_14": float(bar.high) - float(bar.low),
                "volume_sma_10": float(bar.volume)
            }
        
        prices = np.array(self.price_history + [float(bar.close)])
        volumes = np.array(self.volume_history + [float(bar.volume)])
        
        # Simple Moving Averages
        if len(prices) >= 5:
            features["sma_5"] = float(np.mean(prices[-5:]))
        if len(prices) >= 10:
            features["sma_10"] = float(np.mean(prices[-10:]))
        if len(prices) >= 20:
            features["sma_20"] = float(np.mean(prices[-20:]))
        
        # Exponential Moving Averages
        if len(prices) >= 5:
            features["ema_5"] = self._calculate_ema(prices, 5)
        if len(prices) >= 10:
            features["ema_10"] = self._calculate_ema(prices, 10)
        
        # RSI
        if len(prices) >= 14:
            features["rsi_14"] = self._calculate_rsi(prices, 14)
        
        # Bollinger Bands
        if len(prices) >= 20:
            bb_features = self._calculate_bollinger_bands(prices, 20, 2.0)
            features.update(bb_features)
        
        # ATR
        if len(self.price_history) >= 14:
            features["atr_14"] = self._calculate_atr(14)
        
        # Volume Features
        if len(volumes) >= 10:
            features["volume_sma_10"] = float(np.mean(volumes[-10:]))
            features["volume_ratio"] = float(bar.volume) / max(features["volume_sma_10"], 1e-6)
        
        return features
    
    def _extract_volatility_features(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere Volatility Features"""
        if len(self.price_history) < 5:
            return {
                "volatility_5": 0.001,
                "volatility_10": 0.001,
                "volatility_ratio": 1.0,
                "price_velocity": 0.0
            }
        
        prices = np.array(self.price_history + [float(bar.close)])
        
        # Rolling Volatility
        vol_5 = float(np.std(prices[-5:]) / np.mean(prices[-5:]) if len(prices) >= 5 else 0.001)
        vol_10 = float(np.std(prices[-10:]) / np.mean(prices[-10:]) if len(prices) >= 10 else 0.001)
        
        # Volatility Ratio
        vol_ratio = vol_5 / max(vol_10, 1e-6)
        
        # Price Velocity (Rate of Change)
        price_velocity = 0.0
        if len(prices) >= 3:
            price_velocity = float((prices[-1] - prices[-3]) / max(prices[-3], 1e-6))
        
        return {
            "volatility_5": vol_5,
            "volatility_10": vol_10,
            "volatility_ratio": vol_ratio,
            "price_velocity": price_velocity
        }
    
    def _extract_pattern_features(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere Pattern Features mit Visual Pattern Analyzer"""
        try:
            patterns = self.pattern_analyzer.analyze_candlestick_patterns([bar])
            
            features = {
                "pattern_count": float(len(patterns)),
                "pattern_confidence_max": 0.0,
                "pattern_bullish": 0.0,
                "pattern_bearish": 0.0,
                "pattern_neutral": 0.0
            }
            
            if patterns:
                # Max Confidence
                features["pattern_confidence_max"] = max(p.confidence for p in patterns)
                
                # Pattern Type Counts
                for pattern in patterns:
                    if pattern.pattern_type == "bullish":
                        features["pattern_bullish"] += 1.0
                    elif pattern.pattern_type == "bearish":
                        features["pattern_bearish"] += 1.0
                    else:
                        features["pattern_neutral"] += 1.0
            
            return features
            
        except Exception as e:
            self.logger.debug(f"Pattern feature extraction failed: {e}")
            return {
                "pattern_count": 0.0,
                "pattern_confidence_max": 0.0,
                "pattern_bullish": 0.0,
                "pattern_bearish": 0.0,
                "pattern_neutral": 0.0
            }
    
    def _extract_market_regime_features(self, bar: Bar) -> Dict[str, float]:
        """Extrahiere Market Regime Features"""
        if len(self.price_history) < 10:
            return {
                "regime_trending": 0.5,
                "regime_ranging": 0.5,
                "regime_volatile": 0.5,
                "trend_strength": 0.0
            }
        
        prices = np.array(self.price_history + [float(bar.close)])
        
        # Trend Strength (basierend auf linearer Regression)
        x = np.arange(len(prices))
        trend_slope = np.polyfit(x, prices, 1)[0]
        trend_strength = abs(trend_slope) / max(np.mean(prices), 1e-6)
        
        # Volatility für Regime-Detection
        volatility = np.std(prices) / max(np.mean(prices), 1e-6)
        
        # Regime Classification (einfache Heuristik)
        if trend_strength > 0.001:
            regime_trending = 1.0
            regime_ranging = 0.0
        else:
            regime_trending = 0.0
            regime_ranging = 1.0
        
        regime_volatile = 1.0 if volatility > 0.01 else 0.0
        
        return {
            "regime_trending": regime_trending,
            "regime_ranging": regime_ranging,
            "regime_volatile": regime_volatile,
            "trend_strength": float(trend_strength)
        }
    
    def _update_history(self, bar: Bar):
        """Update Price/Volume History"""
        self.price_history.append(float(bar.close))
        self.volume_history.append(float(bar.volume))
        
        # Limit History Size
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        if len(self.volume_history) > self.max_history:
            self.volume_history.pop(0)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Berechne Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return float(ema)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Berechne Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_dev: float) -> Dict[str, float]:
        """Berechne Bollinger Bands"""
        if len(prices) < period:
            mid = float(np.mean(prices))
            return {
                "bb_upper": mid * 1.02,
                "bb_lower": mid * 0.98,
                "bb_position": 0.5
            }
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        # Position innerhalb der Bands
        current_price = prices[-1]
        position = (current_price - lower) / max(upper - lower, 1e-6)
        position = max(0.0, min(1.0, position))  # Clamp to [0, 1]
        
        return {
            "bb_upper": float(upper),
            "bb_lower": float(lower),
            "bb_position": float(position)
        }
    
    def _calculate_atr(self, period: int) -> float:
        """Berechne Average True Range"""
        if len(self.price_history) < period:
            return 0.001
        
        # Vereinfachte ATR-Berechnung (nur Close-basiert)
        prices = np.array(self.price_history[-period:])
        true_ranges = np.abs(np.diff(prices))
        return float(np.mean(true_ranges))
    
    def _get_fallback_features(self, bar: Bar) -> Dict[str, float]:
        """Fallback Features bei Fehlern"""
        return {
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "price_change": float(bar.close) - float(bar.open),
            "is_bullish": float(float(bar.close) > float(bar.open))
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Feature Extractor Statistiken"""
        return {
            "features_extracted": self.features_extracted,
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history),
            "include_time_features": self.include_time_features,
            "include_technical_indicators": self.include_technical_indicators,
            "include_pattern_features": self.include_pattern_features,
            "include_volatility_features": self.include_volatility_features,
            "pattern_analyzer_available": self.pattern_analyzer is not None
        }


# Factory Function
def create_enhanced_feature_extractor(config: Optional[Dict] = None) -> EnhancedFeatureExtractor:
    """
    Factory Function für Enhanced Feature Extractor
    
    Args:
        config: Konfiguration für Feature-Extraktion
    
    Returns:
        EnhancedFeatureExtractor Instance
    """
    return EnhancedFeatureExtractor(config=config)