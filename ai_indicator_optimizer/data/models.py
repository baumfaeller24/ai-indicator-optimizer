"""
Data Models für Forex-Daten und Marktinformationen
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TickData:
    """Tick-Level Forex Daten"""
    timestamp: datetime
    bid: float
    ask: float
    volume: float
    
    @property
    def mid_price(self) -> float:
        """Mittlerer Preis zwischen Bid und Ask"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-Ask Spread"""
        return self.ask - self.bid


@dataclass
class OHLCVData:
    """OHLCV Candlestick Daten"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @property
    def body_size(self) -> float:
        """Größe des Candlestick Body"""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Oberer Schatten"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Unterer Schatten"""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Bullish Candlestick"""
        return self.close > self.open


@dataclass
class MarketData:
    """Kombinierte Marktdaten"""
    symbol: str
    timeframe: str
    ohlcv_data: List[OHLCVData]
    tick_data: Optional[List[TickData]] = None
    
    def to_numpy(self) -> np.ndarray:
        """Konvertiert zu NumPy Array für ML"""
        data = []
        for candle in self.ohlcv_data:
            data.append([
                candle.timestamp.timestamp(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume
            ])
        return np.array(data)


@dataclass
class IndicatorData:
    """Technische Indikatoren"""
    rsi: Optional[List[float]] = None
    macd: Optional[Dict[str, List[float]]] = None
    bollinger: Optional[Dict[str, List[float]]] = None
    sma: Optional[Dict[int, List[float]]] = None
    ema: Optional[Dict[int, List[float]]] = None
    stochastic: Optional[Dict[str, List[float]]] = None
    atr: Optional[List[float]] = None
    adx: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert zu Dictionary"""
        return {
            'rsi': self.rsi,
            'macd': self.macd,
            'bollinger': self.bollinger,
            'sma': self.sma,
            'ema': self.ema,
            'stochastic': self.stochastic,
            'atr': self.atr,
            'adx': self.adx
        }