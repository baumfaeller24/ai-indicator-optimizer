"""
Data Module: Datensammlung, -verarbeitung und -validierung
"""

from .connector import DukascopyConnector
from .processor import DataProcessor
from .models import TickData, OHLCVData, MarketData, IndicatorData

__all__ = ['DukascopyConnector', 'DataProcessor', 'TickData', 'OHLCVData', 'MarketData', 'IndicatorData']