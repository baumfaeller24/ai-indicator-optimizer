"""
Trading Library Models
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import uuid


@dataclass
class VisualPattern:
    """Visuelles Trading Pattern"""
    pattern_id: str
    pattern_type: str
    chart_image: Image.Image
    confidence_score: float
    market_context: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.pattern_id is None:
            self.pattern_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class TradingStrategy:
    """Trading Strategie"""
    strategy_id: str
    strategy_name: str
    indicators: List[Dict[str, Any]]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, float]
    performance_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.strategy_id is None:
            self.strategy_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class PerformanceMetrics:
    """Performance Metriken"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float