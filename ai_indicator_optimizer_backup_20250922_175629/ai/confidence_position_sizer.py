#!/usr/bin/env python3
"""
Confidence-basierte Position-Sizing mit Risk-Score-Integration
Phase 2 Implementation - Core AI Enhancement (CLEAN)

Features:
- Multi-Factor-Confidence-Scoring
- Risk-Score-Integration
- Dynamic Position-Sizing
- Volatility-Adjustment
- Drawdown-Protection
- (Optional) Kelly-Criterion
"""

from __future__ import annotations
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np


class ConfidencePositionSizer:
    """
    Confidence-basierte Position-Sizing für AI Trading System
    
    Phase 2 Core AI Enhancement:
    - Multi-Factor-Confidence-Scoring
    - Risk-Score-Integration
    - Dynamic Position-Sizing basierend auf AI-Confidence
    - Volatility-Adjustment
    - Drawdown-Protection
    - Kelly-Criterion-Integration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Confidence Position Sizer
        
        Args:
            config: Konfiguration für Position-Sizing
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Position-Sizing-Parameter
        self.base_position_size: float = float(self.config.get("base_position_size", 1_000))
        self.max_position_size: float = float(self.config.get("max_position_size", 5_000))
        self.min_position_size: float = float(self.config.get("min_position_size", 100))
        
        # Confidence-Scaling-Parameter (einheitlich!)
        self.confidence_multiplier: float = float(self.config.get("confidence_multiplier", 2.0))
        self.min_confidence: float = float(self.config.get("min_confidence", 0.60))
        self.max_confidence: float = float(self.config.get("max_confidence", 0.95))
        
        # Risk-Management-Parameter
        self.max_risk_per_trade: float = float(self.config.get("max_risk_per_trade", 0.02))  # 2% vom Account
        self.volatility_adjustment: bool = bool(self.config.get("volatility_adjustment", True))
        self.drawdown_protection: bool = bool(self.config.get("drawdown_protection", True))
        
        # Kelly (vereinfachte Integration)
        self.use_kelly_criterion: bool = bool(self.config.get("use_kelly_criterion", False))
        self.kelly_fraction: float = float(self.config.get("kelly_fraction", 0.25))  # max zusätzlicher Boost
        
        # State
        self.position_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Stats
        self.positions_sized: int = 0
        self.total_risk_adjusted: float = 0.0
        
        self.logger.info(
            f"ConfidencePositionSizer initialized | base={self.base_position_size}, max={self.max_position_size}, "
            f"min_conf={self.min_confidence}, max_conf={self.max_confidence}"
        )
    
    # ---- Public API ----------------------------------------------------------
    
    def calculate_position_size(
        self,
        confidence_score: float,
        risk_score: float,
        market_regime: str,
        volatility: float,
        account_balance: float,
        additional_factors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Berechne Position-Size basierend auf Confidence und Risk-Scores
        
        Args:
            confidence_score: AI-Confidence (0.0-1.0)
            risk_score: Risk-Score (0.0-1.0, höher = riskanter)
            market_regime: Market-Regime ("trending", "ranging", "volatile", etc.)
            volatility: Aktuelle Volatility
            account_balance: Account-Balance
            additional_factors: Zusätzliche Faktoren
            
        Returns:
            Dictionary mit Position-Size und Metriken
        """
        try:
            base_size = self._calculate_confidence_based_size(confidence_score)
            risk_adjusted = self._apply_risk_adjustment(base_size, risk_score)
            regime_adjusted = self._apply_market_regime_adjustment(risk_adjusted, market_regime)
            vol_adjusted = (
                self._apply_volatility_adjustment(regime_adjusted, volatility)
                if self.volatility_adjustment else regime_adjusted
            )
            dd_adjusted = (
                self._apply_drawdown_protection(vol_adjusted)
                if self.drawdown_protection else vol_adjusted
            )
            kelly_adjusted = (
                self._apply_kelly_criterion(dd_adjusted, confidence_score, risk_score)
                if self.use_kelly_criterion else dd_adjusted
            )
            final_size = (
                self._apply_additional_factors(kelly_adjusted, additional_factors)
                if additional_factors else kelly_adjusted
            )
            final_size = self._apply_final_constraints(final_size, account_balance)
            risk_metrics = self._calculate_risk_metrics(final_size, account_balance, volatility)
            
            self.positions_sized += 1
            self.total_risk_adjusted += risk_metrics["risk_amount"]
            
            result = {
                "position_size": int(final_size),
                "confidence_score": float(confidence_score),
                "risk_score": float(risk_score),
                "market_regime": str(market_regime),
                "volatility": float(volatility),
                "sizing_steps": {
                    "base_size": base_size,
                    "risk_adjusted": risk_adjusted,
                    "regime_adjusted": regime_adjusted,
                    "volatility_adjusted": vol_adjusted,
                    "drawdown_adjusted": dd_adjusted,
                    "kelly_adjusted": kelly_adjusted,
                    "final_size": final_size,
                },
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now().isoformat(),
            }
            
            self.position_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return self._get_fallback_position_size(account_balance)
    
    def update_performance(self, position_result: Dict[str, float]):
        """
        Update Performance-History für Drawdown-Berechnung
        
        Args:
            position_result: Dictionary mit PnL-Informationen
        """
        try:
            pnl = float(position_result.get("pnl", 0.0))
            self.performance_history.append(pnl)
            
            # kumulative Kurve & aktueller Drawdown (robust gegen 0)
            if self.performance_history:
                cum = np.cumsum(self.performance_history)
                peak = np.maximum.accumulate(cum)
                dd_series = (peak - cum) / np.maximum(np.abs(peak), 1e-6)
                self.current_drawdown = float(dd_series[-1])
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                
        except Exception as e:
            self.logger.error(f"Error updating performance: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Position Sizer Statistiken"""
        avg_size = float(np.mean([p["position_size"] for p in self.position_history])) if self.position_history else 0.0
        return {
            "positions_sized": self.positions_sized,
            "avg_position_size": avg_size,
            "total_risk_adjusted": self.total_risk_adjusted,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "base_position_size": self.base_position_size,
            "max_position_size": self.max_position_size,
            "confidence_multiplier": self.confidence_multiplier,
            "use_kelly_criterion": self.use_kelly_criterion,
            "volatility_adjustment": self.volatility_adjustment,
            "drawdown_protection": self.drawdown_protection,
        }
    
    # ---- Internals -----------------------------------------------------------
    
    def _calculate_confidence_based_size(self, confidence_score: float) -> float:
        """Berechne Basis-Position-Size basierend auf Confidence"""
        c = max(0.0, min(1.0, float(confidence_score)))
        
        if c < self.min_confidence:
            return 0.0
        
        # Div-by-zero guard
        den = max(1e-9, (self.max_confidence - self.min_confidence))
        conf_factor = (c - self.min_confidence) / den
        conf_factor = max(0.0, min(1.0, conf_factor))
        scaled = conf_factor ** 0.8  # leichte Konkavität
        
        return self.base_position_size * (1.0 + scaled * self.confidence_multiplier)
    
    def _apply_risk_adjustment(self, base_size: float, risk_score: float) -> float:
        """Wende Risk-Score-Adjustment an"""
        r = max(0.0, min(1.0, float(risk_score)))
        factor = max(0.1, 1.0 - 0.5 * r)  # bis −50%
        return base_size * factor
    
    def _apply_market_regime_adjustment(self, size: float, regime: str) -> float:
        """Wende Market-Regime-Adjustment an"""
        factors = {"trending": 1.2, "ranging": 0.8, "volatile": 0.6, "quiet": 1.0, "unknown": 0.9}
        return size * factors.get((regime or "unknown").lower(), 0.9)
    
    def _apply_volatility_adjustment(self, size: float, vol: float) -> float:
        """Wende Volatility-Adjustment an"""
        v = max(0.001, min(0.05, float(vol)))
        factor = 0.001 / v
        factor = max(0.2, min(2.0, factor))
        return size * factor
    
    def _apply_drawdown_protection(self, size: float) -> float:
        """Wende Drawdown-Protection an"""
        d = float(self.current_drawdown)
        
        if d <= 0.05:   return size
        if d <= 0.10:   return size * 0.8
        if d <= 0.15:   return size * 0.6
        return size * 0.4
    
    def _apply_kelly_criterion(self, size: float, confidence: float, risk_score: float) -> float:
        """Wende Kelly-Criterion an"""
        p = max(0.0, min(1.0, float(confidence)))
        q = 1.0 - p
        b = 1.0 + (1.0 - max(0.0, min(1.0, float(risk_score))))  # 1..2
        
        # Kelly
        f = (b * p - q) / max(b, 1e-9)
        f = max(0.0, min(self.kelly_fraction, f))
        
        return size * (1.0 + f)
    
    def _apply_additional_factors(self, size: float, factors: Dict[str, float]) -> float:
        """Wende zusätzliche Faktoren an"""
        m = 1.0
        for k, v in (factors or {}).items():
            v = float(v)
            if k == "correlation_penalty": m *= (1.0 - 0.3 * v)
            elif k == "liquidity_bonus":   m *= (1.0 + 0.2 * v)
            elif k == "news_impact":       m *= (1.0 - 0.4 * v)
            elif k == "time_decay":        m *= (1.0 - 0.2 * v)
        
        return size * max(0.1, m)
    
    def _apply_final_constraints(self, size: float, balance: float) -> float:
        """Wende finale Constraints an"""
        size = max(self.min_position_size, min(self.max_position_size, float(size)))
        
        # nicht mehr als 10% des Accounts
        size = min(size, float(balance) * 0.10)
        
        # runde sinnvoll
        return max(self.min_position_size, round(size / 100.0) * 100.0)
    
    def _calculate_risk_metrics(self, pos_size: float, balance: float, vol: float) -> Dict[str, float]:
        """Berechne Risk-Metriken"""
        # simple proxy; capped by max_risk_per_trade
        risk_amount = float(pos_size) * float(vol) * 2.0  # ~2-sigma
        risk_amount_cap = max(float(balance) * self.max_risk_per_trade, 0.0)
        risk_amount = min(risk_amount, risk_amount_cap)
        risk_pct = risk_amount / max(float(balance), 1e-6)
        pos_pct = float(pos_size) / max(float(balance), 1e-6)
        exp_reward = float(pos_size) * 0.01  # 1% proxy
        rr = exp_reward / max(risk_amount, 1e-9)
        
        return {
            "risk_amount": risk_amount,
            "risk_percentage": risk_pct,
            "position_percentage": pos_pct,
            "risk_reward_ratio": rr,
            "volatility_risk": float(vol),
        }
    
    def _get_fallback_position_size(self, balance: float) -> Dict[str, Any]:
        """Fallback Position-Size bei Fehlern"""
        fb = float(min(self.base_position_size, float(balance) * 0.05))
        
        return {
            "position_size": int(fb),
            "confidence_score": 0.5,
            "risk_score": 0.5,
            "market_regime": "unknown",
            "volatility": 0.01,
            "sizing_steps": {"final_size": fb},
            "risk_metrics": {
                "risk_amount": fb * 0.02,
                "risk_percentage": min(0.02, fb / max(float(balance), 1e-6)),
                "position_percentage": fb / max(float(balance), 1e-6),
                "risk_reward_ratio": 0.5,
            },
            "fallback": True,
            "timestamp": datetime.now().isoformat(),
        }


# Factory Function
def create_confidence_position_sizer(config: Optional[Dict] = None) -> ConfidencePositionSizer:
    """
    Factory Function für Confidence Position Sizer
    
    Args:
        config: Konfiguration für Position-Sizing
        
    Returns:
        ConfidencePositionSizer Instance
    """
    return ConfidencePositionSizer(config=config)


if __name__ == "__main__":
    print("🧪 Testing ConfidencePositionSizer (clean)...")
    
    sizer = create_confidence_position_sizer({
        "base_position_size": 1000,
        "max_position_size": 5000,
        "min_confidence": 0.6,
        "max_confidence": 0.95,
        "use_kelly_criterion": False,
    })
    
    scenarios = [
        {"confidence_score": 0.9, "risk_score": 0.1, "market_regime": "trending", "volatility": 0.005, "balance": 50_000},
        {"confidence_score": 0.7, "risk_score": 0.3, "market_regime": "ranging",  "volatility": 0.010, "balance": 50_000},
        {"confidence_score": 0.6, "risk_score": 0.5, "market_regime": "volatile", "volatility": 0.020, "balance": 50_000},
        {"confidence_score": 0.5, "risk_score": 0.2, "market_regime": "trending", "volatility": 0.008, "balance": 50_000},
    ]
    
    for i, sc in enumerate(scenarios, 1):
        r = sizer.calculate_position_size(
            sc["confidence_score"], sc["risk_score"], sc["market_regime"], sc["volatility"], sc["balance"]
        )
        print(f"#{i} → size={r['position_size']}, conf={r['confidence_score']}, risk={r['risk_score']}, rr={r['risk_metrics']['risk_reward_ratio']:.2f}")
    
    sizer.update_performance({"pnl": 50.0})
    sizer.update_performance({"pnl": -30.0})
    print("stats:", sizer.get_statistics())