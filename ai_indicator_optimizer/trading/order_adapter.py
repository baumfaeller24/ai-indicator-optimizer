#!/usr/bin/env python3
"""
Production-Ready Order Adapter für NautilusTrader
Robuste Order-Submission mit Fallback-Mechanismen
"""

from typing import Any, Optional
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.identifiers import InstrumentId
import logging


class OrderAdapter:
    """
    Production-Ready Order Adapter für verschiedene NautilusTrader-Versionen
    
    Bietet zwei Modi:
    1. "convenience" - Nutzt Strategy.submit_market_order() wenn verfügbar
    2. "explicit" - Erstellt explizite MarketOrder-Objekte
    """
    
    def __init__(self, strategy: Any, mode: str = "convenience", logger: Optional[logging.Logger] = None):
        """
        Initialize Order Adapter
        
        Args:
            strategy: NautilusTrader Strategy instance
            mode: "convenience" or "explicit"
            logger: Optional logger
        """
        self.strategy = strategy
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Stats
        self.orders_submitted = 0
        self.orders_failed = 0
    
    def submit_market_order(
        self, 
        instrument_id: InstrumentId, 
        side: OrderSide, 
        quantity: int,
        time_in_force: Optional[TimeInForce] = None
    ) -> bool:
        """
        Submit market order with fallback mechanisms
        
        Args:
            instrument_id: Trading instrument
            side: Order side (BUY/SELL)
            quantity: Order quantity
            time_in_force: Optional time in force
            
        Returns:
            bool: True if order submitted successfully
        """
        try:
            if self.mode == "convenience":
                return self._submit_convenience(instrument_id, side, quantity)
            else:
                return self._submit_explicit(instrument_id, side, quantity, time_in_force)
                
        except Exception as e:
            self.orders_failed += 1
            self.logger.error(f"Order submission failed: {e}")
            return False
    
    def _submit_convenience(self, instrument_id: InstrumentId, side: OrderSide, quantity: int) -> bool:
        """Try convenience API first"""
        try:
            # Method 1: Direct submit_market_order
            if hasattr(self.strategy, 'submit_market_order'):
                self.strategy.submit_market_order(instrument_id, side, quantity)
                self.orders_submitted += 1
                self.logger.info(f"✅ Convenience order submitted: {side.name} {quantity} {instrument_id}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Convenience method failed: {e}, trying explicit...")
            return self._submit_explicit(instrument_id, side, quantity)
        
        return False
    
    def _submit_explicit(
        self, 
        instrument_id: InstrumentId, 
        side: OrderSide, 
        quantity: int,
        time_in_force: Optional[TimeInForce] = None
    ) -> bool:
        """Create explicit MarketOrder"""
        try:
            # Get instrument for proper quantity formatting
            instrument = None
            if hasattr(self.strategy, 'cache') and self.strategy.cache:
                instrument = self.strategy.cache.instrument(instrument_id)
            
            # Format quantity
            if instrument and hasattr(instrument, 'make_qty'):
                formatted_qty = instrument.make_qty(quantity)
            else:
                formatted_qty = quantity
            
            # Create market order
            order = MarketOrder(
                trader_id=self.strategy.trader_id,
                strategy_id=self.strategy.id,
                instrument_id=instrument_id,
                order_side=side,
                quantity=formatted_qty,
                time_in_force=time_in_force or TimeInForce.GTC,
                order_id=self.strategy.generate_order_id(),
                ts_init=self.strategy.clock.timestamp_ns(),
            )
            
            # Submit order
            self.strategy.submit_order(order)
            self.orders_submitted += 1
            self.logger.info(f"✅ Explicit order submitted: {side.name} {quantity} {instrument_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Explicit order submission failed: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get order adapter statistics"""
        return {
            "mode": self.mode,
            "orders_submitted": self.orders_submitted,
            "orders_failed": self.orders_failed,
            "success_rate": (self.orders_submitted / max(self.orders_submitted + self.orders_failed, 1)) * 100
        }


# Convenience functions
def create_order_adapter(strategy: Any, mode: str = "convenience") -> OrderAdapter:
    """
    Factory function for creating OrderAdapter
    
    Args:
        strategy: NautilusTrader Strategy instance
        mode: "convenience" or "explicit"
        
    Returns:
        OrderAdapter instance
    """
    return OrderAdapter(strategy, mode, logger=getattr(strategy, 'log', None))


def submit_market_order_safe(
    strategy: Any, 
    instrument_id: InstrumentId, 
    side: OrderSide, 
    quantity: int,
    mode: str = "convenience"
) -> bool:
    """
    One-shot function for safe market order submission
    
    Args:
        strategy: NautilusTrader Strategy instance
        instrument_id: Trading instrument
        side: Order side
        quantity: Order quantity
        mode: "convenience" or "explicit"
        
    Returns:
        bool: True if successful
    """
    adapter = create_order_adapter(strategy, mode)
    return adapter.submit_market_order(instrument_id, side, quantity)


if __name__ == "__main__":
    # Test the order adapter (mock)
    print("🧪 Testing OrderAdapter...")
    
    class MockStrategy:
        def __init__(self):
            self.trader_id = "test_trader"
            self.id = "test_strategy"
            self.orders_submitted = []
            
        def submit_market_order(self, instrument_id, side, quantity):
            self.orders_submitted.append((instrument_id, side, quantity))
            print(f"Mock order: {side.name} {quantity} {instrument_id}")
            
        def generate_order_id(self):
            return f"order_{len(self.orders_submitted)}"
            
        class MockClock:
            def timestamp_ns(self):
                import time
                return int(time.time() * 1e9)
        
        clock = MockClock()
    
    # Test convenience mode
    strategy = MockStrategy()
    adapter = create_order_adapter(strategy, "convenience")
    
    from nautilus_trader.model.identifiers import InstrumentId
    instrument_id = InstrumentId.from_str("EUR/USD.SIM")
    
    success = adapter.submit_market_order(instrument_id, OrderSide.BUY, 1000)
    print(f"Order success: {success}")
    print(f"Stats: {adapter.get_stats()}")
    
    print("✅ OrderAdapter Test completed!")