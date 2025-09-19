"""
Einfache Buy-and-Hold Strategie für Nautilus
"""
from nautilus_trader.strategy.strategy import Strategy
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.orders import MarketOrder


class BuyAndHoldStrategy(Strategy):
    """
    Einfache Buy-and-Hold Strategie
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.instrument_id = None
        self.position_opened = False
        
    def on_start(self):
        """Called when strategy starts"""
        self.log.info("BuyAndHoldStrategy started")
        
    def on_stop(self):
        """Called when strategy stops"""
        self.log.info("BuyAndHoldStrategy stopped")
        
    def on_bar(self, bar: Bar):
        """Called on each bar"""
        if not self.position_opened:
            # Open position on first bar
            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=bar.bar_type.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(1000),
                time_in_force=self.time_in_force,
                order_id=self.generate_order_id(),
                ts_init=self.clock.timestamp_ns(),
            )
            
            self.submit_order(order)
            self.position_opened = True
            self.log.info(f"Opened position: {order}")
