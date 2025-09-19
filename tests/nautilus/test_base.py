"""
Basis-Test für Nautilus Trading System
"""
import pytest
import asyncio
from nautilus_trader.test_kit.stubs.component import TestComponentStubs
from nautilus_trader.test_kit.stubs.identifiers import TestIdStubs


class TestNautilusBase:
    """Basis-Testklasse für Nautilus-Komponenten"""
    
    def setup_method(self):
        """Setup für jeden Test"""
        self.trader_id = TestIdStubs.trader_id()
        
    def test_basic_setup(self):
        """Test basic setup"""
        assert self.trader_id is not None
        
    @pytest.mark.asyncio
    async def test_async_setup(self):
        """Test async setup"""
        await asyncio.sleep(0.001)  # Minimal async test
        assert True
