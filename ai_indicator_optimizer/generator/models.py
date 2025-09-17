"""
Pine Script Generator Models
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class GeneratedCode:
    """Generierter Pine Script Code"""
    code: str
    strategy_name: str
    version: str = "5"
    indicators_used: List[str] = None
    entry_conditions: List[str] = None
    exit_conditions: List[str] = None
    risk_management: Dict[str, Any] = None
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()
        if self.indicators_used is None:
            self.indicators_used = []
        if self.entry_conditions is None:
            self.entry_conditions = []
        if self.exit_conditions is None:
            self.exit_conditions = []
        if self.risk_management is None:
            self.risk_management = {}


@dataclass
class ValidationResult:
    """Pine Script Validierungs-Ergebnis"""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []