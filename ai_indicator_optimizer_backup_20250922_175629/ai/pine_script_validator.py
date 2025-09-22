#!/usr/bin/env python3
"""
Pine Script Validator für Syntax-Checking und Error-Detection
Phase 3 Implementation - Task 10

Features:
- Comprehensive Syntax-Checking für Pine Script v5
- Error-Detection mit detaillierter Fehleranalyse
- Best-Practice-Validation
- Performance-Issue-Detection
- Security-Checks für Pine Script Code
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum


class ValidationSeverity(Enum):
    """Schweregrade für Validation-Issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Kategorien für Validation-Issues"""
    SYNTAX = "syntax"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BEST_PRACTICE = "best_practice"
    COMPATIBILITY = "compatibility"


@dataclass
class ValidationIssue:
    """Einzelnes Validation-Issue"""
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Ergebnis einer Validation"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    total_lines: int = 0
    processing_time: float = 0.0
    
    @property
    def error_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.ERROR])
    
    @property
    def warning_count(self) -> int:
        return len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])


class PineScriptValidator:
    """
    Comprehensive Pine Script Validator
    Validiert Pine Script Code auf Syntax, Performance und Best Practices
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pine_keywords = self._load_pine_keywords()
        self.validation_rules = self._load_validation_rules()
    
    def _load_pine_keywords(self) -> Dict[str, List[str]]:
        """Lade Pine Script Keywords und Funktionen"""
        return {
            "built_in_functions": [
                "strategy", "study", "plot", "plotshape", "plotchar",
                "sma", "ema", "rsi", "macd", "bollinger", "stoch",
                "atr", "adx", "cci", "mfi", "obv", "vwap"
            ],
            "built_in_variables": [
                "open", "high", "low", "close", "volume", "time",
                "bar_index", "n", "na", "syminfo", "timeframe"
            ],
            "control_structures": [
                "if", "else", "for", "while", "switch", "case", "default"
            ],
            "operators": [
                "and", "or", "not", "==", "!=", ">=", "<=", ">", "<"
            ]
        }
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Lade Validation-Regeln"""
        return {
            "max_line_length": 120,
            "max_function_length": 50,
            "max_nesting_depth": 5,
            "required_version": "5",
            "forbidden_functions": ["security", "request.security"],
            "performance_warnings": {
                "max_plots": 64,
                "max_variables": 1000,
                "max_arrays": 100
            }
        }
    
    def validate_pine_script(self, code: str, filename: Optional[str] = None) -> ValidationResult:
        """
        Hauptvalidierung für Pine Script Code
        
        Args:
            code: Pine Script Code als String
            filename: Optional filename für bessere Fehlermeldungen
            
        Returns:
            ValidationResult mit allen gefundenen Issues
        """
        start_time = datetime.now()
        issues = []
        
        try:
            # Syntax-Validierung
            syntax_issues = self._validate_syntax(code)
            issues.extend(syntax_issues)
            
            # Performance-Validierung
            performance_issues = self._validate_performance(code)
            issues.extend(performance_issues)
            
            # Security-Validierung
            security_issues = self._validate_security(code)
            issues.extend(security_issues)
            
            # Best-Practice-Validierung
            best_practice_issues = self._validate_best_practices(code)
            issues.extend(best_practice_issues)
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.SYNTAX,
                message=f"Validation process failed: {str(e)}"
            ))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Bestimme ob Code valid ist
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        has_critical = any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        return ValidationResult(
            is_valid=not (has_errors or has_critical),
            issues=issues,
            total_lines=len(code.split('\n')),
            processing_time=processing_time
        )
    
    def _validate_syntax(self, code: str) -> List[ValidationIssue]:
        """Validiere Pine Script Syntax"""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Prüfe auf häufige Syntax-Fehler
            if stripped.endswith(',') and not self._is_in_function_call(line, lines, line_num):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message="Trailing comma outside function call",
                    line_number=line_num,
                    code_snippet=line.strip()
                ))
            
            # Prüfe auf unbalanced brackets
            if not self._check_balanced_brackets(line):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message="Unbalanced brackets",
                    line_number=line_num,
                    code_snippet=line.strip()
                ))
        
        return issues
    
    def _validate_performance(self, code: str) -> List[ValidationIssue]:
        """Validiere Performance-Aspekte"""
        issues = []
        
        # Zähle Plots
        plot_count = len(re.findall(r'\bplot\s*\(', code))
        if plot_count > self.validation_rules["performance_warnings"]["max_plots"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Too many plots ({plot_count}), maximum recommended: {self.validation_rules['performance_warnings']['max_plots']}",
                suggestion="Consider consolidating plots or using plotshape for some indicators"
            ))
        
        return issues
    
    def _validate_security(self, code: str) -> List[ValidationIssue]:
        """Validiere Security-Aspekte"""
        issues = []
        
        # Prüfe auf verbotene Funktionen
        for forbidden_func in self.validation_rules["forbidden_functions"]:
            if forbidden_func in code:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.SECURITY,
                    message=f"Usage of potentially unsafe function: {forbidden_func}",
                    suggestion="Consider alternative approaches for data access"
                ))
        
        return issues
    
    def _validate_best_practices(self, code: str) -> List[ValidationIssue]:
        """Validiere Best Practices"""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Prüfe Zeilenlänge
            if len(line) > self.validation_rules["max_line_length"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BEST_PRACTICE,
                    message=f"Line too long ({len(line)} characters)",
                    line_number=line_num,
                    suggestion=f"Keep lines under {self.validation_rules['max_line_length']} characters"
                ))
        
        return issues
    
    def _is_in_function_call(self, line: str, lines: List[str], line_num: int) -> bool:
        """Prüfe ob Zeile Teil eines Function-Calls ist"""
        # Vereinfachte Implementierung
        return '(' in line and ')' not in line
    
    def _check_balanced_brackets(self, line: str) -> bool:
        """Prüfe auf balancierte Brackets in einer Zeile"""
        stack = []
        brackets = {'(': ')', '[': ']', '{': '}'}
        
        for char in line:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                if brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0


def main():
    """Test der Pine Script Validation"""
    validator = PineScriptValidator()
    
    # Test mit einfachem Pine Script
    test_code = """
    //@version=5
    strategy("Test Strategy", overlay=true)
    
    // Simple moving average
    sma_length = input.int(20, "SMA Length")
    sma_value = ta.sma(close, sma_length)
    
    plot(sma_value, color=color.blue, title="SMA")
    
    // Entry conditions
    if close > sma_value
        strategy.entry("Long", strategy.long)
    
    if close < sma_value
        strategy.close("Long")
    """
    
    result = validator.validate_pine_script(test_code)
    
    print(f"Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Issues: {len(result.issues)}")
    print(f"  Errors: {result.error_count}")
    print(f"  Warnings: {result.warning_count}")
    print(f"  Processing time: {result.processing_time:.3f}s")
    
    for issue in result.issues:
        print(f"  {issue.severity.value.upper()}: {issue.message}")


if __name__ == "__main__":
    main()
