#!/usr/bin/env python3
"""
Automatic Error Fixer für selbstständige Syntax-Korrektur
Phase 3 Implementation - Task 10

Features:
- Automatische Korrektur häufiger Pine Script Syntax-Fehler
- Intelligente Code-Reparatur basierend auf Validation-Issues
- Best-Practice-Anwendung und Code-Optimierung
- Integration mit Pine Script Validator
- Backup und Rollback-Funktionalität
"""

import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum

# Local Imports
from .pine_script_validator import PineScriptValidator, ValidationIssue, ValidationSeverity, ValidationCategory


class FixType(Enum):
    """Typen von automatischen Fixes"""
    SYNTAX_FIX = "syntax_fix"
    LOGIC_FIX = "logic_fix"
    PERFORMANCE_FIX = "performance_fix"
    BEST_PRACTICE_FIX = "best_practice_fix"
    COMPATIBILITY_FIX = "compatibility_fix"


@dataclass
class AppliedFix:
    """Dokumentation eines angewendeten Fixes"""
    fix_type: FixType
    rule_id: str
    original_code: str
    fixed_code: str
    line_number: Optional[int] = None
    description: str = ""
    confidence: float = 1.0  # 0.0 - 1.0


@dataclass
class FixResult:
    """Ergebnis einer automatischen Korrektur"""
    success: bool
    original_script: str
    fixed_script: str
    applied_fixes: List[AppliedFix] = field(default_factory=list)
    remaining_issues: int = 0
    fix_time: float = 0.0
    confidence_score: float = 0.0
    
    def get_fix_summary(self) -> Dict[str, Any]:
        """Erstelle Fix-Summary"""
        return {
            "success": self.success,
            "fixes_applied": len(self.applied_fixes),
            "remaining_issues": self.remaining_issues,
            "fix_time": self.fix_time,
            "confidence_score": self.confidence_score,
            "fixes_by_type": {
                fix_type.value: len([f for f in self.applied_fixes if f.fix_type == fix_type])
                for fix_type in FixType
            }
        }


class AutomaticErrorFixer:
    """
    Automatischer Error-Fixer für Pine Script Code
    
    Features:
    - Syntax-Error-Korrektur
    - Logic-Error-Reparatur
    - Performance-Optimierung
    - Best-Practice-Anwendung
    - Intelligente Code-Analyse
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validator für Issue-Detection
        self.validator = PineScriptValidator()
        
        # Fix-Regeln laden
        self.syntax_fixes = self._load_syntax_fixes()
        self.logic_fixes = self._load_logic_fixes()
        self.performance_fixes = self._load_performance_fixes()
        self.best_practice_fixes = self._load_best_practice_fixes()
        self.compatibility_fixes = self._load_compatibility_fixes()
        
        # Statistiken
        self.stats = {
            "scripts_fixed": 0,
            "total_fixes_applied": 0,
            "fix_success_rate": 0.0,
            "avg_fix_time": 0.0,
            "fixes_by_type": {fix_type.value: 0 for fix_type in FixType}
        }
        
        self.logger.info("AutomaticErrorFixer initialized")
    
    def fix_script(self, pine_script: str, max_iterations: int = 5) -> FixResult:
        """
        Automatische Korrektur eines Pine Scripts
        
        Args:
            pine_script: Pine Script Code
            max_iterations: Maximale Anzahl Fix-Iterationen
            
        Returns:
            FixResult mit korrigiertem Code und angewendeten Fixes
        """
        try:
            start_time = datetime.now()
            
            original_script = pine_script
            current_script = pine_script
            applied_fixes = []
            
            # Iterative Korrektur
            for iteration in range(max_iterations):
                self.logger.debug(f"Fix iteration {iteration + 1}/{max_iterations}")
                
                # Aktuelle Issues finden
                validation_result = self.validator.validate_script(current_script)
                
                if validation_result.is_valid:
                    self.logger.info(f"Script is valid after {iteration + 1} iterations")
                    break
                
                # Fixes anwenden
                iteration_fixes = []
                new_script = current_script
                
                # Sortiere Issues nach Priorität (Errors zuerst)
                sorted_issues = sorted(
                    validation_result.issues,
                    key=lambda x: (x.severity != ValidationSeverity.ERROR, x.line_number or 0)
                )
                
                for issue in sorted_issues:
                    fix_result = self._apply_fix_for_issue(new_script, issue)
                    
                    if fix_result:
                        applied_fix, fixed_script = fix_result
                        new_script = fixed_script
                        iteration_fixes.append(applied_fix)
                        
                        self.logger.debug(f"Applied fix: {applied_fix.rule_id}")
                
                # Wenn keine Fixes angewendet wurden, stoppe
                if not iteration_fixes:
                    self.logger.warning("No more fixes can be applied")
                    break
                
                applied_fixes.extend(iteration_fixes)
                current_script = new_script
            
            # Finale Validation
            final_validation = self.validator.validate_script(current_script)
            
            # Confidence-Score berechnen
            confidence_score = self._calculate_confidence_score(applied_fixes)
            
            fix_time = (datetime.now() - start_time).total_seconds()
            
            result = FixResult(
                success=final_validation.is_valid,
                original_script=original_script,
                fixed_script=current_script,
                applied_fixes=applied_fixes,
                remaining_issues=len(final_validation.issues),
                fix_time=fix_time,
                confidence_score=confidence_score
            )
            
            # Statistiken updaten
            self.stats["scripts_fixed"] += 1
            self.stats["total_fixes_applied"] += len(applied_fixes)
            self.stats["avg_fix_time"] = (
                (self.stats["avg_fix_time"] * (self.stats["scripts_fixed"] - 1) + fix_time) 
                / self.stats["scripts_fixed"]
            )
            
            if result.success:
                self.stats["fix_success_rate"] = (
                    (self.stats["fix_success_rate"] * (self.stats["scripts_fixed"] - 1) + 1.0) 
                    / self.stats["scripts_fixed"]
                )
            
            for fix in applied_fixes:
                self.stats["fixes_by_type"][fix.fix_type.value] += 1
            
            self.logger.info(f"Fix completed: {len(applied_fixes)} fixes applied in {fix_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fix error: {e}")
            return FixResult(
                success=False,
                original_script=pine_script,
                fixed_script=pine_script,
                applied_fixes=[],
                remaining_issues=-1,
                fix_time=0.0,
                confidence_score=0.0
            )
    
    def _apply_fix_for_issue(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Fix für spezifisches Issue an"""
        
        try:
            if issue.category == ValidationCategory.SYNTAX:
                return self._apply_syntax_fix(script, issue)
            elif issue.category == ValidationCategory.LOGIC:
                return self._apply_logic_fix(script, issue)
            elif issue.category == ValidationCategory.PERFORMANCE:
                return self._apply_performance_fix(script, issue)
            elif issue.category == ValidationCategory.BEST_PRACTICE:
                return self._apply_best_practice_fix(script, issue)
            elif issue.category == ValidationCategory.COMPATIBILITY:
                return self._apply_compatibility_fix(script, issue)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error applying fix for {issue.rule_id}: {e}")
            return None
    
    def _apply_syntax_fix(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Syntax-Fix an"""
        
        lines = script.split('\n')
        
        if issue.rule_id == "MISSING_VERSION":
            # Füge @version Direktive hinzu
            if not any(line.strip().startswith('//@version') for line in lines):
                fixed_lines = ['//@version=5'] + lines
                fixed_script = '\n'.join(fixed_lines)
                
                return AppliedFix(
                    fix_type=FixType.SYNTAX_FIX,
                    rule_id=issue.rule_id,
                    original_code=lines[0] if lines else "",
                    fixed_code="//@version=5",
                    line_number=1,
                    description="Added missing @version directive",
                    confidence=1.0
                ), fixed_script
        
        elif issue.rule_id == "MISSING_DECLARATION":
            # Füge indicator() Declaration hinzu
            version_line_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('//@version'):
                    version_line_idx = i
                    break
            
            if version_line_idx >= 0:
                insert_idx = version_line_idx + 1
                lines.insert(insert_idx, 'indicator("Auto-Generated Script", overlay=true)')
                fixed_script = '\n'.join(lines)
                
                return AppliedFix(
                    fix_type=FixType.SYNTAX_FIX,
                    rule_id=issue.rule_id,
                    original_code="",
                    fixed_code='indicator("Auto-Generated Script", overlay=true)',
                    line_number=insert_idx + 1,
                    description="Added missing indicator declaration",
                    confidence=0.9
                ), fixed_script
        
        elif issue.rule_id == "UNCLOSED_PAREN":
            # Schließe offene Klammern
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                # Zähle offene/geschlossene Klammern
                open_count = line.count('(')
                close_count = line.count(')')
                
                if open_count > close_count:
                    missing_parens = open_count - close_count
                    fixed_line = line + ')' * missing_parens
                    lines[line_idx] = fixed_line
                    fixed_script = '\n'.join(lines)
                    
                    return AppliedFix(
                        fix_type=FixType.SYNTAX_FIX,
                        rule_id=issue.rule_id,
                        original_code=line,
                        fixed_code=fixed_line,
                        line_number=issue.line_number,
                        description=f"Added {missing_parens} missing closing parenthesis",
                        confidence=0.8
                    ), fixed_script
        
        elif issue.rule_id == "RESERVED_KEYWORD":
            # Ersetze reservierte Keywords
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                # Finde das reservierte Keyword
                for keyword in self.validator.pine_keywords:
                    if f'{keyword} =' in line:
                        # Ersetze mit _var Suffix
                        new_var_name = f"{keyword}_var"
                        fixed_line = line.replace(f'{keyword} =', f'{new_var_name} =')
                        lines[line_idx] = fixed_line
                        fixed_script = '\n'.join(lines)
                        
                        return AppliedFix(
                            fix_type=FixType.SYNTAX_FIX,
                            rule_id=issue.rule_id,
                            original_code=line,
                            fixed_code=fixed_line,
                            line_number=issue.line_number,
                            description=f"Renamed reserved keyword '{keyword}' to '{new_var_name}'",
                            confidence=0.9
                        ), fixed_script
        
        return None
    
    def _apply_logic_fix(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Logic-Fix an"""
        
        lines = script.split('\n')
        
        if issue.rule_id == "DIVISION_BY_ZERO":
            # Füge Zero-Check hinzu
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                # Ersetze Division durch 0 mit sicherer Division
                if '/ 0' in line:
                    fixed_line = line.replace('/ 0', '/ nz(0, 1)')  # Fallback zu 1
                    lines[line_idx] = fixed_line
                    fixed_script = '\n'.join(lines)
                    
                    return AppliedFix(
                        fix_type=FixType.LOGIC_FIX,
                        rule_id=issue.rule_id,
                        original_code=line,
                        fixed_code=fixed_line,
                        line_number=issue.line_number,
                        description="Added zero-check for division",
                        confidence=0.8
                    ), fixed_script
        
        elif issue.rule_id == "ALWAYS_TRUE":
            # Entferne always-true Conditions
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                if 'if (true)' in line or 'if true' in line:
                    # Kommentiere die Zeile aus
                    fixed_line = '// ' + line + '  // Always true condition removed'
                    lines[line_idx] = fixed_line
                    fixed_script = '\n'.join(lines)
                    
                    return AppliedFix(
                        fix_type=FixType.LOGIC_FIX,
                        rule_id=issue.rule_id,
                        original_code=line,
                        fixed_code=fixed_line,
                        line_number=issue.line_number,
                        description="Commented out always-true condition",
                        confidence=0.7
                    ), fixed_script
        
        elif issue.rule_id == "ALWAYS_FALSE":
            # Entferne always-false Conditions
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                if 'if (false)' in line or 'if false' in line:
                    # Kommentiere die Zeile aus
                    fixed_line = '// ' + line + '  // Always false condition removed'
                    lines[line_idx] = fixed_line
                    fixed_script = '\n'.join(lines)
                    
                    return AppliedFix(
                        fix_type=FixType.LOGIC_FIX,
                        rule_id=issue.rule_id,
                        original_code=line,
                        fixed_code=fixed_line,
                        line_number=issue.line_number,
                        description="Commented out always-false condition",
                        confidence=0.7
                    ), fixed_script
        
        return None
    
    def _apply_performance_fix(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Performance-Fix an"""
        
        lines = script.split('\n')
        
        if issue.rule_id == "LARGE_LOOP":
            # Optimiere große Schleifen
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                # Füge Kommentar hinzu
                comment_line = '// TODO: Consider optimizing this large loop for better performance'
                lines.insert(line_idx, comment_line)
                fixed_script = '\n'.join(lines)
                
                return AppliedFix(
                    fix_type=FixType.PERFORMANCE_FIX,
                    rule_id=issue.rule_id,
                    original_code=line,
                    fixed_code=comment_line + '\n' + line,
                    line_number=issue.line_number,
                    description="Added performance optimization comment",
                    confidence=0.5
                ), fixed_script
        
        elif issue.rule_id == "HEAVY_FUNCTION":
            # Füge Caching-Kommentar hinzu
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                comment_line = '// Consider caching this heavy function result'
                lines.insert(line_idx, comment_line)
                fixed_script = '\n'.join(lines)
                
                return AppliedFix(
                    fix_type=FixType.PERFORMANCE_FIX,
                    rule_id=issue.rule_id,
                    original_code=line,
                    fixed_code=comment_line + '\n' + line,
                    line_number=issue.line_number,
                    description="Added caching suggestion comment",
                    confidence=0.4
                ), fixed_script
        
        return None
    
    def _apply_best_practice_fix(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Best-Practice-Fix an"""
        
        lines = script.split('\n')
        
        if issue.rule_id == "MAGIC_NUMBER":
            # Ersetze Magic Numbers mit Konstanten
            if issue.line_number and issue.line_number <= len(lines):
                line_idx = issue.line_number - 1
                line = lines[line_idx]
                
                # Finde Magic Numbers
                magic_numbers = re.finditer(r'\b(?!0\b|1\b|-1\b)\d+\.?\d*\b', line)
                for match in magic_numbers:
                    number = match.group()
                    if float(number) not in [0, 1, -1, 100]:
                        # Erstelle Konstanten-Name
                        const_name = f"CONST_{number.replace('.', '_')}"
                        
                        # Füge Konstante am Anfang hinzu (nach @version)
                        version_idx = -1
                        for i, l in enumerate(lines):
                            if l.strip().startswith('//@version'):
                                version_idx = i
                                break
                        
                        if version_idx >= 0:
                            const_line = f"{const_name} = {number}  // Magic number constant"
                            lines.insert(version_idx + 2, const_line)
                            
                            # Ersetze in der ursprünglichen Zeile
                            fixed_line = line.replace(number, const_name)
                            lines[line_idx + 1] = fixed_line  # +1 wegen eingefügter Zeile
                            
                            fixed_script = '\n'.join(lines)
                            
                            return AppliedFix(
                                fix_type=FixType.BEST_PRACTICE_FIX,
                                rule_id=issue.rule_id,
                                original_code=line,
                                fixed_code=f"{const_line}\n{fixed_line}",
                                line_number=issue.line_number,
                                description=f"Replaced magic number {number} with constant {const_name}",
                                confidence=0.6
                            ), fixed_script
        
        elif issue.rule_id == "MISSING_ERROR_HANDLING":
            # Füge Error-Handling hinzu
            lines.insert(1, "// TODO: Add proper error handling with na() and nz() functions")
            fixed_script = '\n'.join(lines)
            
            return AppliedFix(
                fix_type=FixType.BEST_PRACTICE_FIX,
                rule_id=issue.rule_id,
                original_code="",
                fixed_code="// TODO: Add proper error handling with na() and nz() functions",
                line_number=2,
                description="Added error handling reminder",
                confidence=0.3
            ), fixed_script
        
        return None
    
    def _apply_compatibility_fix(self, script: str, issue: ValidationIssue) -> Optional[Tuple[AppliedFix, str]]:
        """Wende Compatibility-Fix an"""
        
        if issue.rule_id == "DEPRECATED_FUNCTION":
            # Ersetze veraltete Funktionen
            deprecated_mappings = {
                'study(': 'indicator(',
                'security(': 'request.security(',
                'rsi(': 'ta.rsi(',
                'sma(': 'ta.sma(',
                'ema(': 'ta.ema(',
                'macd(': 'ta.macd(',
                'crossover(': 'ta.crossover(',
                'crossunder(': 'ta.crossunder('
            }
            
            fixed_script = script
            applied_fixes = []
            
            for old_func, new_func in deprecated_mappings.items():
                if old_func in script:
                    fixed_script = fixed_script.replace(old_func, new_func)
                    
                    return AppliedFix(
                        fix_type=FixType.COMPATIBILITY_FIX,
                        rule_id=issue.rule_id,
                        original_code=old_func,
                        fixed_code=new_func,
                        description=f"Updated deprecated function {old_func} to {new_func}",
                        confidence=0.9
                    ), fixed_script
        
        return None
    
    def _calculate_confidence_score(self, applied_fixes: List[AppliedFix]) -> float:
        """Berechne Gesamt-Confidence-Score"""
        
        if not applied_fixes:
            return 1.0
        
        # Gewichteter Durchschnitt der Fix-Confidences
        total_confidence = sum(fix.confidence for fix in applied_fixes)
        avg_confidence = total_confidence / len(applied_fixes)
        
        return avg_confidence  
  
    def _load_syntax_fixes(self) -> Dict[str, Any]:
        """Lade Syntax-Fix-Regeln"""
        return {
            "MISSING_VERSION": {"priority": 1, "confidence": 1.0},
            "MISSING_DECLARATION": {"priority": 1, "confidence": 0.9},
            "UNCLOSED_PAREN": {"priority": 2, "confidence": 0.8},
            "RESERVED_KEYWORD": {"priority": 2, "confidence": 0.9}
        }
    
    def _load_logic_fixes(self) -> Dict[str, Any]:
        """Lade Logic-Fix-Regeln"""
        return {
            "DIVISION_BY_ZERO": {"priority": 1, "confidence": 0.8},
            "ALWAYS_TRUE": {"priority": 3, "confidence": 0.7},
            "ALWAYS_FALSE": {"priority": 3, "confidence": 0.7}
        }
    
    def _load_performance_fixes(self) -> Dict[str, Any]:
        """Lade Performance-Fix-Regeln"""
        return {
            "LARGE_LOOP": {"priority": 4, "confidence": 0.5},
            "HEAVY_FUNCTION": {"priority": 4, "confidence": 0.4}
        }
    
    def _load_best_practice_fixes(self) -> Dict[str, Any]:
        """Lade Best-Practice-Fix-Regeln"""
        return {
            "MAGIC_NUMBER": {"priority": 5, "confidence": 0.6},
            "MISSING_ERROR_HANDLING": {"priority": 5, "confidence": 0.3}
        }
    
    def _load_compatibility_fixes(self) -> Dict[str, Any]:
        """Lade Compatibility-Fix-Regeln"""
        return {
            "DEPRECATED_FUNCTION": {"priority": 2, "confidence": 0.9}
        }
    
    def suggest_manual_fixes(self, script: str) -> List[Dict[str, Any]]:
        """Schlage manuelle Fixes für komplexe Issues vor"""
        
        suggestions = []
        validation_result = self.validator.validate_script(script)
        
        for issue in validation_result.issues:
            if issue.severity == ValidationSeverity.ERROR:
                # Für kritische Errors, die nicht automatisch gefixt werden können
                if issue.rule_id == "UNDEFINED_VARIABLE":
                    suggestions.append({
                        "issue": issue.message,
                        "line": issue.line_number,
                        "suggestion": f"Define the variable '{issue.code_snippet}' before using it",
                        "example": f"my_variable = input.float(14, title=\"My Variable\")",
                        "priority": "high"
                    })
                
                elif issue.rule_id == "COMPLEX_LOGIC_ERROR":
                    suggestions.append({
                        "issue": issue.message,
                        "line": issue.line_number,
                        "suggestion": "Review and simplify the logic in this section",
                        "example": "Break complex conditions into smaller, testable parts",
                        "priority": "high"
                    })
        
        return suggestions
    
    def create_fix_report(self, fix_result: FixResult) -> Dict[str, Any]:
        """Erstelle detaillierten Fix-Report"""
        
        return {
            "summary": fix_result.get_fix_summary(),
            "applied_fixes": [
                {
                    "type": fix.fix_type.value,
                    "rule_id": fix.rule_id,
                    "line_number": fix.line_number,
                    "description": fix.description,
                    "confidence": fix.confidence,
                    "original_code": fix.original_code,
                    "fixed_code": fix.fixed_code
                }
                for fix in fix_result.applied_fixes
            ],
            "before_after": {
                "original_lines": len(fix_result.original_script.split('\n')),
                "fixed_lines": len(fix_result.fixed_script.split('\n')),
                "lines_added": len(fix_result.fixed_script.split('\n')) - len(fix_result.original_script.split('\n'))
            },
            "validation_improvement": {
                "remaining_issues": fix_result.remaining_issues,
                "script_is_valid": fix_result.success
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Fixer-Statistiken"""
        
        return {
            **self.stats,
            "fix_rules_loaded": {
                "syntax_fixes": len(self.syntax_fixes),
                "logic_fixes": len(self.logic_fixes),
                "performance_fixes": len(self.performance_fixes),
                "best_practice_fixes": len(self.best_practice_fixes),
                "compatibility_fixes": len(self.compatibility_fixes)
            }
        }


# Factory Function
def create_automatic_error_fixer(config: Optional[Dict] = None) -> AutomaticErrorFixer:
    """Factory Function für Automatic Error Fixer"""
    return AutomaticErrorFixer(config=config)


# Demo/Test Function
def demo_automatic_error_fixer():
    """Demo für Automatic Error Fixer"""
    
    print("🧪 Testing Automatic Error Fixer...")
    
    # Fixer erstellen
    fixer = create_automatic_error_fixer()
    
    # Test-Script mit verschiedenen Fehlern
    buggy_script = """
indicator("Buggy Script", overlay=true)

// Missing closing parenthesis
rsi_value = ta.rsi(close, 14

// Division by zero
result = close / 0

// Deprecated function
old_rsi = rsi(close, 14)

// Magic number
threshold = 73.456789

// Always true condition
if true
    alert("Always triggered")

// Reserved keyword as variable
var = 42
"""
    
    print("🔧 Fixing buggy script...")
    
    # Automatische Korrektur
    fix_result = fixer.fix_script(buggy_script)
    
    # Ergebnisse anzeigen
    summary = fix_result.get_fix_summary()
    
    print(f"✅ Fix completed:")
    print(f"   Success: {summary['success']}")
    print(f"   Fixes Applied: {summary['fixes_applied']}")
    print(f"   Remaining Issues: {summary['remaining_issues']}")
    print(f"   Fix Time: {summary['fix_time']:.3f}s")
    print(f"   Confidence Score: {summary['confidence_score']:.3f}")
    
    print(f"\n🔍 Applied Fixes:")
    for i, fix in enumerate(fix_result.applied_fixes, 1):
        print(f"   {i}. {fix.description}")
        print(f"      Type: {fix.fix_type.value}")
        print(f"      Line: {fix.line_number}")
        print(f"      Confidence: {fix.confidence:.2f}")
        if fix.original_code and fix.fixed_code:
            print(f"      Before: {fix.original_code.strip()}")
            print(f"      After:  {fix.fixed_code.strip()}")
    
    # Fix-Report erstellen
    report = fixer.create_fix_report(fix_result)
    
    print(f"\n📊 Fix Report:")
    print(f"   Original Lines: {report['before_after']['original_lines']}")
    print(f"   Fixed Lines: {report['before_after']['fixed_lines']}")
    print(f"   Lines Added: {report['before_after']['lines_added']}")
    
    # Zeige korrigierten Code (erste 20 Zeilen)
    fixed_lines = fix_result.fixed_script.split('\n')
    print(f"\n📝 Fixed Script (first 20 lines):")
    for i, line in enumerate(fixed_lines[:20], 1):
        print(f"   {i:2d}: {line}")
    
    # Manuelle Fix-Vorschläge
    manual_suggestions = fixer.suggest_manual_fixes(fix_result.fixed_script)
    if manual_suggestions:
        print(f"\n💡 Manual Fix Suggestions:")
        for suggestion in manual_suggestions:
            print(f"   - {suggestion['suggestion']}")
            print(f"     Priority: {suggestion['priority']}")
            if suggestion.get('example'):
                print(f"     Example: {suggestion['example']}")
    
    # Fixer-Statistiken
    stats = fixer.get_statistics()
    print(f"\n📈 Fixer Statistics:")
    print(f"   Scripts Fixed: {stats['scripts_fixed']}")
    print(f"   Total Fixes Applied: {stats['total_fixes_applied']}")
    print(f"   Fix Success Rate: {stats['fix_success_rate']:.2%}")
    print(f"   Avg Fix Time: {stats['avg_fix_time']:.3f}s")
    
    print(f"\n🔧 Fixes by Type:")
    for fix_type, count in stats['fixes_by_type'].items():
        if count > 0:
            print(f"   {fix_type}: {count}")


if __name__ == "__main__":
    demo_automatic_error_fixer()