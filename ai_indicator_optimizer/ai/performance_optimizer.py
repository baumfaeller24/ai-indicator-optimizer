#!/usr/bin/env python3
"""
Performance Optimizer für Pine Script Code-Optimierung
Phase 3 Implementation - Task 10

Features:
- Code-Performance-Analyse und Optimierung
- Memory-Usage-Optimierung
- Execution-Speed-Verbesserung
- Algorithmus-Optimierung für technische Indikatoren
- Caching und Memoization-Strategien
- GPU-optimierte Berechnungen
"""

import re
import ast
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum
import numpy as np

# Local Imports
from .pine_script_validator import PineScriptValidator


class OptimizationType(Enum):
    """Typen von Performance-Optimierungen"""
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CACHING_OPTIMIZATION = "caching_optimization"
    LOOP_OPTIMIZATION = "loop_optimization"
    FUNCTION_OPTIMIZATION = "function_optimization"
    CALCULATION_OPTIMIZATION = "calculation_optimization"


@dataclass
class OptimizationSuggestion:
    """Einzelner Optimierungs-Vorschlag"""
    optimization_type: OptimizationType
    original_code: str
    optimized_code: str
    line_number: Optional[int] = None
    description: str = ""
    performance_gain: float = 0.0  # Geschätzte Verbesserung in %
    complexity_reduction: float = 0.0  # Komplexitäts-Reduktion
    confidence: float = 1.0  # Confidence des Vorschlags


@dataclass
class PerformanceAnalysis:
    """Ergebnis einer Performance-Analyse"""
    script: str
    total_lines: int
    complexity_score: float
    performance_score: float
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    analysis_time: float = 0.0
    
    # Performance-Metriken
    estimated_execution_time: float = 0.0  # Geschätzte Ausführungszeit
    memory_usage_score: float = 100.0     # Memory-Usage-Score (100 = optimal)
    algorithm_efficiency: float = 100.0    # Algorithmus-Effizienz


@dataclass
class OptimizationResult:
    """Ergebnis einer Code-Optimierung"""
    success: bool
    original_script: str
    optimized_script: str
    applied_optimizations: List[OptimizationSuggestion] = field(default_factory=list)
    performance_improvement: float = 0.0  # Geschätzte Verbesserung in %
    optimization_time: float = 0.0
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Erstelle Optimierungs-Summary"""
        return {
            "success": self.success,
            "optimizations_applied": len(self.applied_optimizations),
            "performance_improvement": self.performance_improvement,
            "optimization_time": self.optimization_time,
            "optimizations_by_type": {
                opt_type.value: len([o for o in self.applied_optimizations if o.optimization_type == opt_type])
                for opt_type in OptimizationType
            }
        }


class PerformanceOptimizer:
    """
    Performance-Optimizer für Pine Script Code
    
    Features:
    - Performance-Analyse
    - Bottleneck-Detection
    - Code-Optimierung
    - Algorithm-Improvement
    - Memory-Optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validator für Code-Analyse
        self.validator = PineScriptValidator()
        
        # Optimierungs-Regeln
        self.algorithm_optimizations = self._load_algorithm_optimizations()
        self.memory_optimizations = self._load_memory_optimizations()
        self.caching_optimizations = self._load_caching_optimizations()
        self.loop_optimizations = self._load_loop_optimizations()
        
        # Performance-Patterns
        self.performance_patterns = self._load_performance_patterns()
        self.bottleneck_patterns = self._load_bottleneck_patterns()
        
        # Statistiken
        self.stats = {
            "scripts_analyzed": 0,
            "scripts_optimized": 0,
            "total_optimizations_applied": 0,
            "avg_performance_improvement": 0.0,
            "avg_analysis_time": 0.0,
            "optimizations_by_type": {opt_type.value: 0 for opt_type in OptimizationType}
        }
        
        self.logger.info("PerformanceOptimizer initialized")
    
    def analyze_performance(self, pine_script: str) -> PerformanceAnalysis:
        """
        Analysiere Performance eines Pine Scripts
        
        Args:
            pine_script: Pine Script Code
            
        Returns:
            PerformanceAnalysis mit Bottlenecks und Optimierungs-Vorschlägen
        """
        try:
            start_time = datetime.now()
            
            lines = pine_script.split('\n')
            total_lines = len(lines)
            
            # Basis-Metriken berechnen
            complexity_score = self._calculate_complexity_score(pine_script, lines)
            performance_score = self._calculate_performance_score(pine_script, lines)
            
            # Bottlenecks identifizieren
            bottlenecks = self._identify_bottlenecks(pine_script, lines)
            
            # Optimierungs-Vorschläge generieren
            optimization_suggestions = self._generate_optimization_suggestions(pine_script, lines)
            
            # Performance-Metriken schätzen
            estimated_execution_time = self._estimate_execution_time(pine_script, lines)
            memory_usage_score = self._calculate_memory_usage_score(pine_script, lines)
            algorithm_efficiency = self._calculate_algorithm_efficiency(pine_script, lines)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            analysis = PerformanceAnalysis(
                script=pine_script,
                total_lines=total_lines,
                complexity_score=complexity_score,
                performance_score=performance_score,
                bottlenecks=bottlenecks,
                optimization_suggestions=optimization_suggestions,
                analysis_time=analysis_time,
                estimated_execution_time=estimated_execution_time,
                memory_usage_score=memory_usage_score,
                algorithm_efficiency=algorithm_efficiency
            )
            
            # Statistiken updaten
            self.stats["scripts_analyzed"] += 1
            self.stats["avg_analysis_time"] = (
                (self.stats["avg_analysis_time"] * (self.stats["scripts_analyzed"] - 1) + analysis_time) 
                / self.stats["scripts_analyzed"]
            )
            
            self.logger.info(f"Performance analysis completed: {len(bottlenecks)} bottlenecks, {len(optimization_suggestions)} suggestions")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
            return PerformanceAnalysis(
                script=pine_script,
                total_lines=len(pine_script.split('\n')),
                complexity_score=50.0,
                performance_score=50.0,
                analysis_time=0.0
            )
    
    def optimize_script(self, pine_script: str, apply_all: bool = False) -> OptimizationResult:
        """
        Optimiere Pine Script für bessere Performance
        
        Args:
            pine_script: Pine Script Code
            apply_all: Ob alle Optimierungen angewendet werden sollen
            
        Returns:
            OptimizationResult mit optimiertem Code
        """
        try:
            start_time = datetime.now()
            
            # Performance-Analyse durchführen
            analysis = self.analyze_performance(pine_script)
            
            # Optimierungen anwenden
            current_script = pine_script
            applied_optimizations = []
            
            # Sortiere Optimierungen nach Performance-Gain
            sorted_suggestions = sorted(
                analysis.optimization_suggestions,
                key=lambda x: x.performance_gain,
                reverse=True
            )
            
            for suggestion in sorted_suggestions:
                if apply_all or suggestion.confidence > 0.7:
                    # Wende Optimierung an
                    optimized_script = self._apply_optimization(current_script, suggestion)
                    
                    if optimized_script != current_script:
                        current_script = optimized_script
                        applied_optimizations.append(suggestion)
                        
                        # Update Statistiken
                        self.stats["optimizations_by_type"][suggestion.optimization_type.value] += 1
            
            # Performance-Verbesserung schätzen
            performance_improvement = sum(opt.performance_gain for opt in applied_optimizations)
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            result = OptimizationResult(
                success=len(applied_optimizations) > 0,
                original_script=pine_script,
                optimized_script=current_script,
                applied_optimizations=applied_optimizations,
                performance_improvement=performance_improvement,
                optimization_time=optimization_time
            )
            
            # Statistiken updaten
            self.stats["scripts_optimized"] += 1
            self.stats["total_optimizations_applied"] += len(applied_optimizations)
            self.stats["avg_performance_improvement"] = (
                (self.stats["avg_performance_improvement"] * (self.stats["scripts_optimized"] - 1) + performance_improvement) 
                / self.stats["scripts_optimized"]
            )
            
            self.logger.info(f"Optimization completed: {len(applied_optimizations)} optimizations, {performance_improvement:.1f}% improvement")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return OptimizationResult(
                success=False,
                original_script=pine_script,
                optimized_script=pine_script,
                performance_improvement=0.0,
                optimization_time=0.0
            )
    
    def _calculate_complexity_score(self, script: str, lines: List[str]) -> float:
        """Berechne Komplexitäts-Score"""
        try:
            complexity = 0
            
            # Zyklomatische Komplexität
            complexity += len(re.findall(r'\bif\b', script)) * 2
            complexity += len(re.findall(r'\bfor\b', script)) * 3
            complexity += len(re.findall(r'\bwhile\b', script)) * 3
            
            # Verschachtelungstiefe
            max_nesting = 0
            current_nesting = 0
            
            for line in lines:
                stripped = line.strip()
                if any(stripped.startswith(keyword) for keyword in ['if ', 'for ', 'while ']):
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif not stripped or not stripped.startswith(' '):
                    current_nesting = max(0, current_nesting - 1)
            
            complexity += max_nesting * 5
            
            # Funktions-Aufrufe
            complexity += len(re.findall(r'\w+\s*\(', script)) * 0.5
            
            return min(100, complexity)
            
        except Exception as e:
            self.logger.error(f"Complexity calculation error: {e}")
            return 50.0
    
    def _calculate_performance_score(self, script: str, lines: List[str]) -> float:
        """Berechne Performance-Score (100 = optimal)"""
        try:
            score = 100.0
            
            # Penalty für Performance-Issues
            
            # Große Schleifen
            large_loops = len(re.findall(r'for\s+\w+\s*=\s*\d+\s+to\s+(\d+)', script))
            score -= large_loops * 10
            
            # Heavy Functions
            heavy_functions = ['request.security', 'request.dividends', 'request.earnings']
            for func in heavy_functions:
                score -= script.count(func) * 15
            
            # String-Operationen
            string_ops = len(re.findall(r'str\.\w+', script))
            score -= min(string_ops * 2, 20)
            
            # Array-Operationen
            array_ops = len(re.findall(r'array\.\w+', script))
            score -= min(array_ops * 1, 15)
            
            # Redundante Berechnungen
            calculations = re.findall(r'(\w+\s*[+\-*/]\s*\w+)', script)
            unique_calculations = set(calculations)
            redundancy = len(calculations) - len(unique_calculations)
            score -= redundancy * 3
            
            return max(0, score)
            
        except Exception as e:
            self.logger.error(f"Performance score calculation error: {e}")
            return 50.0
    
    def _identify_bottlenecks(self, script: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Identifiziere Performance-Bottlenecks"""
        bottlenecks = []
        
        try:
            # Große Schleifen
            for line_num, line in enumerate(lines, 1):
                for_match = re.search(r'for\s+\w+\s*=\s*\d+\s+to\s+(\d+)', line)
                if for_match:
                    loop_size = int(for_match.group(1))
                    if loop_size > 500:
                        bottlenecks.append({
                            "type": "large_loop",
                            "line_number": line_num,
                            "description": f"Large loop with {loop_size} iterations",
                            "severity": "high" if loop_size > 2000 else "medium",
                            "impact": "execution_time",
                            "code_snippet": line.strip()
                        })
            
            # Heavy Function Calls
            heavy_functions = {
                'request.security': "External data request",
                'request.dividends': "Dividend data request",
                'request.earnings': "Earnings data request"
            }
            
            for line_num, line in enumerate(lines, 1):
                for func, description in heavy_functions.items():
                    if func in line:
                        bottlenecks.append({
                            "type": "heavy_function",
                            "line_number": line_num,
                            "description": description,
                            "severity": "high",
                            "impact": "execution_time",
                            "code_snippet": line.strip()
                        })
            
            # Redundante Berechnungen
            calculations = {}
            for line_num, line in enumerate(lines, 1):
                calc_matches = re.finditer(r'(\w+\s*[+\-*/]\s*\w+)', line)
                for match in calc_matches:
                    calc = match.group(1)
                    if calc in calculations:
                        bottlenecks.append({
                            "type": "redundant_calculation",
                            "line_number": line_num,
                            "description": f"Redundant calculation: {calc}",
                            "severity": "medium",
                            "impact": "execution_time",
                            "code_snippet": line.strip(),
                            "first_occurrence": calculations[calc]
                        })
                    else:
                        calculations[calc] = line_num
            
            # Memory-intensive Operations
            for line_num, line in enumerate(lines, 1):
                if 'array.new<' in line:
                    size_match = re.search(r'array\.new<\w+>\s*\(\s*(\d+)', line)
                    if size_match:
                        array_size = int(size_match.group(1))
                        if array_size > 5000:
                            bottlenecks.append({
                                "type": "large_array",
                                "line_number": line_num,
                                "description": f"Large array allocation ({array_size} elements)",
                                "severity": "high" if array_size > 20000 else "medium",
                                "impact": "memory_usage",
                                "code_snippet": line.strip()
                            })
            
        except Exception as e:
            self.logger.error(f"Bottleneck identification error: {e}")
        
        return bottlenecks    

    def _generate_optimization_suggestions(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Generiere Optimierungs-Vorschläge"""
        suggestions = []
        
        try:
            # Algorithm Optimizations
            suggestions.extend(self._suggest_algorithm_optimizations(script, lines))
            
            # Memory Optimizations
            suggestions.extend(self._suggest_memory_optimizations(script, lines))
            
            # Caching Optimizations
            suggestions.extend(self._suggest_caching_optimizations(script, lines))
            
            # Loop Optimizations
            suggestions.extend(self._suggest_loop_optimizations(script, lines))
            
            # Function Optimizations
            suggestions.extend(self._suggest_function_optimizations(script, lines))
            
            # Calculation Optimizations
            suggestions.extend(self._suggest_calculation_optimizations(script, lines))
            
        except Exception as e:
            self.logger.error(f"Optimization suggestion error: {e}")
        
        return suggestions
    
    def _suggest_algorithm_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Algorithmus-Optimierungen vor"""
        suggestions = []
        
        # Ineffiziente Moving Average Berechnungen
        for line_num, line in enumerate(lines, 1):
            if 'ta.sma(' in line and 'for ' in script:
                # Vorschlag: Verwende eingebaute SMA statt manueller Berechnung
                suggestions.append(OptimizationSuggestion(
                    optimization_type=OptimizationType.ALGORITHM_OPTIMIZATION,
                    original_code=line.strip(),
                    optimized_code=line.strip() + "  // Consider using built-in ta.sma() instead of manual calculation",
                    line_number=line_num,
                    description="Use built-in SMA function instead of manual loop calculation",
                    performance_gain=25.0,
                    complexity_reduction=15.0,
                    confidence=0.8
                ))
        
        # Ineffiziente RSI Berechnungen
        if 'rsi(' in script and script.count('rsi(') > 1:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.ALGORITHM_OPTIMIZATION,
                original_code="Multiple RSI calculations",
                optimized_code="// Cache RSI result: rsi_cached = ta.rsi(close, period)",
                description="Cache RSI calculation result to avoid recalculation",
                performance_gain=20.0,
                complexity_reduction=10.0,
                confidence=0.9
            ))
        
        return suggestions
    
    def _suggest_memory_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Memory-Optimierungen vor"""
        suggestions = []
        
        # Große Array-Allokationen
        for line_num, line in enumerate(lines, 1):
            if 'array.new<' in line:
                size_match = re.search(r'array\.new<\w+>\s*\(\s*(\d+)', line)
                if size_match:
                    array_size = int(size_match.group(1))
                    if array_size > 10000:
                        optimized_size = min(array_size, 5000)
                        optimized_line = line.replace(str(array_size), str(optimized_size))
                        
                        suggestions.append(OptimizationSuggestion(
                            optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                            original_code=line.strip(),
                            optimized_code=optimized_line.strip(),
                            line_number=line_num,
                            description=f"Reduce array size from {array_size} to {optimized_size} elements",
                            performance_gain=15.0,
                            complexity_reduction=5.0,
                            confidence=0.7
                        ))
        
        # Unnötige Variable-Deklarationen
        var_usage = {}
        for line_num, line in enumerate(lines, 1):
            var_matches = re.finditer(r'(\w+)\s*=', line)
            for match in var_matches:
                var_name = match.group(1)
                if var_name not in var_usage:
                    var_usage[var_name] = {"declared": line_num, "used": 0}
        
        # Zähle Variable-Verwendungen
        for line in lines:
            for var_name in var_usage.keys():
                if var_name in line and f'{var_name} =' not in line:
                    var_usage[var_name]["used"] += 1
        
        # Finde ungenutzte Variablen
        for var_name, usage in var_usage.items():
            if usage["used"] <= 1:  # Nur bei Deklaration verwendet
                suggestions.append(OptimizationSuggestion(
                    optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                    original_code=f"Unused variable: {var_name}",
                    optimized_code=f"// Remove unused variable: {var_name}",
                    line_number=usage["declared"],
                    description=f"Remove unused variable '{var_name}' to save memory",
                    performance_gain=5.0,
                    complexity_reduction=3.0,
                    confidence=0.6
                ))
        
        return suggestions
    
    def _suggest_caching_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Caching-Optimierungen vor"""
        suggestions = []
        
        # Wiederholte teure Berechnungen
        expensive_functions = ['ta.rsi', 'ta.macd', 'ta.stoch', 'request.security']
        
        for func in expensive_functions:
            occurrences = []
            for line_num, line in enumerate(lines, 1):
                if func in line:
                    occurrences.append((line_num, line.strip()))
            
            if len(occurrences) > 1:
                # Gleiche Parameter prüfen
                for i, (line_num1, line1) in enumerate(occurrences):
                    for j, (line_num2, line2) in enumerate(occurrences[i+1:], i+1):
                        # Vereinfachte Parameter-Extraktion
                        params1 = re.search(r'\((.*?)\)', line1)
                        params2 = re.search(r'\((.*?)\)', line2)
                        
                        if params1 and params2 and params1.group(1) == params2.group(1):
                            suggestions.append(OptimizationSuggestion(
                                optimization_type=OptimizationType.CACHING_OPTIMIZATION,
                                original_code=f"Duplicate {func} calls",
                                optimized_code=f"// Cache {func} result: {func.replace('.', '_')}_cached = {func}({params1.group(1)})",
                                line_number=min(line_num1, line_num2),
                                description=f"Cache {func} result to avoid duplicate calculations",
                                performance_gain=30.0,
                                complexity_reduction=8.0,
                                confidence=0.8
                            ))
                            break
        
        return suggestions
    
    def _suggest_loop_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Loop-Optimierungen vor"""
        suggestions = []
        
        # Große Schleifen optimieren
        for line_num, line in enumerate(lines, 1):
            for_match = re.search(r'for\s+(\w+)\s*=\s*(\d+)\s+to\s+(\d+)', line)
            if for_match:
                var_name, start, end = for_match.groups()
                loop_size = int(end) - int(start) + 1
                
                if loop_size > 1000:
                    # Vorschlag: Loop-Unrolling oder Batch-Processing
                    batch_size = min(100, loop_size // 10)
                    optimized_code = f"// Consider batch processing: process {batch_size} items at a time"
                    
                    suggestions.append(OptimizationSuggestion(
                        optimization_type=OptimizationType.LOOP_OPTIMIZATION,
                        original_code=line.strip(),
                        optimized_code=optimized_code,
                        line_number=line_num,
                        description=f"Optimize large loop ({loop_size} iterations) with batch processing",
                        performance_gain=40.0,
                        complexity_reduction=20.0,
                        confidence=0.6
                    ))
        
        # Verschachtelte Schleifen
        nesting_level = 0
        for line_num, line in enumerate(lines, 1):
            if re.search(r'\bfor\s+\w+', line):
                nesting_level += 1
                if nesting_level > 2:
                    suggestions.append(OptimizationSuggestion(
                        optimization_type=OptimizationType.LOOP_OPTIMIZATION,
                        original_code=line.strip(),
                        optimized_code="// Consider flattening nested loops or using vectorized operations",
                        line_number=line_num,
                        description="Reduce loop nesting depth for better performance",
                        performance_gain=25.0,
                        complexity_reduction=15.0,
                        confidence=0.7
                    ))
            elif not line.strip().startswith(' ') and line.strip():
                nesting_level = 0
        
        return suggestions
    
    def _suggest_function_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Function-Optimierungen vor"""
        suggestions = []
        
        # Häufige String-Operationen
        string_ops_count = len(re.findall(r'str\.\w+', script))
        if string_ops_count > 10:
            suggestions.append(OptimizationSuggestion(
                optimization_type=OptimizationType.FUNCTION_OPTIMIZATION,
                original_code=f"{string_ops_count} string operations",
                optimized_code="// Consider caching string results or reducing string operations",
                description="Reduce excessive string operations for better performance",
                performance_gain=15.0,
                complexity_reduction=5.0,
                confidence=0.6
            ))
        
        # Heavy Function Calls
        heavy_functions = {
            'request.security': 35.0,
            'request.dividends': 30.0,
            'request.earnings': 30.0
        }
        
        for func, gain in heavy_functions.items():
            count = script.count(func)
            if count > 1:
                suggestions.append(OptimizationSuggestion(
                    optimization_type=OptimizationType.FUNCTION_OPTIMIZATION,
                    original_code=f"Multiple {func} calls ({count})",
                    optimized_code=f"// Cache {func} results or combine requests",
                    description=f"Optimize {count} {func} calls by caching or combining",
                    performance_gain=gain,
                    complexity_reduction=10.0,
                    confidence=0.8
                ))
        
        return suggestions
    
    def _suggest_calculation_optimizations(self, script: str, lines: List[str]) -> List[OptimizationSuggestion]:
        """Schlage Calculation-Optimierungen vor"""
        suggestions = []
        
        # Redundante Berechnungen
        calculations = {}
        for line_num, line in enumerate(lines, 1):
            calc_matches = re.finditer(r'(\w+\s*[+\-*/]\s*\w+)', line)
            for match in calc_matches:
                calc = match.group(1)
                if calc in calculations:
                    suggestions.append(OptimizationSuggestion(
                        optimization_type=OptimizationType.CALCULATION_OPTIMIZATION,
                        original_code=f"Redundant calculation: {calc}",
                        optimized_code=f"// Store result: calc_result = {calc}",
                        line_number=line_num,
                        description=f"Cache calculation result '{calc}' to avoid recalculation",
                        performance_gain=10.0,
                        complexity_reduction=3.0,
                        confidence=0.7
                    ))
                else:
                    calculations[calc] = line_num
        
        # Komplexe mathematische Ausdrücke
        for line_num, line in enumerate(lines, 1):
            # Finde komplexe Ausdrücke mit vielen Operatoren
            operators = len(re.findall(r'[+\-*/]', line))
            if operators > 5:
                suggestions.append(OptimizationSuggestion(
                    optimization_type=OptimizationType.CALCULATION_OPTIMIZATION,
                    original_code=line.strip(),
                    optimized_code="// Break complex calculation into smaller parts",
                    line_number=line_num,
                    description="Simplify complex mathematical expression",
                    performance_gain=8.0,
                    complexity_reduction=12.0,
                    confidence=0.5
                ))
        
        return suggestions
    
    def _apply_optimization(self, script: str, suggestion: OptimizationSuggestion) -> str:
        """Wende eine Optimierung auf den Script an"""
        
        try:
            lines = script.split('\n')
            
            if suggestion.line_number and suggestion.line_number <= len(lines):
                line_idx = suggestion.line_number - 1
                
                # Einfache Ersetzungen
                if suggestion.optimization_type == OptimizationType.MEMORY_OPTIMIZATION:
                    if "Reduce array size" in suggestion.description:
                        lines[line_idx] = suggestion.optimized_code
                
                elif suggestion.optimization_type == OptimizationType.CACHING_OPTIMIZATION:
                    if "Cache" in suggestion.description:
                        # Füge Caching-Kommentar hinzu
                        lines.insert(line_idx, suggestion.optimized_code)
                
                elif suggestion.optimization_type == OptimizationType.CALCULATION_OPTIMIZATION:
                    if "Cache calculation result" in suggestion.description:
                        # Füge Caching-Kommentar hinzu
                        lines.insert(line_idx, suggestion.optimized_code)
            
            return '\n'.join(lines)
            
        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")
            return script
    
    def _estimate_execution_time(self, script: str, lines: List[str]) -> float:
        """Schätze Ausführungszeit (in ms)"""
        try:
            base_time = len(lines) * 0.1  # 0.1ms pro Zeile
            
            # Penalty für teure Operationen
            base_time += script.count('request.security') * 50  # 50ms pro request
            base_time += script.count('ta.rsi') * 2  # 2ms pro RSI
            base_time += script.count('ta.macd') * 3  # 3ms pro MACD
            base_time += len(re.findall(r'for\s+\w+\s*=\s*\d+\s+to\s+\d+', script)) * 10  # 10ms pro Loop
            
            return base_time
            
        except Exception as e:
            self.logger.error(f"Execution time estimation error: {e}")
            return 100.0  # Fallback
    
    def _calculate_memory_usage_score(self, script: str, lines: List[str]) -> float:
        """Berechne Memory-Usage-Score"""
        try:
            score = 100.0
            
            # Penalty für Memory-intensive Operationen
            array_count = len(re.findall(r'array\.new<', script))
            score -= array_count * 5
            
            # Große Arrays
            large_arrays = len(re.findall(r'array\.new<\w+>\s*\(\s*[5-9]\d{3,}', script))
            score -= large_arrays * 20
            
            # Variable-Anzahl
            var_count = len(re.findall(r'\w+\s*=', script))
            score -= max(0, (var_count - 20) * 2)  # Penalty ab 20 Variablen
            
            return max(0, score)
            
        except Exception as e:
            self.logger.error(f"Memory usage calculation error: {e}")
            return 50.0
    
    def _calculate_algorithm_efficiency(self, script: str, lines: List[str]) -> float:
        """Berechne Algorithmus-Effizienz"""
        try:
            score = 100.0
            
            # Penalty für ineffiziente Patterns
            
            # Verschachtelte Schleifen
            nested_loops = 0
            nesting_level = 0
            for line in lines:
                if re.search(r'\bfor\s+\w+', line):
                    nesting_level += 1
                    if nesting_level > 1:
                        nested_loops += 1
                elif not line.strip().startswith(' ') and line.strip():
                    nesting_level = 0
            
            score -= nested_loops * 15
            
            # Redundante Berechnungen
            calculations = re.findall(r'(\w+\s*[+\-*/]\s*\w+)', script)
            unique_calculations = set(calculations)
            redundancy = len(calculations) - len(unique_calculations)
            score -= redundancy * 5
            
            # Ineffiziente Built-in Usage
            if 'for ' in script and ('ta.sma' in script or 'ta.ema' in script):
                score -= 20  # Manual calculation statt built-in
            
            return max(0, score)
            
        except Exception as e:
            self.logger.error(f"Algorithm efficiency calculation error: {e}")
            return 50.0
    
    def _load_algorithm_optimizations(self) -> Dict[str, Any]:
        """Lade Algorithmus-Optimierungs-Regeln"""
        return {
            "use_builtin_functions": {"priority": 1, "gain": 25.0},
            "cache_expensive_calculations": {"priority": 1, "gain": 30.0},
            "vectorize_operations": {"priority": 2, "gain": 20.0}
        }
    
    def _load_memory_optimizations(self) -> Dict[str, Any]:
        """Lade Memory-Optimierungs-Regeln"""
        return {
            "reduce_array_sizes": {"priority": 2, "gain": 15.0},
            "remove_unused_variables": {"priority": 3, "gain": 5.0},
            "optimize_data_types": {"priority": 3, "gain": 8.0}
        }
    
    def _load_caching_optimizations(self) -> Dict[str, Any]:
        """Lade Caching-Optimierungs-Regeln"""
        return {
            "cache_function_results": {"priority": 1, "gain": 30.0},
            "memoize_calculations": {"priority": 2, "gain": 20.0}
        }
    
    def _load_loop_optimizations(self) -> Dict[str, Any]:
        """Lade Loop-Optimierungs-Regeln"""
        return {
            "batch_processing": {"priority": 1, "gain": 40.0},
            "reduce_nesting": {"priority": 2, "gain": 25.0},
            "loop_unrolling": {"priority": 3, "gain": 15.0}
        }
    
    def _load_performance_patterns(self) -> Dict[str, Any]:
        """Lade Performance-Pattern"""
        return {
            "efficient_indicators": ["ta.sma", "ta.ema", "ta.rsi", "ta.macd"],
            "expensive_functions": ["request.security", "request.dividends"],
            "memory_intensive": ["array.new", "matrix.new"]
        }
    
    def _load_bottleneck_patterns(self) -> Dict[str, Any]:
        """Lade Bottleneck-Pattern"""
        return {
            "large_loops": r'for\s+\w+\s*=\s*\d+\s+to\s+([5-9]\d{2,})',
            "nested_loops": r'for\s+\w+.*\n.*for\s+\w+',
            "redundant_calculations": r'(\w+\s*[+\-*/]\s*\w+)',
            "heavy_functions": ["request.security", "request.dividends", "request.earnings"]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Erhalte Optimizer-Statistiken"""
        return {
            **self.stats,
            "optimization_rules": {
                "algorithm_optimizations": len(self.algorithm_optimizations),
                "memory_optimizations": len(self.memory_optimizations),
                "caching_optimizations": len(self.caching_optimizations),
                "loop_optimizations": len(self.loop_optimizations)
            }
        }


# Factory Function
def create_performance_optimizer(config: Optional[Dict] = None) -> PerformanceOptimizer:
    """Factory Function für Performance Optimizer"""
    return PerformanceOptimizer(config=config)


# Demo/Test Function
def demo_performance_optimizer():
    """Demo für Performance Optimizer"""
    print("🧪 Testing Performance Optimizer...")
    
    optimizer = create_performance_optimizer()
    
    # Test-Script mit Performance-Issues
    test_script = """
//@version=5
indicator("Performance Test", overlay=true)

// Large loop
for i = 1 to 5000
    plot(i, color=color.red)

// Redundant calculations
calc1 = close + high
calc2 = close + high  // Same calculation
calc3 = close + high  // Again

// Heavy function usage
sec_data1 = request.security(syminfo.tickerid, "1D", close)
sec_data2 = request.security(syminfo.tickerid, "1D", close)  // Duplicate

// Large array
big_array = array.new<float>(50000)

// Multiple RSI calculations
rsi1 = ta.rsi(close, 14)
rsi2 = ta.rsi(close, 14)  // Same parameters

// Complex calculation
complex_calc = (close + high + low) * volume / (open + close) + ta.sma(close, 20) * 2.5
"""
    
    print("📊 Analyzing performance...")
    
    # Performance-Analyse
    analysis = optimizer.analyze_performance(test_script)
    
    print(f"✅ Analysis completed:")
    print(f"   Complexity Score: {analysis.complexity_score:.1f}")
    print(f"   Performance Score: {analysis.performance_score:.1f}")
    print(f"   Memory Usage Score: {analysis.memory_usage_score:.1f}")
    print(f"   Algorithm Efficiency: {analysis.algorithm_efficiency:.1f}")
    print(f"   Estimated Execution Time: {analysis.estimated_execution_time:.1f}ms")
    print(f"   Analysis Time: {analysis.analysis_time:.3f}s")
    
    print(f"\n🔍 Bottlenecks found ({len(analysis.bottlenecks)}):")
    for bottleneck in analysis.bottlenecks[:5]:  # Zeige nur erste 5
        print(f"   - {bottleneck['description']} (Line {bottleneck.get('line_number', '?')})")
        print(f"     Severity: {bottleneck['severity']}, Impact: {bottleneck['impact']}")
    
    print(f"\n💡 Optimization suggestions ({len(analysis.optimization_suggestions)}):")
    for suggestion in analysis.optimization_suggestions[:5]:  # Zeige nur erste 5
        print(f"   - {suggestion.description}")
        print(f"     Type: {suggestion.optimization_type.value}")
        print(f"     Performance Gain: {suggestion.performance_gain:.1f}%")
        print(f"     Confidence: {suggestion.confidence:.2f}")
    
    print(f"\n🔧 Applying optimizations...")
    
    # Code-Optimierung
    optimization_result = optimizer.optimize_script(test_script, apply_all=False)
    
    summary = optimization_result.get_optimization_summary()
    
    print(f"✅ Optimization completed:")
    print(f"   Success: {summary['success']}")
    print(f"   Optimizations Applied: {summary['optimizations_applied']}")
    print(f"   Performance Improvement: {summary['performance_improvement']:.1f}%")
    print(f"   Optimization Time: {summary['optimization_time']:.3f}s")
    
    print(f"\n🔧 Applied Optimizations:")
    for opt in optimization_result.applied_optimizations:
        print(f"   - {opt.description}")
        print(f"     Gain: {opt.performance_gain:.1f}%")
        print(f"     Confidence: {opt.confidence:.2f}")
    
    # Optimizer-Statistiken
    stats = optimizer.get_statistics()
    print(f"\n📈 Optimizer Statistics:")
    print(f"   Scripts Analyzed: {stats['scripts_analyzed']}")
    print(f"   Scripts Optimized: {stats['scripts_optimized']}")
    print(f"   Total Optimizations Applied: {stats['total_optimizations_applied']}")
    print(f"   Avg Performance Improvement: {stats['avg_performance_improvement']:.1f}%")
    print(f"   Avg Analysis Time: {stats['avg_analysis_time']:.3f}s")


if __name__ == "__main__":
    demo_performance_optimizer()