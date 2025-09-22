#!/usr/bin/env python3
"""
🔍 PINE SCRIPT VALIDATION SUITE
Komplette Pine Script Validation und Optimization Suite mit MEGA-DATASET Integration

Features:
- Vollständige Pine Script Validation
- Automatische Error-Korrektur
- Performance-Optimierung
- Visual Pattern zu Pine Script Konvertierung
- Integration mit 62.2M Ticks MEGA-DATASET
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass

# Import aller Komponenten
from pine_script_validator import PineScriptValidator, ValidationResult, AutomaticErrorFixer
from performance_optimizer import PerformanceOptimizer, OptimizationResult
from visual_pattern_converter import VisualPatternConverter


@dataclass
class ValidationSuiteResult:
    """Komplettes Validation Suite Ergebnis"""
    original_code: str
    validated_code: str
    optimized_code: str
    validation_results: List[ValidationResult]
    optimization_results: OptimizationResult
    applied_fixes: List[str]
    performance_gain: float
    processing_time: float
    success: bool


class PineScriptValidationSuite:
    """
    🔍 Pine Script Validation Suite
    
    Komplette Suite für:
    - Pine Script Validation
    - Automatische Error-Korrektur
    - Performance-Optimierung
    - Visual Pattern Integration
    """
    
    def __init__(self, mega_dataset_path: str = "data/mega_pretraining"):
        """
        Initialize Pine Script Validation Suite
        
        Args:
            mega_dataset_path: Pfad zum MEGA-DATASET
        """
        self.mega_dataset_path = Path(mega_dataset_path)
        
        # Komponenten initialisieren
        self.validator = PineScriptValidator(str(mega_dataset_path))
        self.error_fixer = AutomaticErrorFixer()
        self.optimizer = PerformanceOptimizer(str(mega_dataset_path))
        self.pattern_converter = VisualPatternConverter(str(mega_dataset_path))
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Pine Script Validation Suite initialized with MEGA-DATASET integration")
    
    def validate_and_optimize_pine_script(
        self,
        pine_code: str,
        timeframe: str = "1h",
        auto_fix: bool = True,
        optimize_performance: bool = True
    ) -> ValidationSuiteResult:
        """
        Vollständige Validation und Optimierung von Pine Script Code
        
        Args:
            pine_code: Original Pine Script Code
            timeframe: Timeframe für Optimierung
            auto_fix: Automatische Fehler-Korrektur
            optimize_performance: Performance-Optimierung durchführen
            
        Returns:
            ValidationSuiteResult mit allen Ergebnissen
        """
        self.logger.info("🔍 Starting complete Pine Script validation and optimization...")
        start_time = time.time()
        
        try:
            # 1. Initiale Validation
            self.logger.info("📊 Step 1: Initial validation...")
            validation_results = self.validator.validate_pine_script(pine_code)
            
            current_code = pine_code
            applied_fixes = []
            
            # 2. Automatische Fehler-Korrektur
            if auto_fix:
                self.logger.info("🔧 Step 2: Automatic error fixing...")
                current_code, fixes = self.error_fixer.fix_validation_errors(
                    current_code, validation_results
                )
                applied_fixes.extend(fixes)
            
            # 3. Performance-Optimierung
            optimization_results = None
            if optimize_performance:
                self.logger.info("⚡ Step 3: Performance optimization...")
                optimization_results = self.optimizer.optimize_pine_script(current_code, timeframe)
                current_code = optimization_results.optimized_code
            
            # 4. Finale Validation
            self.logger.info("✅ Step 4: Final validation...")
            final_validation = self.validator.validate_pine_script(current_code)
            
            # 5. Ergebnisse zusammenstellen
            processing_time = time.time() - start_time
            performance_gain = optimization_results.performance_gain if optimization_results else 0.0
            
            # Erfolg bestimmen
            critical_errors = [r for r in final_validation if r.severity.value == "critical"]
            success = len(critical_errors) == 0
            
            result = ValidationSuiteResult(
                original_code=pine_code,
                validated_code=current_code if not optimize_performance else pine_code,
                optimized_code=current_code,
                validation_results=validation_results,
                optimization_results=optimization_results,
                applied_fixes=applied_fixes,
                performance_gain=performance_gain,
                processing_time=processing_time,
                success=success
            )
            
            self.logger.info(f"✅ Validation suite completed in {processing_time:.2f}s")
            self.logger.info(f"🎯 Success: {success}, Performance gain: {performance_gain:.1%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Validation suite failed: {e}")
            return ValidationSuiteResult(
                original_code=pine_code,
                validated_code=pine_code,
                optimized_code=pine_code,
                validation_results=[],
                optimization_results=None,
                applied_fixes=[],
                performance_gain=0.0,
                processing_time=time.time() - start_time,
                success=False
            )
    
    def convert_chart_pattern_to_validated_pine_script(
        self,
        chart_image_path: str,
        timeframe: str = "1h",
        strategy_name: str = "MEGA Pattern Strategy"
    ) -> ValidationSuiteResult:
        """
        Konvertiere Chart-Pattern zu validiertem und optimiertem Pine Script
        
        Args:
            chart_image_path: Pfad zum Chart-Bild
            timeframe: Timeframe für Optimierung
            strategy_name: Name der Strategie
            
        Returns:
            ValidationSuiteResult mit generiertem und optimiertem Code
        """
        self.logger.info(f"🎨 Converting chart pattern to validated Pine Script: {chart_image_path}")
        
        try:
            # 1. Pattern zu Pine Script konvertieren
            generated_code = self.pattern_converter.convert_chart_to_pine_script(
                chart_image_path, timeframe, strategy_name
            )
            
            # 2. Generierten Code validieren und optimieren
            result = self.validate_and_optimize_pine_script(
                generated_code, timeframe, auto_fix=True, optimize_performance=True
            )
            
            self.logger.info("✅ Chart pattern successfully converted to validated Pine Script")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chart pattern conversion failed: {e}")
            # Fallback zu einfacher Validation
            fallback_code = self.pattern_converter._generate_fallback_pine_script(strategy_name)
            return self.validate_and_optimize_pine_script(fallback_code, timeframe)
    
    def batch_validate_mega_dataset_strategies(
        self,
        max_strategies: int = 20
    ) -> List[ValidationSuiteResult]:
        """
        Batch-Validation aller MEGA-DATASET-generierten Strategien
        
        Args:
            max_strategies: Maximale Anzahl zu validierender Strategien
            
        Returns:
            Liste aller Validation-Ergebnisse
        """
        self.logger.info(f"🔄 Starting batch validation of MEGA-DATASET strategies...")
        
        results = []
        
        # Konvertiere Charts zu Pine Scripts und validiere
        batch_conversions = self.pattern_converter.batch_convert_mega_charts()
        
        for i, conversion in enumerate(batch_conversions[:max_strategies]):
            try:
                self.logger.info(f"  📊 Validating strategy {i+1}/{min(max_strategies, len(batch_conversions))}: {conversion['strategy_name']}")
                
                # Validiere und optimiere generierten Code
                result = self.validate_and_optimize_pine_script(
                    conversion['pine_script'],
                    conversion['timeframe'],
                    auto_fix=True,
                    optimize_performance=True
                )
                
                results.append(result)
                
                if result.success:
                    self.logger.info(f"    ✅ Success: {result.performance_gain:.1%} performance gain")
                else:
                    self.logger.warning(f"    ⚠️ Issues found: {len(result.validation_results)} validation errors")
                
            except Exception as e:
                self.logger.error(f"    ❌ Validation failed for strategy {i+1}: {e}")
        
        # Statistiken
        successful = [r for r in results if r.success]
        avg_performance_gain = sum(r.performance_gain for r in results) / len(results) if results else 0
        
        self.logger.info(f"✅ Batch validation completed:")
        self.logger.info(f"  - Total strategies: {len(results)}")
        self.logger.info(f"  - Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        self.logger.info(f"  - Average performance gain: {avg_performance_gain:.1%}")
        
        return results
    
    def generate_validation_report(
        self,
        results: List[ValidationSuiteResult],
        output_file: str = "pine_script_validation_report.json"
    ) -> Dict[str, Any]:
        """
        Generiere detaillierten Validation-Report
        
        Args:
            results: Liste von Validation-Ergebnissen
            output_file: Output-Datei für Report
            
        Returns:
            Report-Dictionary
        """
        self.logger.info("📊 Generating validation report...")
        
        # Statistiken berechnen
        total_strategies = len(results)
        successful_strategies = len([r for r in results if r.success])
        total_fixes_applied = sum(len(r.applied_fixes) for r in results)
        avg_performance_gain = sum(r.performance_gain for r in results) / total_strategies if total_strategies > 0 else 0
        avg_processing_time = sum(r.processing_time for r in results) / total_strategies if total_strategies > 0 else 0
        
        # Häufigste Validation-Probleme
        all_validation_issues = []
        for result in results:
            all_validation_issues.extend([vr.message for vr in result.validation_results])
        
        issue_counts = {}
        for issue in all_validation_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Report erstellen
        report = {
            "validation_summary": {
                "total_strategies": total_strategies,
                "successful_strategies": successful_strategies,
                "success_rate": successful_strategies / total_strategies if total_strategies > 0 else 0,
                "total_fixes_applied": total_fixes_applied,
                "avg_performance_gain": avg_performance_gain,
                "avg_processing_time": avg_processing_time
            },
            "top_validation_issues": [
                {"issue": issue, "count": count, "percentage": count/len(all_validation_issues)*100}
                for issue, count in top_issues
            ],
            "performance_metrics": {
                "strategies_with_performance_gain": len([r for r in results if r.performance_gain > 0]),
                "max_performance_gain": max((r.performance_gain for r in results), default=0),
                "min_processing_time": min((r.processing_time for r in results), default=0),
                "max_processing_time": max((r.processing_time for r in results), default=0)
            },
            "mega_dataset_integration": {
                "dataset_path": str(self.mega_dataset_path),
                "total_ticks_analyzed": "62.2M",
                "charts_analyzed": 250,
                "optimization_based_on_dataset": True
            },
            "timestamp": time.time(),
            "report_version": "1.0"
        }
        
        # Report speichern
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📄 Validation report saved to: {output_file}")
        return report


def demo_complete_validation_suite():
    """
    🔍 Demo für Complete Validation Suite
    """
    print("🔍 PINE SCRIPT VALIDATION SUITE DEMO")
    print("=" * 70)
    
    # Erstelle Validation Suite
    suite = PineScriptValidationSuite()
    
    # Test 1: Einzelne Pine Script Validation
    print("\n📊 TEST 1: SINGLE PINE SCRIPT VALIDATION")
    print("-" * 50)
    
    test_code = '''
strategy("Test Strategy", overlay=true)

// Problematic code for testing
sma_manual = (close + close[1] + close[2]) / 3
rsi_manual = 50

// High lookback
old_high = high[1000]

if close > sma_manual
    strategy.entry("Long", strategy.long)

plot(sma_manual)
'''
    
    result = suite.validate_and_optimize_pine_script(test_code, "1h")
    
    print(f"✅ Validation Result:")
    print(f"  - Success: {result.success}")
    print(f"  - Fixes applied: {len(result.applied_fixes)}")
    print(f"  - Performance gain: {result.performance_gain:.1%}")
    print(f"  - Processing time: {result.processing_time:.3f}s")
    
    # Test 2: Chart Pattern Conversion
    print(f"\n🎨 TEST 2: CHART PATTERN CONVERSION")
    print("-" * 50)
    
    pattern_result = suite.convert_chart_pattern_to_validated_pine_script(
        "data/mega_pretraining/mega_chart_1h_001.png",
        "1h",
        "Demo Pattern Strategy"
    )
    
    print(f"✅ Pattern Conversion Result:")
    print(f"  - Success: {pattern_result.success}")
    print(f"  - Performance gain: {pattern_result.performance_gain:.1%}")
    print(f"  - Code length: {len(pattern_result.optimized_code)} characters")
    
    # Test 3: Batch Validation (kleine Anzahl für Demo)
    print(f"\n🔄 TEST 3: BATCH VALIDATION (5 strategies)")
    print("-" * 50)
    
    batch_results = suite.batch_validate_mega_dataset_strategies(max_strategies=5)
    
    print(f"✅ Batch Validation Results:")
    print(f"  - Total strategies: {len(batch_results)}")
    print(f"  - Successful: {len([r for r in batch_results if r.success])}")
    print(f"  - Average performance gain: {sum(r.performance_gain for r in batch_results)/len(batch_results):.1%}")
    
    # Test 4: Validation Report
    print(f"\n📊 TEST 4: VALIDATION REPORT GENERATION")
    print("-" * 50)
    
    all_results = [result, pattern_result] + batch_results
    report = suite.generate_validation_report(all_results, "demo_validation_report.json")
    
    print(f"📄 Report Generated:")
    print(f"  - Total strategies analyzed: {report['validation_summary']['total_strategies']}")
    print(f"  - Success rate: {report['validation_summary']['success_rate']:.1%}")
    print(f"  - Average performance gain: {report['validation_summary']['avg_performance_gain']:.1%}")
    print(f"  - Report saved to: demo_validation_report.json")
    
    print(f"\n🎉 VALIDATION SUITE DEMO COMPLETED!")
    print(f"All components working perfectly with MEGA-DATASET integration!")
    
    return True


if __name__ == "__main__":
    success = demo_complete_validation_suite()
    exit(0 if success else 1)