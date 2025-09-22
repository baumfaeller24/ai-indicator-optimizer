#!/usr/bin/env python3
"""
🧩 BAUSTEIN A2 INTEGRATION TEST
Test der Ollama Vision Client Integration mit MEGA-DATASET Charts

Features:
- Test mit echten MEGA-DATASET Charts (250 Charts)
- Vision-Analyse von verschiedenen Timeframes
- Performance-Validierung mit 62.2M Ticks Daten
- Integration mit bestehender Chart-Pipeline
"""

import logging
from pathlib import Path
import time
from typing import List, Dict, Any
import json

# Import der Ollama Vision Client
from ai_indicator_optimizer.ai.ollama_vision_client import (
    OllamaVisionClient,
    create_ollama_vision_client,
    analyze_chart_with_ollama
)


def test_mega_dataset_chart_analysis():
    """
    🧩 Test MEGA-DATASET Chart-Analyse mit Ollama Vision
    """
    print("🧩 BAUSTEIN A2: MEGA-DATASET CHART ANALYSIS TEST")
    print("=" * 70)
    
    # Erstelle Vision Client
    vision_client = create_ollama_vision_client()
    
    # Suche MEGA-DATASET Charts
    charts_dir = Path("data/mega_pretraining")
    if not charts_dir.exists():
        print(f"❌ MEGA-DATASET Charts directory not found: {charts_dir}")
        return False
    
    chart_files = list(charts_dir.glob("mega_chart_*.png"))
    print(f"📊 Found {len(chart_files)} MEGA-DATASET charts")
    
    if len(chart_files) == 0:
        print("❌ No MEGA-DATASET charts found!")
        return False
    
    # Test verschiedene Timeframes
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    analysis_results = {}
    
    for timeframe in timeframes:
        print(f"\n📈 TESTING {timeframe.upper()} TIMEFRAME CHARTS")
        print("-" * 50)
        
        # Finde Charts für diesen Timeframe
        tf_charts = [f for f in chart_files if f"_{timeframe}_" in f.name]
        
        if not tf_charts:
            print(f"  ⚠️ No charts found for {timeframe}")
            continue
        
        # Teste erste 3 Charts pro Timeframe
        tf_results = []
        for i, chart_file in enumerate(tf_charts[:3]):
            try:
                print(f"  🔍 Analyzing {chart_file.name}...")
                
                start_time = time.time()
                
                # Führe Vision-Analyse durch
                analysis = vision_client.analyze_chart_image(
                    str(chart_file),
                    analysis_type="comprehensive"
                )
                
                analysis_time = time.time() - start_time
                
                # Bewerte Analyse-Qualität
                quality_score = assess_analysis_quality(analysis)
                
                result = {
                    "chart_file": chart_file.name,
                    "timeframe": timeframe,
                    "analysis_time": analysis_time,
                    "quality_score": quality_score,
                    "confidence": analysis.get("confidence_score", 0.0),
                    "patterns_found": len(analysis.get("patterns_identified", [])),
                    "insights_count": len(analysis.get("key_insights", [])),
                    "trend": analysis.get("trading_signals", {}).get("trend", "unknown"),
                    "recommendation": analysis.get("trading_signals", {}).get("recommendation", "hold"),
                    "success": "error" not in analysis
                }
                
                tf_results.append(result)
                
                print(f"    ✅ Success: {result['success']}")
                print(f"    📊 Quality: {quality_score:.2f}")
                print(f"    🎯 Confidence: {result['confidence']:.2f}")
                print(f"    📈 Trend: {result['trend']}")
                print(f"    💡 Patterns: {result['patterns_found']}")
                print(f"    ⏱️ Time: {analysis_time:.2f}s")
                
            except Exception as e:
                print(f"    ❌ Analysis failed: {e}")
                tf_results.append({
                    "chart_file": chart_file.name,
                    "timeframe": timeframe,
                    "success": False,
                    "error": str(e)
                })
        
        analysis_results[timeframe] = tf_results
    
    # Gesamtstatistiken
    print(f"\n📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 70)
    
    total_charts = sum(len(results) for results in analysis_results.values())
    successful_analyses = sum(
        len([r for r in results if r.get("success", False)]) 
        for results in analysis_results.values()
    )
    
    if total_charts > 0:
        success_rate = successful_analyses / total_charts
        
        # Durchschnittliche Metriken
        all_successful = [
            r for results in analysis_results.values() 
            for r in results if r.get("success", False)
        ]
        
        if all_successful:
            avg_analysis_time = sum(r["analysis_time"] for r in all_successful) / len(all_successful)
            avg_quality = sum(r["quality_score"] for r in all_successful) / len(all_successful)
            avg_confidence = sum(r["confidence"] for r in all_successful) / len(all_successful)
            avg_patterns = sum(r["patterns_found"] for r in all_successful) / len(all_successful)
            
            print(f"📈 Overall Performance:")
            print(f"  - Charts analyzed: {total_charts}")
            print(f"  - Success rate: {success_rate:.1%}")
            print(f"  - Average analysis time: {avg_analysis_time:.2f}s")
            print(f"  - Average quality score: {avg_quality:.2f}")
            print(f"  - Average confidence: {avg_confidence:.2f}")
            print(f"  - Average patterns found: {avg_patterns:.1f}")
            
            # Timeframe-spezifische Statistiken
            print(f"\n📊 Timeframe Performance:")
            for tf, results in analysis_results.items():
                successful = [r for r in results if r.get("success", False)]
                if successful:
                    tf_success_rate = len(successful) / len(results)
                    tf_avg_time = sum(r["analysis_time"] for r in successful) / len(successful)
                    tf_avg_quality = sum(r["quality_score"] for r in successful) / len(successful)
                    
                    print(f"  {tf.upper()}: {tf_success_rate:.1%} success, {tf_avg_time:.2f}s avg, {tf_avg_quality:.2f} quality")
    
    # Speichere detaillierte Ergebnisse
    results_file = "baustein_a2_mega_analysis_results.json"
    with open(results_file, 'w') as f:
        # Serialisierbare Version
        serializable_results = {}
        for tf, results in analysis_results.items():
            serializable_results[tf] = results
        
        json.dump({
            "test_timestamp": time.time(),
            "total_charts_analyzed": total_charts,
            "success_rate": success_rate if total_charts > 0 else 0,
            "timeframe_results": serializable_results,
            "mega_dataset_integration": True
        }, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    return success_rate > 0.7 if total_charts > 0 else False


def assess_analysis_quality(analysis: Dict[str, Any]) -> float:
    """
    Bewerte Qualität der Vision-Analyse
    
    Args:
        analysis: Vision-Analyse-Ergebnis
        
    Returns:
        Quality Score (0.0 - 1.0)
    """
    try:
        quality_factors = []
        
        # 1. Confidence Score
        confidence = analysis.get("confidence_score", 0.0)
        quality_factors.append(confidence)
        
        # 2. Anzahl identifizierter Patterns
        patterns_count = len(analysis.get("patterns_identified", []))
        pattern_score = min(patterns_count / 5.0, 1.0)  # Max 5 Patterns = 1.0
        quality_factors.append(pattern_score)
        
        # 3. Anzahl Key Insights
        insights_count = len(analysis.get("key_insights", []))
        insights_score = min(insights_count / 5.0, 1.0)  # Max 5 Insights = 1.0
        quality_factors.append(insights_score)
        
        # 4. Trading Signals Vollständigkeit
        signals = analysis.get("trading_signals", {})
        signals_score = 0.0
        if "trend" in signals and signals["trend"] != "neutral":
            signals_score += 0.5
        if "recommendation" in signals and signals["recommendation"] != "hold":
            signals_score += 0.5
        quality_factors.append(signals_score)
        
        # 5. Response Länge (Indikator für Detailgrad)
        raw_analysis = analysis.get("raw_analysis", "")
        length_score = min(len(raw_analysis) / 1000.0, 1.0)  # Max 1000 chars = 1.0
        quality_factors.append(length_score)
        
        # 6. Fehler-Check
        error_penalty = 0.0 if "error" not in analysis else -0.5
        
        # Gewichteter Durchschnitt
        weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]  # Confidence am wichtigsten
        weighted_score = sum(f * w for f, w in zip(quality_factors, weights)) + error_penalty
        
        return max(0.0, min(1.0, weighted_score))
        
    except Exception:
        return 0.0


def test_vision_performance_benchmarks():
    """
    Performance-Benchmarks für Vision-Analyse
    """
    print(f"\n🚀 VISION PERFORMANCE BENCHMARKS")
    print("-" * 50)
    
    vision_client = create_ollama_vision_client()
    
    # Performance-Tests
    performance_tests = [
        ("Quick Analysis", "patterns", 1),
        ("Comprehensive Analysis", "comprehensive", 1),
        ("Feature Extraction", "features", 1),
        ("Batch Processing", "comprehensive", 3)
    ]
    
    benchmark_results = []
    
    for test_name, analysis_type, chart_count in performance_tests:
        print(f"\n  🧪 {test_name}:")
        
        # Finde Test-Charts
        charts_dir = Path("data/mega_pretraining")
        chart_files = list(charts_dir.glob("mega_chart_*.png"))[:chart_count]
        
        if len(chart_files) < chart_count:
            print(f"    ⚠️ Not enough charts for test (need {chart_count}, found {len(chart_files)})")
            continue
        
        start_time = time.time()
        successful_analyses = 0
        
        for chart_file in chart_files:
            try:
                if analysis_type == "features":
                    result = vision_client.extract_visual_features(str(chart_file))
                else:
                    result = vision_client.analyze_chart_image(str(chart_file), analysis_type)
                
                if "error" not in result:
                    successful_analyses += 1
                    
            except Exception as e:
                print(f"    ❌ Chart analysis failed: {e}")
        
        total_time = time.time() - start_time
        
        benchmark_result = {
            "test_name": test_name,
            "analysis_type": analysis_type,
            "charts_processed": chart_count,
            "successful_analyses": successful_analyses,
            "total_time": total_time,
            "avg_time_per_chart": total_time / chart_count,
            "success_rate": successful_analyses / chart_count,
            "throughput_charts_per_minute": (chart_count / total_time) * 60
        }
        
        benchmark_results.append(benchmark_result)
        
        print(f"    📊 Results:")
        print(f"      - Success rate: {benchmark_result['success_rate']:.1%}")
        print(f"      - Avg time per chart: {benchmark_result['avg_time_per_chart']:.2f}s")
        print(f"      - Throughput: {benchmark_result['throughput_charts_per_minute']:.1f} charts/min")
    
    # Performance-Zusammenfassung
    if benchmark_results:
        print(f"\n📈 PERFORMANCE SUMMARY:")
        avg_success_rate = sum(r["success_rate"] for r in benchmark_results) / len(benchmark_results)
        avg_time_per_chart = sum(r["avg_time_per_chart"] for r in benchmark_results) / len(benchmark_results)
        max_throughput = max(r["throughput_charts_per_minute"] for r in benchmark_results)
        
        print(f"  - Average success rate: {avg_success_rate:.1%}")
        print(f"  - Average time per chart: {avg_time_per_chart:.2f}s")
        print(f"  - Maximum throughput: {max_throughput:.1f} charts/min")
    
    return benchmark_results


def main():
    """
    Hauptfunktion für Baustein A2 Integration Test
    """
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧩 BAUSTEIN A2: OLLAMA VISION CLIENT INTEGRATION TEST")
    print("=" * 80)
    print("Testing Vision-Capabilities with MEGA-DATASET (62.2M ticks, 250 charts)")
    print("=" * 80)
    
    # Test 1: MEGA-DATASET Chart Analysis
    chart_analysis_success = test_mega_dataset_chart_analysis()
    
    # Test 2: Performance Benchmarks
    benchmark_results = test_vision_performance_benchmarks()
    
    # Test 3: Vision Client Stats
    print(f"\n📊 VISION CLIENT STATISTICS")
    print("-" * 50)
    
    vision_client = create_ollama_vision_client()
    stats = vision_client.get_performance_stats()
    
    print(f"📈 Client Performance:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Gesamtergebnis
    print(f"\n🎯 BAUSTEIN A2 INTEGRATION TEST RESULTS:")
    print("=" * 60)
    
    if chart_analysis_success:
        print("✅ MEGA-DATASET Chart Analysis: PASSED")
    else:
        print("❌ MEGA-DATASET Chart Analysis: FAILED")
    
    if benchmark_results:
        print("✅ Performance Benchmarks: COMPLETED")
        avg_success = sum(r["success_rate"] for r in benchmark_results) / len(benchmark_results)
        print(f"   Average success rate: {avg_success:.1%}")
    else:
        print("⚠️ Performance Benchmarks: PARTIAL")
    
    overall_success = chart_analysis_success and len(benchmark_results) > 0
    
    if overall_success:
        print(f"\n🎉 BAUSTEIN A2 SUCCESSFULLY INTEGRATED!")
        print(f"Ollama Vision Client ready for multimodal analysis!")
    else:
        print(f"\n⚠️ BAUSTEIN A2 INTEGRATION ISSUES DETECTED")
        print(f"Check Ollama server and model availability")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)