#!/usr/bin/env python3
"""
🧩 BAUSTEIN A2 TEST: Ollama Vision Client mit echtem Chart
Test der Vision-Capabilities mit echten Chart-Daten

Tests:
1. Chart-Bild-Analyse mit verschiedenen Analyse-Typen
2. Strukturierte Feature-Extraktion
3. Performance und Error-Handling
4. Integration mit bestehender Infrastruktur
"""

import logging
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd

# Import des neuen Vision Clients
from ai_indicator_optimizer.ai.ollama_vision_client import (
    OllamaVisionClient,
    create_ollama_vision_client,
    analyze_chart_with_ollama
)


def create_test_chart_image(save_path: str = "test_chart.png") -> str:
    """Erstelle ein Test-Chart-Bild für Vision-Tests"""
    
    # Generiere Test-OHLCV-Daten
    dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
    
    # Simuliere EUR/USD Kursdaten
    np.random.seed(42)
    base_price = 1.0950
    
    prices = []
    current_price = base_price
    
    for i in range(50):
        # Simuliere realistische Preisbewegungen
        change = np.random.normal(0, 0.002)  # 0.2% durchschnittliche Volatilität
        current_price += change
        
        # OHLC für den Tag
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.001))
        low_price = open_price - abs(np.random.normal(0, 0.001))
        close_price = open_price + np.random.normal(0, 0.0005)
        
        prices.append({
            'date': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': np.random.randint(1000, 5000)
        })
    
    df = pd.DataFrame(prices)
    
    # Erstelle Candlestick Chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Candlestick-Darstellung
    for i, row in df.iterrows():
        date = mdates.date2num(row['date'])
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # Farbe bestimmen
        color = 'green' if close_price >= open_price else 'red'
        
        # Wick (High-Low Linie)
        ax.plot([date, date], [low_price, high_price], color='black', linewidth=1)
        
        # Body (Open-Close Rechteck)
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = plt.Rectangle((date - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
    
    # Chart-Formatierung
    ax.set_title('EUR/USD Daily Chart - Test Data', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    # X-Achse formatieren
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Grid hinzufügen
    ax.grid(True, alpha=0.3)
    
    # Support/Resistance Linien hinzufügen
    support_level = df['low'].min() + 0.001
    resistance_level = df['high'].max() - 0.001
    
    ax.axhline(y=support_level, color='blue', linestyle='--', alpha=0.7, label='Support')
    ax.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.7, label='Resistance')
    
    # Moving Average hinzufügen
    df['sma_10'] = df['close'].rolling(window=10).mean()
    ax.plot(mdates.date2num(df['date']), df['sma_10'], color='orange', linewidth=2, label='SMA 10')
    
    ax.legend()
    plt.tight_layout()
    
    # Speichere Chart
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Test chart created: {save_path}")
    return save_path


def test_baustein_a2_comprehensive():
    """
    🧩 BAUSTEIN A2 COMPREHENSIVE TEST
    """
    
    print("🧩 BAUSTEIN A2 COMPREHENSIVE TEST: Ollama Vision Client")
    print("=" * 70)
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. Erstelle Vision Client
        print("\n1️⃣ CREATING OLLAMA VISION CLIENT...")
        client = create_ollama_vision_client()
        
        # Validiere Verbindung
        if not client._validate_connection():
            print("❌ Ollama connection failed - skipping vision tests")
            return False
        
        print("✅ Ollama Vision Client created and validated")
        
        # 2. Erstelle Test-Chart
        print("\n2️⃣ CREATING TEST CHART...")
        test_chart_path = create_test_chart_image("test_chart_baustein_a2.png")
        
        # 3. Teste verschiedene Analyse-Typen
        analysis_types = ["comprehensive", "patterns", "trends", "support_resistance"]
        
        results = {}
        
        for analysis_type in analysis_types:
            print(f"\n3️⃣ TESTING ANALYSIS TYPE: {analysis_type.upper()}")
            
            try:
                result = client.analyze_chart_image(test_chart_path, analysis_type)
                
                if "error" not in result:
                    print(f"✅ {analysis_type} analysis successful")
                    print(f"   - Confidence: {result.get('confidence_score', 0):.2f}")
                    print(f"   - Patterns found: {len(result.get('patterns_identified', []))}")
                    print(f"   - Processing time: {result.get('processing_time', 0):.2f}s")
                    print(f"   - Trend: {result.get('trading_signals', {}).get('trend', 'unknown')}")
                    print(f"   - Recommendation: {result.get('trading_signals', {}).get('recommendation', 'unknown')}")
                    
                    results[analysis_type] = result
                else:
                    print(f"❌ {analysis_type} analysis failed: {result['error']}")
                    results[analysis_type] = None
                    
            except Exception as e:
                print(f"❌ {analysis_type} analysis exception: {e}")
                results[analysis_type] = None
        
        # 4. Teste Visual Feature Extraction
        print(f"\n4️⃣ TESTING VISUAL FEATURE EXTRACTION...")
        
        try:
            visual_features = client.extract_visual_features(test_chart_path)
            
            if "error" not in visual_features:
                print("✅ Visual feature extraction successful")
                print(f"   - Feature vector length: {len(visual_features.get('feature_vector', []))}")
                print(f"   - Visual complexity: {visual_features.get('visual_complexity', 'unknown')}")
                print(f"   - Pattern strength: {visual_features.get('pattern_strength', 0):.3f}")
                
                results["visual_features"] = visual_features
            else:
                print(f"❌ Visual feature extraction failed: {visual_features['error']}")
                results["visual_features"] = None
                
        except Exception as e:
            print(f"❌ Visual feature extraction exception: {e}")
            results["visual_features"] = None
        
        # 5. Teste mit bestehendem Chart (falls vorhanden)
        existing_chart = Path("results/sample_chart_vision_test.png")
        if existing_chart.exists():
            print(f"\n5️⃣ TESTING WITH EXISTING CHART: {existing_chart}")
            
            try:
                existing_result = client.analyze_chart_image(str(existing_chart), "comprehensive")
                
                if "error" not in existing_result:
                    print("✅ Existing chart analysis successful")
                    print(f"   - Confidence: {existing_result.get('confidence_score', 0):.2f}")
                    print(f"   - Analysis quality: {existing_result.get('analysis_quality', 'unknown')}")
                    
                    results["existing_chart"] = existing_result
                else:
                    print(f"❌ Existing chart analysis failed: {existing_result['error']}")
                    
            except Exception as e:
                print(f"❌ Existing chart analysis exception: {e}")
        
        # 6. Performance Stats
        print(f"\n6️⃣ PERFORMANCE STATISTICS...")
        stats = client.get_performance_stats()
        
        print(f"   - Total requests: {stats['total_requests']}")
        print(f"   - Success rate: {stats['success_rate']:.1%}")
        print(f"   - Average inference time: {stats['average_inference_time']:.2f}s")
        print(f"   - Model: {stats['model']}")
        
        # 7. Teste Convenience Functions
        print(f"\n7️⃣ TESTING CONVENIENCE FUNCTIONS...")
        
        try:
            convenience_result = analyze_chart_with_ollama(test_chart_path, "trends")
            
            if "error" not in convenience_result:
                print("✅ Convenience function test successful")
                results["convenience_test"] = convenience_result
            else:
                print(f"❌ Convenience function test failed: {convenience_result['error']}")
                
        except Exception as e:
            print(f"❌ Convenience function test exception: {e}")
        
        # 8. Ergebnis-Zusammenfassung
        print(f"\n8️⃣ TEST SUMMARY...")
        
        successful_tests = sum(1 for result in results.values() if result is not None and "error" not in result)
        total_tests = len(results)
        
        print(f"   - Successful tests: {successful_tests}/{total_tests}")
        print(f"   - Success rate: {successful_tests/total_tests:.1%}")
        
        # Speichere Ergebnisse
        import json
        results_file = "baustein_a2_test_results.json"
        
        # Serialisierbare Version der Ergebnisse
        serializable_results = {}
        for key, value in results.items():
            if value is not None:
                # Entferne nicht-serialisierbare Objekte
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v for k, v in value.items() 
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    }
                else:
                    serializable_results[key] = str(value)
        
        with open(results_file, 'w') as f:
            json.dump({
                "test_timestamp": datetime.now().isoformat(),
                "baustein": "A2",
                "test_type": "comprehensive",
                "results": serializable_results,
                "performance_stats": stats,
                "success_rate": successful_tests/total_tests
            }, f, indent=2)
        
        print(f"   - Results saved to: {results_file}")
        
        # Erfolg bewerten
        if successful_tests >= total_tests * 0.8:  # 80% Erfolgsrate
            print("\n🎉 BAUSTEIN A2 TEST: SUCCESSFUL!")
            print("✅ Ollama Vision Client is ready for production use")
            return True
        else:
            print("\n⚠️ BAUSTEIN A2 TEST: PARTIALLY SUCCESSFUL")
            print("⚠️ Some vision capabilities may need adjustment")
            return False
            
    except Exception as e:
        print(f"\n❌ BAUSTEIN A2 TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    success = test_baustein_a2_comprehensive()
    
    if success:
        print("\n" + "="*70)
        print("🎉 BAUSTEIN A2 READY FOR INTEGRATION!")
        print("Next: Baustein A3 - Chart-Vision-Pipeline-Grundlagen")
        print("="*70)
        exit(0)
    else:
        print("\n" + "="*70)
        print("⚠️ BAUSTEIN A2 NEEDS REVIEW")
        print("Check Ollama setup and model availability")
        print("="*70)
        exit(1)