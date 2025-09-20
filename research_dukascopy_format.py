#!/usr/bin/env python3
"""
Recherche: Was bietet Dukascopy wirklich?
Überprüfung der tatsächlichen Datenstruktur
"""

import requests
import struct
from datetime import datetime, timedelta
import pandas as pd

def research_dukascopy_data_format():
    """Recherchiere echte Dukascopy-Datenstruktur"""
    
    print("🔍 DUKASCOPY-DATENFORMAT RECHERCHE")
    print("="*50)
    
    # Teste echte Dukascopy-URL
    symbol = "EURUSD"
    date = datetime(2025, 9, 19)  # Gestern
    hour = 10  # 10 Uhr UTC
    
    # Dukascopy URL-Format
    year = date.year
    month = date.month - 1  # Dukascopy verwendet 0-basierte Monate
    day = date.day
    
    url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    
    print(f"📡 Teste Dukascopy-URL:")
    print(f"   {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Content-Length: {len(response.content)} bytes")
        
        if response.status_code == 200 and len(response.content) > 0:
            print("✅ Daten verfügbar!")
            
            # Analysiere Binärformat
            data = response.content
            
            # Dukascopy .bi5 Format (komprimiert)
            print(f"\n📊 BINÄRDATEN-ANALYSE:")
            print(f"   Dateigröße: {len(data)} bytes")
            print(f"   Erste 20 bytes (hex): {data[:20].hex()}")
            
            # Versuche zu dekomprimieren (LZMA)
            try:
                import lzma
                decompressed = lzma.decompress(data)
                print(f"   Dekomprimiert: {len(decompressed)} bytes")
                
                # Analysiere Tick-Struktur
                tick_size = 20  # 20 bytes pro Tick laut Dukascopy-Doku
                num_ticks = len(decompressed) // tick_size
                
                print(f"   Geschätzte Ticks: {num_ticks}")
                
                if num_ticks > 0:
                    # Parse ersten Tick
                    tick_bytes = decompressed[:tick_size]
                    
                    timestamp_ms = struct.unpack('>I', tick_bytes[0:4])[0]
                    ask_price_raw = struct.unpack('>I', tick_bytes[4:8])[0]
                    bid_price_raw = struct.unpack('>I', tick_bytes[8:12])[0]
                    ask_volume_raw = struct.unpack('>f', tick_bytes[12:16])[0]
                    bid_volume_raw = struct.unpack('>f', tick_bytes[16:20])[0]
                    
                    print(f"\n📋 ERSTER TICK:")
                    print(f"   Timestamp (ms): {timestamp_ms}")
                    print(f"   Ask Price (raw): {ask_price_raw}")
                    print(f"   Bid Price (raw): {bid_price_raw}")
                    print(f"   Ask Volume: {ask_volume_raw}")
                    print(f"   Bid Volume: {bid_volume_raw}")
                    
                    # Konvertiere Preise
                    point_value = 0.00001
                    ask_price = ask_price_raw * point_value
                    bid_price = bid_price_raw * point_value
                    
                    print(f"\n📊 KONVERTIERTE WERTE:")
                    print(f"   Ask Price: {ask_price:.5f}")
                    print(f"   Bid Price: {bid_price:.5f}")
                    print(f"   Spread: {(ask_price - bid_price):.5f}")
                    print(f"   Ask Volume: {ask_volume_raw:.3f}")
                    print(f"   Bid Volume: {bid_volume_raw:.3f}")
                    print(f"   Total Volume: {ask_volume_raw + bid_volume_raw:.3f}")
                    
                    # Analysiere Volume-Werte
                    if ask_volume_raw == 0 and bid_volume_raw == 0:
                        print("❌ PROBLEM: Volume ist 0 - möglicherweise nicht verfügbar!")
                    elif ask_volume_raw > 0 or bid_volume_raw > 0:
                        print("✅ Volume-Daten verfügbar!")
                    else:
                        print("⚠️ Unklare Volume-Daten")
                
            except Exception as e:
                print(f"❌ Dekomprimierung fehlgeschlagen: {e}")
                print("💡 Möglicherweise anderes Komprimierungsformat")
        
        elif response.status_code == 404:
            print("❌ Daten nicht verfügbar (404)")
            print("💡 Mögliche Gründe:")
            print("   - Wochenende/Feiertag")
            print("   - Zu aktueller Zeitpunkt")
            print("   - Symbol nicht verfügbar")
        
        else:
            print(f"❌ Unerwarteter Status: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Verbindungsfehler: {e}")
    
    # Teste verschiedene Zeitpunkte
    print(f"\n🕐 TESTE VERSCHIEDENE ZEITPUNKTE:")
    
    test_dates = [
        (datetime(2025, 9, 18), 14),  # Gestern, 14 Uhr
        (datetime(2025, 9, 17), 10),  # Vorgestern, 10 Uhr
        (datetime(2025, 9, 16), 16),  # Montag, 16 Uhr
    ]
    
    for test_date, test_hour in test_dates:
        year = test_date.year
        month = test_date.month - 1
        day = test_date.day
        
        test_url = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{test_hour:02d}h_ticks.bi5"
        
        try:
            response = requests.head(test_url, timeout=5)  # Nur Header
            print(f"   {test_date.date()} {test_hour}:00 → {response.status_code}")
        except:
            print(f"   {test_date.date()} {test_hour}:00 → Fehler")

def research_dukascopy_documentation():
    """Recherchiere offizielle Dukascopy-Dokumentation"""
    
    print(f"\n📚 DUKASCOPY-DOKUMENTATION:")
    print("="*40)
    
    print("""
🔍 OFFIZIELLE DUKASCOPY-INFORMATIONEN:

1. **Tick-Datenformat (.bi5):**
   - Komprimiert mit LZMA
   - 20 bytes pro Tick
   - Big-Endian Format

2. **Tick-Struktur:**
   - Bytes 0-3: Timestamp (Millisekunden)
   - Bytes 4-7: Ask Price (Integer, Point-Value)
   - Bytes 8-11: Bid Price (Integer, Point-Value)
   - Bytes 12-15: Ask Volume (Float)
   - Bytes 16-19: Bid Volume (Float)

3. **Volume-Information:**
   ❓ UNKLARHEIT: Ist Volume wirklich verfügbar?
   
   Mögliche Interpretationen:
   a) ✅ Echtes Handelsvolumen
   b) ❌ Tick-Count oder andere Metrik
   c) ⚠️ Nur für Premium-Accounts

4. **Forex-Besonderheit:**
   - Forex ist dezentraler Markt
   - Kein "echtes" Gesamtvolumen
   - Nur Volumen des jeweiligen Brokers/Feeds
""")

def test_volume_reality():
    """Teste ob Volume-Daten realistisch sind"""
    
    print(f"\n🧪 VOLUME-REALITÄTS-TEST:")
    print("="*30)
    
    print("""
🤔 KRITISCHE FRAGEN:

1. **Forex-Markt-Struktur:**
   - Forex ist OTC (Over-the-Counter)
   - Kein zentraler Marktplatz
   - Verschiedene Liquiditätsanbieter
   
   → Wie kann Dukascopy "echtes" Volumen haben?

2. **Dukascopy als Broker:**
   - Dukascopy ist ein Forex-Broker
   - Sieht nur eigene Kunden-Trades
   - Nicht das gesamte Marktvolumen
   
   → Volume = Nur Dukascopy-Kunden?

3. **Tick-Volume vs. Real-Volume:**
   - Tick-Volume = Anzahl Preisänderungen
   - Real-Volume = Tatsächlich gehandelte Menge
   
   → Was liefert Dukascopy wirklich?

🎯 WAHRSCHEINLICHE REALITÄT:
- Dukascopy-Volume = Tick-Count oder synthetisches Volume
- Nicht echtes Handelsvolumen des gesamten Marktes
- Trotzdem nützlich für Analyse (Aktivitäts-Indikator)
""")

def main():
    """Hauptfunktion"""
    
    research_dukascopy_data_format()
    research_dukascopy_documentation()
    test_volume_reality()
    
    print(f"\n🎯 FAZIT:")
    print("="*20)
    print("""
❓ UNSICHER: Ob Dukascopy echtes Handelsvolumen liefert

✅ SICHER: Dukascopy hat Volume-Felder im Datenformat

⚠️ WAHRSCHEINLICH: 
- Volume = Tick-basierte Aktivitäts-Metrik
- Nicht echtes Marktvolumen (Forex ist dezentral)
- Trotzdem nützlich für Analyse

💡 EMPFEHLUNG:
- Verwende Volume als Aktivitäts-Indikator
- Nicht als absolutes Handelsvolumen interpretieren
- Für AI-Features trotzdem wertvoll
""")

if __name__ == "__main__":
    main()