#!/usr/bin/env python3
"""
🧩 BAUSTEIN A2: Ollama Vision Client
Erweiterte Ollama-Integration um Vision-Capabilities für Chart-Analyse

Features:
- Chart-Bild-Analyse mit MiniCPM-4.1-8B Vision Model
- Pattern-Erkennung (Trends, Support/Resistance, Candlestick-Patterns)
- Strukturierte visuelle Feature-Extraktion
- Integration mit bestehender Chart-Generierung
- Error-Handling und Retry-Logic
"""

import requests
import base64
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime
from pathlib import Path
import io
from PIL import Image
import numpy as np


class OllamaVisionClient:
    """
    🧩 BAUSTEIN A2: Ollama Vision Client
    
    Erweitert bestehende Ollama-Integration um Vision-Capabilities:
    - Chart-Bild-Analyse mit MiniCPM-4.1-8B
    - Pattern-Erkennung und Trend-Analyse
    - Strukturierte Feature-Extraktion
    - Integration mit bestehender Infrastruktur
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "openbmb/minicpm4.1:latest",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Ollama Vision Client
        
        Args:
            base_url: Ollama Server URL
            model: Vision-fähiges Model (MiniCPM-4.1-8B)
            timeout: Request Timeout in Sekunden
            max_retries: Maximale Retry-Versuche
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # API Endpoints
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        
        # Performance Tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_inference_time = 0.0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"OllamaVisionClient initialized: {base_url} with model {model}")
        
        # Validiere Ollama-Verbindung
        self._validate_connection()
    
    def _validate_connection(self) -> bool:
        """Validiere Verbindung zu Ollama Server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model in model_names:
                    self.logger.info(f"✅ Ollama connection validated, model {self.model} available")
                    return True
                else:
                    self.logger.warning(f"⚠️ Model {self.model} not found. Available: {model_names}")
                    return False
            else:
                self.logger.error(f"❌ Ollama server not responding: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Ollama: {e}")
            return False
    
    def _encode_image_to_base64(self, image_data: Union[bytes, str, Path, Image.Image]) -> str:
        """
        Konvertiere Bild zu Base64 für Ollama API
        
        Args:
            image_data: Bild als bytes, Pfad, oder PIL Image
            
        Returns:
            Base64-kodiertes Bild
        """
        try:
            if isinstance(image_data, bytes):
                # Direkte bytes
                return base64.b64encode(image_data).decode('utf-8')
            
            elif isinstance(image_data, (str, Path)):
                # Dateipfad
                with open(image_data, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            
            elif isinstance(image_data, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                image_data.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
                
        except Exception as e:
            self.logger.error(f"Failed to encode image to base64: {e}")
            raise
    
    def _make_vision_request(
        self, 
        prompt: str, 
        image_base64: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Führe Vision-Request zu Ollama durch
        
        Args:
            prompt: Text-Prompt für Analyse
            image_base64: Base64-kodiertes Bild
            system_prompt: Optional System-Prompt
            
        Returns:
            Ollama Response Dictionary
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Request Payload für Vision
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Niedrig für konsistente Analyse
                "top_p": 0.9,
                "num_predict": 1000  # Ausreichend für detaillierte Analyse
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Retry-Logic
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.generate_endpoint,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Performance Tracking
                    inference_time = time.time() - start_time
                    self.total_inference_time += inference_time
                    self.successful_requests += 1
                    
                    self.logger.debug(f"Vision request successful: {inference_time:.2f}s")
                    return result
                else:
                    self.logger.warning(f"Ollama request failed: {response.status_code} - {response.text}")
                    last_exception = Exception(f"HTTP {response.status_code}: {response.text}")
                    
            except Exception as e:
                self.logger.warning(f"Vision request attempt {attempt + 1} failed: {e}")
                last_exception = e
                
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # Alle Versuche fehlgeschlagen
        self.failed_requests += 1
        self.logger.error(f"All vision request attempts failed. Last error: {last_exception}")
        raise last_exception
    
    def analyze_chart_image(
        self, 
        chart_image: Union[bytes, str, Path, Image.Image], 
        analysis_type: str = "comprehensive",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analysiere Chart-Bild mit Vision Model
        
        Args:
            chart_image: Chart-Bild (verschiedene Formate unterstützt)
            analysis_type: Art der Analyse ("comprehensive", "patterns", "trends", "support_resistance")
            custom_prompt: Optional custom Prompt
            
        Returns:
            Strukturierte Analyse-Ergebnisse
        """
        try:
            # Bild zu Base64 konvertieren
            image_base64 = self._encode_image_to_base64(chart_image)
            
            # Prompt basierend auf Analyse-Typ
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._get_analysis_prompt(analysis_type)
            
            # System Prompt für Trading-Kontext
            system_prompt = """You are an expert technical analyst specializing in forex chart analysis. 
            Analyze the provided candlestick chart and provide structured, actionable insights for trading decisions.
            Focus on identifying clear patterns, trends, and key levels with high confidence."""
            
            # Vision Request durchführen
            response = self._make_vision_request(prompt, image_base64, system_prompt)
            
            # Response verarbeiten
            analysis_result = self._process_vision_response(response, analysis_type)
            
            # Metadaten hinzufügen
            analysis_result.update({
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "analysis_type": analysis_type,
                "image_processed": True
            })
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Chart analysis failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "image_processed": False
            }
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Generiere Prompt basierend auf Analyse-Typ"""
        
        prompts = {
            "comprehensive": """
            Analyze this forex candlestick chart comprehensively. Provide:
            1. Overall trend direction (bullish/bearish/sideways)
            2. Key candlestick patterns identified
            3. Support and resistance levels
            4. Volume analysis (if visible)
            5. Momentum indicators assessment
            6. Trading recommendation with confidence level
            
            Format your response as structured analysis with clear sections.
            """,
            
            "patterns": """
            Focus specifically on candlestick patterns in this chart. Identify:
            1. Reversal patterns (doji, hammer, shooting star, engulfing, etc.)
            2. Continuation patterns (flags, pennants, triangles)
            3. Pattern completion and reliability
            4. Entry/exit points based on patterns
            
            Provide confidence scores for each identified pattern.
            """,
            
            "trends": """
            Analyze the trend characteristics of this chart:
            1. Primary trend direction and strength
            2. Trend line analysis
            3. Moving average alignment (if visible)
            4. Trend reversal signals
            5. Momentum divergences
            
            Rate trend strength from 1-10 and provide trend continuation probability.
            """,
            
            "support_resistance": """
            Identify key support and resistance levels:
            1. Horizontal support/resistance levels
            2. Dynamic support/resistance (trend lines)
            3. Psychological levels (round numbers)
            4. Previous highs/lows significance
            5. Volume confirmation at key levels
            
            Provide exact price levels and strength ratings.
            """
        }
        
        return prompts.get(analysis_type, prompts["comprehensive"])
    
    def _process_vision_response(self, response: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Verarbeite Ollama Vision Response zu strukturierten Daten
        
        Args:
            response: Raw Ollama Response
            analysis_type: Typ der durchgeführten Analyse
            
        Returns:
            Strukturierte Analyse-Ergebnisse
        """
        try:
            # Extrahiere Text-Response
            raw_text = response.get("response", "")
            
            # Basis-Struktur
            processed_result = {
                "raw_analysis": raw_text,
                "confidence_score": 0.0,
                "key_insights": [],
                "trading_signals": {},
                "technical_levels": {},
                "patterns_identified": [],
                "risk_assessment": "medium"
            }
            
            # Einfache Text-Parsing für strukturierte Extraktion
            # (Kann später durch NLP-Pipeline erweitert werden)
            
            # Confidence Score extrahieren (falls im Text erwähnt)
            confidence_keywords = ["confidence", "certain", "likely", "probable"]
            for keyword in confidence_keywords:
                if keyword in raw_text.lower():
                    # Einfache Heuristik für Confidence
                    if "high" in raw_text.lower():
                        processed_result["confidence_score"] = 0.8
                    elif "medium" in raw_text.lower():
                        processed_result["confidence_score"] = 0.6
                    elif "low" in raw_text.lower():
                        processed_result["confidence_score"] = 0.4
                    else:
                        processed_result["confidence_score"] = 0.5
                    break
            
            # Trading Signals extrahieren
            if "bullish" in raw_text.lower():
                processed_result["trading_signals"]["trend"] = "bullish"
            elif "bearish" in raw_text.lower():
                processed_result["trading_signals"]["trend"] = "bearish"
            else:
                processed_result["trading_signals"]["trend"] = "neutral"
            
            # Buy/Sell Signale
            if "buy" in raw_text.lower() or "long" in raw_text.lower():
                processed_result["trading_signals"]["recommendation"] = "buy"
            elif "sell" in raw_text.lower() or "short" in raw_text.lower():
                processed_result["trading_signals"]["recommendation"] = "sell"
            else:
                processed_result["trading_signals"]["recommendation"] = "hold"
            
            # Key Insights extrahieren (einfache Satz-Segmentierung)
            sentences = [s.strip() for s in raw_text.split('.') if len(s.strip()) > 20]
            processed_result["key_insights"] = sentences[:5]  # Top 5 Insights
            
            # Pattern-Erkennung
            common_patterns = [
                "doji", "hammer", "shooting star", "engulfing", "harami",
                "triangle", "flag", "pennant", "head and shoulders",
                "double top", "double bottom", "support", "resistance"
            ]
            
            identified_patterns = []
            for pattern in common_patterns:
                if pattern in raw_text.lower():
                    identified_patterns.append(pattern)
            
            processed_result["patterns_identified"] = identified_patterns
            
            # Zusätzliche Metadaten
            processed_result.update({
                "processing_time": response.get("total_duration", 0) / 1e9,  # Convert to seconds
                "tokens_generated": len(raw_text.split()),
                "analysis_quality": "high" if len(sentences) > 3 else "medium"
            })
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Failed to process vision response: {e}")
            return {
                "error": f"Response processing failed: {e}",
                "raw_analysis": response.get("response", ""),
                "confidence_score": 0.0
            }
    
    def extract_visual_features(
        self, 
        chart_image: Union[bytes, str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """
        Extrahiere strukturierte visuelle Features aus Chart
        
        Args:
            chart_image: Chart-Bild
            
        Returns:
            Dictionary mit visuellen Features
        """
        try:
            # Führe spezifische Feature-Extraktion durch
            feature_prompt = """
            Extract specific visual features from this chart for algorithmic analysis:
            
            1. Candlestick Statistics:
               - Number of bullish vs bearish candles
               - Average body size relative to total range
               - Wick-to-body ratios
               
            2. Price Action Features:
               - Highest and lowest visible prices
               - Price volatility assessment
               - Gap analysis
               
            3. Volume Features (if visible):
               - Volume trend correlation with price
               - Volume spikes identification
               
            4. Technical Levels:
               - Clear support/resistance levels with prices
               - Trend line angles and strength
               
            Provide numerical values where possible for algorithmic processing.
            """
            
            analysis = self.analyze_chart_image(
                chart_image, 
                analysis_type="features",
                custom_prompt=feature_prompt
            )
            
            # Erweitere um zusätzliche Feature-Extraktion
            visual_features = {
                "chart_analysis": analysis,
                "feature_vector": self._create_feature_vector(analysis),
                "visual_complexity": self._assess_visual_complexity(analysis),
                "pattern_strength": self._calculate_pattern_strength(analysis)
            }
            
            return visual_features
            
        except Exception as e:
            self.logger.error(f"Visual feature extraction failed: {e}")
            return {"error": str(e), "features_extracted": False}
    
    def _create_feature_vector(self, analysis: Dict[str, Any]) -> List[float]:
        """Erstelle numerischen Feature-Vektor aus Analyse"""
        try:
            # Basis Feature-Vektor (kann erweitert werden)
            features = [
                analysis.get("confidence_score", 0.0),
                1.0 if analysis.get("trading_signals", {}).get("trend") == "bullish" else 0.0,
                1.0 if analysis.get("trading_signals", {}).get("trend") == "bearish" else 0.0,
                len(analysis.get("patterns_identified", [])) / 10.0,  # Normalisiert
                len(analysis.get("key_insights", [])) / 10.0,  # Normalisiert
                analysis.get("processing_time", 0.0) / 30.0,  # Normalisiert auf 30s max
            ]
            
            return features
            
        except Exception:
            return [0.0] * 6  # Default Feature-Vektor
    
    def _assess_visual_complexity(self, analysis: Dict[str, Any]) -> str:
        """Bewerte visuelle Komplexität des Charts"""
        try:
            patterns_count = len(analysis.get("patterns_identified", []))
            insights_count = len(analysis.get("key_insights", []))
            
            complexity_score = patterns_count + insights_count
            
            if complexity_score >= 8:
                return "high"
            elif complexity_score >= 4:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    def _calculate_pattern_strength(self, analysis: Dict[str, Any]) -> float:
        """Berechne Gesamtstärke der identifizierten Patterns"""
        try:
            confidence = analysis.get("confidence_score", 0.0)
            patterns_count = len(analysis.get("patterns_identified", []))
            
            # Einfache Heuristik für Pattern-Stärke
            pattern_strength = (confidence * 0.7) + (min(patterns_count / 5.0, 1.0) * 0.3)
            
            return round(pattern_strength, 3)
            
        except Exception:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gebe Performance-Statistiken zurück"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.total_requests),
            "average_inference_time": self.total_inference_time / max(1, self.successful_requests),
            "model": self.model,
            "base_url": self.base_url
        }
    
    def test_vision_capabilities(self) -> Dict[str, Any]:
        """
        Teste Vision-Capabilities mit einem einfachen Test-Bild
        
        Returns:
            Test-Ergebnisse
        """
        try:
            # Erstelle einfaches Test-Chart-Bild
            test_image = Image.new('RGB', (400, 300), color='white')
            
            # Einfacher Test-Prompt
            test_prompt = "Describe what you see in this image. Is this a financial chart?"
            
            # Führe Test durch
            result = self.analyze_chart_image(
                test_image,
                analysis_type="comprehensive",
                custom_prompt=test_prompt
            )
            
            return {
                "test_successful": "error" not in result,
                "model_responsive": True,
                "vision_working": "image" in result.get("raw_analysis", "").lower(),
                "test_result": result
            }
            
        except Exception as e:
            return {
                "test_successful": False,
                "model_responsive": False,
                "vision_working": False,
                "error": str(e)
            }


# Convenience Functions
def create_ollama_vision_client(
    base_url: str = "http://localhost:11434",
    model: str = "openbmb/minicpm4.1:latest"
) -> OllamaVisionClient:
    """Erstelle OllamaVisionClient mit Standard-Konfiguration"""
    return OllamaVisionClient(base_url=base_url, model=model)


def analyze_chart_with_ollama(
    chart_image: Union[bytes, str, Path, Image.Image],
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """Convenience-Funktion für Chart-Analyse"""
    client = create_ollama_vision_client()
    return client.analyze_chart_image(chart_image, analysis_type)


if __name__ == "__main__":
    # Test der OllamaVisionClient
    logging.basicConfig(level=logging.INFO)
    
    print("🧩 BAUSTEIN A2 TEST: OllamaVisionClient")
    print("=" * 50)
    
    # Erstelle Client
    client = create_ollama_vision_client()
    
    # Teste Vision-Capabilities
    test_results = client.test_vision_capabilities()
    print(f"Vision Test Results: {test_results}")
    
    # Performance Stats
    stats = client.get_performance_stats()
    print(f"Performance Stats: {stats}")
    
    print("🎉 BAUSTEIN A2 TEST COMPLETED")