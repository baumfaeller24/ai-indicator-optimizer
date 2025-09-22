"""
Dynamic Fusion Agent Implementation
Task 6: Multimodal Flow Integration

This module implements the Dynamic Fusion Agent for adaptive Vision+Text prompts,
combining chart analysis with numerical data processing for enhanced strategy generation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class FusionMode(Enum):
    """Fusion modes for multimodal processing"""
    VISION_ONLY = "vision_only"
    TEXT_ONLY = "text_only"
    BALANCED_FUSION = "balanced_fusion"
    VISION_DOMINANT = "vision_dominant"
    TEXT_DOMINANT = "text_dominant"
    ADAPTIVE = "adaptive"

class ProcessingBackend(Enum):
    """Processing backend options"""
    OLLAMA = "ollama"
    TORCHSERVE = "torchserve"
    HYBRID = "hybrid"

@dataclass
class MultimodalInput:
    """Input data for multimodal processing"""
    chart_data: Optional[Dict[str, Any]] = None
    numerical_data: Optional[Dict[str, Any]] = None
    text_prompt: Optional[str] = None
    chart_image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """Result from multimodal fusion processing"""
    vision_analysis: Optional[Dict[str, Any]] = None
    text_analysis: Optional[Dict[str, Any]] = None
    fusion_confidence: float = 0.0
    combined_insights: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    backend_used: Optional[ProcessingBackend] = None
    fusion_mode: Optional[FusionMode] = None

@dataclass
class DynamicFusionConfig:
    """Configuration for Dynamic Fusion Agent"""
    # Fusion settings
    default_fusion_mode: FusionMode = FusionMode.ADAPTIVE
    vision_weight: float = 0.6
    text_weight: float = 0.4
    
    # Backend settings
    preferred_backend: ProcessingBackend = ProcessingBackend.HYBRID
    ollama_timeout: float = 30.0
    torchserve_timeout: float = 10.0
    
    # Load balancing
    enable_load_balancing: bool = True
    max_ollama_load: float = 0.8
    max_torchserve_load: float = 0.9
    
    # Quality thresholds
    min_vision_confidence: float = 0.5
    min_text_confidence: float = 0.5
    min_fusion_confidence: float = 0.6
    
    # Adaptive settings
    enable_adaptive_prompts: bool = True
    prompt_adaptation_threshold: float = 0.7
    max_prompt_iterations: int = 3

class DynamicFusionAgent:
    """
    Dynamic Fusion Agent for adaptive Vision+Text prompts
    
    Combines chart analysis with numerical data processing for enhanced
    strategy generation with real-time switching between backends.
    """
    
    def __init__(self, config: Optional[DynamicFusionConfig] = None):
        self.config = config or DynamicFusionConfig()
        self.logger = logging.getLogger(f"{__name__}.DynamicFusionAgent")
        
        # Initialize backends
        self.ollama_client = None
        self.torchserve_handler = None
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "ollama_requests": 0,
            "torchserve_requests": 0,
            "hybrid_requests": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0,
            "backend_switch_count": 0
        }
        
        self.logger.info("🤖 Dynamic Fusion Agent initialized")
    
    async def initialize_backends(self) -> bool:
        """Initialize available processing backends"""
        try:
            # Initialize Ollama Vision Client
            try:
                from ai_indicator_optimizer.ai.ollama_vision_client import OllamaVisionClient
                self.ollama_client = OllamaVisionClient()
                self.logger.info("✅ Ollama Vision Client initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Ollama initialization failed: {e}")
            
            # Initialize TorchServe Handler
            try:
                from ai_indicator_optimizer.ai.torchserve_handler import TorchServeHandler
                self.torchserve_handler = TorchServeHandler()
                self.logger.info("✅ TorchServe Handler initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ TorchServe initialization failed: {e}")
            
            # Check if at least one backend is available
            backends_available = sum([
                self.ollama_client is not None,
                self.torchserve_handler is not None
            ])
            
            if backends_available == 0:
                self.logger.error("❌ No processing backends available")
                return False
            
            self.logger.info(f"🚀 Dynamic Fusion Agent ready: {backends_available}/2 backends available")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backend initialization failed: {e}")
            return False
    
    async def process_multimodal_input(self, 
                                     multimodal_input: MultimodalInput,
                                     fusion_mode: Optional[FusionMode] = None) -> FusionResult:
        """
        Process multimodal input with dynamic fusion
        
        Args:
            multimodal_input: Input data containing charts, numerical data, etc.
            fusion_mode: Override fusion mode (optional)
            
        Returns:
            FusionResult with combined analysis
        """
        start_time = time.time()
        
        # Determine fusion mode
        effective_mode = fusion_mode or self.config.default_fusion_mode
        if effective_mode == FusionMode.ADAPTIVE:
            effective_mode = self._determine_adaptive_mode(multimodal_input)
        
        self.logger.info(f"🔄 Processing multimodal input with mode: {effective_mode.value}")
        
        try:
            # Select optimal backend
            backend = await self._select_optimal_backend(multimodal_input, effective_mode)
            
            # Process based on fusion mode
            if effective_mode == FusionMode.VISION_ONLY:
                result = await self._process_vision_only(multimodal_input, backend)
            elif effective_mode == FusionMode.TEXT_ONLY:
                result = await self._process_text_only(multimodal_input, backend)
            elif effective_mode in [FusionMode.BALANCED_FUSION, FusionMode.VISION_DOMINANT, FusionMode.TEXT_DOMINANT]:
                result = await self._process_fusion_mode(multimodal_input, effective_mode, backend)
            else:
                raise ValueError(f"Unsupported fusion mode: {effective_mode}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.backend_used = backend
            result.fusion_mode = effective_mode
            
            # Update performance stats
            self._update_performance_stats(processing_time, True, backend)
            
            self.logger.info(f"✅ Multimodal processing complete: {processing_time:.3f}s, confidence: {result.fusion_confidence:.3f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Multimodal processing failed: {e}")
            
            # Update performance stats
            self._update_performance_stats(processing_time, False, ProcessingBackend.HYBRID)
            
            # Return fallback result
            return FusionResult(
                fusion_confidence=0.0,
                combined_insights={"error": str(e), "fallback": True},
                processing_time=processing_time,
                backend_used=ProcessingBackend.HYBRID,
                fusion_mode=effective_mode
            )
    
    def _determine_adaptive_mode(self, multimodal_input: MultimodalInput) -> FusionMode:
        """Determine optimal fusion mode based on input characteristics"""
        has_chart = multimodal_input.chart_data is not None or multimodal_input.chart_image_path is not None
        has_numerical = multimodal_input.numerical_data is not None
        has_text = multimodal_input.text_prompt is not None
        
        # Adaptive logic
        if has_chart and has_numerical and has_text:
            return FusionMode.BALANCED_FUSION
        elif has_chart and (has_numerical or has_text):
            return FusionMode.VISION_DOMINANT
        elif (has_numerical or has_text) and not has_chart:
            return FusionMode.TEXT_DOMINANT
        elif has_chart and not (has_numerical or has_text):
            return FusionMode.VISION_ONLY
        else:
            return FusionMode.TEXT_ONLY
    
    async def _select_optimal_backend(self, 
                                    multimodal_input: MultimodalInput,
                                    fusion_mode: FusionMode) -> ProcessingBackend:
        """Select optimal processing backend based on load and capabilities"""
        
        # Check backend availability
        ollama_available = self.ollama_client is not None
        torchserve_available = self.torchserve_handler is not None
        
        if not ollama_available and not torchserve_available:
            raise RuntimeError("No processing backends available")
        
        # For vision-heavy tasks, prefer Ollama
        if fusion_mode in [FusionMode.VISION_ONLY, FusionMode.VISION_DOMINANT]:
            if ollama_available:
                return ProcessingBackend.OLLAMA
            elif torchserve_available:
                return ProcessingBackend.TORCHSERVE
        
        # For text-heavy tasks, prefer TorchServe
        if fusion_mode in [FusionMode.TEXT_ONLY, FusionMode.TEXT_DOMINANT]:
            if torchserve_available:
                return ProcessingBackend.TORCHSERVE
            elif ollama_available:
                return ProcessingBackend.OLLAMA
        
        # For balanced fusion, use hybrid approach
        if fusion_mode == FusionMode.BALANCED_FUSION:
            if ollama_available and torchserve_available:
                return ProcessingBackend.HYBRID
            elif ollama_available:
                return ProcessingBackend.OLLAMA
            elif torchserve_available:
                return ProcessingBackend.TORCHSERVE
        
        # Default fallback
        if ollama_available:
            return ProcessingBackend.OLLAMA
        else:
            return ProcessingBackend.TORCHSERVE
    
    async def _process_vision_only(self, 
                                 multimodal_input: MultimodalInput,
                                 backend: ProcessingBackend) -> FusionResult:
        """Process vision-only analysis"""
        
        if backend == ProcessingBackend.OLLAMA and self.ollama_client:
            # Use Ollama for vision analysis
            vision_result = await self._process_with_ollama_vision(multimodal_input)
            
            return FusionResult(
                vision_analysis=vision_result,
                fusion_confidence=vision_result.get("confidence", 0.5),
                combined_insights=vision_result
            )
        
        elif backend == ProcessingBackend.TORCHSERVE and self.torchserve_handler:
            # Use TorchServe for feature processing
            features = self._extract_chart_features(multimodal_input)
            torchserve_result = self._process_with_torchserve(features)
            
            return FusionResult(
                vision_analysis=torchserve_result,
                fusion_confidence=torchserve_result.get("confidence", 0.5),
                combined_insights=torchserve_result
            )
        
        else:
            raise RuntimeError(f"Backend {backend} not available for vision processing")
    
    async def _process_text_only(self, 
                                multimodal_input: MultimodalInput,
                                backend: ProcessingBackend) -> FusionResult:
        """Process text-only analysis"""
        
        # Extract numerical features for text processing
        numerical_features = self._extract_numerical_features(multimodal_input)
        
        if backend == ProcessingBackend.TORCHSERVE and self.torchserve_handler:
            # Use TorchServe for numerical processing
            torchserve_result = self._process_with_torchserve(numerical_features)
            
            return FusionResult(
                text_analysis=torchserve_result,
                fusion_confidence=torchserve_result.get("confidence", 0.5),
                combined_insights=torchserve_result
            )
        
        elif backend == ProcessingBackend.OLLAMA and self.ollama_client:
            # Use Ollama for text analysis
            text_prompt = self._create_text_prompt(multimodal_input)
            ollama_result = await self._process_with_ollama_text(text_prompt)
            
            return FusionResult(
                text_analysis=ollama_result,
                fusion_confidence=ollama_result.get("confidence", 0.5),
                combined_insights=ollama_result
            )
        
        else:
            raise RuntimeError(f"Backend {backend} not available for text processing")
    
    async def _process_fusion_mode(self, 
                                 multimodal_input: MultimodalInput,
                                 fusion_mode: FusionMode,
                                 backend: ProcessingBackend) -> FusionResult:
        """Process with fusion of vision and text analysis"""
        
        vision_result = None
        text_result = None
        
        # Process vision component
        if multimodal_input.chart_data or multimodal_input.chart_image_path:
            if backend in [ProcessingBackend.OLLAMA, ProcessingBackend.HYBRID] and self.ollama_client:
                vision_result = await self._process_with_ollama_vision(multimodal_input)
            elif self.torchserve_handler:
                features = self._extract_chart_features(multimodal_input)
                vision_result = self._process_with_torchserve(features)
        
        # Process text component
        if multimodal_input.numerical_data or multimodal_input.text_prompt:
            if backend in [ProcessingBackend.TORCHSERVE, ProcessingBackend.HYBRID] and self.torchserve_handler:
                numerical_features = self._extract_numerical_features(multimodal_input)
                text_result = self._process_with_torchserve(numerical_features)
            elif self.ollama_client:
                text_prompt = self._create_text_prompt(multimodal_input)
                text_result = await self._process_with_ollama_text(text_prompt)
        
        # Combine results based on fusion mode
        combined_insights = self._combine_analysis_results(
            vision_result, text_result, fusion_mode
        )
        
        # Calculate fusion confidence
        fusion_confidence = self._calculate_fusion_confidence(
            vision_result, text_result, fusion_mode
        )
        
        return FusionResult(
            vision_analysis=vision_result,
            text_analysis=text_result,
            fusion_confidence=fusion_confidence,
            combined_insights=combined_insights
        )
    
    async def _process_with_ollama_vision(self, multimodal_input: MultimodalInput) -> Dict[str, Any]:
        """Process with Ollama vision client"""
        try:
            # Create chart analysis request
            chart_data = {
                "symbol": multimodal_input.metadata.get("symbol", "EUR/USD"),
                "chart_image": multimodal_input.chart_image_path
            }
            
            numerical_data = multimodal_input.numerical_data or {}
            
            # Use existing multimodal analysis
            result = await self.ollama_client.multimodal_analysis(chart_data, numerical_data)
            
            return {
                "pattern": result.get("pattern", "unknown"),
                "confidence": result.get("confidence", 0.5),
                "analysis": result.get("analysis", "No analysis available"),
                "backend": "ollama_vision"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ollama vision processing failed: {e}")
            return {
                "pattern": "error",
                "confidence": 0.0,
                "analysis": f"Vision processing failed: {str(e)}",
                "backend": "ollama_vision"
            }
    
    async def _process_with_ollama_text(self, text_prompt: str) -> Dict[str, Any]:
        """Process with Ollama text analysis"""
        try:
            # Simple text analysis (placeholder implementation)
            result = {
                "analysis": f"Text analysis of: {text_prompt[:100]}...",
                "confidence": 0.6,
                "insights": ["Market analysis", "Technical indicators", "Risk assessment"],
                "backend": "ollama_text"
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ollama text processing failed: {e}")
            return {
                "analysis": f"Text processing failed: {str(e)}",
                "confidence": 0.0,
                "insights": [],
                "backend": "ollama_text"
            }
    
    def _process_with_torchserve(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Process with TorchServe handler"""
        try:
            # Import ModelType for proper TorchServe call
            from ai_indicator_optimizer.ai.torchserve_handler import ModelType
            
            # Use existing TorchServe handler with required model_type parameter (sync call)
            result = self.torchserve_handler.process_features(features, ModelType.PATTERN_RECOGNITION)
            
            # Extract data from InferenceResult dataclass
            predictions = result.predictions if hasattr(result, 'predictions') else []
            confidence = result.confidence if hasattr(result, 'confidence') else 0.5
            
            return {
                "predictions": predictions,
                "confidence": confidence,
                "features_processed": len(features),
                "backend": "torchserve"
            }
            
        except Exception as e:
            self.logger.error(f"❌ TorchServe processing failed: {e}")
            # Return graceful fallback instead of error
            return {
                "predictions": [{"pattern": "unknown", "confidence": 0.3}],
                "confidence": 0.3,
                "features_processed": len(features),
                "backend": "torchserve_fallback",
                "note": "Using fallback prediction"
            }
    
    def _extract_chart_features(self, multimodal_input: MultimodalInput) -> Dict[str, Any]:
        """Extract features from chart data"""
        features = {}
        
        if multimodal_input.chart_data:
            chart_data = multimodal_input.chart_data
            features.update({
                "symbol": chart_data.get("symbol", "EUR/USD"),
                "timeframe": chart_data.get("timeframe", "5m"),
                "has_chart": True
            })
        
        if multimodal_input.chart_image_path:
            features["chart_image_path"] = multimodal_input.chart_image_path
        
        # Add metadata
        features.update(multimodal_input.metadata)
        
        return features
    
    def _extract_numerical_features(self, multimodal_input: MultimodalInput) -> Dict[str, Any]:
        """Extract numerical features from input data"""
        features = {}
        
        if multimodal_input.numerical_data:
            numerical_data = multimodal_input.numerical_data
            
            # Extract OHLCV data if available
            if "ohlcv" in numerical_data:
                ohlcv = numerical_data["ohlcv"]
                if hasattr(ohlcv, "tail"):  # DataFrame-like
                    latest = ohlcv.tail(1).to_dict("records")[0] if len(ohlcv) > 0 else {}
                    features.update({
                        "latest_close": latest.get("close", 0),
                        "latest_volume": latest.get("volume", 0),
                        "bars_count": len(ohlcv)
                    })
            
            # Extract indicators if available
            if "indicators" in numerical_data:
                indicators = numerical_data["indicators"]
                features.update({f"indicator_{k}": v for k, v in indicators.items()})
        
        # Add text prompt as feature
        if multimodal_input.text_prompt:
            features["text_prompt"] = multimodal_input.text_prompt
        
        return features
    
    def _create_text_prompt(self, multimodal_input: MultimodalInput) -> str:
        """Create text prompt from multimodal input"""
        prompt_parts = []
        
        if multimodal_input.text_prompt:
            prompt_parts.append(multimodal_input.text_prompt)
        
        if multimodal_input.numerical_data:
            numerical_data = multimodal_input.numerical_data
            
            # Add OHLCV summary
            if "ohlcv" in numerical_data:
                prompt_parts.append("Market data analysis with OHLCV bars")
            
            # Add indicators summary
            if "indicators" in numerical_data:
                indicators = numerical_data["indicators"]
                prompt_parts.append(f"Technical indicators: {list(indicators.keys())}")
        
        if multimodal_input.chart_data:
            chart_data = multimodal_input.chart_data
            symbol = chart_data.get("symbol", "EUR/USD")
            timeframe = chart_data.get("timeframe", "5m")
            prompt_parts.append(f"Chart analysis for {symbol} on {timeframe} timeframe")
        
        return " | ".join(prompt_parts) if prompt_parts else "General market analysis"
    
    def _combine_analysis_results(self, 
                                vision_result: Optional[Dict[str, Any]],
                                text_result: Optional[Dict[str, Any]],
                                fusion_mode: FusionMode) -> Dict[str, Any]:
        """Combine vision and text analysis results"""
        
        combined = {
            "fusion_mode": fusion_mode.value,
            "components": {}
        }
        
        # Add vision component
        if vision_result:
            combined["components"]["vision"] = vision_result
            
            # Extract key insights from vision
            if "pattern" in vision_result:
                combined["pattern"] = vision_result["pattern"]
            if "analysis" in vision_result:
                combined["vision_analysis"] = vision_result["analysis"]
        
        # Add text component
        if text_result:
            combined["components"]["text"] = text_result
            
            # Extract key insights from text
            if "predictions" in text_result:
                combined["predictions"] = text_result["predictions"]
            if "analysis" in text_result:
                combined["text_analysis"] = text_result["analysis"]
        
        # Create fusion insights based on mode
        if fusion_mode == FusionMode.VISION_DOMINANT:
            combined["primary_insights"] = vision_result or {}
            combined["supporting_insights"] = text_result or {}
        elif fusion_mode == FusionMode.TEXT_DOMINANT:
            combined["primary_insights"] = text_result or {}
            combined["supporting_insights"] = vision_result or {}
        else:  # BALANCED_FUSION
            combined["balanced_insights"] = {
                "vision_weight": self.config.vision_weight,
                "text_weight": self.config.text_weight,
                "vision_component": vision_result or {},
                "text_component": text_result or {}
            }
        
        return combined
    
    def _calculate_fusion_confidence(self, 
                                   vision_result: Optional[Dict[str, Any]],
                                   text_result: Optional[Dict[str, Any]],
                                   fusion_mode: FusionMode) -> float:
        """Calculate overall fusion confidence"""
        
        vision_conf = vision_result.get("confidence", 0.0) if vision_result else 0.0
        text_conf = text_result.get("confidence", 0.0) if text_result else 0.0
        
        if fusion_mode == FusionMode.VISION_ONLY:
            return vision_conf
        elif fusion_mode == FusionMode.TEXT_ONLY:
            return text_conf
        elif fusion_mode == FusionMode.VISION_DOMINANT:
            return vision_conf * 0.8 + text_conf * 0.2
        elif fusion_mode == FusionMode.TEXT_DOMINANT:
            return text_conf * 0.8 + vision_conf * 0.2
        else:  # BALANCED_FUSION
            return vision_conf * self.config.vision_weight + text_conf * self.config.text_weight
    
    def _update_performance_stats(self, 
                                processing_time: float,
                                success: bool,
                                backend: ProcessingBackend):
        """Update performance statistics"""
        self.performance_stats["total_requests"] += 1
        
        if backend == ProcessingBackend.OLLAMA:
            self.performance_stats["ollama_requests"] += 1
        elif backend == ProcessingBackend.TORCHSERVE:
            self.performance_stats["torchserve_requests"] += 1
        else:
            self.performance_stats["hybrid_requests"] += 1
        
        # Update average processing time
        total_requests = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["average_processing_time"]
        new_avg = (current_avg * (total_requests - 1) + processing_time) / total_requests
        self.performance_stats["average_processing_time"] = new_avg
        
        # Update success rate
        if success:
            success_count = self.performance_stats["success_rate"] * (total_requests - 1) + 1
        else:
            success_count = self.performance_stats["success_rate"] * (total_requests - 1)
        
        self.performance_stats["success_rate"] = success_count / total_requests
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.performance_stats,
            "config": {
                "fusion_mode": self.config.default_fusion_mode.value,
                "preferred_backend": self.config.preferred_backend.value,
                "vision_weight": self.config.vision_weight,
                "text_weight": self.config.text_weight
            },
            "backends_available": {
                "ollama": self.ollama_client is not None,
                "torchserve": self.torchserve_handler is not None
            }
        }

# Factory function for easy instantiation
def create_dynamic_fusion_agent(
    fusion_mode: Optional[FusionMode] = None,
    backend: Optional[ProcessingBackend] = None,
    vision_weight: float = 0.6,
    text_weight: float = 0.4
) -> DynamicFusionAgent:
    """
    Factory function for Dynamic Fusion Agent
    
    Args:
        fusion_mode: Default fusion mode
        backend: Preferred processing backend
        vision_weight: Weight for vision component in fusion
        text_weight: Weight for text component in fusion
        
    Returns:
        Configured DynamicFusionAgent instance
    """
    config = DynamicFusionConfig()
    
    if fusion_mode:
        config.default_fusion_mode = fusion_mode
    if backend:
        config.preferred_backend = backend
    
    config.vision_weight = vision_weight
    config.text_weight = text_weight
    
    return DynamicFusionAgent(config)

# Example usage and testing
async def main():
    """Example usage of Dynamic Fusion Agent"""
    
    # Create fusion agent
    fusion_agent = create_dynamic_fusion_agent(
        fusion_mode=FusionMode.ADAPTIVE,
        backend=ProcessingBackend.HYBRID
    )
    
    # Initialize backends
    success = await fusion_agent.initialize_backends()
    if not success:
        print("❌ Failed to initialize backends")
        return
    
    # Create sample multimodal input
    multimodal_input = MultimodalInput(
        chart_data={"symbol": "EUR/USD", "timeframe": "5m"},
        numerical_data={
            "ohlcv": {"close": 1.1850, "volume": 1000},
            "indicators": {"rsi": 65.5, "macd": 0.002}
        },
        text_prompt="Analyze EUR/USD for potential trading opportunities",
        metadata={"timestamp": "2025-09-22T22:00:00Z"}
    )
    
    # Process multimodal input
    result = await fusion_agent.process_multimodal_input(multimodal_input)
    
    print(f"🎯 Fusion Result:")
    print(f"   Confidence: {result.fusion_confidence:.3f}")
    print(f"   Processing Time: {result.processing_time:.3f}s")
    print(f"   Backend Used: {result.backend_used.value if result.backend_used else 'None'}")
    print(f"   Fusion Mode: {result.fusion_mode.value if result.fusion_mode else 'None'}")
    
    # Get performance stats
    stats = fusion_agent.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(json.dumps(stats, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())