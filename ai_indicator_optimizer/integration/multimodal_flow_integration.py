#!/usr/bin/env python3
"""
Multimodal Flow Integration - Task 6
Top-5-Strategien-Ranking-System (Baustein C2)

Features:
- Dynamic-Fusion-Agent für adaptive Vision+Text-Prompts
- Chart-to-Strategy-Pipeline mit Ollama Vision Client Integration
- Feature-JSON-Processing mit TorchServe Handler (30,933 req/s)
- Multimodal-Confidence-Scoring für kombinierte Vision+Text-Analyse
- Real-time-Switching zwischen Ollama und TorchServe basierend auf Load
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import psutil
import torch

# Import existing components
from ..ai.ollama_vision_client import OllamaVisionClient
from ..ai.torchserve_handler import TorchServeHandler
from ..ai.enhanced_feature_extractor import EnhancedFeatureExtractor
from ..ai.confidence_scoring import EnhancedConfidenceScorer
from ..control.live_control_manager import LiveControlManager


@dataclass
class MultimodalFlowConfig:
    """Configuration for Multimodal Flow Integration"""
    
    # Dynamic Fusion Settings
    fusion_mode: str = "adaptive"  # adaptive, vision_priority, text_priority, balanced
    confidence_threshold: float = 0.7
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "vision": 0.6,
        "text": 0.4,
        "technical": 0.3,
        "pattern": 0.5
    })
    
    # Load Balancing Settings
    ollama_max_load: float = 0.8  # Switch to TorchServe above 80% load
    torchserve_max_load: float = 0.9  # Switch to Ollama above 90% load
    load_check_interval: float = 1.0  # seconds
    
    # Performance Settings
    max_concurrent_requests: int = 32  # Utilize all CPU cores
    request_timeout: float = 30.0
    batch_size: int = 8
    
    # Quality Settings
    min_vision_confidence: float = 0.5
    min_text_confidence: float = 0.6
    min_fusion_confidence: float = 0.65
    
    # Chart Processing Settings
    chart_resolution: Tuple[int, int] = (1200, 800)
    supported_formats: List[str] = field(default_factory=lambda: ["PNG", "JPG", "JPEG"])
    
    # Output Settings
    output_dir: str = "data/multimodal_flow"
    save_intermediate_results: bool = True
    log_performance_metrics: bool = True


@dataclass
class MultimodalAnalysisResult:
    """Result from multimodal analysis"""
    
    # Input Information
    chart_path: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Vision Analysis
    vision_analysis: Dict[str, Any] = field(default_factory=dict)
    vision_confidence: float = 0.0
    vision_processing_time: float = 0.0
    
    # Text Analysis  
    text_analysis: Dict[str, Any] = field(default_factory=dict)
    text_confidence: float = 0.0
    text_processing_time: float = 0.0
    
    # Fusion Results
    fusion_confidence: float = 0.0
    fusion_strategy: str = ""
    fusion_reasoning: str = ""
    
    # Performance Metrics
    total_processing_time: float = 0.0
    processing_method: str = ""  # "ollama", "torchserve", "hybrid"
    load_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Quality Metrics
    quality_score: float = 0.0
    reliability_score: float = 0.0
    actionability_score: float = 0.0


class DynamicFusionAgent:
    """
    Dynamic Fusion Agent für adaptive Vision+Text-Prompts
    
    Intelligente Kombination von Vision- und Text-Analysen mit
    adaptiven Prompts basierend auf Marktkontext und Datenqualität
    """
    
    def __init__(self, config: MultimodalFlowConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Fusion Strategies
        self.fusion_strategies = {
            "adaptive": self._adaptive_fusion,
            "vision_priority": self._vision_priority_fusion,
            "text_priority": self._text_priority_fusion,
            "balanced": self._balanced_fusion
        }
        
        # Performance Tracking
        self.fusion_stats = {
            "total_fusions": 0,
            "successful_fusions": 0,
            "average_confidence": 0.0,
            "processing_times": []
        }
        
        self.logger.info(f"DynamicFusionAgent initialized with mode: {config.fusion_mode}")
    
    def create_adaptive_prompt(self, 
                             chart_analysis: Dict[str, Any],
                             technical_features: Dict[str, Any],
                             market_context: Dict[str, Any]) -> str:
        """
        Create adaptive prompt based on analysis context
        
        Args:
            chart_analysis: Vision analysis results
            technical_features: Technical indicator features
            market_context: Current market conditions
            
        Returns:
            Optimized prompt for current context
        """
        try:
            # Analyze context to determine prompt strategy
            volatility = technical_features.get('volatility_10', 0.001)
            trend_strength = technical_features.get('trend_strength', 0.0)
            pattern_confidence = chart_analysis.get('pattern_confidence', 0.0)
            
            # Base prompt components
            base_prompt = "Analyze this EUR/USD trading chart and provide strategic insights."
            
            # Adaptive prompt modifications
            if volatility > 0.01:  # High volatility
                volatility_context = f" The market shows high volatility ({volatility:.4f}), focus on risk management and breakout patterns."
            elif volatility < 0.005:  # Low volatility
                volatility_context = f" The market shows low volatility ({volatility:.4f}), look for consolidation patterns and range trading opportunities."
            else:
                volatility_context = f" The market shows moderate volatility ({volatility:.4f}), analyze both trend continuation and reversal signals."
            
            # Trend context
            if abs(trend_strength) > 0.002:
                trend_context = f" Strong trend detected (strength: {trend_strength:.4f}), prioritize trend-following strategies."
            else:
                trend_context = f" Weak trend environment (strength: {trend_strength:.4f}), focus on mean-reversion and range-bound strategies."
            
            # Pattern context
            if pattern_confidence > 0.7:
                pattern_context = f" High-confidence patterns detected ({pattern_confidence:.2f}), emphasize pattern-based entry/exit signals."
            else:
                pattern_context = f" Pattern confidence is moderate ({pattern_confidence:.2f}), combine with technical indicators for confirmation."
            
            # Market session context
            current_hour = datetime.now().hour
            if 7 <= current_hour <= 16:  # London session
                session_context = " During London session, expect higher liquidity and stronger trends."
            elif 13 <= current_hour <= 22:  # NY session
                session_context = " During New York session, watch for breakouts and news-driven moves."
            else:
                session_context = " During Asian session, expect lower volatility and range-bound trading."
            
            # Combine all contexts
            adaptive_prompt = (
                base_prompt + 
                volatility_context + 
                trend_context + 
                pattern_context + 
                session_context +
                "\n\nProvide specific entry/exit levels, risk management parameters, and confidence assessment."
            )
            
            return adaptive_prompt
            
        except Exception as e:
            self.logger.error(f"Error creating adaptive prompt: {e}")
            return base_prompt
    
    def fuse_multimodal_analysis(self,
                                vision_result: Dict[str, Any],
                                text_result: Dict[str, Any],
                                technical_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse vision and text analysis results using configured strategy
        
        Args:
            vision_result: Results from vision analysis
            text_result: Results from text analysis  
            technical_features: Technical indicator features
            
        Returns:
            Fused analysis result
        """
        start_time = time.time()
        
        try:
            # Get fusion strategy
            fusion_func = self.fusion_strategies.get(
                self.config.fusion_mode, 
                self._adaptive_fusion
            )
            
            # Execute fusion
            fusion_result = fusion_func(vision_result, text_result, technical_features)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            fusion_result['processing_time'] = processing_time
            
            # Update statistics
            self._update_fusion_stats(fusion_result, processing_time)
            
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"Error in multimodal fusion: {e}")
            return self._create_fallback_fusion(vision_result, text_result)
    
    def _adaptive_fusion(self, 
                        vision_result: Dict[str, Any],
                        text_result: Dict[str, Any],
                        technical_features: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive fusion based on confidence and context"""
        
        vision_conf = vision_result.get('confidence', 0.0)
        text_conf = text_result.get('confidence', 0.0)
        
        # Adaptive weight calculation
        if vision_conf > text_conf + 0.2:  # Vision significantly better
            vision_weight = 0.7
            text_weight = 0.3
        elif text_conf > vision_conf + 0.2:  # Text significantly better
            vision_weight = 0.3
            text_weight = 0.7
        else:  # Similar confidence - use configured weights
            vision_weight = self.config.fusion_weights['vision']
            text_weight = self.config.fusion_weights['text']
        
        # Calculate fusion confidence
        fusion_confidence = (vision_conf * vision_weight + text_conf * text_weight)
        
        # Combine insights
        vision_insights = vision_result.get('insights', [])
        text_insights = text_result.get('insights', [])
        
        # Create fusion reasoning
        reasoning = f"Adaptive fusion (V:{vision_weight:.1f}, T:{text_weight:.1f}) - "
        reasoning += f"Vision confidence: {vision_conf:.2f}, Text confidence: {text_conf:.2f}"
        
        return {
            'fusion_confidence': fusion_confidence,
            'fusion_strategy': 'adaptive',
            'vision_weight': vision_weight,
            'text_weight': text_weight,
            'combined_insights': vision_insights + text_insights,
            'reasoning': reasoning,
            'quality_metrics': {
                'vision_quality': vision_conf,
                'text_quality': text_conf,
                'fusion_quality': fusion_confidence
            }
        }
    
    def _vision_priority_fusion(self, vision_result, text_result, technical_features):
        """Vision-priority fusion strategy"""
        return {
            'fusion_confidence': vision_result.get('confidence', 0.0) * 0.8 + text_result.get('confidence', 0.0) * 0.2,
            'fusion_strategy': 'vision_priority',
            'vision_weight': 0.8,
            'text_weight': 0.2,
            'reasoning': 'Vision-priority fusion for chart pattern emphasis'
        }
    
    def _text_priority_fusion(self, vision_result, text_result, technical_features):
        """Text-priority fusion strategy"""
        return {
            'fusion_confidence': text_result.get('confidence', 0.0) * 0.8 + vision_result.get('confidence', 0.0) * 0.2,
            'fusion_strategy': 'text_priority', 
            'vision_weight': 0.2,
            'text_weight': 0.8,
            'reasoning': 'Text-priority fusion for technical analysis emphasis'
        }
    
    def _balanced_fusion(self, vision_result, text_result, technical_features):
        """Balanced fusion strategy"""
        return {
            'fusion_confidence': (vision_result.get('confidence', 0.0) + text_result.get('confidence', 0.0)) / 2,
            'fusion_strategy': 'balanced',
            'vision_weight': 0.5,
            'text_weight': 0.5,
            'reasoning': 'Balanced fusion with equal vision and text weighting'
        }
    
    def _create_fallback_fusion(self, vision_result, text_result):
        """Create fallback fusion result on error"""
        return {
            'fusion_confidence': 0.3,
            'fusion_strategy': 'fallback',
            'vision_weight': 0.5,
            'text_weight': 0.5,
            'reasoning': 'Fallback fusion due to processing error',
            'error': True
        }
    
    def _update_fusion_stats(self, fusion_result, processing_time):
        """Update fusion statistics"""
        self.fusion_stats['total_fusions'] += 1
        
        if fusion_result.get('fusion_confidence', 0) > self.config.min_fusion_confidence:
            self.fusion_stats['successful_fusions'] += 1
        
        # Update average confidence
        current_avg = self.fusion_stats['average_confidence']
        total = self.fusion_stats['total_fusions']
        new_conf = fusion_result.get('fusion_confidence', 0)
        self.fusion_stats['average_confidence'] = (current_avg * (total - 1) + new_conf) / total
        
        # Track processing times
        self.fusion_stats['processing_times'].append(processing_time)
        if len(self.fusion_stats['processing_times']) > 1000:
            self.fusion_stats['processing_times'] = self.fusion_stats['processing_times'][-1000:]
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion performance statistics"""
        processing_times = self.fusion_stats['processing_times']
        
        return {
            'total_fusions': self.fusion_stats['total_fusions'],
            'successful_fusions': self.fusion_stats['successful_fusions'],
            'success_rate': self.fusion_stats['successful_fusions'] / max(self.fusion_stats['total_fusions'], 1),
            'average_confidence': self.fusion_stats['average_confidence'],
            'average_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'processing_time_std': np.std(processing_times) if processing_times else 0.0,
            'fusion_mode': self.config.fusion_mode
        }


class LoadBalancingManager:
    """
    Load Balancing Manager für Real-time-Switching zwischen Ollama und TorchServe
    
    Intelligente Lastverteilung basierend auf aktueller System-Auslastung,
    Response-Zeiten und Verfügbarkeit der Services
    """
    
    def __init__(self, config: MultimodalFlowConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Service Status Tracking
        self.service_status = {
            'ollama': {'available': True, 'load': 0.0, 'response_time': 0.0, 'error_count': 0},
            'torchserve': {'available': True, 'load': 0.0, 'response_time': 0.0, 'error_count': 0}
        }
        
        # Load Monitoring
        self.load_monitor_active = False
        self.load_monitor_thread = None
        
        # Performance Metrics
        self.routing_stats = {
            'ollama_requests': 0,
            'torchserve_requests': 0,
            'routing_decisions': [],
            'load_switches': 0
        }
        
        self.start_load_monitoring()
    
    def start_load_monitoring(self):
        """Start background load monitoring"""
        if not self.load_monitor_active:
            self.load_monitor_active = True
            self.load_monitor_thread = threading.Thread(target=self._monitor_system_load, daemon=True)
            self.load_monitor_thread.start()
            self.logger.info("Load monitoring started")
    
    def stop_load_monitoring(self):
        """Stop background load monitoring"""
        self.load_monitor_active = False
        if self.load_monitor_thread:
            self.load_monitor_thread.join(timeout=2.0)
        self.logger.info("Load monitoring stopped")
    
    def _monitor_system_load(self):
        """Background thread for monitoring system load"""
        while self.load_monitor_active:
            try:
                # Monitor CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Monitor GPU usage (if available)
                gpu_percent = 0.0
                if torch.cuda.is_available():
                    gpu_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
                
                # Monitor memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Update service load estimates
                self._update_service_loads(cpu_percent, gpu_percent, memory_percent)
                
                time.sleep(self.config.load_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in load monitoring: {e}")
                time.sleep(self.config.load_check_interval)
    
    def _update_service_loads(self, cpu_percent: float, gpu_percent: float, memory_percent: float):
        """Update service load estimates"""
        # Ollama typically uses more CPU and some GPU
        ollama_load = (cpu_percent * 0.6 + gpu_percent * 0.4) / 100.0
        
        # TorchServe typically uses more GPU
        torchserve_load = (cpu_percent * 0.3 + gpu_percent * 0.7) / 100.0
        
        self.service_status['ollama']['load'] = ollama_load
        self.service_status['torchserve']['load'] = torchserve_load
    
    def route_request(self, request_type: str = "multimodal") -> str:
        """
        Route request to optimal service based on current load
        
        Args:
            request_type: Type of request (multimodal, vision, text)
            
        Returns:
            Service name to use ("ollama" or "torchserve")
        """
        try:
            ollama_status = self.service_status['ollama']
            torchserve_status = self.service_status['torchserve']
            
            # Check service availability
            if not ollama_status['available'] and torchserve_status['available']:
                self._record_routing_decision('torchserve', 'ollama_unavailable')
                return 'torchserve'
            elif not torchserve_status['available'] and ollama_status['available']:
                self._record_routing_decision('ollama', 'torchserve_unavailable')
                return 'ollama'
            elif not ollama_status['available'] and not torchserve_status['available']:
                self.logger.warning("Both services unavailable, defaulting to ollama")
                self._record_routing_decision('ollama', 'both_unavailable')
                return 'ollama'
            
            # Load-based routing
            ollama_load = ollama_status['load']
            torchserve_load = torchserve_status['load']
            
            # Decision logic
            if ollama_load > self.config.ollama_max_load and torchserve_load < self.config.torchserve_max_load:
                self._record_routing_decision('torchserve', f'ollama_overloaded_{ollama_load:.2f}')
                return 'torchserve'
            elif torchserve_load > self.config.torchserve_max_load and ollama_load < self.config.ollama_max_load:
                self._record_routing_decision('ollama', f'torchserve_overloaded_{torchserve_load:.2f}')
                return 'ollama'
            else:
                # Both services available - choose based on response time and error rate
                ollama_score = self._calculate_service_score('ollama')
                torchserve_score = self._calculate_service_score('torchserve')
                
                if ollama_score > torchserve_score:
                    self._record_routing_decision('ollama', f'better_score_{ollama_score:.2f}')
                    return 'ollama'
                else:
                    self._record_routing_decision('torchserve', f'better_score_{torchserve_score:.2f}')
                    return 'torchserve'
                    
        except Exception as e:
            self.logger.error(f"Error in request routing: {e}")
            self._record_routing_decision('ollama', 'routing_error')
            return 'ollama'  # Default fallback
    
    def _calculate_service_score(self, service_name: str) -> float:
        """Calculate service performance score"""
        status = self.service_status[service_name]
        
        # Lower load is better (invert)
        load_score = 1.0 - status['load']
        
        # Lower response time is better (invert and normalize)
        response_time = status['response_time']
        response_score = 1.0 / (1.0 + response_time) if response_time > 0 else 1.0
        
        # Lower error count is better
        error_score = 1.0 / (1.0 + status['error_count'])
        
        # Weighted combination
        total_score = (load_score * 0.4 + response_score * 0.4 + error_score * 0.2)
        
        return total_score
    
    def update_service_metrics(self, service_name: str, response_time: float, success: bool):
        """Update service performance metrics"""
        if service_name in self.service_status:
            status = self.service_status[service_name]
            
            # Update response time (exponential moving average)
            if status['response_time'] == 0:
                status['response_time'] = response_time
            else:
                status['response_time'] = 0.7 * status['response_time'] + 0.3 * response_time
            
            # Update error count
            if not success:
                status['error_count'] += 1
            else:
                # Decay error count on success
                status['error_count'] = max(0, status['error_count'] - 0.1)
    
    def _record_routing_decision(self, chosen_service: str, reason: str):
        """Record routing decision for analysis"""
        decision = {
            'timestamp': datetime.now(),
            'chosen_service': chosen_service,
            'reason': reason,
            'ollama_load': self.service_status['ollama']['load'],
            'torchserve_load': self.service_status['torchserve']['load']
        }
        
        self.routing_stats['routing_decisions'].append(decision)
        
        # Keep only last 1000 decisions
        if len(self.routing_stats['routing_decisions']) > 1000:
            self.routing_stats['routing_decisions'] = self.routing_stats['routing_decisions'][-1000:]
        
        # Update counters
        if chosen_service == 'ollama':
            self.routing_stats['ollama_requests'] += 1
        else:
            self.routing_stats['torchserve_requests'] += 1
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total_requests = self.routing_stats['ollama_requests'] + self.routing_stats['torchserve_requests']
        
        return {
            'total_requests': total_requests,
            'ollama_requests': self.routing_stats['ollama_requests'],
            'torchserve_requests': self.routing_stats['torchserve_requests'],
            'ollama_percentage': (self.routing_stats['ollama_requests'] / max(total_requests, 1)) * 100,
            'torchserve_percentage': (self.routing_stats['torchserve_requests'] / max(total_requests, 1)) * 100,
            'load_switches': self.routing_stats['load_switches'],
            'current_loads': {
                'ollama': self.service_status['ollama']['load'],
                'torchserve': self.service_status['torchserve']['load']
            },
            'service_scores': {
                'ollama': self._calculate_service_score('ollama'),
                'torchserve': self._calculate_service_score('torchserve')
            }
        }


class MultimodalFlowIntegration:
    """
    Main Multimodal Flow Integration System
    
    Orchestriert die komplette multimodale Pipeline mit:
    - Dynamic Fusion Agent
    - Chart-to-Strategy Pipeline
    - Load Balancing zwischen Ollama und TorchServe
    - Performance Monitoring und Optimization
    """
    
    def __init__(self, config: MultimodalFlowConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core Components
        self.fusion_agent = DynamicFusionAgent(config)
        self.load_balancer = LoadBalancingManager(config)
        
        # AI Service Clients
        self.ollama_client = None
        self.torchserve_handler = None
        self.feature_extractor = None
        self.confidence_scorer = None
        
        # Processing Components
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_requests)
        self.request_queue = Queue()
        
        # Performance Tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0.0,
            'throughput_per_second': 0.0,
            'processing_times': []
        }
        
        # Setup output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("MultimodalFlowIntegration initialized")
    
    def initialize_services(self) -> bool:
        """Initialize all AI services and components"""
        try:
            self.logger.info("Initializing AI services...")
            
            # Initialize Ollama Vision Client
            try:
                self.ollama_client = OllamaVisionClient(
                    model_name="minicpm-v:8b",
                    base_url="http://localhost:11434",
                    timeout=self.config.request_timeout
                )
                self.logger.info("✅ Ollama Vision Client initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Ollama initialization failed: {e}")
                self.load_balancer.service_status['ollama']['available'] = False
            
            # Initialize TorchServe Handler
            try:
                self.torchserve_handler = TorchServeHandler(
                    endpoint_url="http://localhost:8080/predictions/pattern_model",
                    timeout=self.config.request_timeout
                )
                self.logger.info("✅ TorchServe Handler initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ TorchServe initialization failed: {e}")
                self.load_balancer.service_status['torchserve']['available'] = False
            
            # Initialize Feature Extractor
            self.feature_extractor = EnhancedFeatureExtractor({
                "include_time_features": True,
                "include_technical_indicators": True,
                "include_pattern_features": True,
                "include_volatility_features": True
            })
            
            # Initialize Confidence Scorer
            self.confidence_scorer = EnhancedConfidenceScorer()
            
            # Check if at least one service is available
            if (not self.load_balancer.service_status['ollama']['available'] and 
                not self.load_balancer.service_status['torchserve']['available']):
                self.logger.error("❌ No AI services available")
                return False
            
            self.logger.info("✅ Service initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")
            return False
    
    async def process_chart_to_strategy(self, 
                                      chart_path: str,
                                      technical_features: Dict[str, Any],
                                      market_context: Dict[str, Any]) -> MultimodalAnalysisResult:
        """
        Chart-to-Strategy Pipeline mit Ollama Vision Client Integration
        
        Args:
            chart_path: Path to chart image
            technical_features: Technical indicator features
            market_context: Current market conditions
            
        Returns:
            Complete multimodal analysis result
        """
        start_time = time.time()
        result = MultimodalAnalysisResult(
            chart_path=chart_path,
            features=technical_features,
            timestamp=datetime.now()
        )
        
        try:
            # Route request to optimal service
            chosen_service = self.load_balancer.route_request("multimodal")
            result.processing_method = chosen_service
            
            # Create adaptive prompt
            adaptive_prompt = self.fusion_agent.create_adaptive_prompt(
                chart_analysis={},  # Will be filled by vision analysis
                technical_features=technical_features,
                market_context=market_context
            )
            
            # Process with chosen service
            if chosen_service == "ollama" and self.ollama_client:
                vision_result, text_result = await self._process_with_ollama(
                    chart_path, adaptive_prompt, technical_features
                )
            elif chosen_service == "torchserve" and self.torchserve_handler:
                vision_result, text_result = await self._process_with_torchserve(
                    chart_path, adaptive_prompt, technical_features
                )
            else:
                # Fallback to available service
                if self.ollama_client:
                    vision_result, text_result = await self._process_with_ollama(
                        chart_path, adaptive_prompt, technical_features
                    )
                elif self.torchserve_handler:
                    vision_result, text_result = await self._process_with_torchserve(
                        chart_path, adaptive_prompt, technical_features
                    )
                else:
                    raise RuntimeError("No AI services available")
            
            # Store individual results
            result.vision_analysis = vision_result
            result.vision_confidence = vision_result.get('confidence', 0.0)
            result.text_analysis = text_result
            result.text_confidence = text_result.get('confidence', 0.0)
            
            # Perform multimodal fusion
            fusion_result = self.fusion_agent.fuse_multimodal_analysis(
                vision_result, text_result, technical_features
            )
            
            # Store fusion results
            result.fusion_confidence = fusion_result.get('fusion_confidence', 0.0)
            result.fusion_strategy = fusion_result.get('fusion_strategy', '')
            result.fusion_reasoning = fusion_result.get('reasoning', '')
            
            # Calculate quality scores
            result.quality_score = self._calculate_quality_score(result)
            result.reliability_score = self._calculate_reliability_score(result)
            result.actionability_score = self._calculate_actionability_score(result)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            result.total_processing_time = processing_time
            
            self.load_balancer.update_service_metrics(
                chosen_service, processing_time, True
            )
            
            self._update_performance_metrics(processing_time, True)
            
            self.logger.info(f"✅ Chart-to-Strategy completed in {processing_time:.2f}s with {chosen_service}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            result.total_processing_time = processing_time
            
            self.load_balancer.update_service_metrics(
                result.processing_method, processing_time, False
            )
            
            self._update_performance_metrics(processing_time, False)
            
            self.logger.error(f"❌ Chart-to-Strategy failed: {e}")
            
            # Return partial result with error information
            result.fusion_reasoning = f"Processing failed: {str(e)}"
            return result
    
    async def _process_with_ollama(self, 
                                 chart_path: str,
                                 prompt: str,
                                 technical_features: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process request with Ollama Vision Client"""
        
        # Vision Analysis
        vision_start = time.time()
        try:
            vision_response = await self.ollama_client.analyze_chart_async(
                chart_path=chart_path,
                prompt=f"Analyze this EUR/USD chart for visual patterns and formations: {prompt}",
                include_technical_context=True
            )
            
            vision_result = {
                'confidence': vision_response.get('confidence', 0.0),
                'patterns': vision_response.get('patterns', []),
                'insights': vision_response.get('insights', []),
                'support_resistance': vision_response.get('support_resistance', {}),
                'trend_analysis': vision_response.get('trend_analysis', {}),
                'processing_time': time.time() - vision_start
            }
            
        except Exception as e:
            self.logger.error(f"Ollama vision analysis failed: {e}")
            vision_result = {'confidence': 0.0, 'error': str(e), 'processing_time': time.time() - vision_start}
        
        # Text Analysis (Technical Features)
        text_start = time.time()
        try:
            # Create technical analysis prompt
            tech_prompt = self._create_technical_analysis_prompt(technical_features, prompt)
            
            text_response = await self.ollama_client.generate_text_async(
                prompt=tech_prompt,
                max_tokens=500
            )
            
            text_result = {
                'confidence': self._extract_confidence_from_text(text_response),
                'analysis': text_response,
                'insights': self._extract_insights_from_text(text_response),
                'recommendations': self._extract_recommendations_from_text(text_response),
                'processing_time': time.time() - text_start
            }
            
        except Exception as e:
            self.logger.error(f"Ollama text analysis failed: {e}")
            text_result = {'confidence': 0.0, 'error': str(e), 'processing_time': time.time() - text_start}
        
        return vision_result, text_result
    
    async def _process_with_torchserve(self,
                                     chart_path: str,
                                     prompt: str,
                                     technical_features: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Process request with TorchServe Handler"""
        
        # Prepare features for TorchServe
        feature_json = {
            'chart_path': chart_path,
            'technical_features': technical_features,
            'prompt': prompt,
            'timestamp': datetime.now().isoformat()
        }
        
        # Vision Analysis via TorchServe
        vision_start = time.time()
        try:
            vision_response = await self.torchserve_handler.predict_async(feature_json)
            
            vision_result = {
                'confidence': vision_response.get('vision_confidence', 0.0),
                'patterns': vision_response.get('detected_patterns', []),
                'insights': vision_response.get('vision_insights', []),
                'support_resistance': vision_response.get('levels', {}),
                'trend_analysis': vision_response.get('trend', {}),
                'processing_time': time.time() - vision_start
            }
            
        except Exception as e:
            self.logger.error(f"TorchServe vision analysis failed: {e}")
            vision_result = {'confidence': 0.0, 'error': str(e), 'processing_time': time.time() - vision_start}
        
        # Text Analysis via TorchServe
        text_start = time.time()
        try:
            text_feature_json = {
                'features': technical_features,
                'analysis_type': 'technical',
                'prompt': prompt
            }
            
            text_response = await self.torchserve_handler.predict_async(text_feature_json)
            
            text_result = {
                'confidence': text_response.get('text_confidence', 0.0),
                'analysis': text_response.get('technical_analysis', ''),
                'insights': text_response.get('text_insights', []),
                'recommendations': text_response.get('recommendations', []),
                'processing_time': time.time() - text_start
            }
            
        except Exception as e:
            self.logger.error(f"TorchServe text analysis failed: {e}")
            text_result = {'confidence': 0.0, 'error': str(e), 'processing_time': time.time() - text_start}
        
        return vision_result, text_result
    
    def process_batch_charts(self, 
                           chart_paths: List[str],
                           technical_features_list: List[Dict[str, Any]],
                           market_context: Dict[str, Any]) -> List[MultimodalAnalysisResult]:
        """
        Process multiple charts in parallel for high throughput
        
        Args:
            chart_paths: List of chart image paths
            technical_features_list: List of technical features for each chart
            market_context: Current market conditions
            
        Returns:
            List of analysis results
        """
        self.logger.info(f"Processing batch of {len(chart_paths)} charts...")
        
        # Create async tasks
        async def process_batch():
            tasks = []
            for i, chart_path in enumerate(chart_paths):
                features = technical_features_list[i] if i < len(technical_features_list) else {}
                task = self.process_chart_to_strategy(chart_path, features, market_context)
                tasks.append(task)
            
            # Process with controlled concurrency
            results = []
            for i in range(0, len(tasks), self.config.batch_size):
                batch_tasks = tasks[i:i + self.config.batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Batch processing error: {result}")
                        # Create error result
                        error_result = MultimodalAnalysisResult()
                        error_result.fusion_reasoning = f"Batch processing error: {str(result)}"
                        results.append(error_result)
                    else:
                        results.append(result)
            
            return results
        
        # Run async batch processing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(process_batch())
        finally:
            loop.close()
        
        self.logger.info(f"✅ Batch processing completed: {len(results)} results")
        return results
    
    def _create_technical_analysis_prompt(self, 
                                        technical_features: Dict[str, Any],
                                        base_prompt: str) -> str:
        """Create technical analysis prompt from features"""
        
        # Extract key technical indicators
        rsi = technical_features.get('rsi_14', 50)
        sma_20 = technical_features.get('sma_20', 0)
        volatility = technical_features.get('volatility_10', 0.001)
        trend_strength = technical_features.get('trend_strength', 0.0)
        
        tech_prompt = f"""
        Technical Analysis Context:
        - RSI(14): {rsi:.2f}
        - SMA(20): {sma_20:.5f}
        - Volatility: {volatility:.4f}
        - Trend Strength: {trend_strength:.4f}
        
        {base_prompt}
        
        Based on these technical indicators, provide:
        1. Market condition assessment
        2. Entry/exit signal strength
        3. Risk level evaluation
        4. Confidence in the analysis
        """
        
        return tech_prompt
    
    def _extract_confidence_from_text(self, text_response: str) -> float:
        """Extract confidence score from text response"""
        try:
            # Look for confidence indicators in text
            text_lower = text_response.lower()
            
            if 'high confidence' in text_lower or 'very confident' in text_lower:
                return 0.8
            elif 'confident' in text_lower or 'strong signal' in text_lower:
                return 0.7
            elif 'moderate' in text_lower or 'medium' in text_lower:
                return 0.6
            elif 'low confidence' in text_lower or 'uncertain' in text_lower:
                return 0.4
            else:
                return 0.5  # Default moderate confidence
                
        except Exception:
            return 0.5
    
    def _extract_insights_from_text(self, text_response: str) -> List[str]:
        """Extract key insights from text response"""
        try:
            # Simple extraction based on common patterns
            insights = []
            lines = text_response.split('\n')
            
            for line in lines:
                line = line.strip()
                if (line.startswith('-') or line.startswith('•') or 
                    'signal' in line.lower() or 'pattern' in line.lower()):
                    insights.append(line)
            
            return insights[:5]  # Limit to top 5 insights
            
        except Exception:
            return []
    
    def _extract_recommendations_from_text(self, text_response: str) -> List[str]:
        """Extract recommendations from text response"""
        try:
            recommendations = []
            text_lower = text_response.lower()
            
            # Look for recommendation keywords
            if 'buy' in text_lower or 'long' in text_lower:
                recommendations.append('Consider long position')
            if 'sell' in text_lower or 'short' in text_lower:
                recommendations.append('Consider short position')
            if 'hold' in text_lower or 'wait' in text_lower:
                recommendations.append('Hold current position')
            if 'stop loss' in text_lower:
                recommendations.append('Set appropriate stop loss')
            
            return recommendations
            
        except Exception:
            return []
    
    def _calculate_quality_score(self, result: MultimodalAnalysisResult) -> float:
        """Calculate overall quality score for the analysis"""
        try:
            # Weighted combination of confidence scores
            vision_weight = 0.4
            text_weight = 0.3
            fusion_weight = 0.3
            
            quality_score = (
                result.vision_confidence * vision_weight +
                result.text_confidence * text_weight +
                result.fusion_confidence * fusion_weight
            )
            
            # Penalty for errors
            if 'error' in result.vision_analysis or 'error' in result.text_analysis:
                quality_score *= 0.7
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _calculate_reliability_score(self, result: MultimodalAnalysisResult) -> float:
        """Calculate reliability score based on consistency"""
        try:
            # Check consistency between vision and text analysis
            vision_conf = result.vision_confidence
            text_conf = result.text_confidence
            
            # Higher reliability when both analyses agree
            confidence_diff = abs(vision_conf - text_conf)
            consistency_score = 1.0 - (confidence_diff / 2.0)  # Normalize to [0, 1]
            
            # Factor in processing times (faster = more reliable for real-time use)
            avg_processing_time = (
                result.vision_analysis.get('processing_time', 0) +
                result.text_analysis.get('processing_time', 0)
            ) / 2
            
            time_score = 1.0 / (1.0 + avg_processing_time / 10.0)  # Normalize
            
            reliability_score = (consistency_score * 0.7 + time_score * 0.3)
            
            return min(1.0, max(0.0, reliability_score))
            
        except Exception:
            return 0.0
    
    def _calculate_actionability_score(self, result: MultimodalAnalysisResult) -> float:
        """Calculate actionability score based on clarity of recommendations"""
        try:
            actionability = 0.0
            
            # Check for specific recommendations
            vision_insights = result.vision_analysis.get('insights', [])
            text_insights = result.text_analysis.get('insights', [])
            
            total_insights = len(vision_insights) + len(text_insights)
            if total_insights > 0:
                actionability += 0.3
            
            # Check for specific levels (support/resistance)
            if result.vision_analysis.get('support_resistance'):
                actionability += 0.3
            
            # Check for trend analysis
            if result.vision_analysis.get('trend_analysis'):
                actionability += 0.2
            
            # Check fusion confidence
            if result.fusion_confidence > 0.6:
                actionability += 0.2
            
            return min(1.0, actionability)
            
        except Exception:
            return 0.0
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update performance tracking metrics"""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # Update processing times
        self.performance_metrics['processing_times'].append(processing_time)
        if len(self.performance_metrics['processing_times']) > 1000:
            self.performance_metrics['processing_times'] = self.performance_metrics['processing_times'][-1000:]
        
        # Update average processing time
        times = self.performance_metrics['processing_times']
        self.performance_metrics['average_processing_time'] = np.mean(times) if times else 0.0
        
        # Calculate throughput (requests per second)
        if len(times) >= 10:
            recent_times = times[-10:]
            total_time = sum(recent_times)
            self.performance_metrics['throughput_per_second'] = len(recent_times) / total_time if total_time > 0 else 0.0
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance and routing statistics"""
        
        # Get component statistics
        fusion_stats = self.fusion_agent.get_fusion_statistics()
        routing_stats = self.load_balancer.get_routing_statistics()
        
        # Calculate success rate
        total_requests = self.performance_metrics['total_requests']
        success_rate = (self.performance_metrics['successful_requests'] / max(total_requests, 1)) * 100
        
        return {
            'performance_metrics': {
                'total_requests': total_requests,
                'successful_requests': self.performance_metrics['successful_requests'],
                'failed_requests': self.performance_metrics['failed_requests'],
                'success_rate_percent': success_rate,
                'average_processing_time': self.performance_metrics['average_processing_time'],
                'throughput_per_second': self.performance_metrics['throughput_per_second']
            },
            'fusion_statistics': fusion_stats,
            'routing_statistics': routing_stats,
            'service_status': self.load_balancer.service_status,
            'configuration': {
                'fusion_mode': self.config.fusion_mode,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'batch_size': self.config.batch_size,
                'confidence_thresholds': {
                    'vision': self.config.min_vision_confidence,
                    'text': self.config.min_text_confidence,
                    'fusion': self.config.min_fusion_confidence
                }
            }
        }
    
    def save_analysis_results(self, 
                            results: List[MultimodalAnalysisResult],
                            filename: str = None) -> str:
        """Save analysis results to file"""
        
        if filename is None:
            filename = f"multimodal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(self.config.output_dir) / filename
        
        try:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                result_dict = {
                    'chart_path': result.chart_path,
                    'timestamp': result.timestamp.isoformat(),
                    'vision_confidence': result.vision_confidence,
                    'text_confidence': result.text_confidence,
                    'fusion_confidence': result.fusion_confidence,
                    'fusion_strategy': result.fusion_strategy,
                    'fusion_reasoning': result.fusion_reasoning,
                    'processing_method': result.processing_method,
                    'total_processing_time': result.total_processing_time,
                    'quality_score': result.quality_score,
                    'reliability_score': result.reliability_score,
                    'actionability_score': result.actionability_score,
                    'vision_analysis': result.vision_analysis,
                    'text_analysis': result.text_analysis
                }
                serializable_results.append(result_dict)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump({
                    'results': serializable_results,
                    'metadata': {
                        'total_results': len(results),
                        'generation_time': datetime.now().isoformat(),
                        'config': self.config.__dict__
                    },
                    'statistics': self.get_comprehensive_statistics()
                }, f, indent=2)
            
            self.logger.info(f"✅ Analysis results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop load monitoring
            self.load_balancer.stop_load_monitoring()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Close service connections
            if self.ollama_client:
                # Ollama client cleanup if needed
                pass
            
            if self.torchserve_handler:
                # TorchServe handler cleanup if needed
                pass
            
            self.logger.info("✅ Multimodal Flow Integration cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {e}")


# Factory Functions
def create_multimodal_flow_config(**kwargs) -> MultimodalFlowConfig:
    """Create multimodal flow configuration"""
    return MultimodalFlowConfig(**kwargs)


def create_multimodal_flow_integration(config: MultimodalFlowConfig) -> MultimodalFlowIntegration:
    """Create multimodal flow integration system"""
    return MultimodalFlowIntegration(config)


# Async Helper Functions
async def analyze_single_chart(integration: MultimodalFlowIntegration,
                             chart_path: str,
                             technical_features: Dict[str, Any],
                             market_context: Dict[str, Any]) -> MultimodalAnalysisResult:
    """Async helper for single chart analysis"""
    return await integration.process_chart_to_strategy(chart_path, technical_features, market_context)


async def analyze_multiple_charts(integration: MultimodalFlowIntegration,
                                chart_data: List[Tuple[str, Dict[str, Any]]],
                                market_context: Dict[str, Any]) -> List[MultimodalAnalysisResult]:
    """Async helper for multiple chart analysis"""
    chart_paths = [data[0] for data in chart_data]
    features_list = [data[1] for data in chart_data]
    
    return integration.process_batch_charts(chart_paths, features_list, market_context)