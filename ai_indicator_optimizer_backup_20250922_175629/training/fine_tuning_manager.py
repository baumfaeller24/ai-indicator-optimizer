"""
Fine-Tuning Manager für MiniCPM-4.1-8B Trading-Pattern Anpassung
Optimiert für RTX 5090 + 191GB RAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM,
    TrainingArguments, Trainer, TrainerCallback,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import set_seed
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import os
from datetime import datetime
import wandb
from tqdm import tqdm
import gc

from ..ai.multimodal_ai import MultimodalAI, ModelConfig, GPUMemoryManager
from ..ai.model_factory import ModelFactory
from ..core.resource_manager import ResourceManager
# from ..data.models import TradingData


@dataclass
class FineTuningConfig:
    """Konfiguration für Fine-Tuning Pipeline"""
    
    # Model Settings
    base_model_name: str = "openbmb/MiniCPM-V-2_6"
    output_dir: str = "./models/finetuned_minicpm"
    cache_dir: str = "./models/cache"
    
    # Training Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4  # RTX 5090 optimiert
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # RTX 5090 Optimierungen
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    dataloader_num_workers: int = 8  # 32 CPU Cores optimal nutzen
    
    # Memory Management für 32GB VRAM
    max_memory_per_gpu: str = "30GB"
    gradient_checkpointing: bool = True
    use_cpu_offload: bool = False  # Mit 191GB RAM nicht nötig
    
    # Data Settings
    max_sequence_length: int = 2048
    image_resolution: int = 448
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Logging & Monitoring
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Experiment Tracking
    use_wandb: bool = True
    wandb_project: str = "ai-indicator-optimizer"
    wandb_run_name: Optional[str] = None
    
    # Advanced Settings
    lora_enabled: bool = True  # LoRA für Memory-Effizienz
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Trading-spezifische Settings
    pattern_types: List[str] = field(default_factory=lambda: [
        "double_top", "double_bottom", "head_shoulders", "triangle",
        "support_resistance", "trend_line", "breakout", "reversal"
    ])
    
    # Validation Settings
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001


@dataclass
class TrainingMetrics:
    """Training Metriken Container"""
    epoch: int
    step: int
    train_loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    gpu_memory_gb: float = 0.0
    training_time: float = 0.0
    pattern_accuracy: Optional[Dict[str, float]] = None


class TradingPatternDataset(Dataset):
    """
    Dataset für Trading-Pattern Fine-Tuning
    Kombiniert Chart-Images, Numerical Data und Text-Labels
    """
    
    def __init__(self, 
                 data_samples: List[Dict[str, Any]],
                 processor: Any,
                 tokenizer: Any,
                 config: FineTuningConfig):
        
        self.data_samples = data_samples
        self.processor = processor
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Pattern-zu-ID Mapping
        self.pattern_to_id = {pattern: i for i, pattern in enumerate(config.pattern_types)}
        self.id_to_pattern = {i: pattern for pattern, i in self.pattern_to_id.items()}
        
        self.logger.info(f"Dataset initialized with {len(data_samples)} samples")
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Gibt preprocessed Sample zurück"""
        try:
            sample = self.data_samples[idx]
            
            # Chart Image
            chart_image = sample.get('chart_image')
            if isinstance(chart_image, str):
                chart_image = Image.open(chart_image).convert('RGB')
            elif isinstance(chart_image, np.ndarray):
                chart_image = Image.fromarray(chart_image)
            
            # Resize Image
            if chart_image.size != (self.config.image_resolution, self.config.image_resolution):
                chart_image = chart_image.resize(
                    (self.config.image_resolution, self.config.image_resolution),
                    Image.Resampling.LANCZOS
                )
            
            # Numerical Indicators
            numerical_data = sample.get('numerical_data', {})
            numerical_text = self._format_numerical_data(numerical_data)
            
            # Pattern Label
            pattern_label = sample.get('pattern_label', 'unknown')
            pattern_description = sample.get('pattern_description', '')
            
            # Market Context
            market_context = sample.get('market_context', {})
            context_text = self._format_market_context(market_context)
            
            # Erstelle Training Prompt
            training_prompt = self._create_training_prompt(
                numerical_text, context_text, pattern_label, pattern_description
            )
            
            # Process mit MiniCPM Processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": training_prompt},
                        {"type": "image", "image": chart_image}
                    ]
                }
            ]
            
            # Processor Input
            inputs = self.processor(
                messages,
                return_tensors="pt",
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True
            )
            
            # Labels für Training (gleich wie input_ids für Causal LM)
            labels = inputs["input_ids"].clone()
            
            # Attention Mask
            attention_mask = inputs.get("attention_mask", torch.ones_like(labels))
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
                "pixel_values": inputs.get("pixel_values", torch.zeros(3, self.config.image_resolution, self.config.image_resolution)).squeeze(0),
                "pattern_id": torch.tensor(self.pattern_to_id.get(pattern_label, 0), dtype=torch.long)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing sample {idx}: {e}")
            # Return dummy sample
            return self._get_dummy_sample()
    
    def _format_numerical_data(self, numerical_data: Dict[str, Any]) -> str:
        """Formatiert numerische Daten als Text"""
        if not numerical_data:
            return "No numerical indicators available."
        
        formatted_parts = []
        
        for indicator, value in numerical_data.items():
            if isinstance(value, (int, float)):
                formatted_parts.append(f"{indicator}: {value:.4f}")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        formatted_parts.append(f"{indicator}_{sub_key}: {sub_value:.4f}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[-1], (int, float)):
                    formatted_parts.append(f"{indicator}: {value[-1]:.4f}")
        
        return f"Technical Indicators: {', '.join(formatted_parts)}"
    
    def _format_market_context(self, market_context: Dict[str, Any]) -> str:
        """Formatiert Market Context als Text"""
        if not market_context:
            return "Market context not available."
        
        context_parts = []
        
        # Timeframe
        if 'timeframe' in market_context:
            context_parts.append(f"Timeframe: {market_context['timeframe']}")
        
        # Symbol
        if 'symbol' in market_context:
            context_parts.append(f"Symbol: {market_context['symbol']}")
        
        # Trend
        if 'trend' in market_context:
            context_parts.append(f"Trend: {market_context['trend']}")
        
        # Volatility
        if 'volatility' in market_context:
            context_parts.append(f"Volatility: {market_context['volatility']}")
        
        return f"Market Context: {', '.join(context_parts)}"
    
    def _create_training_prompt(self, 
                              numerical_text: str,
                              context_text: str,
                              pattern_label: str,
                              pattern_description: str) -> str:
        """Erstellt Training-Prompt für Fine-Tuning"""
        
        prompt = f"""Analyze this forex trading chart and identify the pattern.

{context_text}
{numerical_text}

Based on the chart image and technical indicators, identify the trading pattern and provide analysis.

Expected Pattern: {pattern_label.replace('_', ' ').title()}
Analysis: {pattern_description}

Provide a detailed analysis of:
1. Pattern identification and confidence
2. Key support and resistance levels
3. Entry and exit recommendations
4. Risk assessment"""
        
        return prompt
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Gibt Dummy-Sample bei Fehlern zurück"""
        return {
            "input_ids": torch.zeros(self.config.max_sequence_length, dtype=torch.long),
            "attention_mask": torch.zeros(self.config.max_sequence_length, dtype=torch.long),
            "labels": torch.zeros(self.config.max_sequence_length, dtype=torch.long),
            "pixel_values": torch.zeros(3, self.config.image_resolution, self.config.image_resolution),
            "pattern_id": torch.tensor(0, dtype=torch.long)
        }


class FineTuningTrainer(Trainer):
    """
    Custom Trainer für MiniCPM Fine-Tuning mit Trading-spezifischen Metriken
    """
    
    def __init__(self, *args, pattern_types: List[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_types = pattern_types or []
        self.gpu_memory_manager = GPUMemoryManager()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom Loss Computation mit Pattern Classification"""
        
        # Standard Causal LM Loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Optional: Pattern Classification Loss hinzufügen
        if "pattern_id" in inputs and hasattr(model, 'pattern_classifier'):
            pattern_logits = model.pattern_classifier(outputs.hidden_states[-1][:, -1, :])
            pattern_loss = nn.CrossEntropyLoss()(pattern_logits, inputs["pattern_id"])
            loss = loss + 0.1 * pattern_loss  # Gewichtung
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced Logging mit GPU Memory Stats"""
        
        # GPU Memory Stats hinzufügen
        if torch.cuda.is_available():
            memory_stats = self.gpu_memory_manager.get_memory_stats()
            logs["gpu_memory_allocated_gb"] = memory_stats["allocated_gb"]
            logs["gpu_memory_free_gb"] = memory_stats["free_gb"]
        
        super().log(logs)


class FineTuningManager:
    """
    Haupt-Manager für MiniCPM Fine-Tuning Pipeline
    Optimiert für RTX 5090 + 191GB RAM
    """
    
    def __init__(self, 
                 config: FineTuningConfig,
                 resource_manager: Optional[ResourceManager] = None):
        
        self.config = config
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Hardware Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_memory_manager = GPUMemoryManager()
        
        # Model Components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.trainer = None
        
        # Training State
        self.training_metrics: List[TrainingMetrics] = []
        self.best_model_path = None
        
        # Setup Directories
        self._setup_directories()
        
        # Setup Logging
        self._setup_logging()
        
        self.logger.info(f"FineTuningManager initialized on {self.device}")
    
    def _setup_directories(self):
        """Erstellt notwendige Verzeichnisse"""
        directories = [
            self.config.output_dir,
            self.config.cache_dir,
            f"{self.config.output_dir}/checkpoints",
            f"{self.config.output_dir}/logs",
            f"{self.config.output_dir}/metrics"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup Enhanced Logging"""
        log_file = f"{self.config.output_dir}/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Wandb Setup
        if self.config.use_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or f"minicpm_finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config.__dict__
                )
                self.logger.info("Wandb initialized successfully")
            except Exception as e:
                self.logger.warning(f"Wandb initialization failed: {e}")
    
    def load_base_model(self) -> bool:
        """Lädt Base MiniCPM Model für Fine-Tuning"""
        try:
            self.logger.info(f"Loading base model: {self.config.base_model_name}")
            
            # Memory Optimization
            memory_config = self.gpu_memory_manager.optimize_for_model_loading(model_size_gb=8)
            
            # Model Loading mit Training-optimierter Konfiguration
            model_kwargs = {
                "torch_dtype": torch.float16 if self.config.use_mixed_precision else torch.float32,
                "device_map": "auto",
                "trust_remote_code": True,
                "cache_dir": self.config.cache_dir,
                "use_flash_attention_2": self.config.use_flash_attention,
            }
            
            # Memory Management
            if memory_config["use_8bit_loading"]:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load Model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                **model_kwargs
            )
            
            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            # Load Processor
            self.processor = AutoProcessor.from_pretrained(
                self.config.base_model_name,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            # Training Optimizations
            if self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            # LoRA Setup (falls aktiviert)
            if self.config.lora_enabled:
                self._setup_lora()
            
            # Model in Training Mode
            self.model.train()
            
            self.logger.info("Base model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Base model loading failed: {e}")
            return False
    
    def _setup_lora(self):
        """Setup LoRA für Memory-effizientes Fine-Tuning"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # MiniCPM spezifisch
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.logger.info("LoRA setup completed")
            
        except ImportError:
            self.logger.warning("PEFT library not available, skipping LoRA setup")
        except Exception as e:
            self.logger.error(f"LoRA setup failed: {e}")
    
    def prepare_datasets(self, 
                        training_data: List[Dict[str, Any]]) -> Tuple[TradingPatternDataset, TradingPatternDataset, TradingPatternDataset]:
        """Bereitet Training-, Validation- und Test-Datasets vor"""
        
        self.logger.info(f"Preparing datasets from {len(training_data)} samples")
        
        # Shuffle Data
        np.random.shuffle(training_data)
        
        # Split Data
        n_samples = len(training_data)
        train_end = int(n_samples * self.config.train_split)
        val_end = train_end + int(n_samples * self.config.val_split)
        
        train_data = training_data[:train_end]
        val_data = training_data[train_end:val_end]
        test_data = training_data[val_end:]
        
        # Create Datasets
        train_dataset = TradingPatternDataset(
            train_data, self.processor, self.tokenizer, self.config
        )
        
        val_dataset = TradingPatternDataset(
            val_data, self.processor, self.tokenizer, self.config
        )
        
        test_dataset = TradingPatternDataset(
            test_data, self.processor, self.tokenizer, self.config
        )
        
        self.logger.info(f"Datasets prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup Training Arguments für RTX 5090 Optimierung"""
        
        return TrainingArguments(
            output_dir=f"{self.config.output_dir}/checkpoints",
            
            # Training Hyperparameters
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # RTX 5090 Optimizations
            fp16=self.config.use_mixed_precision,
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            
            # Logging & Evaluation
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            # Model Selection
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Experiment Tracking
            report_to="wandb" if self.config.use_wandb else None,
            run_name=self.config.wandb_run_name,
            
            # Memory Optimization
            remove_unused_columns=False,
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
    
    def start_fine_tuning(self, 
                         training_data: List[Dict[str, Any]]) -> bool:
        """Startet Fine-Tuning Pipeline"""
        
        try:
            self.logger.info("Starting MiniCPM Fine-Tuning Pipeline")
            
            # 1. Load Base Model
            if not self.load_base_model():
                return False
            
            # 2. Prepare Datasets
            train_dataset, val_dataset, test_dataset = self.prepare_datasets(training_data)
            
            # 3. Setup Training Arguments
            training_args = self.setup_training_arguments()
            
            # 4. Create Trainer
            self.trainer = FineTuningTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                pattern_types=self.config.pattern_types
            )
            
            # 5. Start Training
            self.logger.info("Starting training...")
            start_time = time.time()
            
            train_result = self.trainer.train()
            
            training_time = time.time() - start_time
            
            # 6. Save Final Model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            self.processor.save_pretrained(self.config.output_dir)
            
            # 7. Evaluate on Test Set
            test_results = self.trainer.evaluate(eval_dataset=test_dataset)
            
            # 8. Save Training Metrics
            self._save_training_results(train_result, test_results, training_time)
            
            self.logger.info(f"Fine-tuning completed in {training_time:.2f}s")
            self.logger.info(f"Final eval loss: {test_results.get('eval_loss', 'N/A')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            return False
    
    def _save_training_results(self, 
                              train_result: Any,
                              test_results: Dict[str, float],
                              training_time: float):
        """Speichert Training-Ergebnisse"""
        
        results = {
            "training_time": training_time,
            "train_results": train_result.metrics if hasattr(train_result, 'metrics') else {},
            "test_results": test_results,
            "config": self.config.__dict__,
            "model_path": self.config.output_dir,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to JSON
        results_file = f"{self.config.output_dir}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log to Wandb
        if self.config.use_wandb:
            wandb.log({"final_results": results})
        
        self.logger.info(f"Training results saved to {results_file}")
    
    def load_finetuned_model(self, model_path: Optional[str] = None) -> MultimodalAI:
        """Lädt fine-tuned Model in MultimodalAI Wrapper"""
        
        model_path = model_path or self.config.output_dir
        
        try:
            # Create ModelConfig für fine-tuned model
            model_config = ModelConfig(
                model_name=model_path,
                torch_dtype=torch.float16,
                use_flash_attention=True,
                enable_mixed_precision=True
            )
            
            # Create MultimodalAI mit fine-tuned model
            multimodal_ai = MultimodalAI(model_config=model_config)
            
            # Load model
            success = multimodal_ai.load_model()
            
            if success:
                self.logger.info(f"Fine-tuned model loaded successfully from {model_path}")
                return multimodal_ai
            else:
                raise RuntimeError("Failed to load fine-tuned model")
                
        except Exception as e:
            self.logger.error(f"Loading fine-tuned model failed: {e}")
            raise
    
    def cleanup(self):
        """Bereinigt Ressourcen nach Training"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        # GPU Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
        
        # Wandb cleanup
        if self.config.use_wandb:
            wandb.finish()
        
        self.logger.info("Fine-tuning cleanup completed")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Gibt Training-Zusammenfassung zurück"""
        
        if not self.training_metrics:
            return {"status": "No training completed"}
        
        latest_metrics = self.training_metrics[-1]
        
        return {
            "status": "completed",
            "total_epochs": latest_metrics.epoch,
            "total_steps": latest_metrics.step,
            "final_train_loss": latest_metrics.train_loss,
            "final_eval_loss": latest_metrics.eval_loss,
            "training_time": latest_metrics.training_time,
            "peak_gpu_memory_gb": latest_metrics.gpu_memory_gb,
            "model_path": self.config.output_dir,
            "pattern_types": self.config.pattern_types
        }