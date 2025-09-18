"""
Numerical Indicator Optimizer für Parameter-Optimierung von Trading-Indikatoren.
Nutzt genetische Algorithmen und Bayesian Optimization für optimale Parameter-Findung.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import optuna
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

from ..data.models import OHLCVData, IndicatorData
from ..core.hardware_detector import HardwareDetector

logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """Unterstützte Indikator-Typen"""
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    SMA = "sma"
    EMA = "ema"
    STOCHASTIC = "stochastic"
    ATR = "atr"
    ADX = "adx"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    MFI = "mfi"
    ROC = "roc"

@dataclass
class IndicatorParameter:
    """Parameter-Definition für Indikatoren"""
    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    default_value: float = None
    parameter_type: str = "int"  # "int", "float", "bool"
    
    def __post_init__(self):
        if self.default_value is None:
            self.default_value = (self.min_value + self.max_value) / 2

@dataclass
class OptimizationResult:
    """Ergebnis der Parameter-Optimierung"""
    indicator_type: IndicatorType
    optimal_parameters: Dict[str, Any]
    performance_score: float
    backtest_results: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    convergence_history: List[float] = field(default_factory=list)

@dataclass
class OptimizationConfig:
    """Konfiguration für Optimierung"""
    optimization_method: str = "bayesian"  # "bayesian", "genetic", "grid", "random"
    max_iterations: int = 100
    population_size: int = 50
    convergence_threshold: float = 1e-6
    parallel_jobs: int = 8
    objective_function: str = "sharpe_ratio"  # "sharpe_ratio", "profit_factor", "max_drawdown"
    validation_split: float = 0.3

class NumericalIndicatorOptimizer:
    """
    Optimiert Parameter von Trading-Indikatoren für maximale Performance.
    Nutzt verschiedene Optimierungsalgorithmen und parallele Verarbeitung.
    """
    
    def __init__(self, hardware_detector: HardwareDetector):
        self.hardware_detector = hardware_detector
        self.cpu_cores = hardware_detector.cpu_info.cores_logical if hardware_detector.cpu_info else 8
        
        # Parameter-Definitionen für verschiedene Indikatoren
        self.indicator_parameters = self._initialize_indicator_parameters()
        
        # Performance-Metriken Cache
        self.performance_cache = {}
        
        logger.info(f"NumericalIndicatorOptimizer initialisiert mit {self.cpu_cores} CPU-Kernen")
    
    def optimize_indicator(self,
                          indicator_type: IndicatorType,
                          ohlcv_data: OHLCVData,
                          config: OptimizationConfig = None) -> OptimizationResult:
        """
        Optimiert Parameter für einen spezifischen Indikator.
        
        Args:
            indicator_type: Typ des zu optimierenden Indikators
            ohlcv_data: Historische OHLCV-Daten
            config: Optimierungs-Konfiguration
            
        Returns:
            OptimizationResult mit optimalen Parametern
        """
        try:
            if config is None:
                config = OptimizationConfig()
            
            logger.info(f"Starte Optimierung für {indicator_type.value}")
            
            # Daten-Validierung
            if not self._validate_data(ohlcv_data):
                raise ValueError("Ungültige OHLCV-Daten für Optimierung")
            
            # Parameter-Space definieren
            parameter_space = self.indicator_parameters.get(indicator_type)
            if not parameter_space:
                raise ValueError(f"Keine Parameter-Definition für {indicator_type.value}")
            
            # Optimierung basierend auf gewählter Methode
            if config.optimization_method == "bayesian":
                result = self._bayesian_optimization(indicator_type, ohlcv_data, config, parameter_space)
            elif config.optimization_method == "genetic":
                result = self._genetic_optimization(indicator_type, ohlcv_data, config, parameter_space)
            elif config.optimization_method == "grid":
                result = self._grid_search_optimization(indicator_type, ohlcv_data, config, parameter_space)
            else:
                result = self._random_search_optimization(indicator_type, ohlcv_data, config, parameter_space)
            
            logger.info(f"Optimierung abgeschlossen: Score = {result.performance_score:.4f}")
            return result
            
        except Exception as e:
            logger.exception(f"Fehler bei Indikator-Optimierung: {e}")
            # Fallback: Default-Parameter zurückgeben
            return self._create_fallback_result(indicator_type, e)
    
    def optimize_multiple_indicators(self,
                                   indicator_types: List[IndicatorType],
                                   ohlcv_data: OHLCVData,
                                   config: OptimizationConfig = None) -> Dict[IndicatorType, OptimizationResult]:
        """
        Optimiert mehrere Indikatoren parallel.
        
        Args:
            indicator_types: Liste der zu optimierenden Indikatoren
            ohlcv_data: Historische OHLCV-Daten
            config: Optimierungs-Konfiguration
            
        Returns:
            Dictionary mit Optimierungsergebnissen pro Indikator
        """
        try:
            if config is None:
                config = OptimizationConfig()
            
            logger.info(f"Starte parallele Optimierung für {len(indicator_types)} Indikatoren")
            
            results = {}
            
            # Parallele Optimierung mit ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(indicator_types), self.cpu_cores)) as executor:
                future_to_indicator = {
                    executor.submit(self.optimize_indicator, indicator_type, ohlcv_data, config): indicator_type
                    for indicator_type in indicator_types
                }
                
                for future in as_completed(future_to_indicator):
                    indicator_type = future_to_indicator[future]
                    try:
                        result = future.result()
                        results[indicator_type] = result
                        logger.info(f"Optimierung für {indicator_type.value} abgeschlossen")
                    except Exception as e:
                        logger.exception(f"Fehler bei Optimierung von {indicator_type.value}: {e}")
                        results[indicator_type] = self._create_fallback_result(indicator_type, e)
            
            return results
            
        except Exception as e:
            logger.exception(f"Fehler bei Multi-Indikator-Optimierung: {e}")
            return {}
    
    def _bayesian_optimization(self,
                              indicator_type: IndicatorType,
                              ohlcv_data: OHLCVData,
                              config: OptimizationConfig,
                              parameter_space: Dict[str, IndicatorParameter]) -> OptimizationResult:
        """Bayesian Optimization mit Optuna"""
        try:
            study = optuna.create_study(direction='maximize')
            
            def objective(trial):
                # Parameter aus Trial extrahieren
                params = {}
                for param_name, param_def in parameter_space.items():
                    if param_def.parameter_type == "int":
                        params[param_name] = trial.suggest_int(
                            param_name, 
                            int(param_def.min_value), 
                            int(param_def.max_value),
                            step=int(param_def.step)
                        )
                    elif param_def.parameter_type == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_def.min_value,
                            param_def.max_value,
                            step=param_def.step
                        )
                    else:  # bool
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
                
                # Performance evaluieren
                return self._evaluate_parameters(indicator_type, params, ohlcv_data, config)
            
            # Optimierung durchführen
            study.optimize(objective, n_trials=config.max_iterations)
            
            # Bestes Ergebnis extrahieren
            best_params = study.best_params
            best_score = study.best_value
            
            # Backtest mit besten Parametern
            backtest_results = self._run_backtest(indicator_type, best_params, ohlcv_data, config)
            
            return OptimizationResult(
                indicator_type=indicator_type,
                optimal_parameters=best_params,
                performance_score=best_score,
                backtest_results=backtest_results,
                optimization_metadata={
                    "method": "bayesian",
                    "trials": len(study.trials),
                    "best_trial": study.best_trial.number
                },
                convergence_history=[trial.value for trial in study.trials if trial.value is not None]
            )
            
        except Exception as e:
            logger.exception(f"Bayesian Optimization fehlgeschlagen: {e}")
            raise
    
    def _genetic_optimization(self,
                             indicator_type: IndicatorType,
                             ohlcv_data: OHLCVData,
                             config: OptimizationConfig,
                             parameter_space: Dict[str, IndicatorParameter]) -> OptimizationResult:
        """Genetische Optimierung mit scipy.optimize.differential_evolution"""
        try:
            # Parameter-Bounds definieren
            bounds = []
            param_names = []
            
            for param_name, param_def in parameter_space.items():
                bounds.append((param_def.min_value, param_def.max_value))
                param_names.append(param_name)
            
            def objective_function(x):
                # Parameter-Dictionary erstellen
                params = {}
                for i, param_name in enumerate(param_names):
                    param_def = parameter_space[param_name]
                    if param_def.parameter_type == "int":
                        params[param_name] = int(round(x[i]))
                    else:
                        params[param_name] = x[i]
                
                # Negative Performance zurückgeben (da differential_evolution minimiert)
                return -self._evaluate_parameters(indicator_type, params, ohlcv_data, config)
            
            # Genetische Optimierung
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=config.max_iterations,
                popsize=config.population_size,
                workers=min(config.parallel_jobs, self.cpu_cores),
                seed=42
            )
            
            # Beste Parameter extrahieren
            best_params = {}
            for i, param_name in enumerate(param_names):
                param_def = parameter_space[param_name]
                if param_def.parameter_type == "int":
                    best_params[param_name] = int(round(result.x[i]))
                else:
                    best_params[param_name] = result.x[i]
            
            best_score = -result.fun
            
            # Backtest mit besten Parametern
            backtest_results = self._run_backtest(indicator_type, best_params, ohlcv_data, config)
            
            return OptimizationResult(
                indicator_type=indicator_type,
                optimal_parameters=best_params,
                performance_score=best_score,
                backtest_results=backtest_results,
                optimization_metadata={
                    "method": "genetic",
                    "iterations": result.nit,
                    "function_evaluations": result.nfev,
                    "success": result.success
                }
            )
            
        except Exception as e:
            logger.exception(f"Genetische Optimierung fehlgeschlagen: {e}")
            raise
    
    def _grid_search_optimization(self,
                                 indicator_type: IndicatorType,
                                 ohlcv_data: OHLCVData,
                                 config: OptimizationConfig,
                                 parameter_space: Dict[str, IndicatorParameter]) -> OptimizationResult:
        """Grid Search Optimierung"""
        try:
            # Parameter-Grid erstellen
            param_grids = {}
            for param_name, param_def in parameter_space.items():
                if param_def.parameter_type == "int":
                    param_grids[param_name] = list(range(
                        int(param_def.min_value),
                        int(param_def.max_value) + 1,
                        int(param_def.step)
                    ))
                elif param_def.parameter_type == "float":
                    param_grids[param_name] = np.arange(
                        param_def.min_value,
                        param_def.max_value + param_def.step,
                        param_def.step
                    ).tolist()
                else:  # bool
                    param_grids[param_name] = [True, False]
            
            # Alle Kombinationen generieren
            import itertools
            param_names = list(param_grids.keys())
            param_combinations = list(itertools.product(*[param_grids[name] for name in param_names]))
            
            # Limitiere Anzahl der Kombinationen
            if len(param_combinations) > config.max_iterations:
                import random
                param_combinations = random.sample(param_combinations, config.max_iterations)
            
            best_score = float('-inf')
            best_params = {}
            scores = []
            
            # Parallele Evaluation
            with ProcessPoolExecutor(max_workers=min(config.parallel_jobs, self.cpu_cores)) as executor:
                futures = []
                for combination in param_combinations:
                    params = dict(zip(param_names, combination))
                    future = executor.submit(self._evaluate_parameters, indicator_type, params, ohlcv_data, config)
                    futures.append((future, params))
                
                for future, params in futures:
                    try:
                        score = future.result()
                        scores.append(score)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                    except Exception as e:
                        logger.warning(f"Evaluation fehlgeschlagen für Parameter {params}: {e}")
                        scores.append(0.0)
            
            # Backtest mit besten Parametern
            backtest_results = self._run_backtest(indicator_type, best_params, ohlcv_data, config)
            
            return OptimizationResult(
                indicator_type=indicator_type,
                optimal_parameters=best_params,
                performance_score=best_score,
                backtest_results=backtest_results,
                optimization_metadata={
                    "method": "grid_search",
                    "total_combinations": len(param_combinations),
                    "evaluated_combinations": len(scores)
                },
                convergence_history=scores
            )
            
        except Exception as e:
            logger.exception(f"Grid Search Optimierung fehlgeschlagen: {e}")
            raise
    
    def _random_search_optimization(self,
                                   indicator_type: IndicatorType,
                                   ohlcv_data: OHLCVData,
                                   config: OptimizationConfig,
                                   parameter_space: Dict[str, IndicatorParameter]) -> OptimizationResult:
        """Random Search Optimierung"""
        try:
            import random
            
            best_score = float('-inf')
            best_params = {}
            scores = []
            
            for _ in range(config.max_iterations):
                # Zufällige Parameter generieren
                params = {}
                for param_name, param_def in parameter_space.items():
                    if param_def.parameter_type == "int":
                        params[param_name] = random.randint(
                            int(param_def.min_value),
                            int(param_def.max_value)
                        )
                    elif param_def.parameter_type == "float":
                        params[param_name] = random.uniform(
                            param_def.min_value,
                            param_def.max_value
                        )
                    else:  # bool
                        params[param_name] = random.choice([True, False])
                
                # Performance evaluieren
                try:
                    score = self._evaluate_parameters(indicator_type, params, ohlcv_data, config)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        
                except Exception as e:
                    logger.warning(f"Evaluation fehlgeschlagen für Parameter {params}: {e}")
                    scores.append(0.0)
            
            # Backtest mit besten Parametern
            backtest_results = self._run_backtest(indicator_type, best_params, ohlcv_data, config)
            
            return OptimizationResult(
                indicator_type=indicator_type,
                optimal_parameters=best_params,
                performance_score=best_score,
                backtest_results=backtest_results,
                optimization_metadata={
                    "method": "random_search",
                    "iterations": config.max_iterations
                },
                convergence_history=scores
            )
            
        except Exception as e:
            logger.exception(f"Random Search Optimierung fehlgeschlagen: {e}")
            raise
    
    def _evaluate_parameters(self,
                           indicator_type: IndicatorType,
                           parameters: Dict[str, Any],
                           ohlcv_data: OHLCVData,
                           config: OptimizationConfig) -> float:
        """Evaluiert Parameter-Set und gibt Performance-Score zurück"""
        try:
            # Cache-Key erstellen
            cache_key = (indicator_type, tuple(sorted(parameters.items())))
            if cache_key in self.performance_cache:
                return self.performance_cache[cache_key]
            
            # Indikator berechnen
            indicator_values = self._calculate_indicator(indicator_type, parameters, ohlcv_data)
            
            # Trading-Signale generieren
            signals = self._generate_signals(indicator_type, indicator_values, ohlcv_data)
            
            # Performance-Metriken berechnen
            performance_score = self._calculate_performance_metrics(signals, ohlcv_data, config)
            
            # Cache speichern
            self.performance_cache[cache_key] = performance_score
            
            return performance_score
            
        except Exception as e:
            logger.warning(f"Parameter-Evaluation fehlgeschlagen: {e}")
            return 0.0
    
    def _calculate_indicator(self,
                           indicator_type: IndicatorType,
                           parameters: Dict[str, Any],
                           ohlcv_data: OHLCVData) -> np.ndarray:
        """Berechnet Indikator-Werte mit gegebenen Parametern"""
        try:
            closes = np.array(ohlcv_data.close)
            highs = np.array(ohlcv_data.high)
            lows = np.array(ohlcv_data.low)
            volumes = np.array(ohlcv_data.volume) if hasattr(ohlcv_data, 'volume') else None
            
            if indicator_type == IndicatorType.RSI:
                return self._calculate_rsi(closes, parameters.get('period', 14))
            elif indicator_type == IndicatorType.SMA:
                return self._calculate_sma(closes, parameters.get('period', 20))
            elif indicator_type == IndicatorType.EMA:
                return self._calculate_ema(closes, parameters.get('period', 20))
            elif indicator_type == IndicatorType.MACD:
                return self._calculate_macd(closes, 
                                          parameters.get('fast_period', 12),
                                          parameters.get('slow_period', 26),
                                          parameters.get('signal_period', 9))
            elif indicator_type == IndicatorType.BOLLINGER_BANDS:
                return self._calculate_bollinger_bands(closes,
                                                     parameters.get('period', 20),
                                                     parameters.get('std_dev', 2.0))
            elif indicator_type == IndicatorType.STOCHASTIC:
                return self._calculate_stochastic(highs, lows, closes,
                                                parameters.get('k_period', 14),
                                                parameters.get('d_period', 3))
            elif indicator_type == IndicatorType.ATR:
                return self._calculate_atr(highs, lows, closes, parameters.get('period', 14))
            else:
                logger.warning(f"Indikator {indicator_type.value} nicht implementiert")
                return np.zeros(len(closes))
                
        except Exception as e:
            logger.exception(f"Indikator-Berechnung fehlgeschlagen: {e}")
            return np.zeros(len(ohlcv_data.close))
    
    def _calculate_rsi(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Berechnet RSI-Indikator"""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Padding für gleiche Länge
        return np.concatenate([np.full(len(closes) - len(rsi), 50), rsi])
    
    def _calculate_sma(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Berechnet Simple Moving Average"""
        sma = np.convolve(closes, np.ones(period)/period, mode='valid')
        return np.concatenate([np.full(period-1, closes[0]), sma])
    
    def _calculate_ema(self, closes: np.ndarray, period: int) -> np.ndarray:
        """Berechnet Exponential Moving Average"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(closes)
        ema[0] = closes[0]
        
        for i in range(1, len(closes)):
            ema[i] = alpha * closes[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_macd(self, closes: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
        """Berechnet MACD-Indikator"""
        ema_fast = self._calculate_ema(closes, fast)
        ema_slow = self._calculate_ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        
        return macd_line - signal_line  # MACD Histogram
    
    def _calculate_bollinger_bands(self, closes: np.ndarray, period: int, std_dev: float) -> np.ndarray:
        """Berechnet Bollinger Bands Position"""
        sma = self._calculate_sma(closes, period)
        rolling_std = np.array([np.std(closes[max(0, i-period+1):i+1]) for i in range(len(closes))])
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        # Position zwischen Bändern (0 = unteres Band, 1 = oberes Band)
        bb_position = (closes - lower_band) / (upper_band - lower_band + 1e-10)
        return np.clip(bb_position, 0, 1)
    
    def _calculate_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                            k_period: int, d_period: int) -> np.ndarray:
        """Berechnet Stochastic Oscillator"""
        k_values = np.zeros(len(closes))
        
        for i in range(k_period-1, len(closes)):
            highest_high = np.max(highs[i-k_period+1:i+1])
            lowest_low = np.min(lows[i-k_period+1:i+1])
            k_values[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        # %D ist SMA von %K
        d_values = self._calculate_sma(k_values, d_period)
        
        return k_values - d_values  # Stochastic Divergence
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """Berechnet Average True Range"""
        true_ranges = np.zeros(len(closes))
        
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges[i] = max(tr1, tr2, tr3)
        
        # ATR ist EMA der True Ranges
        return self._calculate_ema(true_ranges, period)
    
    def _generate_signals(self, indicator_type: IndicatorType, indicator_values: np.ndarray, ohlcv_data: OHLCVData) -> np.ndarray:
        """Generiert Trading-Signale basierend auf Indikator"""
        try:
            signals = np.zeros(len(indicator_values))
            
            if indicator_type == IndicatorType.RSI:
                # RSI Oversold/Overbought Signale
                signals[indicator_values < 30] = 1  # Buy
                signals[indicator_values > 70] = -1  # Sell
            elif indicator_type in [IndicatorType.SMA, IndicatorType.EMA]:
                # Moving Average Crossover
                closes = np.array(ohlcv_data.close)
                signals[closes > indicator_values] = 1  # Buy
                signals[closes < indicator_values] = -1  # Sell
            elif indicator_type == IndicatorType.MACD:
                # MACD Histogram Signale
                signals[indicator_values > 0] = 1  # Buy
                signals[indicator_values < 0] = -1  # Sell
            elif indicator_type == IndicatorType.BOLLINGER_BANDS:
                # Bollinger Bands Mean Reversion
                signals[indicator_values < 0.2] = 1  # Buy (near lower band)
                signals[indicator_values > 0.8] = -1  # Sell (near upper band)
            elif indicator_type == IndicatorType.STOCHASTIC:
                # Stochastic Divergence Signale
                signals[indicator_values > 0] = 1  # Buy
                signals[indicator_values < 0] = -1  # Sell
            
            return signals
            
        except Exception as e:
            logger.warning(f"Signal-Generierung fehlgeschlagen: {e}")
            return np.zeros(len(indicator_values))
    
    def _calculate_performance_metrics(self, signals: np.ndarray, ohlcv_data: OHLCVData, config: OptimizationConfig) -> float:
        """Berechnet Performance-Metriken für Signale"""
        try:
            closes = np.array(ohlcv_data.close)
            returns = np.diff(closes) / closes[:-1]
            
            # Signal-Returns berechnen
            signal_returns = signals[:-1] * returns
            
            if config.objective_function == "sharpe_ratio":
                return self._calculate_sharpe_ratio(signal_returns)
            elif config.objective_function == "profit_factor":
                return self._calculate_profit_factor(signal_returns)
            elif config.objective_function == "max_drawdown":
                return -self._calculate_max_drawdown(signal_returns)  # Negativ weil wir maximieren
            else:
                return np.sum(signal_returns)  # Total Return
                
        except Exception as e:
            logger.warning(f"Performance-Berechnung fehlgeschlagen: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Berechnet Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualisiert
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Berechnet Profit Factor"""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return float('inf') if len(profits) > 0 else 0.0
        
        return np.sum(profits) / abs(np.sum(losses))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Berechnet Maximum Drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _run_backtest(self, indicator_type: IndicatorType, parameters: Dict[str, Any], 
                     ohlcv_data: OHLCVData, config: OptimizationConfig) -> Dict[str, Any]:
        """Führt detaillierten Backtest mit optimalen Parametern durch"""
        try:
            # Indikator und Signale berechnen
            indicator_values = self._calculate_indicator(indicator_type, parameters, ohlcv_data)
            signals = self._generate_signals(indicator_type, indicator_values, ohlcv_data)
            
            # Performance-Metriken
            closes = np.array(ohlcv_data.close)
            returns = np.diff(closes) / closes[:-1]
            signal_returns = signals[:-1] * returns
            
            cumulative_returns = np.cumprod(1 + signal_returns)
            
            return {
                "total_return": cumulative_returns[-1] - 1,
                "sharpe_ratio": self._calculate_sharpe_ratio(signal_returns),
                "profit_factor": self._calculate_profit_factor(signal_returns),
                "max_drawdown": self._calculate_max_drawdown(signal_returns),
                "win_rate": len(signal_returns[signal_returns > 0]) / len(signal_returns[signal_returns != 0]) if len(signal_returns[signal_returns != 0]) > 0 else 0,
                "total_trades": len(signal_returns[signal_returns != 0]),
                "avg_return": np.mean(signal_returns),
                "volatility": np.std(signal_returns) * np.sqrt(252)
            }
            
        except Exception as e:
            logger.exception(f"Backtest fehlgeschlagen: {e}")
            return {"error": str(e)}
    
    def _validate_data(self, ohlcv_data: OHLCVData) -> bool:
        """Validiert OHLCV-Daten für Optimierung"""
        try:
            if not hasattr(ohlcv_data, 'close') or len(ohlcv_data.close) < 50:
                return False
            
            if not all(hasattr(ohlcv_data, attr) for attr in ['open', 'high', 'low', 'close']):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_fallback_result(self, indicator_type: IndicatorType, error: Exception) -> OptimizationResult:
        """Erstellt Fallback-Ergebnis bei Fehlern"""
        default_params = {}
        if indicator_type in self.indicator_parameters:
            for param_name, param_def in self.indicator_parameters[indicator_type].items():
                default_params[param_name] = param_def.default_value
        
        return OptimizationResult(
            indicator_type=indicator_type,
            optimal_parameters=default_params,
            performance_score=0.0,
            backtest_results={"error": str(error)},
            optimization_metadata={"error": str(error), "fallback": True}
        )
    
    def _initialize_indicator_parameters(self) -> Dict[IndicatorType, Dict[str, IndicatorParameter]]:
        """Initialisiert Parameter-Definitionen für alle Indikatoren"""
        return {
            IndicatorType.RSI: {
                "period": IndicatorParameter("period", 5, 50, 1, 14, "int")
            },
            IndicatorType.SMA: {
                "period": IndicatorParameter("period", 5, 200, 1, 20, "int")
            },
            IndicatorType.EMA: {
                "period": IndicatorParameter("period", 5, 200, 1, 20, "int")
            },
            IndicatorType.MACD: {
                "fast_period": IndicatorParameter("fast_period", 5, 20, 1, 12, "int"),
                "slow_period": IndicatorParameter("slow_period", 20, 50, 1, 26, "int"),
                "signal_period": IndicatorParameter("signal_period", 5, 15, 1, 9, "int")
            },
            IndicatorType.BOLLINGER_BANDS: {
                "period": IndicatorParameter("period", 10, 50, 1, 20, "int"),
                "std_dev": IndicatorParameter("std_dev", 1.0, 3.0, 0.1, 2.0, "float")
            },
            IndicatorType.STOCHASTIC: {
                "k_period": IndicatorParameter("k_period", 5, 30, 1, 14, "int"),
                "d_period": IndicatorParameter("d_period", 2, 10, 1, 3, "int")
            },
            IndicatorType.ATR: {
                "period": IndicatorParameter("period", 5, 50, 1, 14, "int")
            }
        }