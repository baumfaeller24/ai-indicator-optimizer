"""
Pattern Validator für automatische Qualitätskontrolle
Validiert Patterns auf Qualität, Konsistenz und Trading-Relevanz
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .historical_pattern_miner import MinedPattern
from .community_strategy_importer import ImportedStrategy


class ValidationLevel(Enum):
    """Validation-Level"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    EXPERT = "expert"


@dataclass
class ValidationResult:
    """Ergebnis der Pattern-Validierung"""
    pattern_id: str
    is_valid: bool
    quality_score: float  # 0.0 - 1.0
    validation_level: ValidationLevel
    
    # Detailed Results
    technical_score: float
    statistical_score: float
    trading_score: float
    consistency_score: float
    
    # Issues and Warnings
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Metrics
    validation_metrics: Dict[str, Any]
    
    def __post_init__(self):
        # Calculate overall quality score
        scores = [self.technical_score, self.statistical_score, 
                 self.trading_score, self.consistency_score]
        self.quality_score = np.mean([s for s in scores if s is not None])


class TechnicalValidator:
    """Technische Validierung von Patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_price_data(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert Preis-Daten Qualität"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            price_data = pattern.price_data.get("ohlcv", [])
            
            if not price_data:
                issues.append("No OHLCV data available")
                return 0.0, issues, metrics
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            
            # Required Columns Check
            required_cols = ["open", "high", "low", "close", "volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
                score -= 0.3
            
            # OHLC Consistency Check
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                
                # High >= Open, Close
                high_violations = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
                if high_violations > 0:
                    issues.append(f"High price violations: {high_violations}")
                    score -= 0.2
                
                # Low <= Open, Close
                low_violations = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
                if low_violations > 0:
                    issues.append(f"Low price violations: {low_violations}")
                    score -= 0.2
                
                # Price Range Check
                price_ranges = df["high"] - df["low"]
                zero_ranges = (price_ranges == 0).sum()
                
                if zero_ranges > len(df) * 0.1:  # More than 10% zero ranges
                    issues.append(f"Too many zero price ranges: {zero_ranges}")
                    score -= 0.1
                
                metrics["price_range_stats"] = {
                    "mean_range": float(price_ranges.mean()),
                    "std_range": float(price_ranges.std()),
                    "zero_ranges": int(zero_ranges)
                }
            
            # Volume Check
            if "volume" in df.columns:
                zero_volume = (df["volume"] == 0).sum()
                
                if zero_volume > len(df) * 0.2:  # More than 20% zero volume
                    issues.append(f"Too many zero volume candles: {zero_volume}")
                    score -= 0.1
                
                metrics["volume_stats"] = {
                    "mean_volume": float(df["volume"].mean()),
                    "zero_volume_pct": float(zero_volume / len(df))
                }
            
            # Data Completeness
            total_nulls = df.isnull().sum().sum()
            if total_nulls > 0:
                issues.append(f"Null values found: {total_nulls}")
                score -= 0.1
            
            # Timestamp Consistency (if available)
            if "timestamp" in df.columns:
                try:
                    timestamps = pd.to_datetime(df["timestamp"])
                    time_diffs = timestamps.diff().dropna()
                    
                    # Check for consistent intervals
                    if len(time_diffs) > 1:
                        mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]
                        irregular_intervals = (time_diffs != mode_diff).sum()
                        
                        if irregular_intervals > len(time_diffs) * 0.1:
                            issues.append(f"Irregular time intervals: {irregular_intervals}")
                            score -= 0.1
                        
                        metrics["timestamp_stats"] = {
                            "regular_intervals": int(len(time_diffs) - irregular_intervals),
                            "irregular_intervals": int(irregular_intervals)
                        }
                
                except Exception as e:
                    issues.append(f"Timestamp validation failed: {e}")
                    score -= 0.05
            
            metrics["data_quality_score"] = max(0.0, score)
            
        except Exception as e:
            issues.append(f"Price data validation failed: {e}")
            score = 0.0
        
        return max(0.0, score), issues, metrics
    
    def validate_indicators(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert Indikator-Daten"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            indicators = pattern.indicators
            
            if not indicators:
                issues.append("No indicator data available")
                return 0.5, issues, metrics  # Not critical
            
            # Check common indicators
            expected_indicators = ["RSI", "MACD", "SMA_20", "Volume"]
            available_indicators = list(indicators.keys())
            
            missing_indicators = [ind for ind in expected_indicators if ind not in available_indicators]
            if missing_indicators:
                issues.append(f"Missing indicators: {missing_indicators}")
                score -= 0.1 * len(missing_indicators) / len(expected_indicators)
            
            # Validate RSI
            if "RSI" in indicators:
                rsi_values = indicators["RSI"]
                if isinstance(rsi_values, list) and len(rsi_values) > 0:
                    rsi_array = np.array([v for v in rsi_values if v is not None])
                    
                    # RSI should be between 0 and 100
                    invalid_rsi = ((rsi_array < 0) | (rsi_array > 100)).sum()
                    if invalid_rsi > 0:
                        issues.append(f"Invalid RSI values: {invalid_rsi}")
                        score -= 0.1
                    
                    metrics["rsi_stats"] = {
                        "mean": float(np.mean(rsi_array)),
                        "std": float(np.std(rsi_array)),
                        "range": [float(np.min(rsi_array)), float(np.max(rsi_array))]
                    }
            
            # Validate MACD
            if "MACD" in indicators and isinstance(indicators["MACD"], dict):
                macd_data = indicators["MACD"]
                
                for key in ["macd", "signal", "histogram"]:
                    if key in macd_data and isinstance(macd_data[key], list):
                        values = np.array([v for v in macd_data[key] if v is not None])
                        
                        # Check for reasonable MACD values (more realistic bounds)
                        if len(values) > 0:
                            if np.any(np.abs(values) > 5.0):  # More realistic MACD bounds
                                issues.append(f"Extreme MACD {key} values detected")
                                score -= 0.05
            
            metrics["indicator_coverage"] = len(available_indicators) / len(expected_indicators)
            
        except Exception as e:
            issues.append(f"Indicator validation failed: {e}")
            score -= 0.2
        
        return max(0.0, score), issues, metrics


class StatisticalValidator:
    """Statistische Validierung von Patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_statistical_significance(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert statistische Signifikanz"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            price_data = pattern.price_data.get("ohlcv", [])
            
            if not price_data or len(price_data) < 10:
                issues.append("Insufficient data for statistical analysis")
                return 0.0, issues, metrics
            
            df = pd.DataFrame(price_data)
            
            if "close" not in df.columns:
                issues.append("No close price data for statistical analysis")
                return 0.0, issues, metrics
            
            close_prices = df["close"].values
            
            # Price Movement Analysis
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Volatility Check
            volatility = np.std(returns)
            if volatility < 0.0001:  # Very low volatility
                issues.append("Extremely low volatility detected")
                score -= 0.2
            elif volatility > 0.1:  # Very high volatility
                issues.append("Extremely high volatility detected")
                score -= 0.1
            
            # Trend Analysis
            if len(close_prices) > 5:
                # Linear trend test
                x = np.arange(len(close_prices))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)
                
                metrics["trend_analysis"] = {
                    "slope": float(slope),
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
                }
            
            # Normality Test (Shapiro-Wilk for small samples)
            if len(returns) >= 3:
                try:
                    if len(returns) <= 5000:  # Shapiro-Wilk limit
                        stat, p_val = stats.shapiro(returns)
                        metrics["normality_test"] = {
                            "statistic": float(stat),
                            "p_value": float(p_val),
                            "is_normal": p_val > 0.05
                        }
                except Exception:
                    pass
            
            # Autocorrelation Check
            if len(returns) > 10:
                try:
                    # Ensure sufficient data for autocorrelation
                    if len(returns) >= 3:
                        # Simple lag-1 autocorrelation
                        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    
                    if not np.isnan(autocorr):
                        metrics["autocorrelation"] = float(autocorr)
                        
                        if abs(autocorr) > 0.8:
                            issues.append("High autocorrelation detected - may indicate data issues")
                            score -= 0.1
                
                except Exception:
                    pass
            
            # Outlier Detection
            if len(returns) > 5:
                q75, q25 = np.percentile(returns, [75, 25])
                iqr = q75 - q25
                
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    outliers = ((returns < lower_bound) | (returns > upper_bound)).sum()
                    outlier_pct = outliers / len(returns)
                    
                    metrics["outlier_analysis"] = {
                        "outlier_count": int(outliers),
                        "outlier_percentage": float(outlier_pct),
                        "iqr": float(iqr)
                    }
                    
                    if outlier_pct > 0.1:  # More than 10% outliers
                        issues.append(f"High outlier percentage: {outlier_pct:.1%}")
                        score -= 0.1
            
            # Data Sufficiency
            data_points = len(close_prices)
            if data_points < 20:
                issues.append("Insufficient data points for reliable analysis")
                score -= 0.2
            elif data_points < 50:
                issues.append("Limited data points - results may be less reliable")
                score -= 0.1
            
            metrics["data_sufficiency"] = {
                "data_points": data_points,
                "sufficiency_level": "high" if data_points >= 100 else "medium" if data_points >= 50 else "low"
            }
            
        except Exception as e:
            issues.append(f"Statistical validation failed: {e}")
            score = 0.0
        
        return max(0.0, score), issues, metrics


class TradingValidator:
    """Trading-spezifische Validierung"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_trading_relevance(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert Trading-Relevanz des Patterns"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            # Pattern Type Validation
            valid_pattern_types = [
                "double_top", "double_bottom", "head_shoulders", 
                "triangle", "support_resistance", "breakout",
                "flag", "pennant", "wedge", "channel"
            ]
            
            if pattern.pattern_type not in valid_pattern_types:
                issues.append(f"Unknown pattern type: {pattern.pattern_type}")
                score -= 0.2
            
            # Confidence Score Check
            if pattern.confidence < 0.5:
                issues.append("Low pattern confidence score")
                score -= 0.3
            elif pattern.confidence < 0.7:
                issues.append("Moderate pattern confidence - consider additional validation")
                score -= 0.1
            
            # Timeframe Validation
            valid_timeframes = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]
            if pattern.timeframe not in valid_timeframes:
                issues.append(f"Invalid timeframe: {pattern.timeframe}")
                score -= 0.1
            
            # Market Context Validation
            market_context = pattern.market_context
            
            if not market_context:
                issues.append("Missing market context")
                score -= 0.1
            else:
                # Symbol validation
                symbol = market_context.get("symbol", "")
                if not symbol or len(symbol) < 3:
                    issues.append("Invalid or missing symbol")
                    score -= 0.1
                
                # Volatility check
                volatility = market_context.get("volatility")
                if volatility is not None:
                    if volatility < 0.0001:
                        issues.append("Extremely low market volatility")
                        score -= 0.1
                    elif volatility > 0.1:
                        issues.append("Extremely high market volatility")
                        score -= 0.1
            
            # Price Movement Validation
            price_data = pattern.price_data.get("ohlcv", [])
            if price_data:
                df = pd.DataFrame(price_data)
                
                if "close" in df.columns and len(df) > 1:
                    # Calculate price movement
                    price_change = (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
                    
                    # Pattern should show some meaningful price movement
                    if abs(price_change) < 0.001:  # Less than 0.1% movement
                        issues.append("Minimal price movement - pattern may not be significant")
                        score -= 0.2
                    
                    metrics["price_movement"] = {
                        "total_change_pct": float(price_change * 100),
                        "significance": "high" if abs(price_change) > 0.02 else "medium" if abs(price_change) > 0.005 else "low"
                    }
            
            # Pattern Duration Validation
            duration = pattern.end_time - pattern.start_time
            duration_hours = duration.total_seconds() / 3600
            
            # Pattern should have reasonable duration
            if duration_hours < 1:
                issues.append("Pattern duration too short")
                score -= 0.2
            elif duration_hours > 24 * 30:  # More than 30 days
                issues.append("Pattern duration very long - may be less actionable")
                score -= 0.1
            
            metrics["pattern_duration"] = {
                "hours": float(duration_hours),
                "days": float(duration_hours / 24)
            }
            
        except Exception as e:
            issues.append(f"Trading validation failed: {e}")
            score = 0.0
        
        return max(0.0, score), issues, metrics
    
    def validate_risk_reward_potential(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert Risk-Reward Potenzial"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            price_data = pattern.price_data.get("ohlcv", [])
            
            if not price_data:
                issues.append("No price data for risk-reward analysis")
                return 0.0, issues, metrics
            
            df = pd.DataFrame(price_data)
            
            if "high" not in df.columns or "low" not in df.columns:
                issues.append("Missing high/low data for risk-reward analysis")
                return 0.0, issues, metrics
            
            # Calculate potential risk-reward based on pattern
            pattern_high = df["high"].max()
            pattern_low = df["low"].min()
            pattern_range = pattern_high - pattern_low
            
            if pattern_range <= 0:
                issues.append("Invalid price range for risk-reward calculation")
                return 0.0, issues, metrics
            
            # Estimate potential targets and stops based on pattern type
            if pattern.pattern_type in ["double_top", "head_shoulders"]:
                # Bearish patterns
                entry_price = pattern_high * 0.99  # Entry below resistance
                target_price = pattern_low  # Target at support
                stop_price = pattern_high * 1.01  # Stop above resistance
                
            elif pattern.pattern_type in ["double_bottom"]:
                # Bullish patterns
                entry_price = pattern_low * 1.01  # Entry above support
                target_price = pattern_high  # Target at resistance
                stop_price = pattern_low * 0.99  # Stop below support
                
            else:
                # Generic calculation
                entry_price = (pattern_high + pattern_low) / 2
                target_price = pattern_high if pattern.confidence > 0.7 else pattern_low
                stop_price = pattern_low if target_price == pattern_high else pattern_high
            
            # Calculate risk-reward ratio
            potential_profit = abs(target_price - entry_price)
            potential_loss = abs(stop_price - entry_price)
            
            if potential_loss > 0 and abs(potential_loss) > 1e-8:  # Avoid division by zero
                risk_reward_ratio = potential_profit / potential_loss
                
                metrics["risk_reward_analysis"] = {
                    "entry_price": float(entry_price),
                    "target_price": float(target_price),
                    "stop_price": float(stop_price),
                    "potential_profit_pct": float((potential_profit / entry_price) * 100),
                    "potential_loss_pct": float((potential_loss / entry_price) * 100),
                    "risk_reward_ratio": float(risk_reward_ratio)
                }
                
                # Score based on risk-reward ratio
                if risk_reward_ratio < 1.0:
                    issues.append(f"Poor risk-reward ratio: {risk_reward_ratio:.2f}")
                    score -= 0.3
                elif risk_reward_ratio < 1.5:
                    issues.append(f"Suboptimal risk-reward ratio: {risk_reward_ratio:.2f}")
                    score -= 0.1
                
                # Check if potential profit/loss is reasonable
                profit_pct = (potential_profit / entry_price) * 100
                loss_pct = (potential_loss / entry_price) * 100
                
                if profit_pct < 0.5:  # Less than 0.5% potential profit
                    issues.append("Very small potential profit")
                    score -= 0.2
                
                if loss_pct > 10:  # More than 10% potential loss
                    issues.append("Very large potential loss")
                    score -= 0.2
            
            else:
                issues.append("Cannot calculate risk-reward - invalid stop loss")
                score -= 0.5
            
        except Exception as e:
            issues.append(f"Risk-reward validation failed: {e}")
            score = 0.0
        
        return max(0.0, score), issues, metrics


class ConsistencyValidator:
    """Konsistenz-Validierung zwischen Pattern-Komponenten"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_internal_consistency(self, pattern: MinedPattern) -> Tuple[float, List[str], Dict[str, Any]]:
        """Validiert interne Konsistenz des Patterns"""
        
        issues = []
        metrics = {}
        score = 1.0
        
        try:
            # Timestamp Consistency
            if pattern.start_time >= pattern.end_time:
                issues.append("Invalid time range: start_time >= end_time")
                score -= 0.3
            
            # Pattern Features Consistency
            pattern_features = pattern.pattern_features
            
            if pattern_features:
                # Confidence consistency
                feature_confidence = pattern_features.get("confidence")
                if feature_confidence is not None and abs(feature_confidence - pattern.confidence) > 0.1:
                    issues.append("Inconsistent confidence scores between pattern and features")
                    score -= 0.1
                
                # Pattern type consistency
                feature_pattern_type = pattern_features.get("pattern_type")
                if feature_pattern_type and feature_pattern_type != pattern.pattern_type:
                    issues.append("Inconsistent pattern types")
                    score -= 0.2
            
            # Market Context Consistency
            market_context = pattern.market_context
            
            if market_context:
                # Symbol consistency
                context_symbol = market_context.get("symbol")
                if context_symbol and context_symbol != pattern.symbol:
                    issues.append("Inconsistent symbols between pattern and market context")
                    score -= 0.1
                
                # Timeframe consistency
                context_timeframe = market_context.get("timeframe")
                if context_timeframe and context_timeframe != pattern.timeframe:
                    issues.append("Inconsistent timeframes")
                    score -= 0.1
            
            # Price Data Consistency
            price_data = pattern.price_data
            
            if price_data and "ohlcv" in price_data:
                ohlcv_data = price_data["ohlcv"]
                
                if ohlcv_data:
                    # Check if price range matches pattern duration
                    df = pd.DataFrame(ohlcv_data)
                    
                    if "timestamp" in df.columns:
                        try:
                            data_start = pd.to_datetime(df["timestamp"].iloc[0])
                            data_end = pd.to_datetime(df["timestamp"].iloc[-1])
                            
                            # Allow some tolerance for timestamp differences
                            start_diff = abs((data_start - pattern.start_time).total_seconds())
                            end_diff = abs((data_end - pattern.end_time).total_seconds())
                            
                            if start_diff > 3600:  # More than 1 hour difference
                                issues.append("Large discrepancy in start times")
                                score -= 0.1
                            
                            if end_diff > 3600:
                                issues.append("Large discrepancy in end times")
                                score -= 0.1
                        
                        except Exception:
                            issues.append("Could not validate timestamp consistency")
                            score -= 0.05
            
            # Indicator Consistency
            indicators = pattern.indicators
            
            if indicators and price_data:
                # Check if indicator length matches price data length
                ohlcv_data = price_data.get("ohlcv", [])
                
                if ohlcv_data:
                    price_length = len(ohlcv_data)
                    
                    for indicator_name, indicator_values in indicators.items():
                        if isinstance(indicator_values, list):
                            indicator_length = len(indicator_values)
                            
                            # Allow some tolerance for indicator calculation lag
                            if abs(indicator_length - price_length) > 20:
                                issues.append(f"Length mismatch for {indicator_name}: {indicator_length} vs {price_length}")
                                score -= 0.05
            
            metrics["consistency_checks"] = {
                "timestamp_valid": pattern.start_time < pattern.end_time,
                "has_price_data": bool(price_data),
                "has_indicators": bool(indicators),
                "has_market_context": bool(market_context)
            }
            
        except Exception as e:
            issues.append(f"Consistency validation failed: {e}")
            score = 0.0
        
        return max(0.0, score), issues, metrics


class PatternValidator:
    """
    Hauptklasse für Pattern-Validierung
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Validators
        self.technical_validator = TechnicalValidator()
        self.statistical_validator = StatisticalValidator()
        self.trading_validator = TradingValidator()
        self.consistency_validator = ConsistencyValidator()
        
        # Validation Thresholds by Level
        self.thresholds = {
            ValidationLevel.BASIC: {
                "min_quality_score": 0.3,
                "min_technical_score": 0.2,
                "min_trading_score": 0.3
            },
            ValidationLevel.STANDARD: {
                "min_quality_score": 0.6,
                "min_technical_score": 0.5,
                "min_trading_score": 0.6,
                "min_statistical_score": 0.4
            },
            ValidationLevel.STRICT: {
                "min_quality_score": 0.8,
                "min_technical_score": 0.7,
                "min_trading_score": 0.8,
                "min_statistical_score": 0.6,
                "min_consistency_score": 0.7
            },
            ValidationLevel.EXPERT: {
                "min_quality_score": 0.9,
                "min_technical_score": 0.8,
                "min_trading_score": 0.9,
                "min_statistical_score": 0.8,
                "min_consistency_score": 0.8
            }
        }
        
        self.logger.info(f"PatternValidator initialized with {validation_level.value} level")
    
    def validate_pattern(self, pattern: MinedPattern) -> ValidationResult:
        """Validiert ein einzelnes Pattern"""
        
        try:
            all_issues = []
            all_warnings = []
            all_recommendations = []
            all_metrics = {}
            
            # Technical Validation
            technical_score, tech_issues, tech_metrics = self.technical_validator.validate_price_data(pattern)
            indicator_score, ind_issues, ind_metrics = self.technical_validator.validate_indicators(pattern)
            
            final_technical_score = (technical_score + indicator_score) / 2
            all_issues.extend(tech_issues)
            all_issues.extend(ind_issues)
            all_metrics.update({"technical": tech_metrics, "indicators": ind_metrics})
            
            # Statistical Validation
            statistical_score = None
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.EXPERT]:
                statistical_score, stat_issues, stat_metrics = self.statistical_validator.validate_statistical_significance(pattern)
                all_issues.extend(stat_issues)
                all_metrics["statistical"] = stat_metrics
            
            # Trading Validation
            trading_score, trade_issues, trade_metrics = self.trading_validator.validate_trading_relevance(pattern)
            risk_score, risk_issues, risk_metrics = self.trading_validator.validate_risk_reward_potential(pattern)
            
            final_trading_score = (trading_score + risk_score) / 2
            all_issues.extend(trade_issues)
            all_issues.extend(risk_issues)
            all_metrics.update({"trading": trade_metrics, "risk_reward": risk_metrics})
            
            # Consistency Validation
            consistency_score = None
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.EXPERT]:
                consistency_score, cons_issues, cons_metrics = self.consistency_validator.validate_internal_consistency(pattern)
                all_issues.extend(cons_issues)
                all_metrics["consistency"] = cons_metrics
            
            # Determine Critical Issues vs Warnings
            critical_issues = []
            warnings = []
            
            for issue in all_issues:
                if any(keyword in issue.lower() for keyword in ["failed", "missing", "invalid", "no data"]):
                    critical_issues.append(issue)
                else:
                    warnings.append(issue)
            
            # Generate Recommendations
            recommendations = self._generate_recommendations(
                final_technical_score, statistical_score, final_trading_score, consistency_score
            )
            
            # Create Validation Result
            result = ValidationResult(
                pattern_id=pattern.pattern_id,
                is_valid=self._determine_validity(final_technical_score, statistical_score, final_trading_score, consistency_score),
                quality_score=0.0,  # Will be calculated in __post_init__
                validation_level=self.validation_level,
                technical_score=final_technical_score,
                statistical_score=statistical_score,
                trading_score=final_trading_score,
                consistency_score=consistency_score,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations,
                validation_metrics=all_metrics
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pattern validation failed: {e}")
            
            return ValidationResult(
                pattern_id=pattern.pattern_id,
                is_valid=False,
                quality_score=0.0,
                validation_level=self.validation_level,
                technical_score=0.0,
                statistical_score=0.0,
                trading_score=0.0,
                consistency_score=0.0,
                critical_issues=[f"Validation failed: {str(e)}"],
                warnings=[],
                recommendations=["Review pattern data quality"],
                validation_metrics={}
            )
    
    def _determine_validity(self, technical_score: float, statistical_score: Optional[float], 
                          trading_score: float, consistency_score: Optional[float]) -> bool:
        """Bestimmt ob Pattern gültig ist basierend auf Scores"""
        
        thresholds = self.thresholds[self.validation_level]
        
        # Technical Score Check
        if technical_score < thresholds.get("min_technical_score", 0.0):
            return False
        
        # Trading Score Check
        if trading_score < thresholds.get("min_trading_score", 0.0):
            return False
        
        # Statistical Score Check (if required)
        if statistical_score is not None:
            if statistical_score < thresholds.get("min_statistical_score", 0.0):
                return False
        
        # Consistency Score Check (if required)
        if consistency_score is not None:
            if consistency_score < thresholds.get("min_consistency_score", 0.0):
                return False
        
        # Overall Quality Score Check
        scores = [s for s in [technical_score, statistical_score, trading_score, consistency_score] if s is not None]
        overall_score = np.mean(scores) if scores else 0.0
        
        return overall_score >= thresholds.get("min_quality_score", 0.0)
    
    def _generate_recommendations(self, technical_score: float, statistical_score: Optional[float],
                                trading_score: float, consistency_score: Optional[float]) -> List[str]:
        """Generiert Empfehlungen basierend auf Scores"""
        
        recommendations = []
        
        if technical_score < 0.7:
            recommendations.append("Improve data quality - check for missing or invalid price data")
        
        if statistical_score is not None and statistical_score < 0.6:
            recommendations.append("Increase sample size or verify statistical significance")
        
        if trading_score < 0.7:
            recommendations.append("Enhance trading relevance - verify pattern type and market context")
        
        if consistency_score is not None and consistency_score < 0.7:
            recommendations.append("Fix data consistency issues between pattern components")
        
        if not recommendations:
            recommendations.append("Pattern meets validation criteria - ready for use")
        
        return recommendations
    
    def validate_pattern_batch(self, patterns: List[MinedPattern]) -> List[ValidationResult]:
        """Validiert mehrere Patterns"""
        
        results = []
        
        self.logger.info(f"Validating {len(patterns)} patterns with {self.validation_level.value} level")
        
        for i, pattern in enumerate(patterns):
            try:
                result = self.validate_pattern(pattern)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Validated {i + 1}/{len(patterns)} patterns")
            
            except Exception as e:
                self.logger.error(f"Batch validation failed for pattern {pattern.pattern_id}: {e}")
        
        # Summary Statistics
        valid_count = sum(1 for r in results if r.is_valid)
        avg_quality = np.mean([r.quality_score for r in results])
        
        self.logger.info(f"Validation complete: {valid_count}/{len(results)} valid patterns, avg quality: {avg_quality:.2f}")
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Gibt Validierungs-Zusammenfassung zurück"""
        
        if not results:
            return {"message": "No validation results available"}
        
        valid_results = [r for r in results if r.is_valid]
        
        summary = {
            "total_patterns": len(results),
            "valid_patterns": len(valid_results),
            "validation_rate": len(valid_results) / len(results),
            "validation_level": self.validation_level.value,
            
            "quality_scores": {
                "mean": float(np.mean([r.quality_score for r in results])),
                "std": float(np.std([r.quality_score for r in results])),
                "min": float(np.min([r.quality_score for r in results])),
                "max": float(np.max([r.quality_score for r in results]))
            },
            
            "score_breakdown": {
                "technical": float(np.mean([r.technical_score for r in results])),
                "statistical": float(np.mean([r.statistical_score for r in results if r.statistical_score is not None])) if any(r.statistical_score is not None for r in results) else None,
                "trading": float(np.mean([r.trading_score for r in results])),
                "consistency": float(np.mean([r.consistency_score for r in results if r.consistency_score is not None])) if any(r.consistency_score is not None for r in results) else None
            },
            
            "common_issues": self._get_common_issues(results),
            "recommendations": self._get_common_recommendations(results)
        }
        
        return summary
    
    def _get_common_issues(self, results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Findet häufige Issues"""
        
        issue_counts = {}
        
        for result in results:
            for issue in result.critical_issues + result.warnings:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"issue": issue, "count": count, "frequency": count / len(results)}
            for issue, count in sorted_issues[:10]  # Top 10
        ]
    
    def _get_common_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Findet häufige Empfehlungen"""
        
        rec_counts = {}
        
        for result in results:
            for rec in result.recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        # Return most common recommendations
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [rec for rec, count in sorted_recs[:5]]  # Top 5


# Convenience Functions
def quick_pattern_validation(pattern: MinedPattern, 
                           level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
    """Schnelle Pattern-Validierung"""
    
    validator = PatternValidator(level)
    return validator.validate_pattern(pattern)


def validate_pattern_library(patterns: List[MinedPattern],
                           level: ValidationLevel = ValidationLevel.STANDARD) -> Tuple[List[ValidationResult], Dict[str, Any]]:
    """Validiert komplette Pattern-Library"""
    
    validator = PatternValidator(level)
    results = validator.validate_pattern_batch(patterns)
    summary = validator.get_validation_summary(results)
    
    return results, summary