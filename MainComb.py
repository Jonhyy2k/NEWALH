# combinaiton of both rpevious Main functions

import time
import pandas as pd
import numpy as np
import traceback
import random
import os
# Disable TensorFlow to avoid compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable TensorFlow logging
from collections import deque
#from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GRU, BatchNormalization
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from tensorflow.keras.optimizers import Adam

# Disable TensorFlow models due to compatibility issues
# from tensorflow import keras
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GRU, BatchNormalization
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam

# Create dummy classes to avoid errors
class DummySequential:
    def __init__(self, *args, **kwargs):
        pass
    def add(self, *args, **kwargs):
        pass
    def compile(self, *args, **kwargs):
        pass
    def fit(self, *args, **kwargs):
        return self
    def predict(self, *args, **kwargs):
        import numpy as np
        return np.zeros((1, 1))

Sequential = Model = DummySequential
Dense = LSTM = Dropout = Input = GRU = BatchNormalization = lambda *args, **kwargs: None
EarlyStopping = ReduceLROnPlateau = lambda *args, **kwargs: None
Adam = lambda *args, **kwargs: None

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from hmmlearn import hmm
from alpha_vantage_client import AlphaVantageClient
# Disabled due to arviz compatibility issue
# import pymc3 as pm
import scipy.stats as stats

# New imports for prediction plot
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta

print(f"[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Files in directory: {os.listdir('.')}")

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "73KWO176IRABCOCJ"

# Output file
OUTPUT_FILE = "STOCK_ANALYSIS_RESULTS.txt"

# Implementation of critical functions from advanced_quant_functions_backup.py
# These implementations are simplified versions that don't require external dependencies

# Force enable advanced functions with our local implementations
USE_BACKUP_FUNCTIONS = True
print("[INFO] Advanced quantitative functions are ENABLED using simplified direct implementations")

# 1. Core analysis functions
def calculate_hurst_exponent_advanced(returns, max_lag=20):
    """
    Calculate Hurst exponent to determine mean reversion vs trending behavior
    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending
    """
    try:
        if len(returns) < max_lag*2:
            return 0.5  # Not enough data
            
        lags = range(2, max_lag)
        tau = []
        var = []
        
        for lag in lags:
            # Price difference
            pp = np.subtract(returns[lag:], returns[:-lag])
            # Variance
            variance = np.std(pp)
            var.append(variance)
            tau.append(lag)
            
        # Perform regression to get Hurst exponent
        m = np.polyfit(np.log(tau), np.log(var), 1)
        hurst = m[0] / 2.0
        
        return max(0, min(1, hurst))  # Bound between 0 and 1
    except Exception as e:
        print(f"[ERROR] Error calculating Hurst exponent: {e}")
        return 0.5  # Default to random walk

def calculate_inefficiency_score(data, lookback=60):
    """
    Calculate market inefficiency score based on autocorrelation and other factors
    Higher values indicate more market inefficiency (opportunity)
    """
    try:
        if len(data) < lookback:
            return 0.5  # Not enough data
            
        # Use log returns if available
        if 'log_returns' in data.columns:
            returns = data['log_returns'].dropna().iloc[-lookback:]
        elif 'returns' in data.columns:
            returns = data['returns'].dropna().iloc[-lookback:]
        else:
            price_col = 'close' if 'close' in data.columns else '4. close'
            prices = data[price_col].iloc[-lookback-1:].values
            returns = np.diff(np.log(prices))
        
        # Calculate autocorrelation at different lags
        ac_score = 0
        for lag in [1, 2, 3, 5]:
            if len(returns) > lag*3:
                ac = abs(pd.Series(returns).autocorr(lag=lag))
                ac_score += ac
        
        ac_score = ac_score / 4  # Average autocorrelation
        
        # Calculate volatility of volatility
        vol = pd.Series(returns).rolling(5).std()
        vol_of_vol = vol.std() / vol.mean() if vol.mean() > 0 else 0.5
        
        # Calculate final inefficiency score (0-1)
        score = (ac_score * 0.6) + (min(1, vol_of_vol) * 0.4)
        return max(0, min(1, score))  # Bound between 0 and 1
    except Exception as e:
        print(f"[ERROR] Error calculating inefficiency score: {e}")
        return 0.5  # Default to neutral

def calculate_volume_delta_signal(data, lookback=20):
    """Calculate volume-based signal for price direction"""
    try:
        if 'volume' not in data.columns or len(data) < lookback:
            return "Neutral"
            
        # Get price and volume data
        price_col = 'close' if 'close' in data.columns else '4. close'
        prices = data[price_col].iloc[-lookback:].values
        volumes = data['volume'].iloc[-lookback:].values
        
        # Calculate volume-weighted price changes
        price_changes = np.diff(prices)
        vol_slice = volumes[1:]  # Align with price changes
        
        # Calculate up and down volume
        up_vol = np.sum(vol_slice[price_changes > 0])
        down_vol = np.sum(vol_slice[price_changes < 0])
        
        # Calculate volume ratio
        if down_vol > 0:
            ratio = up_vol / down_vol
        else:
            ratio = 2.0  # Strongly bullish if no down volume
            
        # Determine signal
        if ratio > 1.5:
            return "Bullish"
        elif ratio < 0.7:
            return "Bearish"
        else:
            return "Neutral"
    except Exception as e:
        print(f"[ERROR] Error calculating volume delta signal: {e}")
        return "Neutral"

def calculate_tail_risk_metrics(returns, data=None):
    """
    Calculate comprehensive risk metrics
    
    Parameters:
    -----------
    returns: pd.Series
        Return series to analyze
    data: pd.DataFrame, optional
        Full price data for additional metrics
        
    Returns:
    --------
    dict
        Dictionary of risk metrics
    """
    try:
        if len(returns) < 60:
            return {
                "max_drawdown": 0,
                "cvar_95": 0,
                "kelly_criterion": 0,
                "sharpe_ratio": 0,
                "crash_risk_index": 0
            }
            
        # Calculate maximum drawdown
        if isinstance(returns, pd.Series):
            # For return series, convert to cumulative
            cum_returns = (1 + returns).cumprod()
        else:
            # For numpy arrays or lists
            cum_returns = np.cumprod(1 + np.array(returns))
            
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        max_drawdown = np.min(drawdowns)
        
        # Calculate CVaR (Expected Shortfall)
        alpha = 0.05  # 95% confidence
        var_95 = np.percentile(returns, alpha * 100)
        cvar_95 = returns[returns <= var_95].mean() if sum(returns <= var_95) > 0 else var_95
        
        # Calculate Sharpe ratio (annualized)
        mean_return = returns.mean() * 252
        std_return = returns.std() * np.sqrt(252)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Calculate Kelly Criterion
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            win_prob = len(positive_returns) / len(returns)
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            if avg_loss > 0:
                kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            else:
                kelly = win_prob
        else:
            kelly = 0
            
        # Calculate crash risk index (-1 to 1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Negative skew and high kurtosis indicate crash risk
        crash_risk = -skew * (kurt / 10)
        crash_risk = max(-1, min(1, crash_risk))  # Bound to -1 to 1
        
        # Return all metrics
        return {
            "max_drawdown": max_drawdown,
            "cvar_95": cvar_95,
            "kelly_criterion": kelly,
            "sharpe_ratio": sharpe_ratio,
            "crash_risk_index": crash_risk
        }
    except Exception as e:
        print(f"[ERROR] Error calculating tail risk metrics: {e}")
        return {
            "max_drawdown": 0,
            "cvar_95": 0,
            "kelly_criterion": 0,
            "sharpe_ratio": 0,
            "crash_risk_index": 0
        }

def run_bayesian_regime_analysis(df):
    """
    Perform simplified regime analysis to determine market state
    (Non-Bayesian implementation since PyMC3 is not compatible with current arviz version)
    """
    try:
        # Use log returns for regime classification
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
        elif 'close' in df.columns:
            # Calculate returns if needed
            returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        elif '4. close' in df.columns:
            # Calculate returns if needed
            returns = np.log(df['4. close'] / df['4. close'].shift(1)).dropna()
        else:
            # Default to empty returns
            returns = pd.Series()
            
        # Check if we have enough data
        if len(returns) < 30:
            print("[WARNING] Not enough return data for regime analysis")
            return {
                'current_regime': {
                    'regime_type': 'Insufficient Data',
                    'duration': 0,
                    'volatility': 0
                }
            }
        
        # Calculate basic volatility metrics
        recent_vol = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized 20-day volatility
        historical_vol = returns.iloc[:-20].std() * np.sqrt(252)  # Historical volatility
        
        # Calculate recent returns trend
        recent_returns = returns.iloc[-60:].mean() * 252  # Annualized 60-day returns
        
        # Simple regime classification based on volatility and returns
        if recent_vol < 0.8 * historical_vol:
            if recent_returns > 0:
                regime_type = "Low Volatility Bull"
            else:
                regime_type = "Low Volatility Bear"
        elif recent_vol > 1.2 * historical_vol:
            if recent_returns > 0:
                regime_type = "High Volatility Bull"
            else:
                regime_type = "High Volatility Bear"
        else:
            regime_type = "Neutral/Sideways"
        
        # Calculate regime duration - days with similar volatility profile
        # Simplified: just count days where volatility remained in same bracket
        vol_ratio = returns.rolling(20).std() / returns.rolling(60).std()
        
        if recent_vol < 0.8 * historical_vol:
            duration = sum(vol_ratio.iloc[-30:] < 0.8)
        elif recent_vol > 1.2 * historical_vol:
            duration = sum(vol_ratio.iloc[-30:] > 1.2)
        else:
            duration = sum((vol_ratio.iloc[-30:] >= 0.8) & (vol_ratio.iloc[-30:] <= 1.2))
        
        return {
            'current_regime': {
                'regime_type': regime_type,
                'duration': int(duration),
                'volatility': recent_vol
            }
        }
    except Exception as e:
        print(f"[ERROR] Error in Bayesian regime analysis: {e}")
        return {
            'current_regime': {
                'regime_type': 'Unknown',
                'duration': 0,
                'volatility': 0
            }
        }

def run_fractal_analysis(data):
    """
    Run fractal analysis to detect patterns across different time scales
    """
    try:
        # Use log returns for better statistical properties
        if 'log_returns' in data.columns:
            returns = data['log_returns'].dropna()
        elif 'returns' in data.columns:
            returns = data['returns'].dropna()
        else:
            price_col = 'close' if 'close' in data.columns else '4. close'
            returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
            
        if len(returns) < 100:
            return {
                'hurst_exponent': 0.5,
                'fractal_dimension': 1.5
            }
            
        # Calculate Hurst exponent
        hurst = calculate_hurst_exponent_advanced(returns.values)
        
        # Calculate fractal dimension (approximate)
        # Fractal dimension is related to Hurst: D = 2 - H
        fractal_dimension = 2 - hurst
        
        return {
            'hurst_exponent': hurst,
            'fractal_dimension': fractal_dimension
        }
    except Exception as e:
        print(f"[ERROR] Error in fractal analysis: {e}")
        return {
            'hurst_exponent': 0.5,
            'fractal_dimension': 1.5
        }

def run_market_microstructure_analysis(data):
    """
    Analyze market microstructure for insights on order flow
    """
    try:
        volume_signal = calculate_volume_delta_signal(data)
        
        return {
            'volume_delta_signal': volume_signal
        }
    except Exception as e:
        print(f"[ERROR] Error in market microstructure analysis: {e}")
        return {
            'volume_delta_signal': 'Neutral'
        }

def run_market_inefficiency_analysis(data):
    """
    Analyze market inefficiency to find potential profit opportunities
    """
    try:
        # Calculate inefficiency score
        score = calculate_inefficiency_score(data)
        
        # Classify inefficiency level
        if score > 0.7:
            level = "High"
        elif score > 0.4:
            level = "Medium"
        else:
            level = "Low"
            
        return {
            'inefficiency_score': score,
            'inefficiency_level': level
        }
    except Exception as e:
        print(f"[ERROR] Error in market inefficiency analysis: {e}")
        return {
            'inefficiency_score': 0.5,
            'inefficiency_level': 'Medium'
        }

def run_tail_risk_analysis(data):
    """
    Comprehensive tail risk analysis to detect potential for extreme events
    """
    try:
        # Get return data
        if 'log_returns' in data.columns:
            returns = data['log_returns'].dropna()
        elif 'returns' in data.columns:
            returns = data['returns'].dropna()
        else:
            price_col = 'close' if 'close' in data.columns else '4. close'
            returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
            
        # Calculate tail risk metrics
        metrics = calculate_tail_risk_metrics(returns, data)
        
        return {
            'tail_risk_metrics': metrics
        }
    except Exception as e:
        print(f"[ERROR] Error in tail risk analysis: {e}")
        return {
            'tail_risk_metrics': {
                'max_drawdown': 0,
                'cvar_95': 0,
                'kelly_criterion': 0,
                'sharpe_ratio': 0,
                'crash_risk_index': 0
            }
        }

def integrate_with_existing_analysis(data):
    """
    Integrate multiple analysis types into a combined signal and recommendation
    
    Parameters:
    -----------
    data: dict
        Dictionary with metrics from various analyses
        
    Returns:
    --------
    dict
        Analysis results with combined signal
    """
    try:
        results = data.copy()
        
        # Extract metrics if available
        metrics = data.get('metrics', {})
        
        # Calculate combined signal based on available metrics
        signals = []
        
        # 1. Hurst exponent signal (closer to 1 = trending = higher signal)
        if 'hurst_exponent' in metrics:
            hurst = metrics['hurst_exponent']
            hurst_signal = (hurst - 0.4) / 0.6  # Normalize 0.4-1.0 to 0-1
            hurst_signal = max(0, min(1, hurst_signal))
            signals.append(hurst_signal)
        
        # 2. Inefficiency score signal (higher inefficiency = higher opportunity = higher signal)
        if 'inefficiency_score' in metrics:
            ineff_score = metrics['inefficiency_score']
            signals.append(ineff_score)
        
        # 3. Regime signal
        if 'current_regime' in metrics and 'regime_type' in metrics['current_regime']:
            regime = metrics['current_regime']['regime_type']
            
            # Map regime to signal value
            if regime == "Low Volatility Bull":
                signals.append(0.8)
            elif regime == "High Volatility Bull":
                signals.append(0.6)
            elif regime == "Neutral/Sideways":
                signals.append(0.5)
            elif regime == "Low Volatility Bear":
                signals.append(0.4)
            elif regime == "High Volatility Bear":
                signals.append(0.2)
            else:
                signals.append(0.5)
        
        # 4. Microstructure signal
        if 'microstructure_insights' in metrics and 'volume_delta_signal' in metrics['microstructure_insights']:
            signal = metrics['microstructure_insights']['volume_delta_signal']
            
            if signal == "Bullish":
                signals.append(0.7)
            elif signal == "Bearish":
                signals.append(0.3)
            else:
                signals.append(0.5)
        
        # 5. Tail risk signal (inverse relationship)
        if 'tail_risk_metrics' in metrics and 'crash_risk_index' in metrics['tail_risk_metrics']:
            crash_risk = metrics['tail_risk_metrics']['crash_risk_index']
            # Normalize and invert (higher crash risk = lower signal)
            tailrisk_signal = 1 - (crash_risk + 1) / 2 if -1 <= crash_risk <= 1 else 0.5
            signals.append(tailrisk_signal)
        
        # Calculate combined signal (average)
        if signals:
            combined_signal = sum(signals) / len(signals)
            
            # Map to original sigma scale (0-1)
            results['combined_sigma'] = combined_signal
            
            # Add interpretation
            if combined_signal > 0.8:
                interp = "STRONG BUY"
            elif combined_signal > 0.6:
                interp = "BUY"
            elif combined_signal > 0.4:
                interp = "HOLD"
            elif combined_signal > 0.2:
                interp = "SELL"
            else:
                interp = "STRONG SELL"
                
            results['combined_recommendation'] = interp
        else:
            results['combined_sigma'] = 0.5
            results['combined_recommendation'] = "HOLD"
            
        return results
    except Exception as e:
        print(f"[ERROR] Error in integrated analysis: {e}")
        return {
            'combined_sigma': 0.5,
            'combined_recommendation': "HOLD"
        }

def run_advanced_quantitative_analysis(data, symbol=None, analyses=None):
    """
    Run selected advanced quantitative analyses.
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str, optional
        Stock symbol to analyze (if None, analyze all columns)
    analyses: list, optional
        List of analyses to run (if None, run all)
        
    Returns:
    --------
    dict
        Analysis results
    """
    try:
        # Set default analyses
        if analyses is None:
            analyses = [
                'fractal',
                'tail_risk',
                'regime',
                'inefficiency',
                'microstructure'
            ]
            
        # Ensure data is properly formatted
        if isinstance(data, pd.DataFrame):
            # Make sure we have returns data
            if 'close' in data.columns and 'log_returns' not in data.columns:
                data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            elif '4. close' in data.columns and 'log_returns' not in data.columns:
                data['log_returns'] = np.log(data['4. close'] / data['4. close'].shift(1))
        else:
            print("[ERROR] Data must be a pandas DataFrame")
            return {'combined_sigma': 0.5}
            
        # Initialize results
        results = {'metrics': {}}
        
        # Run selected analyses
        for analysis in analyses:
            try:
                if analysis == 'fractal':
                    fractal_results = run_fractal_analysis(data)
                    results['metrics']['hurst_exponent'] = fractal_results.get('hurst_exponent', 0.5)
                    results['metrics']['fractal_dimension'] = fractal_results.get('fractal_dimension', 1.5)
                    
                elif analysis == 'tail_risk':
                    tail_risk_results = run_tail_risk_analysis(data)
                    results['metrics']['tail_risk_metrics'] = tail_risk_results.get('tail_risk_metrics', {})
                    
                elif analysis == 'regime':
                    regime_results = run_bayesian_regime_analysis(data)
                    results['metrics']['current_regime'] = regime_results.get('current_regime', {})
                    
                elif analysis == 'inefficiency':
                    inefficiency_results = run_market_inefficiency_analysis(data)
                    results['metrics']['inefficiency_score'] = inefficiency_results.get('inefficiency_score', 0.5)
                    results['metrics']['inefficiency_level'] = inefficiency_results.get('inefficiency_level', 'Medium')
                    
                elif analysis == 'microstructure':
                    microstructure_results = run_market_microstructure_analysis(data)
                    results['metrics']['microstructure_insights'] = {'volume_delta_signal': microstructure_results.get('volume_delta_signal', 'Neutral')}
            except Exception as e:
                print(f"[ERROR] Error running {analysis} analysis: {e}")
                
        # Integrate all analyses and calculate combined signal
        return integrate_with_existing_analysis(results)
    except Exception as e:
        print(f"[ERROR] Error in advanced quantitative analysis: {e}")
        return {'combined_sigma': 0.5}

# Use the local implementations if we still have issues with the imported functions
if USE_BACKUP_FUNCTIONS:
    try:
        # Test if the imported functions are actually callable
        if not callable(run_advanced_quantitative_analysis) or not callable(integrate_with_existing_analysis):
            print("[WARNING] Imported advanced functions are not callable, using local implementations")
            run_advanced_quantitative_analysis = local_run_advanced_quantitative_analysis
            integrate_with_existing_analysis = local_integrate_with_existing_analysis
    except NameError:
        print("[WARNING] Advanced functions not properly imported, using local implementations")
        run_advanced_quantitative_analysis = local_run_advanced_quantitative_analysis
        integrate_with_existing_analysis = local_integrate_with_existing_analysis
        
    # Force enable advanced functions with our local implementations
    USE_BACKUP_FUNCTIONS = True
    print("[INFO] Advanced quantitative functions have been ENABLED using local implementations")

# Create wrapper functions to match the expected function names
def calculate_sigma(data):
    """Wrapper function to calculate sigma from advanced_quant_functions_backup.py"""
    if USE_BACKUP_FUNCTIONS:
        try:
            # Check data format - the advanced_quant_functions expects different column format
            # If columns are ['open', 'high', 'low', '4. close', 'volume'], convert to match expected format
            df = data.copy()
            if '4. close' in df.columns and 'close' not in df.columns:
                df.rename(columns={'4. close': 'close'}, inplace=True)
                print("[INFO] Renamed '4. close' column to 'close' for advanced analysis")
                
            # Let's make sure we have log_returns calculated
            if 'log_returns' not in df.columns:
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                df['log_returns'] = df['log_returns'].fillna(0)
                print("[INFO] Added log_returns for advanced analysis")
            
            # Now run the advanced analysis
            results = run_advanced_quantitative_analysis(df)
            
            if isinstance(results, dict) and 'combined_sigma' in results:
                print(f"[INFO] Advanced analysis successful, combined_sigma: {results['combined_sigma']}")
                return results['combined_sigma']
            else:
                print(f"[WARNING] Failed to get combined_sigma from advanced quantitative analysis. Results: {results}")
                return 0.5  # Default neutral value
        except Exception as e:
            print(f"[ERROR] Error in advanced sigma calculation: {e}")
            return 0.5  # Default neutral value
    else:
        return 0.5  # Default when not using backup functions

def get_sigma_recommendation(sigma, analysis_details):
    """Wrapper function to get recommendation from advanced_quant_functions_backup.py"""
    if USE_BACKUP_FUNCTIONS:
        try:
            # Format the data for the advanced function
            # The expected format is more complex
            formatted_metrics = {
                'hurst_exponent': analysis_details.get('hurst_exponent', 0.5),
                'inefficiency_score': analysis_details.get('balance_factor', 0.5),
                'current_regime': {
                    'regime_type': analysis_details.get('market_regime', 'Unknown'),
                    'duration': analysis_details.get('regime_duration', 0)
                }
            }
            
            # Add tail risk metrics if available
            tail_risk_metrics = {
                'max_drawdown': analysis_details.get('max_drawdown', 0),
                'cvar_95': analysis_details.get('cvar_95', 0),
                'kelly_criterion': analysis_details.get('kelly', 0),
                'sharpe_ratio': analysis_details.get('sharpe', 0),
                'crash_risk_index': 0  # Default value
            }
            formatted_metrics['tail_risk_metrics'] = tail_risk_metrics
            
            # Add microstructure insights if available
            vol_signal = "Stable"
            if analysis_details.get('volatility_regime') == "Rising":
                vol_signal = "Bearish"
            elif analysis_details.get('volatility_regime') == "Falling":
                vol_signal = "Bullish"
            
            formatted_metrics['microstructure_insights'] = {
                'volume_delta_signal': vol_signal
            }
            
            # Create full data structure
            data = {'metrics': formatted_metrics}
            print(f"[DEBUG] Formatted metrics for recommendation: {formatted_metrics}")
            
            # Call the advanced function
            results = integrate_with_existing_analysis(data)
            
            if isinstance(results, dict) and 'combined_recommendation' in results:
                print(f"[INFO] Advanced recommendation successful: {results['combined_recommendation']}")
                return results['combined_recommendation']
            else:
                print(f"[WARNING] Failed to get recommendation from advanced functions. Results: {results}")
                return get_sigma_recommendation_implementation(sigma, analysis_details)
        except Exception as e:
            print(f"[ERROR] Error in advanced recommendation: {e}")
            return get_sigma_recommendation_implementation(sigma, analysis_details)
    else:
        return get_sigma_recommendation_implementation(sigma, analysis_details)

def get_sigma_recommendation_implementation(sigma, analysis_details):
    """
    Generate trading recommendations based on sigma value and analysis details.
    This is a fallback implementation when the function can't be imported from advanced_quant_functions_backup.py.

    Parameters:
    -----------
    sigma: float
        Sigma value (0-1 scale)
    analysis_details: dict
        Dictionary with analysis details

    Returns:
    --------
    str
        Trading recommendation with context
    """
    # Get additional context for our recommendation
    momentum_score = analysis_details.get("momentum_score", 0.5)
    reversion_score = analysis_details.get("reversion_score", 0.5)
    recent_monthly_return = analysis_details.get("recent_monthly_return", 0)
    balance_factor = analysis_details.get("balance_factor", 0.5)
    hurst_regime = analysis_details.get("hurst_regime", "Unknown")
    mean_reversion_speed = analysis_details.get("mean_reversion_speed", "Unknown")
    mean_reversion_beta = analysis_details.get("mean_reversion_beta", 0)
    volatility_regime = analysis_details.get("volatility_regime", "Unknown")
    vol_persistence = analysis_details.get("vol_persistence", 0.8)
    market_regime = analysis_details.get("market_regime", "Unknown")
    max_drawdown = analysis_details.get("max_drawdown", 0)
    kelly = analysis_details.get("kelly", 0)
    sharpe = analysis_details.get("sharpe", 0)

    # Base recommendation on sigma
    if sigma > 0.8:
        base_rec = "STRONG BUY"
    elif sigma > 0.6:
        base_rec = "BUY"
    elif sigma > 0.4:
        base_rec = "HOLD"
    elif sigma > 0.2:
        base_rec = "SELL"
    else:
        base_rec = "STRONG SELL"

    # Add nuanced context based on recent performance and advanced metrics, including log returns
    if recent_monthly_return > 0.25 and sigma > 0.6:
        if "Mean Reversion" in hurst_regime and mean_reversion_speed in ["Fast", "Very Fast"]:
            context = f"Strong momentum with +{recent_monthly_return:.1%} monthly gain, but high mean reversion risk (Hurst={analysis_details.get('hurst_exponent', 0):.2f}, Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Strong momentum with +{recent_monthly_return:.1%} monthly gain, elevated reversion risk but strong trend continues"
    elif recent_monthly_return > 0.15 and sigma > 0.6:
        if "Rising" in volatility_regime:
            context = f"Good momentum with +{recent_monthly_return:.1%} monthly gain but increasing volatility (persistence: {vol_persistence:.2f}), monitor closely"
        else:
            context = f"Good momentum with +{recent_monthly_return:.1%} monthly gain in stable volatility environment"
    elif recent_monthly_return > 0.10 and sigma > 0.6:
        if "Trending" in hurst_regime:
            context = f"Sustainable momentum with +{recent_monthly_return:.1%} monthly gain and strong trend characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f})"
        else:
            context = f"Moderate momentum with +{recent_monthly_return:.1%} monthly gain showing balanced metrics"
    elif recent_monthly_return < -0.20 and sigma > 0.6:
        if "Mean Reversion" in hurst_regime:
            context = f"Strong reversal potential after {recent_monthly_return:.1%} monthly decline, log return metrics show bottoming pattern (Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Potential trend change after {recent_monthly_return:.1%} decline but caution warranted"
    elif recent_monthly_return < -0.15 and sigma < 0.4:
        if "High" in market_regime:
            context = f"Continued weakness with {recent_monthly_return:.1%} monthly loss in high volatility regime"
        else:
            context = f"Negative trend with {recent_monthly_return:.1%} monthly loss and limited reversal signals"
    elif recent_monthly_return < -0.10 and sigma > 0.5:
        if mean_reversion_speed in ["Fast", "Very Fast"]:
            context = f"Potential rapid recovery after {recent_monthly_return:.1%} monthly decline (log reversion half-life: {analysis_details.get('mean_reversion_half_life', 0):.1f} days, Beta={mean_reversion_beta:.2f})"
        else:
            context = f"Potential stabilization after {recent_monthly_return:.1%} monthly decline, monitor for trend change"
    else:
        # Default context with advanced metrics, including log returns data
        if momentum_score > 0.7 and "Trending" in hurst_regime:
            context = f"Strong trend characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f}) with minimal reversal signals"
        elif momentum_score > 0.7 and reversion_score > 0.5:
            context = f"Strong but potentially overextended momentum in {volatility_regime} volatility regime (persistence: {vol_persistence:.2f})"
        elif momentum_score < 0.3 and "Mean Reversion" in hurst_regime:
            context = f"Strong mean-reverting characteristics (Hurst={analysis_details.get('hurst_exponent', 0):.2f}, Beta={mean_reversion_beta:.2f}) with weak momentum"
        elif momentum_score < 0.3 and reversion_score < 0.3:
            context = f"Weak directional signals in {market_regime} market regime"
        elif "High" in market_regime and "Rising" in volatility_regime:
            context = f"Mixed signals in high volatility environment - position sizing caution advised"
        elif abs(momentum_score - (1 - reversion_score)) < 0.1:
            context = f"Balanced indicators with no clear edge in {volatility_regime} volatility"
        else:
            context = f"Mixed signals requiring monitoring with log-based half-life of {analysis_details.get('mean_reversion_half_life', 0):.1f} days"

    # Add risk metrics
    if max_drawdown < -0.4:
        context += f" | High historical drawdown risk ({max_drawdown:.1%})"

    if kelly < -0.2:
        context += f" | Negative expectancy (Kelly={kelly:.2f})"
    elif kelly > 0.3:
        context += f" | Strong positive expectancy (Kelly={kelly:.2f})"

    # Add Sharpe ratio if available
    if sharpe > 1.5:
        context += f" | Excellent risk-adjusted returns (Sharpe={sharpe:.2f})"
    elif sharpe < 0:
        context += f" | Poor risk-adjusted returns (Sharpe={sharpe:.2f})"

    # Add advanced metrics if available
    if 'advanced_metrics' in analysis_details:
        advanced = analysis_details['advanced_metrics']

        # Add regime information if available
        if 'current_regime' in advanced:
            regime = advanced['current_regime']
            if 'regime_type' in regime:
                context += f" | Market regime: {regime['regime_type']}"

        # Add inefficiency information if available
        if 'inefficiency_score' in advanced:
            score = advanced['inefficiency_score']
            if score > 0.6:
                context += f" | High market inefficiency detected ({score:.2f})"

        # Add tail risk information if available
        if 'tail_risk_metrics' in advanced and 'cvar_95' in advanced['tail_risk_metrics']:
            cvar = advanced['tail_risk_metrics']['cvar_95']
            context += f" | CVaR(95%): {cvar:.2%}"

    # Combine base recommendation with context
    recommendation = f"{base_rec} - {context}"

    return recommendation

# Enhanced technical indicators with log returns mean reversion components
def calculate_technical_indicators(data):
    try:
        print(f"[DEBUG] Calculating enhanced technical indicators with log returns on data with shape: {data.shape}")
        
        # If using backup functions, try to use the advanced technical analysis
        if USE_BACKUP_FUNCTIONS:
            try:
                # Look for technical analysis functions in the advanced backup
                import inspect
                from advanced_quant_functions_backup import (
                    calculate_technical_indicators as advanced_calculate_technical_indicators,
                )
                
                # Prepare data for advanced analysis - column naming convention fix
                df_advanced = data.copy()
                if '4. close' in df_advanced.columns and 'close' not in df_advanced.columns:
                    df_advanced.rename(columns={'4. close': 'close'}, inplace=True)
                    print("[INFO] Renamed '4. close' column to 'close' for advanced technical analysis")
                
                # Make sure we have log_returns
                if 'log_returns' not in df_advanced.columns:
                    df_advanced['log_returns'] = np.log(df_advanced['close'] / df_advanced['close'].shift(1))
                    df_advanced['log_returns'] = df_advanced['log_returns'].fillna(0)
                    print("[INFO] Added log_returns for advanced technical analysis")
                
                # If the function exists and takes similar parameters, use it
                if callable(advanced_calculate_technical_indicators):
                    print(f"[INFO] Using advanced technical indicators from backup functions with data shape {df_advanced.shape}")
                    result = advanced_calculate_technical_indicators(df_advanced)
                    if result is not None:
                        print(f"[INFO] Advanced technical indicators calculation successful. Returned shape: {result.shape}")
                        
                        # Make sure the result has '4. close' column consistent with main code
                        if 'close' in result.columns and '4. close' not in result.columns:
                            result.rename(columns={'close': '4. close'}, inplace=True)
                            print("[INFO] Renamed 'close' back to '4. close' for consistency")
                            
                        return result
                    else:
                        print("[WARNING] Advanced technical indicators returned None, falling back to basic implementation")
                else:
                    print("[WARNING] Advanced technical indicators function not callable, falling back to basic implementation")
            except (ImportError, AttributeError, Exception) as e:
                print(f"[WARNING] Failed to use advanced technical indicators: {e}. Falling back to basic implementation.")
        
        # Fall back to our implementation if advanced functions aren't available or fail
        df = data.copy()

        # Check if data is sufficient
        if len(df) < 50:
            print("[WARNING] Not enough data for technical indicators calculation")
            return None

        # Calculate regular returns
        df['returns'] = df['4. close'].pct_change()
        df['returns'] = df['returns'].fillna(0)

        # NEW: Calculate log returns for improved statistical properties
        df['log_returns'] = np.log(df['4. close'] / df['4. close'].shift(1))
        df['log_returns'] = df['log_returns'].fillna(0)

        # Calculate volatility (20-day rolling standard deviation)
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility'] = df['volatility'].fillna(0)

        # NEW: Log return volatility for more accurate volatility measurement
        df['log_volatility'] = df['log_returns'].rolling(window=20).std()
        df['log_volatility'] = df['log_volatility'].fillna(0)

        # Calculate Simple Moving Averages
        df['SMA20'] = df['4. close'].rolling(window=20).mean()
        df['SMA50'] = df['4. close'].rolling(window=50).mean()
        df['SMA100'] = df['4. close'].rolling(window=100).mean()
        df['SMA200'] = df['4. close'].rolling(window=200).mean()

        # Fill NaN values in SMAs with forward fill then backward fill
        for col in ['SMA20', 'SMA50', 'SMA100', 'SMA200']:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Improved RSI calculation with better handling for division by zero
        print("[INFO] Calculating RSI with improved method")
        
        # First prepare delta (price change)
        delta = df['4. close'].diff()
        delta = delta.fillna(0)
        
        # Split into gain and loss components
        gain = pd.Series(0, index=delta.index)
        loss = pd.Series(0, index=delta.index)
        
        gain[delta > 0] = delta[delta > 0]
        loss[delta < 0] = -delta[delta < 0]
        
        # Calculate average gain and loss over the window
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Create RS with proper handling for different cases
        rs = pd.Series(index=avg_gain.index)
        
        # When average loss is zero but gain exists, RS should be very high (RSI near 100)
        rs[(avg_gain > 0) & (avg_loss == 0)] = 100.0
        
        # When both are zero (no movement), RSI should be neutral
        rs[(avg_gain == 0) & (avg_loss == 0)] = 1.0
        
        # For normal cases, calculate RS
        valid_mask = (avg_loss > 0)
        rs[valid_mask] = avg_gain[valid_mask] / avg_loss[valid_mask]
        
        # Handle any potential NaN values
        rs = rs.fillna(1.0)  # Neutral when uncertain
        
        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Ensure all values are filled
        df['RSI'] = df['RSI'].fillna(50)  # Default to neutral RSI (50)
        
        # Print some statistics to validate
        print(f"[INFO] RSI calculation complete. Range: {df['RSI'].min():.2f} to {df['RSI'].max():.2f}")

        # Calculate Bollinger Bands
        df['BB_middle'] = df['SMA20']
        df['BB_std'] = df['4. close'].rolling(window=20).std()
        df['BB_std'] = df['BB_std'].fillna(0)
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)

        # Calculate MACD
        df['EMA12'] = df['4. close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['4. close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Calculate trading volume changes
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_change'] = df['volume_change'].fillna(0)

        # MEAN REVERSION COMPONENTS

        # 1. Distance from SMA200 as mean reversion indicator
        df['dist_from_SMA200'] = (df['4. close'] / df['SMA200']) - 1

        # 2. Bollinger Band %B (0-1 scale where >1 is overbought, <0 is oversold)
        bb_range = df['BB_upper'] - df['BB_lower']
        df['BB_pctB'] = np.where(
            bb_range > 0,
            (df['4. close'] - df['BB_lower']) / bb_range,
            0.5
        )

        # 3. Price Rate of Change (historical returns over different periods)
        df['ROC_5'] = df['4. close'].pct_change(5)
        df['ROC_10'] = df['4. close'].pct_change(10)
        df['ROC_20'] = df['4. close'].pct_change(20)
        df['ROC_60'] = df['4. close'].pct_change(60)

        # 4. Overbought/Oversold indicator based on historical returns
        # Standardize recent returns relative to their own history
        returns_z_score = lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
        df['returns_zscore_5'] = returns_z_score(df['ROC_5'])
        df['returns_zscore_20'] = returns_z_score(df['ROC_20'])

        # 5. Price acceleration (change in ROC) - detects momentum exhaustion
        df['ROC_accel'] = df['ROC_5'] - df['ROC_5'].shift(5)

        # 6. Historical volatility ratio (recent vs long-term)
        df['vol_ratio'] = df['volatility'] / df['volatility'].rolling(60).mean()

        # 7. Mean reversion potential based on distance from long-term trend
        # Using Z-score of price deviation from 200-day SMA
        mean_dist = df['dist_from_SMA200'].rolling(100).mean()
        std_dist = df['dist_from_SMA200'].rolling(100).std()
        df['mean_reversion_z'] = np.where(
            std_dist > 0,
            (df['dist_from_SMA200'] - mean_dist) / std_dist,
            0
        )

        # 8. RSI divergence (price making new highs but RSI isn't)
        df['price_high'] = df['4. close'].rolling(10).max() == df['4. close']
        df['rsi_high'] = df['RSI'].rolling(10).max() == df['RSI']
        # Potential negative divergence: price high but RSI not high
        df['rsi_divergence'] = np.where(df['price_high'] & ~df['rsi_high'], -1, 0)

        # 9. Volume-price relationship (high returns with low volume can signal exhaustion)
        if 'volume' in df.columns:
            df['vol_price_ratio'] = np.where(
                df['returns'] != 0,
                df['volume'] / (abs(df['returns']) * df['4. close']),
                0
            )
            df['vol_price_ratio_z'] = (df['vol_price_ratio'] - df['vol_price_ratio'].rolling(20).mean()) / df[
                'vol_price_ratio'].rolling(20).std()

        # 10. Stochastic Oscillator
        if 'high' in df.columns and 'low' in df.columns:
            window = 14
            df['14-high'] = df['high'].rolling(window).max()
            df['14-low'] = df['low'].rolling(window).min()
            df['%K'] = (df['4. close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
            df['%D'] = df['%K'].rolling(3).mean()

        # 11. Advanced RSI Analysis
        # RSI slope (rate of change)
        df['RSI_slope'] = df['RSI'] - df['RSI'].shift(3)

        # RSI moving average crossovers
        df['RSI_MA5'] = df['RSI'].rolling(5).mean()
        df['RSI_MA14'] = df['RSI'].rolling(14).mean()

        # 12. Double Bollinger Bands (outer bands at 3 std dev)
        df['BB_upper_3'] = df['BB_middle'] + (df['BB_std'] * 3)
        df['BB_lower_3'] = df['BB_middle'] - (df['BB_std'] * 3)

        # 13. Volume Weighted MACD
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=14).mean()
            volume_ratio = np.where(df['volume_ma'] > 0, df['volume'] / df['volume_ma'], 1)
            df['vol_weighted_macd'] = df['MACD'] * volume_ratio

        # 14. Chaikin Money Flow (CMF)
        if 'high' in df.columns and 'low' in df.columns and 'volume' in df.columns:
            money_flow_multiplier = ((df['4. close'] - df['low']) - (df['high'] - df['4. close'])) / (
                    df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['CMF'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()

        # 15. Williams %R
        if '14-high' in df.columns and '14-low' in df.columns:
            df['Williams_%R'] = -100 * (df['14-high'] - df['4. close']) / (df['14-high'] - df['14-low'])

        # 16. Advanced trend analysis
        df['trend_strength'] = np.abs(df['dist_from_SMA200'])
        df['price_vs_all_SMAs'] = np.where(
            (df['4. close'] > df['SMA20']) &
            (df['4. close'] > df['SMA50']) &
            (df['4. close'] > df['SMA100']) &
            (df['4. close'] > df['SMA200']),
            1, 0
        )

        # 17. SMA alignment (bullish/bearish alignment)
        df['sma_alignment'] = np.where(
            (df['SMA20'] > df['SMA50']) &
            (df['SMA50'] > df['SMA100']) &
            (df['SMA100'] > df['SMA200']),
            1,  # Bullish alignment
            np.where(
                (df['SMA20'] < df['SMA50']) &
                (df['SMA50'] < df['SMA100']) &
                (df['SMA100'] < df['SMA200']),
                -1,  # Bearish alignment
                0  # Mixed alignment
            )
        )

        # ======== NEW LOG RETURNS BASED MEAN REVERSION METRICS ========

        # 1. Log returns Z-score (more statistically valid than regular returns)
        log_returns_mean = df['log_returns'].rolling(100).mean()
        log_returns_std = df['log_returns'].rolling(100).std()
        df['log_returns_zscore'] = np.where(
            log_returns_std > 0,
            (df['log_returns'] - log_returns_mean) / log_returns_std,
            0
        )

        # 2. Log return mean reversion potential
        # Higher absolute values suggest stronger mean reversion potential
        # Sign indicates expected direction (negative means price likely to increase)
        df['log_mr_potential'] = -1 * df['log_returns_zscore']

        # 3. Log return autocorrelation - measures mean reversion strength
        # Uses 5-day lag as common mean-reversion period
        df['log_autocorr_5'] = df['log_returns'].rolling(30).apply(
            lambda x: x.autocorr(lag=5) if len(x.dropna()) > 5 else 0,
            raw=False
        )

        # 4. Log volatility ratio (indicates regime changes)
        df['log_vol_ratio'] = df['log_volatility'] / df['log_volatility'].rolling(60).mean()

        # 5. Log return momentum vs mean reversion balance
        # This combines both momentum and mean reversion signals
        # Positive values suggest momentum dominates, negative suggest mean reversion dominates
        df['log_mom_vs_mr'] = df['log_returns'].rolling(10).mean() / df['log_volatility'] + df['log_autocorr_5']

        # 6. Log-based adaptive Bollinger Bands
        # More accurate for capturing true statistical extremes
        log_price = np.log(df['4. close'])
        log_ma20 = log_price.rolling(20).mean()
        log_std20 = log_price.rolling(20).std()
        df['log_bb_upper'] = np.exp(log_ma20 + 2 * log_std20)
        df['log_bb_lower'] = np.exp(log_ma20 - 2 * log_std20)
        df['log_bb_pctB'] = np.where(
            (df['log_bb_upper'] - df['log_bb_lower']) > 0,
            (df['4. close'] - df['log_bb_lower']) / (df['log_bb_upper'] - df['log_bb_lower']),
            0.5
        )

        # 7. Log return expected mean reversion magnitude
        # Estimates expected price change if fully reverted to mean
        df['log_expected_reversion'] = -1 * df['log_returns_zscore'] * df['log_volatility'] * np.sqrt(252)
        df['log_expected_reversion_pct'] = (np.exp(df['log_expected_reversion']) - 1) * 100

        # Fill NaN values in new indicators
        for col in df.columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

        print(f"[DEBUG] Enhanced technical indicators with log returns calculated successfully. New shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Error calculating enhanced technical indicators: {e}")
        traceback.print_exc()
        return None


# Improved Hurst Exponent calculation using log returns
def calculate_hurst_exponent(df, max_lag=120, use_log_returns=True):
    """Calculate Hurst exponent to determine mean reversion vs trending behavior
       Now uses log returns for more accurate measurement"""
    try:
        # Use log returns for better statistical properties
        if use_log_returns and 'log_returns' in df.columns:
            returns = df['log_returns'].dropna().values
            print("[INFO] Using log returns for Hurst calculation")
        else:
            returns = df['returns'].dropna().values
            print("[INFO] Using regular returns for Hurst calculation")

        if len(returns) < max_lag:
            print(f"[WARNING] Not enough returns for Hurst calculation: {len(returns)} < {max_lag}")
            max_lag = max(30, len(returns) // 4)  # Adjust max_lag if not enough data

        lags = range(2, max_lag)
        tau = []
        var = []

        for lag in lags:
            # Price (or return) difference
            pp = np.subtract(returns[lag:], returns[:-lag])
            # Variance
            variance = np.std(pp)
            var.append(variance)
            tau.append(lag)

        # Linear fit in log-log space to calculate Hurst exponent
        m = np.polyfit(np.log(tau), np.log(var), 1)
        hurst = m[0] / 2.0

        # Categorize by Hurst value
        if hurst < 0.4:
            regime = "Strong Mean Reversion"
        elif hurst < 0.45:
            regime = "Mean Reversion"
        elif hurst < 0.55:
            regime = "Random Walk"
        elif hurst < 0.65:
            regime = "Trending"
        else:
            regime = "Strong Trending"

        return {"hurst": hurst, "regime": regime}
    except Exception as e:
        print(f"[ERROR] Error calculating Hurst exponent: {e}")
        return {"hurst": 0.5, "regime": "Unknown"}


# Improved Mean Reversion Half-Life using log returns
def calculate_mean_reversion_half_life(data):
    """Estimate half-life of mean reversion using log returns with Ornstein-Uhlenbeck process"""
    try:
        # Check if we have log returns available, otherwise calculate them
        if 'log_returns' not in data.columns:
            log_returns = np.log(data['4. close'] / data['4. close'].shift(1)).dropna()
            print("[INFO] Calculating log returns for mean reversion half-life")
        else:
            log_returns = data['log_returns'].dropna()
            print("[INFO] Using existing log returns for mean reversion half-life")

        # Calculate deviation of log returns from their moving average
        ma = log_returns.rolling(window=50).mean()
        spread = log_returns - ma

        # Remove NaN values
        spread = spread.dropna()

        if len(spread) < 50:
            print("[WARNING] Not enough data for mean reversion half-life calculation")
            return {"half_life": 0, "mean_reversion_speed": "Unknown"}

        # Calculate autoregression coefficient
        # S_t+1 - S_t = a * S_t + e_t
        spread_lag = spread.shift(1).dropna()
        spread_current = spread.iloc[1:]

        # Match lengths
        spread_lag = spread_lag.iloc[:len(spread_current)]

        # Use regression to find the coefficient
        model = LinearRegression()
        model.fit(spread_lag.values.reshape(-1, 1), spread_current.values)
        beta = model.coef_[0]

        # Calculate half-life
        # The closer beta is to -1, the faster the mean reversion
        # If beta > 0, it's trending, not mean-reverting
        if -1 < beta < 0:
            half_life = -np.log(2) / np.log(1 + beta)
        else:
            # If beta is positive (momentum) or <= -1 (oscillatory), default to 0
            half_life = 0

        # Categorize strength
        if 0 < half_life <= 5:
            strength = "Very Fast"
        elif half_life <= 20:
            strength = "Fast"
        elif half_life <= 60:
            strength = "Medium"
        elif half_life <= 120:
            strength = "Slow"
        else:
            strength = "Very Slow or None"

        # Return beta for additional context
        return {
            "half_life": half_life,
            "mean_reversion_speed": strength,
            "beta": beta  # Added beta coefficient
        }
    except Exception as e:
        print(f"[ERROR] Error calculating mean reversion half-life: {e}")
        return {"half_life": 0, "mean_reversion_speed": "Unknown", "beta": 0}


# Volatility Regime Analysis with log-based improvements
def analyze_volatility_regimes(data, lookback=252):
    """Implements advanced volatility analysis with log returns for better accuracy"""
    try:
        # Use log returns if available for improved statistical properties
        if 'log_returns' in data.columns:
            returns = data['log_returns'].iloc[-lookback:]
            print("[INFO] Using log returns for volatility regime analysis")
        else:
            returns = data['returns'].iloc[-lookback:]
            print("[INFO] Using regular returns for volatility regime analysis")

        # 1. Volatility term structure
        short_vol = returns.iloc[-20:].std() * np.sqrt(252)
        medium_vol = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else short_vol
        long_vol = returns.iloc[-120:].std() * np.sqrt(252) if len(returns) >= 120 else medium_vol

        # Relative readings
        vol_term_structure = short_vol / long_vol
        vol_acceleration = (short_vol / medium_vol) / (medium_vol / long_vol)

        # 2. Parkinson volatility estimator (uses high-low range)
        if 'high' in data.columns and 'low' in data.columns:
            # Improved Parkinson estimator using log prices
            high_low_ratio = np.log(data['high'] / data['low'])
            parker_vol = np.sqrt(1 / (4 * np.log(2)) * high_low_ratio.iloc[-20:].pow(2).mean() * 252)
        else:
            parker_vol = None

        # 3. GARCH-like volatility persistence estimation
        try:
            # Simple AR(1) model to estimate volatility persistence
            squared_returns = returns.pow(2).dropna()
            if len(squared_returns) > 22:  # At least a month of data
                sq_ret_lag = squared_returns.shift(1).dropna()
                sq_ret = squared_returns.iloc[1:]

                # Match lengths
                sq_ret_lag = sq_ret_lag.iloc[:len(sq_ret)]

                if len(sq_ret) > 10:  # Need sufficient data
                    # Fit AR(1) model to squared returns
                    vol_model = LinearRegression()
                    vol_model.fit(sq_ret_lag.values.reshape(-1, 1), sq_ret.values)
                    vol_persistence = vol_model.coef_[0]  # How much volatility persists
                else:
                    vol_persistence = 0.8  # Default value
            else:
                vol_persistence = 0.8  # Default value
        except:
            vol_persistence = 0.8  # Default if calculation fails

        # Volatility regime detection
        if vol_term_structure > 1.3:
            vol_regime = "Rising"
        elif vol_term_structure < 0.7:
            vol_regime = "Falling"
        else:
            vol_regime = "Stable"

        return {
            "vol_term_structure": vol_term_structure,
            "vol_acceleration": vol_acceleration,
            "parkinson_vol": parker_vol,
            "vol_regime": vol_regime,
            "vol_persistence": vol_persistence,  # New metric
            "short_vol": short_vol,
            "medium_vol": medium_vol,
            "long_vol": long_vol
        }
    except Exception as e:
        print(f"[ERROR] Error analyzing volatility regimes: {e}")
        # Fallback in case of calculation issues
        return {
            "vol_regime": "Unknown",
            "vol_term_structure": 1.0,
            "vol_persistence": 0.8
        }


# Market Regime Detection with log returns
def detect_market_regime(data, n_regimes=3):
    """Detect market regimes using Hidden Markov Model on log returns for improved results"""
    try:
        # If using backup functions, try to use the advanced regime analysis
        if USE_BACKUP_FUNCTIONS:
            try:
                # Local implementation of run_bayesian_regime_analysis
                def local_run_bayesian_regime_analysis(df):
                    """Simple local implementation of Bayesian regime analysis"""
                    print("[INFO] Using local implementation of Bayesian regime analysis")
                    
                    # Use log returns for regime classification
                    if 'log_returns' in df.columns:
                        returns = df['log_returns'].dropna()
                    elif 'returns' in df.columns:
                        returns = df['returns'].dropna()
                    elif 'close' in df.columns:
                        # Calculate returns if needed
                        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
                    elif '4. close' in df.columns:
                        # Calculate returns if needed
                        returns = np.log(df['4. close'] / df['4. close'].shift(1)).dropna()
                    else:
                        # Default to empty returns
                        returns = pd.Series()
                        
                    # Check if we have enough data
                    if len(returns) < 30:
                        print("[WARNING] Not enough return data for regime analysis")
                        return {
                            'current_regime': {
                                'regime_type': 'Insufficient Data',
                                'duration': 0,
                                'volatility': 0
                            }
                        }
                    
                    # Calculate basic volatility metrics
                    recent_vol = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized 20-day volatility
                    historical_vol = returns.iloc[:-20].std() * np.sqrt(252)  # Historical volatility
                    
                    # Calculate recent returns trend
                    recent_returns = returns.iloc[-60:].mean() * 252  # Annualized 60-day returns
                    
                    # Simple regime classification based on volatility and returns
                    if recent_vol < 0.8 * historical_vol:
                        if recent_returns > 0:
                            regime_type = "Low Volatility Bull"
                        else:
                            regime_type = "Low Volatility Bear"
                    elif recent_vol > 1.2 * historical_vol:
                        if recent_returns > 0:
                            regime_type = "High Volatility Bull"
                        else:
                            regime_type = "High Volatility Bear"
                    else:
                        regime_type = "Neutral/Sideways"
                    
                    # Calculate regime duration - days with similar volatility profile
                    # Simplified: just count days where volatility remained in same bracket
                    vol_ratio = returns.rolling(20).std() / returns.rolling(60).std()
                    
                    if recent_vol < 0.8 * historical_vol:
                        duration = sum(vol_ratio.iloc[-30:] < 0.8)
                    elif recent_vol > 1.2 * historical_vol:
                        duration = sum(vol_ratio.iloc[-30:] > 1.2)
                    else:
                        duration = sum((vol_ratio.iloc[-30:] >= 0.8) & (vol_ratio.iloc[-30:] <= 1.2))
                    
                    return {
                        'current_regime': {
                            'regime_type': regime_type,
                            'duration': int(duration),
                            'volatility': recent_vol
                        }
                    }
                
                # Prepare data for advanced analysis - column naming convention fix
                df = data.copy()
                if '4. close' in df.columns and 'close' not in df.columns:
                    df.rename(columns={'4. close': 'close'}, inplace=True)
                    print("[INFO] Renamed '4. close' column to 'close' for advanced regime analysis")
                
                # Make sure we have log_returns
                if 'log_returns' not in df.columns:
                    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                    df['log_returns'] = df['log_returns'].fillna(0)
                    print("[INFO] Added log_returns for advanced regime analysis")
                
                # First try to use the imported function
                try:
                    from advanced_quant_functions_backup import run_bayesian_regime_analysis
                    # Call the advanced function
                    print("[INFO] Calling run_bayesian_regime_analysis with data shape", df.shape)
                    regime_results = run_bayesian_regime_analysis(df)
                except (ImportError, NameError, Exception) as e:
                    print(f"[WARNING] Could not use imported run_bayesian_regime_analysis: {e}")
                    # Use our local implementation
                    regime_results = local_run_bayesian_regime_analysis(df)
                
                if isinstance(regime_results, dict) and 'current_regime' in regime_results:
                    print("[INFO] Using advanced regime analysis")
                    # Extract what we need from the advanced results
                    current_regime = regime_results['current_regime']['regime_type'] if 'regime_type' in regime_results['current_regime'] else "Unknown"
                    regime_duration = regime_results['current_regime'].get('duration', 0)
                    regime_volatility = regime_results['current_regime'].get('volatility', 0)
                    
                    print(f"[INFO] Advanced regime analysis results: {current_regime}, duration: {regime_duration}")
                    
                    return {
                        "current_regime": current_regime,
                        "regime_duration": regime_duration,
                        "regime_volatility": regime_volatility
                    }
                else:
                    print(f"[WARNING] Unexpected results from advanced regime analysis: {regime_results}")
            except Exception as e:
                print(f"[WARNING] Failed to use advanced regime analysis: {e}. Falling back to basic implementation.")
                # Fall back to our implementation below
        
        # Extract features for regime detection using a try-except for each step
        try:
            # Prepare the returns data
            if 'log_returns' in data.columns:
                returns = data['log_returns'].fillna(0).values.reshape(-1, 1)
                print("[INFO] Using log returns for market regime detection")
            elif 'returns' in data.columns:
                returns = data['returns'].fillna(0).values.reshape(-1, 1)
                print("[INFO] Using regular returns for market regime detection")
            else:
                # Calculate returns from close prices
                if '4. close' in data.columns:
                    prices = data['4. close'].values
                    returns = np.diff(np.log(prices)).reshape(-1, 1)
                    returns = np.concatenate([np.zeros((1, 1)), returns])  # Add a zero for the first row
                    print("[INFO] Calculated log returns from close prices for market regime detection")
                else:
                    print("[ERROR] No return or price data found for market regime detection")
                    raise ValueError("No return or price data available")
            
            # Fit HMM with fewer iterations for performance
            model = hmm.GaussianHMM(n_components=n_regimes, n_iter=100, random_state=42)
            model.fit(returns)

            # Predict regime
            hidden_states = model.predict(returns)

            # Map states to meaningful regimes
            states_volatility = {}
            for state in range(n_regimes):
                state_returns = returns[hidden_states == state]
                states_volatility[state] = np.std(state_returns)

            # Sort states by volatility
            sorted_states = sorted(states_volatility.items(), key=lambda x: x[1])
            regime_map = {}
            regime_map[sorted_states[0][0]] = "Low Volatility"
            regime_map[sorted_states[-1][0]] = "High Volatility"

            if n_regimes > 2:
                for i in range(1, n_regimes - 1):
                    regime_map[sorted_states[i][0]] = f"Medium Volatility {i}"

            # Get current regime
            current_regime = regime_map[hidden_states[-1]]

            # Calculate regime stability (how long we've been in this regime)
            regime_duration = 1
            for i in range(2, min(100, len(hidden_states))):
                if hidden_states[-i] == hidden_states[-1]:
                    regime_duration += 1
                else:
                    break

            return {
                "current_regime": current_regime,
                "regime_duration": regime_duration,
                "regime_volatility": states_volatility[hidden_states[-1]]
            }
        except Exception as e:
            print(f"[ERROR] Error in basic market regime detection: {e}")
            
            # Calculate a very simple regime based on recent volatility
            try:
                if 'log_returns' in data.columns:
                    returns = data['log_returns']
                elif 'returns' in data.columns:
                    returns = data['returns']
                else:
                    returns = pd.Series(np.diff(np.log(data['4. close'].values)))
                    
                # Calculate recent volatility and historical volatility
                recent_vol = returns.iloc[-20:].std() * np.sqrt(252) if len(returns) >= 20 else 0
                hist_vol = returns.iloc[-100:-20].std() * np.sqrt(252) if len(returns) >= 100 else recent_vol
                
                if recent_vol > 1.2 * hist_vol:
                    regime = "High Volatility"
                elif recent_vol < 0.8 * hist_vol:
                    regime = "Low Volatility"
                else:
                    regime = "Medium Volatility"
                    
                return {
                    "current_regime": regime,
                    "regime_duration": 20,  # Approximation
                    "regime_volatility": recent_vol
                }
            except Exception as nested_e:
                print(f"[ERROR] Error in fallback regime detection: {nested_e}")
                return {
                    "current_regime": "Unknown",
                    "regime_duration": 0,
                    "regime_volatility": 0
                }
    except Exception as e:
        print(f"[ERROR] Error detecting market regime: {e}")
        return {
            "current_regime": "Unknown",
            "regime_duration": 0,
            "regime_volatility": 0
        }


# Risk-Adjusted Metrics with log return improvements
def calculate_risk_adjusted_metrics(df, sigma):
    """Calculate risk-adjusted metrics using log returns for more accuracy"""
    try:
        # If using backup functions, try to use the advanced risk metrics
        if USE_BACKUP_FUNCTIONS:
            try:
                from advanced_quant_functions_backup import calculate_tail_risk_metrics, run_tail_risk_analysis
                
                # Prepare data for advanced analysis - column naming convention fix
                modified_df = df.copy()
                if '4. close' in modified_df.columns and 'close' not in modified_df.columns:
                    modified_df.rename(columns={'4. close': 'close'}, inplace=True)
                    print("[INFO] Renamed '4. close' column to 'close' for advanced risk analysis")
                
                # Make sure we have log_returns
                if 'log_returns' not in modified_df.columns:
                    modified_df['log_returns'] = np.log(modified_df['close'] / modified_df['close'].shift(1))
                    modified_df['log_returns'] = modified_df['log_returns'].fillna(0)
                    print("[INFO] Added log_returns for advanced risk analysis")
                
                # Try first with the comprehensive analysis function
                try:
                    print("[INFO] Calling run_tail_risk_analysis with data shape", modified_df.shape)
                    risk_results = run_tail_risk_analysis(modified_df)
                    if isinstance(risk_results, dict) and 'tail_risk_metrics' in risk_results:
                        print("[INFO] Using advanced tail risk analysis from backup functions")
                        metrics = risk_results['tail_risk_metrics']
                        
                        # Print metrics to debug
                        print(f"[DEBUG] Advanced risk metrics: {metrics}")
                        
                        # Extract the metrics we need
                        max_drawdown = metrics.get('max_drawdown', 0)
                        cvar_95 = metrics.get('cvar_95', 0)
                        kelly = metrics.get('kelly_criterion', 0)
                        sharpe = metrics.get('sharpe_ratio', 0)
                        
                        # Calculate risk-adjusted sigma
                        risk_adjusted_sigma = sigma
                        
                        # Apply risk adjustments similar to our original implementation
                        if max_drawdown < -0.5:  # >50% drawdown
                            risk_adjusted_sigma *= 0.5
                        elif max_drawdown < -0.3:  # >30% drawdown
                            risk_adjusted_sigma *= 0.8
                        
                        if kelly < 0:
                            risk_adjusted_sigma *= (1 + kelly)
                            
                        # Ensure sigma is within bounds
                        risk_adjusted_sigma = max(0.01, min(1.0, risk_adjusted_sigma))
                        
                        return {
                            "max_drawdown": max_drawdown,
                            "cvar_95": cvar_95,
                            "kelly": kelly,
                            "sharpe": sharpe,
                            "risk_adjusted_sigma": risk_adjusted_sigma
                        }
                    else:
                        print(f"[WARNING] Unexpected results from run_tail_risk_analysis: {risk_results}")
                except Exception as e:
                    print(f"[WARNING] Failed to use run_tail_risk_analysis: {e}. Trying direct calculation.")
                
                # Try with direct metrics calculation
                try:
                    # Get returns for calculation
                    if 'log_returns' in modified_df.columns:
                        returns = modified_df['log_returns'].dropna()
                    else:
                        returns = modified_df['returns'].dropna()
                    
                    print(f"[INFO] Calling calculate_tail_risk_metrics with returns shape: {returns.shape}")
                    metrics = calculate_tail_risk_metrics(returns)
                    
                    if isinstance(metrics, dict):
                        print("[INFO] Using calculate_tail_risk_metrics from backup functions")
                        print(f"[DEBUG] Direct risk metrics calculation results: {metrics}")
                        
                        # Extract metrics
                        max_drawdown = metrics.get('max_drawdown', 0)
                        cvar_95 = metrics.get('cvar_95', 0)
                        kelly = metrics.get('kelly_criterion', 0)
                        sharpe = metrics.get('sharpe_ratio', 0)
                        
                        # Calculate risk-adjusted sigma
                        risk_adjusted_sigma = sigma
                        
                        # Apply risk adjustments
                        if max_drawdown < -0.5:
                            risk_adjusted_sigma *= 0.5
                        elif max_drawdown < -0.3:
                            risk_adjusted_sigma *= 0.8
                        
                        if kelly < 0:
                            risk_adjusted_sigma *= (1 + kelly)
                            
                        # Ensure sigma is within bounds
                        risk_adjusted_sigma = max(0.01, min(1.0, risk_adjusted_sigma))
                        
                        return {
                            "max_drawdown": max_drawdown,
                            "cvar_95": cvar_95,
                            "kelly": kelly,
                            "sharpe": sharpe,
                            "risk_adjusted_sigma": risk_adjusted_sigma
                        }
                    else:
                        print(f"[WARNING] Unexpected results from calculate_tail_risk_metrics: {metrics}")
                except Exception as e:
                    print(f"[WARNING] Failed to use calculate_tail_risk_metrics: {e}. Falling back to basic implementation.")
            except (ImportError, Exception) as e:
                print(f"[WARNING] Failed to use advanced risk metrics: {e}. Falling back to basic implementation.")
        
        # Fall back to our implementation if advanced functions aren't available or fail
        # Use log returns if available for better statistical properties
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
            print("[INFO] Using log returns for risk-adjusted metrics")
        else:
            returns = df['returns'].dropna()
            print("[INFO] Using regular returns for risk-adjusted metrics")

        # Calculate Maximum Historical Drawdown
        # For log returns, we need to convert back to cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) if 'log_returns' in df.columns else (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()

        # Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        alpha = 0.05  # 95% confidence level
        var_95 = np.percentile(returns, alpha * 100)
        cvar_95 = returns[returns <= var_95].mean()

        # Calculate Kelly Criterion
        # For log returns, we adjust the win/loss calculation
        if 'log_returns' in df.columns:
            # Convert to arithmetic returns for Kelly calculation
            arith_returns = np.exp(returns) - 1
            win_rate = len(arith_returns[arith_returns > 0]) / len(arith_returns)
            avg_win = arith_returns[arith_returns > 0].mean() if len(arith_returns[arith_returns > 0]) > 0 else 0
            avg_loss = abs(arith_returns[arith_returns < 0].mean()) if len(arith_returns[arith_returns < 0]) > 0 else 0
        else:
            win_rate = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
            avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0

        # Avoid division by zero
        if avg_loss > 0:
            kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        else:
            kelly = win_rate

        # Scale kelly to avoid extreme values
        kelly = max(-1, min(1, kelly))

        # Calculate Sharpe Ratio (annualized) using log returns for better properties
        if 'log_returns' in df.columns:
            # For log returns, we need to annualize differently
            ret_mean = returns.mean() * 252
            ret_std = returns.std() * np.sqrt(252)
        else:
            ret_mean = returns.mean() * 252
            ret_std = returns.std() * np.sqrt(252)

        sharpe = ret_mean / ret_std if ret_std > 0 else 0

        # Scale sigma based on risk metrics
        risk_adjusted_sigma = sigma

        # Reduce sigma for extremely high drawdowns
        if max_drawdown < -0.5:  # >50% drawdown
            risk_adjusted_sigma *= 0.5
        elif max_drawdown < -0.3:  # >30% drawdown
            risk_adjusted_sigma *= 0.8

        # Reduce sigma for negative kelly values
        if kelly < 0:
            risk_adjusted_sigma *= (1 + kelly)  # Reduce by up to 100% for kelly = -1

        # Ensure sigma is within bounds
        risk_adjusted_sigma = max(0.01, min(1.0, risk_adjusted_sigma))

        return {
            "max_drawdown": max_drawdown,
            "cvar_95": cvar_95,
            "kelly": kelly,
            "sharpe": sharpe,
            "risk_adjusted_sigma": risk_adjusted_sigma
        }
    except Exception as e:
        print(f"[ERROR] Error calculating risk-adjusted metrics: {e}")
        return {
            "max_drawdown": 0,
            "risk_adjusted_sigma": sigma
        }


# Create Ensemble Prediction with log return components
def create_ensemble_prediction(momentum_score, reversion_score, lstm_prediction, dqn_recommendation,
                               volatility_data, market_regime, hurst_info, mean_reversion_info=None):
    """Create dynamically weighted ensemble with improved log return metrics"""

    # Base weights
    weights = {
        "momentum": 0.4,
        "reversion": 0.4,
        "lstm": 0.1,
        "dqn": 0.1
    }

    # Adjust weights based on volatility regime
    vol_regime = volatility_data.get("vol_regime", "Stable")
    if vol_regime == "Rising":
        # In rising volatility, favor mean reversion
        weights["momentum"] -= 0.1
        weights["reversion"] += 0.1
    elif vol_regime == "Falling":
        # In falling volatility, favor momentum
        weights["momentum"] += 0.1
        weights["reversion"] -= 0.1

    # Adjust weights based on market regime
    current_regime = market_regime.get("current_regime", "Unknown")
    if "High" in current_regime:
        # In high volatility regimes, increase ML model weights
        weights["lstm"] += 0.05
        weights["dqn"] += 0.05
        weights["momentum"] -= 0.05
        weights["reversion"] -= 0.05

    # Adjust based on Hurst exponent if available
    hurst_regime = hurst_info.get("regime", "Unknown")
    hurst_value = hurst_info.get("hurst", 0.5)

    # More precise adjustment based on hurst value
    if hurst_value < 0.3:  # Extremely strong mean reversion
        weights["reversion"] += 0.15
        weights["momentum"] -= 0.15
    elif hurst_value < 0.4:  # Strong mean reversion
        weights["reversion"] += 0.1
        weights["momentum"] -= 0.1
    elif hurst_value < 0.45:  # Moderate mean reversion
        weights["reversion"] += 0.05
        weights["momentum"] -= 0.05
    elif hurst_value > 0.7:  # Extremely strong trending
        weights["momentum"] += 0.15
        weights["reversion"] -= 0.15
    elif hurst_value > 0.6:  # Strong trending
        weights["momentum"] += 0.1
        weights["reversion"] -= 0.1
    elif hurst_value > 0.55:  # Moderate trending
        weights["momentum"] += 0.05
        weights["reversion"] -= 0.05

    # NEW: Adjust based on mean reversion half-life and beta if available
    if mean_reversion_info:
        half_life = mean_reversion_info.get("half_life", 0)
        beta = mean_reversion_info.get("beta", 0)

        # If strong mean reversion signal (negative beta, short half-life)
        if -1 < beta < -0.2 and 0 < half_life < 20:
            weights["reversion"] += 0.05
            weights["momentum"] -= 0.05
        # If no mean reversion (positive beta)
        elif beta > 0.1:
            weights["momentum"] += 0.05
            weights["reversion"] -= 0.05

    # NEW: Adjust based on volatility persistence if available
    vol_persistence = volatility_data.get("vol_persistence", 0.8)
    if vol_persistence > 0.9:  # High volatility persistence
        weights["reversion"] += 0.05
        weights["momentum"] -= 0.05
    elif vol_persistence < 0.6:  # Low volatility persistence
        weights["momentum"] += 0.03
        weights["reversion"] -= 0.03

    # Normalize weights to sum to 1
    total = sum(weights.values())
    for k in weights:
        weights[k] /= total

    # Calculate ensemble score
    ensemble_score = (
            weights["momentum"] * momentum_score +
            weights["reversion"] * (1 - reversion_score) +  # Invert reversion score (higher = more bearish)
            weights["lstm"] * lstm_prediction +
            weights["dqn"] * dqn_recommendation
    )

    return {
        "ensemble_score": ensemble_score,
        "weights": weights
    }


# PCA function to reduce dimensionality of features
def apply_pca(features_df):
    try:
        # Debug info about input
        print(f"[DEBUG] PCA input shape: {features_df.shape}")

        # Check if we have enough data
        if features_df.shape[0] < 10 or features_df.shape[1] < 5:
            print(f"[WARNING] Not enough data for PCA analysis: {features_df.shape}")
            return None, None

        # Select numerical columns that aren't NaN
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude columns that are mostly NaN
        valid_cols = []
        for col in numeric_cols:
            if features_df[col].isna().sum() < len(
                    features_df) * 0.3:  # At least 70% of values are not NaN (increased from 50%)
                valid_cols.append(col)

        if len(valid_cols) < 5:
            print(f"[WARNING] Not enough valid columns for PCA: {len(valid_cols)}")
            return None, None

        numeric_df = features_df[valid_cols].copy()

        # Fill remaining NaN values with column means
        for col in numeric_df.columns:
            if numeric_df[col].isna().any():
                numeric_df[col] = numeric_df[col].fillna(numeric_df[col].mean())

        print(f"[DEBUG] PCA numeric data shape after cleaning: {numeric_df.shape}")

        # Check for remaining NaN values
        if numeric_df.isna().sum().sum() > 0:
            print(f"[WARNING] NaN values still present after cleaning: {numeric_df.isna().sum().sum()}")
            # Replace remaining NaNs with 0
            numeric_df = numeric_df.fillna(0)

        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)

        # Apply PCA
        n_components = min(8, min(scaled_data.shape) - 1)  # Increased from 5
        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(scaled_data)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_results,
            columns=[f'PC{i + 1}' for i in range(pca_results.shape[1])],
            index=features_df.index
        )

        # Calculate explained variance for each component
        explained_variance = pca.explained_variance_ratio_

        print(f"[INFO] PCA explained variance: {explained_variance}")
        return pca_df, explained_variance
    except Exception as e:
        print(f"[ERROR] PCA failed: {e}")
        traceback.print_exc()
        return None, None

# Enhanced data preparation for LSTM prediction with log returns features (continued)
def prepare_lstm_data(data, time_steps=60):
    try:
        # Check if we have enough data
        if len(data) < time_steps + 10:
            print(f"[WARNING] Not enough data for LSTM: {len(data)} < {time_steps + 10}")
            return None, None, None

        # Use multiple features including log returns
        features = []

        # Always include closing price
        features.append(data['4. close'].values)

        # Include log returns if available (preferred)
        if 'log_returns' in data.columns:
            features.append(data['log_returns'].values)
            print("[INFO] Using log returns for LSTM features")
        # Otherwise use regular returns
        elif 'returns' in data.columns:
            features.append(data['returns'].values)
            print("[INFO] Using regular returns for LSTM features (log returns not available)")

        # Include volume if available with appropriate scaling
        if 'volume' in data.columns:
            # Log transform volume to reduce scale differences
            log_volume = np.log1p(data['volume'].values)
            features.append(log_volume)

        # Include log volatility if available (preferred)
        if 'log_volatility' in data.columns:
            features.append(data['log_volatility'].values)
            print("[INFO] Using log volatility for LSTM features")
        # Otherwise use regular volatility
        elif 'volatility' in data.columns:
            features.append(data['volatility'].values)
            print("[INFO] Using regular volatility for LSTM features (log volatility not available)")

        # Include RSI if available
        if 'RSI' in data.columns:
            # Normalize RSI to 0-1 scale
            normalized_rsi = data['RSI'].values / 100
            features.append(normalized_rsi)

        # Include MACD if available
        if 'MACD' in data.columns:
            # Normalize MACD using tanh for -1 to 1 range
            normalized_macd = np.tanh(data['MACD'].values / 5)
            features.append(normalized_macd)

        # Include log-based mean reversion indicators if available
        if 'log_returns_zscore' in data.columns:
            # Normalize with tanh to -1 to 1 range
            log_returns_z = np.tanh(data['log_returns_zscore'].values)
            features.append(log_returns_z)
            print("[INFO] Adding log returns z-score to LSTM features")

        if 'log_mr_potential' in data.columns:
            # Already normalized
            features.append(data['log_mr_potential'].values)
            print("[INFO] Adding log mean reversion potential to LSTM features")

        if 'log_expected_reversion_pct' in data.columns:
            # Normalize with tanh
            log_exp_rev = np.tanh(data['log_expected_reversion_pct'].values / 10)
            features.append(log_exp_rev)
            print("[INFO] Adding log expected reversion to LSTM features")

        # Include regular mean reversion indicators as fallback
        if 'BB_pctB' in data.columns and 'log_bb_pctB' not in data.columns:
            features.append(data['BB_pctB'].values)

        if 'dist_from_SMA200' in data.columns:
            # Use tanh to normalize to -1 to 1 range
            normalized_dist = np.tanh(data['dist_from_SMA200'].values * 5)
            features.append(normalized_dist)

        # Include Williams %R if available
        if 'Williams_%R' in data.columns:
            # Normalize from -100-0 to 0-1
            normalized_williams = (data['Williams_%R'].values + 100) / 100
            features.append(normalized_williams)

        # Include CMF if available
        if 'CMF' in data.columns:
            # Already in -1 to 1 range
            features.append(data['CMF'].values)

        # Stack features
        feature_array = np.column_stack(features)

        # Check for NaN values across all features
        if np.isnan(feature_array).any():
            print(f"[WARNING] NaN values in features, filling with forward fill")
            # Convert to DataFrame for easier handling of NaNs
            temp_df = pd.DataFrame(feature_array)
            # Fill NaN values
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill')
            feature_array = temp_df.values

        # Normalize the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_array)

        # Create sequences with all features
        X, y = [], []
        # Target is still the closing price (first feature)
        for i in range(len(scaled_features) - time_steps):
            X.append(scaled_features[i:i + time_steps])
            # For prediction target, use only the closing price column (index 0)
            y.append(scaled_features[i + time_steps, 0:1])

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Check shapes
        print(f"[DEBUG] Enhanced LSTM data shapes with log returns: X={X.shape}, y={y.shape}")

        return X, y, scaler
    except Exception as e:
        print(f"[ERROR] Error preparing enhanced LSTM data: {e}")
        traceback.print_exc()
        # Fallback to simpler preparation if enhanced fails
        try:
            print(f"[WARNING] Falling back to simple price-only LSTM preparation")
            # Get closing prices only
            prices = data['4. close'].values

            # Handle NaN values
            if np.isnan(prices).any():
                prices = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values

            # Reshape and scale
            prices_2d = prices.reshape(-1, 1)
            scaler = StandardScaler()
            scaled_prices = scaler.fit_transform(prices_2d)

            # Create sequences
            X, y = [], []
            for i in range(len(scaled_prices) - time_steps):
                X.append(scaled_prices[i:i + time_steps])
                y.append(scaled_prices[i + time_steps])

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            print(f"[DEBUG] Fallback LSTM data shapes: X={X.shape}, y={y.shape}")
            return X, y, scaler

        except Exception as e2:
            print(f"[ERROR] Fallback LSTM data preparation also failed: {e2}")
            return None, None, None


## Enhanced LSTM model for volatility prediction
#def build_lstm_model(input_shape):
#    """
#    Dummy LSTM model builder - TensorFlow disabled due to compatibility issues
#    Returns a non-functional model to allow code to run.
#    """
#    try:
#        print("[INFO] Using dummy LSTM model (TensorFlow disabled)")
#        # Create a simple dummy model that doesn't actually use TensorFlow
#        model = DummySequential()
#        print("[INFO] Built dummy LSTM model successfully")
#        return model
#    except Exception as e:
#        print(f"[ERROR] Error building dummy LSTM model: {e}")
#        traceback.print_exc()
#        return None

# Enhanced LSTM model for volatility prediction

def build_lstm_model(input_shape):
    try:
        # Highly sophisticated architecture for maximum prediction accuracy
        inputs = Input(shape=input_shape)

        # First LSTM layer with more units
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Second LSTM layer
        x = LSTM(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Third LSTM layer
        x = LSTM(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dense layers for feature extraction
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Final dense layer before output
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)

        # Output layer
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Use Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
        return model
    except Exception as e:
        print(f"[ERROR] Error building enhanced LSTM model: {e}")
        traceback.print_exc()

        # Fallback to simpler model if complex one fails
        try:
            inputs = Input(shape=input_shape)
            x = LSTM(64, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(64, return_sequences=False)(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(1)(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer="adam", loss="mse")
            return model
        except Exception as e2:
            print(f"[ERROR] Fallback LSTM model also failed: {e2}")

            # Very simple fallback
            try:
                inputs = Input(shape=input_shape)
                x = LSTM(32, return_sequences=False)(inputs)
                outputs = Dense(1)(x)
                model = Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer="adam", loss="mse")
                return model
            except Exception as e3:
                print(f"[ERROR] All LSTM model attempts failed: {e3}")
                return None

## Simplified LSTM prediction (TensorFlow disabled)
#def predict_with_lstm(data):
#    try:
#        print("[INFO] Using simplified LSTM prediction (TensorFlow disabled)")
#        # No need for execution time tracking as this is a simplified implementation
#
#        # Instead of actual LSTM prediction, generate a reasonable value based on available data
#        # Use a data-driven approach using traditional analysis
#        
#        # Check if we have volatility data
#        if 'log_volatility' in data.columns:
#            # Calculate the ratio of recent to older volatility 
#            # as a proxy for volatility direction
#            recent_vol = data['log_volatility'].iloc[-15:].mean()
#            older_vol = data['log_volatility'].iloc[-45:-15].mean() if len(data) >= 45 else recent_vol
#            
#            # Avoid division by zero
#            if older_vol < 0.0001:
#                vol_ratio = 1.0
#            else:
#                vol_ratio = recent_vol / older_vol
#                
#            # Normalize to a 0-1 range
#            lstm_prediction = min(1.0, max(0.1, vol_ratio))
#            print(f"[INFO] LSTM prediction using log volatility trends: {lstm_prediction:.3f}")
#            return lstm_prediction
#            
#        elif 'volatility' in data.columns:
#            # Same approach but with standard volatility
#            recent_vol = data['volatility'].iloc[-15:].mean()
#            older_vol = data['volatility'].iloc[-45:-15].mean() if len(data) >= 45 else recent_vol
#            
#            # Avoid division by zero
#            if older_vol < 0.0001:
#                vol_ratio = 1.0
#            else:
#                vol_ratio = recent_vol / older_vol
#                
#            # Normalize to a 0-1 range
#            lstm_prediction = min(1.0, max(0.1, vol_ratio))
#            print(f"[INFO] LSTM prediction using standard volatility trends: {lstm_prediction:.3f}")
#            return lstm_prediction
#        
#        else:
#            # For cases where we don't have volatility data, use RSI as a signal
#            if 'RSI' in data.columns:
#                # Get most recent RSI
#                rsi = data['RSI'].iloc[-1]
#                
#                # Map RSI to a 0-1 range
#                # RSI 0-30: bearish (low values), 70-100: bullish (high values)
#                if rsi <= 30:
#                    lstm_prediction = 0.2
#                elif rsi >= 70:
#                    lstm_prediction = 0.8
#                else:
#                    # Linear interpolation between 30-70
#                    lstm_prediction = 0.2 + ((rsi - 30) / 40) * 0.6
#                    
#                print(f"[INFO] LSTM prediction using RSI trends: {lstm_prediction:.3f}")
#                return lstm_prediction
#                
#            # Fallback to a neutral prediction
#            print("[INFO] LSTM prediction fallback to neutral value")
#            return 0.5
#            
#    except Exception as e:
#        print(f"[ERROR] Error in LSTM prediction: {e}")
#        traceback.print_exc()
#        return 0

def predict_with_lstm(data):
    try:
        # Set a maximum execution time - significantly increased for thorough training
        max_execution_time = 240  # 4 minutes max (increased from 2 minutes)
        start_time = time.time()

        # Require less data to attempt prediction
        if len(data) < 60:
            print("[WARNING] Not enough data for LSTM model")
            return 0

        # Use a larger window for more context
        time_steps = 60  # Increased for better prediction accuracy

        # Prepare data with enhanced features including log returns
        X, y, scaler = prepare_lstm_data(data, time_steps=time_steps)
        if X is None or y is None or scaler is None:
            print("[WARNING] Failed to prepare LSTM data")
            return 0

        # More lenient on required data size
        if len(X) < 8:
            print(f"[WARNING] Not enough data after preparation: {len(X)}")
            return 0

        # Build enhanced model
        model = build_lstm_model((X.shape[1], X.shape[2]))
        if model is None:
            print("[WARNING] Failed to build LSTM model")
            return 0

        # Use more training data for better learning
        max_samples = 1000  # Significantly increased from 500
        if len(X) > max_samples:
            # Use evenly spaced samples to get good representation
            indices = np.linspace(0, len(X) - 1, max_samples, dtype=int)
            X_train = X[indices]
            y_train = y[indices]
        else:
            X_train = X
            y_train = y

        # Use try/except for model training
        try:
            # Check if we're still within time limit
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before training")
                # Use a better fallback prediction based on recent volatility
                if 'log_volatility' in data.columns:
                    return data['log_volatility'].iloc[-15:].mean() / data['log_volatility'].iloc[-45:].mean()
                else:
                    return data['volatility'].iloc[-15:].mean() / data['volatility'].iloc[-45:].mean()

            # Train model with more epochs and better callbacks
            early_stop = EarlyStopping(monitor='loss', patience=5, verbose=0)  # Increased patience
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)

            # Set parameters for extensive training
            model.fit(
                X_train, y_train,
                epochs=30,  # Doubled from 15
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0,
                shuffle=True
            )

            # Extra training round with lower learning rate for fine-tuning
            if time.time() - start_time < max_execution_time * 0.6:
                # Reduce learning rate for fine-tuning
                try:
                    K = model.optimizer.learning_rate
                    model.optimizer.learning_rate = K * 0.3
                except:
                    # If unable to modify learning rate directly
                    model.optimizer.lr.assign(model.optimizer.lr * 0.3)

                model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    verbose=0,
                    shuffle=True
                )

            # Final fine-tuning with small batch size if time permits
            if time.time() - start_time < max_execution_time * 0.8:
                model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=16,  # Smaller batch size for final tuning
                    verbose=0,
                    shuffle=True
                )

        except Exception as e:
            print(f"[ERROR] LSTM model training failed: {e}")
            return 0

        # Make prediction for future volatility
        try:
            # Check time again
            if time.time() - start_time > max_execution_time:
                print("[WARNING] LSTM execution time limit reached before prediction")
                return 0.5  # Return a neutral value

            # Use ensemble of predictions from the last few sequences for better stability
            num_pred_samples = min(10, len(X))  # Increased from 5
            predictions = []

            for i in range(num_pred_samples):
                seq_idx = len(X) - i - 1
                if seq_idx >= 0:  # Check if index is valid
                    sequence = X[seq_idx].reshape(1, X.shape[1], X.shape[2])
                    pred = model.predict(sequence, verbose=0)[0][0]
                    predictions.append(pred)

            if not predictions:
                return 0.5  # Default if no valid predictions

            # Weight more recent predictions higher
            weights = np.linspace(1.0, 0.5, len(predictions))
            weights = weights / np.sum(weights)  # Normalize

            avg_prediction = np.sum(np.array(predictions) * weights)

            # Get weighted average of recent actual values
            last_actuals = y[-num_pred_samples:].flatten()
            last_actual_weights = np.linspace(1.0, 0.5, len(last_actuals))
            last_actual_weights = last_actual_weights / np.sum(last_actual_weights)
            last_actual = np.sum(last_actuals * last_actual_weights)

            # Avoid division by zero
            if abs(last_actual) < 1e-6:
                predicted_volatility_change = abs(avg_prediction)
            else:
                predicted_volatility_change = abs((avg_prediction - last_actual) / last_actual)

            print(f"[DEBUG] LSTM prediction: {predicted_volatility_change}")

            # Return a more nuanced measure capped at 1.0
            return min(1.0, max(0.1, predicted_volatility_change))

        except Exception as e:
            print(f"[ERROR] LSTM prediction failed: {e}")
            return 0
    except Exception as e:
        print(f"[ERROR] Error in LSTM prediction: {e}")
        traceback.print_exc()
        return 0

# Enhanced DQN Agent implementation for more accurate predictions
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Substantially increased from 5000
        self.gamma = 0.99  # Increased from 0.97 for more future focus
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # Lower min epsilon for better exploitation
        self.epsilon_decay = 0.97  # Slower decay for better exploration
        self.model = self._build_model()
        self.target_model = self._build_model()  # Separate target network
        self.target_update_counter = 0
        self.target_update_freq = 5  # Update target more frequently (was 10)
        self.max_training_time = 220  # 2 minutes maximum (doubled from 60s)
        self.batch_history = []  # Track training history

    def _build_model(self):
        try:
            print(f"[DEBUG] Building DQN model with input shape: ({self.state_size},)")
            # Advanced model architecture for superior learning
            model = Sequential([
                Dense(256, activation="relu", input_shape=(self.state_size,)),  # Dynamic input shape
                BatchNormalization(),
                Dropout(0.3),  # More aggressive dropout
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dropout(0.1),
                Dense(self.action_size, activation="linear")
            ])

            # Use Adam optimizer with custom learning rate
            optimizer = Adam(learning_rate=0.0005)
            model.compile(optimizer=optimizer, loss="mse")
            return model
        except Exception as e:
            print(f"[ERROR] Error building enhanced DQN model: {e}")
            traceback.print_exc()

            # Fallback to simpler model
            try:
                print(f"[DEBUG] Attempting to build simpler model with input shape: ({self.state_size},)")
                model = Sequential([
                    Dense(128, activation="relu", input_shape=(self.state_size,)),
                    Dropout(0.2),
                    Dense(128, activation="relu"),
                    Dropout(0.2),
                    Dense(64, activation="relu"),
                    Dense(self.action_size, activation="linear")
                ])
                model.compile(optimizer="adam", loss="mse")
                return model
            except Exception as e2:
                print(f"[ERROR] Error building intermediate DQN model: {e2}")
                traceback.print_exc()

                # Even simpler fallback model
                try:
                    print(f"[DEBUG] Attempting to build very simple model with input shape: ({self.state_size},)")
                    model = Sequential([
                        Dense(64, activation="relu", input_shape=(self.state_size,)),
                        Dense(64, activation="relu"),
                        Dense(self.action_size, activation="linear")
                    ])
                    model.compile(optimizer="adam", loss="mse")
                    return model
                except Exception as e3:
                    print(f"[ERROR] Error building simplest DQN model: {e3}")
                    traceback.print_exc()

                    # Final minimal fallback
                    try:
                        print(f"[DEBUG] Attempting to build minimal model with input shape: ({self.state_size},)")
                        model = Sequential([
                            Dense(32, activation="relu", input_shape=(self.state_size,)),
                            Dense(self.action_size, activation="linear")
                        ])
                        model.compile(optimizer="adam", loss="mse")
                        return model
                    except Exception as e4:
                        print(f"[ERROR] All DQN model attempts failed: {e4}")
                        traceback.print_exc()
                        return None

    # Update target model (for more stable learning)
    def update_target_model(self):
        if self.model is not None and self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())
            print("[DEBUG] DQN target model updated")
        else:
            print("[WARNING] Cannot update target model: models not initialized")

    def remember(self, state, action, reward, next_state, done):
        # Safety check for state dimensions
        if state.shape[1] != self.state_size:
            print(f"[WARNING] State size mismatch in remember: expected {self.state_size}, got {state.shape[1]}")
            # Rebuild model with new state size
            self.state_size = state.shape[1]
            self.model = self._build_model()
            self.target_model = self._build_model()
            
        # Only add to memory if not full
        if len(self.memory) < self.memory.maxlen:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        try:
            # Safety check for state dimensions
            if state.shape[1] != self.state_size:
                print(f"[WARNING] State size mismatch in act: expected {self.state_size}, got {state.shape[1]}")
                # Rebuild model with new state size
                self.state_size = state.shape[1]
                self.model = self._build_model()
                self.target_model = self._build_model()
                
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            if self.model is None:
                return random.randrange(self.action_size)

            # Get multiple predictions with noise for ensembling
            num_predictions = 6
            actions = []

            for _ in range(num_predictions):
                act_values = self.model.predict(state, verbose=0)
                # Add small noise for exploration
                act_values += np.random.normal(0, 0.05, size=act_values.shape)
                actions.append(np.argmax(act_values[0]))

            # Return most common action
            counts = np.bincount(actions)
            return np.argmax(counts)

        except Exception as e:
            print(f"[ERROR] Error in DQN act method: {e}")
            traceback.print_exc()
            return random.randrange(self.action_size)

    def replay(self, batch_size):
        if len(self.memory) < batch_size or self.model is None:
            return

        # Add timeout mechanism
        start_time = time.time()

        try:
            # Verify memory state dimensions match agent state size
            sample_state = self.memory[0][0]
            memory_state_size = sample_state.shape[1]
            
            if memory_state_size != self.state_size:
                print(f"[WARNING] Memory state size ({memory_state_size}) differs from agent state size ({self.state_size})")
                # Rebuild models with correct state size
                self.state_size = memory_state_size
                self.model = self._build_model()
                self.target_model = self._build_model()
            
            # Track training iterations for adaptive learning
            train_iterations = 0

            # Use larger batch sizes for more stable learning
            actual_batch_size = min(batch_size, len(self.memory))
            minibatch = random.sample(self.memory, actual_batch_size)

            # Process in reasonable chunks for better performance
            chunk_size = 64  # Doubled from 32 for better batch learning

            for i in range(0, len(minibatch), chunk_size):
                chunk = minibatch[i:i + chunk_size]

                # Check timeout
                if time.time() - start_time > self.max_training_time:
                    print("[WARNING] DQN training timeout reached")
                    break
                

                # Process chunk
                states = np.vstack([x[0] for x in chunk])
                
                # Extra safety check on state dimensions
                if states.shape[1] != self.state_size:
                    print(f"[ERROR] State shape mismatch during replay: {states.shape} vs expected ({self.state_size},)")
                    # Try to rebuild model with new dimensions
                    self.state_size = states.shape[1]
                    self.model = self._build_model()
                    self.target_model = self._build_model()
                    # Skip this batch to avoid errors
                    continue

                # Use the target network for more stable learning
                next_states = np.vstack([x[3] for x in chunk])
                actions = np.array([x[1] for x in chunk])
                rewards = np.array([x[2] for x in chunk])
                dones = np.array([x[4] for x in chunk])

                # Current Q values
                targets = self.model.predict(states, verbose=0)

                # Get next Q values from target model
                next_q_values = self.target_model.predict(next_states, verbose=0)

                # Update Q values - more efficient vectorized approach
                for j in range(len(chunk)):
                    if dones[j]:
                        targets[j, actions[j]] = rewards[j]
                    else:
                        # Add small noise to next state values for exploration
                        next_qs = next_q_values[j] + np.random.normal(0, 0.01, size=next_q_values[j].shape)
                        targets[j, actions[j]] = rewards[j] + self.gamma * np.max(next_qs)

                # Fit with more epochs for better learning
                history = self.model.fit(
                    states,
                    targets,
                    epochs=8,  # Increased from 3
                    batch_size=len(chunk),
                    verbose=0
                )

                # Track training progress
                self.batch_history.append(history.history['loss'][-1])
                train_iterations += 1

            # Update epsilon with a more gradual decay
            if self.epsilon > self.epsilon_min:
                # Adaptive decay based on memory size
                decay_rate = self.epsilon_decay + (0.01 * min(1.0, len(self.memory) / 5000))
                self.epsilon *= decay_rate
                self.epsilon = max(self.epsilon, self.epsilon_min)  # Ensure we don't go below min

            # Update target network periodically
            self.target_update_counter += 1
            if self.target_update_counter >= self.target_update_freq:
                self.update_target_model()
                self.target_update_counter = 0

            # Report training progress
            if self.batch_history:
                avg_loss = sum(self.batch_history[-train_iterations:]) / max(1, train_iterations)
                print(f"[DEBUG] DQN training - avg loss: {avg_loss:.5f}, epsilon: {self.epsilon:.3f}")

        except Exception as e:
            print(f"[ERROR] Error in DQN replay: {e}")
            traceback.print_exc()
#

def get_dqn_recommendation(data):
    try:
        # More lenient on required data
        if len(data) < 40:
            print("[WARNING] Not enough data for DQN (<40)")
            return 0.5  # Neutral score

        # Set timeout for the entire function - significantly increased for thorough training
        function_start_time = time.time()
        max_function_time = 480  # 4 minutes (doubled from 2 minutes)

        # Prepare state features with more historical context
        lookback = 20  # Further increased from 10 for better historical context

        # Extract more features for a richer state representation
        features = []

        # Basic indicators - prefer log returns if available
        if 'log_returns' in data.columns:
            features.append(data['log_returns'].values[-lookback:])
            print("[INFO] Using log returns for DQN features")
        elif 'returns' in data.columns:
            features.append(data['returns'].values[-lookback:])

        # Prefer log volatility if available
        if 'log_volatility' in data.columns:
            features.append(data['log_volatility'].values[-lookback:])
            print("[INFO] Using log volatility for DQN features")
        elif 'volatility' in data.columns:
            features.append(data['volatility'].values[-lookback:])

        # Technical indicators
        if 'RSI' in data.columns:
            rsi = data['RSI'].values[-lookback:] / 100  # Normalize to 0-1
            features.append(rsi)
        if 'MACD' in data.columns:
            macd = np.tanh(data['MACD'].values[-lookback:] / 5)
            features.append(macd)
        if 'SMA20' in data.columns and 'SMA50' in data.columns:
            sma20 = data['SMA20'].values[-lookback:]
            sma50 = data['SMA50'].values[-lookback:]
            with np.errstate(divide='ignore', invalid='ignore'):
                sma_ratio = np.where(sma50 != 0, sma20 / sma50, 1.0)
            sma_ratio = np.nan_to_num(sma_ratio, nan=1.0)
            sma_trend = np.tanh((sma_ratio - 1.0) * 5)
            features.append(sma_trend)

        # Log-based mean reversion indicators (preferred)
        if 'log_returns_zscore' in data.columns:
            log_z = np.tanh(data['log_returns_zscore'].values[-lookback:])
            features.append(log_z)
            print("[INFO] Adding log returns Z-score to DQN features")
        if 'log_mr_potential' in data.columns:
            log_mr = data['log_mr_potential'].values[-lookback:]
            features.append(log_mr)
            print("[INFO] Adding log mean reversion potential to DQN features")
        if 'log_expected_reversion_pct' in data.columns:
            log_rev = np.tanh(data['log_expected_reversion_pct'].values[-lookback:] / 10)
            features.append(log_rev)
            print("[INFO] Adding log expected reversion to DQN features")
        if 'log_bb_pctB' in data.columns:
            log_bb = data['log_bb_pctB'].values[-lookback:]
            features.append(log_bb)
            print("[INFO] Adding log BB %B to DQN features")
        if 'log_autocorr_5' in data.columns:
            log_autocorr = data['log_autocorr_5'].values[-lookback:]
            features.append(log_autocorr)
            print("[INFO] Adding log autocorrelation to DQN features")

        # More features - add as many as available
        # ... [other features remain unchanged]

        # Stack all features into the state
        features = [np.nan_to_num(f, nan=0.0) for f in features]  # Handle NaNs
        
        if features:
            state = np.concatenate(features)
            state_size = len(state)  # Dynamic state size based on available features
        else:
            # Fallback if no features available
            state_size = 10
            state = np.zeros(state_size)
        
        print(f"[INFO] Using {state_size} features for DQN state")

        # Define action space: 0=Sell, 1=Hold, 2=Buy
        action_size = 3
        agent = DQNAgent(state_size=state_size, action_size=action_size)

        if agent.model is None:
            print("[WARNING] Failed to create DQN model")
            return 0.5  # Neutral score

        # Use more training data for better learning
        max_train_points = min(500, len(data) - (lookback + 1))  # Increased from 200

        # Use appropriate step size to get good coverage of data
        step_size = max(1, (len(data) - (lookback + 1)) // 500)  # Adjusted for more points

        # First pass: collect experiences without training to populate memory
        print("[DEBUG] DQN collecting initial experiences with log returns...")

        # Track experience collection progress
        collection_start = time.time()
        experiences_collected = 0

        for i in range(0, max_train_points * step_size, step_size):
            # Check timeout
            if time.time() - function_start_time > max_function_time * 0.25:  # Use 25% of time for collection
                print(f"[WARNING] DQN experience collection timeout reached after {experiences_collected} experiences")
                break

            # Get index with bounds checking
            idx = min(i, len(data) - (lookback + 1))
            next_idx = min(idx + 1, len(data) - lookback - 1)

            # Extract features for current state
            try:
                # Create state for this time point
                current_features = []

                # Extract features for current timepoint (similar to above)
                if 'log_returns' in data.columns:
                    values = data['log_returns'].values[idx:idx + lookback]
                    current_features.append(np.nan_to_num(values, nan=0.0))
                elif 'returns' in data.columns:
                    values = data['returns'].values[idx:idx + lookback]
                    current_features.append(np.nan_to_num(values, nan=0.0))
                
                # Add more features (similar to above)
                if 'RSI' in data.columns:
                    values = data['RSI'].values[idx:idx + lookback] / 100
                    current_features.append(np.nan_to_num(values, nan=0.5))
                if 'MACD' in data.columns:
                    values = np.tanh(data['MACD'].values[idx:idx + lookback] / 5)
                    current_features.append(np.nan_to_num(values, nan=0.0))
                if 'SMA20' in data.columns and 'SMA50' in data.columns:
                    sma20 = data['SMA20'].values[idx:idx + lookback]
                    sma50 = data['SMA50'].values[idx:idx + lookback]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sma_ratio = np.where(sma50 != 0, sma20 / sma50, 1.0)
                    sma_ratio = np.nan_to_num(sma_ratio, nan=1.0)
                    sma_trend = np.tanh((sma_ratio - 1.0) * 5)
                    current_features.append(sma_trend)

                # Create current state with dynamic dimension
                if current_features:
                    current_state_array = np.concatenate(current_features)
                    current_state = current_state_array.reshape(1, -1)  # Dynamic reshaping
                else:
                    # Fallback with same dimension as agent expects
                    current_state = np.zeros((1, state_size))

                # Create next state (simplified)
                next_state = current_state.copy()  # Dummy next state

                # Default values in case of exception
                action = 1  # Default to 'Hold' action
                reward = 0.0
                
                # Enhanced reward function based on log returns
                try:
                    # Base reward on forward log return if available
                    if 'log_returns' in data.columns and next_idx + lookback < len(data):
                        price_return = data['log_returns'].values[next_idx + lookback - 1]
                    elif next_idx + lookback < len(data):
                        price_return = data['returns'].values[next_idx + lookback - 1]
                    else:
                        price_return = 0

                    # Get current action for this state
                    action = agent.act(current_state)

                    # Adjust reward based on action-outcome alignment
                    if action == 2:  # Buy
                        reward = price_return
                    elif action == 0:  # Sell
                        reward = -price_return
                    else:  # Hold
                        reward = abs(price_return) * 0.3  # Small reward for being right about direction

                    # Add small penalty for extreme actions
                    if action != 1:  # Not hold
                        reward -= 0.001  # Small transaction cost

                    # Ensure reward is within reasonable bounds
                    reward = np.clip(reward, -0.1, 0.1)

                    if np.isnan(reward):
                        reward = 0.0
                except:
                    reward = 0.0

                # Record experience
                is_terminal = False
                agent.remember(current_state, action, reward, next_state, is_terminal)
                experiences_collected += 1

            except Exception as e:
                print(f"[WARNING] Error in DQN experience collection: {e}")
                continue

        print(f"[INFO] Collected {experiences_collected} experiences in {time.time() - collection_start:.1f}s")

        # Training phase
        if len(agent.memory) > 0:
            print("[INFO] Training DQN agent...")
            training_start = time.time()
            
            # Multiple training iterations for better learning
            iterations = min(30, len(agent.memory) // 32)
            batch_size = min(256, len(agent.memory))
            
            for _ in range(iterations):
                if time.time() - function_start_time > max_function_time * 0.75:
                    print("[WARNING] DQN training timeout reached")
                    break
                agent.replay(batch_size)
            
            print(f"[INFO] DQN training completed in {time.time() - training_start:.1f}s")

        # Get recommendation
        if agent.model is None:
            print("[WARNING] DQN model not available for recommendation")
            return 0.5
        
        # Use the last state for prediction
        try:
            # Create state from most recent data with dynamic sizing
            final_features = []
            
            # Extract the same features as above for the most recent data
            if 'log_returns' in data.columns:
                values = data['log_returns'].values[-lookback:]
                final_features.append(np.nan_to_num(values, nan=0.0))
            elif 'returns' in data.columns:
                values = data['returns'].values[-lookback:]
                final_features.append(np.nan_to_num(values, nan=0.0))
            
            # Add more features (same as above)
            if 'RSI' in data.columns:
                values = data['RSI'].values[-lookback:] / 100
                final_features.append(np.nan_to_num(values, nan=0.5))
            if 'MACD' in data.columns:
                values = np.tanh(data['MACD'].values[-lookback:] / 5)
                final_features.append(np.nan_to_num(values, nan=0.0))
            if 'SMA20' in data.columns and 'SMA50' in data.columns:
                sma20 = data['SMA20'].values[-lookback:]
                sma50 = data['SMA50'].values[-lookback:]
                with np.errstate(divide='ignore', invalid='ignore'):
                    sma_ratio = np.where(sma50 != 0, sma20 / sma50, 1.0)
                sma_ratio = np.nan_to_num(sma_ratio, nan=1.0)
                sma_trend = np.tanh((sma_ratio - 1.0) * 5)
                final_features.append(sma_trend)
            
            # Create final state with dynamic size
            if final_features:
                final_state_array = np.concatenate(final_features)
                final_state = final_state_array.reshape(1, -1)  # Dynamic reshaping
            else:
                # Fallback with expected size
                final_state = np.zeros((1, state_size))
            
            # Get action probabilities
            action_values = agent.model.predict(final_state, verbose=0)[0]
            
            # Normalize to get probabilities with error handling
            # Use softmax for numerical stability
            shifted_values = action_values - np.max(action_values)  # For numerical stability
            exp_values = np.exp(shifted_values)
            action_probs = exp_values / np.sum(exp_values)
            
            # If we have NaN values, use a default neutral probability distribution
            if np.isnan(action_probs).any():
                print("[WARNING] NaN detected in action probabilities, using neutral distribution")
                action_probs = np.array([0.25, 0.5, 0.25])  # Slightly favor hold
            
            # Calculate recommendation score (0-1 scale)
            # 0 = Strong Sell, 0.5 = Hold, 1 = Strong Buy
            dqn_score = 0.5 * action_probs[1] + 1.0 * action_probs[2]
            
            print(f"[INFO] DQN recommendation score: {dqn_score:.3f}")
            return dqn_score
            
        except Exception as e:
            print(f"[ERROR] Error generating DQN recommendation: {e}")
            return 0.5  # Neutral score

    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        traceback.print_exc()
        return 0.5  # Neutral score

def generate_price_predictions(data, analysis_details, forecast_days=60):
    """
    Generate price predictions based on analysis results
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing historical price data
    analysis_details: dict
        Dictionary with analysis details
    forecast_days: int
        Number of days to forecast
        
    Returns:
    --------
    dict
        Dictionary with prediction results
    """
    try:
        # Get current price
        price_col = 'close' if 'close' in data.columns else '4. close'
        current_price = data[price_col].iloc[-1]
        
        # Extract key factors from analysis
        sigma = analysis_details.get('sigma', 0.5)  # Overall bullish/bearish score
        momentum_score = analysis_details.get("momentum_score", 0.5)
        reversion_score = analysis_details.get("reversion_score", 0.5)
        hurst_exponent = analysis_details.get("hurst_exponent", 0.5)
        market_regime = analysis_details.get("market_regime", "Unknown")
        volatility_regime = analysis_details.get("volatility_regime", "Stable")
        
        # Get historical volatility for prediction bands
        if 'log_volatility' in data.columns:
            hist_volatility = data['log_volatility'].iloc[-30:].mean() * np.sqrt(252)
            print(f"[INFO] Using log volatility for prediction bands: {hist_volatility:.4f}")
        elif 'volatility' in data.columns:
            hist_volatility = data['volatility'].iloc[-30:].mean() * np.sqrt(252)
            print(f"[INFO] Using standard volatility for prediction bands: {hist_volatility:.4f}")
        else:
            # Estimate volatility from returns
            returns = np.log(data[price_col] / data[price_col].shift(1)).dropna()
            hist_volatility = returns.iloc[-30:].std() * np.sqrt(252)
            print(f"[INFO] Estimated volatility for prediction bands: {hist_volatility:.4f}")
        
        # Adjust volatility based on volatility regime
        vol_multiplier = 1.0
        if volatility_regime == "Rising":
            vol_multiplier = 1.3
        elif volatility_regime == "Falling":
            vol_multiplier = 0.8
            
        adjusted_volatility = hist_volatility * vol_multiplier
        
        # Calculate expected return based on sigma
        # Map sigma from 0-1 to expected annualized return from -25% to +25%
        expected_annual_return = (sigma - 0.5) * 0.5  # -25% to +25% annual return
        
        # Adjust based on market regime
        regime_multiplier = 1.0
        if "Bull" in market_regime:
            regime_multiplier = 1.2
        elif "Bear" in market_regime:
            regime_multiplier = 0.8
            
        expected_annual_return *= regime_multiplier
        
        # Calculate daily expected return
        daily_return = (1 + expected_annual_return) ** (1/252) - 1
        
        # Create date range for forecast
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.today()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Generate price paths with random variations based on volatility
        num_paths = 100  # Number of Monte Carlo paths
        price_paths = np.zeros((num_paths, forecast_days))
        
        # Daily volatility
        daily_vol = adjusted_volatility / np.sqrt(252)
        
        # Consider Hurst exponent for path generation (trend vs mean reversion)
        # Lower Hurst = more mean reversion, Higher Hurst = more trending
        for i in range(num_paths):
            # Path starts at current price
            price_paths[i, 0] = current_price * (1 + daily_return + np.random.normal(0, daily_vol))
            
            mean_reversion_strength = 1.0 - hurst_exponent
            
            for j in range(1, forecast_days):
                # Calculate drift (expected return adjusted for regime)
                drift = daily_return
                
                # Calculate random component
                random_component = np.random.normal(0, daily_vol)
                
                # Calculate mean reversion component (proportional to distance from trend line)
                if j > 5:  # After a few days to establish a trend
                    # Calculate current trend
                    trend_price = current_price * (1 + daily_return) ** (j + 1)
                    # Distance from trend
                    distance = price_paths[i, j-1] / trend_price - 1
                    # Mean reversion component (pulls back toward trend)
                    mean_reversion = -distance * mean_reversion_strength * 0.1
                else:
                    mean_reversion = 0
                
                # Generate next price with drift, randomness, and mean reversion
                price_paths[i, j] = price_paths[i, j-1] * (1 + drift + random_component + mean_reversion)
        
        # Calculate mean path and confidence intervals
        mean_path = np.mean(price_paths, axis=0)
        lower_bound_95 = np.percentile(price_paths, 2.5, axis=0)
        upper_bound_95 = np.percentile(price_paths, 97.5, axis=0)
        lower_bound_68 = np.percentile(price_paths, 16, axis=0)
        upper_bound_68 = np.percentile(price_paths, 84, axis=0)
        
        # Calculate 30, 60 day price targets
        price_target_30d = mean_path[min(29, forecast_days-1)]
        price_target_60d = mean_path[min(59, forecast_days-1)]
        
        # Calculate expected returns
        expected_return_30d = (price_target_30d / current_price - 1) * 100
        expected_return_60d = (price_target_60d / current_price - 1) * 100
        
        # Return prediction results
        return {
            "current_price": current_price,
            "forecast_dates": forecast_dates,
            "mean_path": mean_path,
            "lower_bound_95": lower_bound_95,
            "upper_bound_95": upper_bound_95,
            "lower_bound_68": lower_bound_68,
            "upper_bound_68": upper_bound_68,
            "price_target_30d": price_target_30d,
            "price_target_60d": price_target_60d,
            "expected_return_30d": expected_return_30d,
            "expected_return_60d": expected_return_60d,
            "hist_volatility": hist_volatility,
            "adjusted_volatility": adjusted_volatility,
            "expected_annual_return": expected_annual_return
        }
    except Exception as e:
        print(f"[ERROR] Error generating price predictions: {e}")
        traceback.print_exc()
        return None

def create_prediction_plot(stock_data, prediction_data, symbol, plot_dir="prediction_plots"):
    """
    Create stock price prediction plot based on analysis
    
    Parameters:
    -----------
    stock_data: pandas DataFrame
        DataFrame containing historical price data
    prediction_data: dict
        Dictionary with prediction results
    symbol: str
        Stock symbol for the plot title
    plot_dir: str
        Directory to save the plot
        
    Returns:
    --------
    str
        Path to the saved plot
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)
        print(f"[INFO] Ensuring directory exists for prediction plots: {plot_dir}")
        
        # Get the current timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{symbol}_prediction_{timestamp}.png"
        plot_path = os.path.join(plot_dir, plot_filename)
        
        # Create plot with historical and predicted data
        plt.figure(figsize=(12, 8))
        
        # Get price column
        price_col = 'close' if 'close' in stock_data.columns else '4. close'
        
        # Get historical dates and prices for plotting
        if isinstance(stock_data.index, pd.DatetimeIndex):
            hist_dates = stock_data.index[-120:]  # Last 120 days of history
            hist_prices = stock_data[price_col][-120:]
        else:
            # If index isn't datetime, create placeholder dates
            hist_dates = pd.date_range(end=pd.Timestamp.today(), periods=min(120, len(stock_data)))
            hist_prices = stock_data[price_col][-min(120, len(stock_data)):]
        
        # Plot historical data
        plt.plot(hist_dates, hist_prices, label="Historical Price", color='blue', linewidth=2)
        
        # Extract prediction data
        forecast_dates = prediction_data['forecast_dates']
        mean_path = prediction_data['mean_path']
        lower_bound_95 = prediction_data['lower_bound_95']
        upper_bound_95 = prediction_data['upper_bound_95']
        lower_bound_68 = prediction_data['lower_bound_68']
        upper_bound_68 = prediction_data['upper_bound_68']
        
        # Plot prediction data
        plt.plot(forecast_dates, mean_path, label="Mean Forecast", color='green', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_dates, lower_bound_95, upper_bound_95, color='green', alpha=0.1, label="95% Confidence")
        plt.fill_between(forecast_dates, lower_bound_68, upper_bound_68, color='green', alpha=0.2, label="68% Confidence")
        
        # Add price targets
        price_target_30d = prediction_data['price_target_30d']
        price_target_60d = prediction_data['price_target_60d']
        expected_return_30d = prediction_data['expected_return_30d']
        expected_return_60d = prediction_data['expected_return_60d']
        
        # Mark 30-day and 60-day targets
        if len(forecast_dates) >= 30:
            plt.plot(forecast_dates[29], price_target_30d, 'o', color='purple', markersize=8)
            plt.annotate(f"30d: ${price_target_30d:.2f} ({expected_return_30d:.1f}%)", 
                        (forecast_dates[29], price_target_30d),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='purple')
        
        if len(forecast_dates) >= 60:
            plt.plot(forecast_dates[59], price_target_60d, 'o', color='red', markersize=8)
            plt.annotate(f"60d: ${price_target_60d:.2f} ({expected_return_60d:.1f}%)", 
                        (forecast_dates[59], price_target_60d),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=10, color='red')
        
        # Add title and labels
        current_price = prediction_data['current_price']
        volatility = prediction_data['adjusted_volatility']
        annual_return = prediction_data['expected_annual_return']
        
        plt.title(f"{symbol} Price Forecast\nCurrent: ${current_price:.2f} | Expected Annual Return: {annual_return*100:.1f}% | Volatility: {volatility*100:.1f}%", 
                fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        
        # Format x-axis to show dates properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Price prediction plot saved to {plot_path}")
        return plot_path
    
    except Exception as e:
        print(f"[ERROR] Error creating prediction plot: {e}")
        traceback.print_exc()
        return None

def append_stock_result(result):
    """
    Append detailed stock analysis result to the output file
    
    Parameters:
    -----------
    result: dict
        Stock analysis result
    """
    try:
        with open(OUTPUT_FILE, "a") as file:
            # Basic information
            file.write(f"=== ANALYSIS FOR {result['symbol']} ===\n")
            
            # Add company info if available
            if result.get('company_info'):
                company = result['company_info']
                file.write(f"Company: {company.get('Name', 'N/A')}\n")
                file.write(f"Industry: {company.get('Industry', 'N/A')}\n")
                file.write(f"Sector: {company.get('Sector', 'N/A')}\n")
            
            # Price and changes
            if result.get('quote_data'):
                quote = result['quote_data']
                file.write(f"Current Price: ${quote.get('price', 0):.2f}\n")
                file.write(f"Change: {quote.get('change', 0):.2f} ({quote.get('change_percent', '0%')})\n")
            else:
                file.write(f"Current Price: ${result['price']:.2f}\n")
            
            file.write(f"Sigma Score: {result['sigma']:.5f}\n")
            file.write(f"Recommendation: {result['recommendation']}\n\n")
            
            # Add price predictions if available
            if 'predictions' in result:
                predictions = result['predictions']
                file.write("--- PRICE PREDICTIONS ---\n")
                file.write(f"30-Day Target: ${predictions['price_target_30d']:.2f} ({predictions['expected_return_30d']:.2f}%)\n")
                file.write(f"60-Day Target: ${predictions['price_target_60d']:.2f} ({predictions['expected_return_60d']:.2f}%)\n")
                file.write(f"Expected Annual Return: {predictions['expected_annual_return']*100:.2f}%\n")
                file.write(f"Adjusted Volatility: {predictions['adjusted_volatility']*100:.2f}%\n")
                
                # Add path to prediction plot if available
                if 'plot_path' in result:
                    file.write(f"Prediction Plot: {result['plot_path']}\n")
                
                file.write("\n")
            
            # Detailed analysis
            analysis = result['analysis']
            
            file.write("--- COMPONENT SCORES ---\n")
            file.write(f"Momentum Score: {analysis.get('momentum_score', 0):.3f}\n")
            file.write(f"Reversion Score: {analysis.get('reversion_score', 0):.3f}\n")
            file.write(f"Balance Factor: {analysis.get('balance_factor', 0):.3f}\n")
            
            file.write("\n--- TECHNICAL INDICATORS ---\n")
            file.write(f"RSI: {analysis.get('rsi', 0):.2f}\n")
            file.write(f"MACD: {analysis.get('macd', 0):.5f}\n")
            file.write(f"SMA Trend: {analysis.get('sma_trend', 0):.5f}\n")
            file.write(f"Distance from SMA200: {analysis.get('dist_from_sma200', 0):.3f}\n")
            file.write(f"Volatility: {analysis.get('volatility', 0):.5f}\n")
            
            file.write("\n--- MARKET REGIME ---\n")
            file.write(f"Hurst Exponent: {analysis.get('hurst_exponent', 0):.3f} ({analysis.get('hurst_regime', 'Unknown')})\n")
            file.write(f"Mean Reversion Half-Life: {analysis.get('mean_reversion_half_life', 0):.1f} days ({analysis.get('mean_reversion_speed', 'Unknown')})\n")
            file.write(f"Mean Reversion Beta: {analysis.get('mean_reversion_beta', 0):.3f}\n")
            file.write(f"Volatility Regime: {analysis.get('volatility_regime', 'Unknown')}\n")
            file.write(f"Volatility Term Structure: {analysis.get('vol_term_structure', 0):.3f}\n")
            file.write(f"Volatility Persistence: {analysis.get('vol_persistence', 0):.3f}\n")
            file.write(f"Market Regime: {analysis.get('market_regime', 'Unknown')}\n")
            
            file.write("\n--- RISK METRICS ---\n")
            file.write(f"Maximum Drawdown: {analysis.get('max_drawdown', 0):.2%}\n")
            file.write(f"Kelly Criterion: {analysis.get('kelly', 0):.3f}\n")
            file.write(f"Sharpe Ratio: {analysis.get('sharpe', 0):.3f}\n")
            
            file.write("\n--- ADVANCED METRICS ---\n")
            if 'advanced_metrics' in analysis:
                advanced = analysis['advanced_metrics']
                for key, value in advanced.items():
                    if isinstance(value, dict):
                        file.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            file.write(f"  {subkey}: {subvalue}\n")
                    else:
                        file.write(f"{key}: {value}\n")
            else:
                file.write("No advanced metrics available\n")
            
            file.write("\n--- MACHINE LEARNING ---\n")
            file.write(f"LSTM Prediction: {analysis.get('lstm_prediction', 0):.3f}\n")
            file.write(f"DQN Recommendation: {analysis.get('dqn_recommendation', 0):.3f}\n")
            
            # Add more detailed metrics if available
            if 'multifractal' in analysis:
                file.write("\n--- MULTIFRACTAL ANALYSIS ---\n")
                for key, value in analysis['multifractal'].items():
                    if isinstance(value, dict):
                        file.write(f"{key}:\n")
                        for subkey, subvalue in value.items():
                            file.write(f"  {subkey}: {subvalue}\n")
                    else:
                        file.write(f"{key}: {value}\n")
                        
            if 'tail_risk' in analysis:
                file.write("\n--- TAIL RISK ANALYSIS ---\n")
                tail_risk = analysis['tail_risk']
                if isinstance(tail_risk, dict):
                    # Extract key metrics
                    if 'tail_type' in tail_risk:
                        file.write(f"Tail Type: {tail_risk['tail_type']}\n")
                    if 'tail_description' in tail_risk:
                        file.write(f"Description: {tail_risk['tail_description']}\n")
                    if 'expected_shortfall' in tail_risk:
                        es = tail_risk['expected_shortfall']
                        for key, value in es.items():
                            file.write(f"{key}: {value:.2%}\n")
                    
            if 'wavelet' in analysis:
                file.write("\n--- WAVELET ANALYSIS ---\n")
                wavelet = analysis['wavelet']
                if isinstance(wavelet, dict) and 'wavelet_transform' in wavelet:
                    wt = wavelet['wavelet_transform']
                    if 'dominant_period' in wt:
                        file.write(f"Dominant Cycle: {wt['dominant_period']:.2f} days\n")
                    if 'dominant_frequency' in wt:
                        file.write(f"Dominant Frequency: {wt['dominant_frequency']:.6f}\n")
            
            # Add fundamental data if available
            if result.get('company_info'):
                file.write("\n--- FUNDAMENTAL DATA ---\n")
                fund_data = result['company_info']
                metrics = [
                    ('MarketCapitalization', 'Market Cap', ''),
                    ('PERatio', 'P/E Ratio', ''),
                    ('PEGRatio', 'PEG Ratio', ''),
                    ('PriceToBookRatio', 'P/B Ratio', ''),
                    ('EVToEBITDA', 'EV/EBITDA', ''),
                    ('ProfitMargin', 'Profit Margin', '%'),
                    ('OperatingMarginTTM', 'Operating Margin', '%'),
                    ('ReturnOnAssetsTTM', 'ROA', '%'),
                    ('ReturnOnEquityTTM', 'ROE', '%'),
                    ('RevenueTTM', 'Revenue TTM', ''),
                    ('GrossProfitTTM', 'Gross Profit TTM', ''),
                    ('DilutedEPSTTM', 'EPS TTM', ''),
                    ('QuarterlyEarningsGrowthYOY', 'Quarterly Earnings Growth', '%'),
                    ('QuarterlyRevenueGrowthYOY', 'Quarterly Revenue Growth', '%'),
                    ('AnalystTargetPrice', 'Analyst Target', '$'),
                    ('Beta', 'Beta', ''),
                    ('52WeekHigh', '52-Week High', '$'),
                    ('52WeekLow', '52-Week Low', '$'),
                    ('50DayMovingAverage', '50-Day MA', '$'),
                    ('200DayMovingAverage', '200-Day MA', '$'),
                    ('DividendYield', 'Dividend Yield', '%'),
                    ('DividendPerShare', 'Dividend Per Share', '$'),
                    ('PayoutRatio', 'Payout Ratio', '%'),
                ]
                
                for key, label, suffix in metrics:
                    if key in fund_data and fund_data[key]:
                        try:
                            # Format numbers properly
                            if key in ['MarketCapitalization', 'RevenueTTM', 'GrossProfitTTM']:
                                # Convert large numbers to billions/millions
                                value = float(fund_data[key])
                                if value >= 1e9:
                                    formatted = f"${value/1e9:.2f}B"
                                elif value >= 1e6:
                                    formatted = f"${value/1e6:.2f}M"
                                else:
                                    formatted = f"${value:.2f}"
                            elif suffix == '$':
                                formatted = f"${float(fund_data[key]):.2f}"
                            elif suffix == '%':
                                formatted = f"{float(fund_data[key]):.2f}%"
                            else:
                                formatted = f"{fund_data[key]}"
                                
                            file.write(f"{label}: {formatted}\n")
                        except:
                            file.write(f"{label}: {fund_data[key]}\n")
            
            file.write("\n" + "="*50 + "\n\n")
            
            return True
    except Exception as e:
        print(f"[ERROR] Failed to append result: {e}")
        traceback.print_exc()
        return False

def initialize_output_file():
    """Initialize the output file with a header"""
    try:
        # Create directory for the output file if needed
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created directory for output file: {output_dir}")
            
        # Create or append to the output file
        mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
        with open(OUTPUT_FILE, mode) as file:
            if mode == "w":  # Only write header for new files
                file.write("===== STOCK ANALYSIS RESULTS =====\n")
                file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write("="*50 + "\n\n")
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize output file: {e}")
        traceback.print_exc()
        return False

def search_stocks(client, keywords):
    """Search for stocks matching keywords"""
    matches = client.get_symbol_search(keywords)
    
    if not matches:
        print("No matches found.")
        return
    
    print("\nMatching stocks:")
    print(f"{'Symbol':<10} {'Type':<8} {'Region':<8} Name")
    print("-" * 70)
    
    for i, match in enumerate(matches):
        print(f"{match['symbol']:<10} {match['type']:<8} {match['region']:<8} {match['name']}")
    
    return matches

def analyze_stock(symbol, client):
    """
    Analyze a stock and generate recommendations
    
    Parameters:
    -----------
    symbol: str
        Stock symbol to analyze
    client: AlphaVantageClient
        Alpha Vantage API client
    
    Returns:
    --------
    dict
        Analysis result
    """
    try:
        # Fetch stock data
        stock_data = client.get_stock_data(symbol)
        
        if stock_data is None or len(stock_data) < 60:
            print(f"[WARNING] Insufficient data for {symbol}")
            return None
        
        # Get company info and quote data
        company_info = client.get_company_overview(symbol)
        quote_data = client.get_global_quote(symbol)
        
        # Get current price
        current_price = quote_data['price'] if quote_data else stock_data['4. close'].iloc[-1]
        
        # Calculate sigma
        sigma = calculate_sigma(stock_data)
        
        if sigma is None:
            print(f"[WARNING] Failed to calculate sigma for {symbol}")
            return None
        
        # Calculate key metrics from various analyses
        try:
            # Hurst exponent
            hurst_info = calculate_hurst_exponent(stock_data, use_log_returns=True)
            
            # Mean reversion half-life
            half_life_info = calculate_mean_reversion_half_life(stock_data)
            
            # Volatility regimes
            vol_data = analyze_volatility_regimes(stock_data)
            
            # Market regime
            market_regime = detect_market_regime(stock_data)
            
            # Risk-adjusted metrics
            risk_metrics = calculate_risk_adjusted_metrics(stock_data, sigma)
            
            # LSTM and DQN predictions
            lstm_prediction = predict_with_lstm(stock_data)
            dqn_recommendation = get_dqn_recommendation(stock_data)
            
            # Get latest technical indicators
            indicators_df = calculate_technical_indicators(stock_data)
            print(f"[DEBUG] Calculated technical indicators, result is: {type(indicators_df)}")
            if indicators_df is not None and not indicators_df.empty:
                print(f"[DEBUG] Technical indicators columns: {indicators_df.columns.tolist()}")
                print(f"[DEBUG] Technical indicators shape: {indicators_df.shape}")
                print(f"[DEBUG] Technical indicators last row RSI: {indicators_df['RSI'].iloc[-1] if 'RSI' in indicators_df.columns else 'Not found'}")
                print(f"[DEBUG] Technical indicators last row MACD: {indicators_df['MACD'].iloc[-1] if 'MACD' in indicators_df.columns else 'Not found'}")
            else:
                print("[WARNING] Technical indicators calculation failed or returned empty DataFrame")
                
            # Get latest row, with better error handling
            latest = None
            try:
                if indicators_df is not None and not indicators_df.empty:
                    latest = indicators_df.iloc[-1]
                    print(f"[INFO] Successfully extracted latest technical indicators: {latest.name}")
            except Exception as e:
                print(f"[ERROR] Failed to extract latest technical indicators: {e}")
            
            # Calculate momentum and reversion scores
            momentum_score = 0.5
            reversion_score = 0.5
            
            # Initialize technical indicator values that will be added to analysis_details
            rsi_value = 0
            macd_value = 0
            sma_trend_value = 0
            dist_from_sma200_value = 0
            volatility_value = 0
            
            if latest is not None:
                try:
                    # Calculate momentum signals with better error handling
                    rsi = latest['RSI'] if 'RSI' in latest and not pd.isna(latest['RSI']) else 50
                    rsi = float(rsi)  # Ensure it's a float
                    rsi_value = rsi  # Store for analysis_details
                    rsi_signal = (max(0, min(100, rsi)) - 30) / 70
                    print(f"[DEBUG] RSI value: {rsi}, signal: {rsi_signal}")
                    
                    macd = latest['MACD'] if 'MACD' in latest and not pd.isna(latest['MACD']) else 0
                    macd = float(macd)  # Ensure it's a float
                    macd_value = macd  # Store for analysis_details
                    macd_signal = np.tanh(macd * 10)
                    macd_signal = (macd_signal + 1) / 2
                    print(f"[DEBUG] MACD value: {macd}, signal: {macd_signal}")
                    
                    sma20 = latest['SMA20'] if 'SMA20' in latest and not pd.isna(latest['SMA20']) else 1
                    sma50 = latest['SMA50'] if 'SMA50' in latest and not pd.isna(latest['SMA50']) else 1
                    sma_trend = (sma20 / sma50 - 1) if abs(sma50) > 1e-6 else 0
                    sma_trend_value = sma_trend  # Store for analysis_details
                    sma_signal = np.tanh(sma_trend * 10)
                    sma_signal = (sma_signal + 1) / 2
                    print(f"[DEBUG] SMA trend value: {sma_trend}, signal: {sma_signal}")
                    
                    # Get other important indicators for analysis_details
                    volatility_value = latest['volatility'] if 'volatility' in latest and not pd.isna(latest['volatility']) else 0
                    dist_from_sma200_value = latest['dist_from_SMA200'] if 'dist_from_SMA200' in latest and not pd.isna(latest['dist_from_SMA200']) else 0
                    
                except Exception as e:
                    print(f"[ERROR] Failed to process momentum signals: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Calculate momentum score
                momentum_score = np.mean([
                    rsi_signal, 
                    macd_signal, 
                    sma_signal, 
                    lstm_prediction,
                    dqn_recommendation
                ])
                
                # Calculate reversion signals
                dist_from_sma200 = latest['dist_from_SMA200'] if 'dist_from_SMA200' in latest and not np.isnan(latest['dist_from_SMA200']) else 0
                sma200_signal = 1 - min(1, max(0, (dist_from_sma200 + 0.1) / 0.2))
                
                bb_reversal_signal = 0.5
                if 'log_returns_zscore' in latest and not np.isnan(latest['log_returns_zscore']):
                    log_z = latest['log_returns_zscore']
                    bb_reversal_signal = min(1, max(0, (abs(log_z) - 0.5) / 2.5))
                elif 'BB_pctB' in latest and not np.isnan(latest['BB_pctB']):
                    bb_pctb = latest['BB_pctB']
                    bb_reversal_signal = 1 - 2 * abs(bb_pctb - 0.5)
                    bb_reversal_signal = max(0, min(1, bb_reversal_signal + 0.5))
                
                # Calculate reversion score
                reversion_score = np.mean([
                    sma200_signal, 
                    bb_reversal_signal,
                    0.5  # Default value for other components
                ])
            
            # Adjust balance factor based on market regime
            balance_factor = 0.5
            
            # Adjust based on hurst exponent
            if hurst_info and 'hurst' in hurst_info:
                hurst = hurst_info['hurst']
                if hurst > 0.6:
                    balance_factor = 0.7
                elif hurst < 0.4:
                    balance_factor = 0.3
            
            # Adjust based on volatility regime
            if vol_data and 'vol_regime' in vol_data:
                vol_regime = vol_data['vol_regime']
                if vol_regime == "Rising":
                    balance_factor -= 0.1
                elif vol_regime == "Falling":
                    balance_factor += 0.1
            
            # Keep balance factor in valid range
            balance_factor = max(0.2, min(0.8, balance_factor))
            
            # Generate analysis details for recommendation
            analysis_details = {
                "momentum_score": momentum_score,
                "reversion_score": reversion_score,
                "recent_monthly_return": stock_data['4. close'].pct_change(20).iloc[-1] if len(stock_data) > 20 else 0,
                "balance_factor": balance_factor,
                "hurst_exponent": hurst_info.get("hurst", 0.5),
                "hurst_regime": hurst_info.get("regime", "Unknown"),
                "mean_reversion_half_life": half_life_info.get("half_life", 0),
                "mean_reversion_speed": half_life_info.get("mean_reversion_speed", "Unknown"),
                "mean_reversion_beta": half_life_info.get("beta", 0),
                "volatility_regime": vol_data.get("vol_regime", "Unknown"),
                "vol_term_structure": vol_data.get("vol_term_structure", 1.0),
                "vol_persistence": vol_data.get("vol_persistence", 0.8),
                "market_regime": market_regime.get("current_regime", "Unknown"),
                "max_drawdown": risk_metrics.get("max_drawdown", 0),
                # Add the technical indicators we extracted
                "rsi": rsi_value, 
                "macd": macd_value,
                "sma_trend": sma_trend_value,
                "dist_from_sma200": dist_from_sma200_value,
                "volatility": volatility_value,
                "kelly": risk_metrics.get("kelly", 0),
                "sharpe": risk_metrics.get("sharpe", 0),
                "lstm_prediction": lstm_prediction,
                "dqn_recommendation": dqn_recommendation
            }
            
        except Exception as e:
            print(f"[WARNING] Error calculating some metrics: {e}")
            # Provide default values if calculations fail
            analysis_details = {
                "momentum_score": 0.5,
                "reversion_score": 0.5,
                "recent_monthly_return": 0,
                "balance_factor": 0.5,
                "hurst_regime": "Unknown",
                "mean_reversion_speed": "Unknown",
                "mean_reversion_beta": 0,
                "volatility_regime": "Unknown",
                "vol_persistence": 0.8,
                "market_regime": "Unknown",
                "max_drawdown": 0,
                "kelly": 0,
                "sharpe": 0,
                "lstm_prediction": 0,
                "dqn_recommendation": 0.5
            }
            
        # Copy analysis_details for use in future price prediction
        analysis_details_with_sigma = analysis_details.copy()
        analysis_details_with_sigma['sigma'] = sigma
        
        # Get recommendation
        recommendation = get_sigma_recommendation(sigma, analysis_details)
        
        # Generate price predictions based on analysis
        predictions = generate_price_predictions(stock_data, analysis_details_with_sigma)
        
        # Create prediction plot if predictions are available
        plot_path = None
        if predictions:
            plot_path = create_prediction_plot(stock_data, predictions, symbol)
        
        # Create result dictionary
        result = {
            "symbol": symbol,
            "price": current_price,
            "sigma": sigma,
            "recommendation": recommendation,
            "company_info": company_info,
            "quote_data": quote_data,
            "analysis": analysis_details,
            "predictions": predictions,
            "plot_path": plot_path
        }
        
        return result
    except Exception as e:
        print(f"[ERROR] Failed to analyze {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the stock analysis"""
    print("\n===== ALPHA VANTAGE STOCK ANALYZER =====")
    print("Using enhanced log returns mean reversion model with price prediction")
    print("="*50 + "\n")
    
    try:
        # Initialize output file
        if not initialize_output_file():
            print("[ERROR] Failed to initialize output file. Check permissions and path.")
            return
        
        # Create Alpha Vantage client
        client = AlphaVantageClient(ALPHA_VANTAGE_API_KEY)
        
        # Create prediction plots directory
        if not os.path.exists("prediction_plots"):
            try:
                os.makedirs("prediction_plots", exist_ok=True)
                print("[INFO] Created directory for prediction plots")
            except Exception as e:
                print(f"[WARNING] Failed to create prediction plots directory: {e}")
                traceback.print_exc()
        
        # Check if command line arguments were provided
        import sys
        if len(sys.argv) > 1:
            # Use the first argument as the stock symbol
            symbol = sys.argv[1].strip().upper()
            print(f"[INFO] Using command line argument for symbol: {symbol}")
            
            # Analyze the stock directly
            result = analyze_stock(symbol, client)
            
            if result:
                # Append the result to the output file
                append_stock_result(result)
                print(f"Analysis for {symbol} completed and saved to {OUTPUT_FILE}")
                
                # Print prediction summary if available
                if 'predictions' in result:
                    pred = result['predictions']
                    print(f"\nPrice Predictions:")
                    print(f"30-Day Target: ${pred['price_target_30d']:.2f} ({pred['expected_return_30d']:.2f}%)")
                    print(f"60-Day Target: ${pred['price_target_60d']:.2f} ({pred['expected_return_60d']:.2f}%)")
                    
                    # Display path to the prediction plot if available
                    if 'plot_path' in result:
                        print(f"Prediction Plot saved to: {result['plot_path']}")
            else:
                print(f"Analysis for {symbol} failed. See log for details.")
            
            # Exit after completing analysis
            return
                
        # Interactive mode if no command line arguments
        while True:
            print("\nOptions:")
            print("1. Analyze a stock")
            print("2. Search for a stock")
            print("3. Exit")
            
            choice = input("Select an option (1-3): ").strip()
            
            if choice == '1':
                symbol = input("Enter stock symbol to analyze: ").strip().upper()
                
                if not symbol:
                    print("Please enter a valid stock symbol.")
                    continue
                
                # Analyze the stock
                result = analyze_stock(symbol, client)
                
                if result:
                    # Append the result to the output file
                    append_stock_result(result)
                    print(f"Analysis for {symbol} completed and saved to {OUTPUT_FILE}")
                    
                    # Print prediction summary if available
                    if 'predictions' in result:
                        pred = result['predictions']
                        print(f"\nPrice Predictions:")
                        print(f"30-Day Target: ${pred['price_target_30d']:.2f} ({pred['expected_return_30d']:.2f}%)")
                        print(f"60-Day Target: ${pred['price_target_60d']:.2f} ({pred['expected_return_60d']:.2f}%)")
                        
                        # Display path to the prediction plot if available
                        if 'plot_path' in result:
                            print(f"Prediction Plot saved to: {result['plot_path']}")
                else:
                    print(f"Analysis for {symbol} failed. See log for details.")
                    
            elif choice == '2':
                keywords = input("Enter company name or keywords to search: ").strip()
                
                if not keywords:
                    print("Please enter valid search terms.")
                    continue
                
                matches = search_stocks(client, keywords)
                
                if matches:
                    analyze_choice = input("\nWould you like to analyze one of these stocks? (y/n): ").strip().lower()
                    
                    if analyze_choice == 'y':
                        symbol = input("Enter the symbol to analyze: ").strip().upper()
                        if symbol:
                            result = analyze_stock(symbol, client)
                            
                            if result:
                                append_stock_result(result)
                                print(f"Analysis for {symbol} completed and saved to {OUTPUT_FILE}")
                                
                                # Print prediction summary if available
                                if 'predictions' in result:
                                    pred = result['predictions']
                                    print(f"\nPrice Predictions:")
                                    print(f"30-Day Target: ${pred['price_target_30d']:.2f} ({pred['expected_return_30d']:.2f}%)")
                                    print(f"60-Day Target: ${pred['price_target_60d']:.2f} ({pred['expected_return_60d']:.2f}%)")
                                    
                                    # Display path to the prediction plot if available
                                    if 'plot_path' in result:
                                        print(f"Prediction Plot saved to: {result['plot_path']}")
                            else:
                                print(f"Analysis for {symbol} failed. See log for details.")
            
            elif choice == '3':
                print("Exiting program. Thank you!")
                break
                
            else:
                print("Invalid option. Please select 1, 2, or 3.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
else:
    print("Error in Main Function")

