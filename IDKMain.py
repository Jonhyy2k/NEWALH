import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import traceback
import random
from collections import deque
from datetime import datetime, timedelta
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from hmmlearn import hmm
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GRU, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import scipy.stats as stats

warnings.filterwarnings('ignore')

# Enable TensorFlow GPU support and log device placement
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"[INFO] Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("[INFO] GPU memory growth enabled")
else:
    print("[WARNING] No GPU found, using CPU instead")

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "73KWO176IRABCOCJ"

# Output file
OUTPUT_FILE = "STOCK_ANALYSIS_RESULTS.txt"

from alpha_vantage_client import AlphaVantageClient

def calculate_sigma(data):
    """Calculate comprehensive sigma metric using log returns-based mean reversion"""
    try:
        # Set a maximum execution time for the entire function
        max_execution_time = 600  # 10 minutes max
        start_time = time.time()

        # 1. Calculate technical indicators with log returns mean reversion components
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None or len(indicators_df) < 30:
            print("[WARNING] Technical indicators calculation failed or insufficient data")
            return None

        # 2. Calculate Hurst exponent using log returns for more accurate results
        hurst_info = calculate_hurst_exponent(indicators_df, use_log_returns=True)
        print(f"[INFO] Hurst exponent: {hurst_info['hurst']:.3f} - {hurst_info['regime']}")

        # 3. Calculate mean reversion half-life using log returns
        half_life_info = calculate_mean_reversion_half_life(indicators_df)
        print(f"[INFO] Mean reversion half-life: {half_life_info['half_life']:.1f} days - {half_life_info['mean_reversion_speed']} (beta: {half_life_info.get('beta', 0):.3f})")

        # 4. Analyze volatility regimes with log returns
        vol_data = analyze_volatility_regimes(indicators_df)
        print(f"[INFO] Volatility regime: {vol_data['vol_regime']} (Term structure: {vol_data['vol_term_structure']:.2f}, Persistence: {vol_data.get('vol_persistence', 0):.2f})")

        # 5. Detect market regime with log returns
        market_regime = detect_market_regime(indicators_df)
        print(f"[INFO] Market regime: {market_regime['current_regime']} (Duration: {market_regime['regime_duration']} days)")

        # 6. Apply PCA to reduce feature dimensionality
        pca_results = None
        pca_variance = []
        pca_components = None

        # Only skip PCA if very constrained on time
        if time.time() - start_time < max_execution_time * 0.6:  # More generous allocation
            try:
                # Use more historical data for PCA
                lookback_period = min(120, len(indicators_df))  # Doubled from 60
                pca_results, pca_variance = apply_pca(indicators_df.iloc[-lookback_period:])

                if pca_results is not None:
                    # Store pca components for possible use in final sigma calculation
                    pca_components = pca_results.iloc[-1].values
                    print(f"[DEBUG] PCA components for latest datapoint: {pca_components}")
            except Exception as e:
                print(f"[WARNING] PCA calculation failed: {e}, continuing without it")
                pca_variance = []
        else:
            print("[WARNING] Skipping PCA calculation due to significant time constraints")

        # 7. Get LSTM volatility prediction with log returns features
        lstm_prediction = 0
        if time.time() - start_time < max_execution_time * 0.7:
            lstm_prediction = predict_with_lstm(data)
            print(f"[DEBUG] LSTM prediction: {lstm_prediction}")
        else:
            print("[WARNING] Skipping LSTM prediction due to time constraints")

        # 8. Get DQN recommendation with log returns features
        dqn_recommendation = 0.5  # Default neutral
        if time.time() - start_time < max_execution_time * 0.8:
            dqn_recommendation = get_dqn_recommendation(indicators_df)
            print(f"[DEBUG] DQN recommendation: {dqn_recommendation}")
        else:
            print("[WARNING] Skipping DQN recommendation due to time constraints")

        # Get latest technical indicators
        latest = indicators_df.iloc[-1]

        # MOMENTUM INDICATORS
        # Prefer log volatility if available for more statistical robustness
        traditional_volatility = indicators_df['log_volatility'].iloc[-1] if 'log_volatility' in indicators_df.columns else indicators_df['volatility'].iloc[-1] if 'volatility' in indicators_df.columns else 0

        rsi = latest['RSI'] if not np.isnan(latest['RSI']) else 50
        rsi_signal = (max(0, min(100, rsi)) - 30) / 70
        rsi_signal = max(0, min(1, rsi_signal))

        macd = latest['MACD'] if not np.isnan(latest['MACD']) else 0
        macd_signal = np.tanh(macd * 10)

        sma20 = latest['SMA20'] if not np.isnan(latest['SMA20']) else 1
        sma50 = latest['SMA50'] if not np.isnan(latest['SMA50']) else 1
        sma_trend = (sma20 / sma50 - 1) if abs(sma50) > 1e-6 else 0
        sma_signal = np.tanh(sma_trend * 10)

        # Calculate short-term momentum (last 10 days vs previous 10 days)
        try:
            # Prefer log returns for momentum calculation if available
            if 'log_returns' in indicators_df.columns:
                recent_returns = indicators_df['log_returns'].iloc[-10:].mean()
                previous_returns = indicators_df['log_returns'].iloc[-20:-10].mean()
                print("[INFO] Using log returns for momentum calculation")
            else:
                recent_returns = indicators_df['returns'].iloc[-10:].mean()
                previous_returns = indicators_df['returns'].iloc[-20:-10].mean()

            momentum_signal = np.tanh((recent_returns - previous_returns) * 20)  # Scale to approx -1 to 1
            momentum_signal = (momentum_signal + 1) / 2  # Convert to 0-1 scale
        except:
            momentum_signal = 0.5  # Neutral

        # MEAN REVERSION INDICATORS - PREFERRING LOG-BASED METRICS

        # 1. Overbought/Oversold based on distance from SMA200
        dist_from_sma200 = latest['dist_from_SMA200'] if not np.isnan(latest['dist_from_SMA200']) else 0
        # Transform to a 0-1 signal where closer to 0 is more overbought (market reversal potential)
        sma200_signal = 1 - min(1, max(0, (dist_from_sma200 + 0.1) / 0.2))

        # 2. Log returns z-score (preferred) or Bollinger Band %B
        if 'log_returns_zscore' in latest and not np.isnan(latest['log_returns_zscore']):
            # Transform log returns z-score to a mean reversion signal (high absolute z-score = high reversal potential)
            log_z = latest['log_returns_zscore']
            log_z_signal = min(1, max(0, (abs(log_z) - 0.5) / 2.5))  # Scale to 0-1 with 0.5 as neutral point
            print(f"[INFO] Using log returns z-score for mean reversion signal: {log_z:.2f} → {log_z_signal:.2f}")
            bb_reversal_signal = log_z_signal  # Use log_z_signal as the preferred metric
        elif 'BB_pctB' in latest and not np.isnan(latest['BB_pctB']):
            # Fallback to regular BB %B
            bb_pctb = latest['BB_pctB']
            # Transform so that extreme values (near 0 or 1) give higher reversal signals
            bb_reversal_signal = 1 - 2 * abs(bb_pctb - 0.5)
            bb_reversal_signal = max(0, min(1, bb_reversal_signal + 0.5))  # Rescale to 0-1
            print(f"[INFO] Using Bollinger Band %B for mean reversion signal: {bb_pctb:.2f} → {bb_reversal_signal:.2f}")
        else:
            bb_reversal_signal = 0.5  # Neutral if neither is available

        # 3. Log-based expected mean reversion or regular ROC acceleration
        if 'log_expected_reversion_pct' in latest and not np.isnan(latest['log_expected_reversion_pct']):
            # Expected reversion percentage based on log returns
            exp_rev = latest['log_expected_reversion_pct']
            # Transform to a 0-1 scale (higher absolute value = stronger signal)
            accel_signal = min(1, abs(exp_rev) / 10)
            print(f"[INFO] Using log-based expected reversion: {exp_rev:.2f}% → {accel_signal:.2f}")
        elif 'ROC_accel' in latest and not np.isnan(latest['ROC_accel']):
            # Fallback to regular price acceleration
            roc_accel = latest['ROC_accel']
            # Transform to 0-1 signal where negative acceleration gives higher reversal signal
            accel_signal = max(0, min(1, 0.5 - roc_accel * 10))
            print(f"[INFO] Using ROC acceleration: {roc_accel:.4f} → {accel_signal:.2f}")
        else:
            accel_signal = 0.5  # Neutral if neither is available

        # 4. Log-based mean reversion potential or regular z-score
        if 'log_mr_potential' in latest and not np.isnan(latest['log_mr_potential']):
            # Log-based mean reversion potential
            log_mr = latest['log_mr_potential']
            # Higher absolute value = stronger signal, sign indicates direction
            mean_rev_signal = min(1, abs(log_mr) / 2)
            print(f"[INFO] Using log-based mean reversion potential: {log_mr:.2f} → {mean_rev_signal:.2f}")
        elif 'mean_reversion_z' in latest and not np.isnan(latest['mean_reversion_z']):
            # Fallback to regular mean reversion z-score
            mean_rev_z = latest['mean_reversion_z']
            # Transform to 0-1 signal where larger absolute z-score suggests higher reversal potential
            mean_rev_signal = min(1, abs(mean_rev_z) / 2)
            print(f"[INFO] Using regular mean reversion z-score: {mean_rev_z:.2f} → {mean_rev_signal:.2f}")
        else:
            mean_rev_signal = 0.5  # Neutral if neither is available

        # 5. RSI divergence signal
        rsi_div = latest['rsi_divergence'] if 'rsi_divergence' in latest and not np.isnan(
            latest['rsi_divergence']) else 0
        # Transform to a 0-1 signal (1 = strong divergence)
        rsi_div_signal = 1 if rsi_div < 0 else 0

        # 6. Log autocorrelation (direct measure of mean reversion) or returns z-score
        if 'log_autocorr_5' in latest and not np.isnan(latest['log_autocorr_5']):
            # Log return autocorrelation - negative values indicate mean reversion
            log_autocorr = latest['log_autocorr_5']
            # Transform to 0-1 scale where more negative = stronger mean reversion
            overbought_signal = max(0, min(1, 0.5 - log_autocorr))
            print(f"[INFO] Using log returns autocorrelation: {log_autocorr:.2f} → {overbought_signal:.2f}")
        elif 'returns_zscore_20' in latest and not np.isnan(latest['returns_zscore_20']):
            # Fallback to regular returns z-score
            returns_z = latest['returns_zscore_20']
            # High positive z-score suggests overbought conditions
            overbought_signal = max(0, min(1, (returns_z + 1) / 4))
            print(f"[INFO] Using returns z-score: {returns_z:.2f} → {overbought_signal:.2f}")
        else:
            overbought_signal = 0.5  # Neutral if neither is available

        # 7. Log volatility ratio or regular volatility ratio
        if 'log_vol_ratio' in latest and not np.isnan(latest['log_vol_ratio']):
            log_vol_ratio = latest['log_vol_ratio']
            vol_increase_signal = max(0, min(1, (log_vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using log volatility ratio: {log_vol_ratio:.2f} → {vol_increase_signal:.2f}")
        elif 'vol_ratio' in latest and not np.isnan(latest['vol_ratio']):
            vol_ratio = latest['vol_ratio']
            vol_increase_signal = max(0, min(1, (vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using volatility ratio: {vol_ratio:.2f} → {vol_increase_signal:.2f}")
        else:
            vol_increase_signal = 0.5  # Neutral if neither is available

        # 8. Additional indicators if available
        williams_r = (latest['Williams_%R'] + 100) / 100 if 'Williams_%R' in latest and not np.isnan(
            latest['Williams_%R']) else 0.5
        cmf = (latest['CMF'] + 1) / 2 if 'CMF' in latest and not np.isnan(latest['CMF']) else 0.5

        # Component groups for Sigma calculation
        momentum_components = {
            "rsi": rsi_signal,
            "macd": (macd_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "sma_trend": (sma_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "traditional_volatility": min(1, traditional_volatility * 25),
            "momentum": momentum_signal,
            "williams_r": williams_r,
            "cmf": cmf,
            "lstm": lstm_prediction,
            "dqn": dqn_recommendation
        }

        # Mean reversion components (higher value = higher reversal potential)
        reversion_components = {
            "sma200_signal": sma200_signal,
            "bb_reversal": bb_reversal_signal,
            "accel_signal": accel_signal,
            "mean_rev_signal": mean_rev_signal,
            "rsi_div_signal": rsi_div_signal,
            "overbought_signal": overbought_signal,
            "vol_increase_signal": vol_increase_signal
        }

        print(f"[DEBUG] Momentum components: {momentum_components}")
        print(f"[DEBUG] Mean reversion components: {reversion_components}")

        # Calculate momentum score (bullish when high)
        if lstm_prediction > 0 and dqn_recommendation != 0.5:
            # Full momentum score with all advanced components
            momentum_score = (
                    0.15 * momentum_components["traditional_volatility"] +
                    0.10 * momentum_components["rsi"] +
                    0.10 * momentum_components["macd"] +
                    0.10 * momentum_components["sma_trend"] +
                    0.10 * momentum_components["momentum"] +
                    0.05 * momentum_components["williams_r"] +
                    0.05 * momentum_components["cmf"] +
                    0.15 * momentum_components["lstm"] +
                    0.20 * momentum_components["dqn"]
            )
        else:
            # Simplified momentum score without advanced models
            momentum_score = (
                    0.20 * momentum_components["traditional_volatility"] +
                    0.15 * momentum_components["rsi"] +
                    0.15 * momentum_components["macd"] +
                    0.15 * momentum_components["sma_trend"] +
                    0.15 * momentum_components["momentum"] +
                    0.10 * momentum_components["williams_r"] +
                    0.10 * momentum_components["cmf"]
            )

        # Calculate mean reversion score (bearish when high)
        reversion_score = (
                0.20 * reversion_components["sma200_signal"] +
                0.15 * reversion_components["bb_reversal"] +
                0.15 * reversion_components["accel_signal"] +
                0.15 * reversion_components["mean_rev_signal"] +
                0.10 * reversion_components["rsi_div_signal"] +
                0.15 * reversion_components["overbought_signal"] +
                0.10 * reversion_components["vol_increase_signal"]
        )

        # Get recent monthly return using log returns if available
        if 'log_returns' in indicators_df.columns:
            recent_returns = indicators_df['log_returns'].iloc[-20:].sum()  # Sum log returns for approximate monthly return
            recent_returns = np.exp(recent_returns) - 1  # Convert to percentage
            print(f"[INFO] Using accumulated log returns for monthly return: {recent_returns:.2%}")
        else:
            recent_returns = latest['ROC_20'] if 'ROC_20' in latest and not np.isnan(latest['ROC_20']) else 0
            print(f"[INFO] Using ROC_20 for monthly return: {recent_returns:.2%}")

        # Adjust balance factor based on Hurst exponent
        hurst_adjustment = 0
        if hurst_info['hurst'] < 0.4:  # Strong mean reversion
            hurst_adjustment = 0.15  # Significantly more weight to mean reversion
        elif hurst_info['hurst'] < 0.45:  # Mean reversion
            hurst_adjustment = 0.1
        elif hurst_info['hurst'] > 0.65:  # Strong trending
            hurst_adjustment = -0.15  # Significantly more weight to momentum
        elif hurst_info['hurst'] > 0.55:  # Trending
            hurst_adjustment = -0.1

        # Base balance factor (adjusted by Hurst)
        base_balance_factor = 0.5 + hurst_adjustment

        # Add adjustment based on mean reversion half-life and beta
        half_life = half_life_info.get('half_life', 0)
        beta = half_life_info.get('beta', 0)

        mr_speed_adjustment = 0
        # Adjust based on beta (direct measure of mean reversion strength)
        if -1 < beta < -0.5:  # Very strong mean reversion
            mr_speed_adjustment = 0.1  # More weight to mean reversion
        elif -0.5 < beta < -0.2:  # Moderate mean reversion
            mr_speed_adjustment = 0.05
        elif beta > 0.2:  # Momentum behavior
            mr_speed_adjustment = -0.05  # Less weight to mean reversion

        # Also consider half-life (speed of mean reversion)
        if 0 < half_life < 10:  # Very fast mean reversion
            mr_speed_adjustment += 0.05
        elif 10 <= half_life < 30:  # Fast mean reversion
            mr_speed_adjustment += 0.025

        base_balance_factor += mr_speed_adjustment
        print(f"[INFO] Mean reversion adjustment based on beta/half-life: {mr_speed_adjustment:.3f}")

        # For stocks with recent large moves, increase the mean reversion weight
        if recent_returns > 0.15:  # >15% monthly returns
            # Gradually increase mean reversion weight for higher recent returns
            excess_return_factor = min(0.3, (recent_returns - 0.15) * 2)  # Up to 0.3 extra weight
            balance_factor = base_balance_factor + excess_return_factor
            print(f"[INFO] Increasing mean reversion weight by {excess_return_factor:.2f} due to high recent returns ({recent_returns:.1%})")
        elif recent_returns < -0.15:  # <-15% monthly returns (big drop)
            # For big drops, slightly reduce mean reversion weight (they've already reverted)
            balance_factor = max(0.3, base_balance_factor - 0.1)
            print(f"[INFO] Decreasing mean reversion weight due to significant recent decline ({recent_returns:.1%})")
        else:
            balance_factor = base_balance_factor

        # Adjust based on volatility regime
        if vol_data['vol_regime'] == "Rising":
            # In rising volatility, favor mean reversion more
            balance_factor += 0.05
            print("[INFO] Increasing mean reversion weight due to rising volatility regime")
        elif vol_data['vol_regime'] == "Falling":
            # In falling volatility, favor momentum more
            balance_factor -= 0.05
            print("[INFO] Decreasing mean reversion weight due to falling volatility regime")

        # Adjust based on volatility persistence (GARCH-like effect)
        vol_persistence = vol_data.get('vol_persistence', 0.8)
        if vol_persistence > 0.9:  # High volatility persistence
            # In high persistence regimes, increase mean reversion weight
            balance_factor += 0.05
            print(f"[INFO] Increasing mean reversion weight due to high volatility persistence: {vol_persistence:.2f}")
        elif vol_persistence < 0.7:  # Low volatility persistence
            # In low persistence regimes, weight is more neutral
            balance_factor = (balance_factor + 0.5) / 2  # Move closer to 0.5
            print(f"[INFO] Adjusting balance factor toward neutral due to low volatility persistence: {vol_persistence:.2f}")

        # Ensure balance factor is reasonable
        balance_factor = max(0.2, min(0.8, balance_factor))

        # Calculate final sigma with balanced approach
        ensemble_result = create_ensemble_prediction(
            momentum_score,
            reversion_score,
            lstm_prediction,
            dqn_recommendation,
            vol_data,
            market_regime,
            hurst_info,
            half_life_info  # Added half-life info to ensemble
        )

        # Use ensemble score if available, otherwise calculate directly
        if ensemble_result and "ensemble_score" in ensemble_result:
            sigma = ensemble_result["ensemble_score"]
            weights = ensemble_result["weights"]
            print(f"[INFO] Using ensemble model with weights: {weights}")
        else:
            # Calculate directly with balance factor
            sigma = momentum_score * (1 - balance_factor) + (1 - reversion_score) * balance_factor

        # Add small PCA adjustment if available
        if pca_components is not None and len(pca_components) >= 3:
            # Use first few principal components to slightly adjust sigma
            pca_influence = np.tanh(np.sum(pca_components[:3]) / 3) * 0.05
            sigma += pca_influence
            print(f"[DEBUG] PCA adjustment to Sigma: {pca_influence:.3f}")

        # Calculate risk-adjusted metrics with log returns
        risk_metrics = calculate_risk_adjusted_metrics(indicators_df, sigma)

        # Use risk-adjusted sigma
        final_sigma = risk_metrics.get("risk_adjusted_sigma", sigma)

        # Ensure sigma is between 0 and 1
        final_sigma = max(0, min(1, final_sigma))

        print(f"[INFO] Final components: Momentum={momentum_score:.3f}, Reversion={reversion_score:.3f}, Balance={balance_factor:.2f}, Sigma={sigma:.3f}, Final Sigma={final_sigma:.3f}")

        # Analysis details
        analysis_details = {
            "sigma": final_sigma,
            "raw_sigma": sigma,
            "momentum_score": momentum_score,
            "reversion_score": reversion_score,
            "balance_factor": balance_factor,
            "recent_monthly_return": recent_returns,
            "traditional_volatility": traditional_volatility,
            "rsi": rsi,
            "macd": macd,
            "sma_trend": sma_trend,
            "dist_from_sma200": dist_from_sma200,
            "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
            "lstm_prediction": lstm_prediction,
            "dqn_recommendation": dqn_recommendation,
            "hurst_exponent": hurst_info['hurst'],
            "hurst_regime": hurst_info['regime'],
            "mean_reversion_half_life": half_life_info['half_life'],
            "mean_reversion_speed": half_life_info['mean_reversion_speed'],
            "mean_reversion_beta": half_life_info.get('beta', 0),  # Added beta coefficient
            "volatility_regime": vol_data['vol_regime'],
            "vol_term_structure": vol_data['vol_term_structure'],
            "vol_persistence": vol_data.get('vol_persistence', 0.8),  # Added volatility persistence
            "market_regime": market_regime['current_regime'],
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "kelly": risk_metrics.get("kelly", 0),
            "sharpe": risk_metrics.get("sharpe", 0)  # Added Sharpe ratio
        }

        return analysis_details
    except Exception as e:
        print(f"[ERROR] Error calculating balanced Sigma with log returns: {e}")
        traceback.print_exc()
        return None

# Enhanced technical indicators with log returns mean reversion components
def calculate_technical_indicators(data):
    try:
        print(f"[DEBUG] Calculating enhanced technical indicators with log returns on data with shape: {data.shape}")
        df = data.copy()

        # Check if data is sufficient
        if len(df) < 50:
            print("[WARNING] Not enough data for technical indicators calculation")
            return None

        # Calculate regular returns if not already present
        if 'returns' not in df.columns:
            df['returns'] = df['4. close'].pct_change()
            df['returns'] = df['returns'].fillna(0)

        # Calculate log returns if not already present
        if 'log_returns' not in df.columns:
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

        # Calculate Relative Strength Index (RSI)
        delta = df['4. close'].diff()
        delta = delta.fillna(0)

        # Handle division by zero and NaN values in RSI calculation
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Handle zero avg_loss
        rs = np.zeros_like(avg_gain)
        valid_indices = avg_loss != 0
        rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]

        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Default to neutral RSI (50)

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
        else:
            # Generate dummy columns if volume not available
            df['vol_price_ratio'] = 0
            df['vol_price_ratio_z'] = 0

        # 10. Stochastic Oscillator
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
        else:
            df['volume_ma'] = 0
            df['vol_weighted_macd'] = df['MACD']

        # 14. Chaikin Money Flow (CMF)
        if 'volume' in df.columns:
            money_flow_multiplier = ((df['4. close'] - df['low']) - (df['high'] - df['4. close'])) / (
                        df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['CMF'] = money_flow_volume.rolling(20).sum() / df['volume'].rolling(20).sum()
        else:
            df['CMF'] = 0

        # 15. Williams %R
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
        def autocorr(x, lag=5):
            if len(x.dropna()) <= lag:
                return 0
            x = x.dropna()
            return pd.Series(x).autocorr(lag=lag)
        
        df['log_autocorr_5'] = df['log_returns'].rolling(30).apply(
            lambda x: autocorr(x, lag=5),
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
        elif 'returns' in df.columns:
            returns = df['returns'].dropna().values
            print("[INFO] Using regular returns for Hurst calculation")
        else:
            # Fallback to calculating returns from price if neither is available
            price_col = 'close' if 'close' in df.columns else '4. close'
            prices = df[price_col].dropna().values
            returns = np.diff(np.log(prices))
            print("[INFO] Calculated returns from price data for Hurst calculation")

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
        # Ensure we have a DataFrame
        if not isinstance(data, pd.DataFrame):
            print("[WARNING] Input to analyze_volatility_regimes is not a DataFrame")
            if hasattr(data, '__array__'):
                data = pd.DataFrame(data)
            else:
                print("[ERROR] Cannot convert input to DataFrame, returning default")
                return {
                    'vol_term_structure': 0,
                    'vol_persistence': 0.5,
                    'vol_regime': "Stable"
                }
        
        # Handle empty DataFrame
        if data.empty:
            print("[WARNING] Empty DataFrame in analyze_volatility_regimes")
            return {
                'vol_term_structure': 0,
                'vol_persistence': 0.5,
                'vol_regime': "Stable"
            }
            
        # Use log returns if available for improved statistical properties
        if 'log_returns' in data.columns:
            returns = data['log_returns'].iloc[-lookback:]
            print("[INFO] Using log returns for volatility regime analysis")
        elif 'returns' in data.columns:
            returns = data['returns'].iloc[-lookback:]
            print("[INFO] Using regular returns for volatility regime analysis")
        else:
            # Fallback to calculating returns from price if neither is available
            price_col = 'close' if 'close' in data.columns else '4. close'
            if price_col not in data.columns:
                print(f"[WARNING] No price column found in data for volatility regime analysis")
                # Create synthetic returns for a fallback
                returns = pd.Series(np.random.normal(0, 0.01, min(lookback, len(data))),
                                   index=data.index[-min(lookback, len(data)):])
                print("[INFO] Created synthetic returns for volatility regime analysis")
            else:
                prices = data[price_col].iloc[-lookback-1:].values
                log_returns = np.diff(np.log(prices))
                returns = pd.Series(log_returns, index=data.index[-len(log_returns):])
                print("[INFO] Calculated returns from price data for volatility regime analysis")

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
        # Extract features for regime detection
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
            print(f"[ERROR] Error in market regime detection: {e}")
            raise ValueError(f"Market regime detection failed: {e}")
    except Exception as e:
        print(f"[ERROR] Error in market regime detection outer handler: {e}")
        return {
            "current_regime": "Unknown",
            "regime_duration": 0,
            "regime_volatility": 0
        }

# Risk-Adjusted Metrics with log return improvements
def calculate_risk_adjusted_metrics(df, sigma):
    """Calculate risk-adjusted metrics using log returns for more accuracy"""
    try:
        # Use log returns if available for better statistical properties
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
            print("[INFO] Using log returns for risk-adjusted metrics")
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
            print("[INFO] Using regular returns for risk-adjusted metrics")
        else:
            # Fallback to calculating returns from price
            price_col = 'close' if 'close' in df.columns else '4. close'
            prices = df[price_col].dropna().values
            returns = pd.Series(np.diff(np.log(prices)))
            print("[INFO] Calculated returns from price data for risk-adjusted metrics")

        # Calculate Maximum Historical Drawdown
        # For log returns, we need to convert back to cumulative returns
        cum_returns = np.exp(np.cumsum(returns)) if 'log_returns' in df.columns else (1 + returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min()

        # Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        alpha = 0.05  # 95% confidence level
        var_95 = np.percentile(returns, alpha * 100)
        cvar_95 = returns[returns <= var_95].mean() if sum(returns <= var_95) > 0 else var_95

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
            "cvar_95": 0,
            "kelly": 0,
            "sharpe": 0,
            "risk_adjusted_sigma": sigma
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
                    features_df) * 0.3:  # At least 70% of values are not NaN
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
        n_components = min(10, min(scaled_data.shape) - 1)  # Increased from 5
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

# Enhanced data preparation for LSTM prediction with log returns features
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
        return None, None, None

# Enhanced LSTM model for volatility prediction
def build_lstm_model(input_shape, gpu_available=None):
    """
    Build enhanced LSTM model with GPU acceleration support
    
    Parameters:
    -----------
    input_shape: tuple
        Input shape for the model (timesteps, features)
    gpu_available: bool, optional
        Whether GPU acceleration is available. If None, will be auto-detected.
    
    Returns:
    --------
    keras.Model
        Compiled LSTM model
    """
    try:
        # Auto-detect GPU if not specified
        if gpu_available is None:
            try:
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            except Exception as e:
                print(f"[WARNING] Error detecting GPU for LSTM model: {e}")
                gpu_available = False
        
        # Sophisticated deep architecture optimized for GPU when available
        inputs = Input(shape=input_shape)
        
        # Configure LSTM units based on hardware capabilities
        base_units = 320 if gpu_available else 256  # Larger when GPU is available
        
        # Select implementation based on hardware
        implementation = 2 if gpu_available else 1  # CuDNN implementation when GPU available
        
        # First LSTM layer - bidirectional for better pattern recognition
        if gpu_available:
            # Bidirectional LSTM leverages parallel processing on GPU
            x = Bidirectional(LSTM(base_units, return_sequences=True, 
                                implementation=implementation,
                                recurrent_dropout=0))(inputs)
        else:
            # Standard LSTM for CPU
            x = LSTM(base_units, return_sequences=True, 
                    implementation=implementation,
                    recurrent_dropout=0.1)(inputs)
        
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Second LSTM layer with increased units
        x = LSTM(base_units, return_sequences=True, 
                implementation=implementation)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Third LSTM layer
        x = LSTM(base_units//2, return_sequences=True, 
                implementation=implementation)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        # Fourth LSTM layer - no return sequences for final LSTM output
        x = LSTM(base_units//2, return_sequences=False, 
                implementation=implementation)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Dense layers with increased capacity
        x = Dense(160, activation='relu')(x)  # Increased from 128
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Second dense layer
        x = Dense(80, activation='relu')(x)  # Increased from 64
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Final dense layer before output
        x = Dense(40, activation='relu')(x)  # Increased from 32
        x = Dropout(0.1)(x)

        # Output layer
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Adjust learning rate based on hardware
        lr = 0.001 if gpu_available else 0.0005  # Higher learning rate for GPU
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="mse")
        
        print(f"[INFO] Built enhanced LSTM model with {model.count_params():,} parameters")
        print(f"[INFO] Using {'GPU-optimized' if gpu_available else 'CPU-compatible'} architecture")
        
        return model
    except Exception as e:
        print(f"[ERROR] Error building enhanced LSTM model: {e}")
        traceback.print_exc()
        return None

# Enhanced LSTM model training and prediction with extended processing time
def predict_with_lstm(data):
    try:
        # Set a maximum execution time - significantly increased for thorough training
        max_execution_time = 240  # 4 minutes max (increased from 2 minutes)
        start_time = time.time()

        # Check for GPU availability
        try:
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_available:
                print("[INFO] Using GPU acceleration for LSTM training")
                # Optimize TensorFlow performance
                tf.config.optimizer.set_jit(True)  # Enable XLA optimization
            else:
                print("[INFO] Using CPU for LSTM training - no GPU detected")
        except Exception as e:
            # Handle case where TensorFlow GPU check fails
            print(f"[WARNING] Error checking GPU availability: {e}")
            print("[INFO] Continuing with CPU-only mode for LSTM training")
            gpu_available = False

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
        model = build_lstm_model((X.shape[1], X.shape[2]), gpu_available=gpu_available)
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
                K = tf.keras.backend
                K.set_value(model.optimizer.learning_rate, K.get_value(model.optimizer.learning_rate) * 0.3)

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
        self.gamma = 0.98  # Increased from 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.03  # Lower min epsilon for better exploitation
        self.epsilon_decay = 0.97  # Slower decay for better exploration
        
        # Check for GPU availability with error handling
        try:
            self.gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if self.gpu_available:
                print("[INFO] Using GPU acceleration for DQN training")
                # Enable XLA optimization
                tf.config.optimizer.set_jit(True)
            else:
                print("[INFO] Using CPU for DQN training - no GPU detected")
        except Exception as e:
            print(f"[WARNING] Error checking GPU availability for DQN: {e}")
            print("[INFO] Continuing with CPU-only mode for DQN training")
            self.gpu_available = False
        
        # Initialize models
        self.model = self._build_model()
        self.target_model = self._build_model()  # Separate target network for stable learning
        self.target_update_counter = 0
        self.target_update_freq = 5  # Update target more frequently (was 10)
        self.max_training_time = 120  # 2 minutes maximum (doubled from 60s)
        self.batch_history = []  # Track training history

    def _build_model(self):
        # Scale network size based on hardware
        base_units = 512 if self.gpu_available else 256
        
        # Advanced model architecture with deep layers
        model = Sequential([
            # Input layer with batch normalization
            Dense(base_units, activation="relu", input_shape=(self.state_size,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # First hidden layer
            Dense(base_units, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second hidden layer
            Dense(base_units // 2, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            
            # Third hidden layer
            Dense(base_units // 4, activation="relu"),
            Dropout(0.1),
            
            # Output layer
            Dense(self.action_size, activation="linear")
        ])

        # Optimize learning rate based on hardware
        lr = 0.0005 if self.gpu_available else 0.0003
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="huber_loss" if self.gpu_available else "mse")
        
        print(f"[INFO] Built DQN model with {model.count_params():,} parameters")
        return model

    # Update target model (for stable learning)
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("[DEBUG] DQN target model updated")

    def remember(self, state, action, reward, next_state, done):
        # Only add to memory if not full
        if len(self.memory) < self.memory.maxlen:
            self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        try:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            if self.model is None:
                return random.randrange(self.action_size)

            # Get multiple predictions with noise for ensembling
            num_predictions = 3
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
            return random.randrange(self.action_size)

    def replay(self, batch_size):
        if len(self.memory) < batch_size or self.model is None:
            return

        # Add timeout mechanism
        start_time = time.time()

        try:
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
                    epochs=5,  # Increased from 3
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

# Enhanced DQN recommendation with log returns features
def get_dqn_recommendation(data):
    try:
        # More lenient on required data
        if len(data) < 40:
            print("[WARNING] Not enough data for DQN")
            return 0.5  # Neutral score

        # Set timeout for the entire function
        function_start_time = time.time()
        max_function_time = 240  # 4 minutes

        # Prepare state features with more historical context
        lookback = 15  # Further increased from 10 for better historical context

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

        # Regular mean reversion indicators as fallback
        if 'dist_from_SMA200' in data.columns:
            dist_sma200 = np.tanh(data['dist_from_SMA200'].values[-lookback:] * 5)
            features.append(dist_sma200)
        if 'BB_pctB' in data.columns and 'log_bb_pctB' not in data.columns:
            bb_pctb = data['BB_pctB'].values[-lookback:]
            # Transform to deviation from middle (0.5)
            bb_deviation = np.tanh((bb_pctb - 0.5) * 4)
            features.append(bb_deviation)
        if 'ROC_accel' in data.columns:
            roc_accel = np.tanh(data['ROC_accel'].values[-lookback:] * 10)
            features.append(roc_accel)
        if 'mean_reversion_z' in data.columns and 'log_returns_zscore' not in data.columns:
            mean_rev_z = np.tanh(data['mean_reversion_z'].values[-lookback:])
            features.append(mean_rev_z)
        if 'rsi_divergence' in data.columns:
            rsi_div = data['rsi_divergence'].values[-lookback:]
            features.append(rsi_div)
        if 'returns_zscore_20' in data.columns and 'log_returns_zscore' not in data.columns:
            ret_z = np.tanh(data['returns_zscore_20'].values[-lookback:])
            features.append(ret_z)

        # Volatility indicators
        if 'vol_ratio' in data.columns and 'log_vol_ratio' not in data.columns:
            vol_ratio = np.tanh((data['vol_ratio'].values[-lookback:] - 1) * 3)
            features.append(vol_ratio)
        if 'log_vol_ratio' in data.columns:
            log_vol_ratio = np.tanh((data['log_vol_ratio'].values[-lookback:] - 1) * 3)
            features.append(log_vol_ratio)
            print("[INFO] Adding log volatility ratio to DQN features")

        # Additional indicators if available
        if 'Williams_%R' in data.columns:
            williams = (data['Williams_%R'].values[-lookback:] + 100) / 100
            features.append(williams)
        if 'CMF' in data.columns:
            cmf = data['CMF'].values[-lookback:]
            features.append(cmf)
        if 'sma_alignment' in data.columns:
            sma_align = data['sma_alignment'].values[-lookback:] / 2 + 0.5  # Convert -1,0,1 to 0,0.5,1
            features.append(sma_align)

        # Stack all features into the state
        features = [np.nan_to_num(f, nan=0.0) for f in features]  # Handle NaNs
        
        if len(features) > 0:
            state = np.concatenate(features)
            state_size = len(state)
        else:
            # Fallback if no features
            state_size = 5
            state = np.zeros(state_size)
            print("[WARNING] No features available for DQN, using default state")

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

                # Collect features at this time point
                # For brevity, this is simplified - in a full implementation, you would extract
                # all the same features used above at this specific time point

                # Use log returns if available
                if 'log_returns' in data.columns:
                    values = data['log_returns'].values[idx:idx + lookback]
                    current_features.append(np.nan_to_num(values, nan=0.0))
                elif 'returns' in data.columns:
                    values = data['returns'].values[idx:idx + lookback]
                    current_features.append(np.nan_to_num(values, nan=0.0))

                # Add more features dynamically based on what's available in the data
                # This is a simplified implementation compared to the complete feature extraction above
                if 'RSI' in data.columns:
                    values = data['RSI'].values[idx:idx + lookback] / 100
                    current_features.append(np.nan_to_num(values, nan=0.5))
                    
                if 'MACD' in data.columns:
                    values = np.tanh(data['MACD'].values[idx:idx + lookback] / 5)
                    current_features.append(np.nan_to_num(values, nan=0.0))

                # Create current state
                if len(current_features) > 0:
                    current_state = np.concatenate(current_features).reshape(1, -1)
                else:
                    # Fallback if feature creation failed
                    current_state = np.zeros((1, state_size))

                # Simplified next state creation - just use same shape
                next_state = np.zeros_like(current_state)

                # Enhanced reward function based on log returns for more statistical validity
                try:
                    # Base reward on forward log return if available
                    if 'log_returns' in data.columns and next_idx + lookback < len(data):
                        price_return = data['log_returns'].values[next_idx + lookback - 1]
                        print("[INFO] Using log returns for DQN reward calculation")
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
                        reward = abs(price_return) * 0.3  # Small positive reward for being right about direction

                    # Add small penalty for extreme actions to encourage some holding
                    if action != 1:  # Not hold
                        reward -= 0.001  # Small transaction cost/risk penalty

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
                print(f"[WARNING] Error in DQN experience collection sample {i}: {e}")
                continue

        print(f"[INFO] Collected {experiences_collected} experiences in {time.time() - collection_start:.1f}s")

        # Train the agent on collected experiences
        if len(agent.memory) > 0:
            print(f"[INFO] Training DQN agent with {len(agent.memory)} experiences...")
            training_start = time.time()
            
            # Calculate number of training iterations based on memory size
            iterations = min(20, len(agent.memory) // 32)
            batch_size = min(128, len(agent.memory))
            
            for _ in range(iterations):
                if time.time() - function_start_time > max_function_time * 0.75:
                    print("[WARNING] DQN training timeout reached")
                    break
                agent.replay(batch_size)
            
            print(f"[INFO] DQN training completed in {time.time() - training_start:.1f}s")

        # Get recommendation score
        final_state = np.zeros((1, state_size))
        
        # Create state from most recent data
        try:
            current_features = []
            
            # Extract features from the most recent data
            if 'log_returns' in data.columns:
                values = data['log_returns'].values[-lookback:]
                current_features.append(np.nan_to_num(values, nan=0.0))
            elif 'returns' in data.columns:
                values = data['returns'].values[-lookback:]
                current_features.append(np.nan_to_num(values, nan=0.0))
                
            # Add more features - similar to what we did before
            if 'RSI' in data.columns:
                values = data['RSI'].values[-lookback:] / 100
                current_features.append(np.nan_to_num(values, nan=0.5))
                
            if 'MACD' in data.columns:
                values = np.tanh(data['MACD'].values[-lookback:] / 5)
                current_features.append(np.nan_to_num(values, nan=0.0))
                
            # Create final state
            if len(current_features) > 0:
                final_state = np.concatenate(current_features).reshape(1, -1)
            
        except Exception as e:
            print(f"[WARNING] Error creating final state for DQN recommendation: {e}")
        
        # Get action values from the model
        try:
            action_values = agent.model.predict(final_state, verbose=0)[0]
            
            # Normalize to get probabilities
            # Use softmax for numerical stability
            exp_values = np.exp(action_values - np.max(action_values))
            probs = exp_values / np.sum(exp_values)
            
            # Calculate recommendation score (0-1 scale)
            # 0 = Strong Sell, 0.5 = Hold, 1 = Strong Buy
            dqn_score = 0.5 * probs[1] + 1.0 * probs[2]
            
            print(f"[INFO] DQN recommendation score: {dqn_score:.3f}")
            return dqn_score
            
        except Exception as e:
            print(f"[ERROR] Error generating DQN recommendation: {e}")
            return 0.5  # Neutral score if prediction fails
    except Exception as e:
        print(f"[ERROR] Error in DQN recommendation: {e}")
        traceback.print_exc()
        return 0.5  # Neutral score

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

    # Adjust based on mean reversion half-life and beta if available
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

    # Adjust based on volatility persistence if available
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

# Calculate sigma for a stock
# Enhanced sigma recommendation function with rich context
def get_sigma_recommendation(sigma, analysis_details):
    """Enhanced recommendation function with log return and mean reversion context"""
    # Get additional context for our recommendation
    momentum_score = analysis_details.get("momentum_score", 0.5)
    reversion_score = analysis_details.get("reversion_score", 0.5)
    recent_monthly_return = analysis_details.get("recent_monthly_return", 0)
    balance_factor = analysis_details.get("balance_factor", 0.5)
    hurst_regime = analysis_details.get("hurst_regime", "Unknown")
    mean_reversion_speed = analysis_details.get("mean_reversion_speed", "Unknown")
    mean_reversion_beta = analysis_details.get("mean_reversion_beta", 0)  # Added beta coefficient
    volatility_regime = analysis_details.get("volatility_regime", "Unknown")
    vol_persistence = analysis_details.get("vol_persistence", 0.8)  # Added volatility persistence
    market_regime = analysis_details.get("market_regime", "Unknown")
    max_drawdown = analysis_details.get("max_drawdown", 0)
    kelly = analysis_details.get("kelly", 0)
    sharpe = analysis_details.get("sharpe", 0)  # Added Sharpe ratio

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

    # Combine base recommendation with context
    recommendation = f"{base_rec} - {context}"

    return recommendation

# Generate price predictions based on analysis results
def generate_price_predictions(data, analysis_details, forecast_days=60, num_paths=100):
    """
    Generate price predictions based on analysis results with GPU acceleration
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing historical price data
    analysis_details: dict
        Dictionary with analysis details
    forecast_days: int
        Number of days to forecast
    num_paths: int
        Number of Monte Carlo simulation paths
        
    Returns:
    --------
    dict
        Dictionary with prediction results
    """
    try:
        # Get current price
        price_col = 'close' if 'close' in data.columns else '4. close'
        current_price = data[price_col].iloc[-1]
        
        # Extract key factors from analysis - vectorized for efficiency
        analysis_factors = {
            'sigma': analysis_details.get('sigma', 0.5),
            'momentum_score': analysis_details.get("momentum_score", 0.5),
            'reversion_score': analysis_details.get("reversion_score", 0.5),
            'hurst_exponent': analysis_details.get("hurst_exponent", 0.5),
            'market_regime': analysis_details.get("market_regime", "Unknown"),
            'volatility_regime': analysis_details.get("volatility_regime", "Stable")
        }
        
        # Efficient volatility calculation with vectorized operations
        if 'log_volatility' in data.columns:
            hist_volatility = data['log_volatility'].iloc[-30:].mean() * np.sqrt(252)
            print(f"[INFO] Using log volatility for prediction bands: {hist_volatility:.4f}")
        elif 'volatility' in data.columns:
            hist_volatility = data['volatility'].iloc[-30:].mean() * np.sqrt(252)
            print(f"[INFO] Using standard volatility for prediction bands: {hist_volatility:.4f}")
        else:
            # Fast vectorized volatility calculation
            price_array = data[price_col].values
            log_returns = np.zeros_like(price_array)
            valid_indices = np.where(price_array[:-1] > 0)[0]
            log_returns[valid_indices + 1] = np.log(price_array[valid_indices + 1] / price_array[valid_indices])
            hist_volatility = np.std(log_returns[-30:]) * np.sqrt(252)
            print(f"[INFO] Estimated volatility for prediction bands with vectorized calc: {hist_volatility:.4f}")
        
        # Adjust volatility based on volatility regime - use dictionary mapping
        vol_multipliers = {
            "Rising": 1.3,
            "Falling": 0.8,
            "Stable": 1.0
        }
        vol_multiplier = vol_multipliers.get(analysis_factors['volatility_regime'], 1.0)
        adjusted_volatility = hist_volatility * vol_multiplier
        
        # Calculate expected return based on sigma - vectorized calculation
        expected_annual_return = (analysis_factors['sigma'] - 0.5) * 0.5  # -25% to +25% annual return
        
        # Adjust based on market regime - use dictionary mapping
        regime_multipliers = {
            "Bull": 1.2,
            "Bear": 0.8
        }
        # Extract regime type if it contains Bull or Bear
        regime_type = next((key for key in regime_multipliers if key in analysis_factors['market_regime']), None)
        regime_multiplier = regime_multipliers.get(regime_type, 1.0)
        expected_annual_return *= regime_multiplier
        
        # Calculate daily expected return
        daily_return = (1 + expected_annual_return) ** (1/252) - 1
        
        # Create date range for forecast - vectorized
        last_date = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else pd.Timestamp.today()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # GPU acceleration via TensorFlow if available
        try:
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_available:
                # Use TensorFlow for faster Monte Carlo simulation
                print("[INFO] Using GPU acceleration for price prediction simulation")
                
                # Create TensorFlow constants
                tf_current_price = tf.constant(current_price, dtype=tf.float32)
                tf_daily_return = tf.constant(daily_return, dtype=tf.float32)
                tf_daily_vol = tf.constant(adjusted_volatility / np.sqrt(252), dtype=tf.float32)
                tf_hurst = tf.constant(analysis_factors['hurst_exponent'], dtype=tf.float32)
                tf_mean_reversion_strength = tf.constant(1.0 - analysis_factors['hurst_exponent'], dtype=tf.float32)
                
                # Initialize paths tensor
                price_paths = tf.TensorArray(tf.float32, size=num_paths)
                
                # Define the simulation step function
                @tf.function
                def simulate_path(path_idx):
                    # Initialize path
                    path = tf.TensorArray(tf.float32, size=forecast_days)
                    
                    # Calculate first step with random component
                    random_start = tf.random.normal([1], mean=0, stddev=tf_daily_vol)[0]
                    first_price = tf_current_price * (1 + tf_daily_return + random_start)
                    path = path.write(0, first_price)
                    
                    # Generate the rest of the path
                    for j in range(1, forecast_days):
                        # Get previous price
                        prev_price = path.read(j-1)
                        
                        # Calculate random component
                        random_component = tf.random.normal([1], mean=0, stddev=tf_daily_vol)[0]
                        
                        # Calculate mean reversion if needed
                        if j > 5:
                            # Calculate trend price
                            trend_price = tf_current_price * tf.pow(1 + tf_daily_return, tf.cast(j + 1, tf.float32))
                            # Distance from trend
                            distance = prev_price / trend_price - 1
                            # Mean reversion component 
                            mean_reversion = -distance * tf_mean_reversion_strength * 0.1
                        else:
                            mean_reversion = 0.0
                        
                        # Calculate next price
                        next_price = prev_price * (1 + tf_daily_return + random_component + mean_reversion)
                        path = path.write(j, next_price)
                    
                    return path.stack()
                
                # Run simulations in parallel batches
                paths_list = []
                batch_size = 10  # Process 10 paths at a time
                for batch in range(0, num_paths, batch_size):
                    batch_paths = tf.map_fn(simulate_path, tf.range(batch, min(batch + batch_size, num_paths)))
                    paths_list.append(batch_paths)
                
                # Combine batches
                price_paths = tf.concat(paths_list, axis=0)
                
                # Convert to numpy for statistics
                price_paths_np = price_paths.numpy()
                print(f"[INFO] Successfully ran TensorFlow GPU simulation with {num_paths} paths")
            else:
                print("[INFO] No GPU detected, using optimized NumPy for price prediction simulation")
                raise ValueError("Fallback to NumPy implementation")
        except Exception as e:
            print(f"[INFO] TensorFlow GPU acceleration unavailable: {e}")
            gpu_available = False
            
        # Fallback to optimized NumPy implementation if GPU not available or failed
        if not gpu_available:
            print("[INFO] Using optimized NumPy for price prediction simulation")
            
            # Generate all random components at once for efficiency
            daily_vol = adjusted_volatility / np.sqrt(252)
            random_components = np.random.normal(0, daily_vol, (num_paths, forecast_days))
            
            # Initialize price paths array
            price_paths = np.zeros((num_paths, forecast_days))
            
            # Vectorized first day calculation
            price_paths[:, 0] = current_price * (1 + daily_return + random_components[:, 0])
            
            # Mean reversion strength based on Hurst exponent
            mean_reversion_strength = 1.0 - analysis_factors['hurst_exponent']
            
            # Vectorized simulation for remaining days
            for j in range(1, forecast_days):
                # Apply drift to all paths
                drift = daily_return
                
                # Apply random component to all paths
                random_component = random_components[:, j]
                
                # Calculate mean reversion for all paths if past initialization period
                if j > 5:
                    # Calculate current trend price (same for all paths)
                    trend_price = current_price * (1 + daily_return) ** (j + 1)
                    
                    # Calculate distance from trend for all paths
                    distance = price_paths[:, j-1] / trend_price - 1
                    
                    # Calculate mean reversion component for all paths
                    mean_reversion = -distance * mean_reversion_strength * 0.1
                else:
                    mean_reversion = np.zeros(num_paths)
                
                # Generate next price for all paths
                price_paths[:, j] = price_paths[:, j-1] * (1 + drift + random_component + mean_reversion)
            
            # Use NumPy for statistics
            price_paths_np = price_paths
        
        # Calculate statistics on paths - vectorized operations
        mean_path = np.mean(price_paths_np, axis=0)
        lower_bound_95 = np.percentile(price_paths_np, 2.5, axis=0)
        upper_bound_95 = np.percentile(price_paths_np, 97.5, axis=0)
        lower_bound_68 = np.percentile(price_paths_np, 16, axis=0)
        upper_bound_68 = np.percentile(price_paths_np, 84, axis=0)
        
        # Calculate price targets and returns
        price_target_30d = mean_path[min(29, forecast_days-1)]
        price_target_60d = mean_path[min(59, forecast_days-1)]
        
        # Vectorized return calculations
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

# Create prediction plot
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

# Initialize in-memory buffer for stock results
stock_results_buffer = []

def append_stock_result(result):
    """
    Add stock analysis result to memory buffer instead of writing to file immediately
    
    Parameters:
    -----------
    result: dict
        Stock analysis result
    """
    global stock_results_buffer
    
    try:
        # Format the output string in memory
        output_lines = []
        
        # Basic information
        output_lines.append(f"=== ANALYSIS FOR {result['symbol']} ===")
        
        # Add company info if available
        if result.get('company_info'):
            company = result['company_info']
            output_lines.append(f"Company: {company.get('Name', 'N/A')}")
            output_lines.append(f"Industry: {company.get('Industry', 'N/A')}")
            output_lines.append(f"Sector: {company.get('Sector', 'N/A')}")
        
        # Price and changes
        if result.get('quote_data'):
            quote = result['quote_data']
            output_lines.append(f"Current Price: ${quote.get('price', 0):.2f}")
            output_lines.append(f"Change: {quote.get('change', 0):.2f} ({quote.get('change_percent', '0%')})")
        else:
            output_lines.append(f"Current Price: ${result['price']:.2f}")
        
        output_lines.append(f"Sigma Score: {result['sigma']:.5f}")
        output_lines.append(f"Recommendation: {result['recommendation']}")
        output_lines.append("")
        
        # Add price predictions if available
        if 'predictions' in result:
            predictions = result['predictions']
            output_lines.append("--- PRICE PREDICTIONS ---")
            output_lines.append(f"30-Day Target: ${predictions['price_target_30d']:.2f} ({predictions['expected_return_30d']:.2f}%)")
            output_lines.append(f"60-Day Target: ${predictions['price_target_60d']:.2f} ({predictions['expected_return_60d']:.2f}%)")
            output_lines.append(f"Expected Annual Return: {predictions['expected_annual_return']*100:.2f}%")
            output_lines.append(f"Adjusted Volatility: {predictions['adjusted_volatility']*100:.2f}%")
            
            # Add path to prediction plot if available
            if 'plot_path' in result:
                output_lines.append(f"Prediction Plot: {result['plot_path']}")
            
            output_lines.append("")
        
        # Add to buffer for batch writing later
        stock_results_buffer.append('\n'.join(output_lines))
        print(f"[INFO] Added results for {result['symbol']} to output buffer (buffer size: {len(stock_results_buffer)})")
        
        # Write to file if buffer reaches threshold
        if len(stock_results_buffer) >= 10:
            flush_results_buffer()
            
        return True
    except Exception as e:
        print(f"[ERROR] Failed to format stock result: {e}")
        traceback.print_exc()
        return False

def flush_results_buffer():
    """
    Write all buffered results to the output file in a single operation
    """
    global stock_results_buffer
    
    if not stock_results_buffer:
        print("[INFO] No results in buffer to flush")
        return True
    
    try:
        # Combine all results with separators
        combined_output = "\n" + "="*50 + "\n\n".join(stock_results_buffer) + "\n\n"
        
        # Append to file in a single write operation
        with open(OUTPUT_FILE, "a") as file:
            file.write(combined_output)
        
        print(f"[INFO] Successfully flushed {len(stock_results_buffer)} results to {OUTPUT_FILE}")
        
        # Clear the buffer
        stock_results_buffer.clear()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to flush results buffer: {e}")
        traceback.print_exc()
        return False

def initialize_output_file():
    """Initialize the output file with a header"""
    try:
        # Clear the buffer if any previous analysis was running
        global stock_results_buffer
        stock_results_buffer.clear()
        
        # Create directory for the output file if needed
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created directory for output file: {output_dir}")
            
        # Create or append to the output file
        mode = "a" if os.path.exists(OUTPUT_FILE) else "w"
        with open(OUTPUT_FILE, mode) as file:
            if mode == "w":  # Only write header for new files
                file.write("===== OPTIMIZED STOCK ANALYSIS RESULTS =====\n")
                file.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"GPU Acceleration: {'Enabled' if len(tf.config.list_physical_devices('GPU')) > 0 else 'Disabled'}\n")
                file.write("="*50 + "\n\n")
            
        print(f"[INFO] Output file initialized: {OUTPUT_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize output file: {e}")
        traceback.print_exc()
        return False

# Analyze a stock
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
        analysis_details = calculate_sigma(stock_data)
        
        if analysis_details is None:
            print(f"[WARNING] Failed to calculate sigma for {symbol}")
            return None
        
        sigma = analysis_details["sigma"]
        
        # Get recommendation
        recommendation = get_sigma_recommendation(sigma, analysis_details)
        
        # Generate price predictions based on analysis
        predictions = generate_price_predictions(stock_data, analysis_details)
        
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
        traceback.print_exc()
        return None

# Main function
def main():
    """Main function to run the stock analysis"""
    print("\n===== ENHANCED STOCK ANALYZER =====")
    print("Using advanced ML and mean reversion analysis with price prediction")
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
            
            # Flush buffer before exiting
            flush_results_buffer()
            print("[INFO] Results buffer flushed.")
            
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
                
                matches = client.get_symbol_search(keywords)
                
                if matches:
                    print("\nMatching stocks:")
                    print(f"{'Symbol':<10} {'Type':<8} {'Region':<8} Name")
                    print("-" * 70)
                    
                    for match in matches:
                        print(f"{match['symbol']:<10} {match['type']:<8} {match['region']:<8} {match['name']}")
                    
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
                else:
                    print("No matching stocks found.")
            
            elif choice == '3':
                print("Exiting program. Thank you!")
                # Flush any remaining results before exiting
                flush_results_buffer()
                break
                
            else:
                print("Invalid option. Please select 1, 2, or 3.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in the main function: {e}")
        traceback.print_exc()
    finally:
        # Always ensure buffer is flushed at the end
        flush_results_buffer()
        print("[INFO] Final results buffer flushed.")

if __name__ == "__main__":
    main()
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
        analysis_details = calculate_sigma(stock_data)
        
        if analysis_details is None:
            print(f"[WARNING] Failed to calculate sigma for {symbol}")
            return None
        
        sigma = analysis_details["sigma"]
        
        # Get recommendation
        recommendation = get_sigma_recommendation(sigma, analysis_details)
        
        # Generate price predictions based on analysis
        predictions = generate_price_predictions(stock_data, analysis_details)
        
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
        traceback.print_exc()
        return None

# Calculate sigma for a stock
    """Calculate comprehensive sigma metric using log returns-based mean reversion"""
    try:
        # Set a maximum execution time for the entire function
        max_execution_time = 600  # 10 minutes max
        start_time = time.time()

        # 1. Calculate technical indicators with log returns mean reversion components
        indicators_df = calculate_technical_indicators(data)
        if indicators_df is None or len(indicators_df) < 30:
            print("[WARNING] Technical indicators calculation failed or insufficient data")
            return None

        # 2. Calculate Hurst exponent using log returns for more accurate results
        hurst_info = calculate_hurst_exponent(indicators_df, use_log_returns=True)
        print(f"[INFO] Hurst exponent: {hurst_info['hurst']:.3f} - {hurst_info['regime']}")

        # 3. Calculate mean reversion half-life using log returns
        half_life_info = calculate_mean_reversion_half_life(indicators_df)
        print(f"[INFO] Mean reversion half-life: {half_life_info['half_life']:.1f} days - {half_life_info['mean_reversion_speed']} (beta: {half_life_info.get('beta', 0):.3f})")

        # 4. Analyze volatility regimes with log returns
        vol_data = analyze_volatility_regimes(indicators_df)
        print(f"[INFO] Volatility regime: {vol_data['vol_regime']} (Term structure: {vol_data['vol_term_structure']:.2f}, Persistence: {vol_data.get('vol_persistence', 0):.2f})")

        # 5. Detect market regime with log returns
        market_regime = detect_market_regime(indicators_df)
        print(f"[INFO] Market regime: {market_regime['current_regime']} (Duration: {market_regime['regime_duration']} days)")

        # 6. Apply PCA to reduce feature dimensionality
        pca_results = None
        pca_variance = []
        pca_components = None

        # Only skip PCA if very constrained on time
        if time.time() - start_time < max_execution_time * 0.6:  # More generous allocation
            try:
                # Use more historical data for PCA
                lookback_period = min(120, len(indicators_df))  # Doubled from 60
                pca_results, pca_variance = apply_pca(indicators_df.iloc[-lookback_period:])

                if pca_results is not None:
                    # Store pca components for possible use in final sigma calculation
                    pca_components = pca_results.iloc[-1].values
                    print(f"[DEBUG] PCA components for latest datapoint: {pca_components}")
            except Exception as e:
                print(f"[WARNING] PCA calculation failed: {e}, continuing without it")
                pca_variance = []
        else:
            print("[WARNING] Skipping PCA calculation due to significant time constraints")

        # 7. Get LSTM volatility prediction with log returns features
        lstm_prediction = 0
        if time.time() - start_time < max_execution_time * 0.7:
            lstm_prediction = predict_with_lstm(data)
            print(f"[DEBUG] LSTM prediction: {lstm_prediction}")
        else:
            print("[WARNING] Skipping LSTM prediction due to time constraints")

        # 8. Get DQN recommendation with log returns features
        dqn_recommendation = 0.5  # Default neutral
        if time.time() - start_time < max_execution_time * 0.8:
            dqn_recommendation = get_dqn_recommendation(indicators_df)
            print(f"[DEBUG] DQN recommendation: {dqn_recommendation}")
        else:
            print("[WARNING] Skipping DQN recommendation due to time constraints")

        # Get latest technical indicators
        latest = indicators_df.iloc[-1]

        # MOMENTUM INDICATORS
        # Prefer log volatility if available for more statistical robustness
        traditional_volatility = indicators_df['log_volatility'].iloc[-1] if 'log_volatility' in indicators_df.columns else indicators_df['volatility'].iloc[-1] if 'volatility' in indicators_df.columns else 0

        rsi = latest['RSI'] if not np.isnan(latest['RSI']) else 50
        rsi_signal = (max(0, min(100, rsi)) - 30) / 70
        rsi_signal = max(0, min(1, rsi_signal))

        macd = latest['MACD'] if not np.isnan(latest['MACD']) else 0
        macd_signal = np.tanh(macd * 10)

        sma20 = latest['SMA20'] if not np.isnan(latest['SMA20']) else 1
        sma50 = latest['SMA50'] if not np.isnan(latest['SMA50']) else 1
        sma_trend = (sma20 / sma50 - 1) if abs(sma50) > 1e-6 else 0
        sma_signal = np.tanh(sma_trend * 10)

        # Calculate short-term momentum (last 10 days vs previous 10 days)
        try:
            # Prefer log returns for momentum calculation if available
            if 'log_returns' in indicators_df.columns:
                recent_returns = indicators_df['log_returns'].iloc[-10:].mean()
                previous_returns = indicators_df['log_returns'].iloc[-20:-10].mean()
                print("[INFO] Using log returns for momentum calculation")
            else:
                recent_returns = indicators_df['returns'].iloc[-10:].mean()
                previous_returns = indicators_df['returns'].iloc[-20:-10].mean()

            momentum_signal = np.tanh((recent_returns - previous_returns) * 20)  # Scale to approx -1 to 1
            momentum_signal = (momentum_signal + 1) / 2  # Convert to 0-1 scale
        except:
            momentum_signal = 0.5  # Neutral

        # MEAN REVERSION INDICATORS - PREFERRING LOG-BASED METRICS

        # 1. Overbought/Oversold based on distance from SMA200
        dist_from_sma200 = latest['dist_from_SMA200'] if not np.isnan(latest['dist_from_SMA200']) else 0
        # Transform to a 0-1 signal where closer to 0 is more overbought (market reversal potential)
        sma200_signal = 1 - min(1, max(0, (dist_from_sma200 + 0.1) / 0.2))

        # 2. Log returns z-score (preferred) or Bollinger Band %B
        if 'log_returns_zscore' in latest and not np.isnan(latest['log_returns_zscore']):
            # Transform log returns z-score to a mean reversion signal (high absolute z-score = high reversal potential)
            log_z = latest['log_returns_zscore']
            log_z_signal = min(1, max(0, (abs(log_z) - 0.5) / 2.5))  # Scale to 0-1 with 0.5 as neutral point
            print(f"[INFO] Using log returns z-score for mean reversion signal: {log_z:.2f} → {log_z_signal:.2f}")
            bb_reversal_signal = log_z_signal  # Use log_z_signal as the preferred metric
        elif 'BB_pctB' in latest and not np.isnan(latest['BB_pctB']):
            # Fallback to regular BB %B
            bb_pctb = latest['BB_pctB']
            # Transform so that extreme values (near 0 or 1) give higher reversal signals
            bb_reversal_signal = 1 - 2 * abs(bb_pctb - 0.5)
            bb_reversal_signal = max(0, min(1, bb_reversal_signal + 0.5))  # Rescale to 0-1
            print(f"[INFO] Using Bollinger Band %B for mean reversion signal: {bb_pctb:.2f} → {bb_reversal_signal:.2f}")
        else:
            bb_reversal_signal = 0.5  # Neutral if neither is available

        # 3. Log-based expected mean reversion or regular ROC acceleration
        if 'log_expected_reversion_pct' in latest and not np.isnan(latest['log_expected_reversion_pct']):
            # Expected reversion percentage based on log returns
            exp_rev = latest['log_expected_reversion_pct']
            # Transform to a 0-1 scale (higher absolute value = stronger signal)
            accel_signal = min(1, abs(exp_rev) / 10)
            print(f"[INFO] Using log-based expected reversion: {exp_rev:.2f}% → {accel_signal:.2f}")
        elif 'ROC_accel' in latest and not np.isnan(latest['ROC_accel']):
            # Fallback to regular price acceleration
            roc_accel = latest['ROC_accel']
            # Transform to 0-1 signal where negative acceleration gives higher reversal signal
            accel_signal = max(0, min(1, 0.5 - roc_accel * 10))
            print(f"[INFO] Using ROC acceleration: {roc_accel:.4f} → {accel_signal:.2f}")
        else:
            accel_signal = 0.5  # Neutral if neither is available

        # 4. Log-based mean reversion potential or regular z-score
        if 'log_mr_potential' in latest and not np.isnan(latest['log_mr_potential']):
            # Log-based mean reversion potential
            log_mr = latest['log_mr_potential']
            # Higher absolute value = stronger signal, sign indicates direction
            mean_rev_signal = min(1, abs(log_mr) / 2)
            print(f"[INFO] Using log-based mean reversion potential: {log_mr:.2f} → {mean_rev_signal:.2f}")
        elif 'mean_reversion_z' in latest and not np.isnan(latest['mean_reversion_z']):
            # Fallback to regular mean reversion z-score
            mean_rev_z = latest['mean_reversion_z']
            # Transform to 0-1 signal where larger absolute z-score suggests higher reversal potential
            mean_rev_signal = min(1, abs(mean_rev_z) / 2)
            print(f"[INFO] Using regular mean reversion z-score: {mean_rev_z:.2f} → {mean_rev_signal:.2f}")
        else:
            mean_rev_signal = 0.5  # Neutral if neither is available

        # 5. RSI divergence signal
        rsi_div = latest['rsi_divergence'] if 'rsi_divergence' in latest and not np.isnan(
            latest['rsi_divergence']) else 0
        # Transform to a 0-1 signal (1 = strong divergence)
        rsi_div_signal = 1 if rsi_div < 0 else 0

        # 6. Log autocorrelation (direct measure of mean reversion) or returns z-score
        if 'log_autocorr_5' in latest and not np.isnan(latest['log_autocorr_5']):
            # Log return autocorrelation - negative values indicate mean reversion
            log_autocorr = latest['log_autocorr_5']
            # Transform to 0-1 scale where more negative = stronger mean reversion
            overbought_signal = max(0, min(1, 0.5 - log_autocorr))
            print(f"[INFO] Using log returns autocorrelation: {log_autocorr:.2f} → {overbought_signal:.2f}")
        elif 'returns_zscore_20' in latest and not np.isnan(latest['returns_zscore_20']):
            # Fallback to regular returns z-score
            returns_z = latest['returns_zscore_20']
            # High positive z-score suggests overbought conditions
            overbought_signal = max(0, min(1, (returns_z + 1) / 4))
            print(f"[INFO] Using returns z-score: {returns_z:.2f} → {overbought_signal:.2f}")
        else:
            overbought_signal = 0.5  # Neutral if neither is available

        # 7. Log volatility ratio or regular volatility ratio
        if 'log_vol_ratio' in latest and not np.isnan(latest['log_vol_ratio']):
            log_vol_ratio = latest['log_vol_ratio']
            vol_increase_signal = max(0, min(1, (log_vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using log volatility ratio: {log_vol_ratio:.2f} → {vol_increase_signal:.2f}")
        elif 'vol_ratio' in latest and not np.isnan(latest['vol_ratio']):
            vol_ratio = latest['vol_ratio']
            vol_increase_signal = max(0, min(1, (vol_ratio - 0.8) / 1.2))
            print(f"[INFO] Using volatility ratio: {vol_ratio:.2f} → {vol_increase_signal:.2f}")
        else:
            vol_increase_signal = 0.5  # Neutral if neither is available

        # 8. Additional indicators if available
        williams_r = (latest['Williams_%R'] + 100) / 100 if 'Williams_%R' in latest and not np.isnan(
            latest['Williams_%R']) else 0.5
        cmf = (latest['CMF'] + 1) / 2 if 'CMF' in latest and not np.isnan(latest['CMF']) else 0.5

        # Component groups for Sigma calculation
        momentum_components = {
            "rsi": rsi_signal,
            "macd": (macd_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "sma_trend": (sma_signal + 1) / 2,  # Convert from -1:1 to 0:1
            "traditional_volatility": min(1, traditional_volatility * 25),
            "momentum": momentum_signal,
            "williams_r": williams_r,
            "cmf": cmf,
            "lstm": lstm_prediction,
            "dqn": dqn_recommendation
        }

        # Mean reversion components (higher value = higher reversal potential)
        reversion_components = {
            "sma200_signal": sma200_signal,
            "bb_reversal": bb_reversal_signal,
            "accel_signal": accel_signal,
            "mean_rev_signal": mean_rev_signal,
            "rsi_div_signal": rsi_div_signal,
            "overbought_signal": overbought_signal,
            "vol_increase_signal": vol_increase_signal
        }

        print(f"[DEBUG] Momentum components: {momentum_components}")
        print(f"[DEBUG] Mean reversion components: {reversion_components}")

        # Calculate momentum score (bullish when high)
        if lstm_prediction > 0 and dqn_recommendation != 0.5:
            # Full momentum score with all advanced components
            momentum_score = (
                    0.15 * momentum_components["traditional_volatility"] +
                    0.10 * momentum_components["rsi"] +
                    0.10 * momentum_components["macd"] +
                    0.10 * momentum_components["sma_trend"] +
                    0.10 * momentum_components["momentum"] +
                    0.05 * momentum_components["williams_r"] +
                    0.05 * momentum_components["cmf"] +
                    0.15 * momentum_components["lstm"] +
                    0.20 * momentum_components["dqn"]
            )
        else:
            # Simplified momentum score without advanced models
            momentum_score = (
                    0.20 * momentum_components["traditional_volatility"] +
                    0.15 * momentum_components["rsi"] +
                    0.15 * momentum_components["macd"] +
                    0.15 * momentum_components["sma_trend"] +
                    0.15 * momentum_components["momentum"] +
                    0.10 * momentum_components["williams_r"] +
                    0.10 * momentum_components["cmf"]
            )

        # Calculate mean reversion score (bearish when high)
        reversion_score = (
                0.20 * reversion_components["sma200_signal"] +
                0.15 * reversion_components["bb_reversal"] +
                0.15 * reversion_components["accel_signal"] +
                0.15 * reversion_components["mean_rev_signal"] +
                0.10 * reversion_components["rsi_div_signal"] +
                0.15 * reversion_components["overbought_signal"] +
                0.10 * reversion_components["vol_increase_signal"]
        )

        # Get recent monthly return using log returns if available
        if 'log_returns' in indicators_df.columns:
            recent_returns = indicators_df['log_returns'].iloc[-20:].sum()  # Sum log returns for approximate monthly return
            recent_returns = np.exp(recent_returns) - 1  # Convert to percentage
            print(f"[INFO] Using accumulated log returns for monthly return: {recent_returns:.2%}")
        else:
            recent_returns = latest['ROC_20'] if 'ROC_20' in latest and not np.isnan(latest['ROC_20']) else 0
            print(f"[INFO] Using ROC_20 for monthly return: {recent_returns:.2%}")

        # Adjust balance factor based on Hurst exponent
        hurst_adjustment = 0
        if hurst_info['hurst'] < 0.4:  # Strong mean reversion
            hurst_adjustment = 0.15  # Significantly more weight to mean reversion
        elif hurst_info['hurst'] < 0.45:  # Mean reversion
            hurst_adjustment = 0.1
        elif hurst_info['hurst'] > 0.65:  # Strong trending
            hurst_adjustment = -0.15  # Significantly more weight to momentum
        elif hurst_info['hurst'] > 0.55:  # Trending
            hurst_adjustment = -0.1

        # Base balance factor (adjusted by Hurst)
        base_balance_factor = 0.5 + hurst_adjustment

        # Add adjustment based on mean reversion half-life and beta
        half_life = half_life_info.get('half_life', 0)
        beta = half_life_info.get('beta', 0)

        mr_speed_adjustment = 0
        # Adjust based on beta (direct measure of mean reversion strength)
        if -1 < beta < -0.5:  # Very strong mean reversion
            mr_speed_adjustment = 0.1  # More weight to mean reversion
        elif -0.5 < beta < -0.2:  # Moderate mean reversion
            mr_speed_adjustment = 0.05
        elif beta > 0.2:  # Momentum behavior
            mr_speed_adjustment = -0.05  # Less weight to mean reversion

        # Also consider half-life (speed of mean reversion)
        if 0 < half_life < 10:  # Very fast mean reversion
            mr_speed_adjustment += 0.05
        elif 10 <= half_life < 30:  # Fast mean reversion
            mr_speed_adjustment += 0.025

        base_balance_factor += mr_speed_adjustment
        print(f"[INFO] Mean reversion adjustment based on beta/half-life: {mr_speed_adjustment:.3f}")

        # For stocks with recent large moves, increase the mean reversion weight
        if recent_returns > 0.15:  # >15% monthly returns
            # Gradually increase mean reversion weight for higher recent returns
            excess_return_factor = min(0.3, (recent_returns - 0.15) * 2)  # Up to 0.3 extra weight
            balance_factor = base_balance_factor + excess_return_factor
            print(f"[INFO] Increasing mean reversion weight by {excess_return_factor:.2f} due to high recent returns ({recent_returns:.1%})")
        elif recent_returns < -0.15:  # <-15% monthly returns (big drop)
            # For big drops, slightly reduce mean reversion weight (they've already reverted)
            balance_factor = max(0.3, base_balance_factor - 0.1)
            print(f"[INFO] Decreasing mean reversion weight due to significant recent decline ({recent_returns:.1%})")
        else:
            balance_factor = base_balance_factor

        # Adjust based on volatility regime
        if vol_data['vol_regime'] == "Rising":
            # In rising volatility, favor mean reversion more
            balance_factor += 0.05
            print("[INFO] Increasing mean reversion weight due to rising volatility regime")
        elif vol_data['vol_regime'] == "Falling":
            # In falling volatility, favor momentum more
            balance_factor -= 0.05
            print("[INFO] Decreasing mean reversion weight due to falling volatility regime")

        # Adjust based on volatility persistence (GARCH-like effect)
        vol_persistence = vol_data.get('vol_persistence', 0.8)
        if vol_persistence > 0.9:  # High volatility persistence
            # In high persistence regimes, increase mean reversion weight
            balance_factor += 0.05
            print(f"[INFO] Increasing mean reversion weight due to high volatility persistence: {vol_persistence:.2f}")
        elif vol_persistence < 0.7:  # Low volatility persistence
            # In low persistence regimes, weight is more neutral
            balance_factor = (balance_factor + 0.5) / 2  # Move closer to 0.5
            print(f"[INFO] Adjusting balance factor toward neutral due to low volatility persistence: {vol_persistence:.2f}")

        # Ensure balance factor is reasonable
        balance_factor = max(0.2, min(0.8, balance_factor))

        # Calculate final sigma with balanced approach
        ensemble_result = create_ensemble_prediction(
            momentum_score,
            reversion_score,
            lstm_prediction,
            dqn_recommendation,
            vol_data,
            market_regime,
            hurst_info,
            half_life_info  # Added half-life info to ensemble
        )

        # Use ensemble score if available, otherwise calculate directly
        if ensemble_result and "ensemble_score" in ensemble_result:
            sigma = ensemble_result["ensemble_score"]
            weights = ensemble_result["weights"]
            print(f"[INFO] Using ensemble model with weights: {weights}")
        else:
            # Calculate directly with balance factor
            sigma = momentum_score * (1 - balance_factor) + (1 - reversion_score) * balance_factor

        # Add small PCA adjustment if available
        if pca_components is not None and len(pca_components) >= 3:
            # Use first few principal components to slightly adjust sigma
            pca_influence = np.tanh(np.sum(pca_components[:3]) / 3) * 0.05
            sigma += pca_influence
            print(f"[DEBUG] PCA adjustment to Sigma: {pca_influence:.3f}")

        # Calculate risk-adjusted metrics with log returns
        risk_metrics = calculate_risk_adjusted_metrics(indicators_df, sigma)

        # Use risk-adjusted sigma
        final_sigma = risk_metrics.get("risk_adjusted_sigma", sigma)

        # Ensure sigma is between 0 and 1
        final_sigma = max(0, min(1, final_sigma))

        print(f"[INFO] Final components: Momentum={momentum_score:.3f}, Reversion={reversion_score:.3f}, Balance={balance_factor:.2f}, Sigma={sigma:.3f}, Final Sigma={final_sigma:.3f}")

        # Analysis details
        analysis_details = {
            "sigma": final_sigma,
            "raw_sigma": sigma,
            "momentum_score": momentum_score,
            "reversion_score": reversion_score,
            "balance_factor": balance_factor,
            "recent_monthly_return": recent_returns,
            "traditional_volatility": traditional_volatility,
            "rsi": rsi,
            "macd": macd,
            "sma_trend": sma_trend,
            "dist_from_sma200": dist_from_sma200,
            "last_price": latest['4. close'] if not np.isnan(latest['4. close']) else 0,
            "lstm_prediction": lstm_prediction,
            "dqn_recommendation": dqn_recommendation,
            "hurst_exponent": hurst_info['hurst'],
            "hurst_regime": hurst_info['regime'],
            "mean_reversion_half_life": half_life_info['half_life'],
            "mean_reversion_speed": half_life_info['mean_reversion_speed'],
            "mean_reversion_beta": half_life_info.get('beta', 0),  # Added beta coefficient
            "volatility_regime": vol_data['vol_regime'],
            "vol_term_structure": vol_data['vol_term_structure'],
            "vol_persistence": vol_data.get('vol_persistence', 0.8),  # Added volatility persistence
            "market_regime": market_regime['current_regime'],
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "kelly": risk_metrics.get("kelly", 0),
            "sharpe": risk_metrics.get("sharpe", 0)  # Added Sharpe ratio
        }

        return analysis_details
    except Exception as e:
        print(f"[ERROR] Error calculating balanced Sigma with log returns: {e}")
        traceback.print_exc()
        return None
