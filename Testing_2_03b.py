'''
OPTIMIZED TRADING ALGORITHM
PERFORMANCE IMPROVEMENTS
02 April 2025
///
Key Updates:
1. Improved Memory Management with Chunking and Dtype Optimization
2. Streamlined Feature Engineering with Selective Computation
3. Enhanced Feature Selection with Filtering and Caching
4. Optimized Model Training with Early Stopping and Reduced CV
5. Parallelization Improvements for Multi-core Processing
6. Built-in Profiling for Performance Monitoring
'''

# LIBRARIES
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
import lightgbm as lgbm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import pickle
import time
from functools import partial
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

# Add profiling utilities
import cProfile
import pstats
from io import StringIO

print('Importing Libraries OK')

'''
# Performance Monitoring Decorator
def profile_function(func):
    def wrapper(*args, **kwargs):
        prof = cProfile.Profile()
        prof.enable()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        prof.disable()
        
        # Print execution time
        print(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        
        # Get profiling stats
        s = StringIO()
        ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Print top 10 time-consuming functions
        print(s.getvalue())
        
        return result
    return wrapper

# ------------------------------
# 1. Data Loading and Preparation
# ------------------------------
 
'''
def optimized_load_data(file_path, sample_rate=0.25, chunksize=250000, date_filter=None):
    """
    Loads data with memory optimization techniques and optional date filtering.
    
    Args:
        file_path: Path to the CSV file
        sample_rate: Fraction of data to sample (lower means faster processing)
        chunksize: Number of rows to process at once
        date_filter: Optional tuple of (start_date, end_date) to filter data
    """
    print(f"Loading data from {file_path} with sample rate {sample_rate}...")
    
    # Use optimized dtypes and minimal columns initially
    dtypes = {
        'Open': 'float32',
        'High': 'float32',
        'Low': 'float32',
        'Close': 'float32',
        'Volume': 'float32',
    }
    
    # Performance optimization: Only load what we need
    usecols = list(dtypes.keys())
    if 'Timestamp' not in usecols:
        usecols.append('Timestamp')
    
    # Create a generator for chunks
    reader = pd.read_csv(file_path, dtype=dtypes, usecols=usecols, 
                         iterator=True, chunksize=chunksize)
    
    chunk_list = []
    total_rows = 0
    for i, chunk in enumerate(reader):
        if i % 5 == 0:
            print(f"Processing chunk {i}...")
        
        # Process timestamp before potential filtering
        if 'Timestamp' in chunk.columns:
            chunk['Timestamp'] = pd.to_datetime(chunk['Timestamp'])
            chunk.set_index('Timestamp', inplace=True)
        
        # Apply date filter if provided
        if date_filter:
            start_date, end_date = date_filter
            chunk = chunk[(chunk.index >= start_date) & (chunk.index <= end_date)]
        
        # Sample data to reduce processing time
        if len(chunk) > 0 and sample_rate < 1.0:
            chunk = chunk.sample(frac=sample_rate, random_state=42 + i)
        
        if len(chunk) > 0:
            # Add only essential derivatives here
            chunk['log_return'] = np.log(chunk['Close'] / chunk['Close'].shift(1)).astype('float32')
            chunk_list.append(chunk)
            total_rows += len(chunk)
        
        # Aggressive garbage collection
        if i % 3 == 0:
            gc.collect()
    
    if not chunk_list:
        print("No data loaded! Check file path and filters.")
        return None
    
    print(f"Concatenating {len(chunk_list)} chunks with total {total_rows} rows...")
    df = pd.concat(chunk_list)
    del chunk_list
    gc.collect()
    
    # Essential time features (minimal set)
    df['hour'] = df.index.hour.astype('int8')
    df['day_of_week'] = df.index.dayofweek.astype('int8')
    
    # Only calculate time diff for detecting gaps if needed
    df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
    df['gap'] = df['time_diff'].gt(1).astype('int8')
    
    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame loaded: {len(df)} rows, {memory_usage:.2f} MB")
    
    return df

# ------------------------------
# 2. Streamlined Feature Engineering
# ------------------------------
def streamlined_feature_engineering(df, window_sizes=[5, 15, 30], 
                                    feature_set='full'):
    """
    Engineers features with performance optimization.
    
    Args:
        df: DataFrame with OHLCV data
        window_sizes: List of window sizes for indicators
        feature_set: 'minimal', 'essential', or 'full'
    """
    print(f"Engineering {feature_set} features...")
    
    # Clone dataframe for features
    df_feat = df.copy()
    
    # STRATEGY: Only compute features we're likely to use
    # based on the specified feature_set
    
    # Base features that are always computed
    essential_features = []
    
    # For minimal features, focus on price and volume
    for window in window_sizes:
        essential_features.append(f'sma_{window}')
        essential_features.append(f'ema_{window}')
    
    # Add features based on feature set
    if feature_set in ['essential', 'full']:
        for window in window_sizes:
            essential_features.append(f'rsi_{window}')
        
        # Add MACD - typically high value
        df_feat['macd'] = ta.trend.macd(df_feat['Close']).astype('float32')
        df_feat['macd_signal'] = ta.trend.macd_signal(df_feat['Close']).astype('float32')
        df_feat['macd_diff'] = ta.trend.macd_diff(df_feat['Close']).astype('float32')
        
        # Lagged returns - good predictors
        for lag in [1, 5]:
            df_feat[f'log_return_lag_{lag}'] = df_feat['log_return'].shift(lag).astype('float32')
        
        # Volatility - essential
        df_feat['volatility_5'] = df_feat['log_return'].rolling(window=5).std().astype('float32')
        
        # Volume features
        df_feat['volume_ma_ratio_5'] = (df_feat['Volume'] / 
                                        df_feat['Volume'].rolling(5).mean()).astype('float32')
    
    # Full feature set - more comprehensive (but more expensive)
    if feature_set == 'full':
        # Additional momentum features
        for lag in [10]:
            df_feat[f'log_return_lag_{lag}'] = df_feat['log_return'].shift(lag).astype('float32')
        
        # More volatility features
        df_feat['volatility_15'] = df_feat['log_return'].rolling(window=15).std().astype('float32')
        
        # Additional volume features
        df_feat['volume_change'] = df_feat['Volume'].pct_change().astype('float32')
        df_feat['volume_mean_ratio'] = (df_feat['Volume'] / df_feat['Volume'].rolling(window=30).mean()).astype('float32')
        
        
        # Support/resistance - computationally expensive but valuable
        for window in [15, 30]:
            df_feat[f'support_{window}'] = df_feat['Low'].rolling(window=window).min().astype('float32')
            df_feat[f'resistance_{window}'] = df_feat['High'].rolling(window=window).max().astype('float32')
            df_feat[f'price_to_support_{window}'] = ((df_feat['Close'] - df_feat[f'support_{window}']) / df_feat[f'support_{window}']).astype('float32')
            df_feat[f'price_to_resistance_{window}'] = ((df_feat['Close'] - df_feat[f'resistance_{window}']) / df_feat[f'resistance_{window}']).astype('float32')
            
            # Normalize distances
            df_feat[f'close_to_support_{window}'] = ((df_feat['Close'] - df_feat[f'support_{window}']) / 
                                                    df_feat[f'support_{window}']).astype('float32')
            df_feat[f'close_to_resistance_{window}'] = ((df_feat[f'resistance_{window}'] - df_feat['Close']) / 
                                                       df_feat[f'resistance_{window}']).astype('float32')
    
    
    
    # Parallel computation for moving averages and RSI
    # Optimize by computing only the indicators in essential_features
    tasks = []
    for window in window_sizes:
        if f'sma_{window}' in essential_features:
            tasks.append((df_feat, window, f'sma_{window}', ta.trend.sma_indicator))
        if f'ema_{window}' in essential_features:
            tasks.append((df_feat, window, f'ema_{window}', ta.trend.ema_indicator))
        if f'rsi_{window}' in essential_features:
            tasks.append((df_feat, window, f'rsi_{window}', ta.momentum.rsi))
      
        
    # Use joblib for parallel processing with optimized settings
    n_jobs = min(os.cpu_count(), 4)  # Prevent overwhelming the system
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(lambda d, w, n, f: (n, f(d['Close'], window=w).astype('float32')))(*task) 
        for task in tasks
    )
    
    # Assign results to dataframe
    for col_name, result in results:
        df_feat[col_name] = result
    
    # Clean up
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    
    return df_feat
'''

#Testing parameter tuning
def streamlined_feature_engineering(df, window_sizes=[5, 15, 30]):
    """
    Engineers all available features with performance optimization.
    
    Args:
        df: DataFrame with OHLCV data
        window_sizes: List of window sizes for indicators
    """
    print("Engineering all available features...")
    
    # Clone dataframe for features
    df_feat = df.copy()
    
    # Define all features to compute
    all_features = []
    
    # Add all window-based indicators
    for window in window_sizes:
        all_features.append(f'sma_{window}')
        all_features.append(f'ema_{window}')
        all_features.append(f'rsi_{window}')
    
    # Add MACD
    df_feat['macd'] = ta.trend.macd(df_feat['Close']).astype('float32')
    df_feat['macd_signal'] = ta.trend.macd_signal(df_feat['Close']).astype('float32')
    df_feat['macd_diff'] = ta.trend.macd_diff(df_feat['Close']).astype('float32')
    
    # Add all lagged returns
    for lag in [1, 5, 10]:
        df_feat[f'log_return_lag_{lag}'] = df_feat['log_return'].shift(lag).astype('float32')
    
    # Add all volatility metrics
    df_feat['volatility_5'] = df_feat['log_return'].rolling(window=5).std().astype('float32')
    df_feat['volatility_15'] = df_feat['log_return'].rolling(window=15).std().astype('float32')
    
    # Add all volume features
    df_feat['volume_ma_ratio_5'] = (df_feat['Volume'] / 
                                    df_feat['Volume'].rolling(5).mean()).astype('float32')
    df_feat['volume_change'] = df_feat['Volume'].pct_change().astype('float32')
    df_feat['volume_mean_ratio'] = (df_feat['Volume'] / df_feat['Volume'].rolling(window=30).mean()).astype('float32')
    
    # Add all support/resistance features
    for window in [15, 30]:
        df_feat[f'support_{window}'] = df_feat['Low'].rolling(window=window).min().astype('float32')
        df_feat[f'resistance_{window}'] = df_feat['High'].rolling(window=window).max().astype('float32')
        df_feat[f'price_to_support_{window}'] = ((df_feat['Close'] - df_feat[f'support_{window}']) / 
                                              df_feat[f'support_{window}']).astype('float32')
        df_feat[f'price_to_resistance_{window}'] = ((df_feat['Close'] - df_feat[f'resistance_{window}']) / 
                                                 df_feat[f'resistance_{window}']).astype('float32')
        
        # Normalize distances
        df_feat[f'close_to_support_{window}'] = ((df_feat['Close'] - df_feat[f'support_{window}']) / 
                                              df_feat[f'support_{window}']).astype('float32')
        df_feat[f'close_to_resistance_{window}'] = ((df_feat[f'resistance_{window}'] - df_feat['Close']) / 
                                                 df_feat[f'resistance_{window}']).astype('float32')
    
    # Parallel computation for moving averages and RSI
    tasks = []
    for window in window_sizes:
        tasks.append((df_feat, window, f'sma_{window}', ta.trend.sma_indicator))
        tasks.append((df_feat, window, f'ema_{window}', ta.trend.ema_indicator))
        tasks.append((df_feat, window, f'rsi_{window}', ta.momentum.rsi))
    
    # Use joblib for parallel processing with optimized settings
    n_jobs = min(os.cpu_count(), 4)  # Prevent overwhelming the system
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(lambda d, w, n, f: (n, f(d['Close'], window=w).astype('float32')))(*task) 
        for task in tasks
    )
    
    # Assign results to dataframe
    for col_name, result in results:
        df_feat[col_name] = result
    
    # Clean up
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    
    return df_feat

''' 
# ------------------------------
# 3. Efficient Feature Selection
# ------------------------------
 
def efficient_feature_selection(df, target_col, mi_threshold=0.01, max_features=20, 
                                use_cache=True, cache_file='feature_importance_cache.pkl'):
    """
    Selects features efficiently with optional caching of results.
    """
    print("Selecting features...")
    
    # Define potential features, excluding the target and non-feature columns
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'time_diff', 
                   'gap', 'log_return', 'Timestamp', target_col]
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Try to load cached feature importance if available
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data.get('features') == features:
                    print("Using cached feature importance")
                    mi_dict = cached_data.get('mi_dict')
                    # Continue with feature selection using cached values
                else:
                    mi_dict = None
        except Exception as e:
            print(f"Error loading cache: {e}")
            mi_dict = None
    else:
        mi_dict = None
    
    # If no cache, calculate mutual information
    if mi_dict is None:
        print("Calculating mutual information...")
        
        # Use a sample to speed up calculation if dataset is large
        if len(df) > 10000:
            sample_size = min(10000, int(len(df) * 0.3))
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df
        
        X = df_sample[features].fillna(0)
        y = df_sample[target_col].fillna(0)
        
        # Calculate mutual information with error handling
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_dict = dict(zip(features, mi_scores))
            
            # Cache results
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump({'features': features, 'mi_dict': mi_dict}, f)
        except Exception as e:
            print(f"Error calculating mutual information: {e}")
            print("Falling back to default feature selection")
            # Return default features if MI calculation fails
            return sorted(features)[:max_features]
    
    # Select features based on MI threshold and max count
    selected_features = sorted([(f, score) for f, score in mi_dict.items() 
                               if score > mi_threshold], 
                              key=lambda x: x[1], reverse=True)
    
    # Limit to max_features if specified
    if max_features > 0:
        selected_features = selected_features[:max_features]
    
    # Extract feature names only
    selected_feature_names = [f[0] for f in selected_features]
    
    print(f"Selected {len(selected_feature_names)} features")
    
    # Return top features
    return selected_feature_names

# ------------------------------
# 4. Optimized Model Training
# ------------------------------
class PurgedTimeSeriesKFold:
    def __init__(self, n_splits=3, purge_gap=30):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Explicitly convert to integer arrays
            train_indices = np.concatenate([
                indices[:max(0, test_start - self.purge_gap)],
                indices[test_end + self.purge_gap:] if test_end + self.purge_gap < n_samples else np.array([], dtype=int)
            ]).astype(int)
            
            # Ensure both are integer type
            yield train_indices.astype(int), test_indices.astype(int)

 
def train_optimized_model(df, features, target_col, horizon, 
                           cv_splits=3, purge_gap=30, use_ensemble=False):
    """
    Trains models with optimized performance.
    
    Args:
        df: DataFrame with features
        features: List of feature columns
        target_col: Target column name
        horizon: Prediction horizon
        cv_splits: Number of CV splits
        purge_gap: Gap for purging in time series CV
        use_ensemble: Whether to train an ensemble (slower but potentially better)
    """
    print(f"Training optimized model for {target_col} (horizon: {horizon})...")
    
    # Use only necessary data
    df_subset = df[features + [target_col]].copy()
    df_subset = df_subset.dropna()
    
    X = df_subset[features].values
    y = df_subset[target_col].values
    
    # Check for any issues in data
    assert not np.isnan(X).any(), "NaN values found in features"
    assert not np.isnan(y).any(), "NaN values found in target"
    
    # Handle class imbalance
    class_counts = np.bincount(y)
    print(f"Class distribution: {class_counts}")
    
    # Check if we have both classes
    if len(class_counts) < 2:
        print("WARNING: Only one class found in target!")
        if len(class_counts) == 1 and class_counts[0] == len(y):
            print("All samples are class 0")
            # Create a dummy model that always predicts class 0
            class DummyModel:
                def predict(self, X):
                    return np.zeros(len(X))
                def predict_proba(self, X):
                    return np.vstack([np.ones(len(X)), np.zeros(len(X))]).T
            
            return DummyModel(), {'dummy': {'metrics': {'accuracy': 1.0}}}, None
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define the primary model (LightGBM is generally faster)
    lgb_model = lgbm.LGBMClassifier(
        n_estimators=1000,  # Reduced from 500
        learning_rate=0.25,  # Increased for faster convergence
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        importance_type='gain',
        # Remove early_stopping_rounds from here as it needs to be in fit()
        verbose=-1
    )
    
    # Cross-validation with fewer splits
    tscv = PurgedTimeSeriesKFold(n_splits=cv_splits, purge_gap=purge_gap)
    
    # Train and evaluate the model
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    feature_importances = np.zeros(len(features))
    
    # Train across folds
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        print(f"Training fold {i+1}/{cv_splits}...")
        
        # Ensure indices are integer type
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create validation set from training data
        val_size = int(len(X_train) * 0.2)  # Use 20% of training data as validation
        X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]
        
        # Fit with validation data
        lgb_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgbm.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        # Predict
        y_pred = lgb_model.predict(X_test)
        y_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        # Add try-except for AUC as it requires both classes
        try:
            fold_metrics['auc'].append(roc_auc_score(y_test, y_proba))
        except Exception as e:
            print(f"Warning: Could not calculate AUC: {e}")
            fold_metrics['auc'].append(0.5)  # Default to random
        
        # Update feature importances
        if hasattr(lgb_model, 'feature_importances_'):
            feature_importances += lgb_model.feature_importances_ / cv_splits
        
        # Clear memory
        gc.collect()
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
    std_metrics = {f"{metric}_std": np.std(values) for metric, values in fold_metrics.items()}
    
    print("Model training complete!")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f} (±{std_metrics[f'{metric}_std']:.4f})")
    
    # Train final model on all data if metrics are acceptable
    if avg_metrics['auc'] > 0.52:  # Only if better than random
        print("Training final model on all data...")
        # Split data for final training with validation set for early stopping
        train_size = int(len(X_scaled) * 0.8)
        X_train_final, X_val_final = X_scaled[:train_size], X_scaled[train_size:]
        y_train_final, y_val_final = y[:train_size], y[train_size:]
        
        # Create a new model instance for final training
        final_lgb_model = lgbm.LGBMClassifier(
            n_estimators=2000,
            learning_rate=1,
            max_depth=10,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )
        
        # Fit with validation data for early stopping
        final_lgb_model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            eval_metric='auc',
            callbacks=[lgbm.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        lgb_model = final_lgb_model
    
    # Only train ensemble if requested and we have good performance
    ensemble = None
    if use_ensemble and avg_metrics['auc'] > 0.55:
        print("Training ensemble (this may take some time)...")
        
        # Create a simple ensemble with the already trained model
        # No need to create a new one since we already have a final model
        
        print("Ensemble training complete!")
    
    results = {
        'LightGBM': {
            'model': lgb_model,
            'metrics': {**avg_metrics, **std_metrics}
        }
    }
    
    if ensemble:
        results['Ensemble'] = {'model': ensemble}
        return ensemble, results, scaler
    else:
        return lgb_model, results, scaler

# ------------------------------
# 5. Optimized Backtesting
# ------------------------------
 
def optimized_backtest(df, model, features, scaler, threshold=0.6, cost=0.001):
    """
    Performs backtesting with transaction costs and improved performance.
    """
    print("Running optimized backtest...")
    
    # Ensure we only use complete rows
    df_backtest = df[features].dropna()
    
    # Scale features
    X_scaled = scaler.transform(df_backtest.values)
    
    # Generate predictions
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Generate signals with threshold
    signals = (y_proba > threshold).astype(int)
    
    # Get returns data aligned with our predictions
    returns_data = df.loc[df_backtest.index, 'log_return'].shift(-1).dropna()
    
    # Align signals with returns
    signals = signals[:len(returns_data)]
    
    # Calculate strategy returns with transaction costs
    signal_series = pd.Series(signals, index=returns_data.index)
    costs = cost * np.abs(signal_series.diff().fillna(0))
    strategy_returns = returns_data * signal_series - costs
    
    # Calculate performance metrics
    total_return = strategy_returns.sum()
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 24 * 60)
    win_rate = (strategy_returns > 0).mean()
    
    print(f"Backtest Results:")
    print(f"  Total Return: {total_return:.4f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Win Rate: {win_rate:.4f}")
    
    # Return all metrics
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'win_rate': win_rate,
        'strategy_returns': strategy_returns
    }

# ------------------------------
# 6. Main Optimized Pipeline
# ------------------------------
 
def run_optimized_analysis(file_path, output_dir='./model_outputs', 
                           sample_rate=0.25, horizon=1, threshold=0.6,
                           feature_set='full', date_filter=None):
    """
    Executes the optimized pipeline with improved performance.
    
    Args:
        file_path: Path to the data file
        output_dir: Directory to save outputs
        sample_rate: Fraction of data to sample
        horizon: Prediction horizon
        threshold: Threshold for trading signals
        feature_set: 'minimal', 'essential', or 'full'
        date_filter: Optional tuple of (start_date, end_date)
    """
    start_time = time.time()
    print(f"Starting optimized analysis pipeline: {datetime.now()}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Load data
    df = optimized_load_data(file_path, sample_rate=sample_rate, date_filter=date_filter)
    if df is None or len(df) == 0:
        print("Error: No data loaded")
        return None
    
    # 2. Feature engineering - use streamlined approach
    df_featured = streamlined_feature_engineering(df, feature_set=feature_set)
    
    # 3. Create target variable
    print("Creating target variable...")
    df_featured['price_up'] = (df_featured['Close'].shift(-horizon) > df_featured['Close']).astype(int)
    df_featured = df_featured.dropna(subset=['price_up'])
    
    # Print class distribution
    class_counts = df_featured['price_up'].value_counts()
    print(f"Target distribution: {class_counts}")
    
    # 4. Select features
    selected_features = efficient_feature_selection(
        df_featured, 'price_up', mi_threshold=0.01, max_features=20
    )
    
    # 5. Train model - use optimized approach
    model, results, scaler = train_optimized_model(
        df=df_featured,
        features=selected_features,
        target_col='price_up',
        horizon=horizon,
        cv_splits=3  # Reduced for speed
    )
    
    # 6. Backtest - use optimized approach
    backtest_results = optimized_backtest(
        df_featured, model, selected_features, scaler, threshold=threshold
    )
    '''
    # 7. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{output_dir}/model_{timestamp}.pkl"
    results_filename = f"{output_dir}/results_{timestamp}.pkl"
    
    with open(model_filename, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': selected_features}, f)
    
    with open(results_filename, 'wb') as f:
        pickle.dump({
            'backtest_results': backtest_results,
            'model_results': results,
            'features': selected_features,
            'settings': {
                'sample_rate': sample_rate,
                'horizon': horizon,
                'threshold': threshold,
                'feature_set': feature_set
            }
        }, f)
    '''
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Analysis complete in {total_time:.2f} seconds")
    #print(f"Results saved to {output_dir}")
    
    return {
        'model': model,
        'results': results,
        'backtest_results': backtest_results,
        'features': selected_features,
        'execution_time': total_time
    }

# ------------------------------
# 7. Execution
# ------------------------------
if __name__ == "__main__":
    print("Starting trading algorithm optimization...")
    
    # Set parameters for faster execution
    results = run_optimized_analysis(
        file_path='BTC_1min.csv',  # Replace with your file path
        sample_rate=0.25,          # Reduced sampling for speed
        horizon=1,                 # Predict 1-minute ahead
        threshold=0.6,             # Probability threshold
        feature_set='full',   # Use essential features only
        date_filter=None           # Optional: ('2023-01-01', '2023-03-31')
    )
    
    print("Analysis complete!")