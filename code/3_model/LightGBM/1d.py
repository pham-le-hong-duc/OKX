"""
LightGBM Model Training - 1d Interval

Walk-Forward Validation with Feature Selection for daily trading strategy.
Fixed test size with expanding train window approach.

WARNING: Very small dataset - results have high variance and low statistical significance.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import argparse


class LightGBMTrainer:
    """LightGBM Walk-Forward Validation for 1d interval"""
    
    def __init__(self, data_path: str = "datalake/4_diamond/1d.parquet"):
        self.data_path = Path(data_path)
        self.interval = "1d"
        
        # Walk-Forward Configuration for 1d
        self.test_size = 55             # ~2 months of daily data
        self.min_train_size = 90        # ~3 months (minimum train)
        self.step_size = 14             # ~2 weeks step
        
        # Model Configuration (heavily adjusted for very small dataset)
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,        # Much reduced
            'learning_rate': 0.1,       # Higher learning rate
            'num_leaves': 15,           # Very simple trees
            'max_depth': 4,             # Shallow trees
            'subsample': 0.9,           # Use more data
            'colsample_bytree': 0.9,    # Use more features
            'min_child_samples': 5,     # Small minimum samples
            'reg_alpha': 0.1,           # L1 regularization
            'reg_lambda': 0.1,          # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Feature selection threshold - more conservative
        self.feature_threshold = 'median'
        
        # Results storage
        self.results = []
        
        # Trading simulation parameters
        self.trading_fee = 0.0004  # 0.04% per trade
        
        # Prediction threshold for precision optimization
        self.prediction_threshold = 0.55  # Relaxed confidence for 1d
        
    def load_data(self):
        """Load and prepare data"""
        print(f"📖 Loading {self.interval} data from {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Critical warning for small dataset
        if len(df) < 200:
            print("🚨 CRITICAL WARNING: Very small dataset!")
            print("   Results will have extremely high variance")
            print("   Consider this for educational purposes only")
        
        # Sort by timestamp
        df = df.sort_values('timestamp_dt').reset_index(drop=True)
        
        # Extract features and target
        target_col = f"label_{self.interval}"
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found!")
        
        # Remove non-feature columns
        exclude_cols = ['timestamp_dt'] + [col for col in df.columns if col.startswith('label_')]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        timestamps = df['timestamp_dt'].copy()
        
        print(f"✓ Features: {len(feature_cols)}, Target: {target_col}")
        print(f"✓ Target distribution: {y.mean():.1%} positive")
        print(f"🚨 Dataset size warning: Only {len(df)} samples available")
        
        return X, y, timestamps
    
    def correlation_filter(self, X_train, X_test, threshold=0.95):
        """Remove highly correlated features before feature selection"""
        print(f"🧹 Correlation filter (removing features with correlation > {threshold})...")
        
        # Calculate correlation matrix
        corr_matrix = X_train.corr().abs()
        
        # Find pairs of features with high correlation
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop (keep first occurrence, drop others)
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        print(f"✓ Removing {len(to_drop)} highly correlated features")
        
        # Keep features that are not highly correlated
        keep_features = [col for col in X_train.columns if col not in to_drop]
        
        return X_train[keep_features], X_test[keep_features], keep_features
    
    def feature_selection(self, X_train, y_train, X_test):
        """Feature selection: Correlation filter + LightGBM importance (very conservative for small data)"""
        print("🔧 Step 1: Correlation filter...")
        
        # Step 1: Remove highly correlated features
        X_train_filtered, X_test_filtered, remaining_features = self.correlation_filter(X_train, X_test)
        
        print("🔧 Step 2: LightGBM importance selection (conservative for small dataset)...")
        
        # Step 2: LightGBM feature importance on filtered features
        train_data = lgb.Dataset(X_train_filtered, label=y_train)
        
        # Minimal model for feature selection
        fs_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 30,        # Very few estimators
            'learning_rate': 0.2,      # Fast learning
            'num_leaves': 10,          # Simple trees
            'max_depth': 3,            # Shallow
            'verbose': -1
        }
        
        model = lgb.train(
            fs_params,
            train_data,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Get feature importance
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = X_train_filtered.columns
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Very conservative feature selection for tiny datasets
        if self.feature_threshold == 'median':
            # Use top 70% instead of 50% for very small datasets
            threshold_percentile = 30  # Keep top 70%
            threshold_value = np.percentile(importance_df['importance'], threshold_percentile)
            selected_features = importance_df[importance_df['importance'] >= threshold_value]['feature'].tolist()
            
            # Ensure minimum features for stability
            if len(selected_features) < 20 and len(feature_names) >= 20:
                selected_features = importance_df.head(20)['feature'].tolist()
        else:
            # If numeric threshold provided
            n_features = int(self.feature_threshold)
            selected_features = importance_df.head(n_features)['feature'].tolist()
        
        print(f"✓ Final selection: {len(selected_features)} features")
        print(f"  (Original: {len(X_train.columns)} → Correlation filtered: {len(feature_names)} → Final: {len(selected_features)})")
        
        # Get top feature for logging
        top_feature = importance_df.iloc[0]['feature'] if len(importance_df) > 0 else "N/A"
        
        return X_train_filtered[selected_features], X_test_filtered[selected_features], selected_features, top_feature