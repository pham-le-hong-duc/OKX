"""
LightGBM Model Training - 15m Interval

Walk-Forward Validation with Feature Selection for 15-minute trading strategy.
Fixed test size with expanding train window approach.
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
    """LightGBM Walk-Forward Validation for 15m interval"""
    
    def __init__(self, data_path: str = "datalake/4_diamond/15m.parquet"):
        self.data_path = Path(data_path)
        self.interval = "15m"
        
        # Walk-Forward Configuration for 15m
        self.test_size = 2900           # ~1 month of 15m data
        self.min_train_size = 8640      # ~3 months (minimum train)
        self.step_size = 1344           # ~2 weeks step
        
        # Model Configuration
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Feature selection threshold
        self.feature_threshold = 'median'  # Top 50% features
        
        # Results storage
        self.results = []
        
        # Trading simulation parameters
        self.trading_fee = 0.0004  # 0.04% per trade
        
        # Prediction threshold for precision optimization
        self.prediction_threshold = 0.65  # Strict confidence for 15m
        
    def load_data(self):
        """Load and prepare data"""
        print(f"📖 Loading {self.interval} data from {self.data_path}")
        
        df = pd.read_parquet(self.data_path)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
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
        """Feature selection: LightGBM importance only (correlation already applied)"""
        print("🔧 LightGBM importance selection on correlation-filtered features...")
        
        # LightGBM feature importance on correlation-filtered features
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Use a smaller, faster model for feature selection
        fs_params = self.params.copy()
        fs_params.update({
            'n_estimators': 100,  # Faster for feature selection
            'learning_rate': 0.1,
            'verbose': -1
        })
        
        model = lgb.train(
            fs_params,
            train_data,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)]
        )
        
        # Get feature importance
        feature_importance = model.feature_importance(importance_type='gain')
        feature_names = X_train.columns
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Select features based on threshold
        if self.feature_threshold == 'median':
            threshold_value = importance_df['importance'].median()
            selected_features = importance_df[importance_df['importance'] >= threshold_value]['feature'].tolist()
        else:
            # If numeric threshold provided
            n_features = int(self.feature_threshold)
            selected_features = importance_df.head(n_features)['feature'].tolist()
        
        print(f"✓ Final selection: {len(selected_features)} features")
        print(f"  (Original: {len(X_train.columns)} → Correlation filtered: {len(feature_names)} → Final: {len(selected_features)})")
        
        # Get top feature for logging
        top_feature = importance_df.iloc[0]['feature'] if len(importance_df) > 0 else "N/A"
        
        return X_train[selected_features], X_test[selected_features], selected_features, top_feature
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
        
        model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0)
            ]
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > self.prediction_threshold).astype(int)
        
        # Accuracy
        accuracy = (y_pred == y_test).mean()
        
        # Precision, Recall (for additional metrics)
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def simulate_trading(self, y_test, y_pred, returns_test):
        """Simulate trading performance"""
        trade_signals = y_pred
        actual_returns = returns_test if returns_test is not None else np.zeros(len(y_test))
        
        # Calculate PnL for each trade
        trade_returns = []
        total_trades = 0
        
        for i in range(len(trade_signals)):
            if trade_signals[i] == 1:  # BUY signal
                ret = actual_returns[i] if i < len(actual_returns) else 0
                trade_return = ret - self.trading_fee
                trade_returns.append(trade_return)
                total_trades += 1
        
        if total_trades == 0:
            return {'total_pnl': 0, 'avg_trade_return': 0, 'total_trades': 0}
        
        total_pnl = sum(trade_returns)
        avg_trade_return = np.mean(trade_returns)
        
        return {
            'total_pnl': total_pnl,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades
        }
    
    def walk_forward_validation(self):
        """Execute walk-forward validation"""
        print(f"🚀 Starting Walk-Forward Validation for {self.interval}")
        print("="*60)
        
        # Load data
        X, y, timestamps = self.load_data()
        
        # Pre-compute correlation filter ONCE on entire dataset
        print("🧹 Pre-computing correlation filter on entire dataset...")
        X_corr_filtered, _, self.selected_features_corr = self.correlation_filter(X, X.iloc[:100])
        print(f"✓ Correlation filter: {len(X.columns)} → {len(X_corr_filtered.columns)} features")
        
        # Use filtered features for all folds
        X = X_corr_filtered
        
        total_samples = len(X)
        print(f"📊 Total samples: {total_samples:,}")
        print(f"📊 Test size: {self.test_size:,}")
        print(f"📊 Min train size: {self.min_train_size:,}")
        print(f"📊 Step size: {self.step_size:,}")
        
        # Calculate number of folds
        max_start_idx = total_samples - self.test_size
        start_indices = list(range(self.min_train_size, max_start_idx, self.step_size))
        
        if not start_indices:
            print("❌ Not enough data for walk-forward validation")
            return
        
        print(f"📊 Number of folds: {len(start_indices)}")
        print()
        
        fold_id = 0
        
        for train_end_idx in start_indices:
            fold_id += 1
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + self.test_size, total_samples)
            
            # Skip if test set too small
            if test_end_idx - test_start_idx < self.test_size * 0.8:
                continue
                
            print(f"📁 Fold {fold_id}/{len(start_indices)}")
            
            # Split data
            X_train = X.iloc[:train_end_idx].copy()
            y_train = y.iloc[:train_end_idx].copy()
            X_test = X.iloc[test_start_idx:test_end_idx].copy()
            y_test = y.iloc[test_start_idx:test_end_idx].copy()
            
            train_start_time = timestamps.iloc[0]
            train_end_time = timestamps.iloc[train_end_idx-1]
            test_start_time = timestamps.iloc[test_start_idx]
            test_end_time = timestamps.iloc[test_end_idx-1]
            
            print(f"   📅 Train: {train_start_time} to {train_end_time} ({len(X_train):,} samples)")
            print(f"   📅 Test:  {test_start_time} to {test_end_time} ({len(X_test):,} samples)")
            
            # Feature Selection
            X_train_selected, X_test_selected, selected_features, top_feature = self.feature_selection(
                X_train, y_train, X_test
            )
            
            # Train Model
            print("   🤖 Training model...")
            model = self.train_model(X_train_selected, y_train)
            
            # Evaluate
            print("   📊 Evaluating...")
            metrics = self.evaluate_model(model, X_test_selected, y_test)
            
            # Trading Simulation (simplified)
            trading_metrics = self.simulate_trading(y_test, metrics['y_pred'], None)
            
            # Store results
            result = {
                'fold_id': fold_id,
                'train_start': train_start_time,
                'train_end': train_end_time,
                'test_start': test_start_time,
                'test_end': test_end_time,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'selected_features': len(selected_features),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'total_pnl': trading_metrics['total_pnl'],
                'avg_trade_return': trading_metrics['avg_trade_return'],
                'total_trades': trading_metrics['total_trades'],
                'top_feature': top_feature
            }
            
            self.results.append(result)
            
            print(f"   ✓ Accuracy: {metrics['accuracy']:.3f}")
            print(f"   ✓ Precision: {metrics['precision']:.3f}")
            print(f"   ✓ Total Trades: {trading_metrics['total_trades']}")
            print(f"   ✓ Top Feature: {top_feature}")
            print()
        
        # Summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        # Train final model on all data
        self.train_final_model(X, y)
    
    def print_summary(self):
        """Print validation summary"""
        if not self.results:
            print("❌ No results to summarize")
            return
        
        print("="*60)
        print(f"📊 WALK-FORWARD VALIDATION SUMMARY - {self.interval}")
        print("="*60)
        
        accuracies = [r['accuracy'] for r in self.results]
        pnls = [r['total_pnl'] for r in self.results]
        trades = [r['total_trades'] for r in self.results]
        
        print(f"📈 Total Folds: {len(self.results)}")
        print(f"📈 Avg Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"📈 Best Accuracy: {np.max(accuracies):.3f}")
        print(f"📈 Worst Accuracy: {np.min(accuracies):.3f}")
        print(f"💰 Avg PnL per Fold: {np.mean(pnls):.4f}")
        print(f"💰 Total PnL: {np.sum(pnls):.4f}")
        print(f"📊 Avg Trades per Fold: {np.mean(trades):.0f}")
        print(f"📊 Total Trades: {np.sum(trades)}")
        
        # Profitable folds
        profitable_folds = sum(1 for pnl in pnls if pnl > 0)
        print(f"💵 Profitable Folds: {profitable_folds}/{len(self.results)} ({profitable_folds/len(self.results):.1%})")
    
    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Create output directory
        output_dir = Path("results/lightgbm")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        output_file = output_dir / f"training_log_{self.interval}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Save summary
        summary_file = output_dir / f"summary_{self.interval}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"LightGBM Walk-Forward Validation Summary - {self.interval}\\n")
            f.write(f"Generated: {datetime.now()}\\n\\n")
            
            accuracies = [r['accuracy'] for r in self.results]
            pnls = [r['total_pnl'] for r in self.results]
            
            f.write(f"Total Folds: {len(self.results)}\\n")
            f.write(f"Avg Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\\n")
            f.write(f"Best Accuracy: {np.max(accuracies):.3f}\\n")
            f.write(f"Worst Accuracy: {np.min(accuracies):.3f}\\n")
            f.write(f"Total PnL: {np.sum(pnls):.4f}\\n")
            
        print(f"📄 Summary saved to: {summary_file}")
    
    def train_final_model(self, X, y):
        """Train final model on all data"""
        print("🎯 Training final model on all data...")
        
        # Feature selection on full dataset
        X_selected, _, selected_features, _ = self.feature_selection(X, y, X.iloc[:100])  # Dummy test set
        
        # Train final model
        final_model = self.train_model(X_selected, y)
        
        # Save final model
        output_dir = Path("models/lightgbm")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = output_dir / f"final_model_{self.interval}.txt"
        final_model.save_model(str(model_file))
        
        # Save selected features
        features_file = output_dir / f"selected_features_{self.interval}.txt"
        with open(features_file, 'w') as f:
            for feature in selected_features:
                f.write(f"{feature}\\n")
        
        print(f"💾 Final model saved to: {model_file}")
        print(f"💾 Selected features saved to: {features_file}")
        print(f"✓ Final model uses {len(selected_features)} features")
    
    def run(self):
        """Run the complete training pipeline"""
        try:
            self.walk_forward_validation()
            return True
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description=f'LightGBM Walk-Forward Validation for 15m interval')
    parser.add_argument('--data-path', default='datalake/4_diamond/15m.parquet', help='Path to training data')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting LightGBM Training for 15m interval")
    print(f"📁 Data path: {args.data_path}")
    print()
    
    trainer = LightGBMTrainer(data_path=args.data_path)
    success = trainer.run()
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()