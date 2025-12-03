"""
Gold Layer - Stage 3: Create Lag Features

Create lag features for all intervals based on historical lookback windows.
Each interval has different lag periods optimized for that timeframe.
"""

import polars as pl
from pathlib import Path
import argparse
import sys
from typing import List, Dict


class LagFeatureCreator:
    """Create lag features for multi-timeframe analysis"""
    
    def __init__(self, datalake_path: str = "datalake"):
        self.datalake_path = Path(datalake_path)
        
        # Input and output paths
        self.input_file = self.datalake_path / "3_gold" / "step_2.parquet"
        self.output_file = self.datalake_path / "3_gold" / "step_3.parquet"
        
        # Define lag periods for each interval (in number of periods)
        self.lag_configs = {
            "5m": [1, 3, 12, 48, 288],     # 5m, 15m, 1h, 4h, 1d
            "15m": [1, 4, 16, 48, 288],    # 15m, 1h, 4h, 1d, 3d  
            "1h": [1, 4, 24, 72, 288],     # 1h, 4h, 1d, 3d, 12d
            "4h": [1, 6, 18, 72, 144],     # 4h, 1d, 3d, 12d, 24d
            "1d": [1, 3, 12, 24, 48]       # 1d, 3d, 12d, 24d, 48d
        }
        
        # Convert to human readable names
        self.lag_names = {
            "5m": ["5m", "15m", "1h", "4h", "1d"],
            "15m": ["15m", "1h", "4h", "1d", "3d"],
            "1h": ["1h", "4h", "1d", "3d", "12d"],
            "4h": ["4h", "1d", "3d", "12d", "24d"],
            "1d": ["1d", "3d", "12d", "24d", "48d"]
        }
        
        # Define feature groups to create lags for
        self.feature_groups = {
            # Group A: Price Momentum
            "momentum": [
                "feat_log_return_trade",
                "feat_price_velocity", 
                "feat_ma_divergence",
                "feat_rsi_proxy"
            ],
            
            # Group B: Order Flow (Most Important)
            "order_flow": [
                "feat_trade_imbalance",
                "feat_smart_money_div",
                "feat_buy_consumption", 
                "feat_sell_consumption"
            ],
            
            # Group C: Orderbook Structure  
            "orderbook": [
                "feat_depth_imbalance",
                "feat_bid_slope",
                "feat_ask_slope",
                "feat_liq_density"
            ],
            
            # Group D: Macro & Sentiment
            "macro": [
                "feat_basis_ratio",
                "feat_funding_trend",
                "feat_log_return_index"
            ],
            
            # Group E: Volatility
            "volatility": [
                "feat_candle_range",
                "feat_z_volume"
            ]
        }
    
    def load_data(self) -> pl.DataFrame:
        """Load enhanced data with derived features"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📖 Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def get_feature_columns_by_interval(self, df: pl.DataFrame, interval: str) -> Dict[str, List[str]]:
        """Get all feature columns for a specific interval"""
        interval_features = {}
        
        for group_name, base_features in self.feature_groups.items():
            interval_features[group_name] = []
            
            for base_feature in base_features:
                # Look for columns with this base feature and interval
                feature_col = f"{base_feature}_{interval}"
                if feature_col in df.columns:
                    interval_features[group_name].append(feature_col)
        
        return interval_features
    
    def create_lags_for_interval(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:
        """Create lag features for a specific interval"""
        print(f"🔧 Creating lag features for {interval}...")
        
        # Get lag configuration for this interval
        lag_periods = self.lag_configs[interval]
        lag_labels = self.lag_names[interval]
        
        # Get feature columns for this interval
        interval_features = self.get_feature_columns_by_interval(df, interval)
        
        lag_expressions = []
        
        for group_name, feature_cols in interval_features.items():
            if not feature_cols:
                continue
                
            print(f"  📊 {group_name}: {len(feature_cols)} features")
            
            for feature_col in feature_cols:
                for lag_period, lag_label in zip(lag_periods, lag_labels):
                    # Create lag feature name
                    lag_feature_name = f"{feature_col}_lag_{lag_label}"
                    
                    # Create lag expression
                    lag_expressions.append(
                        pl.col(feature_col).shift(lag_period).alias(lag_feature_name)
                    )
        
        # Apply all lag transformations
        if lag_expressions:
            df = df.with_columns(lag_expressions)
            print(f"  ✓ Created {len(lag_expressions)} lag features for {interval}")
        
        return df
    
    def create_special_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create special lag features and combinations"""
        print("🔧 Creating special lag features...")
        
        special_features = []
        
        # For each interval, create some special lag combinations
        for interval in ["5m", "15m", "1h", "4h", "1d"]:
            
            # Momentum persistence (Current vs Recent Past)
            log_return_col = f"feat_log_return_trade_{interval}"
            log_return_lag1 = f"feat_log_return_trade_{interval}_lag_{self.lag_names[interval][0]}"
            
            if all(col in df.columns for col in [log_return_col, log_return_lag1]):
                special_features.extend([
                    # Momentum persistence: Same direction as previous period?
                    (pl.col(log_return_col) * pl.col(log_return_lag1)).alias(f"feat_momentum_persistence_{interval}"),
                    
                    # Momentum acceleration: Getting stronger or weaker?
                    (pl.col(log_return_col) - pl.col(log_return_lag1)).alias(f"feat_momentum_acceleration_{interval}"),
                ])
            
            # Order flow persistence
            trade_imbal_col = f"feat_trade_imbalance_{interval}"
            trade_imbal_lag1 = f"feat_trade_imbalance_{interval}_lag_{self.lag_names[interval][0]}"
            
            if all(col in df.columns for col in [trade_imbal_col, trade_imbal_lag1]):
                special_features.extend([
                    # Order flow persistence: Same buying/selling pressure?
                    (pl.col(trade_imbal_col) * pl.col(trade_imbal_lag1)).alias(f"feat_flow_persistence_{interval}"),
                    
                    # Order flow reversal signal
                    ((pl.col(trade_imbal_col) > 0) & (pl.col(trade_imbal_lag1) < 0)).cast(pl.Int8).alias(f"feat_flow_reversal_bullish_{interval}"),
                    ((pl.col(trade_imbal_col) < 0) & (pl.col(trade_imbal_lag1) > 0)).cast(pl.Int8).alias(f"feat_flow_reversal_bearish_{interval}"),
                ])
            
            # Basis trend (Futures vs Spot divergence over time)
            basis_ratio_col = f"feat_basis_ratio_{interval}"
            basis_ratio_lag_long = f"feat_basis_ratio_{interval}_lag_{self.lag_names[interval][-2]}"  # Second longest lag
            
            if all(col in df.columns for col in [basis_ratio_col, basis_ratio_lag_long]):
                special_features.append(
                    # Basis expansion/contraction over longer timeframe
                    (pl.col(basis_ratio_col) - pl.col(basis_ratio_lag_long)).alias(f"feat_basis_expansion_{interval}")
                )
        
        if special_features:
            df = df.with_columns(special_features)
            print(f"  ✓ Created {len(special_features)} special lag features")
        
        return df
    
    def create_cross_interval_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create features comparing different intervals"""
        print("🔧 Creating cross-interval features...")
        
        cross_features = []
        
        # Compare short vs long term momentum
        pairs = [("5m", "1h"), ("15m", "4h"), ("1h", "1d")]
        
        for short_interval, long_interval in pairs:
            short_momentum = f"feat_log_return_trade_{short_interval}"
            long_momentum = f"feat_log_return_trade_{long_interval}"
            
            if all(col in df.columns for col in [short_momentum, long_momentum]):
                cross_features.extend([
                    # Momentum alignment: Are short and long term moving same direction?
                    (pl.col(short_momentum) * pl.col(long_momentum)).alias(f"feat_momentum_alignment_{short_interval}_{long_interval}"),
                    
                    # Momentum divergence: Short term vs long term
                    (pl.col(short_momentum) - pl.col(long_momentum)).alias(f"feat_momentum_divergence_{short_interval}_{long_interval}"),
                ])
            
            # Compare order flow across timeframes
            short_flow = f"feat_trade_imbalance_{short_interval}"
            long_flow = f"feat_trade_imbalance_{long_interval}"
            
            if all(col in df.columns for col in [short_flow, long_flow]):
                cross_features.append(
                    # Flow alignment across timeframes
                    (pl.col(short_flow) * pl.col(long_flow)).alias(f"feat_flow_alignment_{short_interval}_{long_interval}")
                )
        
        if cross_features:
            df = df.with_columns(cross_features)
            print(f"  ✓ Created {len(cross_features)} cross-interval features")
        
        return df
    
    def create_rolling_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create rolling statistics over lag features"""
        print("🔧 Creating rolling lag features...")
        
        rolling_features = []
        
        for interval in ["5m", "15m", "1h"]:  # Focus on more active intervals
            # Get some key lag features to create rolling stats
            key_features = [
                f"feat_log_return_trade_{interval}",
                f"feat_trade_imbalance_{interval}",
                f"feat_basis_ratio_{interval}"
            ]
            
            for feature in key_features:
                if feature not in df.columns:
                    continue
                
                # Create rolling statistics over recent periods
                window_short = 3  # Short term trend
                window_medium = 6  # Medium term trend
                
                rolling_features.extend([
                    # Rolling mean - trend over recent periods
                    pl.col(feature).rolling_mean(window_short).alias(f"{feature}_roll_mean_{window_short}p"),
                    pl.col(feature).rolling_mean(window_medium).alias(f"{feature}_roll_mean_{window_medium}p"),
                    
                    # Rolling std - consistency/volatility of the feature
                    pl.col(feature).rolling_std(window_short).alias(f"{feature}_roll_std_{window_short}p"),
                    
                    # Current vs rolling mean (normalized)
                    ((pl.col(feature) - pl.col(feature).rolling_mean(window_short)) / 
                     (pl.col(feature).rolling_std(window_short) + 1e-9)).alias(f"{feature}_vs_trend_z"),
                ])
        
        if rolling_features:
            df = df.with_columns(rolling_features)
            print(f"  ✓ Created {len(rolling_features)} rolling lag features")
        
        return df
    
    def create_all_lag_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create all lag features"""
        print("🚀 Creating lag features for all intervals...")
        
        # Sort by timestamp to ensure proper ordering for lag operations
        df = df.sort("timestamp_dt")
        
        # Create basic lag features for each interval
        for interval in ["5m", "15m", "1h", "4h", "1d"]:
            df = self.create_lags_for_interval(df, interval)
        
        # Create special lag features and combinations
        df = self.create_special_lag_features(df)
        
        # Create cross-interval features
        df = self.create_cross_interval_features(df)
        
        # Create rolling lag features
        df = self.create_rolling_lag_features(df)
        
        return df
    
    def save_data(self, df: pl.DataFrame) -> None:
        """Save enhanced data with lag features"""
        output_dir = self.output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving enhanced data to {self.output_file}")
        df.write_parquet(self.output_file, compression="snappy")
        
        # Count new features
        original_cols = len(pl.read_parquet(self.input_file).columns)
        new_cols = len(df.columns) - original_cols
        
        print(f"✅ Success! Enhanced {len(df):,} rows")
        print(f"📊 Features: {original_cols} original + {new_cols} lag = {len(df.columns)} total")
        print(f"💾 File size: {self.output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Show time range
        if not df.is_empty():
            min_time = df.select(pl.col("timestamp_dt").min()).item()
            max_time = df.select(pl.col("timestamp_dt").max()).item()
            print(f"📅 Time range: {min_time} to {max_time}")
        
        # Show lag feature counts by interval
        print(f"\n📊 Lag features by interval:")
        for interval in ["5m", "15m", "1h", "4h", "1d"]:
            lag_cols = [col for col in df.columns if f"_{interval}_lag_" in col]
            if lag_cols:
                print(f"  {interval}: {len(lag_cols)} lag features")
    
    def run(self) -> bool:
        """Run the lag feature creation process"""
        try:
            df = self.load_data()
            df_enhanced = self.create_all_lag_features(df)
            self.save_data(df_enhanced)
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Create lag features for multi-timeframe analysis")
    parser.add_argument("--datalake-path", default="datalake", help="Path to datalake directory")
    
    args = parser.parse_args()
    
    creator = LagFeatureCreator(datalake_path=args.datalake_path)
    success = creator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()