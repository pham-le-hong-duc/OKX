"""
Gold Layer - Stage 2: Create Derived Features

Create advanced features for all intervals from the merged multi-timeframe data.
Includes macro indicators, sentiment analysis, momentum, order flow, and statistical features.
"""

import polars as pl
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime


class FeatureCreator:
    """Create derived features for multi-timeframe analysis"""
    
    def __init__(self, datalake_path: str = "datalake"):
        self.datalake_path = Path(datalake_path)
        self.intervals = ["5m", "15m", "1h", "4h", "1d"]
        
        # Input and output paths
        self.input_file = self.datalake_path / "3_gold" / "step_1.parquet"
        self.output_file = self.datalake_path / "3_gold" / "step_2.parquet"
    
    def load_data(self) -> pl.DataFrame:
        """Load merged multi-timeframe data"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📖 Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def create_macro_basis_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Macro & Basis features (Index Price & Mark Price)"""
        print("🔧 Creating Macro & Basis features...")
        
        features = []
        for interval in self.intervals:
            prefix_trade = f"trades_{interval}_"
            prefix_index = f"index_price_{interval}_"
            prefix_mark = f"mark_price_{interval}_"
            
            # Base columns - fix column names based on actual data
            price_mean_trade = f"{prefix_trade}price_mean_trade"
            index_mean = f"{prefix_index}mean"  
            mark_mean = f"{prefix_mark}mean"
            price_std_trade = f"{prefix_trade}price_std_trade"
            index_std = f"{prefix_index}std"
            
            # Check if required columns exist
            if not all(col in df.columns for col in [price_mean_trade, index_mean, mark_mean]):
                print(f"⚠️  Missing columns for {interval}, skipping...")
                continue
            
            features.extend([
                # feat_basis_spread: Basis absolute difference
                (pl.col(price_mean_trade) - pl.col(index_mean)).alias(f"feat_basis_spread_{interval}"),
                
                # feat_basis_ratio: Normalized % difference
                ((pl.col(price_mean_trade) - pl.col(index_mean)) / (pl.col(index_mean) + 1e-9)).alias(f"feat_basis_ratio_{interval}"),
                
                # feat_premium_index: Premium index for funding rate calculation
                ((pl.col(mark_mean) - pl.col(index_mean)) / (pl.col(index_mean) + 1e-9)).alias(f"feat_premium_index_{interval}"),
                
                # feat_volatility_spread: Volatility difference between futures and spot
                (pl.col(price_std_trade) - pl.col(index_std)).alias(f"feat_volatility_spread_{interval}"),
            ])
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_log_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create log returns for trend divergence calculation"""
        print("🔧 Creating log returns...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            prefix_index = f"index_price_{interval}_"
            
            wmp_last = f"{prefix_orderbook}wmp_last"
            index_close = f"{prefix_index}close"
            
            if not all(col in df.columns for col in [wmp_last, index_close]):
                continue
            
            features.extend([
                # Log return for trades (futures)
                (pl.col(wmp_last) / (pl.col(wmp_last).shift(1) + 1e-9)).log().alias(f"feat_log_return_trade_{interval}"),
                
                # Log return for index (spot)
                (pl.col(index_close) / (pl.col(index_close).shift(1) + 1e-9)).log().alias(f"feat_log_return_index_{interval}"),
            ])
        
        if features:
            df = df.with_columns(features)
        
        # Create trend divergence after log returns are computed
        divergence_features = []
        for interval in self.intervals:
            log_return_trade = f"feat_log_return_trade_{interval}"
            log_return_index = f"feat_log_return_index_{interval}"
            
            if all(col in df.columns for col in [log_return_trade, log_return_index]):
                divergence_features.append(
                    (pl.col(log_return_trade) - pl.col(log_return_index)).alias(f"feat_trend_divergence_{interval}")
                )
        
        if divergence_features:
            df = df.with_columns(divergence_features)
        
        return df
    
    def create_funding_sentiment_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Funding & Sentiment features"""
        print("🔧 Creating Funding & Sentiment features...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            
            wmp_last = f"{prefix_orderbook}wmp_last"
            funding_rate = "funding_funding_rate"  # Single funding rate column
            basis_ratio = f"feat_basis_ratio_{interval}"
            
            if not all(col in df.columns for col in [wmp_last, funding_rate]):
                continue
            
            features.extend([
                # feat_funding_cost: Cost of holding position overnight
                (pl.col(funding_rate) * pl.col(wmp_last)).alias(f"feat_funding_cost_{interval}"),
                
                # feat_funding_trend: Funding rate trend vs 12-period average
                (pl.col(funding_rate) - pl.col(funding_rate).rolling_mean(12)).alias(f"feat_funding_trend_{interval}"),
            ])
            
            # feat_funding_basis_corr: Funding-basis correlation
            if basis_ratio in df.columns:
                features.append(
                    (pl.col(funding_rate) * pl.col(basis_ratio)).alias(f"feat_funding_basis_corr_{interval}")
                )
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_momentum_trend_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Momentum & Trend features"""
        print("🔧 Creating Momentum & Trend features...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            
            wmp_last = f"{prefix_orderbook}wmp_last"
            wmp_min = f"{prefix_orderbook}wmp_min"
            wmp_max = f"{prefix_orderbook}wmp_max"
            
            if not all(col in df.columns for col in [wmp_last, wmp_min, wmp_max]):
                continue
            
            features.extend([
                # feat_price_velocity: Price velocity over 3 periods
                ((pl.col(wmp_last) - pl.col(wmp_last).shift(3)) / 3).alias(f"feat_price_velocity_{interval}"),
                
                # feat_ma_divergence: Distance from 12-period MA
                (pl.col(wmp_last) / (pl.col(wmp_last).rolling_mean(12) + 1e-9) - 1).alias(f"feat_ma_divergence_{interval}"),
                
                # feat_rsi_proxy: Relative position in current window
                ((pl.col(wmp_last) - pl.col(wmp_min)) / (pl.col(wmp_max) - pl.col(wmp_min) + 1e-9)).alias(f"feat_rsi_proxy_{interval}"),
            ])
        
        # Add price acceleration after velocity is computed
        if features:
            df = df.with_columns(features)
        
        # Create acceleration features
        accel_features = []
        for interval in self.intervals:
            velocity_col = f"feat_price_velocity_{interval}"
            if velocity_col in df.columns:
                accel_features.append(
                    (pl.col(velocity_col) - pl.col(velocity_col).shift(1)).alias(f"feat_price_accel_{interval}")
                )
        
        if accel_features:
            df = df.with_columns(accel_features)
        
        return df
    
    def create_order_flow_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Order Flow & Interaction features"""
        print("🔧 Creating Order Flow features...")
        
        features = []
        for interval in self.intervals:
            prefix_trades = f"trades_{interval}_"
            prefix_orderbook = f"orderbook_{interval}_"
            
            volume_buy = f"{prefix_trades}volume_buy"
            volume_sell = f"{prefix_trades}volume_sell"
            volume_total = f"{prefix_trades}volume_total"
            count_buy = f"{prefix_trades}count_buy"
            count_sell = f"{prefix_trades}count_sell"
            
            sum_bid_50 = f"{prefix_orderbook}sum_bid_50"
            sum_ask_50 = f"{prefix_orderbook}sum_ask_50"
            total_depth_50 = f"{prefix_orderbook}total_depth_50"
            
            if not all(col in df.columns for col in [volume_buy, volume_sell, volume_total]):
                continue
            
            features.extend([
                # feat_trade_imbalance: Volume order imbalance
                ((pl.col(volume_buy) - pl.col(volume_sell)) / (pl.col(volume_total) + 1e-9)).alias(f"feat_trade_imbalance_{interval}"),
            ])
            
            # Order book features if available
            if all(col in df.columns for col in [sum_bid_50, sum_ask_50, total_depth_50]):
                features.extend([
                    # feat_depth_imbalance: Order book depth imbalance
                    ((pl.col(sum_bid_50) - pl.col(sum_ask_50)) / (pl.col(total_depth_50) + 1e-9)).alias(f"feat_depth_imbalance_{interval}"),
                    
                    # feat_buy_consumption: Buy pressure vs ask liquidity
                    (pl.col(volume_buy) / (pl.col(sum_ask_50) + 1e-9)).alias(f"feat_buy_consumption_{interval}"),
                    
                    # feat_sell_consumption: Sell pressure vs bid liquidity
                    (pl.col(volume_sell) / (pl.col(sum_bid_50) + 1e-9)).alias(f"feat_sell_consumption_{interval}"),
                ])
            
            # feat_aggressiveness: Trade frequency ratio
            if all(col in df.columns for col in [count_buy, count_sell]):
                features.append(
                    (pl.col(count_buy) / (pl.col(count_sell) + 1e-9)).alias(f"feat_aggressiveness_{interval}")
                )
        
        if features:
            df = df.with_columns(features)
        
        # Create smart money divergence after imbalances are computed
        smart_money_features = []
        for interval in self.intervals:
            trade_imbal = f"feat_trade_imbalance_{interval}"
            depth_imbal = f"feat_depth_imbalance_{interval}"
            
            if all(col in df.columns for col in [trade_imbal, depth_imbal]):
                smart_money_features.append(
                    (pl.col(trade_imbal) - pl.col(depth_imbal)).alias(f"feat_smart_money_div_{interval}")
                )
        
        if smart_money_features:
            df = df.with_columns(smart_money_features)
        
        return df
    
    def create_volatility_liquidity_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Volatility & Liquidity features"""
        print("🔧 Creating Volatility & Liquidity features...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            prefix_trades = f"trades_{interval}_"
            
            spread_mean = f"{prefix_orderbook}spread_mean"
            wmp_mean = f"{prefix_orderbook}wmp_mean"
            total_depth_50 = f"{prefix_orderbook}total_depth_50"
            
            price_max_trade = f"{prefix_trades}price_max_trade"
            price_min_trade = f"{prefix_trades}price_min_trade"
            price_mean_trade = f"{prefix_trades}price_mean_trade"
            price_last_trade = f"{prefix_trades}price_last_trade"
            
            if not all(col in df.columns for col in [spread_mean, wmp_mean]):
                continue
            
            features.extend([
                # feat_rel_spread: Normalized spread
                (pl.col(spread_mean) / (pl.col(wmp_mean) + 1e-9)).alias(f"feat_rel_spread_{interval}"),
            ])
            
            # Liquidity density if depth available
            if total_depth_50 in df.columns:
                features.append(
                    (pl.col(total_depth_50) / (pl.col(spread_mean) + 1e-9)).alias(f"feat_liq_density_{interval}")
                )
            
            # Candle features if trade prices available - fix column names
            price_max_trade_actual = f"{prefix_trades}price_max_trade"
            price_min_trade_actual = f"{prefix_trades}price_min_trade"
            price_mean_trade_actual = f"{prefix_trades}price_mean_trade"
            price_last_trade_actual = f"{prefix_trades}price_last_trade"
            
            if all(col in df.columns for col in [price_max_trade_actual, price_min_trade_actual, price_mean_trade_actual, price_last_trade_actual]):
                features.extend([
                    # feat_candle_range: Realized volatility
                    ((pl.col(price_max_trade_actual) - pl.col(price_min_trade_actual)) / (pl.col(price_mean_trade_actual) + 1e-9)).alias(f"feat_candle_range_{interval}"),
                    
                    # feat_tail_extension: Upper tail ratio
                    ((pl.col(price_max_trade_actual) - pl.col(price_last_trade_actual)) / 
                     (pl.col(price_max_trade_actual) - pl.col(price_min_trade_actual) + 1e-9)).alias(f"feat_tail_extension_{interval}"),
                ])
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Time features"""
        print("🔧 Creating Time features...")
        
        # Extract hour from timestamp
        df = df.with_columns([
            pl.col("timestamp_dt").dt.hour().alias("hour")
        ])
        
        # Create cyclical time features
        time_features = [
            # Hour sin/cos encoding
            (2 * np.pi * pl.col("hour") / 24).sin().alias("feat_hour_sin"),
            (2 * np.pi * pl.col("hour") / 24).cos().alias("feat_hour_cos"),
        ]
        
        df = df.with_columns(time_features)
        
        # Drop temporary hour column
        df = df.drop("hour")
        
        return df
    
    def create_efficiency_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Market Efficiency & Fractal features"""
        print("🔧 Creating Market Efficiency features...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            wmp_last = f"{prefix_orderbook}wmp_last"
            
            if wmp_last not in df.columns:
                continue
            
            # Simple efficiency ratio approximation
            # Efficiency = |net_movement| / sum(|movements|)
            window_size = 12
            
            features.extend([
                # feat_efficiency_ratio: Kaufman efficiency approximation
                (pl.col(wmp_last).diff().abs().rolling_sum(window_size) / 
                 (pl.col(wmp_last).diff().abs().rolling_sum(window_size) + 1e-9)).alias(f"feat_efficiency_ratio_{interval}"),
                
                # feat_price_entropy: Price entropy approximation using std
                (pl.col(wmp_last).rolling_std(window_size) / (pl.col(wmp_last).rolling_mean(window_size) + 1e-9)).alias(f"feat_price_entropy_{interval}"),
            ])
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_orderbook_shape_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Advanced Orderbook Shape features"""
        print("🔧 Creating Orderbook Shape features...")
        
        features = []
        for interval in self.intervals:
            prefix = f"orderbook_{interval}_"
            
            # Required columns for slope calculation
            sum_bid_20 = f"{prefix}sum_bid_20"
            sum_ask_20 = f"{prefix}sum_ask_20"
            sum_bid_5 = f"{prefix}sum_bid_5"
            sum_bid_50 = f"{prefix}sum_bid_50"
            best_bid = f"{prefix}best_bid"
            best_ask = f"{prefix}best_ask"
            bid_px_20 = f"{prefix}bid_px_20"
            ask_px_20 = f"{prefix}ask_px_20"
            
            slope_cols = [sum_bid_20, sum_ask_20, best_bid, best_ask, bid_px_20, ask_px_20]
            if all(col in df.columns for col in slope_cols):
                features.extend([
                    # feat_bid_slope: Bid side slope
                    (pl.col(sum_bid_20) / (pl.col(best_bid) - pl.col(bid_px_20) + 1e-9)).alias(f"feat_bid_slope_{interval}"),
                    
                    # feat_ask_slope: Ask side slope
                    (pl.col(sum_ask_20) / (pl.col(ask_px_20) - pl.col(best_ask) + 1e-9)).alias(f"feat_ask_slope_{interval}"),
                ])
            
            # Depth convexity if available
            if all(col in df.columns for col in [sum_bid_5, sum_bid_50]):
                features.append(
                    (pl.col(sum_bid_5) / (pl.col(sum_bid_50) + 1e-9)).alias(f"feat_depth_convexity_{interval}")
                )
        
        if features:
            df = df.with_columns(features)
        
        # Create slope imbalance after slopes are computed
        slope_imbal_features = []
        for interval in self.intervals:
            bid_slope = f"feat_bid_slope_{interval}"
            ask_slope = f"feat_ask_slope_{interval}"
            
            if all(col in df.columns for col in [bid_slope, ask_slope]):
                slope_imbal_features.append(
                    (pl.col(bid_slope) / (pl.col(ask_slope) + 1e-9)).alias(f"feat_slope_imbalance_{interval}")
                )
        
        if slope_imbal_features:
            df = df.with_columns(slope_imbal_features)
        
        return df
    
    def create_statistical_normalization(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create Statistical Normalization (Z-Scores)"""
        print("🔧 Creating Statistical Normalization features...")
        
        features = []
        for interval in self.intervals:
            prefix_trades = f"trades_{interval}_"
            prefix_orderbook = f"orderbook_{interval}_"
            
            volume_total = f"{prefix_trades}volume_total"
            spread_mean = f"{prefix_orderbook}spread_mean"
            trade_imbalance = f"feat_trade_imbalance_{interval}"
            
            window_1h = 12  # Assuming 5-minute base intervals for 1h window
            
            if volume_total in df.columns:
                features.extend([
                    # feat_z_volume: Volume Z-score
                    ((pl.col(volume_total) - pl.col(volume_total).rolling_mean(window_1h)) / 
                     (pl.col(volume_total).rolling_std(window_1h) + 1e-9)).alias(f"feat_z_volume_{interval}"),
                ])
            
            if spread_mean in df.columns:
                features.extend([
                    # feat_z_spread: Spread Z-score
                    ((pl.col(spread_mean) - pl.col(spread_mean).rolling_mean(window_1h)) / 
                     (pl.col(spread_mean).rolling_std(window_1h) + 1e-9)).alias(f"feat_z_spread_{interval}"),
                ])
            
            if trade_imbalance in df.columns:
                features.extend([
                    # feat_z_imbalance: Imbalance Z-score
                    ((pl.col(trade_imbalance) - pl.col(trade_imbalance).rolling_mean(window_1h)) / 
                     (pl.col(trade_imbalance).rolling_std(window_1h) + 1e-9)).alias(f"feat_z_imbalance_{interval}"),
                ])
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_vwap_pivot_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create VWAP & Pivot features"""
        print("🔧 Creating VWAP & Pivot features...")
        
        features = []
        for interval in self.intervals:
            prefix_orderbook = f"orderbook_{interval}_"
            prefix_trades = f"trades_{interval}_"
            
            wmp_last = f"{prefix_orderbook}wmp_last"
            vwap = f"{prefix_trades}vwap"
            
            if not all(col in df.columns for col in [wmp_last, vwap]):
                continue
            
            window_1h = 12
            
            features.extend([
                # feat_dist_vwap: Distance from VWAP
                ((pl.col(wmp_last) - pl.col(vwap)) / (pl.col(vwap) + 1e-9)).alias(f"feat_dist_vwap_{interval}"),
                
                # feat_dist_max: Distance from recent high
                ((pl.col(wmp_last) - pl.col(wmp_last).rolling_max(window_1h)) / 
                 (pl.col(wmp_last).rolling_max(window_1h) + 1e-9)).alias(f"feat_dist_max_{interval}"),
                
                # feat_dist_min: Distance from recent low
                ((pl.col(wmp_last) - pl.col(wmp_last).rolling_min(window_1h)) / 
                 (pl.col(wmp_last).rolling_min(window_1h) + 1e-9)).alias(f"feat_dist_min_{interval}"),
            ])
        
        if features:
            df = df.with_columns(features)
        
        return df
    
    def create_all_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create all derived features"""
        print("🚀 Creating derived features for all intervals...")
        
        # Sort by timestamp to ensure proper ordering for lag operations
        df = df.sort("timestamp_dt")
        
        # Create features in logical order
        df = self.create_log_returns(df)  # Must be first for trend divergence
        df = self.create_macro_basis_features(df)
        df = self.create_funding_sentiment_features(df)
        df = self.create_momentum_trend_features(df)
        df = self.create_order_flow_features(df)
        df = self.create_volatility_liquidity_features(df)
        df = self.create_time_features(df)
        df = self.create_efficiency_features(df)
        df = self.create_orderbook_shape_features(df)
        df = self.create_statistical_normalization(df)
        df = self.create_vwap_pivot_features(df)
        
        return df
    
    def save_data(self, df: pl.DataFrame) -> None:
        """Save enhanced data with derived features"""
        output_dir = self.output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving enhanced data to {self.output_file}")
        df.write_parquet(self.output_file, compression="snappy")
        
        # Count new features
        original_cols = len(pl.read_parquet(self.input_file).columns)
        new_cols = len(df.columns) - original_cols
        
        print(f"✅ Success! Enhanced {len(df):,} rows")
        print(f"📊 Features: {original_cols} original + {new_cols} derived = {len(df.columns)} total")
        print(f"💾 File size: {self.output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Show time range
        if not df.is_empty():
            min_time = df.select(pl.col("timestamp_dt").min()).item()
            max_time = df.select(pl.col("timestamp_dt").max()).item()
            print(f"📅 Time range: {min_time} to {max_time}")
    
    def run(self) -> bool:
        """Run the feature creation process"""
        try:
            df = self.load_data()
            df_enhanced = self.create_all_features(df)
            self.save_data(df_enhanced)
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Create derived features for multi-timeframe analysis")
    parser.add_argument("--datalake-path", default="datalake", help="Path to datalake directory")
    
    args = parser.parse_args()
    
    creator = FeatureCreator(datalake_path=args.datalake_path)
    success = creator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()