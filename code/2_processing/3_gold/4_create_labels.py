"""
Gold Layer - Stage 4: Create Labels

Create binary labels (1 = price up, 0 = price down) for all intervals.
Labels are based on future price movement for each timeframe.
"""

import polars as pl
from pathlib import Path
import argparse
import sys
from typing import List, Dict


class LabelCreator:
    """Create binary labels for multi-timeframe prediction"""
    
    def __init__(self, datalake_path: str = "datalake"):
        self.datalake_path = Path(datalake_path)
        
        # Input and output paths
        self.input_file = self.datalake_path / "3_gold" / "step_3.parquet"
        self.output_file = self.datalake_path / "3_gold" / "step_4.parquet"
        
        # Define intervals and their corresponding periods for label creation
        self.intervals = ["5m", "15m", "1h", "4h", "1d"]
        
        # Define how many periods ahead to look for each interval
        self.label_periods = {
            "5m": 1,    # Look 1 period (5m) ahead
            "15m": 1,   # Look 1 period (15m) ahead  
            "1h": 1,    # Look 1 period (1h) ahead
            "4h": 1,    # Look 1 period (4h) ahead
            "1d": 1     # Look 1 period (1d) ahead
        }
        
        # Define price columns to use for label creation (current price reference)
        self.price_columns = {
            "5m": "orderbook_5m_wmp_last",      # Current market price
            "15m": "orderbook_15m_wmp_last",
            "1h": "orderbook_1h_wmp_last", 
            "4h": "orderbook_4h_wmp_last",
            "1d": "orderbook_1d_wmp_last"
        }
    
    def load_data(self) -> pl.DataFrame:
        """Load enhanced data with lag features"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📖 Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def create_labels_for_interval(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:
        """Create binary labels for a specific interval"""
        print(f"🏷️  Creating labels for {interval}...")
        
        # Get price column and label period for this interval
        price_col = self.price_columns[interval]
        periods_ahead = self.label_periods[interval]
        
        # Check if price column exists
        if price_col not in df.columns:
            print(f"⚠️  Warning: Price column {price_col} not found, skipping {interval}")
            return df
        
        # Create future price column (shifted backwards)
        future_price_col = f"{price_col}_future_{periods_ahead}p"
        current_price = pl.col(price_col)
        future_price = pl.col(price_col).shift(-periods_ahead)
        
        # Create label expressions
        label_expressions = [
            # Future price for reference
            future_price.alias(future_price_col),
            
            # Binary label: 1 if price goes up, 0 if price goes down
            (future_price > current_price).cast(pl.Int8).alias(f"label_{interval}"),
            
            # Return magnitude for analysis (optional)
            ((future_price - current_price) / (current_price + 1e-9)).alias(f"return_{interval}"),
            
            # Absolute return for analysis
            (((future_price - current_price) / (current_price + 1e-9)).abs()).alias(f"return_abs_{interval}"),
        ]
        
        # Add label expressions to dataframe
        df = df.with_columns(label_expressions)
        
        print(f"✓ Created labels for {interval}")
        return df
    
    def create_multi_period_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create labels for multiple periods ahead"""
        print("🏷️  Creating multi-period labels...")
        
        multi_expressions = []
        
        # For shorter timeframes, also create longer horizon labels
        extended_periods = {
            "5m": [1, 3, 6, 12],     # 5m, 15m, 30m, 1h ahead
            "15m": [1, 4, 8, 16],    # 15m, 1h, 2h, 4h ahead
            "1h": [1, 4, 12, 24],    # 1h, 4h, 12h, 1d ahead
        }
        
        for interval, periods_list in extended_periods.items():
            price_col = self.price_columns[interval]
            
            if price_col not in df.columns:
                continue
                
            print(f"  📊 Multi-period labels for {interval}")
            
            for periods in periods_list:
                if periods == 1:  # Skip single period as it's already created
                    continue
                    
                current_price = pl.col(price_col)
                future_price = pl.col(price_col).shift(-periods)
                
                # Calculate time horizon in readable format
                if interval == "5m":
                    horizon = f"{periods * 5}m"
                elif interval == "15m":
                    horizon = f"{periods * 15}m" if periods <= 4 else f"{periods * 15 // 60}h"
                else:  # 1h
                    horizon = f"{periods}h" if periods <= 24 else f"{periods // 24}d"
                
                multi_expressions.extend([
                    # Multi-period binary label
                    (future_price > current_price).cast(pl.Int8).alias(f"label_{interval}_{horizon}"),
                    
                    # Multi-period return
                    ((future_price - current_price) / (current_price + 1e-9)).alias(f"return_{interval}_{horizon}"),
                ])
        
        if multi_expressions:
            df = df.with_columns(multi_expressions)
            print(f"✓ Created {len(multi_expressions)} multi-period labels")
        
        return df
    
    def create_categorical_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create categorical labels (strong up, weak up, weak down, strong down)"""
        print("🏷️  Creating categorical labels...")
        
        categorical_expressions = []
        
        # Define thresholds for categorization (in percentage)
        thresholds = {
            "5m": 0.001,    # 0.1% for 5-minute moves
            "15m": 0.002,   # 0.2% for 15-minute moves  
            "1h": 0.005,    # 0.5% for 1-hour moves
            "4h": 0.01,     # 1.0% for 4-hour moves
            "1d": 0.02      # 2.0% for daily moves
        }
        
        for interval in self.intervals:
            return_col = f"return_{interval}"
            threshold = thresholds[interval]
            
            if return_col not in df.columns:
                continue
                
            # Create categorical label:
            # 3 = Strong Up (return > threshold)
            # 2 = Weak Up (0 < return <= threshold)  
            # 1 = Weak Down (-threshold <= return < 0)
            # 0 = Strong Down (return < -threshold)
            categorical_expr = (
                pl.when(pl.col(return_col) > threshold).then(3)
                .when(pl.col(return_col) > 0).then(2)
                .when(pl.col(return_col) >= -threshold).then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias(f"label_cat_{interval}")
            )
            
            categorical_expressions.append(categorical_expr)
        
        if categorical_expressions:
            df = df.with_columns(categorical_expressions)
            print(f"✓ Created {len(categorical_expressions)} categorical labels")
        
        return df
    
    def create_volatility_adjusted_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create volatility-adjusted labels"""
        print("🏷️  Creating volatility-adjusted labels...")
        
        vol_expressions = []
        
        for interval in ["5m", "15m", "1h"]:  # Focus on active intervals
            return_col = f"return_{interval}"
            
            if return_col not in df.columns:
                continue
            
            # Calculate rolling volatility
            window = 24  # 24 periods rolling window
            vol_col = f"vol_{interval}_{window}p"
            
            vol_expressions.extend([
                # Rolling volatility
                pl.col(return_col).rolling_std(window).alias(vol_col),
                
                # Volatility-adjusted return (return / volatility)
                ((pl.col(return_col)) / (pl.col(return_col).rolling_std(window) + 1e-9)).alias(f"return_vol_adj_{interval}"),
            ])
        
        # Apply volatility calculations first
        if vol_expressions:
            df = df.with_columns(vol_expressions)
        
        # Create volatility-adjusted labels
        vol_label_expressions = []
        for interval in ["5m", "15m", "1h"]:
            vol_adj_return_col = f"return_vol_adj_{interval}"
            
            if vol_adj_return_col not in df.columns:
                continue
                
            # Label based on volatility-adjusted return
            # Threshold of 1.0 means 1 standard deviation move
            vol_label_expressions.extend([
                # Binary vol-adjusted label
                (pl.col(vol_adj_return_col) > 0).cast(pl.Int8).alias(f"label_vol_adj_{interval}"),
                
                # Strong move label (> 1 std dev)
                (pl.col(vol_adj_return_col).abs() > 1.0).cast(pl.Int8).alias(f"label_strong_move_{interval}"),
            ])
        
        if vol_label_expressions:
            df = df.with_columns(vol_label_expressions)
            print(f"✓ Created {len(vol_label_expressions)} volatility-adjusted labels")
        
        return df
    
    def create_optimal_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create THE KING of Labels: Binary with Trading Fee Threshold"""
        print("👑 Creating OPTIMAL Binary Labels with Fee Threshold...")
        
        # Sort by timestamp to ensure proper ordering
        df = df.sort("timestamp_dt")
        
        # Define trading fee + profit thresholds (optimized for precision)
        fee_thresholds = {
            "5m": 0.0018,   # 0.18% - Tinh chỉnh để balance cơ hội và chất lượng
            "15m": 0.004,   # 0.4% - Balance between opportunity và quality
            "1h": 0.008,    # 0.8% - Clear swing moves only
            "4h": 0.015,    # 1.5% - Hạ nhẹ vì 2.0% hơi hiếm
            "1d": 0.02      # 2.0% - Conservative long-term moves
        }
        
        optimal_expressions = []
        
        for interval in self.intervals:
            price_col = self.price_columns[interval]
            threshold = fee_thresholds[interval]
            periods_ahead = self.label_periods[interval]
            
            if price_col not in df.columns:
                print(f"⚠️  Warning: Price column {price_col} not found, skipping {interval}")
                continue
            
            current_price = pl.col(price_col)
            future_price = pl.col(price_col).shift(-periods_ahead)
            
            # Calculate return
            return_pct = (future_price - current_price) / (current_price + 1e-9)
            
            optimal_expressions.extend([
                # THE KING LABEL: Binary with Fee Threshold
                # 1 (BUY) = Return > threshold (enough to beat fees + profit)
                # 0 (SKIP) = Return <= threshold (not profitable after fees)
                (return_pct > threshold).cast(pl.Int8).alias(f"label_{interval}"),
            ])
            
            print(f"✓ Created optimal label for {interval} (threshold: {threshold:.3f}%)")
        
        if optimal_expressions:
            df = df.with_columns(optimal_expressions)
        
        return df
    
    def create_all_labels(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create ONLY the most important labels"""
        print("👑 Creating THE KING of Labels: Profitable Binary Classification")
        
        # Create only the optimal labels
        df = self.create_optimal_labels(df)
        
        return df
    
    def analyze_label_distribution(self, df: pl.DataFrame) -> None:
        """Analyze THE KING label distributions"""
        print("\n👑 THE KING Label Distribution Analysis:")
        print("(Shows % of profitable opportunities after fees)")
        
        for interval in self.intervals:
            label_col = f"label_{interval}"
            
            if label_col not in df.columns:
                continue
                
            # Calculate distributions - CLEAN VERSION
            stats = df.select([
                pl.col(label_col).filter(pl.col(label_col).is_not_null()).mean().alias("profitable_ratio"),
                pl.col(label_col).filter(pl.col(label_col).is_not_null()).count().alias("total_count"),
                pl.col(label_col).is_null().sum().alias("null_count"),
            ]).to_dict(as_series=False)
            
            profitable_ratio = stats["profitable_ratio"][0]
            total_count = stats["total_count"][0] 
            null_count = stats["null_count"][0]
            
            if profitable_ratio is not None:
                print(f"  📊 {interval}:")
                print(f"    💰 Profitable trades: {profitable_ratio:.1%} ({int(total_count * profitable_ratio):,} opportunities)")
                print(f"    📝 Total samples: {total_count:,} | Nulls: {null_count:,}")
                print()
    
    def save_data(self, df: pl.DataFrame) -> None:
        """Save enhanced data with labels"""
        output_dir = self.output_file.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving labeled data to {self.output_file}")
        df.write_parquet(self.output_file, compression="snappy")
        
        # Count new features
        original_cols = len(pl.read_parquet(self.input_file).columns)
        new_cols = len(df.columns) - original_cols
        
        print(f"✅ Success! Enhanced {len(df):,} rows")
        print(f"👑 THE KING Labels: {original_cols} original + {new_cols} optimal = {len(df.columns)} total")
        print(f"💾 File size: {self.output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Show time range
        if not df.is_empty():
            min_time = df.select(pl.col("timestamp_dt").min()).item()
            max_time = df.select(pl.col("timestamp_dt").max()).item()
            print(f"📅 Time range: {min_time} to {max_time}")
    
    def run(self) -> bool:
        """Run the label creation process"""
        try:
            df = self.load_data()
            df_labeled = self.create_all_labels(df)
            self.analyze_label_distribution(df_labeled)
            self.save_data(df_labeled)
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Create labels for multi-timeframe prediction")
    parser.add_argument("--datalake-path", default="datalake", help="Path to datalake directory")
    
    args = parser.parse_args()
    
    creator = LabelCreator(datalake_path=args.datalake_path)
    success = creator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()