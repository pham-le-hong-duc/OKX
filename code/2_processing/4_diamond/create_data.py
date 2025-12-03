"""
Diamond Layer: Create Model-Ready Data

Split step_4.parquet into interval-specific files optimized for ML training.
Each file contains only relevant features and labels for its specific timeframe.
"""

import polars as pl
from pathlib import Path
import argparse
import sys
from typing import List, Dict, Set
import numpy as np


class DiamondDataCreator:
    """Create model-ready interval-specific datasets"""
    
    def __init__(self, datalake_path: str = "datalake"):
        self.datalake_path = Path(datalake_path)
        
        # Input and output paths
        self.input_file = self.datalake_path / "3_gold" / "step_4.parquet"
        self.output_dir = self.datalake_path / "4_diamond"
        
        # Define intervals and their sampling ratios
        self.intervals = ["5m", "15m", "1h", "4h", "1d"]
        self.sampling_ratios = {
            "5m": 1,     # Every row (5m base)
            "15m": 3,    # Every 3rd row (15m = 3×5m)
            "1h": 12,    # Every 12th row (1h = 12×5m)  
            "4h": 48,    # Every 48th row (4h = 48×5m)
            "1d": 288    # Every 288th row (1d = 288×5m)
        }
        
        # Base features that should be kept in all files
        self.base_feature_patterns = [
            "timestamp_dt",
            "funding_",
            "feat_hour_sin",
            "feat_hour_cos"
        ]
    
    def load_data(self) -> pl.DataFrame:
        """Load step_4 data and filter non-null rows"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        print(f"📖 Loading data from {self.input_file}")
        df = pl.read_parquet(self.input_file)
        print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Filter only completely non-null rows
        print("🧹 Filtering non-null rows...")
        null_count_expr = pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int32) for col in df.columns])
        df_with_null_count = df.with_columns(null_count_expr.alias("null_count"))
        
        df_clean = df_with_null_count.filter(pl.col("null_count") == 0).drop("null_count")
        
        print(f"✓ Clean data: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}% of original)")
        
        # Sort by timestamp for proper sampling
        df_clean = df_clean.sort("timestamp_dt")
        
        return df_clean
    
    def get_interval_features(self, all_columns: List[str], target_interval: str) -> Set[str]:
        """Get relevant feature columns for a specific interval"""
        relevant_features = set()
        
        # Add base features
        for col in all_columns:
            for pattern in self.base_feature_patterns:
                if pattern in col:
                    relevant_features.add(col)
                    break
        
        # Add target label
        target_label = f"label_{target_interval}"
        if target_label in all_columns:
            relevant_features.add(target_label)
        
        # Add interval-specific features (current interval and larger)
        target_idx = self.intervals.index(target_interval)
        allowed_intervals = self.intervals[target_idx:]  # Current and larger intervals
        
        for col in all_columns:
            # Check for interval-specific features
            for allowed_interval in allowed_intervals:
                if f"_{allowed_interval}_" in col:
                    relevant_features.add(col)
                    break
        
        # Add cross-interval features (only if current interval is the smallest one mentioned)
        for col in all_columns:
            if "feat_" in col and "_" in col:
                # Check for cross-interval features like feat_momentum_alignment_5m_1h
                if self.is_cross_interval_feature(col, target_interval):
                    relevant_features.add(col)
        
        return relevant_features
    
    def is_cross_interval_feature(self, feature_name: str, target_interval: str) -> bool:
        """Check if a cross-interval feature should be included in target interval file"""
        if not any(cross_pattern in feature_name for cross_pattern in ["_alignment_", "_divergence_", "_expansion_"]):
            return False
        
        # Extract intervals from feature name
        parts = feature_name.split("_")
        found_intervals = []
        
        for part in parts:
            if part in self.intervals:
                found_intervals.append(part)
        
        if len(found_intervals) >= 2:
            # Include if target_interval is the smallest interval mentioned
            smallest_interval = min(found_intervals, key=lambda x: self.intervals.index(x))
            return smallest_interval == target_interval
        
        return False
    
    def downsample_data(self, df: pl.DataFrame, target_interval: str) -> pl.DataFrame:
        """Downsample data according to target interval"""
        sampling_ratio = self.sampling_ratios[target_interval]
        
        if sampling_ratio == 1:
            return df  # No sampling needed for 5m
        
        # Sample every Nth row where N = sampling_ratio
        # We want to align with proper timestamps (e.g., top of hour for 1h)
        print(f"📊 Downsampling by factor {sampling_ratio} for {target_interval}")
        
        # Method: Take every Nth row starting from a properly aligned timestamp
        total_rows = len(df)
        sampled_indices = list(range(0, total_rows, sampling_ratio))
        
        df_sampled = df[sampled_indices]
        
        print(f"✓ Sampled {len(df_sampled):,} rows from {total_rows:,} ({len(df_sampled)/total_rows*100:.1f}%)")
        
        return df_sampled
    
    def create_interval_dataset(self, df: pl.DataFrame, target_interval: str) -> pl.DataFrame:
        """Create dataset for specific interval"""
        print(f"\n💎 Creating {target_interval} dataset...")
        
        # Get relevant features for this interval
        all_columns = df.columns
        relevant_features = self.get_interval_features(all_columns, target_interval)
        
        print(f"📊 Features: {len(relevant_features)} selected from {len(all_columns)} total")
        
        # Filter columns
        df_filtered = df.select(sorted(list(relevant_features)))
        
        # Downsample data
        df_final = self.downsample_data(df_filtered, target_interval)
        
        # Verify target label exists
        target_label = f"label_{target_interval}"
        if target_label not in df_final.columns:
            print(f"⚠️  Warning: Target label {target_label} not found!")
        else:
            label_stats = df_final.select([
                pl.col(target_label).filter(pl.col(target_label).is_not_null()).mean().alias("positive_ratio"),
                pl.col(target_label).filter(pl.col(target_label).is_not_null()).count().alias("total_count")
            ]).to_dict(as_series=False)
            
            pos_ratio = label_stats["positive_ratio"][0]
            total_count = label_stats["total_count"][0]
            
            if pos_ratio is not None:
                print(f"🎯 Label distribution: {pos_ratio:.1%} positive ({int(total_count * pos_ratio):,}/{total_count:,})")
        
        return df_final
    
    def save_datasets(self, datasets: Dict[str, pl.DataFrame]) -> None:
        """Save all interval datasets"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving datasets to {self.output_dir}")
        
        total_original_size = 0
        total_compressed_size = 0
        
        for interval, df in datasets.items():
            output_file = self.output_dir / f"{interval}.parquet"
            
            print(f"📁 Saving {interval}.parquet...")
            df.write_parquet(output_file, compression="snappy")
            
            file_size = output_file.stat().st_size / (1024*1024)  # MB
            total_compressed_size += file_size
            
            print(f"   ✓ {len(df):,} rows × {len(df.columns)} columns")
            print(f"   📏 File size: {file_size:.1f} MB")
            
            # Show feature breakdown
            feature_types = {}
            for col in df.columns:
                if col.startswith("label_"):
                    feature_types["labels"] = feature_types.get("labels", 0) + 1
                elif any(pattern in col for pattern in self.base_feature_patterns):
                    feature_types["base"] = feature_types.get("base", 0) + 1
                elif f"_{interval}_" in col:
                    feature_types[f"{interval}_features"] = feature_types.get(f"{interval}_features", 0) + 1
                else:
                    # Other interval features
                    for check_interval in self.intervals:
                        if f"_{check_interval}_" in col:
                            feature_types[f"{check_interval}_features"] = feature_types.get(f"{check_interval}_features", 0) + 1
                            break
                    else:
                        feature_types["other"] = feature_types.get("other", 0) + 1
            
            print(f"   📊 Feature breakdown: {feature_types}")
            print()
        
        # Summary
        print("="*60)
        print("💎 DIAMOND LAYER SUMMARY")
        print("="*60)
        print(f"✅ Created {len(datasets)} interval-specific datasets")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Total size: {total_compressed_size:.1f} MB")
        
        # Show file sizes comparison
        print(f"\n📊 Dataset sizes:")
        for interval in self.intervals:
            if interval in datasets:
                df = datasets[interval]
                output_file = self.output_dir / f"{interval}.parquet"
                file_size = output_file.stat().st_size / (1024*1024)
                print(f"   {interval:>3}: {len(df):>7,} rows × {len(df.columns):>4} features = {file_size:>6.1f} MB")
        
        print(f"\n🎯 Ready for model training!")
        print(f"   Each dataset optimized for {', '.join(self.intervals)} trading strategies")
    
    def run(self) -> bool:
        """Run the diamond data creation process"""
        try:
            print("💎 Creating Diamond Layer - Model-Ready Data")
            print("="*60)
            
            # Load and clean data
            df_clean = self.load_data()
            
            # Create datasets for each interval
            datasets = {}
            for interval in self.intervals:
                datasets[interval] = self.create_interval_dataset(df_clean, interval)
            
            # Save all datasets
            self.save_datasets(datasets)
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Create model-ready interval-specific datasets")
    parser.add_argument("--datalake-path", default="datalake", help="Path to datalake directory")
    
    args = parser.parse_args()
    
    creator = DiamondDataCreator(datalake_path=args.datalake_path)
    success = creator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()