"""
Gold Layer - Stage 1: Multi-timeframe analysis

Multi-timeframe analysis: Một hàng chứa tất cả thông tin từ 5m đến 1d
All intervals (5m, 15m, 1h, 4h, 1d) merged by timestamp_dt into single rows.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys
from typing import List, Optional, Dict, Any
import numpy as np


class DataMerger:
    """Multi-timeframe analysis: Một hàng chứa tất cả thông tin từ 5m đến 1d"""
    
    def __init__(self, 
                 datalake_path: str = "datalake"):
        self.datalake_path = Path(datalake_path)
        
        # Multi-timeframe intervals with 5m stepsize alignment
        self.intervals = ["5m", "15m", "1h", "4h", "1d"]
        
        # Fixed symbols for different data types
        self.index_symbol = "btc-usdt"           # For indexPriceKlines
        self.main_symbol = "btc-usdt-swap"       # For orderBook, trades, markPriceKlines
        
        # Build silver sources with correct symbols
        self.silver_sources = {}
        for interval in self.intervals:
            self.silver_sources.update({
                f"orderbook_{interval}": f"2_silver/orderBook/{self.main_symbol}/{interval}",
                f"trades_{interval}": f"2_silver/trades/{self.main_symbol}/{interval}",
                f"index_price_{interval}": f"2_silver/indexPriceKlines/{self.index_symbol}/{interval}",
                f"mark_price_{interval}": f"2_silver/markPriceKlines/{self.main_symbol}/{interval}"
            })
        
        # Bronze funding path
        self.bronze_funding_path = f"1_bronze/fundingRate/{self.main_symbol}"
        self.output_path = "3_gold"
    
    
    def get_file_list(self, path: Path, pattern: str = "*.parquet") -> List[Path]:
        """Get list of parquet files in directory"""
        if not path.exists():
            print(f"Warning: Path {path} does not exist")
            return []
        
        files = list(path.glob(pattern))
        return sorted(files)
    
    def load_silver_data(self, source_name: str) -> Optional[pl.DataFrame]:
        """Load silver layer data for a specific source"""
        source_path = self.datalake_path / self.silver_sources[source_name]
        files = self.get_file_list(source_path)
        
        if not files:
            print(f"Warning: No files found for {source_name} at {source_path}")
            return None
        
        # Read all files and concatenate
        dfs = []
        for file in files:
            try:
                df = pl.read_parquet(file)
                if not df.is_empty():
                    dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        
        if not dfs:
            print(f"Warning: No valid data found for {source_name}")
            return None
        
        # Concatenate and sort by timestamp
        result = pl.concat(dfs, rechunk=True).sort("timestamp_dt")
        
        # Ensure timestamp_dt is in microseconds to match other sources
        timestamp_dtype = result.schema["timestamp_dt"]
        if timestamp_dtype != pl.Datetime(time_unit="us"):
            result = result.with_columns([
                pl.col("timestamp_dt").dt.cast_time_unit("us")
            ])
        
        # Add prefix to column names (except timestamp_dt)
        columns_to_rename = [col for col in result.columns if col != "timestamp_dt"]
        rename_mapping = {col: f"{source_name}_{col}" for col in columns_to_rename}
        result = result.rename(rename_mapping)
        
        print(f"✓ {source_name}: {len(result)} rows")
        return result
    
    def load_bronze_funding_data(self) -> Optional[pl.DataFrame]:
        """Load bronze funding rate data"""
        funding_path = self.datalake_path / self.bronze_funding_path
        files = self.get_file_list(funding_path)
        
        if not files:
            print(f"Warning: No funding rate files found at {funding_path}")
            return None
        
        # Read all files and concatenate
        dfs = []
        for file in files:
            try:
                df = pl.read_parquet(file)
                if not df.is_empty():
                    dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
        
        if not dfs:
            print("Warning: No valid funding rate data found")
            return None
        
        # Concatenate and process
        result = pl.concat(dfs, rechunk=True)
        
        # Convert funding_time to timestamp_dt for merging (match silver data microseconds)
        result = result.with_columns([
            pl.from_epoch(pl.col("funding_time"), time_unit="ms").dt.cast_time_unit("us").alias("timestamp_dt")
        ])
        
        # Sort and add prefix
        result = result.sort("timestamp_dt")
        columns_to_rename = [col for col in result.columns if col != "timestamp_dt"]
        rename_mapping = {col: f"funding_{col}" for col in columns_to_rename}
        result = result.rename(rename_mapping)
        
        print(f"✓ funding: {len(result)} rows")
        return result
    
    def merge_all_data(self) -> Optional[pl.DataFrame]:
        """Merge all silver and bronze funding data by timestamp"""
        print("Loading data sources...")
        
        # Load all data sources
        data_sources = {}
        
        # Load silver layer data
        for source_name in self.silver_sources.keys():
            df = self.load_silver_data(source_name)
            if df is not None:
                data_sources[source_name] = df
        
        # Load bronze funding data
        funding_df = self.load_bronze_funding_data()
        if funding_df is not None:
            data_sources["funding"] = funding_df
        
        if not data_sources:
            print("Error: No data sources available for merging")
            return None
        
        print(f"\nMerging {len(data_sources)} data sources...")
        
        # Start with the first available dataset
        first_key = list(data_sources.keys())[0]
        merged_df = data_sources[first_key]
        
        # Merge remaining datasets
        for source_name, df in list(data_sources.items())[1:]:
            merged_df = merged_df.join(
                df,
                on="timestamp_dt",
                how="full",
                suffix="_temp"
            )
            
            # Remove any duplicate timestamp columns that might have been created
            if "timestamp_dt_temp" in merged_df.columns:
                merged_df = merged_df.drop("timestamp_dt_temp")
        
        # Sort final result by timestamp
        merged_df = merged_df.sort("timestamp_dt")
        
        # Fill forward for funding rate data (funding rates don't change frequently)
        funding_columns = [col for col in merged_df.columns if col.startswith("funding_")]
        if funding_columns:
            merged_df = merged_df.with_columns([
                pl.col(col).forward_fill() for col in funding_columns
            ])
        
        print(f"✓ Merged: {len(merged_df)} rows, {len(merged_df.columns)} columns")
        return merged_df
    
    def save_merged_data(self, df: pl.DataFrame) -> None:
        """Save merged data to single parquet file"""
        output_dir = self.datalake_path / self.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "step_1.parquet"
        
        df.write_parquet(output_file, compression="snappy")
        
        print(f"\n✅ Success! Saved {len(df):,} rows to: {output_file}")
        
        # Show time range
        if not df.is_empty():
            min_time = df.select(pl.col("timestamp_dt").min()).item()
            max_time = df.select(pl.col("timestamp_dt").max()).item()
            print(f"📅 Time range: {min_time} to {max_time}")
            print(f"📊 Columns: {len(df.columns)}")
            print(f"💾 File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    
    def run(self) -> bool:
        """Run the merge process"""
        try:
            print("🚀 Multi-timeframe data merge")
            
            merged_df = self.merge_all_data()
            
            if merged_df is None or merged_df.is_empty():
                print("❌ Error: No data to merge")
                return False
            
            self.save_merged_data(merged_df)
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Multi-timeframe analysis: Merge tất cả symbols & intervals vào single file")
    parser.add_argument("--datalake-path", default="datalake", help="Path to datalake directory")
    
    args = parser.parse_args()
    
    merger = DataMerger(datalake_path=args.datalake_path)
    
    success = merger.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()