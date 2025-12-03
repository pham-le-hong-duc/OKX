import polars as pl
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import gc
import sys

class OrderBookAggregator:
    """
    OrderBook Aggregator using Pre-sort + Binary Search + Sliding Window
    """
    @staticmethod
    def _parse_interval_to_ms(interval: str) -> int:
        """Convert interval string to milliseconds"""
        unit = interval[-1].lower()
        try:
            value = int(interval[:-1])
        except ValueError:
            return 60 * 1000 

        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        else:
            return 60 * 1000

    def __init__(self, interval: str = "5m", step_size: str = "5m", max_workers: int = None):
        self.interval = interval
        self.step_size = step_size
        self.interval_ms = self._parse_interval_to_ms(interval)
        self.step_ms = self._parse_interval_to_ms(step_size)
        self.max_workers = max_workers or mp.cpu_count()
    
    def load_and_sort_data(self, df_lazy: pl.LazyFrame) -> pl.DataFrame:
        """Load and sort data once"""
        schema = df_lazy.collect_schema()
        
        # Map column names
        if "created_time" not in schema.names() and "ts" in schema.names():
            df_lazy = df_lazy.with_columns([
                pl.col("ts").alias("created_time"),
                pl.col("ts").alias("time")
            ])
        
        return df_lazy.sort("created_time").collect()
    
    def create_time_index(self, df_sorted: pl.DataFrame) -> np.ndarray:
        """Create numpy array for binary search"""
        return df_sorted["created_time"].to_numpy()
    
    def create_windows_for_date(self, target_date: datetime) -> list:
        """Generate window timestamps for target date (UTC)"""
        from datetime import timezone
        
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        next_day = target_date + timedelta(days=1)
        end_of_day = next_day.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        
        start_ms = int(start_of_day.timestamp() * 1000)
        end_ms = int(end_of_day.timestamp() * 1000)
        
        windows = []
        current_window_end = start_ms + self.step_ms
        
        while current_window_end <= end_ms:
            windows.append(current_window_end)
            current_window_end += self.step_ms
            
        return windows
    
    def get_window_data_slice(self, df_sorted: pl.DataFrame, timestamps: np.ndarray, window_end_time: int):
        """Get window data using binary search"""
        window_start_time = window_end_time - self.interval_ms
        
        start_idx = np.searchsorted(timestamps, window_start_time, side='left')
        end_idx = np.searchsorted(timestamps, window_end_time, side='left')
        
        if start_idx >= end_idx:
            return None
            
        return df_sorted[start_idx:end_idx]

    def prepare_features(self, df_raw: pl.LazyFrame) -> pl.LazyFrame:
        """Prepare OrderBook features - filter for snapshot action only"""
        schema = df_raw.collect_schema()
        
        # Filter for snapshot action only
        if "action" in schema.names():
            df_raw = df_raw.filter(pl.col("action") == "snapshot")
        
        # Map column names
        if "time" not in schema.names() and "ts" in schema.names():
            df_raw = df_raw.with_columns([
                pl.col("ts").alias("time"),
                pl.col("ts").alias("created_time")
            ])
        
        # Parse real orderbook features from JSON string bid/ask data
        df_features = df_raw.with_columns([
            pl.from_epoch(pl.col("time"), time_unit="ms").alias("timestamp_dt"),
            
            # Parse JSON strings to arrays and cast to Float64
            pl.col("bids").str.json_decode(dtype=pl.List(pl.List(pl.String))).list.eval(
                pl.element().list.eval(pl.element().cast(pl.Float64))
            ).alias("bids_float"),
            pl.col("asks").str.json_decode(dtype=pl.List(pl.List(pl.String))).list.eval(
                pl.element().list.eval(pl.element().cast(pl.Float64))  
            ).alias("asks_float"),
        ]).with_columns([
            # Parse orderbook levels from float arrays
            pl.col("bids_float").list.first().list.first().alias("best_bid"),  # bids[0][0] = best bid price
            pl.col("asks_float").list.first().list.first().alias("best_ask"),  # asks[0][0] = best ask price
            
            # Sum quantities for depth analysis (safely handle missing data)
            pl.col("bids_float").list.slice(0, 50).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_bid_50"),
            pl.col("asks_float").list.slice(0, 50).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_ask_50"),
            pl.col("bids_float").list.slice(0, 5).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_bid_5"),
            pl.col("asks_float").list.slice(0, 5).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_ask_5"),
            pl.col("bids_float").list.slice(0, 20).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_bid_20"),
            pl.col("asks_float").list.slice(0, 20).list.eval(
                pl.element().list.get(1).fill_null(0.0)
            ).list.sum().alias("sum_ask_20"),
            
            # Price at depth levels (safely handle out of bounds)
            pl.when(pl.col("bids_float").list.len() > 19)
            .then(pl.col("bids_float").list.get(19).list.first())
            .otherwise(pl.col("bids_float").list.first().list.first()).alias("bid_px_20"),
            pl.when(pl.col("asks_float").list.len() > 19)
            .then(pl.col("asks_float").list.get(19).list.first())
            .otherwise(pl.col("asks_float").list.first().list.first()).alias("ask_px_20"),
            
            # Total money (price * quantity for each level, then sum)
            pl.col("bids_float").list.slice(0, 50).list.eval(
                (pl.element().list.get(0).fill_null(0.0) * pl.element().list.get(1).fill_null(0.0))
            ).list.sum().alias("total_money_bid_50"),
            pl.col("asks_float").list.slice(0, 50).list.eval(
                (pl.element().list.get(0).fill_null(0.0) * pl.element().list.get(1).fill_null(0.0))
            ).list.sum().alias("total_money_ask_50"),
        ]).with_columns([
            # Calculate total money (bid + ask)
            (pl.col("total_money_bid_50") + pl.col("total_money_ask_50")).alias("total_money_50")
        ])

        # Calculate indicators with safe division (add epsilon to prevent divide by zero)
        return df_features.with_columns([
            # Spread
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            
            # Weighted Mid Price (safe division)
            (pl.col("total_money_50") / (pl.col("sum_bid_50") + pl.col("sum_ask_50") + 1e-9)).alias("weighted_mid_price"),
            
            # Deep Imbalance Ratio (safe division)
            (pl.col("sum_bid_50") / (pl.col("sum_bid_50") + pl.col("sum_ask_50") + 1e-9)).alias("deep_imbalance"),
            
            # Total Depth 50
            (pl.col("sum_bid_50") + pl.col("sum_ask_50")).alias("total_depth_50"),
            
            # Liquidity Concentration (safe division)
            ((pl.col("sum_bid_5") + pl.col("sum_ask_5")) / (pl.col("sum_bid_50") + pl.col("sum_ask_50") + 1e-9)).alias("concentration"),
            
            # Price Impact Slope (safe division)
            ((pl.col("ask_px_20") - pl.col("best_ask")) / (pl.col("sum_ask_20") + 1e-9)).alias("impact_slope_ask"),
            ((pl.col("best_bid") - pl.col("bid_px_20")) / (pl.col("sum_bid_20") + 1e-9)).alias("impact_slope_bid"),
            
            # Book Pressure (safe division)
            ((pl.col("total_money_50") / (pl.col("sum_bid_50") + pl.col("sum_ask_50") + 1e-9)) - 
             ((pl.col("best_ask") + pl.col("best_bid")) / 2)).alias("book_pressure")
        ])

    def aggregate_window_data(self, window_data: pl.DataFrame, window_end_time: int) -> dict:
        """Aggregate OrderBook window data with all features preserved"""
        if window_data is None or len(window_data) == 0:
            return {"timestamp_dt": datetime.fromtimestamp(window_end_time/1000), "snapshot_count": 0}
        
        try:
            agg_result = window_data.select([
                # Weighted Mid Price
                pl.col("weighted_mid_price").mean().alias("wmp_mean"),
                pl.col("weighted_mid_price").std().alias("wmp_std"),
                pl.col("weighted_mid_price").min().alias("wmp_min"),
                pl.col("weighted_mid_price").quantile(0.25).alias("wmp_0.25"),
                pl.col("weighted_mid_price").quantile(0.50).alias("wmp_0.50"),
                pl.col("weighted_mid_price").quantile(0.75).alias("wmp_0.75"),
                pl.col("weighted_mid_price").max().alias("wmp_max"),
                pl.col("weighted_mid_price").first().alias("wmp_first"),
                pl.col("weighted_mid_price").last().alias("wmp_last"),
                pl.col("weighted_mid_price").skew().alias("wmp_skew"),
                pl.col("weighted_mid_price").kurtosis().alias("wmp_kurtosis"),

                # Spread
                pl.col("spread").mean().alias("spread_mean"),
                pl.col("spread").std().alias("spread_std"),
                pl.col("spread").min().alias("spread_min"),
                pl.col("spread").quantile(0.25).alias("spread_0.25"),
                pl.col("spread").quantile(0.50).alias("spread_0.50"),
                pl.col("spread").quantile(0.75).alias("spread_0.75"),
                pl.col("spread").max().alias("spread_max"),
                pl.col("spread").first().alias("spread_first"),
                pl.col("spread").last().alias("spread_last"),
                pl.col("spread").skew().alias("spread_skew"),
                pl.col("spread").kurtosis().alias("spread_kurtosis"),

                # Imbalance
                pl.col("deep_imbalance").mean().alias("imbal_mean"),
                pl.col("deep_imbalance").std().alias("imbal_std"),
                pl.col("deep_imbalance").min().alias("imbal_min"),
                pl.col("deep_imbalance").quantile(0.25).alias("imbal_0.25"),
                pl.col("deep_imbalance").quantile(0.50).alias("imbal_0.50"),
                pl.col("deep_imbalance").quantile(0.75).alias("imbal_0.75"),
                pl.col("deep_imbalance").max().alias("imbal_max"),
                pl.col("deep_imbalance").first().alias("imbal_first"),
                pl.col("deep_imbalance").last().alias("imbal_last"),
                pl.col("deep_imbalance").skew().alias("imbal_skew"),
                pl.col("deep_imbalance").kurtosis().alias("imbal_kurtosis"),

                # Depth & Concentration
                pl.col("total_depth_50").mean().alias("depth_mean"),
                pl.col("total_depth_50").std().alias("depth_std"),
                pl.col("total_depth_50").min().alias("depth_min"),
                pl.col("total_depth_50").max().alias("depth_max"),
                pl.col("total_depth_50").first().alias("depth_first"),
                pl.col("total_depth_50").last().alias("depth_last"),
                
                pl.col("concentration").mean().alias("conc_mean"),
                pl.col("concentration").std().alias("conc_std"),
                pl.col("concentration").min().alias("conc_min"),
                pl.col("concentration").max().alias("conc_max"),

                # Impact Slope
                pl.col("impact_slope_ask").mean().alias("impact_ask_mean"),
                pl.col("impact_slope_ask").std().alias("impact_ask_std"),
                pl.col("impact_slope_ask").min().alias("impact_ask_min"),
                pl.col("impact_slope_ask").max().alias("impact_ask_max"),
                
                pl.col("impact_slope_bid").mean().alias("impact_bid_mean"),
                pl.col("impact_slope_bid").std().alias("impact_bid_std"),
                pl.col("impact_slope_bid").min().alias("impact_bid_min"),
                pl.col("impact_slope_bid").max().alias("impact_bid_max"),

                # Book Pressure
                pl.col("book_pressure").sum().alias("pressure_sum"),
                pl.col("book_pressure").mean().alias("pressure_mean"),
                pl.col("book_pressure").std().alias("pressure_std"),
                pl.col("book_pressure").min().alias("pressure_min"),
                pl.col("book_pressure").quantile(0.25).alias("pressure_0.25"),
                pl.col("book_pressure").quantile(0.50).alias("pressure_0.50"),
                pl.col("book_pressure").quantile(0.75).alias("pressure_0.75"),
                pl.col("book_pressure").max().alias("pressure_max"),
                pl.col("book_pressure").first().alias("pressure_first"),
                pl.col("book_pressure").last().alias("pressure_last"),
                pl.col("book_pressure").skew().alias("pressure_skew"),
                pl.col("book_pressure").kurtosis().alias("pressure_kurtosis"),

                # Count & Rate
                pl.len().alias("snapshot_count"),
                pl.col("time").diff().mean().alias("rate_mean_ms"),
                pl.col("time").diff().std().alias("rate_std_ms"),
                pl.col("time").diff().min().alias("rate_min_ms"),
                pl.col("time").diff().max().alias("rate_max_ms"),

                # Correlations
                pl.corr("weighted_mid_price", "deep_imbalance").alias("corr_wmp_imbal"),
                pl.corr("weighted_mid_price", "total_depth_50").alias("corr_wmp_depth"),
                pl.corr("weighted_mid_price", "book_pressure").alias("corr_wmp_pressure"),
                pl.corr("weighted_mid_price", (pl.col("time").max() - pl.col("time"))).alias("corr_wmp_time"),
                pl.corr("spread", "deep_imbalance").alias("corr_spread_imbal"),
                pl.corr("spread", (pl.col("impact_slope_ask") + pl.col("impact_slope_bid"))/2).alias("corr_spread_impact"),
                pl.corr("concentration", (pl.col("time").max() - pl.col("time"))).alias("corr_conc_time"),
                pl.corr("book_pressure", (pl.col("time").max() - pl.col("time"))).alias("corr_pressure_time"),

                # Meta
                pl.col("time").max().alias("last_update_time_ms")
            ])
            
            result = agg_result.to_dict(as_series=False)
            result["timestamp_dt"] = datetime.fromtimestamp(window_end_time/1000)
            
            # Extract single values from lists
            for key, value in result.items():
                if isinstance(value, list) and len(value) == 1:
                    result[key] = value[0]
                    
            return result
            
        except Exception as e:
            return {"timestamp_dt": datetime.fromtimestamp(window_end_time/1000), "snapshot_count": 0}
    
    def process_window(self, df_sorted: pl.DataFrame, timestamps: np.ndarray, window_end_time: int) -> dict:
        """Process single window: binary search + aggregate"""
        window_data = self.get_window_data_slice(df_sorted, timestamps, window_end_time)
        return self.aggregate_window_data(window_data, window_end_time)

    def run_pipeline(self, df_raw: pl.LazyFrame, target_date: datetime = None) -> Optional[pl.DataFrame]:
        """
        Execute optimal OrderBook aggregation pipeline
        """
        df_prepared = None
        df_sorted = None
        timestamps = None
        try:
            # Prepare features
            df_prepared = self.prepare_features(df_raw)
            
            # Load and sort data
            df_sorted = self.load_and_sort_data(df_prepared)
            
            # Create time index
            timestamps = self.create_time_index(df_sorted)
            
            # Generate windows for target date
            window_times = self.create_windows_for_date(target_date)
            
            # Process windows with threading
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(
                    lambda w: self.process_window(df_sorted, timestamps, w), 
                    window_times
                ))
            
            # Filter valid results
            valid_results = [r for r in results if r.get("snapshot_count", 0) > 0]
            
            # Create DataFrame
            final_df = pl.DataFrame(valid_results).sort("timestamp_dt")
            
            return final_df

        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        finally:
            del df_prepared, df_sorted, timestamps
            gc.collect()


def get_previous_file(current_file: Path, all_files: list) -> Optional[Path]:
    """Find the previous day's file"""
    current_date = datetime.strptime(current_file.stem, "%Y-%m-%d")
    previous_date = current_date - timedelta(days=1)
    previous_filename = f"{previous_date.strftime('%Y-%m-%d')}.parquet"
    
    for file_path in all_files:
        if file_path.name == previous_filename:
            return file_path
    
    return None

def load_concat_data(current_file: Path, previous_file: Optional[Path] = None) -> pl.LazyFrame:
    """Load and concat current + previous day data"""
    try:
        if previous_file is None or not previous_file.exists():
            return pl.scan_parquet(str(current_file))
        else:
            print(f"   📂 + {previous_file.name}")
            df_prev = pl.scan_parquet(str(previous_file))
            df_curr = pl.scan_parquet(str(current_file))
            return pl.concat([df_prev, df_curr], how="vertical")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def process_single_file(input_file: Path, output_dir: Path, symbol: str, all_files: list):
    """Process single orderbook file for all intervals"""
    intervals = ["5m", "15m", "1h", "4h", "1d"]
    step_size = "5m"
    
    # Parse target date from filename
    try:
        target_date = datetime.strptime(input_file.stem, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Cannot parse date from filename: {input_file}")
        return False
    
    print(f"🚀 {input_file.name}")
    
    try:
        # Get previous file for concat
        previous_file = get_previous_file(input_file, all_files)
        
        # Load data with concat
        df_data = load_concat_data(input_file, previous_file)
        if df_data is None:
            print(f"❌ Failed to load data")
            return False
            
        processed_count = 0
        
        for interval in intervals:
            try:
                aggregator = OrderBookAggregator(interval=interval, step_size=step_size, max_workers=4)
                result_df = aggregator.run_pipeline(df_data, target_date=target_date)
                
                if result_df is not None and len(result_df) > 0:
                    output_dir_path = output_dir / "orderBook" / symbol / interval
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    
                    output_file = output_dir_path / input_file.name
                    result_df.write_parquet(str(output_file))
                    
                    print(f"   ✅ {interval}: {result_df.shape[0]} windows")
                    processed_count += 1
                else:
                    print(f"   ❌ {interval}: No data")
                    
            except Exception as e:
                print(f"   ❌ {interval}: {e}")
            finally:
                del aggregator, result_df
                gc.collect()
        
        return processed_count > 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return False


def auto_process_orderbook():
    """Auto discover and process all orderbook files in bronze"""
    
    # Auto-detect paths
    bronze_orderbook_dir = Path("datalake/1_bronze/orderBook/btc-usdt-swap")
    silver_output_dir = Path("datalake/2_silver")
    symbol = "btc-usdt-swap"
    
    print("🎯 AUTO PROCESSING ORDERBOOK FILES")
    print(f"📂 {bronze_orderbook_dir}")
    print("=" * 50)
    
    # Check if input directory exists
    if not bronze_orderbook_dir.exists():
        print(f"❌ Bronze directory not found: {bronze_orderbook_dir}")
        return False
    
    # Get all parquet files
    parquet_files = sorted(list(bronze_orderbook_dir.glob("*.parquet")))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {bronze_orderbook_dir}")
        return False
    
    print(f"📋 Found {len(parquet_files)} files")
    
    # Process each file
    success_count = 0
    start_time = time.time()
    
    for i, file_path in enumerate(parquet_files, 1):
        print(f"\n[{i}/{len(parquet_files)}]", end=" ")
        
        # Use multiprocessing to ensure complete memory cleanup
        p = mp.Process(target=_worker_wrapper, args=(file_path, silver_output_dir, symbol, parquet_files))
        p.start()
        p.join()  # Wait for process to completely finish and release RAM
        
        if p.exitcode == 0:
            success_count += 1
        
        # Progress update
        if i % 10 == 0 or i == len(parquet_files):
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(parquet_files) - i) * avg_time
            print(f"⏱️ {success_count}/{i} OK | {remaining:.0f}s left")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n🎉 COMPLETED: {success_count}/{len(parquet_files)} files | {total_time:.1f}s")
    
    return success_count == len(parquet_files)


def _worker_wrapper(file_path, output_dir, symbol, all_files):
    """Worker wrapper to isolate process and ensure complete memory cleanup"""
    try:
        return process_single_file(file_path, output_dir, symbol, all_files)
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        return False


if __name__ == "__main__":
    success = auto_process_orderbook()
    
    if success:
        print("\n🏁 All orderbook files processed successfully!")
    else:
        print("\n❌ Some files failed to process!")
        sys.exit(1)