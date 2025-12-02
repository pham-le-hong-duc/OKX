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

class MarkPriceKlinesAggregator:
    """
    MarkPriceKlines aggregator using Pre-sort + Binary Search + Sliding Window
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
        
        # Map column names if needed
        if "open_time" in schema.names():
            df_lazy = df_lazy.with_columns(pl.col("open_time").alias("timestamp"))
        
        return df_lazy.sort("timestamp").collect()
    
    def create_time_index(self, df_sorted: pl.DataFrame) -> np.ndarray:
        """Create numpy array for binary search"""
        return df_sorted["timestamp"].to_numpy()
    
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

    def aggregate_window_data(self, window_data: pl.DataFrame, window_end_time: int) -> dict:
        """Aggregate window data with OHLC features"""
        if window_data is None or len(window_data) == 0:
            return {"timestamp_dt": datetime.fromtimestamp(window_end_time/1000)}
        
        try:
            agg_result = window_data.select([
                # Reconstructed OHLC
                pl.col("open").first().alias("open"),       # Giá mở cửa window
                pl.col("high").max().alias("high"),         # Giá cao nhất window
                pl.col("low").min().alias("low"),           # Giá thấp nhất window
                pl.col("close").last().alias("close"),      # Giá đóng cửa window
                
                # Stats
                pl.col("close").mean().alias("mean"),       # Giá trung bình
                pl.col("close").std().alias("std"),         # Độ biến động
            ])
            
            result = agg_result.to_dict(as_series=False)
            result["timestamp_dt"] = datetime.fromtimestamp(window_end_time/1000)
            
            # Extract single values from lists
            for key, value in result.items():
                if isinstance(value, list) and len(value) == 1:
                    result[key] = value[0]
                    
            return result
            
        except Exception as e:
            return {"timestamp_dt": datetime.fromtimestamp(window_end_time/1000)}
    
    def process_window(self, df_sorted: pl.DataFrame, timestamps: np.ndarray, window_end_time: int) -> dict:
        """Process single window: binary search + aggregate"""
        window_data = self.get_window_data_slice(df_sorted, timestamps, window_end_time)
        return self.aggregate_window_data(window_data, window_end_time)

    def run_pipeline(self, df_prepared: pl.LazyFrame, target_date: datetime = None) -> Optional[pl.DataFrame]:
        """
        Execute optimal aggregation pipeline
        """
        df_sorted = None
        timestamps = None
        try:
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
            
            # Filter valid results (has OHLC data)
            valid_results = [r for r in results if "open" in r and r["open"] is not None]
            
            # Create DataFrame
            final_df = pl.DataFrame(valid_results).sort("timestamp_dt")
            
            return final_df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        finally:
            del df_sorted, timestamps
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
    """Process single markPriceKlines file for all intervals"""
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
                aggregator = MarkPriceKlinesAggregator(interval=interval, step_size=step_size, max_workers=4)
                result_df = aggregator.run_pipeline(df_data, target_date=target_date)
                
                if result_df is not None and len(result_df) > 0:
                    output_dir_path = output_dir / "markPriceKlines" / symbol / interval
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


def auto_process_markpriceklines():
    """Auto discover and process all markPriceKlines files in bronze"""
    
    # Auto-detect paths
    bronze_mark_dir = Path("datalake/1_bronze/markPriceKlines/btc-usd-swap/1m")
    silver_output_dir = Path("datalake/2_silver")
    symbol = "btc-usd-swap"
    
    print("🎯 AUTO PROCESSING MARKPRICEKLINES FILES")
    print(f"📂 {bronze_mark_dir}")
    print("=" * 50)
    
    # Check if input directory exists
    if not bronze_mark_dir.exists():
        print(f"❌ Bronze directory not found: {bronze_mark_dir}")
        return False
    
    # Get all parquet files
    parquet_files = sorted(list(bronze_mark_dir.glob("*.parquet")))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {bronze_mark_dir}")
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
    success = auto_process_markpriceklines()
    
    if success:
        print("\n🏁 All markPriceKlines files processed successfully!")
    else:
        print("\n❌ Some files failed to process!")
        sys.exit(1)