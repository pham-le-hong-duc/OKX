import asyncio
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import okx.api.public as PublicData

class RestAPI:
    
    def __init__(self, symbol="BTC-USD-SWAP", base_data_path="../../../datalake/1_bronze", buffer_size=2000, base_start_date="2025-01-01"):
        self.symbol = symbol
        self.base_data_path = Path(base_data_path)
        self.publicAPI = PublicData.Public(flag="0")
        self.buffer_size = buffer_size
        self.base_start_date = base_start_date
        
        # Gap thresholds for each interval (in milliseconds) - 1.5x interval
        self.gap_thresholds = {
            "1m": int(1.5 * 60 * 1000),           # 1.5 minutes
            "5m": int(1.5 * 5 * 60 * 1000),       # 7.5 minutes  
            "15m": int(1.5 * 15 * 60 * 1000),     # 22.5 minutes
            "1H": int(1.5 * 60 * 60 * 1000),      # 1.5 hours
            "4H": int(1.5 * 4 * 60 * 60 * 1000),  # 6 hours
            "1D": int(1.5 * 24 * 60 * 60 * 1000)  # 36 hours
        }
    
    def detect_gaps(self, interval):
        # Get threshold for current interval
        gap_threshold = self.gap_thresholds.get(interval, 60000)
        
        output_path = self.base_data_path / "markPriceKlines" / self.symbol.lower() / interval
        files = sorted(output_path.glob("*.parquet"))
        gaps = []
        
        # If no files exist, skip gap detection
        if not files:
            return gaps
        
        # Check gap from base_start_date to first file
        first_file_df = pl.scan_parquet(files[0]).select(pl.col("open_time").min()).collect()
        first_file_start = first_file_df["open_time"][0]
        base_start_ms = int(datetime.strptime(self.base_start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        if base_start_ms < first_file_start:
            gap_duration = first_file_start - base_start_ms
            # Only create pre-history gap if duration exceeds threshold
            if gap_duration > gap_threshold:
                gaps.append({
                    'type': 'pre_history',
                    'start': base_start_ms,
                    'end': first_file_start - 1,
                    'duration_hours': gap_duration / (1000 * 3600),
                    'description': f"{self.base_start_date} → {files[0].name}",
                })
        
        gaps.extend(self._detect_boundary_gaps(files, gap_threshold))
        gaps.extend(self._detect_internal_gaps(files, gap_threshold))
        return gaps
    
    def _detect_boundary_gaps(self, files, gap_threshold):
        gaps = []
        for i in range(len(files) - 1):
            current_df = pl.scan_parquet(files[i]).select(pl.col("open_time").max()).collect()
            next_df = pl.scan_parquet(files[i + 1]).select(pl.col("open_time").min()).collect()
            
            current_end = current_df["open_time"][0]
            next_start = next_df["open_time"][0]
            
            gap_duration = next_start - current_end
            if gap_duration > gap_threshold:
                gaps.append({
                    'type': 'boundary',
                    'start': current_end + 1,
                    'end': next_start - 1,
                    'duration_hours': gap_duration / (1000 * 3600),
                    'between_files': f"{files[i].name} → {files[i+1].name}"
                })
        return gaps
    
    def _detect_internal_gaps(self, files, gap_threshold):
        gaps = []
        for file in files:
            file_gaps = self._check_single_file_gaps(file, gap_threshold)
            gaps.extend(file_gaps)
        return gaps
    
    def _check_single_file_gaps(self, file_path, gap_threshold):
        try:
            df = pl.read_parquet(file_path, columns=["open_time"])
            
            if len(df) > 10_000_000:
                df = df.sample(n=50_000)
            
            timestamps = df["open_time"].sort().to_list()
            gaps = []
            
            for i in range(len(timestamps) - 1):
                gap_duration = timestamps[i+1] - timestamps[i]
                if gap_duration > gap_threshold:
                    gaps.append({
                        'type': 'internal',
                        'start': timestamps[i] + 1,
                        'end': timestamps[i+1] - 1,
                        'duration_hours': gap_duration / (1000 * 3600),
                        'inside_file': file_path.name
                    })
            
            return gaps
            
        except:
            return []
    
    def show_gaps(self, gaps):
        if not gaps:
            print("No gaps found")
            return
        
        # Count different types of gaps
        pre_history_count = len([g for g in gaps if g['type'] == 'pre_history']) 
        boundary_count = len([g for g in gaps if g['type'] == 'boundary'])
        internal_count = len([g for g in gaps if g['type'] == 'internal'])
        
        if pre_history_count > 0:
            print(f"Pre-history gaps: {pre_history_count}")
        if boundary_count > 0:
            print(f"Boundary gaps: {boundary_count}")
        if internal_count > 0:
            print(f"Internal gaps: {internal_count}")
        
        for i, gap in enumerate(gaps):
            start_dt = datetime.fromtimestamp(gap['start'] / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(gap['end'] / 1000, tz=timezone.utc)
            
            print(f"\nGap {i+1}/{len(gaps)} ({gap['type']}):")
            print(f"    Time: {start_dt} → {end_dt}")
            print(f"    Duration: {gap['duration_hours']:.2f} hours")            
            if gap['type'] == 'boundary':
                print(f"    Between files: {gap['between_files']}")
            elif gap['type'] in ['initial', 'pre_history']:
                print(f"    Description: {gap['description']}")
            else:
                print(f"    Inside file: {gap['inside_file']}")
    
    async def fill_gaps(self, gaps, interval):
        if not gaps:
            return 0
            
        total_filled = 0
        for i, gap in enumerate(gaps):
            print(f"\nFilling gap {i+1}/{len(gaps)}:")
            
            filled_count = await self._fetch_gap_data(gap['start'], gap['end'], interval)
            if not filled_count:
                print("\tNo data found")
                continue
            
            print(f"    Completed: {filled_count} klines filled")
            total_filled += filled_count
        
        return total_filled
    
    async def _fetch_gap_data(self, start_ms, end_ms, interval):
        klines_buffer = []
        current_after = str(end_ms)
        total_saved = 0
        
        while True:            
            try:
                # Check for cancellation
                if asyncio.current_task().cancelled():
                    break
                    
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.publicAPI.get_history_mark_price_candles,
                    self.symbol, current_after, "", interval, "100"
                )
                
                if result.get('code') != '0' or not result.get('data'):
                    break
                    
                batch = result['data']
                min_ts = None
                
                for k in batch:
                    ts = int(k[0])  # timestamp is first element in kline array
                    if start_ms <= ts <= end_ms:
                        klines_buffer.append({
                            'open_time': ts,
                            'open': float(k[1]),
                            'high': float(k[2]),
                            'low': float(k[3]),
                            'close': float(k[4])
                        })
                    
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                
                if len(klines_buffer) >= self.buffer_size:
                    saved_count = self._save_gap_data(klines_buffer, interval)
                    total_saved += saved_count
                    klines_buffer.clear()
                
                if min_ts and min_ts < start_ms:
                    break
                    
                current_after = str(min_ts)
                
                # Small delay to allow interruption
                await asyncio.sleep(0.01)
                
            except KeyboardInterrupt:
                print("Fetching interrupted by user")
                break
            except asyncio.CancelledError:
                print("Fetching cancelled")
                break
            except Exception as e:
                print(f"API error: {e}")
                break
        
        if klines_buffer:
            saved_count = self._save_gap_data(klines_buffer, interval)
            total_saved += saved_count
        
        return total_saved
    
    def _save_gap_data(self, klines, interval):
        klines_by_date = {}
        for kline in klines:
            date_str = datetime.fromtimestamp(kline['open_time'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            klines_by_date.setdefault(date_str, []).append(kline)
        
        total_saved = 0
        for date_str, day_klines in klines_by_date.items():
            self._save_to_daily_file(date_str, day_klines, interval)
            total_saved += len(day_klines)
        
        return total_saved
    
    def _save_to_daily_file(self, date_str, klines, interval):
        df = pl.DataFrame(klines).sort("open_time")
        output_path = self.base_data_path / "markPriceKlines" / self.symbol.lower() / interval
        file_path = output_path / f"{date_str}.parquet"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df]).unique(subset=["open_time"], keep="last").sort("open_time")
            combined_df.write_parquet(file_path, compression="snappy")
        else:
            df.write_parquet(file_path, compression="snappy")
        
        print(f"    {len(klines)} → {date_str}.parquet")
    
    async def run_single_interval(self, interval):
        """Run gap detection and filling for current interval"""
        gaps = self.detect_gaps(interval)
        self.show_gaps(gaps)
        
        if gaps:
            await self.fill_gaps(gaps, interval)
    
    async def run(self):
        """Run gap detection and filling for all intervals"""
        intervals = list(self.gap_thresholds.keys())
        
        for interval in intervals:
            print(f"\n{'='*60}")
            print(f"Processing {self.symbol} - {interval}")
            print(f"{'='*60}")
            
            # Ensure output directory exists
            output_path = self.base_data_path / "markPriceKlines" / self.symbol.lower() / interval
            output_path.mkdir(parents=True, exist_ok=True)
            
            await self.run_single_interval(interval)
            
        print(f"\n{'='*60}")
        print("Completed all intervals!")
        print(f"{'='*60}")

if __name__ == "__main__":
    try:
        asyncio.run(RestAPI().run())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user (Ctrl+C)")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
