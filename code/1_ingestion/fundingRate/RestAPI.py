import asyncio
import polars as pl
from pathlib import Path
from datetime import datetime, timezone, timedelta
import okx.api.public as PublicData

class RestAPI:
    
    def __init__(self, symbol="BTC-USDT-SWAP", base_start_date="2025-01-01", base_data_path="../../../datalake/1_bronze", buffer_size=2000):
        self.symbol = symbol
        self.output_path = Path(base_data_path) / "fundingRate" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.publicAPI = PublicData.Public(flag="0")  # Production trading
        self.buffer_size = buffer_size
        
        # Gap threshold for funding rate (1.5 * 8 hours = 12 hours in milliseconds)
        # Funding rate updates every 8 hours, so 1.5x = 12 hours threshold
        self.gap_threshold = int(1.5 * 8 * 60 * 60 * 1000)
    
    def detect_gaps(self):
        files = sorted(self.output_path.glob("*.parquet"))
        if not files:
            return []
        
        gaps = []
        gaps.extend(self._detect_boundary_gaps(files, self.gap_threshold))
        gaps.extend(self._detect_internal_gaps(files, self.gap_threshold))
        return gaps
    
    def _detect_boundary_gaps(self, files, gap_threshold):
        gaps = []
        for i in range(len(files) - 1):
            current_df = pl.scan_parquet(files[i]).select(pl.col("funding_time").max()).collect()
            next_df = pl.scan_parquet(files[i + 1]).select(pl.col("funding_time").min()).collect()
            
            current_end = current_df["funding_time"][0]
            next_start = next_df["funding_time"][0]
            
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
            df = pl.read_parquet(file_path, columns=["funding_time"])
            
            if len(df) > 10_000_000:
                df = df.sample(n=50_000)
            
            timestamps = df["funding_time"].sort().to_list()
            gaps = []
            
            for i in range(len(timestamps) - 1):
                gap_duration = timestamps[i+1] - timestamps[i]
                if gap_duration > gap_threshold * 3:
                    gaps.append({
                        'type': 'internal',
                        'start': timestamps[i] + 1,
                        'end': timestamps[i+1] - 1,
                        'duration_hours': gap_duration / (1000 * 3600),
                        'inside_file': file_path.name
                    })
            
            return gaps
            
        except Exception as e:
            print(f"Error checking {file_path.name}: {e}")
            return []
    
    def show_gaps(self, gaps):
        if not gaps:
            print("No gaps found")
            return
        
        # Show threshold being used
        threshold_hours = self.gap_threshold / (1000 * 60 * 60)
        print(f"Gap detection for funding rate (threshold: {threshold_hours:.1f} hours)")
        
        boundary_count = len([g for g in gaps if g['type'] == 'boundary'])
        internal_count = len([g for g in gaps if g['type'] == 'internal'])
        
        print(f"Total gaps found: {len(gaps)}")
        if boundary_count > 0:
            print(f"  Boundary gaps: {boundary_count}")
        if internal_count > 0:
            print(f"  Internal gaps: {internal_count}")
        
        for i, gap in enumerate(gaps):
            start_dt = datetime.fromtimestamp(gap['start'] / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(gap['end'] / 1000, tz=timezone.utc)
            
            print(f"\nGap {i+1}/{len(gaps)} ({gap['type']}):")
            print(f"    Time: {start_dt} → {end_dt}")
            print(f"    Duration: {gap['duration_hours']:.2f} hours")            
            if gap['type'] == 'boundary':
                print(f"    Between files: {gap['between_files']}")
            else:
                print(f"    Inside file: {gap['inside_file']}")
    
    async def fill_gaps(self, gaps):
        if not gaps:
            return 0
            
        total_filled = 0
        for i, gap in enumerate(gaps):
            print(f"\nFilling gap {i+1}/{len(gaps)}:")
            
            filled_count = await self._fetch_gap_data(gap['start'], gap['end'])
            if not filled_count:
                print("\tNo data found")
                continue
            
            print(f"    Completed: {filled_count} trades filled")
            total_filled += filled_count
        
        return total_filled
    
    async def _fetch_gap_data(self, start_ms, end_ms):
        funding_buffer = []
        current_after = str(end_ms)
        total_saved = 0
        
        while True:            
            try:
                # Check for cancellation
                if asyncio.current_task().cancelled():
                    break
                    
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.publicAPI.get_funding_rate_history,
                    self.symbol, "", current_after, "100"
                )
                
                if result.get('code') != '0' or not result.get('data'):
                    break
                    
                batch = result['data']
                min_ts = None
                
                for f in batch:
                    ts = int(f['fundingTime'])  # Changed from 'fundingTs' to 'fundingTime'
                    if start_ms <= ts <= end_ms:
                        funding_buffer.append({
                            'instrument_name': self.symbol,
                            'funding_rate': float(f['fundingRate']),
                            'funding_time': ts
                        })
                    
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                
                if len(funding_buffer) >= self.buffer_size:
                    saved_count = self._save_gap_data(funding_buffer)
                    total_saved += saved_count
                    funding_buffer.clear()
                
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
        
        if funding_buffer:
            saved_count = self._save_gap_data(funding_buffer)
            total_saved += saved_count
        
        return total_saved
    
    def _save_gap_data(self, funding_rates):
        funding_by_month = {}
        for funding in funding_rates:
            month_str = datetime.fromtimestamp(funding['funding_time'] / 1000, tz=timezone.utc).strftime('%Y-%m')
            funding_by_month.setdefault(month_str, []).append(funding)
        
        total_saved = 0
        for month_str, month_funding in funding_by_month.items():
            self._save_to_monthly_file(month_str, month_funding)
            total_saved += len(month_funding)
        
        return total_saved
    
    def _save_to_monthly_file(self, month_str, funding_rates):
        df = pl.DataFrame(funding_rates).sort("funding_time")
        file_path = self.output_path / f"{month_str}.parquet"
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df]).unique(subset=["funding_time"], keep="last").sort("funding_time")
            combined_df.write_parquet(file_path, compression="snappy")
            print(f"{len(funding_rates)} → {month_str}.parquet")
        else:
            df.write_parquet(file_path, compression="snappy")
            print(f"{len(funding_rates)} → {month_str}.parquet")
    
    async def run(self):
        gaps = self.detect_gaps()
        self.show_gaps(gaps)
        
        if gaps:
            await self.fill_gaps(gaps)

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