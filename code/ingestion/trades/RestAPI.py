import asyncio
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import okx.api.market as MarketData

class RestAPI:
    
    def __init__(self, symbol="BTC-USDT-SWAP", base_start_date="2025-01-01", base_data_path="../../../datalake/1_bronze", buffer_size=2000):
        self.symbol = symbol
        self.output_path = Path(base_data_path) / "trades" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.marketAPI = MarketData.Market()
        self.buffer_size = buffer_size
    
    def detect_gaps(self, gap_threshold=60000):
        files = sorted(self.output_path.glob("*.parquet"))
        if not files:
            return []
        
        gaps = []
        gaps.extend(self._detect_boundary_gaps(files, gap_threshold))
        gaps.extend(self._detect_internal_gaps(files, gap_threshold))
        return gaps
    
    def _detect_boundary_gaps(self, files, gap_threshold):
        gaps = []
        for i in range(len(files) - 1):
            current_df = pl.scan_parquet(files[i]).select(pl.col("created_time").max()).collect()
            next_df = pl.scan_parquet(files[i + 1]).select(pl.col("created_time").min()).collect()
            
            current_end = current_df["created_time"][0]
            next_start = next_df["created_time"][0]
            
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
            df = pl.read_parquet(file_path, columns=["created_time"])
            
            if len(df) > 10_000_000:
                df = df.sample(n=50_000)
            
            timestamps = df["created_time"].sort().to_list()
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
            
        except:
            return []
    
    def show_gaps(self, gaps):
        if not gaps:
            print("No gaps found")
            return
        
        boundary_count = len([g for g in gaps if g['type'] == 'boundary'])
        internal_count = len([g for g in gaps if g['type'] == 'internal'])
        
        print(f"Boundary gaps: {boundary_count}")
        print(f"Internal gaps: {internal_count}")
        
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
        trades_buffer = []
        current_after = str(end_ms)
        total_saved = 0
        
        while True:            
            try:
                # Check for cancellation
                if asyncio.current_task().cancelled():
                    break
                    
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self.marketAPI.get_history_trades,
                    self.symbol, "2", current_after, "", "100"
                )
                
                if result.get('code') != '0' or not result.get('data'):
                    break
                    
                batch = result['data']
                min_ts = None
                
                for t in batch:
                    ts = int(t['ts'])
                    if start_ms <= ts <= end_ms:
                        trades_buffer.append({
                            'instrument_name': self.symbol,
                            'trade_id': int(t['tradeId']),
                            'side': t['side'],
                            'price': float(t['px']),  # Price before size to match existing schema
                            'size': float(t['sz']),
                            'created_time': ts
                        })
                    
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                
                if len(trades_buffer) >= self.buffer_size:
                    saved_count = self._save_gap_data(trades_buffer)
                    total_saved += saved_count
                    trades_buffer.clear()
                
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
        
        if trades_buffer:
            saved_count = self._save_gap_data(trades_buffer)
            total_saved += saved_count
        
        return total_saved
    
    def _save_gap_data(self, trades):
        trades_by_date = {}
        for trade in trades:
            date_str = datetime.fromtimestamp(trade['created_time'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
            trades_by_date.setdefault(date_str, []).append(trade)
        
        total_saved = 0
        for date_str, day_trades in trades_by_date.items():
            self._save_to_daily_file(date_str, day_trades)
            total_saved += len(day_trades)
        
        return total_saved
    
    def _save_to_daily_file(self, date_str, trades):
        df = pl.DataFrame(trades).sort("created_time")
        file_path = self.output_path / f"{date_str}.parquet"
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            combined_df = pl.concat([existing_df, df]).unique(subset=["trade_id"], keep="last").sort("created_time")
            combined_df.write_parquet(file_path, compression="snappy")
        else:
            df.write_parquet(file_path, compression="snappy")
        
        print(f"{len(trades)} → {date_str}.parquet")
    
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
