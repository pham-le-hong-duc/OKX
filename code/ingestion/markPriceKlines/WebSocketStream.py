import json
import asyncio
import signal
from pathlib import Path
from datetime import datetime
import websockets
import polars as pl

class WebSocketStream:
    def __init__(self, symbol="BTC-USD-SWAP", base_data_path="../../../datalake/1_bronze",
                 buffer_size=100):
        self.symbol = symbol
        self.base_data_path = Path(base_data_path)
        
        self.url = "wss://wspap.okx.com:8443/ws/v5/business"
        
        # Map intervals to channel names
        self.interval_mapping = {
            "1m": "mark-price-candle1m",
            "5m": "mark-price-candle5m", 
            "15m": "mark-price-candle15m",
            "1H": "mark-price-candle1H",
            "4H": "mark-price-candle4H",
            "1D": "mark-price-candle1D"
        }
        
        # Subscribe to all intervals
        args = []
        for interval, channel in self.interval_mapping.items():
            args.append({"channel": channel, "instId": symbol})
            
        self.subscribe_msg = json.dumps({"op": "subscribe", "args": args})
        
        self.running = False
        self.buffer_size = buffer_size
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Separate buffers for each interval
        self.buffers = {interval: [] for interval in self.interval_mapping.keys()}
        
        # Track processed open_time for each interval to avoid duplicates
        self.processed_open_times = {interval: set() for interval in self.interval_mapping.keys()}
        
        # Create output paths for all intervals
        for interval in self.interval_mapping.keys():
            output_path = self.base_data_path / "markPriceKlines" / symbol.lower() / interval
            output_path.mkdir(parents=True, exist_ok=True)
    
    def _normalize(self, kline_data, interval):
        try:
            if isinstance(kline_data, list) and len(kline_data) >= 5:
                return {
                    "open_time": int(kline_data[0]),
                    "open": float(kline_data[1]),
                    "high": float(kline_data[2]),
                    "low": float(kline_data[3]),
                    "close": float(kline_data[4])
                }
            return None
        except:
            return None
    
    def _check_date_change(self):
        """Check if date has changed and flush all buffers if needed"""
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self.current_date:
            # Flush all buffers with old date before updating
            for interval in self.interval_mapping.keys():
                if self.buffers[interval]:
                    self._flush_buffer_with_date(interval, self.current_date)
            self.current_date = new_date
            # Reset processed open_times for new date
            for interval in self.interval_mapping.keys():
                self.processed_open_times[interval].clear()
            return True
        return False
    
    def _handle_message(self, message_str):
        try:
            # Check if date has changed before processing new messages
            self._check_date_change()
            
            data = json.loads(message_str)
            
            if 'arg' in data and 'data' in data:
                channel = data.get('arg', {}).get('channel', '')
                
                # Determine interval from channel name
                interval = None
                for int_key, channel_name in self.interval_mapping.items():
                    if channel == channel_name:
                        interval = int_key
                        break
                
                if interval:
                    for kline in data['data']:
                        normalized_kline = self._normalize(kline, interval)
                        
                        if normalized_kline:
                            open_time = normalized_kline['open_time']
                            
                            # Check for duplicate open_time globally for this interval
                            if open_time not in self.processed_open_times[interval]:
                                self.buffers[interval].append(normalized_kline)
                                self.processed_open_times[interval].add(open_time)
                            
                            if len(self.buffers[interval]) >= self.buffer_size:
                                self._flush_buffer(interval)
                            
        except:
            pass
    
    def _flush_buffer_with_date(self, interval, target_date):
        """Flush buffer to a specific date file for given interval"""
        if not self.buffers[interval]:
            return
        
        try:
            # Deduplicate klines by open_time (keep last occurrence)
            klines_dict = {kline["open_time"]: kline for kline in self.buffers[interval]}
            final_klines = list(klines_dict.values())
            
            if final_klines:
                df = pl.DataFrame(final_klines)
                
                output_path = self.base_data_path / "markPriceKlines" / self.symbol.lower() / interval
                output_file = output_path / f"{target_date}.parquet"
                
                if output_file.exists():
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, df])
                    final_df = combined_df.unique(subset=["open_time"], keep="last")
                else:
                    final_df = df
                
                final_df = final_df.sort("open_time")
                final_df.write_parquet(output_file)
                
                print(f"{len(final_klines)} → {interval}/{target_date}.parquet")
                
            self.buffers[interval].clear()
            
        except:
            pass
    
    def _flush_buffer(self, interval):
        """Flush buffer to current date file for given interval"""
        self._flush_buffer_with_date(interval, self.current_date)
    
    def run(self):
        async def _stream():
            self.running = True
            reconnect_delay = 1
            
            while self.running:
                try:
                    async with websockets.connect(self.url) as ws:
                        print("Connected")
                        
                        await ws.send(self.subscribe_msg)
                        
                        reconnect_delay = 1
                        
                        async for message in ws:
                            if not self.running:  # Check if we should stop
                                break
                                
                            self._handle_message(message)
                            
                            # Check for date change
                            self._check_date_change()
                                
                except websockets.exceptions.ConnectionClosed:
                    if self.running:
                        print("Connection lost, reconnecting...")
                except Exception as e:
                    if self.running:
                        print(f"Error: {e}")
                
                if self.running:
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
        
        def signal_handler():
            print("Disconnected")
            self.running = False
        
        async def _run_with_signal_handling():
            # Set up signal handlers for clean shutdown
            loop = asyncio.get_running_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, signal_handler)
            
            try:
                await _stream()
            except KeyboardInterrupt:
                signal_handler()
            finally:
                self.stop()
        
        try:
            import platform
            if platform.system() == 'Windows':
                # Windows doesn't support signal handlers properly
                try:
                    asyncio.run(_stream())
                except KeyboardInterrupt:
                    print("\nReceived interrupt signal, stopping...")
                    self.stop()
            else:
                # Unix systems
                asyncio.run(_run_with_signal_handling())
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        self.running = False
        # Flush all buffers when stopping
        for interval in self.interval_mapping.keys():
            if self.buffers[interval]:
                self._flush_buffer(interval)
    
if __name__ == "__main__":
    WebSocketStream(buffer_size=1).run()
