import json
import asyncio
import signal
import platform
from pathlib import Path
from datetime import datetime
import websockets
import polars as pl

class WebSocketStream:
    def __init__(self, symbol="BTC-USDT-SWAP", base_data_path="../../../datalake/1_bronze",
                 buffer_size=100):
        self.symbol = symbol
        self.base_data_path = Path(base_data_path)
        self.output_path = self.base_data_path / "fundingRate" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.url = "wss://wspap.okx.com:8443/ws/v5/public"
        self.channel = "funding-rate"
        
        args = [{"channel": self.channel, "instId": symbol}]
        self.subscribe_msg = json.dumps({"op": "subscribe", "args": args})
        
        self.running = False
        
        self.buffer = []
        self.buffer_size = buffer_size
        self.current_month = datetime.now().strftime("%Y-%m")
        self.processed_funding_times = set()  # Track processed funding times
    
    def _normalize(self, funding_data):
        try:
            if isinstance(funding_data, dict):
                return {
                    "instrument_name": self.symbol,
                    "funding_rate": float(funding_data["fundingRate"]),
                    "funding_time": int(funding_data["fundingTime"])
                }
            return None
        except:
            return None
    
    def _check_month_change(self):
        """Check if month has changed and flush buffer if needed"""
        new_month = datetime.now().strftime("%Y-%m")
        if new_month != self.current_month:
            # Flush buffer with old month before updating
            if self.buffer:
                self._flush_buffer_with_month(self.current_month)
            self.current_month = new_month
            # Reset processed funding times for new month
            self.processed_funding_times.clear()
            return True
        return False
    
    def _handle_message(self, message_str):
        try:
            # Check if month has changed before processing new messages
            self._check_month_change()
            
            data = json.loads(message_str)
            
            if (data.get('arg', {}).get('channel') == self.channel and 'data' in data):
                for funding_record in data['data']:
                    normalized_funding = self._normalize(funding_record)
                    
                    if normalized_funding:
                        funding_time = normalized_funding['funding_time']
                        
                        # Check for duplicate funding_time globally
                        if funding_time not in self.processed_funding_times:
                            self.buffer.append(normalized_funding)
                            self.processed_funding_times.add(funding_time)
                        
                        if len(self.buffer) >= self.buffer_size:
                            self._flush_buffer()
                            
        except:
            pass
    
    def _flush_buffer_with_month(self, target_month):
        """Flush buffer to a specific month file"""
        if not self.buffer:
            return
        
        try:
            # Deduplicate funding records by funding_time (keep last occurrence)
            funding_dict = {funding["funding_time"]: funding for funding in self.buffer}
            final_funding = list(funding_dict.values())
            
            if final_funding:
                df = pl.DataFrame(final_funding)
                
                output_file = self.output_path / f"{target_month}.parquet"
                
                if output_file.exists():
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, df])
                    final_df = combined_df.unique(subset=["funding_time"], keep="last")
                else:
                    final_df = df
                
                final_df = final_df.sort("funding_time")
                final_df.write_parquet(output_file)
                
                print(f"{len(final_funding)} → {output_file.name}")
            
            self.buffer.clear()
            
        except Exception as e:
            print(f"Error flushing buffer with month {target_month}: {e}")
    
    def _flush_buffer(self):
        """Flush buffer to current month file"""
        self._flush_buffer_with_month(self.current_month)
    
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
                            
                            # Check for month change
                            self._check_month_change()
                                
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
            print("\nReceived interrupt signal, stopping...")
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
        if self.buffer:
            self._flush_buffer()

if __name__ == "__main__":
    WebSocketStream(buffer_size=1).run()