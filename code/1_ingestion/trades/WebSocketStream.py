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
                 buffer_size=100, buffer_timeout=60):
        self.symbol = symbol
        self.base_data_path = Path(base_data_path)
        self.output_path = self.base_data_path / "trades" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.url = "wss://ws.okx.com:8443/ws/v5/public"
        self.channel = "trades"
        
        args = [{"channel": self.channel, "instId": symbol}]
        self.subscribe_msg = json.dumps({"op": "subscribe", "args": args})
        
        self.running = False
        
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_timeout = buffer_timeout
        self.last_flush_time = datetime.now()
        self.current_date = datetime.now().strftime("%Y-%m-%d")
    
    def _normalize(self, trade):
        try:
            if isinstance(trade, dict):
                return {
                    "instrument_name": self.symbol,
                    "trade_id": int(trade["tradeId"]),  # Ép kiểu thành int để đồng nhất với RestAPI
                    "side": trade["side"],
                    "price": float(trade["px"]),
                    "size": float(trade["sz"]),
                    "created_time": int(trade["ts"])
                }
            return None
        except:
            return None
    
    def _check_date_change(self):
        """Check if date has changed and flush buffer if needed"""
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self.current_date:
            # Flush buffer with old date before updating
            if self.buffer:
                self._flush_buffer_with_date(self.current_date)
            self.current_date = new_date
            return True
        return False
    
    def _handle_message(self, message_str):
        try:
            # Check if date has changed before processing new messages
            self._check_date_change()
            
            data = json.loads(message_str)
            
            if (data.get('arg', {}).get('channel') == self.channel and 'data' in data):
                for trade in data['data']:
                    normalized_trade = self._normalize(trade)
                    
                    if normalized_trade:
                        self.buffer.append(normalized_trade)
                        
                        if len(self.buffer) >= self.buffer_size:
                            self._flush_buffer()
                            
        except:
            pass
    
    def _flush_buffer_with_date(self, target_date):
        """Flush buffer to a specific date file"""
        if not self.buffer:
            return
        
        try:
            # Deduplicate trades by trade_id (keep last occurrence)
            trades_dict = {trade["trade_id"]: trade for trade in self.buffer}
            final_trades = list(trades_dict.values())
            
            if final_trades:
                df = pl.DataFrame(final_trades)
                
                output_file = self.output_path / f"{target_date}.parquet"
                
                if output_file.exists():
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, df])
                    final_df = combined_df.unique(subset=["trade_id"], keep="last")
                else:
                    final_df = df
                
                final_df = final_df.sort("created_time")
                final_df.write_parquet(output_file)
                
                print(f"{len(final_trades)} → {output_file.name}")
                
            self.buffer.clear()
            self.last_flush_time = datetime.now()
            
        except:
            pass
    
    def _flush_buffer(self):
        """Flush buffer to current date file"""
        self._flush_buffer_with_date(self.current_date)
    
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
                            
                            # Check for date change and timeout flush
                            self._check_date_change()
                            time_since_flush = (datetime.now() - self.last_flush_time).total_seconds()
                            if len(self.buffer) > 0 and time_since_flush >= self.buffer_timeout:
                                self._flush_buffer()
                                
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
    WebSocketStream(buffer_size=128, buffer_timeout=8).run()
