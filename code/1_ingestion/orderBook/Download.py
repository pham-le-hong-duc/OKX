import tempfile
import gc
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import orjson

import requests
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

class Download:
    def __init__(self, symbol="BTC-USDT-SWAP", base_start_date="2025-01-01", 
                 base_data_path="../../../datalake/1_bronze", chunk_size=2000000):
        self.symbol = symbol
        self.base_start_date = base_start_date
        self.base_data_path = Path(base_data_path)
        self.chunk_size = chunk_size
        self.daily_url_template = "https://static.okx.com/cdn/okx/match/orderbook/L2/400lv/daily/{YYYYMMDD}/{SYMBOL}-L2orderbook-400lv-{YYYY_MM_DD}.tar.gz"
        
        self.output_path = self.base_data_path / "orderBook" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.schema = pa.schema([
            ('instId', pa.string()),
            ('action', pa.string()),
            ('ts', pa.int64()),
            ('asks', pa.string()),
            ('bids', pa.string())
        ])
    
    def get_url(self, date_obj):
        """Generate URL for a specific date"""
        return self.daily_url_template.format(
            YYYYMMDD=date_obj.strftime("%Y%m%d"),
            SYMBOL=self.symbol,
            YYYY_MM_DD=date_obj.strftime("%Y-%m-%d")
        )
    
    def _is_download_available(self, url):
        """Check if download is available using HEAD request"""
        try:
            return requests.head(url, timeout=30).status_code == 200
        except:
            return False
    
    def get_available_files_list(self):
        """Get list of available files from 2025-01-01 to today"""
        today = datetime.now(timezone.utc).date()
        start_date = datetime(2025, 1, 1).date()
        available_files = []
        
        current_date = start_date
        
        while current_date <= today:
            if self._is_download_available(self.get_url(current_date)):
                available_files.append(current_date)
            current_date += timedelta(days=1)
        
        print(f"Found {len(available_files)} available files")
        return available_files
    
    def download_date_range(self, dates_list):
        """Download all dates in list"""
        results = {"success": 0, "failed": 0}
        
        for date_obj in dates_list:
            output_file = self.output_path / f"{date_obj.strftime('%Y-%m-%d')}.parquet"
            
            if self._download_and_convert_sync(self.get_url(date_obj), output_file):
                print(f"{date_obj.strftime('%Y-%m-%d')}.parquet")
                results["success"] += 1
            else:
                results["failed"] += 1
        
        return results["success"]
    
    def _download_and_convert_sync(self, url, output_file):
        """Download and convert file"""
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            return self._convert_sync(response.content, output_file) > 0
        except:
            return False
    
    def _convert_sync(self, content, output_file):
        """Convert tar.gz content to parquet"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tar.gz') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                import tarfile
                with tarfile.open(tmp_path, 'r:gz') as tar_file:
                    files = tar_file.getnames()
                    data_files = [f for f in files if f.endswith('.data')]
                    csv_files = [f for f in files if f.endswith('.csv')]
                    
                    if data_files:
                        return self._process_data_file(tar_file, data_files[0], tmp_path, output_file)
                    elif csv_files:
                        return self._process_csv_file(tar_file, csv_files[0], tmp_path, output_file)
                    return 0
                        
            finally:
                Path(tmp_path).unlink(missing_ok=True)
                
        except:
            return 0
    
    def _process_data_file(self, tar_file, data_file, tmp_path, output_file):
        """Process .data file"""
        tar_file.extract(data_file, Path(tmp_path).parent)
        data_path = Path(tmp_path).parent / data_file
        
        try:
            return self._convert_l400_to_l50_optimized(data_path, output_file)
        finally:
            data_path.unlink(missing_ok=True)
    
    def _process_csv_file(self, tar_file, csv_file, tmp_path, output_file):
        """Process .csv file"""
        tar_file.extract(csv_file, Path(tmp_path).parent)
        csv_path = Path(tmp_path).parent / csv_file
        
        try:
            return self._convert_csv_optimized_from_file(csv_path, output_file)
        finally:
            csv_path.unlink(missing_ok=True)
    
    def _convert_l400_to_l50_optimized(self, data_path, output_file):
        """Convert .data file (JSON lines) to L50 parquet"""
        total_written = 0
        
        try:
            with pq.ParquetWriter(output_file, self.schema, compression='snappy') as writer:
                with open(data_path, 'r') as f:
                    lines_batch = []
                    
                    for line in f:
                        lines_batch.append(line.strip())
                        
                        if len(lines_batch) >= self.chunk_size:
                            total_written += self._process_json_lines_batch(lines_batch, writer)
                            lines_batch = []
                    
                    if lines_batch:
                        total_written += self._process_json_lines_batch(lines_batch, writer)
            
        except:
            return 0
            
        return total_written
    
    def _process_json_lines_batch(self, lines_batch, writer):
        """Process batch of JSON lines"""
        data = {'inst_ids': [], 'actions': [], 'timestamps': [], 'asks': [], 'bids': []}
        
        for line in lines_batch:
            if not line:
                continue
                
            try:
                json_data = orjson.loads(line.encode())
                
                asks = json_data.get('asks', [])
                bids = json_data.get('bids', [])
                
                if asks and bids:
                    data['inst_ids'].append(json_data.get('instId', ''))
                    data['actions'].append(json_data.get('action', ''))
                    data['timestamps'].append(int(json_data.get('ts', 0)))
                    
                    # Convert to L50 and serialize
                    asks_l50 = orjson.dumps(asks[:50]).decode()
                    bids_l50 = orjson.dumps(bids[:50]).decode()
                    
                    data['asks'].append(asks_l50)
                    data['bids'].append(bids_l50)
                    
            except:
                continue
        
        if data['inst_ids']:
            table = pa.Table.from_arrays([
                pa.array(data['inst_ids']),
                pa.array(data['actions']),
                pa.array(data['timestamps']),
                pa.array(data['asks']),
                pa.array(data['bids'])
            ], schema=self.schema)
            
            writer.write_table(table)
        
        return len(data['inst_ids'])
    
    def _convert_csv_optimized_from_file(self, csv_file_path, output_file):
        """Convert CSV file to parquet"""
        total_written = 0
        
        with pq.ParquetWriter(output_file, self.schema, compression='snappy') as writer:
            with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
                lines_batch = []
                
                for line in csv_file:
                    line = line.strip()
                    if not line or line.startswith('ts'):
                        continue
                    
                    lines_batch.append(line)
                    
                    if len(lines_batch) >= self.chunk_size:
                        total_written += self._process_csv_batch(lines_batch, writer)
                        lines_batch = []
                
                if lines_batch:
                    total_written += self._process_csv_batch(lines_batch, writer)
        
        return total_written
    
    def _process_csv_batch(self, lines_batch, writer):
        """Process batch of CSV lines"""
        data = {'inst_ids': [], 'actions': [], 'timestamps': [], 'asks': [], 'bids': []}
        
        for line in lines_batch:
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    ts = int(parts[0])
                    bids_data = orjson.loads(parts[1].encode())
                    asks_data = orjson.loads(parts[2].encode())
                    
                    if bids_data and asks_data:
                        data['inst_ids'].append(self.symbol)
                        data['actions'].append("snapshot")
                        data['timestamps'].append(ts)
                        
                        # Convert to L50 and serialize
                        bids_l50 = orjson.dumps(bids_data[:50]).decode()
                        asks_l50 = orjson.dumps(asks_data[:50]).decode()
                        
                        data['asks'].append(asks_l50)
                        data['bids'].append(bids_l50)
                        
                except:
                    continue
        
        if data['inst_ids']:
            table = pa.Table.from_arrays([
                pa.array(data['inst_ids']),
                pa.array(data['actions']),
                pa.array(data['timestamps']),
                pa.array(data['asks']),
                pa.array(data['bids'])
            ], schema=self.schema)
            
            writer.write_table(table)
        
        return len(data['inst_ids'])
    
    def run_bootstrap(self):
        """Phase 1: Bootstrap initial data"""
        print("Phase 1: Bootstrapping initial data...")
        
        try:
            available_dates = self.get_available_files_list()
            
            # Filter files that don't exist or are smaller than 200MB
            files_to_download = []
            for date in available_dates:
                file_path = self.output_path / f"{date.strftime('%Y-%m-%d')}.parquet"
                if not file_path.exists():
                    # File doesn't exist - need to download
                    files_to_download.append(date)
                else:
                    # File exists - check size (200MB = 200 * 1024 * 1024 bytes)
                    file_size = file_path.stat().st_size
                    if file_size < 200 * 1024 * 1024:
                        # File is smaller than 200MB - treat as incomplete and re-download
                        print(f"File {file_path.name} is {file_size / (1024*1024):.2f}MB < 200MB, re-downloading...")
                        files_to_download.append(date)
            
            if files_to_download:
                self.download_date_range(files_to_download)
            
            return set(available_dates)
            
        except Exception as e:
            print(f"Bootstrap error: {e}")
            return set()
    
    def run_monitor(self, initial_reference_list):
        """Phase 2: Monitor for new files every 8 hours"""
        print("Phase 2: Starting continuous monitoring...")
        
        saved_available_files = initial_reference_list.copy()
        
        while True:
            try:
                current_available_files = set(self.get_available_files_list())
                
                if not current_available_files:
                    print("No data available, waiting 8 hours...")
                    time.sleep(8 * 3600)
                    continue
                
                new_files = current_available_files - saved_available_files
                
                if not new_files:
                    print("No new files detected since initial check")
                else:
                    self.download_date_range(sorted(list(new_files)))
                
                saved_available_files = current_available_files.copy()
                
                print("Waiting 8 hours...")
                time.sleep(8 * 3600)
                
            except KeyboardInterrupt:
                print("Interrupted by user")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Waiting 1 hour before retry...")
                time.sleep(3600)
    
    def run(self):
        """Main run method: Bootstrap + Monitor"""
        initial_reference_list = self.run_bootstrap()
        self.run_monitor(initial_reference_list)

if __name__ == "__main__":
    downloader = Download()
    downloader.run()