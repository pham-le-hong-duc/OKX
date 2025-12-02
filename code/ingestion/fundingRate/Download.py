import tempfile
import zipfile
import gc
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta

import requests
import polars as pl

class Download:
    def __init__(self, symbol="BTC-USDT-SWAP", base_start_date="2025-01-01", base_data_path="../../../datalake/1_bronze"):
        self.symbol = symbol
        self.base_start_date = base_start_date
        self.base_data_path = Path(base_data_path)
        self.monthly_url_template = "https://static.okx.com/cdn/okex/traderecords/swaprates/monthly/{YYYYMM}/{SYMBOL}-fundingrates-{YYYY_MM}.zip"
        
        self.output_path = self.base_data_path / "fundingRate" / symbol.lower()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.interrupted = False
    
    def _get_files(self):
        available_files = []
        start_date = datetime.strptime(self.base_start_date, "%Y-%m-%d")
        current_date = datetime.now()
        
        # Set to first day of start month
        current_month = start_date.replace(day=1)
        
        while current_month <= current_date:
            year_month = current_month.strftime("%Y-%m")
            year_month_compact = current_month.strftime("%Y%m")
            
            url = self.monthly_url_template.format(
                YYYYMM=year_month_compact,
                SYMBOL=self.symbol,
                YYYY_MM=year_month
            )
            
            try:
                if requests.head(url, timeout=10).status_code == 200:
                    available_files.append({
                        "date": current_month,
                        "url": url,
                        "filename": f"{year_month}.parquet"
                    })
            except:
                pass
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        return available_files
    
    def _download_and_convert(self, url, output_file):
        """Download and convert funding rate files (sync version)"""
        with tempfile.TemporaryDirectory(dir=self.output_path) as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                zip_path = temp_path / "temp.zip"
                zip_path.write_bytes(response.content)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        return False
                    
                    csv_path = temp_path / csv_files[0]
                    zip_ref.extract(csv_files[0], temp_path)
                
                # Read and convert CSV to parquet
                df = pl.read_csv(csv_path)
                df.write_parquet(output_file, compression="snappy")
                
                return True
                
            except Exception as e:
                return False
    
    def run(self):
        """Main run method for funding rate monthly files"""
        available_files = self._get_files()
        
        files_to_download = [f for f in available_files 
                            if not (self.output_path / f["filename"]).exists()]
        
        if not files_to_download:
            return
            
        print(f"Downloading {len(files_to_download)} monthly files...")
        
        # Sort by date
        monthly_files = sorted(files_to_download, key=lambda x: x["date"], reverse=False)
        
        for file_info in monthly_files:
            if self.interrupted:
                print("\n⚠️ Download interrupted by user")
                break
                
            output_file = self.output_path / file_info["filename"]
            
            if output_file.exists():
                continue
            
            if self._download_and_convert(file_info["url"], output_file):
                print(f"✓ {file_info['filename']}")
            else:
                print(f"✗ {file_info['filename']}")
            gc.collect()

if __name__ == "__main__":
    downloader = Download()
    
    def signal_handler(sig, frame):
        downloader.interrupted = True
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        downloader.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass