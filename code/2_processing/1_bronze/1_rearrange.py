#!/usr/bin/env python3
"""
Automated UTC data placement fixer - rearranges misplaced records to correct files.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pyarrow.parquet as pq
from collections import defaultdict

class AutoUTCFixer:
    def __init__(self, base_path="datalake/1_bronze"):
        self.base_path = Path(base_path)
        
        self.data_configs = {
            'orderBook': {'timestamp_column': 'ts', 'file_pattern': 'daily', 'path_pattern': '*'},
            'fundingRate': {'timestamp_column': 'funding_time', 'file_pattern': 'monthly', 'path_pattern': '*'},
            'trades': {'timestamp_column': 'created_time', 'file_pattern': 'daily', 'path_pattern': '*'},
            'indexPriceKlines': {'timestamp_column': 'open_time', 'file_pattern': 'daily', 'path_pattern': '*/[0-9]*'},
            'markPriceKlines': {'timestamp_column': 'open_time', 'file_pattern': 'daily', 'path_pattern': '*/[0-9]*'}
        }
        
        self.stats = {'files_processed': 0, 'files_fixed': 0, 'records_moved': 0, 'errors': 0}
    
    def parse_filename_date(self, filename, file_pattern):
        """Parse expected UTC date range from filename."""
        try:
            if file_pattern == 'daily':
                date_obj = datetime.strptime(filename, "%Y-%m-%d").date()
                utc_start = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
                utc_end = datetime.combine(date_obj, datetime.max.time()).replace(tzinfo=timezone.utc)
                return utc_start, utc_end
            elif file_pattern == 'monthly':
                year, month = filename.split('-')
                date_obj = datetime(int(year), int(month), 1).date()
                if int(month) == 12:
                    next_month = datetime(int(year) + 1, 1, 1)
                else:
                    next_month = datetime(int(year), int(month) + 1, 1)
                last_day = (next_month - timedelta(days=1)).date()
                utc_start = datetime.combine(date_obj, datetime.min.time()).replace(tzinfo=timezone.utc)
                utc_end = datetime.combine(last_day, datetime.max.time()).replace(tzinfo=timezone.utc)
                return utc_start, utc_end
        except ValueError:
            return None, None
    
    def get_target_filename(self, timestamp, file_pattern):
        """Generate target filename for a timestamp."""
        if file_pattern == 'daily':
            return timestamp.strftime("%Y-%m-%d") + ".parquet"
        elif file_pattern == 'monthly':
            return timestamp.strftime("%Y-%m") + ".parquet"
    
    def analyze_file(self, file_path, data_type):
        """Analyze file and return misplaced records grouped by target files."""
        file_path = Path(file_path)
        config = self.data_configs[data_type]
        timestamp_column = config['timestamp_column']
        file_pattern = config['file_pattern']
        
        try:
            utc_start, utc_end = self.parse_filename_date(file_path.stem, file_pattern)
            if utc_start is None:
                return {'status': 'error', 'message': 'Could not parse filename'}
            
            utc_start_ms = int(utc_start.timestamp() * 1000)
            utc_end_ms = int(utc_end.timestamp() * 1000)
            
            # Check timestamp column exists
            parquet_file = pq.ParquetFile(file_path)
            if timestamp_column not in parquet_file.schema.names:
                return {'status': 'error', 'message': f'Missing {timestamp_column} column'}
            
            # Read timestamp column first
            df_ts = pd.read_parquet(file_path, columns=[timestamp_column])
            if df_ts.empty:
                return {'status': 'empty'}
            
            # Check for misplaced records
            ts_values = df_ts[timestamp_column].values
            if df_ts[timestamp_column].dtype == 'int64':
                misplaced_mask = (ts_values < utc_start_ms) | (ts_values > utc_end_ms)
            else:
                ts_series = pd.to_datetime(df_ts[timestamp_column], utc=True)
                misplaced_mask = (ts_series < utc_start) | (ts_series > utc_end)
            
            if not misplaced_mask.any():
                return {'status': 'valid', 'records_total': len(df_ts), 'records_misplaced': 0}
            
            # Read full data if misplaced records found
            df_full = pd.read_parquet(file_path)
            if len(df_full) > 1_000_000:
                print(f"    📊 Large file detected ({len(df_full):,} records)")
            
            # Debug: Check data types in the loaded dataframe
            if 'trade_id' in df_full.columns:
                print(f"    🔍 trade_id dtype: {df_full['trade_id'].dtype}")
            
            # Process timestamps
            if df_full[timestamp_column].dtype == 'int64':
                ts_series_full = pd.to_datetime(df_full[timestamp_column], unit='ms', utc=True)
            else:
                ts_series_full = pd.to_datetime(df_full[timestamp_column], utc=True)
            
            # Split records
            correct_records = df_full[~misplaced_mask].copy()
            misplaced_records = df_full[misplaced_mask].copy()
            misplaced_timestamps = ts_series_full[misplaced_mask]
            
            # Group by target files
            misplaced_groups = defaultdict(list)
            for idx, ts in zip(misplaced_records.index, misplaced_timestamps):
                target_filename = self.get_target_filename(ts, file_pattern)
                misplaced_groups[target_filename].append(idx)
            
            return {
                'status': 'needs_fix',
                'records_total': len(df_full),
                'records_misplaced': len(misplaced_records),
                'correct_records': correct_records,
                'misplaced_records': misplaced_records,
                'misplaced_groups': dict(misplaced_groups)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    
    def append_to_file(self, target_path, new_records_df, timestamp_column):
        """Append records to target file with schema compatibility."""
        target_path = Path(target_path)
        
        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
            
            # Align schemas with better type handling
            for col in existing_df.columns:
                if col in new_records_df.columns and existing_df[col].dtype != new_records_df[col].dtype:
                    existing_dtype = existing_df[col].dtype
                    new_dtype = new_records_df[col].dtype
                    
                    try:
                        # Handle specific problematic cases
                        if col == 'trade_id':
                            # Ensure trade_id stays as int64
                            if existing_dtype != 'int64':
                                existing_df[col] = existing_df[col].astype('int64')
                            if new_dtype != 'int64':
                                new_records_df[col] = new_records_df[col].astype('int64')
                        elif 'int' in str(existing_dtype) and 'int' in str(new_dtype):
                            # Both are integers, use the larger type
                            target_dtype = 'int64' if 'int64' in [str(existing_dtype), str(new_dtype)] else existing_dtype
                            new_records_df[col] = new_records_df[col].astype(target_dtype)
                            if existing_dtype != target_dtype:
                                existing_df[col] = existing_df[col].astype(target_dtype)
                        elif 'float' in str(existing_dtype) and ('int' in str(new_dtype) or 'float' in str(new_dtype)):
                            # Convert to float for compatibility
                            new_records_df[col] = new_records_df[col].astype(existing_dtype)
                        else:
                            # Try direct conversion first
                            new_records_df[col] = new_records_df[col].astype(existing_dtype)
                    except (ValueError, TypeError) as e:
                        # Fallback to string conversion only if other methods fail
                        print(f"    ⚠️  Schema mismatch for {col}: {existing_dtype} vs {new_dtype}, converting to string")
                        existing_df[col] = existing_df[col].astype(str)
                        new_records_df[col] = new_records_df[col].astype(str)
            
            combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
            combined_df = combined_df.sort_values(timestamp_column)
            combined_df.to_parquet(target_path, index=False)
        else:
            new_records_df = new_records_df.sort_values(timestamp_column)
            new_records_df.to_parquet(target_path, index=False)
    
    def fix_single_file(self, file_path, data_type):
        """Fix a single file automatically."""
        file_path = Path(file_path)
        config = self.data_configs[data_type]
        
        self.stats['files_processed'] += 1
        analysis = self.analyze_file(file_path, data_type)
        
        if analysis['status'] != 'needs_fix':
            return analysis
        
        try:
            print(f"  🔧 Fixing: {file_path.name} ({analysis['records_misplaced']:,} misplaced records)")
            
            # Move misplaced records to target files
            for target_filename, record_indices in analysis['misplaced_groups'].items():
                target_path = file_path.parent / target_filename
                misplaced_subset = analysis['misplaced_records'].loc[record_indices].copy()
                self.append_to_file(target_path, misplaced_subset, config['timestamp_column'])
                print(f"    📤 Moved {len(record_indices):,} records to {target_filename}")
            
            # Update source file
            if len(analysis['correct_records']) > 0:
                analysis['correct_records'].to_parquet(file_path, index=False)
            else:
                print(f"    🗑️  Deleting original file (no correct records to keep)")
                file_path.unlink()
            
            self.stats['files_fixed'] += 1
            self.stats['records_moved'] += analysis['records_misplaced']
            print(f"    ✅ Fixed successfully")
            analysis['status'] = 'fixed'
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            self.stats['errors'] += 1
            analysis['status'] = 'error'
            analysis['message'] = str(e)
        
        return analysis
    
    def find_data_paths(self, data_type):
        """Find all data paths for a given data type."""
        data_type_path = self.base_path / data_type
        if not data_type_path.exists():
            return []
        
        path_pattern = self.data_configs[data_type]['path_pattern']
        
        if path_pattern == '*':
            return [d for d in data_type_path.iterdir() 
                   if d.is_dir() and not d.name.endswith(('_backup', '_reorganized'))]
        else:
            paths = []
            for instrument_dir in data_type_path.iterdir():
                if instrument_dir.is_dir() and not instrument_dir.name.endswith(('_backup', '_reorganized')):
                    for timeframe_dir in instrument_dir.iterdir():
                        if timeframe_dir.is_dir() and timeframe_dir.name[0].isdigit():
                            paths.append(timeframe_dir)
            return paths
    
    def fix_data_type(self, data_type):
        """Automatically fix all files for a data type."""
        print(f"\n{'='*80}")
        print(f"🚀 AUTO FIXING: {data_type}")
        print(f"{'='*80}")
        
        data_paths = self.find_data_paths(data_type)
        if not data_paths:
            print(f"❌ No valid paths found for {data_type}")
            return []
        
        all_results = []
        for data_path in data_paths:
            path_name = str(data_path.relative_to(self.base_path))
            print(f"\n📁 Processing path: {path_name}")
            
            parquet_files = [f for f in data_path.glob("*.parquet") if not f.name.endswith('_backup.parquet')]
            parquet_files.sort()
            
            if not parquet_files:
                print("  No parquet files found")
                continue
            
            print(f"  📊 Found {len(parquet_files)} files")
            files_fixed_in_path = 0
            
            for i, file_path in enumerate(parquet_files, 1):
                if i % 50 == 0 or i == len(parquet_files):
                    print(f"  Progress: {i}/{len(parquet_files)} ({i*100//len(parquet_files)}%)")
                
                result = self.fix_single_file(file_path, data_type)
                result['path'] = path_name
                all_results.append(result)
                
                if result.get('status') == 'fixed':
                    files_fixed_in_path += 1
            
            print(f"  📈 Path summary: {files_fixed_in_path}/{len(parquet_files)} files fixed")
        
        return all_results
    
    def fix_all_data_types(self):
        """Automatically fix all data types in recommended order."""
        print("🚀 AUTOMATED UTC DATA PLACEMENT FIXER")
        print("=" * 80)
        print("Order: orderBook → indexPriceKlines → markPriceKlines → fundingRate → trades\n")
        
        data_types = ['orderBook', 'indexPriceKlines', 'markPriceKlines', 'fundingRate', 'trades']
        all_results = {}
        start_time = datetime.now()
        
        for data_type in data_types:
            try:
                results = self.fix_data_type(data_type)
                all_results[data_type] = results
                data_fixed = len([r for r in results if r.get('status') == 'fixed'])
                print(f"✅ {data_type} completed: {data_fixed}/{len(results)} files fixed")
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user during {data_type}")
                break
            except Exception as e:
                print(f"❌ Error processing {data_type}: {str(e)}")
        
        duration = datetime.now() - start_time
        self.print_final_summary(all_results, duration)
        return all_results
    
    def print_final_summary(self, all_results, duration):
        """Print final summary to terminal."""
        print(f"\n{'='*80}")
        print("🎉 AUTOMATED FIX COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️  Duration: {duration}")
        print(f"📁 Files processed: {self.stats['files_processed']:,}")
        print(f"🔧 Files fixed: {self.stats['files_fixed']:,}")
        print(f"📦 Records moved: {self.stats['records_moved']:,}")
        print(f"❌ Errors: {self.stats['errors']:,}")
        
        # Summary by data type
        for data_type, results in all_results.items():
            if not results:
                continue
            fixed_files = len([r for r in results if r.get('status') == 'fixed'])
            error_files = len([r for r in results if r.get('status') == 'error'])
            if fixed_files > 0 or error_files > 0:
                print(f"📊 {data_type}: {fixed_files} fixed, {error_files} errors")

def main():
    """Main automatic fixer."""
    AutoUTCFixer().fix_all_data_types()

if __name__ == "__main__":
    main()