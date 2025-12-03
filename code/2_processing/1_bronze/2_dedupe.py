#!/usr/bin/env python3
"""
Check and remove duplicate records in Silver layer parquet files.
Supports all data types with different schemas.
"""

import polars as pl
from pathlib import Path
from datetime import datetime
import traceback

class DuplicateChecker:
    def __init__(self, base_path="datalake/1_bronze"):
        self.base_path = Path(base_path)
        
        # Data configurations with their unique key columns
        self.data_configs = {
            'orderBook': {'unique_columns': ['ts'], 'path_pattern': '*'},
            'fundingRate': {'unique_columns': ['funding_time'], 'path_pattern': '*'},
            'trades': {'unique_columns': ['trade_id'], 'path_pattern': '*'},
            'indexPriceKlines': {'unique_columns': ['open_time'], 'path_pattern': '*/[0-9]*'},
            'markPriceKlines': {'unique_columns': ['open_time'], 'path_pattern': '*/[0-9]*'}
        }
    
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
                    paths.extend([tf for tf in instrument_dir.iterdir() 
                                if tf.is_dir() and tf.name[0].isdigit()])
            return paths
    
    def check_file_duplicates(self, file_path, data_type):
        """Check duplicates in a single parquet file."""
        try:
            unique_columns = self.data_configs[data_type]['unique_columns']
            df = pl.read_parquet(file_path)
            
            if df.is_empty():
                return {'status': 'empty', 'file': file_path.name, 'total_records': 0, 'duplicate_records': 0}
            
            total_records = len(df)
            missing_cols = [col for col in unique_columns if col not in df.columns]
            if missing_cols:
                return {'status': 'error', 'file': file_path.name, 'message': f'Missing columns: {missing_cols}', 'total_records': total_records}
            
            # Find duplicates using group_by for consistency
            grouped = df.group_by(unique_columns).len()
            duplicated_groups = grouped.filter(pl.col('len') > 1)
            
            duplicate_count = 0
            if len(duplicated_groups) > 0:
                duplicate_count = int(duplicated_groups['len'].sum() - len(duplicated_groups))
            
            return {
                'status': 'checked',
                'file': file_path.name,
                'total_records': total_records,
                'duplicate_records': duplicate_count
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path.name, 'message': str(e), 'traceback': traceback.format_exc()}
    
    def process_files_for_data_type(self, data_type, action, create_backup=False):
        """Process files for a specific data type."""
        data_paths = self.find_data_paths(data_type)
        if not data_paths:
            print(f"❌ No valid paths found for {data_type}")
            return []
        
        all_results = []
        for data_path in data_paths:
            path_name = str(data_path.relative_to(self.base_path))
            print(f"\n📁 Processing path: {path_name}")
            
            # Find parquet files
            parquet_files = [f for f in data_path.glob("*.parquet") 
                           if not f.name.endswith(('_backup.parquet', '.backup_before_dedup'))]
            parquet_files.sort()
            
            if not parquet_files:
                print("  No parquet files found")
                continue
            
            print(f"  📊 Found {len(parquet_files)} files")
            files_processed = 0
            
            for i, file_path in enumerate(parquet_files, 1):
                if i % 50 == 0 or i == len(parquet_files):
                    print(f"  Progress: {i}/{len(parquet_files)} ({i*100//len(parquet_files)}%)")
                
                if action == 'check':
                    result = self.check_file_duplicates(file_path, data_type)
                    if result.get('duplicate_records', 0) > 0:
                        files_processed += 1
                        print(f"  ⚠️  {file_path.name}: {result['duplicate_records']:,} duplicates in {result['total_records']:,} records")
                else:
                    result = self.remove_duplicates_from_file(file_path, data_type, create_backup)
                    if result.get('had_duplicates', False):
                        files_processed += 1
                
                all_results.append(result)
            
            summary_text = "files have duplicates" if action == 'check' else "files had duplicates"
            print(f"  📈 Path summary: {files_processed}/{len(parquet_files)} {summary_text}")
        
        return all_results
    
    def remove_duplicates_from_file(self, file_path, data_type, create_backup=False):
        """Remove duplicates from a single parquet file, keeping only one record."""
        try:
            unique_columns = self.data_configs[data_type]['unique_columns']
            df = pl.read_parquet(file_path)
            
            if df.is_empty():
                return {'status': 'empty', 'file': file_path.name, 'original_records': 0, 'removed_duplicates': 0}
            
            original_count = len(df)
            missing_cols = [col for col in unique_columns if col not in df.columns]
            if missing_cols:
                return {'status': 'error', 'file': file_path.name, 'message': f'Missing columns: {missing_cols}'}
            
            # Remove duplicates - keep first occurrence
            df_deduplicated = df.unique(subset=unique_columns, keep='first')
            final_count = len(df_deduplicated)
            removed_count = original_count - final_count
            
            if removed_count > 0:
                # Create backup only if requested
                if create_backup:
                    backup_path = file_path.with_suffix('.parquet.backup_before_dedup')
                    df.write_parquet(backup_path)
                
                df_deduplicated.write_parquet(file_path)
                print(f"    ✅ {file_path.name}: Removed {removed_count:,} duplicates (kept {final_count:,}/{original_count:,} records)")
            
            return {
                'status': 'processed',
                'file': file_path.name,
                'original_records': original_count,
                'final_records': final_count,
                'removed_duplicates': removed_count,
                'had_duplicates': removed_count > 0
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path.name, 'message': str(e), 'traceback': traceback.format_exc()}
    
    def check_and_remove_all_duplicates(self, create_backup=False):
        """Check duplicates first, then remove them without backup by default."""
        print("🔍🧹 CHECK AND REMOVE DUPLICATES FOR SILVER LAYER")
        print("=" * 80)
        print("Processing: orderBook → trades → fundingRate → indexPriceKlines → markPriceKlines")
        print()
        
        data_types = ['orderBook', 'trades', 'fundingRate', 'indexPriceKlines', 'markPriceKlines']
        start_time = datetime.now()
        
        for data_type in data_types:
            try:
                print(f"\n{'='*80}")
                print(f"🔍 CHECKING DUPLICATES: {data_type}")
                print(f"{'='*80}")
                
                # First check for duplicates
                check_results = self.process_files_for_data_type(data_type, 'check')
                files_with_dups = len([r for r in check_results if r.get('duplicate_records', 0) > 0])
                total_dups = sum(r.get('duplicate_records', 0) for r in check_results)
                
                if files_with_dups > 0:
                    print(f"\n🧹 REMOVING DUPLICATES: {data_type}")
                    print(f"Found {total_dups:,} duplicates in {files_with_dups} files. Removing now...")
                    
                    # Then remove duplicates
                    self.process_files_for_data_type(data_type, 'remove', create_backup)
                    print(f"✅ {data_type} completed: Removed duplicates from {files_with_dups} files")
                else:
                    print(f"✅ {data_type} completed: No duplicates found")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Interrupted by user during {data_type}")
                break
            except Exception as e:
                print(f"❌ Error processing {data_type}: {str(e)}")
                continue
        
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n🎉 CHECK AND REMOVE COMPLETED! Duration: {duration}")
    

def main():
    """Main function - automatically check and remove duplicates without backup."""
    checker = DuplicateChecker()
    print("🚀 Starting automatic duplicate check and removal (no backup)")
    checker.check_and_remove_all_duplicates(create_backup=False)

if __name__ == "__main__":
    main()