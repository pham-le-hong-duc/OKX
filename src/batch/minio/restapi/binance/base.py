"""
Base RestAPI class for Binance data backfill with gap detection.

This class provides 3-tier gap detection and filling:
1. Boundary gaps: Between consecutive files
2. Internal gaps: Within individual files
3. Recent gaps: From latest file to current time

Subclasses implement data-specific methods for API calls and transformations.
"""
import polars as pl
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod

from src.utils.s3_client import MinIOWriter


class RestAPI(ABC):
    """
    Base class for REST API data ingestion with gap detection and filling capabilities.
    
    Subclasses must implement 2 abstract methods:
    - get_api_data(start_date, end_date): Fetch data from API
    - transform_data(api_response): Transform API response to Polars DataFrame
    """
    
    def __init__(self, 
                 symbol,
                 data_type,
                 client_type="futures",
                 file_pattern="daily",
                 timestamp_field=None,
                 unique_field=None,
                 api_limit_days=None,
                 gap_threshold_ms=60000):
        """
        Initialize base REST API handler.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSDT")
            data_type: Type of data (e.g., "futures/um/daily/aggTrades", "futures/um/monthly/fundingRate")
            client_type: "futures" or "spot"
            file_pattern: "daily" or "monthly"
            timestamp_field: Name of timestamp field in data
            unique_field: Name of unique identifier field
            api_limit_days: API limit in days (None = full history, 365 = 1 year, 30 = 30 days)
            gap_threshold_ms: Gap threshold in milliseconds (default 60000 = 1 minute)
        """
        self.symbol = symbol
        self.data_type = data_type
        self.client_type = client_type
        self.file_pattern = file_pattern
        self.timestamp_field = timestamp_field
        self.unique_field = unique_field
        self.api_limit_days = api_limit_days
        self.gap_threshold_ms = gap_threshold_ms
        
        # Initialize MinIO writer
        self.minio_writer = MinIOWriter()
        
        self._init_binance_client()

    @staticmethod
    def _drop_embedded_header_rows(df: pl.DataFrame) -> pl.DataFrame:
        """Drop rows whose values are exactly the column names repeated."""
        if df.is_empty():
            return df

        header_checks = [
            pl.col(column).cast(pl.Utf8, strict=False).eq(pl.lit(column))
            for column in df.columns
        ]
        if not header_checks:
            return df

        return (
            df.with_columns(pl.all_horizontal(header_checks).alias("_is_embedded_header"))
            .filter(~pl.col("_is_embedded_header"))
            .drop("_is_embedded_header")
        )

    @staticmethod
    def _to_timestamp_ms(value):
        """Convert int/string/datetime timestamp values to epoch milliseconds."""
        if value is None:
            return None

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

            try:
                return int(value)
            except ValueError:
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)

        if hasattr(value, "timestamp"):
            return int(value.timestamp() * 1000)

        return int(value)

    @classmethod
    def _normalize_timestamp_value_to_ms(cls, value):
        """Normalize seconds/milliseconds/microseconds timestamps to epoch milliseconds."""
        timestamp = cls._to_timestamp_ms(value)
        if timestamp is None:
            return None

        abs_timestamp = abs(timestamp)
        if abs_timestamp >= cls.MICROSECOND_THRESHOLD:
            return timestamp // 1000
        if abs_timestamp < cls.SECOND_THRESHOLD:
            return timestamp * 1000
        return timestamp

    def _prepare_gap_detection_df(self, df: pl.DataFrame) -> pl.DataFrame | None:
        """Remove embedded headers and rows with invalid timestamps before gap detection."""
        if df is None or df.is_empty():
            return None

        df = self._drop_embedded_header_rows(df)
        if df.is_empty() or self.timestamp_field not in df.columns:
            return None

        timestamp_expr = pl.col(self.timestamp_field)
        if df.schema[self.timestamp_field] == pl.Utf8:
            timestamp_expr = (
                pl.when(pl.col(self.timestamp_field).str.contains(r"^\d+$"))
                .then(pl.col(self.timestamp_field).cast(pl.Int64, strict=False))
                .otherwise(
                    pl.col(self.timestamp_field)
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                    .dt.timestamp("ms")
                )
            )
        elif df.schema[self.timestamp_field] == pl.Datetime:
            timestamp_expr = pl.col(self.timestamp_field).dt.timestamp("ms")
        elif df.schema[self.timestamp_field] == pl.Date:
            timestamp_expr = pl.col(self.timestamp_field).cast(pl.Datetime).dt.timestamp("ms")
        else:
            timestamp_expr = pl.col(self.timestamp_field).cast(pl.Int64, strict=False)

        cleaned = (
            df.with_columns(timestamp_expr.alias(self.timestamp_field))
            .filter(pl.col(self.timestamp_field).is_not_null())
        )
        return cleaned if not cleaned.is_empty() else None
    
    def _init_binance_client(self):
        """Initialize Binance API client."""
        if self.client_type == "futures":
            from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
                DerivativesTradingUsdsFutures,
                ConfigurationRestAPI,
                DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
            )
            configuration_rest_api = ConfigurationRestAPI(
                api_key="",
                api_secret="",
                base_path=DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
                timeout=30000,  # 30 seconds timeout for API calls
            )
            self.client = DerivativesTradingUsdsFutures(config_rest_api=configuration_rest_api)
        else:  # spot
            from binance_sdk_spot.spot import (
                Spot,
                ConfigurationRestAPI,
                SPOT_REST_API_PROD_URL,
            )
            configuration_rest_api = ConfigurationRestAPI(
                api_key="",
                api_secret="",
                base_path=SPOT_REST_API_PROD_URL,
                timeout=30000,  # 30 seconds timeout for API calls
            )
            self.client = Spot(config_rest_api=configuration_rest_api)
    
    # ==================== Abstract Methods (must be implemented by subclasses) ====================
    
    @abstractmethod
    def get_api_data(self, start_date, end_date):
        """
        Fetch data from API for the given date range.
        
        This method can return either:
        1. A generator that yields batches of data (memory efficient)
        2. A list/dict of all data (legacy compatibility)
        
        Args:
            start_date: Start datetime
            end_date: End datetime
        
        Yields or Returns:
            Generator yielding batches of API data, or list/dict of all data
            
        Example (Generator pattern - recommended):
            def get_api_data(self, start_date, end_date):
                while True:
                    response = api.call()
                    if response:
                        yield response.data()  # Yield batch
                    if done:
                        break
        
        Example (Legacy pattern):
            def get_api_data(self, start_date, end_date):
                all_data = []
                # ... fetch all data
                return all_data
        """
    
    @abstractmethod
    def transform_data(self, api_response):
        """
        Transform API response to Polars DataFrame.
        
        Args:
            api_response: Raw API response
        
        Returns:
            Polars DataFrame
        """
    
    # ==================== Common Gap Detection Methods ====================
    
    def detect_gaps(self):
        """
        Detect gaps in data stored in MinIO.
        
        This method scans all parquet files in MinIO and detects 3 types of gaps:
        1. Boundary gaps: Between consecutive files
        2. Internal gaps: Within files
        3. Recent gaps: From latest file to now
        
        Uses self.gap_threshold_ms for gap detection.
        Only returns gaps within api_limit_days.
        
        Returns:
            List of gap dictionaries with structure:
            {
                'type': 'boundary' or 'internal',
                'start': start_timestamp_ms,
                'end': end_timestamp_ms,
                'duration_hours': gap_duration_in_hours,
                'between_files': 'file1.parquet → file2.parquet' (for boundary gaps),
                'inside_file': 'file.parquet' (for internal gaps)
            }
        """
        try:
            print(f"Starting gap detection...")
            
            # Calculate cutoff date based on API limit (filter early to avoid listing all files)
            cutoff_date = None
            cutoff_ms = 0
            if self.api_limit_days is not None:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.api_limit_days)
                cutoff_ms = int(cutoff_date.timestamp() * 1000)
                if self.file_pattern == 'daily':
                    cutoff_str = cutoff_date.strftime('%Y-%m-%d')
                else:  # monthly
                    cutoff_str = cutoff_date.strftime('%Y-%m')
                print(f"API limit: {self.api_limit_days} days (cutoff: {cutoff_str})")
            
            # List files in MinIO
            # data_type already includes full path with symbol
            prefix = f"{self.data_type}/"
            print(f"Listing files from MinIO: {prefix}")
            objects = self.minio_writer.client.list_objects(
                self.minio_writer.bucket,
                prefix=prefix,
                recursive=False
            )
            
            # Get parquet files and filter by date range
            file_dates = []
            total_files = 0
            for obj in objects:
                total_files += 1
                filename = obj.object_name.split('/')[-1]
                if filename.endswith('.parquet'):
                    date_str = filename.replace('.parquet', '')
                    
                    # Skip files older than API limit (optimization)
                    if cutoff_date is not None and date_str < cutoff_str:
                        continue
                    
                    file_dates.append((date_str, obj.object_name))
            
            print(f"Found {total_files} total files, {len(file_dates)} within range")
            
            if not file_dates:
                return []
            
            file_dates.sort(key=lambda x: x[0])
            
            # Detect all 3 tiers (recent first to ensure latest data is read)
            gaps = []
            print(f"Detecting recent gaps...")
            gaps.extend(self._detect_recent_gaps(file_dates, self.gap_threshold_ms))
            print(f"Detecting boundary gaps...")
            gaps.extend(self._detect_boundary_gaps_minio(file_dates, self.gap_threshold_ms))
            print(f"Detecting internal gaps...")
            gaps.extend(self._detect_internal_gaps_minio(file_dates, self.gap_threshold_ms))
            
            # Filter by API limit
            if self.api_limit_days is not None:
                gaps = [g for g in gaps if g['start'] >= cutoff_ms]
            
            return gaps
            
        except Exception as e:
            print(f"Error detecting gaps: {e}")
            return []
    
    def _detect_boundary_gaps_minio(self, file_dates, gap_threshold):
        """Detect gaps between consecutive files."""
        gaps = []
        timestamp_field = self.timestamp_field
        
        for i in range(len(file_dates) - 1):
            try:
                current_date, current_path = file_dates[i]
                next_date, next_path = file_dates[i + 1]
                
                # Read timestamp columns from MinIO
                current_df = self.minio_writer.read_parquet(current_path)
                next_df = self.minio_writer.read_parquet(next_path)
                current_df = self._prepare_gap_detection_df(current_df)
                next_df = self._prepare_gap_detection_df(next_df)
                
                if current_df is None or next_df is None:
                    continue
                
                # Get last timestamp of current file and first timestamp of next file
                current_end = current_df[timestamp_field].max()
                next_start = next_df[timestamp_field].min()
                if current_end is None or next_start is None:
                    continue
                
                # Auto-detect timestamp unit
                current_end_ms = self._normalize_timestamp_value_to_ms(current_end)
                next_start_ms = self._normalize_timestamp_value_to_ms(next_start)
                if current_end_ms is None or next_start_ms is None:
                    continue
                
                # Check if gap exceeds threshold
                gap_duration = next_start_ms - current_end_ms
                if gap_duration > gap_threshold:
                    gaps.append({
                        'type': 'boundary',
                        'start': current_end_ms + 1,
                        'end': next_start_ms - 1,
                        'duration_hours': gap_duration / (1000 * 3600),
                        'between_files': f"{current_date}.parquet → {next_date}.parquet"
                    })
                    
            except Exception as exc:
                print(f"Warning: failed boundary gap detection for {current_path}: {exc}")
                continue
        
        return gaps
    
    def _detect_internal_gaps_minio(self, file_dates, gap_threshold):
        """Detect gaps within files."""
        gaps = []
        timestamp_field = self.timestamp_field
        
        for date_str, object_path in file_dates:
            try:
                df = self.minio_writer.read_parquet(object_path)
                df = self._prepare_gap_detection_df(df)
                if df is None:
                    continue
                
                # Sample large files to avoid memory issues
                if len(df) > 10_000_000:
                    df = df.sample(n=50_000)
                
                # Get all timestamps
                timestamps = df[timestamp_field].sort().to_list()
                
                if not timestamps:
                    continue
                
                # Auto-detect timestamp unit (milliseconds vs microseconds)
                # Timestamps > 10^15 are likely microseconds (16+ digits)
                # Timestamps <= 10^15 are likely milliseconds (13 digits) or seconds (10 digits)
                first_ts = self._normalize_timestamp_value_to_ms(timestamps[0])
                if first_ts is None:
                    continue
                
                for i in range(len(timestamps) - 1):
                    ts_i_ms = self._normalize_timestamp_value_to_ms(timestamps[i])
                    ts_i1_ms = self._normalize_timestamp_value_to_ms(timestamps[i + 1])
                    if ts_i_ms is None or ts_i1_ms is None:
                        continue
                    
                    gap_duration = ts_i1_ms - ts_i_ms
                    if gap_duration > gap_threshold:
                        gaps.append({
                            'type': 'internal',
                            'start': ts_i_ms + 1,
                            'end': ts_i1_ms - 1,
                            'duration_hours': gap_duration / (1000 * 3600),
                            'inside_file': f"{date_str}.parquet"
                        })
                        
            except Exception as exc:
                print(f"Warning: failed internal gap detection for {object_path}: {exc}")
                continue
        
        return gaps
    
    def _detect_recent_gaps(self, file_dates, gap_threshold):
        """Detect gaps from latest file to now."""
        if not file_dates:
            return []
        
        gaps = []
        timestamp_field = self.timestamp_field
        
        try:
            latest_date, latest_path = file_dates[-1]
            latest_df = self.minio_writer.read_parquet(latest_path)
            latest_df = self._prepare_gap_detection_df(latest_df)
            
            if latest_df is None:
                return []
            
            latest_timestamp = latest_df[timestamp_field].max()
            if latest_timestamp is None:
                return []
            
            # Auto-detect timestamp unit
            latest_timestamp_ms = self._normalize_timestamp_value_to_ms(latest_timestamp)
            if latest_timestamp_ms is None:
                return []
            
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Apply buffer time to avoid race conditions with concurrent writes
            # Buffer = 3x gap_threshold to ensure data consistency
            buffer_ms = gap_threshold * 3
            safe_current_time_ms = current_time_ms - buffer_ms
            
            gap_duration = safe_current_time_ms - latest_timestamp_ms
            
            if gap_duration > gap_threshold:
                gaps.append({
                    'type': 'recent',
                    'start': latest_timestamp_ms + 1,
                    'end': safe_current_time_ms,
                    'duration_hours': gap_duration / (1000 * 3600),
                    'description': f"Gap from latest file ({latest_date}.parquet) to now (with {buffer_ms/1000}s buffer)"
                })
            
            return gaps
            
        except Exception as e:
            print(f"Error detecting recent gaps: {e}")
            return []
    
    def show_gaps(self, gaps):
        """Display detected gaps."""
        if not gaps:
            return
        
        print(f"Found {len(gaps)} gaps:")
        for i, gap in enumerate(gaps):
            start_dt = datetime.fromtimestamp(gap['start'] / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(gap['end'] / 1000, tz=timezone.utc)
            print(f"  {i+1}. [{gap['type']}] {start_dt} → {end_dt} ({gap['duration_hours']:.1f}h)")
    
    def fill_gaps(self, gaps):
        """
        Fill gaps by fetching from API using streaming pattern.
        
        Generator pattern enables memory-efficient processing of large datasets.
        """
        if not gaps:
            return 0
            
        total_filled = 0
        for i, gap in enumerate(gaps, 1):
            try:
                print(f"  [{i}/{len(gaps)}] Filling gap...", end=' ')
                
                start_date = datetime.fromtimestamp(gap['start'] / 1000, tz=timezone.utc)
                end_date = datetime.fromtimestamp(gap['end'] / 1000, tz=timezone.utc)
                data_generator = self.get_api_data(start_date, end_date)
                
                # Stream processing: process batches as they arrive
                gap_count = self._fill_gap_streaming(data_generator, start_date, end_date)
                
                if gap_count == 0:
                    print("No data")
                else:
                    total_filled += gap_count
                    print(f"OK ({gap_count} records)")
                
            except KeyboardInterrupt:
                print("\nInterrupted")
                return total_filled
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return total_filled
    
    def _fill_gap_streaming(self, data_generator, start_date, end_date):
        """
        Fill gap using streaming generator pattern.
        
        Args:
            data_generator: Generator yielding batches of API data
            start_date: Start datetime (for logging)
            end_date: End datetime (for logging)
        
        Returns:
            Total number of records filled
        """
        period_format = '%Y-%m-%d' if self.file_pattern == 'daily' else '%Y-%m'
        total_count = 0
        batch_num = 0
        
        for batch in data_generator:
            if not batch:
                continue
            
            batch_num += 1
            
            # Transform batch to DataFrame
            df = self.transform_data(batch)
            if df is None or len(df) == 0:
                continue
            
            # Add period column for grouping
            # Handle different timestamp types
            if df[self.timestamp_field].dtype == pl.Int64:
                # Int64: milliseconds → datetime → string
                df = df.with_columns(
                    pl.from_epoch(pl.col(self.timestamp_field), time_unit='ms')
                      .dt.strftime(period_format).alias('_period')
                )
            elif df[self.timestamp_field].dtype == pl.Utf8:
                # String: parse datetime string → datetime → format
                df = df.with_columns(
                    pl.col(self.timestamp_field).str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S')
                      .dt.strftime(period_format).alias('_period')
                )
            else:
                # Datetime: use strftime
                df = df.with_columns(
                    pl.col(self.timestamp_field).dt.strftime(period_format).alias('_period')
                )
            
            # Save each period immediately (streaming write)
            for period_str, period_df in df.group_by('_period'):
                period_df = period_df.drop('_period').sort(self.timestamp_field)
                new_records = self._save_to_minio(period_str[0], period_df)
                total_count += new_records  # Count only new records added
        return total_count
    
    def _save_to_minio(self, period_str, df):
        """
        Save to MinIO, merge with existing if present.
        
        Returns:
            Number of new records actually added to MinIO
        """
        # data_type already includes full path with symbol
        object_path = f"{self.data_type}/{period_str}.parquet"
        
        existing_df = self.minio_writer.read_parquet(object_path)
        if existing_df is not None:
            existing_count = len(existing_df)
            merged_df = pl.concat([existing_df, df]).unique(
                subset=[self.unique_field], 
                keep="last"
            ).sort(self.timestamp_field)
            self.minio_writer.write_parquet(merged_df, object_path)
            return len(merged_df) - existing_count  # Only count new records
        else:
            self.minio_writer.write_parquet(df, object_path)
            return len(df)  # All records are new
    
    def run(self):
        """Run 3-tier gap detection and filling."""
        print(f"Gap detection: {self.symbol} - {self.data_type}")
        
        gaps = self.detect_gaps()
        self.show_gaps(gaps)
        
        if gaps:
            print(f"Filling {len(gaps)} gaps...")
            total_filled = self.fill_gaps(gaps)
            print(f"Filled {total_filled} records\n")
        else:
            print("No gaps\n")

    SECOND_THRESHOLD = 100_000_000_000
    MICROSECOND_THRESHOLD = 10_000_000_000_000
