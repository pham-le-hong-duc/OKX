"""
Shared Redpanda -> TimescaleDB realtime aggregation consumer.

This keeps the old reference flow:
- load recent MinIO parquet files into RAM
- warm up with recent Kafka messages
- aggregate on UTC time boundaries
- upsert into fixed TimescaleDB tables

Unlike the old reference code, this consumer never auto-creates tables from
incoming data. Target tables must already exist.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone

import polars as pl
from kafka import KafkaConsumer

from src.utils.s3_client import MinIOWriter
from src.utils.timescaledb_client import TimescaleDBClient

logger = logging.getLogger(__name__)


class Consumer:
    """Base class for realtime TimescaleDB aggregation consumers."""
    SECOND_THRESHOLD = 100_000_000_000
    MICROSECOND_THRESHOLD = 10_000_000_000_000

    def __init__(
        self,
        timestamp_field,
        topic=None,
        data_type=None,
        topics=None,
        symbol=None,
        intervals=None,
        boundary_interval="1m",
        window_timestamp_mode="start",
        bootstrap_servers="redpanda:9092",
        group_id=None,
        historical_files_to_load=2,
        dedupe_columns=None,
        warmup_messages=1000,
        minio_bucket="binance",
        minio_prefix=None,
        historical_sources=None,
        schema_name="dashboard",
        key_column="ts_ms",
        max_poll_records=1000,
    ):
        self.topic = topic
        self.topics = topics or ([topic] if topic else [])
        if not self.topics:
            raise ValueError("At least one topic must be provided")
        self.data_type = data_type
        self.symbol = symbol.lower() if isinstance(symbol, str) else symbol
        self.timestamp_field = timestamp_field
        self.intervals = intervals or ["5m", "15m", "1h", "4h", "1d"]
        self.boundary_interval = boundary_interval
        if window_timestamp_mode not in {"start", "end"}:
            raise ValueError("window_timestamp_mode must be 'start' or 'end'")
        self.window_timestamp_mode = window_timestamp_mode
        self.historical_files_to_load = historical_files_to_load
        self.dedupe_columns = dedupe_columns
        self.warmup_messages = warmup_messages
        self.schema_name = schema_name
        self.key_column = key_column
        self.max_poll_records = max_poll_records
        self.minio_prefix = minio_prefix or self._default_minio_prefix()
        self.historical_sources = historical_sources or [(self.minio_prefix, None)]
        self.retention_buffer_ms = 5 * 60 * 1000
        self.base_boundary_ms = self._parse_interval_to_ms(self.boundary_interval)

        self.db_client = TimescaleDBClient()
        self.s3_client = MinIOWriter(bucket=minio_bucket)

        logger.info(f"Loading {historical_files_to_load} recent file(s) from MinIO...")
        self.df_historical = self._load_historical_from_s3()
        logger.info(f"Loaded {len(self.df_historical):,} records into RAM")

        self.current_date = self._initialize_current_date()
        self.next_boundary = self._get_next_boundary(
            int(datetime.now(timezone.utc).timestamp() * 1000),
            self.boundary_interval,
        )

        consumer_group_id = group_id or f"timescaledb-{self.topics[0]}"
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=consumer_group_id,
            value_deserializer=lambda message: json.loads(message.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=False,
            max_poll_records=self.max_poll_records,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            request_timeout_ms=40000,
            metadata_max_age_ms=300000,
            consumer_timeout_ms=-1,
        )
        self.consumer.subscribe(self.topics)

        self._wait_for_partition_assignment()
        self._warmup_recent_messages()
        self._trim_historical_to_active_windows()

        self.running = True
        self.total_consumed = 0
        self.total_aggregated = 0

    def _default_minio_prefix(self):
        return self.data_type

    @classmethod
    def _normalize_epoch_to_ms_expr(cls, column_name, alias=None):
        target_name = alias or column_name
        return (
            pl.when(pl.col(column_name).abs() >= cls.MICROSECOND_THRESHOLD)
            .then((pl.col(column_name) // 1000).cast(pl.Int64))
            .when(pl.col(column_name).abs() < cls.SECOND_THRESHOLD)
            .then((pl.col(column_name) * 1000).cast(pl.Int64))
            .otherwise(pl.col(column_name).cast(pl.Int64))
            .alias(target_name)
        )

    @classmethod
    def _normalize_epoch_value_to_ms(cls, value):
        if value is None:
            return None
        normalized = int(value)
        abs_value = abs(normalized)
        if abs_value >= cls.MICROSECOND_THRESHOLD:
            return normalized // 1000
        if abs_value < cls.SECOND_THRESHOLD:
            return normalized * 1000
        return normalized

    def _parse_interval_to_ms(self, interval):
        unit = interval[-1].lower()
        value = int(interval[:-1])

        if unit == "m":
            return value * 60 * 1000
        if unit == "h":
            return value * 60 * 60 * 1000
        if unit == "d":
            return value * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported interval: {interval}")

    def _initialize_current_date(self):
        if len(self.df_historical) == 0:
            return datetime.now(timezone.utc).date()
        max_ts = self.df_historical[self.timestamp_field].max()
        return self._get_date_utc(max_ts)

    def _wait_for_partition_assignment(self):
        partitions = []
        for _ in range(30):
            partitions = self.consumer.assignment()
            if partitions:
                break
            self.consumer.poll(timeout_ms=1000)

        if not partitions:
            raise RuntimeError("Failed to get partition assignment")

        self.partitions = partitions

    def _warmup_recent_messages(self):
        logger.info(
            f"Container startup - loading last {self.warmup_messages} messages for warmup"
        )
        for partition in self.partitions:
            self.consumer.seek_to_end(partition)
            end_offset = self.consumer.position(partition)
            target_offset = max(0, end_offset - self.warmup_messages)
            self.consumer.seek(partition, target_offset)

        warmup_records = []
        while True:
            messages = self.consumer.poll(
                timeout_ms=5000,
                max_records=self.warmup_messages,
            )
            if not messages:
                break

            for _, records in messages.items():
                for record in records:
                    normalized = self.transform_record(record.value, record.topic)
                    if normalized is not None:
                        warmup_records.append(normalized)

            if len(warmup_records) >= self.warmup_messages * len(self.partitions):
                break

        if not warmup_records:
            logger.info("No warmup records available")
            return

        df_warmup = pl.DataFrame(warmup_records)
        self.df_historical = self._append_and_dedupe(self.df_historical, df_warmup)
        logger.info(f"Loaded {len(warmup_records):,} warmup records")

    def _load_historical_from_s3(self):
        dfs = []
        for prefix, source_name in self.historical_sources:
            all_files = sorted(
                self.s3_client.list_objects(prefix=prefix, recursive=True)
            )
            parquet_files = [path for path in all_files if path.endswith(".parquet")]

            if not parquet_files:
                continue

            recent_files = parquet_files[-self.historical_files_to_load :]
            for file_path in recent_files:
                df = self.s3_client.read_parquet(file_path)
                if df is not None and len(df) > 0:
                    df = self.transform_historical_df(df, source_name or prefix)
                    if df is not None and len(df) > 0:
                        dfs.append(df)

        if not dfs:
            logger.info("No MinIO parquet files found, starting with empty DataFrame")
            return pl.DataFrame()

        return pl.concat(dfs, how="vertical_relaxed").sort(self.timestamp_field)

    def _append_and_dedupe(self, df_existing, df_new):
        if len(df_existing) == 0:
            combined = df_new
        else:
            combined = pl.concat([df_existing, df_new], how="vertical_relaxed")

        if self.dedupe_columns:
            combined = combined.unique(
                subset=self.dedupe_columns,
                keep="last",
                maintain_order=False,
            )

        return combined.sort(self.timestamp_field)

    def _get_date_utc(self, ts_ms):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.date()

    def _get_next_boundary(self, current_ts_ms, interval):
        """Get the next UTC boundary timestamp for the configured base interval."""
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        step_ms = self._parse_interval_to_ms(interval)
        step_seconds = step_ms // 1000
        current_seconds = int(dt.timestamp())
        boundary_seconds = ((current_seconds // step_seconds) + 1) * step_seconds
        boundary_dt = datetime.fromtimestamp(boundary_seconds, tz=timezone.utc)
        return int(boundary_dt.timestamp() * 1000)

    def _get_window_size_ms(self, interval):
        sizes = {
            "1m": 1 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        if interval not in sizes:
            raise ValueError(f"Unsupported interval: {interval}")
        return sizes[interval]

    def _should_aggregate_interval(self, boundary_ts_ms, interval):
        """Return True when an interval should be emitted at the given boundary."""
        if interval == "1m":
            return True

        dt = datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)

        if interval == "5m":
            return dt.minute % 5 == 0
        if interval == "15m":
            return dt.minute % 15 == 0
        if interval == "1h":
            return dt.minute == 0
        if interval == "4h":
            return dt.minute == 0 and dt.hour % 4 == 0
        if interval == "1d":
            return dt.minute == 0 and dt.hour == 0

        raise ValueError(f"Unsupported interval: {interval}")

    def _active_intervals_for_boundary(self, boundary_ts_ms):
        return [
            interval
            for interval in self.intervals
            if self._should_aggregate_interval(boundary_ts_ms, interval)
        ]

    def _trim_historical_to_active_windows(self):
        """
        Keep only the data needed for the largest active aggregation window,
        plus a small safety buffer.
        """
        if len(self.df_historical) == 0:
            return

        max_window_size_ms = max(self._get_window_size_ms(interval) for interval in self.intervals)
        cutoff_ts = self.next_boundary - max_window_size_ms - self.retention_buffer_ms
        self.df_historical = self.df_historical.filter(
            pl.col(self.timestamp_field) >= cutoff_ts
        )


    def _format_ts(self, ts_ms):
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _slice_window(self, window_start, window_end):
        if self.window_timestamp_mode == "end":
            return self.df_historical.filter(
                (pl.col(self.timestamp_field) > window_start)
                & (pl.col(self.timestamp_field) <= window_end)
            )

        return self.df_historical.filter(
            (pl.col(self.timestamp_field) >= window_start)
            & (pl.col(self.timestamp_field) < window_end)
        )

    def aggregate_window(self, df_window, window_ts, interval):
        """Subclasses must convert one window into an aggregated DataFrame."""
        raise NotImplementedError("Subclass must implement aggregate_window()")

    def transform_record(self, record, topic):
        """Normalize one raw Kafka record into the in-memory schema."""
        return record

    def transform_historical_df(self, df, source_name):
        """Normalize one historical parquet dataframe into the in-memory schema."""
        return df

    def can_process_boundary(self, boundary_ts_ms, max_ts):
        """Hook for subclasses to delay aggregation until external conditions are met."""
        return True

    def boundary_ready_ts(self, max_ts):
        """
        Hook for subclasses that can safely advance boundaries based on external
        control signals instead of only raw event timestamps.
        """
        return max_ts

    def should_evaluate_boundaries_without_new_records(self):
        """Hook for subclasses that can be triggered by control/status topics."""
        return False

    def on_boundary_processed(self, boundary_ts_ms):
        """Hook called after a boundary has been processed and before advancing."""
        return

    def resolve_table_target(self, interval):
        """
        Subclasses must return the fixed target table for an interval.

        Returns:
            tuple[str, str]: (schema_name, table_name)
        """
        raise NotImplementedError("Subclass must implement resolve_table_target()")

    def consume(self):
        logger.info("=" * 60)
        logger.info("STARTING REAL-TIME AGGREGATION CONSUMER")
        logger.info("=" * 60)
        logger.info(f"Topic: {self.topic}")
        logger.info(f"Data Type: {self.data_type}")
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Intervals: {self.intervals}")
        logger.info(f"Current RAM: {len(self.df_historical):,} records")
        logger.info(f"Current Date: {self.current_date}")
        logger.info(f"Next Boundary: {self._format_ts(self.next_boundary)}")
        logger.info("=" * 60)

        def signal_handler(signum, frame):
            logger.info("Shutdown signal received")
            self.running = False

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                messages = self.consumer.poll(timeout_ms=0)
                if not messages:
                    time.sleep(0.001)
                    continue

                polled_any_records = any(records for records in messages.values())

                new_records = []
                for _, records in messages.items():
                    for record in records:
                        normalized = self.transform_record(record.value, record.topic)
                        if normalized is not None:
                            new_records.append(normalized)

                if new_records:
                    self.total_consumed += len(new_records)
                    df_new = pl.DataFrame(new_records)

                    self.df_historical = self._append_and_dedupe(self.df_historical, df_new)

                elif not self.should_evaluate_boundaries_without_new_records():
                    continue

                if len(self.df_historical) == 0:
                    continue

                max_ts = self.df_historical[self.timestamp_field].max()
                ready_ts = self.boundary_ready_ts(max_ts)
                active_intervals = self._active_intervals_for_boundary(self.next_boundary)
                boundary_ready_ts = (
                    ready_ts
                    if active_intervals
                    else int(datetime.now(timezone.utc).timestamp() * 1000)
                )
                if boundary_ready_ts < self.next_boundary:
                    if polled_any_records:
                        self.consumer.commit()
                    continue

                while boundary_ready_ts >= self.next_boundary:
                    active_intervals = self._active_intervals_for_boundary(self.next_boundary)
                    if active_intervals and not self.can_process_boundary(self.next_boundary, max_ts):
                        break
                    for interval in active_intervals:
                        window_ts = self.next_boundary
                        try:
                            window_size_ms = self._get_window_size_ms(interval)
                            window_end = self.next_boundary
                            window_start = window_end - window_size_ms

                            df_window = self._slice_window(window_start, window_end)

                            if len(df_window) == 0:
                                continue

                            process_start = time.perf_counter()
                            aggregate_start = time.perf_counter()
                            aggregated = self.aggregate_window(df_window, window_ts, interval)
                            aggregate_seconds = time.perf_counter() - aggregate_start
                            if aggregated is None or len(aggregated) == 0:
                                logger.warning(
                                    f"  {interval:>3s} @ {self._format_ts(window_ts)} "
                                    " - Aggregation failed"
                                )
                                continue

                            schema_name, table_name = self.resolve_table_target(interval)

                            current_time_ms = int(
                                datetime.now(timezone.utc).timestamp() * 1000
                            )
                            latency_seconds = (current_time_ms - max_ts) / 1000

                            db_write_start = time.perf_counter()
                            self.db_client.upsert_dataframe(
                                aggregated,
                                table_name=table_name,
                                key_column=self.key_column,
                                schema_name=schema_name,
                            )
                            db_write_seconds = time.perf_counter() - db_write_start
                            process_seconds = time.perf_counter() - process_start
                            logger.info(
                                f"  {interval:>3s} @ {self._format_ts(window_ts)} - "
                                f"Aggregated {len(df_window):,} records -> "
                                f"Upserted {len(aggregated)} row(s) into "
                                f"{schema_name}.{table_name} | "
                                f"Data lag: {latency_seconds:.2f}s | "
                                f"Aggregate: {aggregate_seconds:.3f}s | "
                                f"DB write: {db_write_seconds:.3f}s | "
                                f"Total process: {process_seconds:.3f}s"
                            )
                            self.total_aggregated += 1
                        except Exception as exc:
                            logger.error(
                                f"  {interval:>3s} @ {self._format_ts(window_ts)} - Error: {exc}"
                            )
                            logger.exception("Interval aggregation failed")

                    self.on_boundary_processed(self.next_boundary)
                    self.next_boundary += self.base_boundary_ms
                    ready_ts = self.boundary_ready_ts(max_ts)
                    active_intervals = self._active_intervals_for_boundary(self.next_boundary)
                    boundary_ready_ts = (
                        ready_ts
                        if active_intervals
                        else int(datetime.now(timezone.utc).timestamp() * 1000)
                    )

                self._trim_historical_to_active_windows()

                max_date = self._get_date_utc(max_ts)
                if max_date > self.current_date:
                    cutoff_date = max_date - timedelta(days=1)
                    cutoff_ts = int(
                        datetime.combine(cutoff_date, datetime.min.time())
                        .replace(tzinfo=timezone.utc)
                        .timestamp()
                        * 1000
                    )

                    before_trim = len(self.df_historical)
                    self.df_historical = self.df_historical.filter(
                        pl.col(self.timestamp_field) >= cutoff_ts
                    )
                    self.current_date = max_date

                self.consumer.commit()

        except Exception as exc:
            logger.exception(f"FATAL ERROR: {exc}")
        finally:
            self._shutdown()

    def _shutdown(self):
        logger.info("=" * 60)
        logger.info("CONSUMER SHUTDOWN")
        logger.info("=" * 60)
        logger.info(f"Total Consumed: {self.total_consumed:,} records")
        logger.info(f"Total Aggregated: {self.total_aggregated:,} windows")
        logger.info("=" * 60)

        try:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        except Exception as exc:
            logger.error(f"Failed to close Kafka consumer: {exc}")

        try:
            self.db_client.close()
        except Exception as exc:
            logger.error(f"Failed to close TimescaleDB client: {exc}")

        logger.info("Shutdown complete.")

    def stop(self):
        self.running = False
