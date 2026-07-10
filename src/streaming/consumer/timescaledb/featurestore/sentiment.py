"""
Realtime TimescaleDB consumer for Reddit sentiment featurestore aggregates.

Flow:
- Read reddit submissions, comments, and reddit-status from Redpanda
- Load recent submissions/comments history from MinIO
- Only process boundaries that have a matching reddit-status ping
- Aggregate daily sentiment features
- Load recent featurestore history from TimescaleDB
- Recompute feature columns for combined history + current batch
- Upsert only the current interval row into featurestore.sentiment_<interval>
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging

import polars as pl

from .base import Consumer

logger = logging.getLogger(__name__)


class SentimentConsumer(Consumer):
    ROLLING_WINDOWS = [4, 8, 16, 32]
    LAG_WINDOWS = [1, 2, 4, 8, 16, 32]
    TEMPORAL_FEATURE_SOURCE_COLUMNS = [
        "log_return_count",
        "score",
        "confidence",
        "pct_negative",
        "pct_positive",
        "pct_neutral",
    ]

    def __init__(self, **kwargs):
        self.pending_status_boundaries = set()
        super().__init__(
            topics=["reddit-submissions", "reddit-comments", "reddit-status"],
            group_id="timescaledb-featurestore-reddit-sentiment",
            data_type="reddit-sentiment",
            timestamp_field="event_ts_ms",
            intervals=["1h", "4h", "1d"],
            boundary_interval="10m",
            dedupe_columns=["source", "record_id"],
            warmup_messages=200,
            minio_bucket="reddit",
            historical_sources=[
                ("submissions", "submission"),
                ("comments", "comment"),
            ],
            schema_name="featurestore",
            key_column="create_time",
            timescaledb_history_rows=60,
            **kwargs,
        )

    def feature_steps(self):
        return [
            ("log_return", self.log_return),
            ("rolling", self.rolling),
            ("momentum", self.momentum),
            ("lag", self.lag),
        ]

    def transform_record(self, record, topic):
        if topic == "reddit-status":
            timestamp_utc = record.get("timestamp_utc")
            if timestamp_utc is None:
                return None

            timestamp_ms = int(timestamp_utc) * 1000
            boundary_ts_ms = (
                (timestamp_ms + self.base_boundary_ms - 1)
                // self.base_boundary_ms
            ) * self.base_boundary_ms
            self.pending_status_boundaries.add(boundary_ts_ms)
            return None

        if topic == "reddit-submissions":
            return self._normalize_submission_record(record)

        if topic == "reddit-comments":
            return self._normalize_comment_record(record)

        return None

    def transform_historical_df(self, df, source_name):
        if source_name == "submission":
            records = []
            for row in df.to_dicts():
                normalized = self._normalize_submission_record(row)
                if normalized is not None:
                    records.append(normalized)
            return pl.DataFrame(records) if records else pl.DataFrame()

        if source_name == "comment":
            records = []
            for row in df.to_dicts():
                normalized = self._normalize_comment_record(row)
                if normalized is not None:
                    records.append(normalized)
            return pl.DataFrame(records) if records else pl.DataFrame()

        return pl.DataFrame()

    def should_evaluate_boundaries_without_new_records(self):
        return bool(self.pending_status_boundaries)

    def _boundary_requires_status(self, boundary_ts_ms):
        return any(
            self._should_aggregate_interval(boundary_ts_ms, interval)
            for interval in self.intervals
        )

    def can_process_boundary(self, boundary_ts_ms, max_ts):
        if not self._boundary_requires_status(boundary_ts_ms):
            return True
        return boundary_ts_ms in self.pending_status_boundaries

    def boundary_ready_ts(self, max_ts):
        if not self.pending_status_boundaries:
            return max_ts
        return max(max_ts, max(self.pending_status_boundaries))

    def on_boundary_processed(self, boundary_ts_ms):
        self.pending_status_boundaries.discard(boundary_ts_ms)

    @staticmethod
    def _safe_zscore_expr(value, mean, std):
        return (
            pl.when(value.is_null() | mean.is_null() | std.is_null() | std.eq(0))
            .then(None)
            .otherwise((value - mean) / std)
        )

    def combine_history(self, aggregated_df, timescaledb_historical_df):
        return self._combine_history_with_batch(
            aggregated_df,
            timescaledb_historical_df,
            aggregated_df.columns,
        )

    def aggregate_window(self, df_window, window_ts, interval):
        total = len(df_window)
        if total == 0:
            return None

        sentiments = df_window["sentiment"].to_list()
        negative = sum(1 for value in sentiments if value == 0)
        neutral = sum(1 for value in sentiments if value == 1)
        positive = sum(1 for value in sentiments if value == 2)

        aggregated_df = pl.DataFrame(
            [
                {
                    "create_time": datetime.fromtimestamp(window_ts / 1000, tz=timezone.utc),
                    "count": total,
                    "score": (positive - negative) / total,
                    "confidence": (positive + negative) / total,
                    "pct_negative": negative / total,
                    "pct_positive": positive / total,
                    "pct_neutral": neutral / total,
                }
            ]
        )

        current_create_time = aggregated_df["create_time"][0]
        _, table_name = self.resolve_table_target(interval)
        timescaledb_historical_df = self._load_timescaledb_history(
            table_name=table_name,
            current_time=current_create_time,
            time_column="create_time",
            schema_name="featurestore",
        )
        self._log_irregular_history(timescaledb_historical_df, current_create_time, interval)

        combined_df = self.combine_history(aggregated_df, timescaledb_historical_df)
        result_df = self._run_feature_steps(combined_df)
        if result_df is None or result_df.is_empty():
            return None

        return result_df.filter(pl.col("create_time") == current_create_time).sort("create_time")

    def _log_irregular_history(self, history_df, current_time, interval):
        """
        Reddit sentiment is event-driven, so a quiet hour can legitimately leave
        gaps. Do not block the current row; rolling/lag features will be null
        until enough previous rows exist.
        """
        is_valid_history, history_reason = self._validate_timescaledb_history(
            history_df,
            current_time=current_time,
            interval=interval,
            time_column="create_time",
        )
        if not is_valid_history:
            logger.info(
                f"Featurestore sentiment {interval} @ {current_time}: "
                f"using partial/irregular history ({history_reason})"
            )

    def resolve_table_target(self, interval):
        return ("featurestore", f"sentiment_{interval}")

    def log_return(self, combined_df):
        if combined_df.is_empty():
            return combined_df

        return combined_df.with_columns(
            pl.when(pl.col("count") <= 0)
            .then(None)
            .otherwise(pl.col("count").cast(pl.Float64).log())
            .alias("log_return_count")
        )

    def rolling(self, combined_df):
        if combined_df.is_empty():
            return combined_df

        rolling_exprs = []
        temp_columns = []

        for window in self.ROLLING_WINDOWS:
            for column in self.TEMPORAL_FEATURE_SOURCE_COLUMNS:
                mean_col = f"_{column}_mean_{window}"
                std_col = f"{column}_rolling_std_{window}"
                temp_columns.append(mean_col)
                rolling_exprs.extend(
                    [
                        pl.col(column).rolling_mean(window_size=window).alias(mean_col),
                        pl.col(column).rolling_std(window_size=window).alias(std_col),
                    ]
                )

        combined_df = combined_df.with_columns(rolling_exprs)

        zscore_exprs = []
        for window in self.ROLLING_WINDOWS:
            for column in self.TEMPORAL_FEATURE_SOURCE_COLUMNS:
                zscore_exprs.append(
                    self._safe_zscore_expr(
                        pl.col(column),
                        pl.col(f"_{column}_mean_{window}"),
                        pl.col(f"{column}_rolling_std_{window}"),
                    ).alias(f"{column}_rolling_zscore_{window}")
                )

        return combined_df.with_columns(zscore_exprs).drop(temp_columns)

    def momentum(self, combined_df):
        if combined_df.is_empty():
            return combined_df

        return combined_df.with_columns(
            [
                (pl.col(column) - pl.col(column).shift(window)).alias(
                    f"{column}_momentum_{window}"
                )
                for window in self.ROLLING_WINDOWS
                for column in self.TEMPORAL_FEATURE_SOURCE_COLUMNS
            ]
        )

    def lag(self, combined_df):
        if combined_df.is_empty():
            return combined_df

        return combined_df.with_columns(
            [
                pl.col(column).shift(window).alias(f"{column}_lag_{window}")
                for window in self.LAG_WINDOWS
                for column in self.TEMPORAL_FEATURE_SOURCE_COLUMNS
            ]
        )

    def _normalize_submission_record(self, record):
        sentiment = record.get("sentiment")
        if sentiment is None or sentiment == "":
            return None

        created_utc = record.get("created_utc")
        if created_utc is None:
            return None

        return {
            "record_id": record.get("id"),
            "source": "submission",
            "event_ts_ms": int(created_utc) * 1000,
            "sentiment": int(sentiment),
        }

    def _normalize_comment_record(self, record):
        sentiment = record.get("sentiment")
        if sentiment is None or sentiment == "":
            return None

        created_utc = record.get("created_utc")
        if created_utc is None:
            return None

        return {
            "record_id": record.get("id"),
            "source": "comment",
            "event_ts_ms": int(created_utc) * 1000,
            "sentiment": int(sentiment),
        }


def main():
    consumer = SentimentConsumer()
    consumer.consume()


if __name__ == "__main__":
    main()
