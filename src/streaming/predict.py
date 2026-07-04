"""
Realtime TimescaleDB consumer for dashboard predictions.

Flow:
- Wake every 20m boundary
- For active 1h/4h/1d boundaries, read exact featurestore rows from TimescaleDB
- Merge features, align with the trained model input schema, and predict trend
- Upsert into fixed dashboard.predict_<interval> tables
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import signal
import threading
import time

import joblib
import pandas as pd
import polars as pl

from src.utils.timescaledb_client import TimescaleDBClient

logging.Formatter.converter = time.gmtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeaturestoreSource:
    table_prefix: str
    time_column: str


class PredictConsumer:
    MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
    FEATURESTORE_SOURCES = (
        FeaturestoreSource("futures_klines", "open_time"),
        FeaturestoreSource("futures_metrics", "create_time"),
        FeaturestoreSource("futures_premiumindexklines", "open_time"),
        FeaturestoreSource("spot_klines", "open_time"),
        FeaturestoreSource("sentiment", "create_time"),
        FeaturestoreSource("futures_aggtrades", "create_time"),
    )
    BASIC_COLUMNS = [
        "futures_aggtrades_buy_price_mean",
        "futures_aggtrades_buy_price_min",
        "futures_aggtrades_buy_price_p25",
        "futures_aggtrades_buy_price_p50",
        "futures_aggtrades_buy_price_p75",
        "futures_aggtrades_buy_price_max",
        "futures_aggtrades_trade_price_mean",
        "futures_aggtrades_trade_price_min",
        "futures_aggtrades_trade_price_p25",
        "futures_aggtrades_trade_price_p50",
        "futures_aggtrades_trade_price_p75",
        "futures_aggtrades_trade_price_max",
        "futures_aggtrades_buy_quantity_mean",
        "futures_aggtrades_buy_quantity_min",
        "futures_aggtrades_buy_quantity_p25",
        "futures_aggtrades_buy_quantity_p50",
        "futures_aggtrades_buy_quantity_p75",
        "futures_aggtrades_buy_quantity_max",
        "futures_aggtrades_trade_quantity_mean",
        "futures_aggtrades_trade_quantity_min",
        "futures_aggtrades_trade_quantity_p25",
        "futures_aggtrades_trade_quantity_p50",
        "futures_aggtrades_trade_quantity_p75",
        "futures_aggtrades_trade_quantity_max",
        "futures_aggtrades_buy_rate_mean",
        "futures_aggtrades_trade_rate_mean",
        "futures_aggtrades_trade_count",
        "futures_aggtrades_buy_count",
        "futures_aggtrades_tickup_count",
        "futures_aggtrades_buy_vwap",
        "futures_aggtrades_trade_vwap",
        "futures_aggtrades_log_return_buy_price_min",
        "futures_aggtrades_log_return_buy_price_max",
        "futures_aggtrades_log_return_trade_price_min",
        "futures_aggtrades_log_return_trade_price_max",
        "futures_klines_open",
        "futures_klines_high",
        "futures_klines_low",
        "futures_klines_close",
        "futures_klines_trade_quantity",
        "futures_klines_trade_turnover",
        "futures_klines_trade_count",
        "futures_klines_buy_quantity",
        "futures_klines_buy_turnover",
        "futures_metrics_sum_open_interest_mean",
        "futures_metrics_sum_open_interest_min",
        "futures_metrics_sum_open_interest_p25",
        "futures_metrics_sum_open_interest_p50",
        "futures_metrics_sum_open_interest_p75",
        "futures_metrics_sum_open_interest_max",
        "futures_metrics_sum_open_interest_last",
        "futures_metrics_sum_open_interest_value_mean",
        "futures_metrics_sum_open_interest_value_min",
        "futures_metrics_sum_open_interest_value_p25",
        "futures_metrics_sum_open_interest_value_p50",
        "futures_metrics_sum_open_interest_value_p75",
        "futures_metrics_sum_open_interest_value_max",
        "futures_metrics_sum_open_interest_value_last",
        "spot_klines_trade_quantity",
        "spot_klines_trade_turnover",
        "spot_klines_trade_count",
        "spot_klines_buy_quantity",
        "spot_klines_buy_turnover",
        "sentiment_count",
    ]
    UNIQUE_VARIANCE_COLUMNS = [
        "futures_aggtrades_log_return_trade_quantity_min",
        "futures_aggtrades_log_return_buy_quantity_min",
    ]
    MODEL_PATHS = {
        "1h": MODEL_DIR / "model_1h.joblib",
        "4h": MODEL_DIR / "model_4h.joblib",
        "1d": MODEL_DIR / "model_1d.joblib",
    }
    INTERVALS = ["1h", "4h", "1d"]
    BOUNDARY_INTERVAL = "20m"
    RETRY_WAIT_SECONDS = 15

    def __init__(self) -> None:
        self.db_client = TimescaleDBClient()
        self._models: dict[str, object] = {}
        for interval in self.INTERVALS:
            self._get_model(interval)
        self.running = True
        self.total_aggregated = 0
        self.base_boundary_ms = self._parse_interval_to_ms(self.BOUNDARY_INTERVAL)
        self.next_boundary = self._get_next_boundary(
            int(datetime.now(timezone.utc).timestamp() * 1000),
            self.BOUNDARY_INTERVAL,
        )

    def _parse_interval_to_ms(self, interval: str) -> int:
        unit = interval[-1].lower()
        value = int(interval[:-1])
        if unit == "m":
            return value * 60 * 1000
        if unit == "h":
            return value * 60 * 60 * 1000
        if unit == "d":
            return value * 24 * 60 * 60 * 1000
        raise ValueError(f"Unsupported interval: {interval}")

    def _format_ts(self, ts_ms: int) -> str:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _get_next_boundary(self, current_ts_ms: int, interval: str) -> int:
        dt = datetime.fromtimestamp(current_ts_ms / 1000, tz=timezone.utc)
        step_ms = self._parse_interval_to_ms(interval)
        step_seconds = step_ms // 1000
        current_seconds = int(dt.timestamp())
        boundary_seconds = ((current_seconds // step_seconds) + 1) * step_seconds
        boundary_dt = datetime.fromtimestamp(boundary_seconds, tz=timezone.utc)
        return int(boundary_dt.timestamp() * 1000)

    def _should_aggregate_interval(self, boundary_ts_ms: int, interval: str) -> bool:
        dt = datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)
        if interval == "1h":
            return dt.minute == 0
        if interval == "4h":
            return dt.minute == 0 and dt.hour % 4 == 0
        if interval == "1d":
            return dt.minute == 0 and dt.hour == 0
        raise ValueError(f"Unsupported interval: {interval}")

    def _active_intervals_for_boundary(self, boundary_ts_ms: int) -> list[str]:
        return [
            interval
            for interval in self.INTERVALS
            if self._should_aggregate_interval(boundary_ts_ms, interval)
        ]

    def _load_exact_featurestore_row(
        self,
        *,
        table_name: str,
        time_column: str,
        target_time: datetime,
    ) -> pl.DataFrame | None:
        query = (
            f'SELECT * FROM featurestore.{table_name} '
            f'WHERE "{time_column}" = %s '
            "LIMIT 1"
        )
        with self.db_client.conn.cursor() as cur:
            cur.execute(query, (target_time,))
            data = cur.fetchall()
            if not data:
                return None
            columns = [desc[0] for desc in cur.description]
        return pl.DataFrame(data, schema=columns, orient="row")

    def _load_featurestore_inputs(
        self,
        *,
        interval: str,
        boundary_ts_ms: int,
    ) -> dict[str, pl.DataFrame] | None:
        boundary_dt = datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)
        interval_ms = self._parse_interval_to_ms(interval)
        open_dt = datetime.fromtimestamp(
            (boundary_ts_ms - interval_ms) / 1000,
            tz=timezone.utc,
        )

        loaded: dict[str, pl.DataFrame] = {}
        for source in self.FEATURESTORE_SOURCES:
            table_name = f"{source.table_prefix}_{interval}"
            target_time = open_dt if source.time_column == "open_time" else boundary_dt
            df = self._load_exact_featurestore_row(
                table_name=table_name,
                time_column=source.time_column,
                target_time=target_time,
            )
            if df is None or df.is_empty():
                return None
            loaded[source.table_prefix] = df

        return loaded

    def _merge_featurestore_inputs(
        self,
        *,
        interval: str,
        boundary_ts_ms: int,
        featurestore_inputs: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        boundary_dt = datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)
        merged_row: dict[str, object] = {
            "create_time": boundary_dt,
        }

        for source_name, df in featurestore_inputs.items():
            row = df.to_dicts()[0]
            for column_name, value in row.items():
                if column_name in {"create_time", "open_time", "close_time"}:
                    continue
                merged_row[f"{source_name}_{column_name}"] = value

        merged_df = pl.DataFrame([merged_row])
        merged_df = self._add_time_features(merged_df, interval)
        columns_to_drop = [
            column_name
            for column_name in (self.BASIC_COLUMNS + self.UNIQUE_VARIANCE_COLUMNS)
            if column_name in merged_df.columns
        ]
        return merged_df.drop(columns_to_drop) if columns_to_drop else merged_df

    def _add_time_features(self, df: pl.DataFrame, interval: str) -> pl.DataFrame:
        if df.is_empty():
            return df

        create_time = df["create_time"][0]
        if create_time.tzinfo is None:
            create_time = create_time.replace(tzinfo=timezone.utc)
        else:
            create_time = create_time.astimezone(timezone.utc)

        features: dict[str, float] = {}
        if interval in {"1h", "4h"}:
            hour_of_day = create_time.hour
            features["hour_of_day_sin"] = math.sin(2 * math.pi * hour_of_day / 24)
            features["hour_of_day_cos"] = math.cos(2 * math.pi * hour_of_day / 24)

        day_of_week = create_time.weekday()
        features["day_of_week_sin"] = math.sin(2 * math.pi * day_of_week / 7)
        features["day_of_week_cos"] = math.cos(2 * math.pi * day_of_week / 7)

        day_of_month = create_time.day - 1
        features["day_of_month_sin"] = math.sin(2 * math.pi * day_of_month / 31)
        features["day_of_month_cos"] = math.cos(2 * math.pi * day_of_month / 31)

        month_of_year = create_time.month - 1
        features["month_of_year_sin"] = math.sin(2 * math.pi * month_of_year / 12)
        features["month_of_year_cos"] = math.cos(2 * math.pi * month_of_year / 12)

        return df.with_columns(
            [pl.lit(value).alias(column_name) for column_name, value in features.items()]
        )

    def _get_model(self, interval: str) -> object:
        model = self._models.get(interval)
        if model is not None:
            return model

        model = joblib.load(self.MODEL_PATHS[interval])
        self._models[interval] = model
        return model

    def _prepare_model_input(
        self,
        *,
        interval: str,
        merged_df: pl.DataFrame,
    ) -> pd.DataFrame:
        model = self._get_model(interval)
        feature_names = list(getattr(model, "feature_names_in_", []))
        if not feature_names:
            raise ValueError(f"Model for interval {interval} has no feature_names_in_.")

        feature_df = merged_df.drop("create_time") if "create_time" in merged_df.columns else merged_df
        model_input = feature_df.to_pandas()
        model_input = model_input.reindex(columns=feature_names, fill_value=0.0)
        model_input = model_input.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return model_input

    def _build_prediction_df(
        self,
        *,
        interval: str,
        boundary_ts_ms: int,
        merged_df: pl.DataFrame,
    ) -> pl.DataFrame:
        model = self._get_model(interval)
        model_input = self._prepare_model_input(
            interval=interval,
            merged_df=merged_df,
        )
        prediction = int(model.predict(model_input)[0])

        return pl.DataFrame(
            [
                {
                    "create_time": datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc),
                    "trend": prediction,
                }
            ]
        )

    def _predict_interval(self, interval: str, boundary_ts_ms: int) -> bool:
        featurestore_inputs = self._load_featurestore_inputs(
            interval=interval,
            boundary_ts_ms=boundary_ts_ms,
        )
        if featurestore_inputs is None:
            return False

        merged_df = self._merge_featurestore_inputs(
            interval=interval,
            boundary_ts_ms=boundary_ts_ms,
            featurestore_inputs=featurestore_inputs,
        )
        prediction_df = self._build_prediction_df(
            interval=interval,
            boundary_ts_ms=boundary_ts_ms,
            merged_df=merged_df,
        )
        upserted = self.db_client.upsert_dataframe(
            prediction_df,
            table_name=f"predict_{interval}",
            key_column="create_time",
            schema_name="dashboard",
        )
        logger.info(
            f"  {interval:>3s} @ {self._format_ts(boundary_ts_ms)} - "
            f"Upserted {upserted} row(s) into dashboard.predict_{interval}"
        )
        self.total_aggregated += 1
        return True

    def consume(self) -> None:
        logger.info("=" * 60)
        logger.info("STARTING REAL-TIME PREDICTION CONSUMER")
        logger.info("=" * 60)
        logger.info(f"Intervals: {self.INTERVALS}")
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
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                if now_ms < self.next_boundary:
                    time.sleep(0.5)
                    continue

                active_intervals = self._active_intervals_for_boundary(self.next_boundary)
                if not active_intervals:
                    self.next_boundary += self.base_boundary_ms
                    continue

                processed_all = True
                for interval in active_intervals:
                    success = self._predict_interval(interval, self.next_boundary)
                    if not success:
                        processed_all = False
                        logger.warning(
                            f"  {interval:>3s} @ {self._format_ts(self.next_boundary)} - "
                            f"Featurestore rows are not ready yet, retrying in {self.RETRY_WAIT_SECONDS}s"
                        )
                        break

                if processed_all:
                    self.next_boundary += self.base_boundary_ms
                else:
                    time.sleep(self.RETRY_WAIT_SECONDS)

        except Exception as exc:
            logger.exception(f"FATAL ERROR: {exc}")
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        logger.info("=" * 60)
        logger.info("PREDICTION CONSUMER SHUTDOWN")
        logger.info("=" * 60)
        logger.info(f"Total Aggregated: {self.total_aggregated:,} windows")
        logger.info("=" * 60)
        try:
            self.db_client.close()
        except Exception as exc:
            logger.error(f"Failed to close TimescaleDB client: {exc}")

    def stop(self) -> None:
        self.running = False


def main() -> None:
    consumer = PredictConsumer()
    consumer.consume()


if __name__ == "__main__":
    main()
