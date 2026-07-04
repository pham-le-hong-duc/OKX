from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path

import joblib
import pandas as pd
import polars as pl

from .base import HistoricalTimescaleBatch, INTERVAL_TO_MS


@dataclass(frozen=True)
class FeaturestoreSource:
    table_prefix: str
    time_column: str


class PredictBatch(HistoricalTimescaleBatch):
    MODEL_DIR = Path(__file__).resolve().parents[3] / "model"
    """
    Dashboard prediction backfill fed by TimescaleDB featurestore tables.

    For each interval:
    - detect missing rows in dashboard.predict_<interval>
    - read matching rows from featurestore tables with the same interval
    - if every required featurestore row exists, emit one placeholder prediction

    Current output is still temporary:
    - create_time
    - trend=0
    """

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

    def __init__(self) -> None:
        super().__init__(
            schema_name="dashboard",
            time_column="create_time",
            intervals=["1h", "4h", "1d"],
            historical_sources=[],
            base_start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
            minio_bucket="binance",
        )
        self._models: dict[str, object] = {}

    def table_name(self, interval: str) -> str:
        return f"predict_{interval}"

    def aggregate_timestamps(
        self,
        interval: str,
        timestamps: list[int],
        historical_frames: dict[str, pl.DataFrame],
    ) -> pl.DataFrame | None:
        raise NotImplementedError("PredictBatch uses TimescaleDB featurestore tables directly.")

    def _align_boundary(self, ts_ms: int, interval_ms: int) -> int:
        return (ts_ms // interval_ms) * interval_ms

    def _expected_timestamps(self, interval: str) -> set[int]:
        interval_ms = INTERVAL_TO_MS[interval]
        start_ms = self._align_boundary(
            int(self.base_start_date.timestamp() * 1000),
            interval_ms,
        )
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        last_closed_boundary_ms = (now_ms // interval_ms) * interval_ms

        if last_closed_boundary_ms < start_ms:
            return set()

        return set(range(start_ms, last_closed_boundary_ms + interval_ms, interval_ms))

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _load_featurestore_range(
        self,
        *,
        table_name: str,
        time_column: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        query = (
            f'SELECT * FROM featurestore.{table_name} '
            f'WHERE "{time_column}" >= %s AND "{time_column}" <= %s '
            f'ORDER BY "{time_column}"'
        )

        with self._ts_client.conn.cursor() as cur:
            cur.execute(query, (start_time, end_time))
            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

        if not data:
            return pl.DataFrame(schema=columns)

        return pl.DataFrame(data, schema=columns, orient="row")

    def _load_featurestore_inputs_batch(
        self,
        *,
        interval: str,
        boundary_ts_list: list[int],
    ) -> dict[str, dict[datetime, dict[str, object]]]:
        if not boundary_ts_list:
            return {}

        interval_ms = INTERVAL_TO_MS[interval]
        boundary_dts = [
            datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)
            for boundary_ts_ms in boundary_ts_list
        ]
        open_dts = [
            datetime.fromtimestamp((boundary_ts_ms - interval_ms) / 1000, tz=timezone.utc)
            for boundary_ts_ms in boundary_ts_list
        ]

        loaded: dict[str, dict[datetime, dict[str, object]]] = {}
        for source in self.FEATURESTORE_SOURCES:
            table_name = f"{source.table_prefix}_{interval}"
            target_dts = open_dts if source.time_column == "open_time" else boundary_dts
            df = self._load_featurestore_range(
                table_name=table_name,
                time_column=source.time_column,
                start_time=min(target_dts),
                end_time=max(target_dts),
            )
            rows_by_time: dict[datetime, dict[str, object]] = {}
            if not df.is_empty():
                for row in df.to_dicts():
                    row_time = row.get(source.time_column)
                    if row_time is None:
                        continue
                    rows_by_time[self._normalize_datetime(row_time)] = row
            loaded[source.table_prefix] = rows_by_time

        return loaded

    def _merge_featurestore_inputs_batch(
        self,
        *,
        interval: str,
        boundary_ts_list: list[int],
        featurestore_inputs: dict[str, dict[datetime, dict[str, object]]],
    ) -> pl.DataFrame:
        interval_ms = INTERVAL_TO_MS[interval]
        merged_rows: list[dict[str, object]] = []

        for boundary_ts_ms in boundary_ts_list:
            boundary_dt = datetime.fromtimestamp(boundary_ts_ms / 1000, tz=timezone.utc)
            open_dt = datetime.fromtimestamp((boundary_ts_ms - interval_ms) / 1000, tz=timezone.utc)
            merged_row: dict[str, object] = {
                "create_time": boundary_dt,
            }

            missing_source = False
            for source in self.FEATURESTORE_SOURCES:
                source_name = source.table_prefix
                target_time = open_dt if source.time_column == "open_time" else boundary_dt
                row = featurestore_inputs.get(source_name, {}).get(target_time)
                if row is None:
                    missing_source = True
                    break

                for column_name, value in row.items():
                    if column_name in {"create_time", "open_time", "close_time"}:
                        continue
                    merged_row[f"{source_name}_{column_name}"] = value

            if not missing_source:
                merged_rows.append(merged_row)

        if not merged_rows:
            return pl.DataFrame()

        merged_df = pl.DataFrame(merged_rows).sort("create_time")
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

        exprs = [
            pl.col("create_time")
            .map_elements(
                lambda dt: math.sin(2 * math.pi * self._normalize_datetime(dt).weekday() / 7),
                return_dtype=pl.Float64,
            )
            .alias("day_of_week_sin"),
            pl.col("create_time")
            .map_elements(
                lambda dt: math.cos(2 * math.pi * self._normalize_datetime(dt).weekday() / 7),
                return_dtype=pl.Float64,
            )
            .alias("day_of_week_cos"),
            pl.col("create_time")
            .map_elements(
                lambda dt: math.sin(2 * math.pi * (self._normalize_datetime(dt).day - 1) / 31),
                return_dtype=pl.Float64,
            )
            .alias("day_of_month_sin"),
            pl.col("create_time")
            .map_elements(
                lambda dt: math.cos(2 * math.pi * (self._normalize_datetime(dt).day - 1) / 31),
                return_dtype=pl.Float64,
            )
            .alias("day_of_month_cos"),
            pl.col("create_time")
            .map_elements(
                lambda dt: math.sin(2 * math.pi * (self._normalize_datetime(dt).month - 1) / 12),
                return_dtype=pl.Float64,
            )
            .alias("month_of_year_sin"),
            pl.col("create_time")
            .map_elements(
                lambda dt: math.cos(2 * math.pi * (self._normalize_datetime(dt).month - 1) / 12),
                return_dtype=pl.Float64,
            )
            .alias("month_of_year_cos"),
        ]

        if interval in {"1h", "4h"}:
            exprs.extend(
                [
                    pl.col("create_time")
                    .map_elements(
                        lambda dt: math.sin(2 * math.pi * self._normalize_datetime(dt).hour / 24),
                        return_dtype=pl.Float64,
                    )
                    .alias("hour_of_day_sin"),
                    pl.col("create_time")
                    .map_elements(
                        lambda dt: math.cos(2 * math.pi * self._normalize_datetime(dt).hour / 24),
                        return_dtype=pl.Float64,
                    )
                    .alias("hour_of_day_cos"),
                ]
            )

        return df.with_columns(exprs)

    def _get_model(self, interval: str) -> object:
        model = self._models.get(interval)
        if model is not None:
            return model

        model_path = self.MODEL_PATHS[interval]
        model = joblib.load(model_path)
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

    def _build_prediction_rows(
        self,
        *,
        interval: str,
        merged_df: pl.DataFrame,
    ) -> pl.DataFrame:
        model = self._get_model(interval)
        model_input = self._prepare_model_input(
            interval=interval,
            merged_df=merged_df,
        )
        predictions = model.predict(model_input)

        return pl.DataFrame(
            {
                "create_time": merged_df["create_time"].to_list(),
                "trend": [int(prediction) for prediction in predictions],
            }
        ).sort("create_time")

    def fill_gaps(self) -> None:
        print(f"{'=' * 60}")
        print(f"HISTORICAL GAP FILL: {self.schema_name}")
        print(f"{'=' * 60}")

        for interval in self.intervals:
            date_groups = self.missing_ts.get(interval, {})
            if not date_groups:
                continue

            print(f"[{interval}] Processing {len(date_groups)} days")
            for date_str in sorted(date_groups.keys()):
                timestamps = sorted(date_groups[date_str])
                featurestore_inputs = self._load_featurestore_inputs_batch(
                    interval=interval,
                    boundary_ts_list=timestamps,
                )
                merged_df = self._merge_featurestore_inputs_batch(
                    interval=interval,
                    boundary_ts_list=timestamps,
                    featurestore_inputs=featurestore_inputs,
                )

                if merged_df.is_empty():
                    print(f"  {date_str}: no featurestore-aligned rows")
                    continue

                result_df = self._build_prediction_rows(
                    interval=interval,
                    merged_df=merged_df,
                )
                upserted = self._ts_client.upsert_dataframe(
                    result_df,
                    self.table_name(interval),
                    key_column=self.time_column,
                    schema_name=self.schema_name,
                )
                print(f"  {date_str}: upserted {upserted} rows")

        print(f"{'=' * 60}")
        print("HISTORICAL GAP FILL COMPLETED")
        print(f"{'=' * 60}")


def main() -> None:
    batch = PredictBatch()
    try:
        batch.detect_all_gaps_and_propagate()
        batch.fill_gaps()
    finally:
        batch.close()


if __name__ == "__main__":
    main()
