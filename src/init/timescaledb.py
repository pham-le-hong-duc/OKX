import sys

from src.utils.timescaledb_client import TimescaleDBClient, wait_for_timescaledb


SCHEMAS = ("dashboard", "featurestore")
KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
METRICS_TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
SENTIMENT_TIMEFRAMES = ("1h", "4h", "1d")
PREDICT_TIMEFRAMES = ("1h", "4h", "1d")
FEATURESTORE_ROLLING_WINDOWS = (4, 8, 16, 32)
FEATURESTORE_LAG_WINDOWS = (1, 2, 4, 8, 16, 32)
FEATURESTORE_TIMEFRAMES = ("1h", "4h", "1d")
FEATURESTORE_TEMPORAL_SOURCE_COLUMNS = (
    "relative_range",
    "imbalance_buy_quantity",
    "imbalance_buy_turnover",
    "log_return_close",
    "log_return_trade_quantity",
    "log_return_trade_turnover",
    "log_return_trade_count",
    "log_return_buy_quantity",
    "log_return_buy_turnover",
)


def ensure_hypertable(
    client: TimescaleDBClient,
    *,
    schema_name: str,
    table_name: str,
    time_column: str,
    columns: list[tuple[str, str]],
) -> None:
    columns_sql = ",\n        ".join(
        f"{column_name} {column_type}"
        for column_name, column_type in columns
    )

    client.ensure_table(
        schema_name=schema_name,
        table_name=table_name,
        columns_sql=columns_sql,
        hypertable_time_column=time_column,
    )

    for column_name, column_type in columns[1:]:
        client.execute(
            f"""
            ALTER TABLE {schema_name}.{table_name}
            ADD COLUMN IF NOT EXISTS {column_name} {column_type}
            """
        )


def initialize_schemas(client: TimescaleDBClient) -> None:
    for schema_name in SCHEMAS:
        if not client.create_schema(schema_name):
            raise RuntimeError(f"Failed to create schema '{schema_name}'")


def initialize_dashboard_tables(client: TimescaleDBClient) -> None:
    create_dashboard_futures_klines_tables(client)
    create_dashboard_futures_metrics_tables(client)
    create_dashboard_sentiment_tables(client)
    create_dashboard_predict_tables(client)


def initialize_featurestore_tables(client: TimescaleDBClient) -> None:
    create_featurestore_futures_klines_table(client)
    create_featurestore_futures_metrics_table(client)
    create_featurestore_futures_premiumindexklines_table(client)
    create_featurestore_spot_klines_table(client)
    create_featurestore_sentiment_table(client)
    create_featurestore_futures_aggtrades_table(client)


def create_dashboard_futures_klines_tables(client: TimescaleDBClient) -> None:
    columns = [
        ("open_time", "TIMESTAMP PRIMARY KEY"),
        ("close_time", "TIMESTAMP"),
        ("open", "DOUBLE PRECISION"),
        ("high", "DOUBLE PRECISION"),
        ("low", "DOUBLE PRECISION"),
        ("close", "DOUBLE PRECISION"),
    ]
    for timeframe in KLINE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="dashboard",
            table_name=f"futures_klines_{timeframe}",
            time_column="open_time",
            columns=columns,
        )


def create_dashboard_futures_metrics_tables(client: TimescaleDBClient) -> None:
    columns = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("sum_open_interest", "DOUBLE PRECISION"),
        ("sum_open_interest_value", "DOUBLE PRECISION"),
        ("count_toptrader_long_short_ratio", "DOUBLE PRECISION"),
        ("sum_toptrader_long_short_ratio", "DOUBLE PRECISION"),
        ("count_long_short_ratio", "DOUBLE PRECISION"),
        ("sum_taker_long_short_vol_ratio", "DOUBLE PRECISION"),
    ]
    for timeframe in METRICS_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="dashboard",
            table_name=f"futures_metrics_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def create_dashboard_sentiment_tables(client: TimescaleDBClient) -> None:
    columns = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("word_frequency", "JSONB"),
        ("count", "BIGINT"),
        ("score", "DOUBLE PRECISION"),
        ("confidence", "DOUBLE PRECISION"),
        ("pct_negative", "DOUBLE PRECISION"),
        ("pct_positive", "DOUBLE PRECISION"),
        ("pct_neutral", "DOUBLE PRECISION"),
    ]
    for timeframe in SENTIMENT_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="dashboard",
            table_name=f"sentiment_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def create_dashboard_predict_tables(client: TimescaleDBClient) -> None:
    columns = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("trend", "INTEGER"),
    ]
    for timeframe in PREDICT_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="dashboard",
            table_name=f"predict_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def build_featurestore_futures_klines_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("open_time", "TIMESTAMP PRIMARY KEY"),
        ("close_time", "TIMESTAMP"),
        ("open", "DOUBLE PRECISION"),
        ("high", "DOUBLE PRECISION"),
        ("low", "DOUBLE PRECISION"),
        ("close", "DOUBLE PRECISION"),
        ("trade_quantity", "DOUBLE PRECISION"),
        ("trade_turnover", "DOUBLE PRECISION"),
        ("trade_count", "DOUBLE PRECISION"),
        ("buy_quantity", "DOUBLE PRECISION"),
        ("buy_turnover", "DOUBLE PRECISION"),
        ("relative_range", "DOUBLE PRECISION"),
        ("imbalance_buy_quantity", "DOUBLE PRECISION"),
        ("imbalance_buy_turnover", "DOUBLE PRECISION"),
        ("body_percentage", "DOUBLE PRECISION"),
        ("upper_wick_percentage", "DOUBLE PRECISION"),
        ("lower_wick_percentage", "DOUBLE PRECISION"),
        ("gap_percentage", "DOUBLE PRECISION"),
        ("ema20", "DOUBLE PRECISION"),
        ("ema50", "DOUBLE PRECISION"),
        ("macd", "DOUBLE PRECISION"),
        ("macd_signal", "DOUBLE PRECISION"),
        ("atr14", "DOUBLE PRECISION"),
        ("rsi14", "DOUBLE PRECISION"),
        ("macd_histogram", "DOUBLE PRECISION"),
        ("log_return_open", "DOUBLE PRECISION"),
        ("log_return_high", "DOUBLE PRECISION"),
        ("log_return_low", "DOUBLE PRECISION"),
        ("log_return_close", "DOUBLE PRECISION"),
        ("log_return_trade_quantity", "DOUBLE PRECISION"),
        ("log_return_trade_turnover", "DOUBLE PRECISION"),
        ("log_return_trade_count", "DOUBLE PRECISION"),
        ("log_return_buy_quantity", "DOUBLE PRECISION"),
        ("log_return_buy_turnover", "DOUBLE PRECISION"),
    ]

    for window in FEATURESTORE_ROLLING_WINDOWS:
        columns.extend(
            [
                (f"relative_range_rolling_std_{window}", "DOUBLE PRECISION"),
                (f"relative_range_rolling_zscore_{window}", "DOUBLE PRECISION"),
                (f"imbalance_buy_quantity_rolling_std_{window}", "DOUBLE PRECISION"),
                (f"imbalance_buy_quantity_rolling_zscore_{window}", "DOUBLE PRECISION"),
                (f"imbalance_buy_turnover_rolling_std_{window}", "DOUBLE PRECISION"),
                (f"imbalance_buy_turnover_rolling_zscore_{window}", "DOUBLE PRECISION"),
                (f"log_return_close_rolling_mean_{window}", "DOUBLE PRECISION"),
                (f"log_return_close_rolling_std_{window}", "DOUBLE PRECISION"),
                (f"log_return_close_rolling_zscore_{window}", "DOUBLE PRECISION"),
            ]
        )

        for column in (
            "log_return_trade_quantity",
            "log_return_trade_turnover",
            "log_return_trade_count",
            "log_return_buy_quantity",
            "log_return_buy_turnover",
        ):
            columns.extend(
                [
                    (f"{column}_rolling_std_{window}", "DOUBLE PRECISION"),
                    (f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"),
                ]
            )

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in FEATURESTORE_TEMPORAL_SOURCE_COLUMNS:
            columns.append((f"{column}_momentum_{window}", "DOUBLE PRECISION"))

    for window in FEATURESTORE_LAG_WINDOWS:
        for column in FEATURESTORE_TEMPORAL_SOURCE_COLUMNS:
            columns.append((f"{column}_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_futures_klines_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_futures_klines_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"futures_klines_{timeframe}",
            time_column="open_time",
            columns=columns,
        )


def build_featurestore_futures_metrics_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("sum_open_interest_mean", "DOUBLE PRECISION"),
        ("sum_open_interest_std", "DOUBLE PRECISION"),
        ("sum_open_interest_min", "DOUBLE PRECISION"),
        ("sum_open_interest_p25", "DOUBLE PRECISION"),
        ("sum_open_interest_p50", "DOUBLE PRECISION"),
        ("sum_open_interest_p75", "DOUBLE PRECISION"),
        ("sum_open_interest_max", "DOUBLE PRECISION"),
        ("sum_open_interest_skew", "DOUBLE PRECISION"),
        ("sum_open_interest_kurtosis", "DOUBLE PRECISION"),
        ("sum_open_interest_last", "DOUBLE PRECISION"),
        ("sum_open_interest_value_mean", "DOUBLE PRECISION"),
        ("sum_open_interest_value_std", "DOUBLE PRECISION"),
        ("sum_open_interest_value_min", "DOUBLE PRECISION"),
        ("sum_open_interest_value_p25", "DOUBLE PRECISION"),
        ("sum_open_interest_value_p50", "DOUBLE PRECISION"),
        ("sum_open_interest_value_p75", "DOUBLE PRECISION"),
        ("sum_open_interest_value_max", "DOUBLE PRECISION"),
        ("sum_open_interest_value_skew", "DOUBLE PRECISION"),
        ("sum_open_interest_value_kurtosis", "DOUBLE PRECISION"),
        ("sum_open_interest_value_last", "DOUBLE PRECISION"),
        ("count_toptrader_long_short_ratio_mean", "DOUBLE PRECISION"),
        ("count_toptrader_long_short_ratio_std", "DOUBLE PRECISION"),
        ("count_toptrader_long_short_ratio_last", "DOUBLE PRECISION"),
        ("sum_toptrader_long_short_ratio_mean", "DOUBLE PRECISION"),
        ("sum_toptrader_long_short_ratio_std", "DOUBLE PRECISION"),
        ("sum_toptrader_long_short_ratio_last", "DOUBLE PRECISION"),
        ("count_long_short_ratio_mean", "DOUBLE PRECISION"),
        ("count_long_short_ratio_std", "DOUBLE PRECISION"),
        ("count_long_short_ratio_last", "DOUBLE PRECISION"),
        ("sum_taker_long_short_vol_ratio_mean", "DOUBLE PRECISION"),
        ("sum_taker_long_short_vol_ratio_std", "DOUBLE PRECISION"),
        ("sum_taker_long_short_vol_ratio_last", "DOUBLE PRECISION"),
    ]

    log_return_columns = (
        "sum_open_interest_mean",
        "sum_open_interest_min",
        "sum_open_interest_p25",
        "sum_open_interest_p50",
        "sum_open_interest_p75",
        "sum_open_interest_max",
        "sum_open_interest_last",
        "sum_open_interest_value_mean",
        "sum_open_interest_value_min",
        "sum_open_interest_value_p25",
        "sum_open_interest_value_p50",
        "sum_open_interest_value_p75",
        "sum_open_interest_value_max",
        "sum_open_interest_value_last",
    )
    for column in log_return_columns:
        columns.append((f"log_return_{column}", "DOUBLE PRECISION"))

    rolling_mean_columns = (
        "log_return_sum_open_interest_last",
        "log_return_sum_open_interest_value_last",
    )
    rolling_std_zscore_columns = (
        "log_return_sum_open_interest_last",
        "log_return_sum_open_interest_value_last",
        "count_toptrader_long_short_ratio_last",
        "sum_toptrader_long_short_ratio_last",
        "count_long_short_ratio_last",
        "sum_taker_long_short_vol_ratio_last",
    )

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in rolling_std_zscore_columns:
            columns.append((f"{column}_rolling_std_{window}", "DOUBLE PRECISION"))
            if column in rolling_mean_columns:
                columns.append((f"{column}_rolling_mean_{window}", "DOUBLE PRECISION"))
            columns.append((f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"))

    temporal_columns = (
        "log_return_sum_open_interest_last",
        "log_return_sum_open_interest_value_last",
        "count_toptrader_long_short_ratio_last",
        "sum_toptrader_long_short_ratio_last",
        "count_long_short_ratio_last",
        "sum_taker_long_short_vol_ratio_last",
    )

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in temporal_columns:
            columns.append((f"{column}_momentum_{window}", "DOUBLE PRECISION"))

    for window in FEATURESTORE_LAG_WINDOWS:
        for column in temporal_columns:
            columns.append((f"{column}_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_futures_metrics_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_futures_metrics_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"futures_metrics_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def build_featurestore_futures_premiumindexklines_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("open_time", "TIMESTAMP PRIMARY KEY"),
        ("close_time", "TIMESTAMP"),
        ("open", "DOUBLE PRECISION"),
        ("high", "DOUBLE PRECISION"),
        ("low", "DOUBLE PRECISION"),
        ("close", "DOUBLE PRECISION"),
        ("body_percentage", "DOUBLE PRECISION"),
        ("gap_percentage", "DOUBLE PRECISION"),
    ]

    for window in FEATURESTORE_ROLLING_WINDOWS:
        columns.extend(
            [
                (f"close_rolling_std_{window}", "DOUBLE PRECISION"),
                (f"close_rolling_zscore_{window}", "DOUBLE PRECISION"),
                (f"close_momentum_{window}", "DOUBLE PRECISION"),
            ]
        )

    for window in FEATURESTORE_LAG_WINDOWS:
        columns.append((f"close_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_futures_premiumindexklines_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_futures_premiumindexklines_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"futures_premiumindexklines_{timeframe}",
            time_column="open_time",
            columns=columns,
        )


def build_featurestore_spot_klines_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("open_time", "TIMESTAMP PRIMARY KEY"),
        ("close_time", "TIMESTAMP"),
        ("trade_quantity", "DOUBLE PRECISION"),
        ("trade_turnover", "DOUBLE PRECISION"),
        ("trade_count", "DOUBLE PRECISION"),
        ("buy_quantity", "DOUBLE PRECISION"),
        ("buy_turnover", "DOUBLE PRECISION"),
        ("imbalance_buy_quantity", "DOUBLE PRECISION"),
        ("imbalance_buy_turnover", "DOUBLE PRECISION"),
        ("log_return_trade_quantity", "DOUBLE PRECISION"),
        ("log_return_trade_turnover", "DOUBLE PRECISION"),
        ("log_return_trade_count", "DOUBLE PRECISION"),
        ("log_return_buy_quantity", "DOUBLE PRECISION"),
        ("log_return_buy_turnover", "DOUBLE PRECISION"),
    ]

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in (
            "imbalance_buy_quantity",
            "imbalance_buy_turnover",
        ):
            columns.extend(
                [
                    (f"{column}_rolling_std_{window}", "DOUBLE PRECISION"),
                    (f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"),
                    (f"{column}_momentum_{window}", "DOUBLE PRECISION"),
                ]
            )

        for column in (
            "log_return_trade_quantity",
            "log_return_trade_turnover",
            "log_return_trade_count",
            "log_return_buy_quantity",
            "log_return_buy_turnover",
        ):
            columns.extend(
                [
                    (f"{column}_rolling_std_{window}", "DOUBLE PRECISION"),
                    (f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"),
                    (f"{column}_momentum_{window}", "DOUBLE PRECISION"),
                ]
            )

    for window in FEATURESTORE_LAG_WINDOWS:
        for column in (
            "imbalance_buy_quantity",
            "imbalance_buy_turnover",
            "log_return_trade_quantity",
            "log_return_trade_turnover",
            "log_return_trade_count",
            "log_return_buy_quantity",
            "log_return_buy_turnover",
        ):
            columns.append((f"{column}_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_spot_klines_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_spot_klines_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"spot_klines_{timeframe}",
            time_column="open_time",
            columns=columns,
        )


def build_featurestore_sentiment_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("count", "BIGINT"),
        ("score", "DOUBLE PRECISION"),
        ("confidence", "DOUBLE PRECISION"),
        ("pct_negative", "DOUBLE PRECISION"),
        ("pct_positive", "DOUBLE PRECISION"),
        ("pct_neutral", "DOUBLE PRECISION"),
        ("log_return_count", "DOUBLE PRECISION"),
    ]

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in (
            "log_return_count",
            "score",
            "confidence",
            "pct_negative",
            "pct_positive",
            "pct_neutral",
        ):
            columns.extend(
                [
                    (f"{column}_rolling_std_{window}", "DOUBLE PRECISION"),
                    (f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"),
                    (f"{column}_momentum_{window}", "DOUBLE PRECISION"),
                ]
            )

    for window in FEATURESTORE_LAG_WINDOWS:
        for column in (
            "log_return_count",
            "score",
            "confidence",
            "pct_negative",
            "pct_positive",
            "pct_neutral",
        ):
            columns.append((f"{column}_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_sentiment_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_sentiment_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"sentiment_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def build_featurestore_futures_aggtrades_columns() -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = [
        ("create_time", "TIMESTAMP PRIMARY KEY"),
        ("trade_price_mean", "DOUBLE PRECISION"),
        ("trade_price_std", "DOUBLE PRECISION"),
        ("trade_price_min", "DOUBLE PRECISION"),
        ("trade_price_p25", "DOUBLE PRECISION"),
        ("trade_price_p50", "DOUBLE PRECISION"),
        ("trade_price_p75", "DOUBLE PRECISION"),
        ("trade_price_max", "DOUBLE PRECISION"),
        ("trade_price_skew", "DOUBLE PRECISION"),
        ("trade_price_kurtosis", "DOUBLE PRECISION"),
        ("buy_price_mean", "DOUBLE PRECISION"),
        ("buy_price_std", "DOUBLE PRECISION"),
        ("buy_price_min", "DOUBLE PRECISION"),
        ("buy_price_p25", "DOUBLE PRECISION"),
        ("buy_price_p50", "DOUBLE PRECISION"),
        ("buy_price_p75", "DOUBLE PRECISION"),
        ("buy_price_max", "DOUBLE PRECISION"),
        ("buy_price_skew", "DOUBLE PRECISION"),
        ("buy_price_kurtosis", "DOUBLE PRECISION"),
        ("trade_quantity_mean", "DOUBLE PRECISION"),
        ("trade_quantity_std", "DOUBLE PRECISION"),
        ("trade_quantity_min", "DOUBLE PRECISION"),
        ("trade_quantity_p25", "DOUBLE PRECISION"),
        ("trade_quantity_p50", "DOUBLE PRECISION"),
        ("trade_quantity_p75", "DOUBLE PRECISION"),
        ("trade_quantity_max", "DOUBLE PRECISION"),
        ("trade_quantity_skew", "DOUBLE PRECISION"),
        ("trade_quantity_kurtosis", "DOUBLE PRECISION"),
        ("buy_quantity_mean", "DOUBLE PRECISION"),
        ("buy_quantity_std", "DOUBLE PRECISION"),
        ("buy_quantity_min", "DOUBLE PRECISION"),
        ("buy_quantity_p25", "DOUBLE PRECISION"),
        ("buy_quantity_p50", "DOUBLE PRECISION"),
        ("buy_quantity_p75", "DOUBLE PRECISION"),
        ("buy_quantity_max", "DOUBLE PRECISION"),
        ("buy_quantity_skew", "DOUBLE PRECISION"),
        ("buy_quantity_kurtosis", "DOUBLE PRECISION"),
        ("trade_vwap", "DOUBLE PRECISION"),
        ("buy_vwap", "DOUBLE PRECISION"),
        ("trade_rate_mean", "DOUBLE PRECISION"),
        ("trade_rate_std", "DOUBLE PRECISION"),
        ("trade_count", "DOUBLE PRECISION"),
        ("buy_rate_mean", "DOUBLE PRECISION"),
        ("buy_rate_std", "DOUBLE PRECISION"),
        ("buy_count", "DOUBLE PRECISION"),
        ("tickup_count", "DOUBLE PRECISION"),
        ("trade_turnover_std", "DOUBLE PRECISION"),
        ("buy_turnover_std", "DOUBLE PRECISION"),
        ("trade_corr_price_quantity", "DOUBLE PRECISION"),
        ("trade_corr_price_time", "DOUBLE PRECISION"),
        ("trade_corr_quantity_time", "DOUBLE PRECISION"),
        ("buy_corr_price_quantity", "DOUBLE PRECISION"),
        ("buy_corr_price_time", "DOUBLE PRECISION"),
        ("buy_corr_quantity_time", "DOUBLE PRECISION"),
        ("imbalance_buy_count", "DOUBLE PRECISION"),
        ("imbalance_tickup_count", "DOUBLE PRECISION"),
    ]

    log_return_columns = (
        "trade_price_mean",
        "trade_price_min",
        "trade_price_p25",
        "trade_price_p50",
        "trade_price_p75",
        "trade_price_max",
        "buy_price_mean",
        "buy_price_min",
        "buy_price_p25",
        "buy_price_p50",
        "buy_price_p75",
        "buy_price_max",
        "trade_quantity_mean",
        "trade_quantity_min",
        "trade_quantity_p25",
        "trade_quantity_p50",
        "trade_quantity_p75",
        "trade_quantity_max",
        "buy_quantity_mean",
        "buy_quantity_min",
        "buy_quantity_p25",
        "buy_quantity_p50",
        "buy_quantity_p75",
        "buy_quantity_max",
        "trade_rate_mean",
        "buy_rate_mean",
        "buy_count",
        "tickup_count",
        "trade_vwap",
        "buy_vwap",
    )

    for column in log_return_columns:
        columns.append((f"log_return_{column}", "DOUBLE PRECISION"))

    rolling_mean_columns = (
        "log_return_trade_price_mean",
        "log_return_buy_price_mean",
        "log_return_trade_vwap",
        "log_return_buy_vwap",
    )
    rolling_std_zscore_columns = (
        "log_return_trade_price_mean",
        "log_return_buy_price_mean",
        "log_return_trade_rate_mean",
        "log_return_buy_rate_mean",
        "log_return_buy_count",
        "log_return_tickup_count",
        "imbalance_buy_count",
        "imbalance_tickup_count",
        "log_return_trade_vwap",
        "log_return_buy_vwap",
    )

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in rolling_std_zscore_columns:
            columns.append((f"{column}_rolling_std_{window}", "DOUBLE PRECISION"))
            if column in rolling_mean_columns:
                columns.append((f"{column}_rolling_mean_{window}", "DOUBLE PRECISION"))
            columns.append((f"{column}_rolling_zscore_{window}", "DOUBLE PRECISION"))

    temporal_columns = (
        "log_return_trade_price_mean",
        "log_return_buy_price_mean",
        "log_return_trade_rate_mean",
        "log_return_buy_rate_mean",
        "log_return_buy_count",
        "log_return_tickup_count",
        "log_return_trade_vwap",
        "log_return_buy_vwap",
        "imbalance_buy_count",
        "imbalance_tickup_count",
    )

    for window in FEATURESTORE_ROLLING_WINDOWS:
        for column in temporal_columns:
            columns.append((f"{column}_momentum_{window}", "DOUBLE PRECISION"))

    for window in FEATURESTORE_LAG_WINDOWS:
        for column in temporal_columns:
            columns.append((f"{column}_lag_{window}", "DOUBLE PRECISION"))

    return columns


def create_featurestore_futures_aggtrades_table(client: TimescaleDBClient) -> None:
    columns = build_featurestore_futures_aggtrades_columns()
    for timeframe in FEATURESTORE_TIMEFRAMES:
        ensure_hypertable(
            client,
            schema_name="featurestore",
            table_name=f"futures_aggtrades_{timeframe}",
            time_column="create_time",
            columns=columns,
        )


def main() -> None:
    print("TIMESCALEDB INITIALIZATION")
    client = None

    try:
        client = wait_for_timescaledb()
        initialize_schemas(client)
        initialize_dashboard_tables(client)
        initialize_featurestore_tables(client)
        print("TIMESCALEDB INITIALIZATION COMPLETED!")
    except Exception as exc:
        print(f"TimescaleDB initialization failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
