
# Data Schema Documentation

## Overview
- **Bronze Layer**: Raw data from exchange APIs
- **Silver Layer**: Aggregated data with time-based intervals
- **Available Intervals for Silver**: `5m`, `15m`, `1h`, `4h`, `1d` 
- **Step Size**: All intervals use 5m step size for aggregation windows
- **Aggregation Method**: Sliding window with previous day concat for boundary completeness

---

## Bronze Layer Schemas
Bronze layer contains raw ingested data from external sources.

### Fundingrate Data
**Path**: `datalake/1_bronze/fundingRate/{symbol}/YYYY-MM.parquet`
**Intervals**: None (monthly files)
| Column | Data Type |
|--------|-----------|
| `instrument_name` | `String` |
| `funding_rate` | `Float64` |
| `funding_time` | `Int64` |

### Indexpriceklines Data
**Path**: `datalake/1_bronze/indexPriceKlines/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` (original from exchange)
| Column | Data Type |
|--------|-----------|
| `open_time` | `Int64` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |

### Markpriceklines Data
**Path**: `datalake/1_bronze/markPriceKlines/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `1m`, `5m`, `15m`, `1h`, `4h`, `1d` (original from exchange)
| Column | Data Type |
|--------|-----------|
| `open_time` | `Int64` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |

### Orderbook Data
**Path**: `datalake/1_bronze/orderBook/{symbol}/YYYY-MM-DD.parquet`
**Intervals**: None (raw tick data)
| Column | Data Type |
|--------|-----------|
| `instId` | `String` |
| `action` | `String` |
| `ts` | `Int64` |
| `asks` | `String` |
| `bids` | `String` |

### Trades Data
**Path**: `datalake/1_bronze/trades/{symbol}/YYYY-MM-DD.parquet`
**Intervals**: None (raw tick data)
| Column | Data Type |
|--------|-----------|
| `instrument_name` | `String` |
| `trade_id` | `Int64` |
| `side` | `String` |
| `price` | `Float64` |
| `size` | `Float64` |
| `created_time` | `Int64` |


---

## Silver Layer Schemas

Silver layer contains cleaned and aggregated data with business logic applied.

### Orderbook Data (Aggregated)
**Path**: `datalake/2_silver/orderBook/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `5m`, `15m`, `1h`, `4h`, `1d` (step size: 5m for all)
**Source**: Aggregated from bronze orderbook snapshots (action='snapshot' only)
| Column | Data Type |
|--------|-----------|
| `timestamp_dt` | `Datetime(time_unit='us', time_zone=None)` |
| `wmp_mean` | `Float64` |
| `wmp_std` | `Float64` |
| `wmp_min` | `Float64` |
| `wmp_0.25` | `Float64` |
| `wmp_0.50` | `Float64` |
| `wmp_0.75` | `Float64` |
| `wmp_max` | `Float64` |
| `wmp_first` | `Float64` |
| `wmp_last` | `Float64` |
| `wmp_skew` | `Float64` |
| `wmp_kurtosis` | `Float64` |
| `spread_mean` | `Float64` |
| `spread_std` | `Float64` |
| `spread_min` | `Float64` |
| `spread_0.25` | `Float64` |
| `spread_0.50` | `Float64` |
| `spread_0.75` | `Float64` |
| `spread_max` | `Float64` |
| `spread_first` | `Float64` |
| `spread_last` | `Float64` |
| `spread_skew` | `Float64` |
| `spread_kurtosis` | `Float64` |
| `imbal_mean` | `Float64` |
| `imbal_std` | `Float64` |
| `imbal_min` | `Float64` |
| `imbal_0.25` | `Float64` |
| `imbal_0.50` | `Float64` |
| `imbal_0.75` | `Float64` |
| `imbal_max` | `Float64` |
| `imbal_first` | `Float64` |
| `imbal_last` | `Float64` |
| `imbal_skew` | `Float64` |
| `imbal_kurtosis` | `Float64` |
| `depth_mean` | `Float64` |
| `depth_std` | `Float64` |
| `depth_min` | `Float64` |
| `depth_max` | `Float64` |
| `depth_first` | `Float64` |
| `depth_last` | `Float64` |
| `conc_mean` | `Float64` |
| `conc_std` | `Float64` |
| `conc_min` | `Float64` |
| `conc_max` | `Float64` |
| `impact_ask_mean` | `Float64` |
| `impact_ask_std` | `Float64` |
| `impact_ask_min` | `Float64` |
| `impact_ask_max` | `Float64` |
| `impact_bid_mean` | `Float64` |
| `impact_bid_std` | `Float64` |
| `impact_bid_min` | `Float64` |
| `impact_bid_max` | `Float64` |
| `pressure_sum` | `Float64` |
| `pressure_mean` | `Float64` |
| `pressure_std` | `Float64` |
| `pressure_min` | `Float64` |
| `pressure_0.25` | `Float64` |
| `pressure_0.50` | `Float64` |
| `pressure_0.75` | `Float64` |
| `pressure_max` | `Float64` |
| `pressure_first` | `Float64` |
| `pressure_last` | `Float64` |
| `pressure_skew` | `Float64` |
| `pressure_kurtosis` | `Float64` |
| `snapshot_count` | `Int64` |
| `rate_mean_ms` | `Float64` |
| `rate_std_ms` | `Float64` |
| `rate_min_ms` | `Int64` |
| `rate_max_ms` | `Int64` |
| `corr_wmp_imbal` | `Float64` |
| `corr_wmp_depth` | `Float64` |
| `corr_wmp_pressure` | `Float64` |
| `corr_wmp_time` | `Float64` |
| `corr_spread_imbal` | `Float64` |
| `corr_spread_impact` | `Float64` |
| `corr_conc_time` | `Float64` |
| `corr_pressure_time` | `Float64` |
| `last_update_time_ms` | `Int64` |



### Trades Data (Aggregated)
**Path**: `datalake/2_silver/trades/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `5m`, `15m`, `1h`, `4h`, `1d` (step size: 5m for all)
**Source**: Aggregated from bronze trades tick data with sliding window algorithm
| Column | Data Type |
|--------|-----------|
| `timestamp_dt` | `Datetime(time_unit='us', time_zone=None)` |
| `volume_buy` | `Float64` |
| `volume_sell` | `Float64` |
| `turnover_buy` | `Float64` |
| `turnover_sell` | `Float64` |
| `count_buy` | `Int64` |
| `count_sell` | `Int64` |
| `count_tick_up` | `Int64` |
| `count_tick_down` | `Int64` |
| `price_first_trade` | `Float64` |
| `price_last_trade` | `Float64` |
| `price_mean_trade` | `Float64` |
| `price_std_trade` | `Float64` |
| `price_min_trade` | `Float64` |
| `price_max_trade` | `Float64` |
| `price_0.25_trade` | `Float64` |
| `price_0.50_trade` | `Float64` |
| `price_0.75_trade` | `Float64` |
| `price_skew_trade` | `Float64` |
| `price_kurtosis_trade` | `Float64` |
| `nunique_price_trade` | `Int64` |
| `price_first_buy` | `Float64` |
| `price_last_buy` | `Float64` |
| `price_mean_buy` | `Float64` |
| `price_std_buy` | `Float64` |
| `price_min_buy` | `Float64` |
| `price_max_buy` | `Float64` |
| `price_0.25_buy` | `Float64` |
| `price_0.50_buy` | `Float64` |
| `price_0.75_buy` | `Float64` |
| `price_skew_buy` | `Float64` |
| `price_kurtosis_buy` | `Float64` |
| `nunique_price_buy` | `Int64` |
| `price_first_sell` | `Float64` |
| `price_last_sell` | `Float64` |
| `price_mean_sell` | `Float64` |
| `price_std_sell` | `Float64` |
| `price_min_sell` | `Float64` |
| `price_max_sell` | `Float64` |
| `price_0.25_sell` | `Float64` |
| `price_0.50_sell` | `Float64` |
| `price_0.75_sell` | `Float64` |
| `price_skew_sell` | `Float64` |
| `price_kurtosis_sell` | `Float64` |
| `nunique_price_sell` | `Int64` |
| `size_mean_trade` | `Float64` |
| `size_std_trade` | `Float64` |
| `size_min_trade` | `Float64` |
| `size_max_trade` | `Float64` |
| `size_0.25_trade` | `Float64` |
| `size_0.50_trade` | `Float64` |
| `size_0.75_trade` | `Float64` |
| `size_skew_trade` | `Float64` |
| `size_kurtosis_trade` | `Float64` |
| `nunique_size_trade` | `Int64` |
| `size_mean_buy` | `Float64` |
| `size_std_buy` | `Float64` |
| `size_min_buy` | `Float64` |
| `size_max_buy` | `Float64` |
| `size_0.25_buy` | `Float64` |
| `size_0.50_buy` | `Float64` |
| `size_0.75_buy` | `Float64` |
| `size_skew_buy` | `Float64` |
| `size_kurtosis_buy` | `Float64` |
| `nunique_size_buy` | `Int64` |
| `size_mean_sell` | `Float64` |
| `size_std_sell` | `Float64` |
| `size_min_sell` | `Float64` |
| `size_max_sell` | `Float64` |
| `size_0.25_sell` | `Float64` |
| `size_0.50_sell` | `Float64` |
| `size_0.75_sell` | `Float64` |
| `size_skew_sell` | `Float64` |
| `size_kurtosis_sell` | `Float64` |
| `nunique_size_sell` | `Int64` |
| `rate_mean_ms_trade` | `Float64` |
| `rate_std_ms_trade` | `Float64` |
| `rate_max_ms_trade` | `Int64` |
| `rate_min_ms_trade` | `Int64` |
| `rate_mean_ms_buy` | `Float64` |
| `rate_std_ms_buy` | `Float64` |
| `rate_max_ms_buy` | `Int64` |
| `rate_min_ms_buy` | `Int64` |
| `rate_mean_ms_sell` | `Float64` |
| `rate_std_ms_sell` | `Float64` |
| `rate_max_ms_sell` | `Int64` |
| `rate_min_ms_sell` | `Int64` |
| `corr_price_size_trade` | `Float64` |
| `corr_price_time_trade` | `Float64` |
| `corr_size_time_trade` | `Float64` |
| `corr_price_size_buy` | `Float64` |
| `corr_price_time_buy` | `Float64` |
| `corr_size_time_buy` | `Float64` |
| `corr_price_size_sell` | `Float64` |
| `corr_price_time_sell` | `Float64` |
| `corr_size_time_sell` | `Float64` |
| `last_trade_time_ms` | `Int64` |
| `trade_count` | `Int64` |


### IndexPriceKlines Data (Aggregated)
**Path**: `datalake/2_silver/indexPriceKlines/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `5m`, `15m`, `1h`, `4h`, `1d` (step size: 5m for all)
**Source**: Aggregated from bronze indexPriceKlines 1m data with sliding window algorithm
| Column | Data Type |
|--------|-----------|
| `timestamp_dt` | `Datetime(time_unit='us', time_zone=None)` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |
| `mean` | `Float64` |
| `std` | `Float64` |


### MarkPriceKlines Data (Aggregated)
**Path**: `datalake/2_silver/markPriceKlines/{symbol}/{interval}/YYYY-MM-DD.parquet`
**Intervals**: `5m`, `15m`, `1h`, `4h`, `1d` (step size: 5m for all)
**Source**: Aggregated from bronze markPriceKlines 1m data with sliding window algorithm
| Column | Data Type |
|--------|-----------|
| `timestamp_dt` | `Datetime(time_unit='us', time_zone=None)` |
| `open` | `Float64` |
| `high` | `Float64` |
| `low` | `Float64` |
| `close` | `Float64` |
| `mean` | `Float64` |
| `std` | `Float64` |