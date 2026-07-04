"""
Airflow DAG for batch backfill of dashboard prediction tables.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_dashboard_predict_backfill(**kwargs):
    """Backfill dashboard predict tables from featurestore tables."""
    from src.batch.timescaledb.dashboard.predict import main

    main()


default_args = {
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(seconds=5),
}


with DAG(
    dag_id="batch_predict",
    default_args=default_args,
    description="Backfill dashboard predict tables in TimescaleDB from featurestore tables.",
    schedule_interval=None,
    catchup=False,
) as dag:
    backfill_dashboard_predict = PythonOperator(
        task_id="backfill_dashboard_predict",
        python_callable=run_dashboard_predict_backfill,
        provide_context=True,
    )
