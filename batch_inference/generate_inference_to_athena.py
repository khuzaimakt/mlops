#!/usr/bin/env python
"""
Generate synthetic inference data (with ground-truth churn label) and publish it to an Athena table.

- Database: ml_metrics
- Table:    mlops_poc_inference_data

Flow:
1. Generate inference dataset (no target column)
2. Write to local CSV (no header)
3. Upload CSV to S3 (bucket/prefix from batch_inference_poc/pipeline_config.yaml)
4. Run Athena DDL to create/replace external table pointing at that S3 location
"""

import os
import sys
import argparse
import logging
import uuid
import time

import boto3
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_inference_dataset(n_samples: int = 10000, start_customer_id: int = 1000) -> pd.DataFrame:
    """Generate synthetic inference dataset with ground-truth churn label."""
    import numpy as np

    logger.info(f"Generating inference dataset with {n_samples} samples (customer_id from {start_customer_id})...")

    np.random.seed(42)

    customer_ids = [f"CUST_{i:06d}" for i in range(start_customer_id, start_customer_id + n_samples)]

    data = {
        "customer_id": customer_ids,
        "age": np.random.randint(18, 80, n_samples),
        "tenure_months": np.random.randint(0, 72, n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples).round(2),
        "total_charges": np.random.uniform(100, 8640, n_samples).round(2),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "payment_method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
        ),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "phone_service": np.random.choice(["Yes", "No"], n_samples),
        "multiple_lines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "online_security": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "device_protection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
    }

    df = pd.DataFrame(data)

    # Generate ground-truth churn similar to training logic
    churn_prob = (
        (df["monthly_charges"] / 120) * 0.3
        + (1 - df["tenure_months"] / 72) * 0.4
        + (df["contract_type"] == "Month-to-month").astype(int) * 0.3
    )
    df["churn"] = (np.random.rand(n_samples) < churn_prob).astype(int)

    logger.info("Dataset generated: %d samples, churn rate=%.1f%%", len(df), df["churn"].mean() * 100)
    return df


def upload_to_s3(df: pd.DataFrame, bucket: str, prefix: str, region: str) -> str:
    """Write DataFrame to local CSV (no header) and upload to S3. Return S3 prefix (folder)."""
    session = boto3.Session(region_name=region)
    s3 = session.client("s3")

    run_id = uuid.uuid4().hex[:8]
    s3_prefix = f"{prefix}/athena/mlops_poc_inference_data/run={run_id}"
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    local_file = f"/tmp/mlops_poc_inference_data_{run_id}.csv"

    logger.info(f"Writing local CSV (no header): {local_file}")
    # Write without header so Athena doesn't treat the header row as data
    df.to_csv(local_file, index=False, header=False)

    key = f"{s3_prefix}/data.csv"
    logger.info(f"Uploading to s3://{bucket}/{key}")
    s3.upload_file(local_file, bucket, key)

    logger.info(f"Data uploaded to {s3_uri}")
    return s3_uri


def build_create_table_sql(database: str, table: str, s3_location: str) -> str:
    """
    Create/replace external table DDL for the inference dataset.

    Schema includes ground-truth churn for evaluation.
    """
    return f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table} (
  customer_id       string,
  age               int,
  tenure_months     int,
  monthly_charges   double,
  total_charges     double,
  contract_type     string,
  payment_method    string,
  internet_service  string,
  phone_service     string,
  multiple_lines    string,
  online_security   string,
  tech_support      string,
  device_protection string,
  churn             int
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
WITH SERDEPROPERTIES (
  'serialization.format' = ',',
  'field.delim' = ','
)
LOCATION '{s3_location}'
TBLPROPERTIES ('has_encrypted_data'='false', 'skip.header.line.count'='0');
""".strip()


def run_athena_ddl(ddl: str, database: str, region: str, output_location: str) -> None:
    """Execute a DDL query in Athena and wait for completion."""
    session = boto3.Session(region_name=region)
    athena = session.client("athena")

    logger.info("Submitting Athena DDL...")
    resp = athena.start_query_execution(
        QueryString=ddl,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    qid = resp["QueryExecutionId"]
    logger.info(f"Athena QueryExecutionId: {qid}")

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            logger.info(f"Athena query finished with status: {status}")
            if status != "SUCCEEDED":
                reason = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"].get(
                    "StateChangeReason", "Unknown"
                )
                raise RuntimeError(f"Athena DDL failed: {reason}")
            break
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "configs",
            "pipeline_config.yaml",
        ),
        help="Path to batch_inference_poc pipeline_config.yaml (for bucket/prefix/region)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="ml_metrics",
        help="Athena database name",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="mlops_poc_inference_data",
        help="Athena table name",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of inference samples to generate",
    )
    parser.add_argument(
        "--start-customer-id",
        type=int,
        default=1000,
        help="Starting customer_id numeric component (inclusive)",
    )
    parser.add_argument(
        "--athena-output",
        type=str,
        default="s3://mlbucket-sagemaker/athena-query-results/",
        help="S3 location for Athena query results (must exist and be writable)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["region"]
    bucket = config["bucket"]
    prefix = config["prefix"]

    df = generate_inference_dataset(args.n_samples, args.start_customer_id)
    s3_location = upload_to_s3(df, bucket=bucket, prefix=prefix, region=region)

    ddl = build_create_table_sql(
        database=args.database,
        table=args.table,
        s3_location=s3_location,
    )
    logger.info("Athena DDL:\n%s", ddl)

    run_athena_ddl(
        ddl=ddl,
        database=args.database,
        region=region,
        output_location=args.athena_output,
    )

    logger.info(
        "Athena table %s.%s now points to data at %s",
        args.database,
        args.table,
        s3_location,
    )


if __name__ == "__main__":
    main()

