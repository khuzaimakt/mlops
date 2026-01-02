#!/usr/bin/env python
"""Publish batch predictions to Athena table."""
import os
import sys
import argparse
import time
import logging
import boto3
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")


def send_failure_notification(job_name: str, error_message: str) -> None:
    """Publish a failure notification to SNS."""
    try:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        sns = boto3.client("sns", region_name=region)
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[SageMaker Pipeline Failed] {job_name}",
            Message=(
                f"Step: {job_name}\n"
                f"Error: {error_message}\n"
                "See CloudWatch Logs for full traceback."
            ),
        )
        logger.info("Sent failure notification for %s", job_name)
    except Exception as notify_err:  # pragma: no cover
        logger.error("Failed to send SNS notification: %s", notify_err)


def infer_athena_type(dtype) -> str:
    """Map pandas dtype to Athena type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "bigint"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    return "string"


def build_create_table_sql(database: str, table: str, s3_path: str, df: pd.DataFrame) -> str:
    cols = []
    for col in df.columns:
        cols.append(f"`{col}` {infer_athena_type(df[col].dtype)}")
    cols_sql = ",\n  ".join(cols)
    # Use OpenCSVSerde for safer CSV handling
    return f"""
CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table} (
  {cols_sql}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '\"',
  'escapeChar' = '\\\\'
)
LOCATION '{s3_path}'
TBLPROPERTIES ('skip.header.line.count'='1');
""".strip()


def run_athena(query: str, database: str, output_location: str, region: str) -> None:
    athena = boto3.client("athena", region_name=region)
    q = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    qid = q["QueryExecutionId"]
    logger.info("Started Athena query %s", qid)
    while True:
        resp = athena.get_query_execution(QueryExecutionId=qid)
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ["SUCCEEDED", "FAILED", "CANCELLED"]:
            break
        time.sleep(2)
    if state != "SUCCEEDED":
        reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
        raise RuntimeError(f"Athena query failed: {reason}")
    logger.info("Athena query %s succeeded", qid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--input-s3", type=str, required=True, help="S3 path with predictions CSV (output of batch inference)")
    parser.add_argument("--athena-output", type=str, required=True, help="S3 path for Athena query results")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
    parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))
    args = parser.parse_args()

    global SNS_TOPIC_ARN, NOTIFICATION_EMAIL
    if args.sns_topic_arn:
        SNS_TOPIC_ARN = args.sns_topic_arn
    if args.notification_email:
        NOTIFICATION_EMAIL = args.notification_email

    try:
        # Load a sample of predictions to infer schema
        # We expect a single CSV file in input-s3 path
        s3 = boto3.client("s3", region_name=args.region)
        if not args.input_s3.startswith("s3://"):
            raise RuntimeError("input-s3 must be an s3:// URI")
        bucket, prefix = args.input_s3.replace("s3://", "").split("/", 1)
        objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
        csv_keys = [o["Key"] for o in objs if o["Key"].endswith(".csv")]
        if not csv_keys:
            raise RuntimeError(f"No CSV predictions found under {args.input_s3}")
        csv_key = csv_keys[0]
        local_csv = "/tmp/predictions.csv"
        s3.download_file(bucket, csv_key, local_csv)
        df = pd.read_csv(local_csv)
        logger.info("Loaded predictions sample with columns: %s", list(df.columns))

        create_sql = build_create_table_sql(args.database, args.table, args.input_s3, df)
        logger.info("Creating/Updating Athena table %s.%s", args.database, args.table)
        run_athena(create_sql, args.database, args.athena_output, args.region)

    except Exception as exc:
        logger.exception("Publish to Athena step failed")
        send_failure_notification("Publish to Athena - Churn Classification Model Batch Pipeline", str(exc))
        raise


if __name__ == "__main__":
    main()

