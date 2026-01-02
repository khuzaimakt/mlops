#!/usr/bin/env python
"""Data sourcing: load training data from SageMaker Feature Store.

This script reads training data from SageMaker Feature Store offline store
(via Athena) and writes it as a CSV to a local output path. The Feature Store
offline store is queried using Athena, filtering by customer_id range.
"""
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
    except Exception as notify_err:
        logger.error("Failed to send SNS notification: %s", notify_err)


def describe_feature_group(region: str, feature_group_name: str) -> dict:
    """Describe a Feature Group to get offline store Glue table information."""
    sm = boto3.client("sagemaker", region_name=region)
    desc = sm.describe_feature_group(FeatureGroupName=feature_group_name)
    return desc


def run_athena_query(query: str, database: str, region: str, output_location: str) -> pd.DataFrame:
    athena = boto3.client("athena", region_name=region)
    start = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    qid = start["QueryExecutionId"]
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

    # Fetch results
    paginator = athena.get_paginator("get_query_results")
    rows = []
    columns = []
    first_page = True
    for page in paginator.paginate(QueryExecutionId=qid):
        result_set = page["ResultSet"]
        if first_page:
            columns = [col["Name"] for col in result_set["ResultSetMetadata"]["ColumnInfo"]]
            first_page = False
            data_rows = result_set.get("Rows", [])[1:]  # skip header
        else:
            data_rows = result_set.get("Rows", [])

        for row in data_rows:
            values = [field.get("VarCharValue", None) for field in row["Data"]]
            rows.append(values)

    df = pd.DataFrame(rows, columns=columns)
    logger.info("Athena query returned %d rows", len(df))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-group-name",
        type=str,
        default="mlops_poc_churn_classification_feature",
        help="SageMaker Feature Store feature group name",
    )
    parser.add_argument(
        "--customer-id-start",
        type=int,
        default=1,
        help="Starting customer_id (inclusive) for training data",
    )
    parser.add_argument(
        "--customer-id-end",
        type=int,
        default=4000,
        help="Ending customer_id (inclusive) for training data",
    )
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output/raw")
    parser.add_argument("--athena-output", type=str, required=True, help="S3 path for Athena query results")
    parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
    parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))
    args = parser.parse_args()

    global SNS_TOPIC_ARN, NOTIFICATION_EMAIL
    if args.sns_topic_arn:
        SNS_TOPIC_ARN = args.sns_topic_arn
    if args.notification_email:
        NOTIFICATION_EMAIL = args.notification_email

    try:
        # 1) Describe feature group to get offline store Glue table
        logger.info("Describing feature group: %s", args.feature_group_name)
        desc = describe_feature_group(args.region, args.feature_group_name)
        offline_cfg = desc.get("OfflineStoreConfig", {})
        data_catalog = offline_cfg.get("DataCatalogConfig", {})

        database = data_catalog.get("Database")
        table_name = data_catalog.get("TableName")

        if not database or not table_name:
            raise RuntimeError(
                "Offline DataCatalogConfig is missing Database/TableName. "
                "Ensure offline store is enabled and Glue integration is not disabled."
            )

        logger.info("Using Glue/Athena table: %s.%s", database, table_name)

        # 2) Build query with customer_id filter
        start_id_str = f"CUST_{args.customer_id_start:06d}"
        end_id_str = f"CUST_{args.customer_id_end:06d}"
        
        query = f'''
        SELECT * FROM "{database}"."{table_name}"
        WHERE customer_id >= '{start_id_str}' AND customer_id <= '{end_id_str}'
        '''
        
        logger.info("Running Athena query with customer_id filter: %s to %s", start_id_str, end_id_str)

        df = run_athena_query(
            query=query,
            database=database,
            region=args.region,
            output_location=args.athena_output,
        )

        if df.empty:
            raise RuntimeError(f"Feature Store query returned no rows for customer_id range {start_id_str} to {end_id_str}")

        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, "data.csv")
        df.to_csv(output_file, index=False)
        logger.info("Wrote %d rows to %s", len(df), output_file)
    except Exception as exc:
        logger.exception("Data Sourcing step failed")
        send_failure_notification("Data Sourcing - AutoML Training Pipeline", str(exc))
        raise


if __name__ == "__main__":
    main()
