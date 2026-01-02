#!/usr/bin/env python
"""
Ingest churn prediction features from Athena into SageMaker Feature Store.

Flow:
1. Read raw (messy) churn data from Athena table `churn_prediction_data`
2. Apply light preprocessing/cleaning to align with training_poc schema
3. Upsert records into Feature Store feature group `mlops_poc_churn_classification_feature`

This POC is designed so the resulting feature group can later replace the
Athena tables used in:
  - mlops_poc/training_poc
  - mlops_poc/batch_inference_poc
  - mlops_poc/automl_training_poc
"""

import subprocess
import sys
import os
import logging

# Set up logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SNS configuration (will be overridden via CLI args when run in pipeline)
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")


def send_failure_notification_early(job_name: str, error_message: str) -> None:
    """Publish a failure notification to SNS - early version that doesn't require imports."""
    try:
        import boto3
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        sns = boto3.client("sns", region_name=region)
        if SNS_TOPIC_ARN:
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
        else:
            logger.warning("SNS_TOPIC_ARN not set; skipping notification")
    except Exception as notify_err:
        logger.error("Failed to send SNS notification: %s", notify_err)


# PySparkProcessor has sagemaker pre-installed, but we may need pyyaml
try:
    logger.info("Installing pyyaml (sagemaker is pre-installed in PySparkProcessor)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "pyyaml",
        "--quiet"
    ], check=True)
    logger.info("Dependencies installed successfully")
except Exception as e:
    error_msg = f"Failed to install pyyaml: {str(e)}"
    logger.error(error_msg)
    send_failure_notification_early("Feature Store Ingestion - Churn Classification", error_msg)
    raise

import argparse
import time
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import yaml
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session


def send_failure_notification(job_name: str, error_message: str) -> None:
    """Publish a failure notification to SNS."""
    try:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        sns = boto3.client("sns", region_name=region)
        if SNS_TOPIC_ARN:
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
        else:
            logger.warning("SNS_TOPIC_ARN not set; skipping notification")
    except Exception as notify_err:  # best-effort notification
        logger.error("Failed to send SNS notification: %s", notify_err)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_athena_query(query: str, database: str, region: str, output_location: str) -> pd.DataFrame:
    """Execute an Athena query and return results as a pandas DataFrame."""
    athena = boto3.client("athena", region_name=region)

    logger.info("Starting Athena query...")
    resp = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    qid = resp["QueryExecutionId"]
    logger.info("Athena QueryExecutionId: %s", qid)

    # Wait for query to complete
    while True:
        status_resp = athena.get_query_execution(QueryExecutionId=qid)
        state = status_resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            logger.info("Athena query finished with state: %s", state)
            if state != "SUCCEEDED":
                reason = status_resp["QueryExecution"]["Status"].get("StateChangeReason", "Unknown")
                raise RuntimeError(f"Athena query failed: {reason}")
            break
        time.sleep(2)

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
    logger.info("Fetched %d rows from Athena", len(df))
    return df


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light preprocessing:
    - Strip whitespace from string columns
    - Normalize casing for categorical features
    - Coerce numeric columns to proper types
    - Ensure churn is int (0/1)
    """
    logger.info("Cleaning raw dataframe before loading to Feature Store...")
    df_clean = df.copy()

    # Strip whitespace on all object columns
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()

    # Coerce numeric columns
    numeric_cols = ["age", "tenure_months", "monthly_charges", "total_charges"]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Coerce churn to int
    if "churn" in df_clean.columns:
        df_clean["churn"] = pd.to_numeric(df_clean["churn"], errors="coerce").fillna(0).astype(int)

    # Normalize common categorical columns to title-case
    cat_cols = [
        "contract_type",
        "payment_method",
        "internet_service",
        "phone_service",
        "multiple_lines",
        "online_security",
        "tech_support",
        "device_protection",
    ]
    for col in cat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace("", np.nan)
            df_clean[col] = df_clean[col].dropna().astype(str).str.strip()
            df_clean[col] = df_clean[col].str.title()

    logger.info("Cleaned dataframe shape: %s", df_clean.shape)
    return df_clean


def ensure_feature_group(
    sagemaker_session: Session,
    feature_group_name: str,
    df: pd.DataFrame,
    record_identifier_name: str,
    event_time_feature_name: str,
    offline_store_s3_uri: str,
    online_store_enabled: bool,
    role_arn: str,
) -> FeatureGroup:
    """Create the feature group if it does not exist."""
    fg = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    sm_client = sagemaker_session.boto_session.client("sagemaker")

    try:
        desc = fg.describe()
        logger.info("Feature group '%s' already exists (status: %s)", feature_group_name, desc["FeatureGroupStatus"])
        return fg
    except sm_client.exceptions.ResourceNotFound:
        logger.info("Feature group '%s' does not exist. Creating...", feature_group_name)

    # Build feature definitions from dataframe dtypes
    from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum

    feature_definitions = []
    for col, dtype in df.dtypes.items():
        if col == event_time_feature_name:
            # Event time is stored as string/ISO timestamp in the feature group
            feature_type = FeatureTypeEnum.STRING
        elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating):
            feature_type = FeatureTypeEnum.INTEGRAL if np.issubdtype(dtype, np.integer) else FeatureTypeEnum.FRACTIONAL
        else:
            feature_type = FeatureTypeEnum.STRING
        feature_definitions.append(FeatureDefinition(feature_name=col, feature_type=feature_type))

    # Ensure the event time feature is present in feature definitions even if not in the raw dataframe
    if event_time_feature_name not in [fd.feature_name for fd in feature_definitions]:
        feature_definitions.append(
            FeatureDefinition(feature_name=event_time_feature_name, feature_type=FeatureTypeEnum.STRING)
        )

    # Older SageMaker SDK versions do not accept record_identifier_name/event_time_feature_name
    # in the constructor; they are passed only to create().
    fg = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sagemaker_session,
        feature_definitions=feature_definitions,
    )

    fg.create(
        s3_uri=offline_store_s3_uri,
        record_identifier_name=record_identifier_name,
        event_time_feature_name=event_time_feature_name,
        role_arn=role_arn,
        enable_online_store=online_store_enabled,
    )

    logger.info("Waiting for feature group '%s' to be in 'Created' status...", feature_group_name)
    # Older SDKs may not have fg.wait_for_create(), so poll describe_feature_group directly.
    while True:
        desc = sm_client.describe_feature_group(FeatureGroupName=feature_group_name)
        status = desc.get("FeatureGroupStatus")
        logger.info("Current feature group status: %s", status)
        if status in ("Created", "CreateFailed", "Deleted", "DeleteFailed"):
            break
        time.sleep(5)

    if status != "Created":
        raise RuntimeError(f"Feature group creation ended in status {status}")

    logger.info("Feature group '%s' created.", feature_group_name)
    return fg


def ingest_dataframe_to_feature_group(
    fg: FeatureGroup,
    df: pd.DataFrame,
    event_time_feature_name: str,
    batch_size: int = 500,
):
    """Ingest a DataFrame into the feature group in mini-batches."""
    logger.info("Ingesting %d records into feature group '%s'...", len(df), fg.name)

    # Ensure event_time column exists and is ISO-string
    if event_time_feature_name not in df.columns:
        now = datetime.now(timezone.utc).isoformat()
        df[event_time_feature_name] = now
    else:
        df[event_time_feature_name] = pd.to_datetime(df[event_time_feature_name], errors="coerce").fillna(
            datetime.now(timezone.utc)
        )
        df[event_time_feature_name] = df[event_time_feature_name].dt.tz_convert("UTC").dt.strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )

    total = len(df)
    ingested_count = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]
        batch_size_actual = len(batch_df)
        logger.info("Ingesting records %d-%d (%d rows)...", start, end - 1, batch_size_actual)
        try:
            fg.ingest(data_frame=batch_df, max_workers=4, wait=True)
            ingested_count += batch_size_actual
            logger.info("Successfully ingested batch %d-%d (total ingested: %d/%d)", start, end - 1, ingested_count, total)
        except Exception as e:
            logger.error("Failed to ingest batch %d-%d: %s", start, end - 1, str(e))
            raise

    logger.info("All ingestion batches completed. Total records ingested: %d/%d", ingested_count, total)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest churn features from Athena into SageMaker Feature Store")
    # Try pipeline input path first, then fallback to local config
    default_config = os.path.join("/opt/ml/processing/input/config", "pipeline_config.yaml")
    if not os.path.exists(default_config):
        default_config = os.path.join(os.path.dirname(__file__), "..", "configs", "pipeline_config.yaml")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to feature_store_poc pipeline_config.yaml",
    )
    parser.add_argument(
        "--sns-topic-arn",
        type=str,
        default=os.environ.get("SNS_TOPIC_ARN"),
        help="SNS topic ARN for failure notifications (optional)",
    )
    parser.add_argument(
        "--notification-email",
        type=str,
        default=os.environ.get("NOTIFICATION_EMAIL"),
        help="Email address for failure notifications (for informational purposes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Override SNS configuration from arguments if provided
    global SNS_TOPIC_ARN, NOTIFICATION_EMAIL
    if args.sns_topic_arn:
        SNS_TOPIC_ARN = args.sns_topic_arn
    if args.notification_email:
        NOTIFICATION_EMAIL = args.notification_email
    
    try:
        config = load_config(args.config)

        region = config["region"]
        athena_db = config["athena_database"]
        athena_table = config["athena_table"]
        athena_output = config["athena_query_results"]

        feature_group_name = config["feature_group_name"]
        record_identifier_name = config["record_identifier_name"]
        event_time_feature_name = config["event_time_feature_name"]
        offline_store_s3_uri = config["offline_store_s3_uri"]
        online_store_enabled = bool(config.get("online_store_enabled", False))
        role_arn = config["role_arn"]

        logger.info("Region: %s", region)
        logger.info("Athena source: %s.%s", athena_db, athena_table)
        logger.info("Feature group: %s", feature_group_name)

        # 1) Read raw data from Athena
        query = f"SELECT * FROM {athena_table}"
        df_raw = run_athena_query(query, database=athena_db, region=region, output_location=athena_output)
        if df_raw.empty:
            error_msg = "Athena query returned no rows; aborting."
            logger.error(error_msg)
            send_failure_notification("Feature Store Ingestion - Churn Classification", error_msg)
            return 1

        # 2) Clean / normalize data (but keep raw categorical semantics)
        df_clean = clean_raw_dataframe(df_raw)

        # 3) Initialize SageMaker session and feature group
        boto_sess = boto3.Session(region_name=region)
        sm_session = Session(boto_session=boto_sess)

        fg = ensure_feature_group(
            sagemaker_session=sm_session,
            feature_group_name=feature_group_name,
            df=df_clean,
            record_identifier_name=record_identifier_name,
            event_time_feature_name=event_time_feature_name,
            offline_store_s3_uri=offline_store_s3_uri,
            online_store_enabled=online_store_enabled,
            role_arn=role_arn,
        )

        # 4) Ingest cleaned data
        ingest_dataframe_to_feature_group(
            fg=fg,
            df=df_clean,
            event_time_feature_name=event_time_feature_name,
            batch_size=500,
        )

        logger.info("Feature Store ingestion completed.")
        return 0
    except Exception as exc:
        logger.exception("Feature Store ingestion step failed")
        send_failure_notification("Feature Store Ingestion - Churn Classification", str(exc))
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        # Catch any unhandled exceptions (including import errors)
        error_msg = f"Unhandled exception: {str(exc)}"
        logger.exception(error_msg)
        send_failure_notification_early("Feature Store Ingestion - Churn Classification", error_msg)
        sys.exit(1)


