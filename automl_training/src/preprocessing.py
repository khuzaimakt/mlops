#!/usr/bin/env python
"""Data preprocessing for AutoML training."""
import subprocess
import sys
import logging
import os
import argparse
import pandas as pd
import numpy as np
import json
import joblib
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure pyarrow for potential parquet ops
subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow", "--quiet"], check=False)

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


def detect_target_column(df):
    common_targets = ["target", "churn", "loan_approved", "label", "y", "class"]
    for col in common_targets:
        if col in df.columns:
            return col
    return df.columns[-1]


def preprocess_data(df, target_col):
    if len(df) == 0:
        logger.error("No data to preprocess")
        return None, None, None

    processed_df = df.copy()

    # Exclude identifier and metadata columns (not predictive features)
    meta_columns = [
        "event_time",
        "write_time",
        "api_invocation_time",
        "is_deleted",
    ]
    id_columns = ['customer_id', 'id', 'ID', 'CustomerID', 'Customer_ID']
    for col in id_columns + meta_columns:
        if col in processed_df.columns and col != target_col:
            logger.info(f"Dropping non-feature column: {col}")
            processed_df = processed_df.drop(columns=[col])

    # Handle missing values
    for col in processed_df.columns:
        if processed_df[col].dtype == "object":
            processed_df[col] = processed_df[col].fillna("Unknown")
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    # Identify categorical columns
    categorical_cols = processed_df.select_dtypes(include=["object"]).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    label_encoders = {}
    if categorical_cols:
        for col in categorical_cols:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le

    # Encode target if object
    if processed_df[target_col].dtype == "object":
        from sklearn.preprocessing import LabelEncoder

        target_le = LabelEncoder()
        processed_df[target_col] = target_le.fit_transform(processed_df[target_col])

    return processed_df, label_encoders, categorical_cols


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
        parser.add_argument("--output-train", type=str, default="/opt/ml/processing/output/train")
        parser.add_argument("--output-test", type=str, default="/opt/ml/processing/output/test")
        parser.add_argument("--output-meta", type=str, default="/opt/ml/processing/output/meta")
        parser.add_argument("--test-split", type=float, default=0.2)
        parser.add_argument("--target-col", type=str, default=None)
        parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
        parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))
        args = parser.parse_args()

        
        if args.sns_topic_arn:
            SNS_TOPIC_ARN = args.sns_topic_arn
        if args.notification_email:
            NOTIFICATION_EMAIL = args.notification_email

        # Load data
        input_files = []
        if os.path.isdir(args.input_data):
            import glob

            csv_files = glob.glob(f"{args.input_data}/*.csv")
            if csv_files:
                input_files = csv_files
        elif os.path.isfile(args.input_data) and args.input_data.endswith(".csv"):
            input_files = [args.input_data]
        else:
            logger.error(f"No CSV files found in {args.input_data}")
            sys.exit(1)

        if not input_files:
            logger.error("No input files found")
            sys.exit(1)

        dfs = []
        for file in input_files:
            df_part = pd.read_csv(file)
            dfs.append(df_part)
            logger.info("Loaded %d rows from %s", len(df_part), file)

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        logger.info("Total rows loaded: %d", len(df))

        target_col = args.target_col or detect_target_column(df)
        logger.info("Using target column: %s", target_col)

        if target_col not in df.columns:
            logger.error("Target column '%s' not found", target_col)
            sys.exit(1)

        processed_df, label_encoders, categorical_cols = preprocess_data(df, target_col)
        if processed_df is None:
            logger.error("Preprocessing failed")
            sys.exit(1)

        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(
            processed_df, test_size=args.test_split, random_state=42, stratify=processed_df[target_col]
        )

        os.makedirs(args.output_train, exist_ok=True)
        os.makedirs(args.output_test, exist_ok=True)
        os.makedirs(args.output_meta, exist_ok=True)

        train_df.to_csv(f"{args.output_train}/data.csv", index=False)
        test_df.to_csv(f"{args.output_test}/data.csv", index=False)

        metadata = {
            "target_column": target_col,
            "categorical_columns": categorical_cols,
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "class_distribution": processed_df[target_col].value_counts().to_dict(),
        }
        with open(f"{args.output_meta}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if label_encoders:
            joblib.dump(label_encoders, f"{args.output_meta}/label_encoders.pkl")

        logger.info("Preprocessing completed successfully")
    except Exception as exc:
        logger.exception("PreprocessData step failed")
        send_failure_notification("Preprocess Data - AutoML Training Pipeline", str(exc))
        raise

