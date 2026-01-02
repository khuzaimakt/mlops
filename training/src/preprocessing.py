#!/usr/bin/env python
"""Data preprocessing script for Classification Model MLOps Pipeline."""
import subprocess
import sys
import logging
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Install required dependencies (SKLearnProcessor has scikit-learn 1.2.1, but we need pyarrow)
logger.info("Installing additional dependencies (pyarrow)...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "pyarrow",
    "--quiet"
], check=True)
logger.info("Dependencies installed successfully")

import os
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

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
    except Exception as notify_err:  # pragma: no cover - best-effort notification
        logger.error("Failed to send SNS notification: %s", notify_err)


def detect_target_column(df):
    """Detect target column name."""
    common_targets = ['target', 'churn', 'loan_approved', 'label', 'y', 'class']
    for col in common_targets:
        if col in df.columns:
            return col
    # If not found, assume last column is target
    return df.columns[-1]


def preprocess_data(df, target_col):
    """Preprocess data for classification."""
    logger.info("Starting data preprocessing...")
    
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
    logger.info("Handling missing values...")
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].fillna('Unknown')
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    
    # Identify categorical and numerical columns
    categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    if categorical_cols:
        logger.info("Encoding categorical variables...")
        for col in categorical_cols:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle target variable
    if processed_df[target_col].dtype == 'object':
        target_le = LabelEncoder()
        processed_df[target_col] = target_le.fit_transform(processed_df[target_col])
        logger.info(f"Encoded target: {target_le.classes_}")
    
    logger.info(f"Preprocessing complete. Dataset shape: {processed_df.shape}")
    
    return processed_df, label_encoders, categorical_cols


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
        parser.add_argument("--output-train", type=str, default="/opt/ml/processing/output/train")
        parser.add_argument("--output-test", type=str, default="/opt/ml/processing/output/test")
        parser.add_argument("--test-split", type=float, default=0.2)
        parser.add_argument("--target-col", type=str, default=None, help="Target column name (auto-detected if not provided)")
        parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
        parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))

        args = parser.parse_args()

        # Use SNS configuration from arguments or environment
        sns_topic_arn = args.sns_topic_arn or SNS_TOPIC_ARN
        notification_email = args.notification_email or NOTIFICATION_EMAIL

        # Load data
        logger.info(f"Loading data from {args.input_data}...")
        input_files = []
        if os.path.isdir(args.input_data):
            import glob
            csv_files = glob.glob(f"{args.input_data}/*.csv")
            if csv_files:
                input_files = csv_files
        elif os.path.isfile(args.input_data) and args.input_data.endswith('.csv'):
            input_files = [args.input_data]
        else:
            logger.error(f"No CSV files found in {args.input_data}")
            sys.exit(1)

        if not input_files:
            logger.error("No input files found")
            sys.exit(1)

        # Load and combine CSV files
        dfs = []
        for file in input_files:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        logger.info(f"Total rows loaded: {len(df)}")

        # Detect or use provided target column
        target_col = args.target_col or detect_target_column(df)
        logger.info(f"Using target column: {target_col}")

        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            sys.exit(1)

        # Preprocess
        processed_df, label_encoders, categorical_cols = preprocess_data(df, target_col)

        if processed_df is None:
            logger.error("Preprocessing failed")
            sys.exit(1)

        # Split data
        logger.info(f"Splitting data (test_size={args.test_split})...")
        train_df, test_df = train_test_split(
            processed_df,
            test_size=args.test_split,
            random_state=42,
            stratify=processed_df[target_col]  # Stratified split for classification
        )

        logger.info(f"Split data: Train={len(train_df)}, Test={len(test_df)}")
        logger.info(f"Train target distribution:\n{train_df[target_col].value_counts()}")
        logger.info(f"Test target distribution:\n{test_df[target_col].value_counts()}")

        # Save preprocessed data
        os.makedirs(args.output_train, exist_ok=True)
        os.makedirs(args.output_test, exist_ok=True)

        train_df.to_csv(f"{args.output_train}/data.csv", index=False)
        train_df.to_parquet(f"{args.output_train}/data.parquet", index=False)

        test_df.to_csv(f"{args.output_test}/data.csv", index=False)
        test_df.to_parquet(f"{args.output_test}/data.parquet", index=False)

        # Save label encoders and metadata
        import json

        metadata = {
            "target_column": target_col,
            "categorical_columns": categorical_cols,
            "n_train_samples": len(train_df),
            "n_test_samples": len(test_df),
            "n_classes": processed_df[target_col].nunique(),
            "class_distribution": processed_df[target_col].value_counts().to_dict()
        }

        with open(f"{args.output_train}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        if label_encoders:
            joblib.dump(label_encoders, f"{args.output_train}/label_encoders.pkl")
            logger.info(f"Saved {len(label_encoders)} label encoders")

        logger.info(f"Saved training data to {args.output_train}")
        logger.info(f"Saved test data to {args.output_test}")
        logger.info("Preprocessing step completed successfully")
    except Exception as exc:  # Ensure failures are reported via SNS
        logger.exception("PreprocessData step failed")
        send_failure_notification("Preprocess Data - Classification Model Training Pipeline", str(exc))
        raise

