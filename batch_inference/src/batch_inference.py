#!/usr/bin/env python
"""Batch inference: preprocess using training artifacts, predict, optional evaluation."""
import os
import argparse
import logging
import tarfile
import tempfile
import json
import sys
import boto3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    except Exception as notify_err:  # pragma: no cover
        logger.error("Failed to send SNS notification: %s", notify_err)


def get_latest_approved_model_package(model_package_group: str, region: str, approval_status: str = "Approved") -> str:
    sm = boto3.client("sagemaker", region_name=region)
    resp = sm.list_model_packages(
        ModelPackageGroupName=model_package_group,
        ModelApprovalStatus=approval_status,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    summaries = resp.get("ModelPackageSummaryList", [])
    if not summaries:
        raise RuntimeError(f"No {approval_status} models found in group {model_package_group}")
    return summaries[0]["ModelPackageArn"]


def download_model_artifacts(model_package_arn: str, region: str, target_dir: str) -> str:
    """Download model.tar.gz from a model package to target_dir; return path to extracted directory."""
    sm = boto3.client("sagemaker", region_name=region)
    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    model_url = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

    s3 = boto3.client("s3", region_name=region)
    bucket, key = model_url.replace("s3://", "").split("/", 1)
    local_tar = os.path.join(target_dir, "model.tar.gz")
    s3.download_file(bucket, key, local_tar)
    logger.info("Downloaded model artifacts to %s", local_tar)

    extract_dir = os.path.join(target_dir, "model")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
    logger.info("Extracted model to %s", extract_dir)
    return extract_dir


def load_model_bundle(model_dir: str):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))

    model_metadata = None
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            model_metadata = json.load(f)

    prep_metadata = None
    prep_meta_path = os.path.join(model_dir, "preprocessing_metadata.json")
    if os.path.exists(prep_meta_path):
        with open(prep_meta_path, "r") as f:
            prep_metadata = json.load(f)

    label_encoders = None
    enc_path = os.path.join(model_dir, "label_encoders.pkl")
    if os.path.exists(enc_path):
        label_encoders = joblib.load(enc_path)

    return {
        "model": model,
        "model_metadata": model_metadata,
        "prep_metadata": prep_metadata,
        "label_encoders": label_encoders,
    }


from typing import Optional


def preprocess(df: pd.DataFrame, prep_metadata: Optional[dict], label_encoders: Optional[dict], target_col: Optional[str]):
    """Apply training-time preprocessing to inference data."""
    if prep_metadata is None:
        return df

    categorical_cols = prep_metadata.get("categorical_columns", []) or []
    processed_df = df.copy()

    # Exclude customer_id from features (it's an identifier, not a feature)
    id_columns = ['customer_id', 'id', 'ID', 'CustomerID', 'Customer_ID']
    for id_col in id_columns:
        if id_col in processed_df.columns and id_col != target_col:
            logger.info(f"Dropping identifier column: {id_col}")
            processed_df = processed_df.drop(columns=[id_col])

    # Missing values: object -> "Unknown", numeric -> median
    for col in processed_df.columns:
        if processed_df[col].dtype == "object":
            processed_df[col] = processed_df[col].fillna("Unknown")
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    # Encode categorical with stored label encoders
    if label_encoders:
        for col in categorical_cols:
            if col in processed_df.columns and col in label_encoders:
                le = label_encoders[col]
                def encode_val(v):
                    v_str = str(v)
                    if v_str in le.classes_:
                        return int(le.transform([v_str])[0])
                    return int(le.transform([le.classes_[0]])[0])
                processed_df[col] = processed_df[col].astype(str).map(encode_val)

    # Drop target if present (for prediction)
    if target_col and target_col in processed_df.columns:
        processed_df = processed_df.drop(columns=[target_col])

    return processed_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-group", type=str, required=True)
    parser.add_argument("--model-approval-status", type=str, default="Approved")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
    parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))
    args = parser.parse_args()

    global SNS_TOPIC_ARN, NOTIFICATION_EMAIL
    if args.sns_topic_arn:
        SNS_TOPIC_ARN = args.sns_topic_arn
    if args.notification_email:
        NOTIFICATION_EMAIL = args.notification_email

    try:
        logger.info("Fetching latest approved model...")
        model_package_arn = get_latest_approved_model_package(
            args.model_package_group, args.region, args.model_approval_status
        )
        logger.info("Using model package: %s", model_package_arn)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = download_model_artifacts(model_package_arn, args.region, tmpdir)
            bundle = load_model_bundle(model_dir)

            model = bundle["model"]
            model_metadata = bundle.get("model_metadata") or {}
            prep_metadata = bundle.get("prep_metadata")
            label_encoders = bundle.get("label_encoders")
            target_col = model_metadata.get("target_column")
            feature_cols = model_metadata.get("feature_columns")

            # Load inference data
            input_files = []
            if os.path.isdir(args.input_data):
                import glob
                csv_files = glob.glob(f"{args.input_data}/*.csv")
                if csv_files:
                    input_files = csv_files
            elif os.path.isfile(args.input_data) and args.input_data.endswith(".csv"):
                input_files = [args.input_data]

            if not input_files:
                raise RuntimeError(f"No CSV files found in {args.input_data}")

            dfs = []
            for file in input_files:
                df_part = pd.read_csv(file)
                dfs.append(df_part)
                logger.info("Loaded %d rows from %s", len(df_part), file)

            df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
            logger.info("Total inference rows: %d", len(df))

            # Keep original for writing predictions with possible target
            original_df = df.copy()

            processed_df = preprocess(df, prep_metadata, label_encoders, target_col)

            if feature_cols:
                missing = set(feature_cols) - set(processed_df.columns)
                if missing:
                    raise ValueError(f"Missing expected feature columns in input: {missing}")
                processed_df = processed_df[feature_cols]

            predictions = model.predict(processed_df.values)
            original_df["prediction"] = predictions

            # Optional evaluation if target present
            evaluation = {}
            if target_col and target_col in original_df.columns:
                try:
                    y_true = original_df[target_col]
                    evaluation = {
                        "accuracy": float(accuracy_score(y_true, predictions)),
                        "precision": float(precision_score(y_true, predictions, average="weighted")),
                        "recall": float(recall_score(y_true, predictions, average="weighted")),
                        "f1": float(f1_score(y_true, predictions, average="weighted")),
                        "n": int(len(y_true)),
                    }
                    logger.info("Computed evaluation metrics: %s", evaluation)
                except Exception as eval_err:
                    logger.warning("Could not compute evaluation metrics: %s", eval_err)

            # Write outputs
            pred_dir = os.path.join(args.output_path, "predictions")
            eval_dir = os.path.join(args.output_path, "evaluation")
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)

            pred_path = os.path.join(pred_dir, "predictions.csv")
            original_df.to_csv(pred_path, index=False)
            logger.info("Saved predictions to %s", pred_path)

            eval_path = os.path.join(eval_dir, "evaluation.json")
            with open(eval_path, "w") as f:
                json.dump(evaluation, f, indent=2)
            logger.info("Saved evaluation to %s", eval_path)

    except Exception as exc:
        logger.exception("Batch inference step failed")
        send_failure_notification("Batch Inference - Churn Classification Model Batch Pipeline", str(exc))
        raise


if __name__ == "__main__":
    main()

