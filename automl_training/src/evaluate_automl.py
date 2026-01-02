#!/usr/bin/env python
"""Evaluate AutoML model using batch transform."""
import os
import sys
import argparse
import logging
import boto3
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SNS configuration (overridden in main() via CLI args when run in pipeline)
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")


def send_failure_notification(job_name: str, error_message: str, sns_topic_arn: str = None) -> None:
    """Publish a failure notification to SNS."""
    try:
        topic_arn = sns_topic_arn or SNS_TOPIC_ARN
        if not topic_arn:
            logger.warning("SNS topic ARN not provided, skipping notification")
            return
            
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        sns = boto3.client("sns", region_name=region)
        sns.publish(
            TopicArn=topic_arn,
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


def run_batch_transform(
    model_name: str,
    test_s3_uri: str,
    output_s3_uri: str,
    region: str,
    instance_type: str = "ml.m5.xlarge"
) -> str:
    """Run batch transform job and return output S3 URI."""
    sm = boto3.client("sagemaker", region_name=region)
    job_name = f"{model_name}-eval-{int(time.time())}"
    
    logger.info(f"Starting batch transform job: {job_name}")
    sm.create_transform_job(
        TransformJobName=job_name,
        ModelName=model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": test_s3_uri}},
            "ContentType": "text/csv",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "AssembleWith": "Line",
            "Accept": "text/csv",
        },
        TransformResources={
            "InstanceType": instance_type,
            "InstanceCount": 1
        },
    )
    
    # Wait for completion
    logger.info("Waiting for batch transform to complete...")
    while True:
        resp = sm.describe_transform_job(TransformJobName=job_name)
        status = resp["TransformJobStatus"]
        if status in ["Completed", "Failed", "Stopped"]:
            if status != "Completed":
                reason = resp.get("FailureReason", "Unknown")
                raise RuntimeError(f"Batch transform failed: {reason}")
            break
        time.sleep(10)
    
    logger.info(f"Batch transform completed: {job_name}")
    return output_s3_uri


def download_predictions(output_s3_uri: str, region: str) -> pd.DataFrame:
    """Download predictions from S3."""
    s3 = boto3.client("s3", region_name=region)
    
    # Parse S3 URI
    bucket_key = output_s3_uri.replace("s3://", "").split("/", 1)
    bucket = bucket_key[0]
    prefix = bucket_key[1] if len(bucket_key) > 1 else ""
    
    # List and download prediction files
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    predictions = []
    
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".out"):
            local_file = f"/tmp/{os.path.basename(key)}"
            s3.download_file(bucket, key, local_file)
            
            # Read predictions (AutoML outputs CSV format)
            pred_df = pd.read_csv(local_file, header=None)
            predictions.append(pred_df)
    
    if not predictions:
        raise RuntimeError("No prediction files found in S3 output")
    
    return pd.concat(predictions, ignore_index=True)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-info-path", type=str, required=True)
        parser.add_argument("--test-data", type=str, required=True)
        parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
        parser.add_argument("--evaluation-output", type=str, default="/opt/ml/processing/evaluation")
        parser.add_argument("--region", type=str, default="us-west-2")
        parser.add_argument("--instance-type", type=str, default="ml.m5.xlarge")
        parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
        parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))

        args = parser.parse_args()

        # Use SNS configuration from arguments or environment
        sns_topic_arn = args.sns_topic_arn or SNS_TOPIC_ARN
        notification_email = args.notification_email or NOTIFICATION_EMAIL
        
        # Update global SNS variables for use in send_failure_notification
        
        if args.sns_topic_arn:
            SNS_TOPIC_ARN = args.sns_topic_arn
        if args.notification_email:
            NOTIFICATION_EMAIL = args.notification_email

        # Load model name from model_info.json
        logger.info(f"Loading model info from: {args.model_info_path}")
        with open(args.model_info_path, "r") as f:
            model_info = json.load(f)
        model_name = model_info["model_name"]
        logger.info(f"Using model name: {model_name}")

        # Load test data to get ground truth
        logger.info("Loading test data...")
        s3 = boto3.client("s3", region_name=args.region)
        
        # Handle S3 URI or local path
        if args.test_data.startswith("s3://"):
            # Download from S3
            bucket_key = args.test_data.replace("s3://", "").split("/", 1)
            bucket = bucket_key[0]
            prefix = bucket_key[1] if len(bucket_key) > 1 else ""
            
            # List objects to find data file
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            data_file = None
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".csv") or key.endswith(".parquet"):
                    data_file = key
                    break
            
            if not data_file:
                raise RuntimeError(f"No data file found in {args.test_data}")
            
            local_file = f"/tmp/{os.path.basename(data_file)}"
            s3.download_file(bucket, data_file, local_file)
            
            if data_file.endswith(".parquet"):
                test_df = pd.read_parquet(local_file)
            else:
                test_df = pd.read_csv(local_file)
        else:
            # Local path
            if os.path.exists(f"{args.test_data}/data.parquet"):
                test_df = pd.read_parquet(f"{args.test_data}/data.parquet")
            else:
                test_df = pd.read_csv(f"{args.test_data}/data.csv")
        
        # Detect target column
        target_col = "churn"  # Based on config
        if target_col not in test_df.columns:
            # Try common names
            for col in ["target", "label", "y", "class"]:
                if col in test_df.columns:
                    target_col = col
                    break
        
        if target_col not in test_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        y_test = test_df[target_col]
        
        # Prepare test data without target for batch transform
        test_features = test_df.drop(columns=[target_col])
        
        # Upload test features to S3 for batch transform
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            test_features.to_csv(f.name, index=False, header=False)
            temp_test_file = f.name
        
        # Determine bucket from test_data S3 URI or use a default
        if args.test_data.startswith("s3://"):
            bucket_key = args.test_data.replace("s3://", "").split("/", 1)
            bucket = bucket_key[0]
        else:
            # Extract bucket from config or use default
            bucket = "mlbucket-sagemaker"  # Default bucket
        
        test_prefix = f"{model_name}-test-features-{int(time.time())}"
        test_s3_key = f"{test_prefix}/test.csv"
        
        s3.upload_file(temp_test_file, bucket, test_s3_key)
        test_s3_uri = f"s3://{bucket}/{test_prefix}/"
        logger.info(f"Uploaded test features to: {test_s3_uri}")
        
        # Run batch transform
        output_s3_uri = f"s3://{bucket}/{model_name}-predictions-{int(time.time())}/"
        run_batch_transform(
            model_name,
            test_s3_uri,
            output_s3_uri,
            args.region,
            args.instance_type
        )
        
        # Download predictions
        logger.info("Downloading predictions...")
        pred_df = download_predictions(output_s3_uri, args.region)
        
        # AutoML outputs predictions in CSV format (first column is prediction)
        if pred_df.shape[1] > 0:
            y_pred = pred_df.iloc[:, 0].values
        else:
            raise RuntimeError("No predictions found in output")
        
        # Ensure predictions match test data length
        if len(y_pred) != len(y_test):
            logger.warning(f"Prediction length ({len(y_pred)}) != test length ({len(y_test)})")
            min_len = min(len(y_pred), len(y_test))
            y_pred = y_pred[:min_len]
            y_test = y_test[:min_len]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # ROC-AUC (if binary classification and we have probabilities)
        roc_auc = None
        if len(np.unique(y_test)) == 2 and pred_df.shape[1] > 1:
            try:
                # If second column exists, it might be probabilities
                y_pred_proba = pred_df.iloc[:, 1].values[:len(y_test)]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                logger.info(f"ROC-AUC: {roc_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Save evaluation results
        evaluation = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "test_samples": int(len(y_test)),
            "n_classes": int(len(np.unique(y_test))),
            "class_distribution": y_test.value_counts().to_dict()
        }
        
        if roc_auc is not None:
            evaluation["roc_auc"] = float(roc_auc)
        
        os.makedirs(args.evaluation_output, exist_ok=True)
        with open(f"{args.evaluation_output}/evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as exc:
        logger.exception("EvaluateModel step failed")
        send_failure_notification("Evaluate Model - AutoML Training Pipeline", str(exc), sns_topic_arn)
        raise

