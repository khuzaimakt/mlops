#!/usr/bin/env python
"""Evaluation script for Classification Model."""
import os
import sys
import argparse
import logging
import json
import boto3
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

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


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
        parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
        parser.add_argument("--evaluation-output", type=str, default="/opt/ml/processing/evaluation")
        parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
        parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))

        args = parser.parse_args()

        # Use SNS configuration from arguments or environment
        sns_topic_arn = args.sns_topic_arn or SNS_TOPIC_ARN
        notification_email = args.notification_email or NOTIFICATION_EMAIL

        # Extract model tarball if needed (SKLearn estimator saves as tar.gz)
        logger.info(f"Checking model directory: {args.model_dir}")
        if os.path.exists(args.model_dir):
            logger.info(f"Contents: {os.listdir(args.model_dir)}")
        else:
            logger.error(f"Model directory not found: {args.model_dir}")
            sys.exit(1)

        # Look for model file or tarball
        model_path = None
        if os.path.exists(f"{args.model_dir}/model.joblib"):
            model_path = f"{args.model_dir}/model.joblib"
        elif os.path.exists(f"{args.model_dir}/model.tar.gz"):
            logger.info("Found model.tar.gz, extracting...")
            import tarfile
            with tarfile.open(f"{args.model_dir}/model.tar.gz", "r:gz") as tar:
                # Use filter parameter if available (Python 3.12+), otherwise use extractall without it
                try:
                    tar.extractall(path=args.model_dir, filter='data')
                except TypeError:
                    # Older Python versions don't support filter parameter
                    tar.extractall(path=args.model_dir)
            # After extraction, model should be in the directory
            if os.path.exists(f"{args.model_dir}/model.joblib"):
                model_path = f"{args.model_dir}/model.joblib"
            else:
                # Sometimes it's in a subdirectory
                for root, dirs, files in os.walk(args.model_dir):
                    if "model.joblib" in files:
                        model_path = os.path.join(root, "model.joblib")
                        break
        else:
            # Search for model.joblib in subdirectories
            for root, dirs, files in os.walk(args.model_dir):
                if "model.joblib" in files:
                    model_path = os.path.join(root, "model.joblib")
                    break

        if model_path is None or not os.path.exists(model_path):
            logger.error(f"Model file not found in {args.model_dir}")
            logger.error(f"Directory contents: {os.listdir(args.model_dir) if os.path.exists(args.model_dir) else 'N/A'}")
            raise FileNotFoundError(f"Model file (model.joblib) not found in {args.model_dir}")

        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # Load model metadata
        metadata_path = f"{args.model_dir}/model_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
            target_col = model_metadata["target_column"]
            feature_cols = model_metadata["feature_columns"]
            logger.info(f"Loaded model metadata: target={target_col}, features={len(feature_cols)}")
        else:
            logger.warning("Model metadata not found, will infer from data")
            target_col = None
            feature_cols = None

        # Load test data
        logger.info("Loading test data...")
        if os.path.exists(f"{args.test_data}/data.parquet"):
            test_df = pd.read_parquet(f"{args.test_data}/data.parquet")
        else:
            test_df = pd.read_csv(f"{args.test_data}/data.csv")

        logger.info(f"Loaded {len(test_df)} test samples")

        # Detect target column if needed
        if target_col is None:
            common_targets = ['target', 'churn', 'loan_approved', 'label', 'y', 'class']
            for col in common_targets:
                if col in test_df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = test_df.columns[-1]
            logger.info(f"Detected target column: {target_col}")

        if target_col not in test_df.columns:
            logger.error(f"Target column '{target_col}' not found in test data")
            sys.exit(1)

        # Prepare features
        if feature_cols:
            # Use features from model metadata
            missing_cols = set(feature_cols) - set(test_df.columns)
            if missing_cols:
                logger.warning(f"Missing features in test data: {missing_cols}")
                feature_cols = [col for col in feature_cols if col in test_df.columns]

            X_test = test_df[feature_cols]
        else:
            # Infer features (all columns except target)
            feature_cols = [col for col in test_df.columns if col != target_col]
            X_test = test_df[feature_cols]
            logger.info(f"Inferred {len(feature_cols)} features")

        y_test = test_df[target_col]

        # Ensure feature order matches training
        if hasattr(model, 'feature_names_in_'):
            # Reorder features to match training order
            X_test = X_test[model.feature_names_in_]

        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")

        # Make predictions
        logger.info("Making predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # ROC-AUC for binary classification
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        else:
            roc_auc = None

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
            "test_samples": int(len(test_df)),
            "n_classes": int(len(np.unique(y_test))),
            "class_distribution": y_test.value_counts().to_dict()
        }

        if roc_auc is not None:
            evaluation["roc_auc"] = float(roc_auc)

        # Add per-class metrics if multiclass
        if len(np.unique(y_test)) > 2:
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

            evaluation["precision_per_class"] = precision_per_class.tolist()
            evaluation["recall_per_class"] = recall_per_class.tolist()
            evaluation["f1_per_class"] = f1_per_class.tolist()

        os.makedirs(args.evaluation_output, exist_ok=True)
        with open(f"{args.evaluation_output}/evaluation.json", "w") as f:
            json.dump(evaluation, f, indent=2)

        logger.info("Evaluation completed successfully")
    except Exception as exc:  # Ensure failures are reported via SNS
        logger.exception("EvaluateModel step failed")
        send_failure_notification("Evaluate Model - Classification Model Training Pipeline", str(exc))
        raise
