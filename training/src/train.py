#!/usr/bin/env python
"""Training script for Classification Model - adapted for SageMaker."""
import subprocess
import sys
import logging
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SKLearnProcessor already has scikit-learn, pandas, numpy pre-installed
# Only need to install pyarrow if needed for parquet support
logger.info("Installing pyarrow for parquet support...")
subprocess.run([sys.executable, "-m", "pip", "install", "pyarrow", "--quiet"], check=False)

import os
import argparse
import logging
import json
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
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


def load_metadata(metadata_path):
    """Load preprocessing metadata."""
    with open(metadata_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()

        # SageMaker environment variables
        parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
        parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
        parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

        # Hyperparameters
        parser.add_argument("--n-estimators", type=int, default=100)
        parser.add_argument("--max-depth", type=int, default=10)
        parser.add_argument("--min-samples-split", type=int, default=2)
        parser.add_argument("--min-samples-leaf", type=int, default=1)
        parser.add_argument("--test-split", type=float, default=0.2)
        parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
        parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))

        args = parser.parse_args()

        # Use SNS configuration from arguments or environment
        sns_topic_arn = args.sns_topic_arn or SNS_TOPIC_ARN
        notification_email = args.notification_email or NOTIFICATION_EMAIL

        # Load metadata
        metadata_path = f"{args.train}/metadata.json"
        if os.path.exists(metadata_path):
            metadata = load_metadata(metadata_path)
            target_col = metadata["target_column"]
            logger.info(f"Loaded metadata: {metadata}")
        else:
            logger.warning("Metadata file not found, attempting to detect target column")
            target_col = None

        # Load training data
        logger.info(f"Loading training data from {args.train}...")
        if os.path.exists(f"{args.train}/data.parquet"):
            train_df = pd.read_parquet(f"{args.train}/data.parquet")
        else:
            train_df = pd.read_csv(f"{args.train}/data.csv")

        logger.info(f"Loaded {len(train_df)} training rows")

        # Detect target column if not in metadata
        if target_col is None:
            common_targets = ['target', 'churn', 'loan_approved', 'label', 'y', 'class']
            for col in common_targets:
                if col in train_df.columns:
                    target_col = col
                    break
            if target_col is None:
                target_col = train_df.columns[-1]
            logger.info(f"Detected target column: {target_col}")

        if target_col not in train_df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            sys.exit(1)

        # Prepare features and target
        feature_cols = [col for col in train_df.columns if col != target_col]
        X_train_full = train_df[feature_cols]
        y_train_full = train_df[target_col]

        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
        logger.info(f"Target classes: {sorted(y_train_full.unique())}")
        logger.info(f"Class distribution:\n{y_train_full.value_counts()}")

        # Create validation split from training data
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=args.test_split,
            random_state=42,
            stratify=y_train_full
        )

        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Train model
        logger.info("Training RandomForest classifier...")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        model.fit(X_train, y_train)

        # Evaluate on validation set
        logger.info("Evaluating model on validation set...")
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_proba_val = model.predict_proba(X_val)[:, 1] if len(np.unique(y_train)) == 2 else None

        train_accuracy = accuracy_score(y_train, y_pred_train)
        val_accuracy = accuracy_score(y_val, y_pred_val)
        val_precision = precision_score(y_val, y_pred_val, average='weighted')
        val_recall = recall_score(y_val, y_pred_val, average='weighted')
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')

        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        logger.info(f"Validation Precision: {val_precision:.4f}")
        logger.info(f"Validation Recall: {val_recall:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")

        if y_pred_proba_val is not None:
            val_roc_auc = roc_auc_score(y_val, y_pred_proba_val)
            logger.info(f"Validation ROC-AUC: {val_roc_auc:.4f}")
        else:
            val_roc_auc = None

        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_val, y_pred_val)}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']:<30} {row['importance']:.4f}")

        # Save metrics
        metrics = {
            "accuracy": float(val_accuracy),
            "precision": float(val_precision),
            "recall": float(val_recall),
            "f1_score": float(val_f1),
            "train_accuracy": float(train_accuracy),
            "n_classes": int(len(np.unique(y_train))),
            "n_features": int(len(feature_cols)),
            "n_train_samples": int(len(X_train)),
            "n_val_samples": int(len(X_val))
        }

        if val_roc_auc is not None:
            metrics["roc_auc"] = float(val_roc_auc)

        os.makedirs(args.output_data_dir, exist_ok=True)
        with open(f"{args.output_data_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model and metadata
        logger.info(f"Saving model to {args.model_dir}...")
        os.makedirs(args.model_dir, exist_ok=True)

        joblib.dump(model, f"{args.model_dir}/model.joblib")

        # Save feature columns and target info (used by inference)
        model_metadata = {
            "target_column": target_col,
            "feature_columns": feature_cols,
            "n_classes": int(len(np.unique(y_train))),
            "class_names": [str(c) for c in sorted(np.unique(y_train))],
            "feature_importance": feature_importance.to_dict("records")[:20],  # Top 20
        }

        with open(f"{args.model_dir}/model_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        # Copy preprocessing artifacts (metadata + label encoders) into model_dir for inference
        try:
            prep_meta_src = os.path.join(args.train, "metadata.json")
            prep_enc_src = os.path.join(args.train, "label_encoders.pkl")

            prep_meta_dst = os.path.join(args.model_dir, "preprocessing_metadata.json")
            if os.path.exists(prep_meta_src):
                shutil.copy2(prep_meta_src, prep_meta_dst)
                logger.info(f"Copied preprocessing metadata to {prep_meta_dst}")
            else:
                logger.warning(f"Preprocessing metadata not found at {prep_meta_src}")

            if os.path.exists(prep_enc_src):
                enc_dst = os.path.join(args.model_dir, "label_encoders.pkl")
                shutil.copy2(prep_enc_src, enc_dst)
                logger.info(f"Copied label_encoders.pkl to {enc_dst}")
            else:
                logger.warning(f"label_encoders.pkl not found at {prep_enc_src}; "
                               "inference will expect pre-encoded features.")
        except Exception as copy_err:
            logger.warning(f"Failed to copy preprocessing artifacts for inference: {copy_err}")

        logger.info("Training completed successfully")
    except Exception as exc:  # Ensure failures are reported via SNS
        logger.exception("TrainModel step failed")
        send_failure_notification("Train Model - Classification Model Training Pipeline", str(exc))
        raise

