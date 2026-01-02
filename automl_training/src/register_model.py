#!/usr/bin/env python
"""Register AutoML model in Model Registry with conditional approval."""
import os
import sys
import argparse
import logging
import boto3
import json

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


def ensure_model_package_group_exists(model_package_group_name: str, region: str) -> None:
    """Create model package group if it doesn't exist."""
    sm = boto3.client("sagemaker", region_name=region)
    
    try:
        sm.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
        logger.info(f"Model package group '{model_package_group_name}' already exists")
    except sm.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            # Group doesn't exist, create it
            logger.info(f"Creating model package group: {model_package_group_name}")
            sm.create_model_package_group(
                ModelPackageGroupName=model_package_group_name,
                ModelPackageGroupDescription=f"Model package group for AutoML models"
            )
            logger.info(f"Created model package group: {model_package_group_name}")
        else:
            raise


def register_model(
    model_name: str,
    model_package_group_name: str,
    evaluation_s3_uri: str,
    accuracy_threshold: float,
    region: str,
    approval_status: str = "PendingManualApproval"
) -> str:
    """Register model in Model Registry with conditional approval."""
    sm = boto3.client("sagemaker", region_name=region)
    
    # Ensure model package group exists
    ensure_model_package_group_exists(model_package_group_name, region)
    
    # Load evaluation metrics
    import urllib.parse
    s3 = boto3.client("s3", region_name=region)
    bucket_key = evaluation_s3_uri.replace("s3://", "").split("/", 1)
    bucket = bucket_key[0]
    key = f"{bucket_key[1]}/evaluation.json" if len(bucket_key) > 1 else "evaluation.json"
    
    response = s3.get_object(Bucket=bucket, Key=key)
    evaluation = json.loads(response["Body"].read())
    
    accuracy = evaluation.get("accuracy", 0.0)
    logger.info(f"Model accuracy: {accuracy:.4f}, Threshold: {accuracy_threshold:.4f}")
    
    # Auto-approve if accuracy >= threshold
    if accuracy >= accuracy_threshold:
        approval_status = "Approved"
        logger.info(f"Model accuracy ({accuracy:.4f}) >= threshold ({accuracy_threshold:.4f}), auto-approving")
    else:
        logger.info(f"Model accuracy ({accuracy:.4f}) < threshold ({accuracy_threshold:.4f}), setting to PendingManualApproval")
    
    # Get model details
    model_desc = sm.describe_model(ModelName=model_name)
    
    # AutoML models may have Containers (plural) or PrimaryContainer
    if "Containers" in model_desc:
        raw_containers = model_desc["Containers"]
    elif "PrimaryContainer" in model_desc:
        raw_containers = [model_desc["PrimaryContainer"]]
    else:
        raise RuntimeError("Model does not have Containers or PrimaryContainer")
    
    # Filter container fields to only include valid ones for create_model_package
    # Valid fields: ContainerHostname, Image, ImageDigest, ModelDataUrl, ProductId, 
    # Environment, ModelInput, Framework, FrameworkVersion, NearestModelName, AdditionalS3DataSource
    valid_container_fields = {
        "ContainerHostname", "Image", "ImageDigest", "ModelDataUrl", "ProductId",
        "Environment", "ModelInput", "Framework", "FrameworkVersion", 
        "NearestModelName", "AdditionalS3DataSource"
    }
    
    containers = []
    for container in raw_containers:
        filtered_container = {
            k: v for k, v in container.items() 
            if k in valid_container_fields
        }
        containers.append(filtered_container)
    
    logger.info(f"Filtered containers: {len(containers)} container(s) with valid fields")
    
    # Construct model metrics dictionary directly (no SageMaker SDK needed)
    # Build the full S3 URI to the evaluation.json file
    metrics_s3_uri = f"{evaluation_s3_uri}/evaluation.json" if not evaluation_s3_uri.endswith(".json") else evaluation_s3_uri
    
    # Use ModelQuality instead of ModelStatistics (correct parameter name)
    model_metrics = {
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": metrics_s3_uri,
            }
        }
    }
    
    # Create model package
    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=f"AutoML model with accuracy={accuracy:.4f}",
        InferenceSpecification={
            "Containers": containers,
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv", "application/json"],
        },
        ModelMetrics=model_metrics,
        ModelApprovalStatus=approval_status,
    )
    
    model_package_arn = response["ModelPackageArn"]
    logger.info(f"Registered model package: {model_package_arn}")
    logger.info(f"Approval status: {approval_status}")
    
    return model_package_arn


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-info-path", type=str, required=True)
        parser.add_argument("--model-package-group", type=str, required=True)
        parser.add_argument("--evaluation-s3-uri", type=str, required=True)
        parser.add_argument("--accuracy-threshold", type=float, required=True)
        parser.add_argument("--region", type=str, default="us-west-2")
        parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
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
        logger.info(f"Registering model: {model_name}")
        
        model_package_arn = register_model(
            model_name,
            args.model_package_group,
            args.evaluation_s3_uri,
            args.accuracy_threshold,
            args.region
        )
        
        # Save registration info
        registration_info = {
            "model_package_arn": model_package_arn,
            "model_name": model_name,
            "model_package_group": args.model_package_group,
        }
        
        os.makedirs(args.output_path, exist_ok=True)
        with open(f"{args.output_path}/registration.json", "w") as f:
            json.dump(registration_info, f, indent=2)
        
        logger.info(f"Model registered successfully: {model_package_arn}")
        
    except Exception as exc:
        logger.exception("RegisterModel step failed")
        send_failure_notification("Register Model - AutoML Training Pipeline", str(exc), sns_topic_arn)
        raise

