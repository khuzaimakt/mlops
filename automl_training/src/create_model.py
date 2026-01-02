#!/usr/bin/env python
"""Create a SageMaker model from AutoML best candidate."""
import os
import sys
import argparse
import logging
import boto3
import json
import time

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


def get_best_candidate_from_automl_job(automl_job_name: str, region: str) -> dict:
    """Get the best candidate from an AutoML job."""
    sm = boto3.client("sagemaker", region_name=region)
    
    job = sm.describe_auto_ml_job(AutoMLJobName=automl_job_name)
    
    if job["AutoMLJobStatus"] != "Completed":
        raise RuntimeError(f"AutoML job is not completed. Status: {job['AutoMLJobStatus']}")
    
    best_candidate = job.get("BestCandidate")
    if not best_candidate:
        raise RuntimeError("No best candidate found in AutoML job")
    
    return best_candidate


def create_model_from_candidate(
    best_candidate: dict,
    model_name: str,
    role_arn: str,
    region: str
) -> str:
    """Create a SageMaker model from the best AutoML candidate."""
    sm = boto3.client("sagemaker", region_name=region)
    
    # Get inference containers from best candidate
    inference_containers = best_candidate.get("InferenceContainers", [])
    if not inference_containers:
        raise RuntimeError("No inference containers found in best candidate")
    
    # Create model with the inference containers
    # With timestamp-based names, models should be unique, so we don't need to handle "already exists"
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=inference_containers,
    )
    logger.info(f"Created model: {model_name}")
    
    return model_name


def get_automl_job_from_pipeline(pipeline_name: str, region: str) -> str:
    """Get AutoML job name from the latest pipeline execution."""
    sm = boto3.client("sagemaker", region_name=region)
    
    executions = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        MaxResults=1,
        SortOrder="Descending"
    )
    
    if not executions.get("PipelineExecutionSummaries"):
        raise RuntimeError(f"No executions found for pipeline: {pipeline_name}")
    
    execution_arn = executions["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
    
    # List step executions to find AutoMLTraining step
    step_executions = sm.list_pipeline_execution_steps(
        PipelineExecutionArn=execution_arn
    )
    
    for step in step_executions.get("PipelineExecutionSteps", []):
        if step["StepName"] == "AutoMLTraining":
            step_metadata = step.get("Metadata", {})
            if "AutoMLJob" in step_metadata:
                automl_job = step_metadata["AutoMLJob"]
                if isinstance(automl_job, dict) and "Arn" in automl_job:
                    arn = automl_job["Arn"]
                    return arn.split("/")[-1]
    
    raise RuntimeError("Could not find AutoML job name from pipeline execution")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--automl-job-name", type=str, default=None)
        parser.add_argument("--pipeline-name", type=str, default=None)
        parser.add_argument("--model-name-base", type=str, default="classification-model-automl")
        parser.add_argument("--role-arn", type=str, required=True)
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

        # Get AutoML job name
        if args.automl_job_name:
            automl_job_name = args.automl_job_name
        elif args.pipeline_name:
            logger.info(f"Looking up AutoML job from pipeline: {args.pipeline_name}")
            automl_job_name = get_automl_job_from_pipeline(args.pipeline_name, args.region)
        else:
            raise ValueError("Either --automl-job-name or --pipeline-name must be provided")

        logger.info(f"Getting best candidate from AutoML job: {automl_job_name}")
        best_candidate = get_best_candidate_from_automl_job(automl_job_name, args.region)
        
        # Generate timestamp-based unique model name
        timestamp = int(time.time())
        model_name = f"{args.model_name_base}-{timestamp}"
        logger.info(f"Creating model with unique name: {model_name}")
        
        model_name = create_model_from_candidate(
            best_candidate,
            model_name,
            args.role_arn,
            args.region
        )
        
        # Save model info
        model_info = {
            "model_name": model_name,
            "automl_job_name": automl_job_name,
            "candidate_name": best_candidate.get("CandidateName"),
            "metric": best_candidate.get("FinalAutoMLJobObjectiveMetric", {}).get("MetricName"),
            "metric_value": best_candidate.get("FinalAutoMLJobObjectiveMetric", {}).get("Value"),
        }
        
        os.makedirs(args.output_path, exist_ok=True)
        with open(f"{args.output_path}/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model created successfully: {model_name}")
        logger.info(f"Best candidate metric: {model_info['metric']} = {model_info['metric_value']}")
        
    except Exception as exc:
        logger.exception("CreateModel step failed")
        send_failure_notification("Create Model - AutoML Training Pipeline", str(exc), sns_topic_arn)
        raise

