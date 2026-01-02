#!/usr/bin/env python
"""SageMaker Pipeline for Feature Store Ingestion."""
import os
import argparse
import logging
import yaml
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.spark.processing import PySparkProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_session(region: str) -> PipelineSession:
    boto_session = boto3.Session(region_name=region)
    return PipelineSession(boto_session=boto_session)


def build_pipeline(config: dict, config_path: str = None) -> Pipeline:
    """Build the Feature Store Ingestion SageMaker Pipeline."""
    region = config["region"]
    bucket = config["bucket"]
    prefix = config["prefix"]
    role = config["role_arn"]
    pipeline_session = get_pipeline_session(region)

    processing_instance_type = config.get("processing_instance_type", "ml.m5.xlarge")
    sns_topic_arn = config.get("sns_topic_arn")
    notification_email = config.get("notification_email")
    
    # Only pass SNS arguments if they are provided (not None or empty)
    job_args = []
    if sns_topic_arn:
        job_args.extend(["--sns-topic-arn", sns_topic_arn])
    if notification_email:
        job_args.extend(["--notification-email", notification_email])

    # Step: Ingest to Feature Store
    # Use PySparkProcessor which has sagemaker pre-installed (avoids dependency conflicts)
    ingest_processor = PySparkProcessor(
        framework_version="3.1",
        py_version="py37",
        container_version="1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-ingest-feature-store",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )

    # Upload config file to S3 so it can be passed as input to the processing job
    config_s3_path = f"s3://{bucket}/{prefix}/config/pipeline_config.yaml"
    s3 = boto3.client("s3", region_name=region)
    config_bucket, config_key = config_s3_path.replace("s3://", "").split("/", 1)
    config_local_path = config_path or os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml")
    try:
        s3.upload_file(config_local_path, config_bucket, config_key)
        logger.info(f"Uploaded config to {config_s3_path}")
    except Exception as e:
        logger.warning(f"Could not upload config to S3 (may already exist): {e}")

    step_ingest = ProcessingStep(
        name="IngestToFeatureStore",
        processor=ingest_processor,
        inputs=[
            ProcessingInput(
                source=config_s3_path,
                destination="/opt/ml/processing/input/config",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/ingestion_output",
                output_name="ingestion_output",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "ingest_to_feature_store.py"),
        job_arguments=job_args,
    )

    # Build pipeline
    pipeline = Pipeline(
        name=config.get("pipeline_name", f"{config['project_name']}-pipeline"),
        steps=[step_ingest],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(config, config_path=args.config)

    pipeline.upsert(role_arn=config["role_arn"])
    print(f"Pipeline '{config.get('pipeline_name', config['project_name'] + '-pipeline')}' created/updated successfully")
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print("View execution in SageMaker Studio or AWS Console")


if __name__ == "__main__":
    main()

