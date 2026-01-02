#!/usr/bin/env python
"""SageMaker Pipeline for Batch Inference."""
import os
import argparse
import yaml
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_session(region: str) -> PipelineSession:
    """Get SageMaker Pipeline session."""
    boto_session = boto3.Session(region_name=region)
    return PipelineSession(boto_session=boto_session)


def build_pipeline(config: dict) -> Pipeline:
    """Build the Batch Inference SageMaker Pipeline."""
    region = config["region"]
    bucket = config["bucket"]
    prefix = config["prefix"]
    role = config["role_arn"]
    pipeline_session = get_pipeline_session(region)

    processing_instance_type = config["processing_instance_type"]

    sns_topic_arn = config.get("sns_topic_arn")
    notification_email = config.get("notification_email")

    # Step 0: Data Sourcing from Feature Store (replaces Athena input table)
    data_source_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-data-sourcing",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )

    step_data_source = ProcessingStep(
        name="DataSource",
        processor=data_source_processor,
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/raw",
                destination=f"s3://{bucket}/{prefix}/data/raw",
                output_name="raw_data",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "data_source.py"),
        job_arguments=[
            "--feature-group-name", config["feature_group_name"],
            "--customer-id-start", str(config["customer_id_start"]),
            "--customer-id-end", str(config["customer_id_end"]),
            "--region", region,
            "--output-path", "/opt/ml/processing/output/raw",
            "--athena-output", config["athena_query_results"],
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
    )

    # Step 1: Batch Inference (preprocess + predict + optional evaluation)
    inference_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-batch-inference",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )

    step_batch_inference = ProcessingStep(
        name="BatchInference",
        processor=inference_processor,
        inputs=[
            ProcessingInput(
                source=step_data_source.properties.ProcessingOutputConfig.Outputs["raw_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/predictions",
                destination=f"s3://{bucket}/{prefix}/predictions",
                output_name="predictions",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/evaluation",
                destination=f"s3://{bucket}/{prefix}/evaluation",
                output_name="evaluation",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "batch_inference.py"),
        job_arguments=[
            "--model-package-group", config["model_package_group"],
            "--model-approval-status", config.get("model_approval_status", "Approved"),
            "--region", region,
            "--input-data", "/opt/ml/processing/input",
            "--output-path", "/opt/ml/processing/output",
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_data_source.name],
    )

    # Step 2: Publish predictions to Athena
    publish_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-publish-athena",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=10,
    )

    step_publish = ProcessingStep(
        name="PublishToAthena",
        processor=publish_processor,
        inputs=[
            ProcessingInput(
                source=step_batch_inference.properties.ProcessingOutputConfig.Outputs["predictions"].S3Output.S3Uri,
                destination="/opt/ml/processing/predictions",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/athena-publish",
                output_name="athena_publish",
            )
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "publish_to_athena.py"),
        job_arguments=[
            "--database", config["athena_database"],
            "--table", config["athena_output_table"],
            "--input-s3", step_batch_inference.properties.ProcessingOutputConfig.Outputs["predictions"].S3Output.S3Uri,
            "--athena-output", config["athena_query_results"],
            "--region", region,
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_batch_inference.name],
    )

    pipeline_steps = [step_data_source, step_batch_inference, step_publish]

    pipeline = Pipeline(
        name=config["pipeline_name"],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml")
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(config)

    pipeline.upsert(role_arn=config["role_arn"])
    print(f"Pipeline '{config['pipeline_name']}' created/updated successfully")

    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print("View execution in SageMaker Studio or AWS Console")


if __name__ == "__main__":
    main()

