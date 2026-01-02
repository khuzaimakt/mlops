#!/usr/bin/env python
"""Execute the Feature Store Ingestion Pipeline."""
import os
import sys
import argparse
import yaml
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_session(region: str) -> PipelineSession:
    boto_session = boto3.Session(region_name=region)
    return PipelineSession(boto_session=boto_session)


def main():
    parser = argparse.ArgumentParser(description="Execute Feature Store Ingestion Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml"),
        help="Path to pipeline_config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    region = config["region"]
    pipeline_name = config.get("pipeline_name", f"{config['project_name']}-pipeline")

    # Load pipeline by name
    pipeline_session = get_pipeline_session(region)
    pipeline = Pipeline(
        name=pipeline_name,
        sagemaker_session=pipeline_session,
    )

    # Start execution
    print(f"Starting pipeline execution: {pipeline_name}")
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print(f"View execution in SageMaker Studio or AWS Console: {execution.arn}")


if __name__ == "__main__":
    main()

