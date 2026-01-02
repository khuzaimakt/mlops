#!/usr/bin/env python
"""Execute an existing SageMaker Pipeline."""
import os
import argparse
import yaml
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_session(region: str) -> PipelineSession:
    """Get SageMaker Pipeline session."""
    boto_session = boto3.Session(region_name=region)
    return PipelineSession(
        boto_session=boto_session,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml")
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default=None,
        help="Override pipeline name from config"
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    pipeline_name = args.pipeline_name or config["pipeline_name"]
    region = config["region"]
    
    # Use boto3 client directly to start pipeline execution
    sagemaker_client = boto3.client("sagemaker", region_name=region)
    
    try:
        # Start pipeline execution
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name
        )
        
        execution_arn = response["PipelineExecutionArn"]
        print(f"Pipeline execution started: {execution_arn}")
        print(f"View execution status at:")
        print(f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline_name}/executions")
    except Exception as e:
        print(f"Error starting pipeline execution: {e}")
        print(f"\nMake sure the pipeline '{pipeline_name}' exists in region {region}")
        raise


if __name__ == "__main__":
    main()

