#!/usr/bin/env python
"""Execute the Batch Inference SageMaker Pipeline."""
import os
import argparse
from pipeline import load_config, build_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml"),
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

