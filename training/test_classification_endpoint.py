#!/usr/bin/env python3
"""
Test script for the 'classification-model-endpoint' SageMaker endpoint.

Usage:
  python3 test_classification_endpoint.py --region us-west-2 --endpoint-name classification-model-endpoint --input-csv sample_input.csv

The input CSV should have the same feature columns (and order) as the model expects.
Each row will be sent as a separate inference request.
"""

import argparse
import csv
import json
import sys

import boto3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        type=str,
        default="us-west-2",
        help="AWS region where the SageMaker endpoint is deployed",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="classification-model-endpoint",
        help="Name of the SageMaker endpoint to call",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to a CSV file containing input rows (header + feature columns)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create a SageMaker Runtime client (uses your AWS credentials from env/CLI)
    runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    # Read the CSV file
    try:
        with open(args.input_csv, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("Input CSV is empty.")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    print(f"Loaded {len(data_rows)} rows from {args.input_csv}")
    print(f"Header: {header}")
    print(f"Sending requests to endpoint: {args.endpoint_name} in region {args.region}")
    print("-" * 80)

    for i, row in enumerate(data_rows, start=1):
        # Build CSV payload with header + single row so the endpoint sees column names
        payload = ",".join(header) + "\n" + ",".join(row)
        try:
            response = runtime.invoke_endpoint(
                EndpointName=args.endpoint_name,
                ContentType="text/csv",
                Body=payload.encode("utf-8"),
            )
            result = response["Body"].read().decode("utf-8")
            try:
                parsed = json.loads(result)
            except:
                parsed = result
            print(f"Row {i}: input={row}")
            print(f"  prediction: {parsed}\n")
        except Exception as e:
            print(f"Error invoking endpoint for row {i}: {e}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()