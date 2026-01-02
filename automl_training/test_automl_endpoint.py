#!/usr/bin/env python3
"""
Test script for the AutoML SageMaker endpoint.

This script:
1. Loads raw categorical data from CSV (with headers)
2. Applies preprocessing (missing values, label encoding) using encoders from S3
3. Sends preprocessed data to the AutoML endpoint

Usage:
  python test_automl_endpoint.py \
    --region us-west-2 \
    --endpoint-name automl-classification-endpoint \
    --input-csv sample_input.csv \
    --preprocessing-s3-uri s3://mlbucket-sagemaker/mlops/automl_training/data/processed/meta/
"""

import argparse
import csv
import json
import sys
import tempfile
import os
import yaml

import boto3
import pandas as pd
import joblib
import numpy as np


def find_latest_preprocessing_artifacts(pipeline_name: str, region: str) -> str:
    """Find the latest preprocessing artifacts S3 URI from pipeline execution."""
    sm = boto3.client("sagemaker", region_name=region)
    
    # Get latest pipeline execution
    executions = sm.list_pipeline_executions(
        PipelineName=pipeline_name,
        MaxResults=1,
        SortOrder="Descending"
    )
    
    if not executions.get("PipelineExecutionSummaries"):
        raise RuntimeError(f"No executions found for pipeline: {pipeline_name}")
    
    execution_arn = executions["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
    print(f"Found latest pipeline execution: {execution_arn}")
    
    # Get step executions
    steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)
    
    # Find PreprocessData step
    for step in steps.get("PipelineExecutionSteps", []):
        if step["StepName"] == "PreprocessData":
            metadata = step.get("Metadata", {})
            outputs = metadata.get("ProcessingOutputConfig", {}).get("Outputs", [])
            
            # Find the "meta" output
            for output in outputs:
                if output.get("OutputName") == "meta":
                    s3_uri = output.get("S3Output", {}).get("S3Uri")
                    if s3_uri:
                        print(f"Found preprocessing artifacts at: {s3_uri}")
                        return s3_uri
    
    raise RuntimeError("Could not find preprocessing artifacts from pipeline execution")


def load_preprocessing_artifacts(s3_uri: str, region: str) -> tuple:
    """Load preprocessing metadata and label encoders from S3."""
    s3 = boto3.client("s3", region_name=region)
    
    # Parse S3 URI
    bucket_key = s3_uri.replace("s3://", "").split("/", 1)
    bucket = bucket_key[0]
    prefix = bucket_key[1] if len(bucket_key) > 1 else ""
    
    # Download files to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # List objects in the S3 prefix to find the actual file names
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            objects = response.get("Contents", [])
            
            if not objects:
                # Try listing with trailing slash
                if not prefix.endswith("/"):
                    prefix = f"{prefix}/"
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                objects = response.get("Contents", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list objects in {s3_uri}: {e}")
        
        metadata_path = None
        encoders_path = None
        
        # Find metadata.json and label_encoders.pkl
        for obj in objects:
            key = obj["Key"]
            filename = os.path.basename(key)
            
            if filename == "metadata.json":
                metadata_path = os.path.join(tmpdir, "metadata.json")
                s3.download_file(bucket, key, metadata_path)
            elif filename == "label_encoders.pkl":
                encoders_path = os.path.join(tmpdir, "label_encoders.pkl")
                s3.download_file(bucket, key, encoders_path)
        
        # Load metadata
        if not metadata_path or not os.path.exists(metadata_path):
            # Try direct download
            metadata_key = f"{prefix}/metadata.json" if prefix else "metadata.json"
            metadata_path = os.path.join(tmpdir, "metadata.json")
            try:
                s3.download_file(bucket, metadata_key, metadata_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download metadata.json from {s3_uri}. Available files: {[obj['Key'] for obj in objects]}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load label encoders
        label_encoders = None
        if encoders_path and os.path.exists(encoders_path):
            label_encoders = joblib.load(encoders_path)
        else:
            # Try direct download
            encoders_key = f"{prefix}/label_encoders.pkl" if prefix else "label_encoders.pkl"
            encoders_path = os.path.join(tmpdir, "label_encoders.pkl")
            try:
                s3.download_file(bucket, encoders_key, encoders_path)
                label_encoders = joblib.load(encoders_path)
            except Exception as e:
                print(f"Warning: Could not load label_encoders.pkl: {e}")
                print("Continuing without label encoders (assuming data is already encoded)")
        
        return metadata, label_encoders


def preprocess_data(df: pd.DataFrame, metadata: dict, label_encoders: dict = None) -> pd.DataFrame:
    """Apply preprocessing to match preprocessing.py logic."""
    processed_df = df.copy()
    
    target_col = metadata.get("target_column", "churn")
    categorical_cols = metadata.get("categorical_columns", [])
    
    # Drop identifier and metadata columns (not used as features)
    meta_columns = [
        "event_time",
        "write_time",
        "api_invocation_time",
        "is_deleted",
    ]
    id_columns = ["customer_id", "id", "ID", "CustomerID", "Customer_ID"]
    for col in id_columns + meta_columns:
        if col in processed_df.columns and col != target_col:
            processed_df = processed_df.drop(columns=[col])
    
    # Remove target column if present (not needed for inference)
    if target_col in processed_df.columns:
        processed_df = processed_df.drop(columns=[target_col])
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
    
    # Handle missing values - exactly as in preprocessing.py
    for col in processed_df.columns:
        if processed_df[col].dtype == "object":
            processed_df[col] = processed_df[col].fillna("Unknown")
        else:
            median_val = processed_df[col].median()
            processed_df[col] = processed_df[col].fillna(median_val)
    
    # Apply label encoding to categorical columns
    if label_encoders and categorical_cols:
        for col in categorical_cols:
            if col in processed_df.columns and col in label_encoders:
                le = label_encoders[col]
                # Convert to string and encode
                def encode_val(v):
                    v_str = str(v)
                    if v_str in le.classes_:
                        return int(le.transform([v_str])[0])
                    # Unseen category: use first class as fallback
                    print(f"Warning: Unseen category '{v_str}' in column '{col}', using fallback")
                    return int(le.transform([le.classes_[0]])[0])
                
                processed_df[col] = processed_df[col].astype(str).map(encode_val)
    
    return processed_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test AutoML SageMaker endpoint with raw categorical data"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-west-2",
        help="AWS region where the SageMaker endpoint is deployed",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="automl-classification-endpoint",
        help="Name of the SageMaker endpoint to call",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to a CSV file containing input rows (header + feature columns with raw categorical data)",
    )
    parser.add_argument(
        "--preprocessing-s3-uri",
        type=str,
        default=None,
        help="S3 URI to preprocessing metadata and label encoders. If not provided, will auto-detect from latest pipeline execution.",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        default="classification-model-automl-pipeline",
        help="Pipeline name to use for auto-detecting preprocessing artifacts",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML (used to get pipeline name if not specified)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("AutoML Endpoint Test Script")
    print("=" * 80)
    print(f"Endpoint: {args.endpoint_name}")
    print(f"Region: {args.region}")
    print(f"Input CSV: {args.input_csv}")
    print("-" * 80)

    # Determine preprocessing S3 URI
    preprocessing_s3_uri = args.preprocessing_s3_uri
    
    if not preprocessing_s3_uri:
        # Auto-detect from pipeline
        pipeline_name = args.pipeline_name
        
        # Try to get pipeline name from config if provided
        if args.config:
            try:
                with open(args.config, "r") as f:
                    config = yaml.safe_load(f)
                    pipeline_name = config.get("pipeline_name", pipeline_name)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        print(f"\nAuto-detecting preprocessing artifacts from pipeline: {pipeline_name}")
        try:
            preprocessing_s3_uri = find_latest_preprocessing_artifacts(pipeline_name, args.region)
        except Exception as e:
            print(f"✗ Failed to auto-detect preprocessing artifacts: {e}", file=sys.stderr)
            print("\nPlease provide --preprocessing-s3-uri explicitly")
            sys.exit(1)
    
    print(f"Preprocessing artifacts: {preprocessing_s3_uri}")

    # Load preprocessing artifacts from S3
    print("\nLoading preprocessing artifacts from S3...")
    try:
        metadata, label_encoders = load_preprocessing_artifacts(preprocessing_s3_uri, args.region)
        print(f"✓ Loaded metadata (target: {metadata.get('target_column')}, categorical columns: {len(metadata.get('categorical_columns', []))})")
        if label_encoders:
            print(f"✓ Loaded {len(label_encoders)} label encoder(s)")
        else:
            print("⚠ No label encoders found (assuming data is already encoded)")
    except Exception as e:
        print(f"✗ Failed to load preprocessing artifacts: {e}", file=sys.stderr)
        sys.exit(1)

    # Read input CSV
    print(f"\nLoading input data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
        print(f"✓ Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply preprocessing
    print("\nApplying preprocessing (missing values, label encoding)...")
    try:
        processed_df = preprocess_data(df, metadata, label_encoders)
        print(f"✓ Preprocessed data shape: {processed_df.shape}")
        print(f"  Columns: {list(processed_df.columns)}")
        print(f"  Data types: {processed_df.dtypes.to_dict()}")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create SageMaker Runtime client
    runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    print(f"\nSending requests to endpoint: {args.endpoint_name}")
    print("-" * 80)

    # Send each row as a separate inference request
    for i, (idx, row) in enumerate(processed_df.iterrows(), start=1):
        # Convert row to CSV format (no header, comma-separated values)
        # AutoML expects numeric values in the same order as training features
        payload = ",".join([str(val) for val in row.values])
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=args.endpoint_name,
                ContentType="text/csv",
                Body=payload.encode("utf-8"),
            )
            
            result = response["Body"].read().decode("utf-8")
            
            # AutoML typically returns CSV format predictions
            try:
                # Try parsing as JSON first
                parsed = json.loads(result)
            except:
                # If not JSON, it's likely CSV format
                parsed = result.strip()
            
            print(f"Row {i} (index {idx}):")
            print(f"  Raw input: {dict(df.iloc[idx])}")
            print(f"  Preprocessed: {dict(row)}")
            print(f"  Prediction: {parsed}\n")
            
        except Exception as e:
            print(f"✗ Error invoking endpoint for row {i}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()

