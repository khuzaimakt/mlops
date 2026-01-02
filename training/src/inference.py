#!/usr/bin/env python
"""Inference script for SKLearn SageMaker endpoint."""
import os
import json
import joblib
import pandas as pd
import numpy as np
from io import StringIO
from typing import Optional


def model_fn(model_dir):
    """Load the model and preprocessing artifacts from the model_dir directory."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)

    # Load model metadata (feature columns, target, etc.)
    model_metadata = None
    model_meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(model_meta_path):
        with open(model_meta_path, "r") as f:
            model_metadata = json.load(f)

    # Load preprocessing metadata and label encoders (if available)
    prep_metadata = None
    prep_meta_path = os.path.join(model_dir, "preprocessing_metadata.json")
    if os.path.exists(prep_meta_path):
        with open(prep_meta_path, "r") as f:
            prep_metadata = json.load(f)

    label_encoders = None
    enc_path = os.path.join(model_dir, "label_encoders.pkl")
    if os.path.exists(enc_path):
        label_encoders = joblib.load(enc_path)

    return {
        "model": model,
        "model_metadata": model_metadata,
        "prep_metadata": prep_metadata,
        "label_encoders": label_encoders,
    }


def input_fn(request_body, request_content_type):
    """Deserialize and prepare the prediction input.

    For text/csv we expect a header row with column names matching the raw data schema.
    """
    if request_content_type == "text/csv":
        # Assume first line is header; create DataFrame with column names
        data = pd.read_csv(StringIO(request_body))
        return data
    elif request_content_type == "application/json":
        # Read JSON data
        data = json.loads(request_body)
        # Handle different JSON formats
        if isinstance(data, dict):
            if "instances" in data:
                return np.array(data["instances"])
            elif "data" in data:
                return np.array(data["data"])
            else:
                # Assume the dict contains feature values
                return np.array([list(data.values())])
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def _apply_preprocessing(df: pd.DataFrame, prep_metadata: Optional[dict], label_encoders: Optional[dict]) -> pd.DataFrame:
    """Match the preprocessing logic from preprocessing.py on incoming data.
    
    This function replicates the exact preprocessing steps:
    1. Handle missing values (object -> 'Unknown', numeric -> median)
    2. Encode categorical variables using stored LabelEncoders
    3. Drop target column if present
    """
    if prep_metadata is None:
        # No metadata; assume data is already numeric
        return df

    target_col = prep_metadata.get("target_column")
    categorical_cols = prep_metadata.get("categorical_columns", [])

    processed_df = df.copy()

    # Exclude identifier and metadata columns (align with training-time preprocessing)
    meta_columns = [
        "event_time",
        "write_time",
        "api_invocation_time",
        "is_deleted",
    ]
    id_columns = ['customer_id', 'id', 'ID', 'CustomerID', 'Customer_ID']
    for col in id_columns + meta_columns:
        if col in processed_df.columns and col != target_col:
            processed_df = processed_df.drop(columns=[col])

    # Handle missing values - exactly as in preprocessing.py
    for col in processed_df.columns:
        if processed_df[col].dtype == 'object':
            processed_df[col] = processed_df[col].fillna('Unknown')
        else:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())

    # Encode categorical variables using stored label encoders
    # This matches preprocessing.py: le.fit_transform(processed_df[col].astype(str))
    if label_encoders and categorical_cols:
        for col in categorical_cols:
            if col in processed_df.columns and col in label_encoders:
                le = label_encoders[col]
                # Convert to string and encode, matching preprocessing.py behavior
                # Handle unseen categories: use the first class as fallback
                def encode_val(v):
                    v_str = str(v)
                    if v_str in le.classes_:
                        return int(le.transform([v_str])[0])
                    # Unseen category: use first class as fallback
                    # This is safer than failing, but ideally unseen values should be rare
                    return int(le.transform([le.classes_[0]])[0])
                
                processed_df[col] = processed_df[col].astype(str).map(encode_val)

    # Drop target column if present (as it's not a feature)
    if target_col and target_col in processed_df.columns:
        processed_df = processed_df.drop(columns=[target_col])

    return processed_df


def predict_fn(input_data, model_bundle):
    """Perform prediction on the deserialized input, applying preprocessing if available."""
    model = model_bundle["model"]
    model_metadata = model_bundle.get("model_metadata") or {}
    prep_metadata = model_bundle.get("prep_metadata")
    label_encoders = model_bundle.get("label_encoders")

    # Columns that are useful metadata but can be safely defaulted when absent in requests
    optional_metadata_defaults = {
        "event_time": lambda: pd.Timestamp.utcnow().isoformat(),
        "write_time": lambda: pd.Timestamp.utcnow().isoformat(),
        "api_invocation_time": lambda: pd.Timestamp.utcnow().isoformat(),
        "is_deleted": lambda: 0,
    }

    # If we get a DataFrame and have preprocessing metadata, treat input as raw features
    if isinstance(input_data, pd.DataFrame):
        df = _apply_preprocessing(input_data, prep_metadata, label_encoders)

        # Reorder/limit to expected feature columns if available
        feature_cols = model_metadata.get("feature_columns")
        if feature_cols:
            # Add optional metadata columns with reasonable defaults if missing
            for col in feature_cols:
                if col not in df.columns and col in optional_metadata_defaults:
                    df[col] = optional_metadata_defaults[col]()

            # Re-check for any still-missing required columns (i.e., not optional)
            missing = [col for col in feature_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected feature columns in input: {set(missing)}")

            # Now reorder/limit to the expected feature order
            df = df[feature_cols]

        X = df.values
    else:
        # Fallback: assume already-preprocessed numeric array
        X = np.asarray(input_data)

    predictions = model.predict(X)
    return predictions


def output_fn(prediction, content_type):
    """Serialize the prediction result."""
    if content_type == "application/json":
        return json.dumps({"predictions": prediction.tolist()})
    elif content_type == "text/csv":
        return ",".join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

