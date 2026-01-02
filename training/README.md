# Training Pipeline - Classification Model MLOps

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Setup and Installation](#setup-and-installation)
7. [Pipeline Steps](#pipeline-steps)
8. [Execution](#execution)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [IAM Permissions](#iam-permissions)
11. [Model Registry](#model-registry)
12. [Endpoint Usage](#endpoint-usage)
13. [Troubleshooting](#troubleshooting)
14. [Cost Optimization](#cost-optimization)
15. [Clean Up](#clean-up)
16. [Integration with Other Pipelines](#integration-with-other-pipelines)

---

## Overview

The Training Pipeline is an end-to-end MLOps pipeline for training, evaluating, and deploying a binary classification model using AWS SageMaker. This pipeline automates the complete machine learning lifecycle from data sourcing to endpoint deployment, with integrated quality gates and automatic model registration.

### Key Features

- **Automated ML Workflow**: Fully automated pipeline from data ingestion to model deployment
- **Feature Store Integration**: Sources training data from SageMaker Feature Store (customer_id 1-4000)
- **Quality Gates**: Conditional model registration based on accuracy threshold
- **Model Registry**: Automatic model versioning and approval workflow
- **Real-time Inference**: Automatic endpoint deployment with autoscaling support
- **Failure Notifications**: SNS email notifications for pipeline step failures
- **Data Preprocessing**: Automatic handling of missing values, categorical encoding, and feature engineering

### Use Case

This pipeline is designed for churn prediction classification, but can be adapted for any binary classification problem. It trains a RandomForest classifier using scikit-learn and deploys it as a real-time inference endpoint.

---

## Architecture

### Pipeline Flow

```
┌─────────────────┐
│  Data Source    │ → Retrieves data from Feature Store (customer_id 1-4000)
│  (Feature Store)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PreprocessData │ → Cleans data, handles missing values, encodes categories
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  TrainModel     │ → Trains RandomForest classifier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EvaluateModel  │ → Evaluates on test set, calculates metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ CheckModelQuality│ → Conditional: accuracy >= threshold?
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
┌─────────┐  └─→ Pipeline stops (model not registered)
│Register │
│  Model  │
└────┬────┘
     │
     ▼
┌──────────────┐
│DeployEndpoint│ → Creates/updates endpoint with autoscaling
└──────────────┘
```

### Data Flow

1. **Input**: Feature Store feature group `mlops_poc_churn_classification_feature` (customer_id 1-4000)
2. **Processing**: Data cleaning, encoding, train/test split
3. **Training**: RandomForest classifier with configurable hyperparameters
4. **Evaluation**: Metrics calculation (accuracy, precision, recall, F1, ROC-AUC)
5. **Output**: Registered model in Model Registry, deployed endpoint

---

## Prerequisites

### AWS Account Requirements

1. **AWS Account** with appropriate permissions (see [IAM Permissions](#iam-permissions))
2. **S3 Bucket** for storing data and model artifacts
3. **IAM Role** with SageMaker execution permissions
4. **SageMaker Feature Store** feature group must exist (created by `feature_store_poc`)

### Local Environment Requirements

1. **Python 3.8+** with the following packages:
   - `boto3` (AWS SDK)
   - `sagemaker` (SageMaker SDK)
   - `pyyaml` (Configuration parsing)
   - `pandas` (Data manipulation)
   - `numpy` (Numerical operations)
   - `scikit-learn` (Machine learning)

2. **AWS CLI** configured with credentials:
   ```bash
   aws configure
   ```

3. **Access to AWS Console** for monitoring pipeline executions

### Feature Store Prerequisites

- Feature group `mlops_poc_churn_classification_feature` must exist
- Feature group must have data for customer_id range 1-4000
- Feature group offline store must be accessible via Athena

---

## Project Structure

```
training_poc/
├── configs/
│   └── pipeline_config.yaml      # Pipeline configuration (YAML)
├── src/
│   ├── data_source.py            # Data sourcing from Feature Store
│   ├── preprocessing.py          # Data preprocessing and encoding
│   ├── train.py                  # Model training (RandomForest)
│   ├── evaluate.py               # Model evaluation and metrics
│   ├── _repack_model.py          # Model repacking for deployment
│   ├── inference.py              # Inference script for endpoint
│   └── deploy_endpoint.py        # Endpoint deployment with autoscaling
├── pipeline.py                   # Main pipeline definition
├── execute_pipeline.py           # Pipeline execution script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Key Files

- **`pipeline.py`**: Defines the SageMaker Pipeline with all steps, conditions, and dependencies
- **`execute_pipeline.py`**: Script to create/update and start pipeline execution
- **`configs/pipeline_config.yaml`**: Centralized configuration for all pipeline parameters
- **`src/data_source.py`**: Queries Feature Store offline store via Athena
- **`src/preprocessing.py`**: Handles missing values, categorical encoding, train/test split
- **`src/train.py`**: Trains RandomForest classifier, saves model artifacts
- **`src/evaluate.py`**: Evaluates model on test set, calculates metrics
- **`src/inference.py`**: Inference code for real-time endpoint (handles preprocessing)
- **`src/deploy_endpoint.py`**: Deploys model to endpoint, configures autoscaling

---

## Configuration

### Configuration File: `configs/pipeline_config.yaml`

Edit this file to customize the pipeline behavior:

```yaml
# Project settings
project_name: classification-model-mlops
region: us-west-2
bucket: mlbucket-sagemaker                    # Your S3 bucket
prefix: mlops/classification_model            # S3 prefix for artifacts
pipeline_name: classification-model-pipeline
model_package_group: classification-model-group

# IAM Role - must have SageMaker, S3, Feature Store, Athena, Glue permissions
role_arn: arn:aws:iam::ACCOUNT_ID:role/ML-Role-Pipelines

# Instance types
training_instance_type: ml.m5.xlarge          # For model training
processing_instance_type: ml.m5.xlarge         # For data processing steps
inference_instance_type: ml.m5.xlarge           # For endpoint (must be non-burstable for autoscaling)

# Model Registry settings
accuracy_threshold: 0.5                        # Minimum accuracy to register model
approval_status: Approved                     # Auto-approve models meeting threshold

# Deployment settings
endpoint_name: classification-model-endpoint
enable_autoscaling: true                      # Enable autoscaling for endpoint
autoscaling_min_capacity: 1                    # Minimum number of instances
autoscaling_max_capacity: 5                    # Maximum number of instances

# Failure notification settings
sns_topic_arn: arn:aws:sns:us-west-2:ACCOUNT_ID:ml-model-pipeline-failures
notification_email: your-email@example.com

# Feature Store settings (replaces Athena table)
feature_group_name: mlops_poc_churn_classification_feature
customer_id_start: 1                           # Training data: customer_id 1-4000
customer_id_end: 4000
athena_query_results: s3://mlbucket-sagemaker/athena-query-results/

# Training hyperparameters (RandomForest Classifier)
hyperparameters:
  n_estimators: 100                            # Number of trees
  max_depth: 10                                 # Maximum tree depth
  min_samples_split: 2                          # Minimum samples to split
  min_samples_leaf: 1                           # Minimum samples in leaf
```

### Configuration Parameters Explained

#### Instance Types

- **`training_instance_type`**: Instance type for model training. Larger instances train faster but cost more.
- **`processing_instance_type`**: Instance type for data processing steps (data sourcing, preprocessing, evaluation).
- **`inference_instance_type`**: Instance type for the endpoint. **Important**: Must be non-burstable (e.g., `ml.m5.xlarge`, `ml.c5.xlarge`) to enable autoscaling. Burstable instances (t2, t3, t3a, t4g) do not support autoscaling.

#### Model Registry

- **`accuracy_threshold`**: Minimum accuracy required to register the model. Models below this threshold are not registered or deployed.
- **`approval_status`**: Approval status for models that meet the threshold. Options: `Approved`, `PendingManualApproval`, `Rejected`.

#### Autoscaling

- **`enable_autoscaling`**: Whether to configure autoscaling for the endpoint.
- **`autoscaling_min_capacity`**: Minimum number of instances (must be >= 1).
- **`autoscaling_max_capacity`**: Maximum number of instances.

#### Feature Store

- **`feature_group_name`**: Name of the SageMaker Feature Store feature group.
- **`customer_id_start`**: Starting customer ID for training data (inclusive).
- **`customer_id_end`**: Ending customer ID for training data (inclusive).
- **`athena_query_results`**: S3 path for Athena query results (must be writable by the execution role).

---

## Setup and Installation

### 1. Install Dependencies

```bash
cd sagemaker/mlops_poc/training_poc
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Ensure AWS CLI is configured:

```bash
aws configure
```

Verify access:

```bash
aws sts get-caller-identity
```

### 3. Update Configuration

Edit `configs/pipeline_config.yaml`:

1. Update `role_arn` with your IAM role ARN
2. Update `bucket` with your S3 bucket name
3. Update `sns_topic_arn` and `notification_email` for failure notifications
4. Adjust instance types and hyperparameters as needed

### 4. Verify Feature Store

Ensure the feature group exists and has data:

```bash
aws sagemaker describe-feature-group \
  --feature-group-name mlops_poc_churn_classification_feature \
  --region us-west-2
```

### 5. Create/Update Pipeline

```bash
python3 pipeline.py --config configs/pipeline_config.yaml
```

This will:
- Create or update the SageMaker Pipeline definition
- Start a pipeline execution

---

## Pipeline Steps

### Step 1: DataSource

**Purpose**: Retrieves training data from SageMaker Feature Store offline store.

**Process**:
1. Describes the feature group to get Glue database and table names
2. Constructs an Athena query to select data from the offline store
3. Filters by `customer_id` range (1-4000) using `CUST_XXXXXX` format
4. Executes the query via Athena
5. Saves raw CSV data to S3

**Inputs**:
- Feature group name: `mlops_poc_churn_classification_feature`
- Customer ID range: 1-4000

**Outputs**:
- Raw CSV data: `s3://<bucket>/<prefix>/data/raw/`

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:DescribeFeatureGroup`
- `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults`
- `glue:GetDatabase`, `glue:GetTable`
- S3 read/write access

---

### Step 2: PreprocessData

**Purpose**: Cleans and prepares data for training.

**Process**:
1. Loads raw data from S3
2. **Drops identifier and metadata columns**: `customer_id`, `event_time`, `write_time`, `api_invocation_time`, `is_deleted`
3. Handles missing values:
   - Object columns → `'Unknown'`
   - Numeric columns → median value
4. Encodes categorical variables using `LabelEncoder`
5. Splits into train/test sets (80/20, stratified by target)
6. Saves:
   - Processed train/test data
   - Label encoders (for inference)
   - Metadata (feature names, target column, etc.)

**Inputs**:
- Raw CSV data from DataSource step

**Outputs**:
- Train data: `s3://<bucket>/<prefix>/data/train/`
- Test data: `s3://<bucket>/<prefix>/data/test/`
- Preprocessing artifacts: `s3://<bucket>/<prefix>/preprocessing/`

**Key Features**:
- Automatic categorical detection and encoding
- Stratified train/test split
- Metadata preservation for inference

**Failure Handling**: Sends SNS email notification on failure.

---

### Step 3: TrainModel

**Purpose**: Trains a RandomForest classifier.

**Process**:
1. Loads preprocessed training data
2. Trains RandomForest classifier with hyperparameters from config
3. Copies preprocessing artifacts (label encoders, metadata) into model directory
4. Saves model artifacts and metadata to S3

**Inputs**:
- Train data from PreprocessData step
- Preprocessing artifacts (label encoders, metadata)

**Outputs**:
- Model artifacts: `s3://<bucket>/<prefix>/model/`
- Model metadata: `s3://<bucket>/<prefix>/model/model_info.json`

**Hyperparameters** (from config):
- `n_estimators`: Number of trees (default: 100)
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Minimum samples to split (default: 2)
- `min_samples_leaf`: Minimum samples in leaf (default: 1)

**Failure Handling**: Sends SNS email notification on failure.

---

### Step 4: EvaluateModel

**Purpose**: Evaluates model performance on test set.

**Process**:
1. Loads trained model and test data
2. Generates predictions on test set
3. Calculates metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC (if applicable)
4. Saves evaluation report to S3

**Inputs**:
- Model artifacts from TrainModel step
- Test data from PreprocessData step

**Outputs**:
- Evaluation report: `s3://<bucket>/<prefix>/evaluation/evaluation.json`

**Metrics Format**:
```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.78,
  "f1_score": 0.80,
  "roc_auc": 0.88
}
```

**Failure Handling**: Sends SNS email notification on failure.

---

### Step 5: CheckModelQuality (Conditional)

**Purpose**: Checks if model accuracy meets the threshold.

**Process**:
1. Reads accuracy from evaluation report
2. Compares with `accuracy_threshold` from config
3. If `accuracy >= threshold`: Proceeds to RegisterModel and DeployEndpoint
4. If `accuracy < threshold`: Pipeline stops (model not registered)

**Inputs**:
- Evaluation report from EvaluateModel step
- Accuracy threshold from config

**Outputs**:
- Conditional branch to RegisterModel or pipeline termination

**Default Threshold**: 0.5 (configurable in `pipeline_config.yaml`)

---

### Step 6: RegisterModel (Conditional)

**Purpose**: Registers model in SageMaker Model Registry.

**Process**:
1. Creates Model Package Group if it doesn't exist
2. Registers model package with:
   - Model artifacts
   - Evaluation metrics
   - Approval status (from config)
3. Saves registration info to S3

**Inputs**:
- Model artifacts from TrainModel step
- Evaluation metrics from EvaluateModel step

**Outputs**:
- Registered model in Model Registry: `classification-model-group`
- Registration info: `s3://<bucket>/<prefix>/registration/registration.json`

**Approval Status**: Automatically set to `Approved` if accuracy meets threshold (configurable).

**Failure Handling**: Sends SNS email notification on failure.

---

### Step 7: DeployEndpoint (Conditional)

**Purpose**: Deploys model as real-time inference endpoint.

**Process**:
1. Retrieves latest approved model from Model Package Group
2. Creates or updates SageMaker endpoint
3. Waits for endpoint to be `InService`
4. Configures autoscaling (if enabled and instance type supports it):
   - Registers scalable target
   - Creates scaling policy
   - Sets min/max capacity

**Inputs**:
- Registered model from RegisterModel step

**Outputs**:
- SageMaker endpoint: `classification-model-endpoint`

**Autoscaling Notes**:
- **Non-burstable instances** (e.g., `ml.m5.xlarge`, `ml.c5.xlarge`): Autoscaling is configured
- **Burstable instances** (e.g., `ml.t2.medium`, `ml.t3.medium`): Autoscaling is skipped (not supported)

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:CreateEndpoint`, `sagemaker:UpdateEndpoint`
- `application-autoscaling:RegisterScalableTarget`
- `application-autoscaling:PutScalingPolicy`
- `iam:PassRole` for autoscaling service role

---

## Execution

### Method 1: Using `execute_pipeline.py` (Recommended)

```bash
python3 sagemaker/mlops_poc/training_poc/execute_pipeline.py \
  --config sagemaker/mlops_poc/training_poc/configs/pipeline_config.yaml
```

This script:
- Starts an execution of the existing pipeline
- Does not create/update the pipeline definition

### Method 2: Using `pipeline.py` Directly

```bash
python3 sagemaker/mlops_poc/training_poc/pipeline.py \
  --config sagemaker/mlops_poc/training_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 3: Using AWS CLI

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name classification-model-pipeline \
  --region us-west-2
```

### Method 4: Using EventBridge (Scheduled Execution)

The pipeline can be scheduled to run automatically using AWS EventBridge:

```bash
# Create EventBridge rule (runs on 1st of each month at midnight UTC)
aws events put-rule \
  --name training-poc-monthly \
  --schedule-expression "cron(0 0 1 * ? *)" \
  --state ENABLED \
  --region us-west-2

# Add pipeline as target
aws events put-targets \
  --rule training-poc-monthly \
  --targets '[
    {
      "Id": "training-poc-target",
      "Arn": "arn:aws:sagemaker:us-west-2:ACCOUNT_ID:pipeline/classification-model-pipeline",
      "RoleArn": "arn:aws:iam::ACCOUNT_ID:role/EventBridgeSageMakerStartPipelineRole",
      "Input": "{\"PipelineExecutionDisplayName\":\"scheduled-monthly-training\"}"
    }
  ]' \
  --region us-west-2
```

**Note**: The EventBridge role must have `sagemaker:StartPipelineExecution` permission.

---

## Monitoring and Logging

### View Pipeline Executions

**AWS Console**:
1. Navigate to SageMaker → Pipelines
2. Select `classification-model-pipeline`
3. View execution history and step status

**AWS CLI**:
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name classification-model-pipeline \
  --region us-west-2
```

### CloudWatch Logs

Each pipeline step generates logs in CloudWatch:

**Log Group Pattern**: `/aws/sagemaker/ProcessingJobs/<job-name>`

**View Logs**:
```bash
aws logs tail /aws/sagemaker/ProcessingJobs/<job-name> --follow --region us-west-2
```

### Failure Notifications

Every pipeline step script sends SNS email notifications on failure:

**Configuration**:
- SNS topic ARN: `sns_topic_arn` in `pipeline_config.yaml`
- Notification email: `notification_email` in `pipeline_config.yaml`

**Email Format**:
- **Subject**: `[SageMaker Pipeline Failed] <Step Name>`
- **Body**: Includes step name, error message, and CloudWatch logs reference

**IAM Requirements**:
- Execution role must have `sns:Publish` permission on the SNS topic
- SNS topic policy must allow the execution role to publish

### Check Endpoint Status

```bash
aws sagemaker describe-endpoint \
  --endpoint-name classification-model-endpoint \
  --region us-west-2
```

### Monitor Endpoint Metrics

View endpoint metrics in CloudWatch:
- Invocations
- Model latency
- 4xx/5xx errors
- CPU/GPU utilization

---

## IAM Permissions

The execution role (`ML-Role-Pipelines`) requires the following permissions:

### SageMaker Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "sagemaker:CreatePipeline",
    "sagemaker:UpdatePipeline",
    "sagemaker:StartPipelineExecution",
    "sagemaker:DescribePipelineExecution",
    "sagemaker:CreateProcessingJob",
    "sagemaker:DescribeProcessingJob",
    "sagemaker:CreateTrainingJob",
    "sagemaker:DescribeTrainingJob",
    "sagemaker:CreateModel",
    "sagemaker:CreateEndpoint",
    "sagemaker:UpdateEndpoint",
    "sagemaker:DescribeEndpoint",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:DescribeModel",
    "sagemaker:CreateModelPackage",
    "sagemaker:DescribeModelPackage",
    "sagemaker:CreateModelPackageGroup",
    "sagemaker:DescribeModelPackageGroup"
  ],
  "Resource": "*"
}
```

### Feature Store Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "sagemaker:DescribeFeatureGroup",
    "sagemaker:GetRecord",
    "sagemaker:PutRecord"
  ],
  "Resource": "arn:aws:sagemaker:*:*:feature-group/mlops_poc_churn_classification_feature"
}
```

### Athena Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "athena:StartQueryExecution",
    "athena:GetQueryExecution",
    "athena:GetQueryResults",
    "athena:StopQueryExecution"
  ],
  "Resource": "*"
}
```

### Glue Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "glue:GetDatabase",
    "glue:GetTable",
    "glue:GetPartitions"
  ],
  "Resource": "*"
}
```

### S3 Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::mlbucket-sagemaker/*",
    "arn:aws:s3:::mlbucket-sagemaker"
  ]
}
```

### SNS Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "sns:Publish"
  ],
  "Resource": "arn:aws:sns:us-west-2:ACCOUNT_ID:ml-model-pipeline-failures"
}
```

### Application Auto Scaling Permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "application-autoscaling:RegisterScalableTarget",
    "application-autoscaling:PutScalingPolicy",
    "application-autoscaling:DescribeScalingPolicies",
    "application-autoscaling:DescribeScalableTargets"
  ],
  "Resource": "*"
}
```

### IAM PassRole Permission

```json
{
  "Effect": "Allow",
  "Action": [
    "iam:PassRole"
  ],
  "Resource": "arn:aws:iam::ACCOUNT_ID:role/*",
  "Condition": {
    "StringEquals": {
      "iam:PassedToService": "application-autoscaling.amazonaws.com"
    }
  }
}
```

---

## Model Registry

### View Registered Models

**AWS Console**:
1. Navigate to SageMaker → Model Registry
2. Select `classification-model-group`
3. View model versions, metrics, and approval status

**AWS CLI**:
```bash
aws sagemaker list-model-packages \
  --model-package-group-name classification-model-group \
  --region us-west-2
```

### Model Approval Workflow

1. **Automatic Approval**: Models meeting the accuracy threshold are automatically approved
2. **Manual Approval**: Models can be manually approved/rejected in the Model Registry
3. **Endpoint Deployment**: Only approved models are deployed to the endpoint

### Model Versioning

Each pipeline execution creates a new model version if the quality threshold is met. Models are versioned automatically by SageMaker.

---

## Endpoint Usage

### Python Example

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Prepare input (CSV format with header row)
# The endpoint expects raw features (categorical values as strings)
header = "age,tenure_months,monthly_charges,total_charges,contract_type,payment_method,internet_service,phone_service,multiple_lines,online_security,tech_support,device_protection"
data_row = "35,12,70.5,840.0,Month-to-month,Electronic check,DSL,Yes,No,Yes,No"
payload = f"{header}\n{data_row}"

response = runtime.invoke_endpoint(
    EndpointName='classification-model-endpoint',
    ContentType='text/csv',
    Body=payload.encode('utf-8')
)

result = response['Body'].read().decode('utf-8')
print(result)  # Prediction result (e.g., "0" or "1")
```

### Inference Features

The endpoint automatically:
- Accepts raw categorical features (e.g., "Month-to-month", "Yes", "No")
- Performs the same preprocessing as training (missing value handling, categorical encoding)
- Drops identifier and metadata columns (`customer_id`, `event_time`, etc.) if present
- Returns predictions in CSV format (0 or 1 for binary classification)

### Input Format

**CSV with Header Row**:
```csv
age,tenure_months,monthly_charges,total_charges,contract_type,payment_method,internet_service,phone_service,multiple_lines,online_security,tech_support,device_protection
35,12,70.5,840.0,Month-to-month,Electronic check,DSL,Yes,No,Yes,No,No
```

**Note**: The header row must be included with each request.

### Output Format

**CSV**:
```
0
```

Or:
```
1
```

### Testing the Endpoint

Use the provided test script:

```bash
python3 test_classification_endpoint.py \
  --region us-west-2 \
  --endpoint-name classification-model-endpoint \
  --input-csv sample_input.csv
```

---

## Troubleshooting

### Pipeline Fails at DataSource Step

**Symptoms**: DataSource step fails with Athena or Feature Store errors.

**Solutions**:
1. Verify feature group exists:
   ```bash
   aws sagemaker describe-feature-group \
     --feature-group-name mlops_poc_churn_classification_feature \
     --region us-west-2
   ```
2. Check IAM permissions for Feature Store, Athena, and Glue
3. Verify customer_id range has data in the feature group
4. Check CloudWatch logs for detailed error messages

### Model Not Registered

**Symptoms**: Pipeline completes but model is not registered.

**Solutions**:
1. Check evaluation metrics in S3: `s3://bucket/prefix/evaluation/evaluation.json`
2. Verify accuracy threshold in config (default: 0.5)
3. Check `CheckModelQuality` step status in pipeline execution
4. Review evaluation logs for calculation errors

### Endpoint Deployment Fails

**Symptoms**: DeployEndpoint step fails or endpoint is not created.

**Solutions**:
1. Verify IAM role has endpoint creation permissions
2. Check instance type availability in your region
3. Review CloudWatch logs for deployment step
4. **Autoscaling not configured**: If using burstable instance types (t2, t3, t3a, t4g), autoscaling will be skipped. Use non-burstable instance types (ml.m5.xlarge, ml.c5.xlarge) to enable autoscaling.
5. **Endpoint update timeout**: Endpoint updates can take 5-10 minutes. The deployment script waits for the endpoint to be InService before configuring autoscaling.

### Import Errors in Processing Jobs

**Symptoms**: Processing jobs fail with `ModuleNotFoundError`.

**Solutions**:
1. Ensure all dependencies are listed in scripts
2. Check Python version compatibility (3.8+)
3. Verify SKLearnProcessor framework version matches scikit-learn version
4. **Type hints**: The codebase uses `Optional[dict]` instead of `dict | None` for Python 3.9 compatibility

### Inference Endpoint Errors

**Symptoms**: Endpoint returns errors during invocation.

**Solutions**:
1. **Missing feature columns**: Ensure input CSV includes a header row with column names
2. **Categorical encoding errors**: The endpoint handles unseen categories by mapping to the first known class
3. **ID columns missing**: ID columns (like `customer_id`) are automatically dropped if present
4. **Metadata columns**: Metadata columns (`event_time`, `write_time`, etc.) are automatically dropped

### Feature Store Query Errors

**Symptoms**: DataSource step fails when querying Feature Store.

**Solutions**:
1. Verify feature group offline store is accessible
2. Check Glue database and table names match
3. Ensure Athena query results location is writable
4. Verify customer_id filtering format matches `CUST_XXXXXX` pattern

---

## Cost Optimization

### Instance Type Selection

- **Testing**: Use smaller instance types (`ml.t3.medium` for testing)
- **Production**: Use appropriate instance types (`ml.m5.xlarge` for production)
- **Autoscaling**: Configure min/max capacity based on expected traffic

### Storage Optimization

- Clean up old pipeline executions and model versions
- Delete unused model artifacts from S3
- Monitor S3 storage costs

### Endpoint Optimization

- **Autoscaling**: Configure min/max capacity to scale based on traffic
- **Single Instance**: Set `enable_autoscaling: false` to use single instance (lower cost, no scaling)
- **Burstable Instances**: Use `ml.t2.medium` or `ml.t3.medium` for lower cost (autoscaling not supported)
- **Non-burstable Instances**: Use `ml.m5.xlarge` or `ml.c5.xlarge` for autoscaling support

### Pipeline Execution

- Schedule pipelines during off-peak hours
- Monitor pipeline execution costs
- Clean up failed executions

---

## Clean Up

### Delete Pipeline

```bash
aws sagemaker delete-pipeline \
  --pipeline-name classification-model-pipeline \
  --region us-west-2
```

### Delete Endpoint

```bash
aws sagemaker delete-endpoint \
  --endpoint-name classification-model-endpoint \
  --region us-west-2

aws sagemaker delete-endpoint-config \
  --endpoint-config-name <CONFIG_NAME> \
  --region us-west-2
```

### Delete Model Registry Group

```bash
aws sagemaker delete-model-package-group \
  --model-package-group-name classification-model-group \
  --region us-west-2
```

**Note**: You must delete all model packages in the group before deleting the group.

### Delete S3 Artifacts

```bash
aws s3 rm s3://mlbucket-sagemaker/mlops/classification_model/ --recursive
```

### Delete EventBridge Rule (if scheduled)

```bash
aws events remove-targets \
  --rule training-poc-monthly \
  --ids training-poc-target \
  --region us-west-2

aws events delete-rule \
  --name training-poc-monthly \
  --region us-west-2
```

---

## Integration with Other Pipelines

### Feature Store Integration

This pipeline sources data from the Feature Store feature group created by `feature_store_poc`:
- **Feature Group**: `mlops_poc_churn_classification_feature`
- **Customer ID Range**: 1-4000 (training data)

### Model Registry Integration

The trained model is registered in:
- **Model Package Group**: `classification-model-group`
- **Used By**: `batch_inference_poc` (for batch predictions)

### Endpoint Integration

The deployed endpoint:
- **Endpoint Name**: `classification-model-endpoint`
- **Model Source**: Latest approved model from `classification-model-group`
- **Usage**: Real-time inference for churn prediction

### Data Flow Across Pipelines

```
feature_store_poc
    ↓ (ingests data)
Feature Store (mlops_poc_churn_classification_feature)
    ↓ (customer_id 1-4000)
training_poc
    ↓ (trains model)
Model Registry (classification-model-group)
    ↓ (approved model)
batch_inference_poc (uses model for batch predictions)
    ↓ (customer_id 4000-5000)
Feature Store (same feature group, different customer_id range)
```

---

## Additional Resources

- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [Application Auto Scaling for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [Feature Store Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
- [Athena Documentation](https://docs.aws.amazon.com/athena/latest/ug/what-is.html)
- [SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)

---

## License

[Your License Here]
