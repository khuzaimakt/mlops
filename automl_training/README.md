# AutoML Training Pipeline - Classification Model MLOps

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
14. [Cost Considerations](#cost-considerations)
15. [Differences from Training Pipeline](#differences-from-training-pipeline)
16. [Clean Up](#clean-up)
17. [Integration with Other Pipelines](#integration-with-other-pipelines)

---

## Overview

The AutoML Training Pipeline is an end-to-end MLOps pipeline that uses SageMaker AutoML to automatically train and select the best model for binary classification. This pipeline mirrors the `training_poc` flow but replaces manual training with AutoML, which automatically explores multiple algorithms and hyperparameters to find the optimal model.

### Key Features

- **Automated Model Selection**: SageMaker AutoML automatically explores multiple algorithms and hyperparameters
- **Feature Store Integration**: Sources training data from SageMaker Feature Store (customer_id 1-4000)
- **Quality Gates**: Conditional model registration based on accuracy threshold
- **Model Registry**: Automatic model versioning and approval workflow
- **Real-time Inference**: Automatic endpoint deployment with autoscaling support
- **Failure Notifications**: SNS email notifications for pipeline step failures
- **Data Preprocessing**: Automatic handling of missing values, categorical encoding, and feature engineering
- **Unique Model Naming**: Timestamp-based model names prevent conflicts on re-runs

### Use Case

This pipeline is designed for churn prediction classification using AutoML. It automatically selects the best algorithm and hyperparameters, then deploys the model as a real-time inference endpoint.

### Advantages of AutoML

- **No Manual Tuning**: AutoML automatically explores algorithms and hyperparameters
- **Best Model Selection**: AutoML selects the best candidate based on objective metric
- **Multiple Algorithms**: AutoML explores XGBoost, Linear Learner, Deep Learning, and more
- **Ensemble Models**: AutoML can create ensemble models for better performance

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
│ AutoMLTraining  │ → Runs AutoML, explores algorithms, selects best candidate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CreateModel    │ → Creates SageMaker model from AutoML best candidate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EvaluateModel  │ → Runs batch transform, calculates metrics
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
3. **AutoML Training**: Automatic algorithm and hyperparameter exploration
4. **Model Creation**: Creates SageMaker model from AutoML best candidate
5. **Evaluation**: Batch transform on test set, metrics calculation
6. **Output**: Registered model in Model Registry, deployed endpoint

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
automl_training_poc/
├── configs/
│   └── pipeline_config.yaml          # Pipeline configuration (YAML)
├── src/
│   ├── data_source.py                # Data sourcing from Feature Store
│   ├── preprocessing.py              # Data preprocessing and encoding
│   ├── create_model.py              # Create SageMaker model from AutoML best candidate
│   ├── evaluate_automl.py           # Evaluate model using batch transform
│   ├── register_model.py            # Register model in Model Registry
│   └── deploy_endpoint.py            # Deploy endpoint with autoscaling
├── pipeline.py                       # Main pipeline definition
├── execute_pipeline.py               # Pipeline execution script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

### Key Files

- **`pipeline.py`**: Defines the SageMaker Pipeline with all steps, conditions, and dependencies
- **`execute_pipeline.py`**: Script to create/update and start pipeline execution
- **`configs/pipeline_config.yaml`**: Centralized configuration for all pipeline parameters
- **`src/data_source.py`**: Queries Feature Store offline store via Athena
- **`src/preprocessing.py`**: Handles missing values, categorical encoding, train/test split
- **`src/create_model.py`**: Creates SageMaker model from AutoML best candidate with unique timestamp-based name
- **`src/evaluate_automl.py`**: Evaluates model using batch transform, calculates metrics
- **`src/register_model.py`**: Registers model in Model Registry, creates group if needed
- **`src/deploy_endpoint.py`**: Deploys model to endpoint, configures autoscaling

---

## Configuration

### Configuration File: `configs/pipeline_config.yaml`

Edit this file to customize the pipeline behavior:

```yaml
# Project settings
project_name: classification-model-automl
region: us-west-2
bucket: mlbucket-sagemaker
prefix: mlops/automl_training
pipeline_name: classification-model-automl-pipeline
model_package_group: classification-model-automl-group

# IAM Role - should have SageMaker, S3, Feature Store, Athena, Glue permissions
role_arn: arn:aws:iam::ACCOUNT_ID:role/ML-Role-Pipelines

# Instance types
processing_instance_type: ml.m5.xlarge

# Inference/endpoint settings
endpoint_name: automl-classification-endpoint
inference_instance_type: ml.m5.xlarge  # Must be non-burstable for autoscaling
enable_autoscaling: true
autoscaling_min_capacity: 1
autoscaling_max_capacity: 5

# AutoML settings
target_attribute_name: churn
problem_type: BinaryClassification
objective_metric_name: Accuracy
max_runtime_sec: 3600  # Maximum runtime per training job (seconds)
accuracy_threshold: 0.5  # Minimum accuracy to register and deploy model

# Feature Store settings (replaces Athena table)
feature_group_name: mlops_poc_churn_classification_feature
customer_id_start: 1  # Training data: customer_id 1-4000
customer_id_end: 4000
athena_query_results: s3://mlbucket-sagemaker/athena-query-results/

# Failure notification settings
sns_topic_arn: arn:aws:sns:us-west-2:ACCOUNT_ID:ml-model-pipeline-failures
notification_email: your-email@example.com
```

### Configuration Parameters Explained

#### AutoML Settings

- **`target_attribute_name`**: Name of the target column (e.g., "churn")
- **`problem_type`**: Type of ML problem (`BinaryClassification`, `MulticlassClassification`, `Regression`)
- **`objective_metric_name`**: Metric to optimize (`Accuracy`, `F1`, `Precision`, `Recall`, `AUC`, `RMSE`, `MAE`)
- **`max_runtime_sec`**: Maximum runtime per training job in seconds (default: 3600 = 1 hour)

#### Instance Types

- **`processing_instance_type`**: Instance type for data processing steps
- **`inference_instance_type`**: Instance type for the endpoint. **Important**: Must be non-burstable (e.g., `ml.m5.xlarge`, `ml.c5.xlarge`) to enable autoscaling. Burstable instances (t2, t3, t3a, t4g) do not support autoscaling.

#### Model Registry

- **`accuracy_threshold`**: Minimum accuracy required to register the model
- **`model_package_group`**: Name of the Model Package Group (created automatically if it doesn't exist)

#### Feature Store

- **`feature_group_name`**: Name of the SageMaker Feature Store feature group
- **`customer_id_start`**: Starting customer ID for training data (inclusive)
- **`customer_id_end`**: Ending customer ID for training data (inclusive)
- **`athena_query_results`**: S3 path for Athena query results

---

## Setup and Installation

### 1. Install Dependencies

```bash
cd sagemaker/mlops_poc/automl_training_poc
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
4. Adjust instance types and AutoML settings as needed

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

**Purpose**: Cleans and prepares data for AutoML training.

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
- Preprocessing artifacts saved separately to prevent AutoML from reading them

**Failure Handling**: Sends SNS email notification on failure.

---

### Step 3: AutoMLTraining

**Purpose**: Runs SageMaker AutoML to automatically explore algorithms and select the best model.

**Process**:
1. Configures AutoML job with:
   - Problem type: Binary Classification
   - Objective metric: Accuracy
   - Maximum runtime per training job
   - Target attribute name
2. Runs AutoML in ENSEMBLING mode (required for AutoMLStep)
3. AutoML automatically:
   - Explores multiple algorithms (XGBoost, Linear Learner, Deep Learning, etc.)
   - Tunes hyperparameters for each algorithm
   - Trains multiple candidate models
   - Selects the best candidate based on objective metric
4. Saves AutoML job artifacts to S3

**Inputs**:
- Train data from PreprocessData step

**Outputs**:
- AutoML job artifacts: `s3://<bucket>/<prefix>/automl/`
- Best candidate model information

**Key Features**:
- **Automatic Algorithm Selection**: AutoML explores multiple algorithms
- **Hyperparameter Tuning**: AutoML automatically tunes hyperparameters
- **Ensemble Models**: AutoML can create ensemble models
- **Best Candidate Selection**: AutoML selects the best candidate based on objective metric

**Note**: AutoML may launch multiple parallel training jobs. This is expected behavior.

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:CreateAutoMLJob`
- `sagemaker:DescribeAutoMLJob`
- `sagemaker:ListCandidatesForAutoMLJob`
- S3 read/write access

---

### Step 4: CreateModel

**Purpose**: Creates a SageMaker model from the AutoML best candidate.

**Process**:
1. Retrieves the best candidate from the AutoML job
2. Generates a unique model name using timestamp (e.g., `classification-model-automl-1734619820`)
3. Creates a SageMaker model from the AutoML best candidate
4. Saves model information to `model_info.json` for downstream steps

**Inputs**:
- AutoML job artifacts from AutoMLTraining step

**Outputs**:
- Model information: `s3://<bucket>/<prefix>/model/model_info.json`
- Model artifacts: Stored by SageMaker

**Key Features**:
- **Unique Model Names**: Timestamp-based naming prevents conflicts on re-runs
- **Model Information**: Saves model name and ARN for downstream steps

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:CreateModel`
- `sagemaker:DescribeModel`
- `sagemaker:DescribeAutoMLJob`
- `sagemaker:ListCandidatesForAutoMLJob`

---

### Step 5: EvaluateModel

**Purpose**: Evaluates model performance using batch transform.

**Process**:
1. Loads the model name from `model_info.json`
2. Creates a batch transform job
3. Runs batch transform on the test dataset
4. Calculates metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
5. Saves evaluation report to S3

**Inputs**:
- Model information from CreateModel step
- Test data from PreprocessData step

**Outputs**:
- Evaluation report: `s3://<bucket>/<prefix>/evaluation/evaluation.json`
- Batch transform predictions: `s3://<bucket>/<prefix>/evaluation/predictions/`

**Metrics Format**:
```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.78,
  "f1_score": 0.80
}
```

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:CreateTransformJob`
- `sagemaker:DescribeTransformJob`
- `sagemaker:DescribeModel`
- S3 read/write access

---

### Step 6: CheckModelQuality (Conditional)

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

### Step 7: RegisterModel (Conditional)

**Purpose**: Registers model in SageMaker Model Registry.

**Process**:
1. Creates Model Package Group if it doesn't exist
2. Registers model package with:
   - Model artifacts
   - Evaluation metrics
   - Approval status (from config)
3. Saves registration info to S3

**Inputs**:
- Model information from CreateModel step
- Evaluation metrics from EvaluateModel step

**Outputs**:
- Registered model in Model Registry: `classification-model-automl-group`
- Registration info: `s3://<bucket>/<prefix>/registration/registration.json`

**Approval Status**: Automatically set to `Approved` if accuracy meets threshold (configurable).

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:CreateModelPackage`
- `sagemaker:CreateModelPackageGroup`
- `sagemaker:DescribeModelPackage`
- `sagemaker:DescribeModel`

---

### Step 8: DeployEndpoint (Conditional)

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
- SageMaker endpoint: `automl-classification-endpoint`

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
python3 sagemaker/mlops_poc/automl_training_poc/execute_pipeline.py \
  --config sagemaker/mlops_poc/automl_training_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 2: Using `pipeline.py` Directly

```bash
python3 sagemaker/mlops_poc/automl_training_poc/pipeline.py \
  --config sagemaker/mlops_poc/automl_training_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 3: Using AWS CLI

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name classification-model-automl-pipeline \
  --region us-west-2
```

### Method 4: Using EventBridge (Scheduled Execution)

The pipeline can be scheduled to run automatically using AWS EventBridge:

```bash
# Create EventBridge rule (runs on 1st of each month at midnight UTC)
aws events put-rule \
  --name automl-training-poc-monthly \
  --schedule-expression "cron(0 0 1 * ? *)" \
  --state ENABLED \
  --region us-west-2

# Add pipeline as target
aws events put-targets \
  --rule automl-training-poc-monthly \
  --targets '[
    {
      "Id": "automl-training-poc-target",
      "Arn": "arn:aws:sagemaker:us-west-2:ACCOUNT_ID:pipeline/classification-model-automl-pipeline",
      "RoleArn": "arn:aws:iam::ACCOUNT_ID:role/EventBridgeSageMakerStartPipelineRole",
      "Input": "{\"PipelineExecutionDisplayName\":\"scheduled-monthly-automl-training\"}"
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
2. Select `classification-model-automl-pipeline`
3. View execution history and step status

**AWS CLI**:
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name classification-model-automl-pipeline \
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
  --endpoint-name automl-classification-endpoint \
  --region us-west-2
```

### Monitor AutoML Job

```bash
aws sagemaker describe-auto-ml-job \
  --auto-ml-job-name <job-name> \
  --region us-west-2
```

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
    "sagemaker:CreateAutoMLJob",
    "sagemaker:DescribeAutoMLJob",
    "sagemaker:ListCandidatesForAutoMLJob",
    "sagemaker:CreateModel",
    "sagemaker:DescribeModel",
    "sagemaker:CreateTransformJob",
    "sagemaker:DescribeTransformJob",
    "sagemaker:CreateEndpoint",
    "sagemaker:UpdateEndpoint",
    "sagemaker:DescribeEndpoint",
    "sagemaker:CreateEndpointConfig",
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
2. Select `classification-model-automl-group`
3. View model versions, metrics, and approval status

**AWS CLI**:
```bash
aws sagemaker list-model-packages \
  --model-package-group-name classification-model-automl-group \
  --region us-west-2
```

### Model Approval Workflow

1. **Automatic Approval**: Models meeting the accuracy threshold are automatically approved
2. **Manual Approval**: Models can be manually approved/rejected in the Model Registry
3. **Endpoint Deployment**: Only approved models are deployed to the endpoint

### Model Versioning

Each pipeline execution creates a new model version if the quality threshold is met. Models are versioned automatically by SageMaker with unique timestamp-based names.

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
    EndpointName='automl-classification-endpoint',
    ContentType='text/csv',
    Body=payload.encode('utf-8')
)

result = response['Body'].read().decode('utf-8')
print(result)  # Prediction result
```

### Testing the Endpoint

Use the provided test script:

```bash
python3 test_automl_endpoint.py \
  --region us-west-2 \
  --endpoint-name automl-classification-endpoint \
  --input-csv sample_input.csv
```

The test script:
- Auto-detects preprocessing artifacts from the latest pipeline execution
- Applies preprocessing (missing values, label encoding) to your raw CSV
- Sends preprocessed data to the endpoint
- Displays predictions

### Inference Features

The endpoint automatically:
- Accepts raw categorical features (e.g., "Month-to-month", "Yes", "No")
- Performs the same preprocessing as training (missing value handling, categorical encoding)
- Drops identifier and metadata columns (`customer_id`, `event_time`, etc.) if present
- Returns predictions in CSV format

---

## Troubleshooting

### AutoML Launches Multiple Jobs

**Symptoms**: AutoML appears to launch multiple training jobs.

**Explanation**: This is expected behavior. AutoML explores multiple algorithms and hyperparameters in parallel to find the best model.

**Solutions**: No action needed. AutoML will select the best candidate automatically.

### Model Already Exists Error

**Symptoms**: CreateModel step fails with "model already exists" error.

**Solutions**:
1. The pipeline uses timestamp-based model names to prevent conflicts. If you see this error, it may indicate a clock synchronization issue.
2. Check the model name in `model_info.json` to verify it's unique
3. Review CloudWatch logs for detailed error messages

### Autoscaling Not Working

**Symptoms**: Endpoint is deployed but autoscaling is not configured.

**Solutions**:
1. Ensure instance type is non-burstable (e.g., `ml.m5.xlarge`, not `ml.t2.medium`)
2. Check that endpoint status is `InService` before autoscaling configuration
3. Verify IAM permissions for Application Auto Scaling
4. Review deployment logs for autoscaling errors

### Preprocessing Artifacts Not Found

**Symptoms**: Evaluation or inference fails due to missing preprocessing artifacts.

**Solutions**:
1. The preprocessing step saves metadata and label encoders to a separate S3 path (`meta/`) to prevent AutoML from reading them as training data
2. Ensure this path is accessible
3. Verify preprocessing step completed successfully
4. Check S3 permissions

### Batch Transform Fails

**Symptoms**: EvaluateModel step fails during batch transform.

**Solutions**:
1. Verify model was created successfully in CreateModel step
2. Check test data format matches training data format
3. Review batch transform logs in CloudWatch
4. Ensure IAM permissions for batch transform

### Feature Store Query Errors

**Symptoms**: DataSource step fails when querying Feature Store.

**Solutions**:
1. Verify feature group offline store is accessible
2. Check Glue database and table names match
3. Ensure Athena query results location is writable
4. Verify customer_id filtering format matches `CUST_XXXXXX` pattern

---

## Cost Considerations

### AutoML Costs

- **AutoML**: Can be expensive due to multiple parallel training jobs
- **Training Jobs**: Each candidate model training job incurs costs
- **Instance Types**: Use appropriate instance types for your workload
- **Runtime**: Set `max_runtime_sec` to limit AutoML exploration time

### Instance Type Selection

- **Processing**: Use `ml.m5.xlarge` for data processing steps
- **Inference**: Use `ml.m5.xlarge` or `ml.c5.xlarge` for endpoints (non-burstable for autoscaling)
- **Testing**: Use smaller instance types for testing (`ml.t3.medium`)

### Storage Costs

- Monitor S3 storage for model artifacts and data
- Clean up old pipeline executions and model versions
- Delete unused model artifacts from S3

### Endpoint Costs

- **Autoscaling**: Configure min/max capacity based on expected traffic
- **Single Instance**: Use single instance for lower cost (no autoscaling)
- **Burstable Instances**: Use `ml.t2.medium` or `ml.t3.medium` for lower cost (autoscaling not supported)

### Pipeline Execution

- Schedule pipelines during off-peak hours
- Monitor pipeline execution costs
- Clean up failed executions

---

## Differences from Training Pipeline

### 1. Training Method

- **Training Pipeline**: Uses manual scikit-learn RandomForest training
- **AutoML Pipeline**: Uses SageMaker AutoML to automatically explore algorithms and hyperparameters

### 2. Model Creation

- **Training Pipeline**: Model is created directly during training
- **AutoML Pipeline**: Requires separate `CreateModel` step to create model from AutoML best candidate

### 3. Evaluation

- **Training Pipeline**: Direct model evaluation on test set
- **AutoML Pipeline**: Uses batch transform for evaluation

### 4. Inference Script

- **Training Pipeline**: Custom inference script (`inference.py`) handles preprocessing
- **AutoML Pipeline**: AutoML models handle preprocessing internally; test script preprocesses data locally for consistency

### 5. Model Naming

- **Training Pipeline**: Uses fixed model names (may cause conflicts on re-runs)
- **AutoML Pipeline**: Uses timestamp-based unique names to avoid conflicts

### 6. Model Package Group

- **Training Pipeline**: `classification-model-group`
- **AutoML Pipeline**: `classification-model-automl-group`

### 7. Endpoint Name

- **Training Pipeline**: `classification-model-endpoint`
- **AutoML Pipeline**: `automl-classification-endpoint`

---

## Clean Up

### Delete Pipeline

```bash
aws sagemaker delete-pipeline \
  --pipeline-name classification-model-automl-pipeline \
  --region us-west-2
```

### Delete Endpoint

```bash
aws sagemaker delete-endpoint \
  --endpoint-name automl-classification-endpoint \
  --region us-west-2

aws sagemaker delete-endpoint-config \
  --endpoint-config-name <CONFIG_NAME> \
  --region us-west-2
```

### Delete Model Registry Group

```bash
aws sagemaker delete-model-package-group \
  --model-package-group-name classification-model-automl-group \
  --region us-west-2
```

**Note**: You must delete all model packages in the group before deleting the group.

### Delete S3 Artifacts

```bash
aws s3 rm s3://mlbucket-sagemaker/mlops/automl_training/ --recursive
```

### Delete EventBridge Rule (if scheduled)

```bash
aws events remove-targets \
  --rule automl-training-poc-monthly \
  --ids automl-training-poc-target \
  --region us-west-2

aws events delete-rule \
  --name automl-training-poc-monthly \
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
- **Model Package Group**: `classification-model-automl-group`
- **Separate from Training Pipeline**: Uses a different model package group than `training_poc`

### Endpoint Integration

The deployed endpoint:
- **Endpoint Name**: `automl-classification-endpoint`
- **Model Source**: Latest approved model from `classification-model-automl-group`
- **Usage**: Real-time inference for churn prediction using AutoML-selected model

### Data Flow Across Pipelines

```
feature_store_poc
    ↓ (ingests data)
Feature Store (mlops_poc_churn_classification_feature)
    ↓ (customer_id 1-4000)
automl_training_poc
    ↓ (trains model with AutoML)
Model Registry (classification-model-automl-group)
    ↓ (approved model)
Endpoint (automl-classification-endpoint)
    ↓ (real-time inference)
Applications
```

---

## Additional Resources

- [SageMaker AutoML Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-machine-learning.html)
- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [Application Auto Scaling for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [Feature Store Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
- [Athena Documentation](https://docs.aws.amazon.com/athena/latest/ug/what-is.html)

---

## License

[Your License Here]
