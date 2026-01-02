# Feature Store POC – Churn Classification

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Setup and Installation](#setup-and-installation)
7. [Data Generation](#data-generation)
8. [Pipeline Steps](#pipeline-steps)
9. [Execution](#execution)
10. [Monitoring and Logging](#monitoring-and-logging)
11. [IAM Permissions](#iam-permissions)
12. [Feature Store Details](#feature-store-details)
13. [Troubleshooting](#troubleshooting)
14. [Cost Considerations](#cost-considerations)
15. [Clean Up](#clean-up)
16. [Integration with Other Pipelines](#integration-with-other-pipelines)

---

## Overview

The Feature Store POC demonstrates how to move from Athena-based data sourcing to SageMaker Feature Store for the churn classification use case. This pipeline ingests cleaned features into SageMaker Feature Store, which then serves as the centralized data source for downstream ML pipelines.

### Key Features

- **Centralized Feature Storage**: Single source of truth for features used across multiple pipelines
- **Data Cleaning**: Automatically cleans messy data (missing values, whitespace, mixed casing)
- **Offline Store**: Stores features in S3 for batch processing and training
- **Online Store** (optional): Can be enabled for real-time feature lookups
- **Versioning**: Feature Store automatically versions features with timestamps
- **Failure Notifications**: SNS email notifications for pipeline failures
- **Pipeline Integration**: Seamlessly integrates with training, AutoML, and batch inference pipelines

### Use Case

This POC creates a feature group that replaces Athena tables currently used in:
- `mlops_poc/training_poc` (customer_id 1-4000 for training)
- `mlops_poc/batch_inference_poc` (customer_id 4000-5000 for inference)
- `mlops_poc/automl_training_poc` (customer_id 1-4000 for training)

### Benefits of Feature Store

1. **Centralized Management**: Single source of truth for features
2. **Data Quality**: Automatic validation and cleaning
3. **Versioning**: Automatic feature versioning with timestamps
4. **Consistency**: Ensures same features are used across training and inference
5. **Scalability**: Handles large-scale feature storage and retrieval
6. **Integration**: Seamless integration with SageMaker pipelines

---

## Architecture

### Pipeline Flow

```
┌─────────────────────────┐
│ Generate Churn Data     │ → Creates synthetic messy churn data
│ (Athena Table)          │   Stores in Athena table: churn_prediction_data
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ IngestToFeatureStore    │ → Reads from Athena table
│ (Processing Job)        │   Cleans data (whitespace, missing values, etc.)
└───────────┬─────────────┘   Creates feature group if needed
            │                 Ingests records into Feature Store
            ▼
┌─────────────────────────┐
│ Feature Store           │ → Offline store (S3) for batch processing
│ (mlops_poc_churn_       │   Online store (optional) for real-time lookups
│  classification_feature) │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Downstream Pipelines    │ → training_poc (customer_id 1-4000)
│                          │   automl_training_poc (customer_id 1-4000)
│                          │   batch_inference_poc (customer_id 4000-5000)
└─────────────────────────┘
```

### Data Flow

1. **Input**: Messy churn data in Athena table `ml_metrics.churn_prediction_data`
2. **Processing**: Data cleaning, normalization, feature validation
3. **Output**: Cleaned features in Feature Store feature group `mlops_poc_churn_classification_feature`
4. **Usage**: Downstream pipelines query Feature Store offline store via Athena

---

## Prerequisites

### AWS Account Requirements

1. **AWS Account** with appropriate permissions (see [IAM Permissions](#iam-permissions))
2. **S3 Bucket** for storing Feature Store offline store and Athena query results
3. **IAM Role** with SageMaker execution permissions
4. **Athena Database** (`ml_metrics`) must exist
5. **Glue Catalog** for Feature Store offline store

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

### Athena Prerequisites

- Athena database `ml_metrics` must exist
- S3 location for Athena query results must be writable
- Glue catalog must be accessible

---

## Project Structure

```
feature_store_poc/
├── configs/
│   └── pipeline_config.yaml        # Pipeline configuration (YAML)
├── src/
│   └── ingest_to_feature_store.py  # Processing/ingestion script
├── pipeline.py                     # SageMaker Pipeline definition
├── execute_pipeline.py            # Pipeline execution script
├── generate_churn_to_athena.py     # Messy churn data generator → Athena
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Key Files

- **`pipeline.py`**: Defines the SageMaker Pipeline with ingestion step
- **`execute_pipeline.py`**: Script to start pipeline execution
- **`configs/pipeline_config.yaml`**: Centralized configuration for all pipeline parameters
- **`src/ingest_to_feature_store.py`**: Main ingestion script that cleans data and ingests into Feature Store
- **`generate_churn_to_athena.py`**: Utility script to generate synthetic messy churn data (optional, for testing)

---

## Configuration

### Configuration File: `configs/pipeline_config.yaml`

Edit this file to customize the pipeline behavior:

```yaml
# Project settings
project_name: feature-store-churn-poc
region: us-west-2

# S3 bucket/prefix used for Athena and Feature Store offline store
bucket: mlbucket-sagemaker
prefix: mlops/feature_store_poc

# IAM Role - should have SageMaker, S3, Athena, Glue, and Feature Store permissions
role_arn: arn:aws:iam::ACCOUNT_ID:role/ML-Role-Pipelines

# Athena settings for raw "messy" churn data
athena_database: ml_metrics
athena_table: churn_prediction_data
athena_query_results: s3://mlbucket-sagemaker/athena-query-results/

# SageMaker Feature Store
feature_group_name: mlops_poc_churn_classification_feature
record_identifier_name: customer_id
event_time_feature_name: event_time
offline_store_s3_uri: s3://mlbucket-sagemaker/mlops/feature_store_poc/offline_store/
online_store_enabled: false  # Set to true for real-time feature lookups

# Processing instance type
processing_instance_type: ml.m5.xlarge

# Pipeline name
pipeline_name: feature-store-ingestion-pipeline

# Failure notification settings
sns_topic_arn: arn:aws:sns:us-west-2:ACCOUNT_ID:ml-model-pipeline-failures
notification_email: your-email@example.com
```

### Configuration Parameters Explained

#### Feature Store Settings

- **`feature_group_name`**: Name of the Feature Store feature group (must be unique)
- **`record_identifier_name`**: Primary key for feature records (e.g., `customer_id`)
- **`event_time_feature_name`**: Timestamp column for feature versioning (e.g., `event_time`)
- **`offline_store_s3_uri`**: S3 location for Feature Store offline store (must be writable)
- **`online_store_enabled`**: Whether to enable online store for real-time lookups (default: `false`)

#### Athena Settings

- **`athena_database`**: Athena database name (default: `ml_metrics`)
- **`athena_table`**: Athena table name containing raw churn data
- **`athena_query_results`**: S3 path for Athena query results (must be writable)

#### Processing Settings

- **`processing_instance_type`**: Instance type for the ingestion processing job (default: `ml.m5.xlarge`)

#### Notification Settings

- **`sns_topic_arn`**: SNS topic ARN for failure notifications
- **`notification_email`**: Email address to receive failure notifications

---

## Setup and Installation

### 1. Install Dependencies

```bash
cd sagemaker/mlops_poc/feature_store_poc
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
4. Adjust instance types and Feature Store settings as needed

### 4. Verify Athena Database

Ensure the Athena database exists:

```bash
aws athena get-database \
  --catalog-name AwsDataCatalog \
  --database-name ml_metrics \
  --region us-west-2
```

If it doesn't exist, create it:

```bash
aws glue create-database \
  --database-input '{
    "Name": "ml_metrics",
    "Description": "ML metrics database"
  }' \
  --region us-west-2
```

### 5. Generate Test Data (Optional)

If you need to generate test data:

```bash
python3 generate_churn_to_athena.py \
  --config configs/pipeline_config.yaml \
  --database ml_metrics \
  --table churn_prediction_data \
  --n-samples 5000 \
  --start-customer-id 1 \
  --athena-output s3://mlbucket-sagemaker/athena-query-results/
```

---

## Data Generation

### Generate Churn Data to Athena

The `generate_churn_to_athena.py` script creates synthetic messy churn data and stores it in an Athena table.

**Usage**:
```bash
python3 sagemaker/mlops_poc/feature_store_poc/generate_churn_to_athena.py \
  --config sagemaker/mlops_poc/feature_store_poc/configs/pipeline_config.yaml \
  --database ml_metrics \
  --table churn_prediction_data \
  --n-samples 5000 \
  --start-customer-id 1 \
  --athena-output s3://mlbucket-sagemaker/athena-query-results/
```

**Parameters**:
- `--config`: Path to pipeline config file
- `--database`: Athena database name (default: `ml_metrics`)
- `--table`: Athena table name (default: `churn_prediction_data`)
- `--n-samples`: Number of samples to generate (default: 5000)
- `--start-customer-id`: Starting customer ID (default: 1)
- `--athena-output`: S3 path for Athena query results

**Data Schema**:
- `customer_id` (string): Customer identifier (format: `CUST_XXXXXX`)
- `age` (int): Customer age
- `tenure_months` (int): Tenure in months
- `monthly_charges` (double): Monthly charges
- `total_charges` (double): Total charges
- `contract_type` (string): Contract type (e.g., "Month-to-month", "One year", "Two year")
- `payment_method` (string): Payment method (e.g., "Electronic check", "Credit card")
- `internet_service` (string): Internet service type (e.g., "DSL", "Fiber optic", "No")
- `phone_service` (string): Phone service (e.g., "Yes", "No")
- `multiple_lines` (string): Multiple lines (e.g., "Yes", "No")
- `online_security` (string): Online security (e.g., "Yes", "No")
- `tech_support` (string): Tech support (e.g., "Yes", "No")
- `device_protection` (string): Device protection (e.g., "Yes", "No")
- `churn` (int): Churn label (0 or 1)

**Data Characteristics**:
- Intentionally includes missing values
- Extra whitespace in string columns
- Mixed-casing categories
- Designed to test data cleaning capabilities

---

## Pipeline Steps

### Step 1: IngestToFeatureStore

**Purpose**: Ingests cleaned features from Athena table into SageMaker Feature Store.

**Process**:
1. **Installs Dependencies**: Installs required Python packages (`pyyaml`, `sagemaker`, etc.) in the processing container
2. **Loads Configuration**: Reads pipeline config from S3 input
3. **Queries Athena**: Executes `SELECT * FROM churn_prediction_data` to fetch raw data
4. **Cleans Data**:
   - Trims whitespace on string columns
   - Coerces numeric columns (`age`, `tenure_months`, `monthly_charges`, `total_charges`) to numeric
   - Ensures `churn` is 0/1 integer
   - Normalizes categorical columns to title-case
5. **Adds Event Time**: Ensures `event_time` column exists (UTC ISO timestamp)
6. **Creates Feature Group** (if needed):
   - Name: `mlops_poc_churn_classification_feature`
   - Record identifier: `customer_id`
   - Event time: `event_time`
   - Feature definitions inferred from cleaned DataFrame dtypes
   - Offline store: S3 location for batch processing
   - Online store: Optional (disabled by default)
7. **Waits for Feature Group**: Polls until feature group is `Created`
8. **Ingests Records**: Ingests records in batches into Feature Store (offline store; online optional)
9. **Progress Logging**: Logs ingestion progress

**Inputs**:
- Pipeline config file from S3
- Athena table: `ml_metrics.churn_prediction_data`

**Outputs**:
- Feature Store feature group: `mlops_poc_churn_classification_feature`
- Ingestion logs: `s3://<bucket>/<prefix>/ingestion_output/`

**Key Features**:
- **Automatic Dependency Installation**: Installs required packages in the processing container
- **Robust Error Handling**: Sends SNS notifications on failure (including early import errors)
- **Batch Ingestion**: Ingests records in batches for efficiency
- **Progress Tracking**: Logs ingestion progress

**Failure Handling**: 
- Sends SNS email notification on failure (including early import errors)
- Catches all exceptions and sends notifications before exiting

**IAM Requirements**:
- `sagemaker:CreateProcessingJob`
- `sagemaker:DescribeProcessingJob`
- `sagemaker:DescribeFeatureGroup`
- `sagemaker:CreateFeatureGroup`
- `sagemaker:PutRecord`
- `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults`
- `glue:GetDatabase`, `glue:GetTable`
- S3 read/write access
- `sns:Publish`

---

## Execution

### Method 1: Using `execute_pipeline.py` (Recommended)

```bash
python3 sagemaker/mlops_poc/feature_store_poc/execute_pipeline.py \
  --config sagemaker/mlops_poc/feature_store_poc/configs/pipeline_config.yaml
```

This script:
- Loads the existing pipeline by name
- Starts a pipeline execution

### Method 2: Using `pipeline.py` Directly

```bash
python3 sagemaker/mlops_poc/feature_store_poc/pipeline.py \
  --config sagemaker/mlops_poc/feature_store_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 3: Using AWS CLI

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name feature-store-ingestion-pipeline \
  --region us-west-2
```

### Method 4: Using EventBridge (Scheduled Execution)

The pipeline can be scheduled to run automatically using AWS EventBridge:

```bash
# Create EventBridge rule (runs daily at midnight UTC)
aws events put-rule \
  --name feature-store-daily-ingestion \
  --schedule-expression "cron(0 0 * * ? *)" \
  --state ENABLED \
  --region us-west-2

# Add pipeline as target
aws events put-targets \
  --rule feature-store-daily-ingestion \
  --targets '[
    {
      "Id": "feature-store-target",
      "Arn": "arn:aws:sagemaker:us-west-2:ACCOUNT_ID:pipeline/feature-store-ingestion-pipeline",
      "RoleArn": "arn:aws:iam::ACCOUNT_ID:role/EventBridgeSageMakerStartPipelineRole",
      "Input": "{\"PipelineExecutionDisplayName\":\"scheduled-daily-ingestion\"}"
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
2. Select `feature-store-ingestion-pipeline`
3. View execution history and step status

**AWS CLI**:
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name feature-store-ingestion-pipeline \
  --region us-west-2
```

### CloudWatch Logs

The ingestion processing job generates logs in CloudWatch:

**Log Group Pattern**: `/aws/sagemaker/ProcessingJobs/<job-name>`

**View Logs**:
```bash
aws logs tail /aws/sagemaker/ProcessingJobs/<job-name> --follow --region us-west-2
```

### Failure Notifications

The ingestion script sends SNS email notifications on failure:

**Configuration**:
- SNS topic ARN: `sns_topic_arn` in `pipeline_config.yaml`
- Notification email: `notification_email` in `pipeline_config.yaml`

**Email Format**:
- **Subject**: `[SageMaker Pipeline Failed] Feature Store Ingestion - Churn Classification`
- **Body**: Includes step name, error message, and CloudWatch logs reference

**Early Failure Handling**: The script includes `send_failure_notification_early()` to catch import errors and other early failures before the main execution.

**IAM Requirements**:
- Execution role must have `sns:Publish` permission on the SNS topic
- SNS topic policy must allow the execution role to publish

### Check Feature Group Status

```bash
aws sagemaker describe-feature-group \
  --feature-group-name mlops_poc_churn_classification_feature \
  --region us-west-2
```

### Monitor Feature Store Metrics

View Feature Store metrics in CloudWatch:
- Ingestion latency
- Ingestion errors
- Record count
- Storage size

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
    "sagemaker:DescribeFeatureGroup",
    "sagemaker:CreateFeatureGroup",
    "sagemaker:PutRecord",
    "sagemaker:GetRecord",
    "sagemaker:DeleteRecord"
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
    "sagemaker:CreateFeatureGroup",
    "sagemaker:PutRecord",
    "sagemaker:GetRecord",
    "sagemaker:DeleteRecord"
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
    "glue:GetPartitions",
    "glue:CreateTable",
    "glue:UpdateTable"
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

---

## Feature Store Details

### Feature Group Information

- **Name**: `mlops_poc_churn_classification_feature`
- **Record Identifier**: `customer_id` (primary key)
- **Event Time**: `event_time` (timestamp for versioning)
- **Offline Store**: S3 location for batch processing
- **Online Store**: Disabled by default (can be enabled for real-time lookups)

### Feature Definitions

The feature group includes the following features:
- `customer_id` (STRING): Customer identifier (record identifier)
- `event_time` (STRING): Event timestamp (event time feature)
- `age` (INT): Customer age
- `tenure_months` (INT): Tenure in months
- `monthly_charges` (FLOAT): Monthly charges
- `total_charges` (FLOAT): Total charges
- `contract_type` (STRING): Contract type
- `payment_method` (STRING): Payment method
- `internet_service` (STRING): Internet service type
- `phone_service` (STRING): Phone service
- `multiple_lines` (STRING): Multiple lines
- `online_security` (STRING): Online security
- `tech_support` (STRING): Tech support
- `device_protection` (STRING): Device protection
- `churn` (INT): Churn label (0 or 1)

### Offline Store

- **Location**: `s3://mlbucket-sagemaker/mlops/feature_store_poc/offline_store/`
- **Format**: Parquet (default)
- **Partitioning**: By event time
- **Query Method**: Via Athena (Glue table created automatically)

### Online Store (Optional)

- **Status**: Disabled by default
- **Enable**: Set `online_store_enabled: true` in config
- **Use Case**: Real-time feature lookups for inference
- **Cost**: Additional cost for DynamoDB storage

### Viewing Feature Store in AWS Console

1. Navigate to SageMaker → Feature Store
2. Select `mlops_poc_churn_classification_feature`
3. View feature definitions, offline/online store details, and ingestion history

### Querying Feature Store

**Via Athena** (Offline Store):
```sql
SELECT * FROM "sagemaker_featurestore"."mlops_poc_churn_classification_feature"
WHERE customer_id >= 'CUST_000001' AND customer_id <= 'CUST_000400'
```

**Via SDK** (Online Store, if enabled):
```python
import boto3
featurestore_runtime = boto3.client('sagemaker-featurestore-runtime')
response = featurestore_runtime.get_record(
    FeatureGroupName='mlops_poc_churn_classification_feature',
    RecordIdentifierValueAsString='CUST_000001'
)
```

---

## Troubleshooting

### ModuleNotFoundError in Processing Job

**Symptoms**: Processing job fails with `ModuleNotFoundError` (e.g., `yaml`, `sagemaker`).

**Solutions**:
1. The ingestion script automatically installs required packages at runtime
2. Check CloudWatch logs for installation errors
3. Verify the script has internet access to download packages
4. Review the `subprocess.run` calls in `ingest_to_feature_store.py`

### Feature Group Creation Fails

**Symptoms**: Feature group creation fails with validation errors.

**Solutions**:
1. Verify `event_time` column exists in the DataFrame
2. Check feature definitions match DataFrame dtypes
3. Ensure `record_identifier_name` and `event_time_feature_name` are in feature definitions
4. Review CloudWatch logs for detailed error messages

### Athena Query Fails

**Symptoms**: DataSource step fails when querying Athena.

**Solutions**:
1. Verify Athena database and table exist
2. Check IAM permissions for Athena and Glue
3. Ensure Athena query results location is writable
4. Verify table schema matches expected format

### Ingestion Timeout

**Symptoms**: Ingestion takes too long or times out.

**Solutions**:
1. Increase processing instance size (e.g., `ml.m5.2xlarge`)
2. Reduce batch size in ingestion script
3. Check network connectivity to S3
4. Monitor CloudWatch metrics for throttling

### Feature Store Offline Store Not Accessible

**Symptoms**: Downstream pipelines cannot query Feature Store offline store.

**Solutions**:
1. Verify Glue table was created for the offline store
2. Check Glue database name (usually `sagemaker_featurestore`)
3. Ensure IAM permissions for Glue and Athena
4. Verify offline store S3 location is accessible

### SNS Notification Not Received

**Symptoms**: Pipeline fails but no email notification is received.

**Solutions**:
1. Verify SNS topic ARN is correct in config
2. Check SNS topic policy allows the execution role to publish
3. Verify notification email is subscribed to the SNS topic
4. Check CloudWatch logs for SNS publish errors

---

## Cost Considerations

### Processing Costs

- **Instance Type**: Use `ml.m5.xlarge` for processing (configurable)
- **Runtime**: Ingestion time depends on data volume
- **Storage**: Feature Store offline store incurs S3 storage costs

### Feature Store Costs

- **Offline Store**: S3 storage costs (Parquet format, compressed)
- **Online Store** (if enabled): DynamoDB storage and read/write costs
- **Ingestion**: Processing job costs

### Optimization Tips

- Use appropriate instance types for your data volume
- Schedule ingestion during off-peak hours
- Monitor S3 storage and clean up old partitions if needed
- Disable online store if not needed for real-time lookups

---

## Clean Up

### Delete Pipeline

```bash
aws sagemaker delete-pipeline \
  --pipeline-name feature-store-ingestion-pipeline \
  --region us-west-2
```

### Delete Feature Group

```bash
aws sagemaker delete-feature-group \
  --feature-group-name mlops_poc_churn_classification_feature \
  --region us-west-2
```

**Note**: This will delete both offline and online stores. Ensure downstream pipelines are updated before deleting.

### Delete S3 Artifacts

```bash
# Delete offline store
aws s3 rm s3://mlbucket-sagemaker/mlops/feature_store_poc/offline_store/ --recursive

# Delete ingestion outputs
aws s3 rm s3://mlbucket-sagemaker/mlops/feature_store_poc/ingestion_output/ --recursive
```

### Delete Athena Table (if needed)

```bash
aws glue delete-table \
  --database-name ml_metrics \
  --name churn_prediction_data \
  --region us-west-2
```

### Delete EventBridge Rule (if scheduled)

```bash
aws events remove-targets \
  --rule feature-store-daily-ingestion \
  --ids feature-store-target \
  --region us-west-2

aws events delete-rule \
  --name feature-store-daily-ingestion \
  --region us-west-2
```

---

## Integration with Other Pipelines

### Training Pipeline Integration

The `training_poc` pipeline sources data from this Feature Store:
- **Feature Group**: `mlops_poc_churn_classification_feature`
- **Customer ID Range**: 1-4000 (training data)
- **Query Method**: Via Athena (offline store)

### AutoML Training Pipeline Integration

The `automl_training_poc` pipeline sources data from this Feature Store:
- **Feature Group**: `mlops_poc_churn_classification_feature`
- **Customer ID Range**: 1-4000 (training data)
- **Query Method**: Via Athena (offline store)

### Batch Inference Pipeline Integration

The `batch_inference_poc` pipeline sources data from this Feature Store:
- **Feature Group**: `mlops_poc_churn_classification_feature`
- **Customer ID Range**: 4000-5000 (inference data)
- **Query Method**: Via Athena (offline store)

### Data Flow Across Pipelines

```
generate_churn_to_athena.py
    ↓ (generates messy data)
Athena Table (churn_prediction_data)
    ↓ (ingestion)
Feature Store (mlops_poc_churn_classification_feature)
    ↓ (offline store)
    ├─→ training_poc (customer_id 1-4000)
    ├─→ automl_training_poc (customer_id 1-4000)
    └─→ batch_inference_poc (customer_id 4000-5000)
```

### Alignment with Existing Pipelines

The feature columns mirror the schema used in:
- `training_poc/src/preprocessing.py`
- `automl_training_poc/src/preprocessing.py`
- `batch_inference_poc/src/batch_inference.py`

This ensures:
- Same preprocessing logic can be used downstream
- Feature consistency across training and inference
- Easy migration from Athena tables to Feature Store

---

## Additional Resources

- [SageMaker Feature Store Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [Athena Documentation](https://docs.aws.amazon.com/athena/latest/ug/what-is.html)
- [Glue Catalog Documentation](https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html)

---

## License

[Your License Here]
