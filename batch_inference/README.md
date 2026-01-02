# Batch Inference Pipeline - Classification Model MLOps

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
11. [Output Format](#output-format)
12. [Troubleshooting](#troubleshooting)
13. [Cost Considerations](#cost-considerations)
14. [Clean Up](#clean-up)
15. [Integration with Other Pipelines](#integration-with-other-pipelines)

---

## Overview

The Batch Inference Pipeline is an end-to-end SageMaker pipeline for running batch predictions on large datasets. This pipeline sources inference data from SageMaker Feature Store, applies the same preprocessing as training, runs batch predictions using the latest approved model from the Model Registry, and publishes results to an Athena table for downstream consumption.

### Key Features

- **Batch Processing**: Efficient batch predictions on large datasets
- **Feature Store Integration**: Sources inference data from SageMaker Feature Store (customer_id 4000-5000)
- **Model Registry Integration**: Automatically uses the latest approved model from Model Registry
- **Preprocessing Consistency**: Applies the same preprocessing as training (missing values, categorical encoding)
- **Evaluation Support**: Optional evaluation metrics if ground-truth labels are present
- **Athena Integration**: Publishes predictions to Athena table for easy querying
- **Failure Notifications**: SNS email notifications for pipeline step failures

### Use Case

This pipeline is designed for batch churn prediction on customer data. It processes inference data in batches, generates predictions, and stores results in an Athena table for downstream analytics and reporting.

### When to Use Batch Inference

- **Large Datasets**: Processing thousands or millions of records
- **Non-Real-Time**: Predictions don't need to be returned immediately
- **Cost Efficiency**: More cost-effective than real-time endpoints for large volumes
- **Scheduled Processing**: Regular batch predictions (daily, weekly, etc.)

---

## Architecture

### Pipeline Flow

```
┌─────────────────┐
│  Data Source    │ → Retrieves inference data from Feature Store (customer_id 4000-5000)
│  (Feature Store)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BatchInference  │ → Loads latest approved model from Model Registry
│                 │   Applies preprocessing (missing values, encoding)
│                 │   Generates predictions
│                 │   Optionally evaluates if labels present
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PublishToAthena │ → Creates/updates Athena table
│                 │   Writes predictions to table
└─────────────────┘
```

### Data Flow

1. **Input**: Feature Store feature group `mlops_poc_churn_classification_feature` (customer_id 4000-5000)
2. **Processing**: Data cleaning, encoding (same as training)
3. **Prediction**: Batch predictions using latest approved model
4. **Output**: Predictions in S3 and Athena table `ml_metrics.mlops_poc_daily_inference`

---

## Prerequisites

### AWS Account Requirements

1. **AWS Account** with appropriate permissions (see [IAM Permissions](#iam-permissions))
2. **S3 Bucket** for storing predictions and intermediate data
3. **IAM Role** with SageMaker execution permissions
4. **SageMaker Feature Store** feature group must exist (created by `feature_store_poc`)
5. **Model Registry** must have at least one approved model (from `training_poc`)

### Local Environment Requirements

1. **Python 3.8+** with the following packages:
   - `boto3` (AWS SDK)
   - `sagemaker` (SageMaker SDK)
   - `pyyaml` (Configuration parsing)
   - `pandas` (Data manipulation)
   - `numpy` (Numerical operations)
   - `scikit-learn` (For preprocessing and evaluation)

2. **AWS CLI** configured with credentials:
   ```bash
   aws configure
   ```

3. **Access to AWS Console** for monitoring pipeline executions

### Feature Store Prerequisites

- Feature group `mlops_poc_churn_classification_feature` must exist
- Feature group must have data for customer_id range 4000-5000
- Feature group offline store must be accessible via Athena

### Model Registry Prerequisites

- Model Package Group `classification-model-group` must exist
- At least one model must be approved in the Model Registry
- Model must have been trained using the same preprocessing as this pipeline

---

## Project Structure

```
batch_inference_poc/
├── configs/
│   └── pipeline_config.yaml      # Batch inference pipeline configuration
├── src/
│   ├── data_source.py            # Fetches inference data from Feature Store
│   ├── batch_inference.py        # Preprocess + predict + optional evaluation
│   └── publish_to_athena.py      # Creates/updates Athena table for predictions
├── pipeline.py                   # Pipeline definition
├── execute_pipeline.py           # Run the pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Key Files

- **`pipeline.py`**: Defines the SageMaker Pipeline with all steps
- **`execute_pipeline.py`**: Script to create/update and start pipeline execution
- **`configs/pipeline_config.yaml`**: Centralized configuration for all pipeline parameters
- **`src/data_source.py`**: Queries Feature Store offline store via Athena
- **`src/batch_inference.py`**: Loads model, applies preprocessing, generates predictions, optionally evaluates
- **`src/publish_to_athena.py`**: Creates/updates Athena table and writes predictions

---

## Configuration

### Configuration File: `configs/pipeline_config.yaml`

Edit this file to customize the pipeline behavior:

```yaml
# Project settings
project_name: classification-model-batch-inference
region: us-west-2
bucket: mlbucket-sagemaker
prefix: mlops/batch_inference
pipeline_name: classification-model-batch-inference-pipeline
model_package_group: classification-model-group  # Source of the trained model

# IAM Role - should have SageMaker, S3, Feature Store, Athena, Glue permissions
role_arn: arn:aws:iam::ACCOUNT_ID:role/ML-Role-Pipelines

# Instance types (processing only for batch inference)
processing_instance_type: ml.m5.xlarge

# Feature Store settings (replaces Athena input table)
feature_group_name: mlops_poc_churn_classification_feature
customer_id_start: 4000  # Inference data: customer_id 4000-5000
customer_id_end: 5000
athena_query_results: s3://mlbucket-sagemaker/athena-query-results/

# Athena settings (for output table only)
athena_database: ml_metrics
athena_output_table: mlops_poc_daily_inference

# Failure notification settings
sns_topic_arn: arn:aws:sns:us-west-2:ACCOUNT_ID:ml-model-pipeline-failures
notification_email: your-email@example.com

# Deployment settings (model source)
model_approval_status: Approved  # Latest approved model will be used for inference
```

### Configuration Parameters Explained

#### Feature Store Settings

- **`feature_group_name`**: Name of the SageMaker Feature Store feature group
- **`customer_id_start`**: Starting customer ID for inference data (inclusive)
- **`customer_id_end`**: Ending customer ID for inference data (inclusive)
- **`athena_query_results`**: S3 path for Athena query results (must be writable)

#### Athena Settings

- **`athena_database`**: Athena database name for output table (default: `ml_metrics`)
- **`athena_output_table`**: Athena table name for predictions (default: `mlops_poc_daily_inference`)

#### Model Registry Settings

- **`model_package_group`**: Model Package Group name (must match training pipeline)
- **`model_approval_status`**: Approval status filter (default: `Approved`)

#### Processing Settings

- **`processing_instance_type`**: Instance type for batch inference processing (default: `ml.m5.xlarge`)

---

## Setup and Installation

### 1. Install Dependencies

```bash
cd sagemaker/mlops_poc/batch_inference_poc
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
4. Adjust instance types and customer ID ranges as needed
5. Verify `model_package_group` matches the training pipeline

### 4. Verify Feature Store

Ensure the feature group exists and has data:

```bash
aws sagemaker describe-feature-group \
  --feature-group-name mlops_poc_churn_classification_feature \
  --region us-west-2
```

### 5. Verify Model Registry

Ensure at least one model is approved:

```bash
aws sagemaker list-model-packages \
  --model-package-group-name classification-model-group \
  --model-approval-status Approved \
  --region us-west-2
```

### 6. Create/Update Pipeline

```bash
python3 pipeline.py --config configs/pipeline_config.yaml
```

This will:
- Create or update the SageMaker Pipeline definition
- Start a pipeline execution

---

## Pipeline Steps

### Step 1: DataSource

**Purpose**: Retrieves inference data from SageMaker Feature Store offline store.

**Process**:
1. Describes the feature group to get Glue database and table names
2. Constructs an Athena query to select data from the offline store
3. Filters by `customer_id` range (4000-5000) using `CUST_XXXXXX` format
4. Executes the query via Athena
5. Saves raw CSV data to S3

**Inputs**:
- Feature group name: `mlops_poc_churn_classification_feature`
- Customer ID range: 4000-5000

**Outputs**:
- Raw CSV data: `s3://<bucket>/<prefix>/data/raw/`

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:DescribeFeatureGroup`
- `athena:StartQueryExecution`, `athena:GetQueryExecution`, `athena:GetQueryResults`
- `glue:GetDatabase`, `glue:GetTable`
- S3 read/write access

---

### Step 2: BatchInference

**Purpose**: Applies preprocessing and generates batch predictions.

**Process**:
1. **Loads Latest Approved Model**:
   - Queries Model Registry for latest approved model
   - Downloads model artifacts from S3
   - Loads model, preprocessing metadata, and label encoders
2. **Loads Inference Data**:
   - Reads raw inference data from S3
   - Drops identifier and metadata columns (`customer_id`, `event_time`, etc.)
3. **Applies Preprocessing**:
   - Handles missing values (same as training)
   - Applies label encoding (using training encoders)
   - Ensures feature order matches training
4. **Generates Predictions**:
   - Runs predictions on preprocessed data
   - Saves predictions to S3
5. **Optional Evaluation** (if target column exists):
   - Calculates accuracy, precision, recall, F1-score
   - Saves evaluation metrics to S3

**Inputs**:
- Raw inference data from DataSource step
- Model artifacts from Model Registry

**Outputs**:
- Predictions: `s3://<bucket>/<prefix>/predictions/`
- Evaluation (optional): `s3://<bucket>/<prefix>/evaluation/`

**Key Features**:
- **Preprocessing Consistency**: Uses the same preprocessing as training
- **Model Versioning**: Automatically uses latest approved model
- **Evaluation Support**: Optional evaluation if ground-truth labels are present
- **Batch Processing**: Efficient processing of large datasets

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `sagemaker:DescribeModelPackage`
- `sagemaker:DescribeModel`
- `sagemaker:CreateProcessingJob`
- `sagemaker:DescribeProcessingJob`
- S3 read/write access

---

### Step 3: PublishToAthena

**Purpose**: Creates/updates Athena table and writes predictions.

**Process**:
1. **Loads Predictions**:
   - Reads predictions from S3
   - Merges with original customer IDs (if needed)
2. **Creates/Updates Athena Table**:
   - Checks if table exists
   - Creates table with inferred schema if it doesn't exist
   - Updates table if it exists (appends or overwrites based on configuration)
3. **Writes Predictions**:
   - Writes predictions to S3 (Parquet or CSV format)
   - Updates Glue table metadata
   - Ensures table is queryable via Athena

**Inputs**:
- Predictions from BatchInference step

**Outputs**:
- Athena table: `ml_metrics.mlops_poc_daily_inference`
- Predictions in S3: `s3://<bucket>/<prefix>/athena_output/`

**Key Features**:
- **Automatic Schema Inference**: Infers column types from predictions
- **Table Management**: Creates or updates table as needed
- **Queryable Results**: Predictions are immediately queryable via Athena

**Failure Handling**: Sends SNS email notification on failure.

**IAM Requirements**:
- `glue:CreateTable`, `glue:UpdateTable`, `glue:GetTable`
- `athena:StartQueryExecution`, `athena:GetQueryExecution`
- S3 read/write access

---

## Execution

### Method 1: Using `execute_pipeline.py` (Recommended)

```bash
python3 sagemaker/mlops_poc/batch_inference_poc/execute_pipeline.py \
  --config sagemaker/mlops_poc/batch_inference_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 2: Using `pipeline.py` Directly

```bash
python3 sagemaker/mlops_poc/batch_inference_poc/pipeline.py \
  --config sagemaker/mlops_poc/batch_inference_poc/configs/pipeline_config.yaml
```

This script:
- Creates or updates the pipeline definition
- Starts a pipeline execution

### Method 3: Using AWS CLI

```bash
aws sagemaker start-pipeline-execution \
  --pipeline-name classification-model-batch-inference-pipeline \
  --region us-west-2
```

### Method 4: Using EventBridge (Scheduled Execution)

The pipeline can be scheduled to run automatically using AWS EventBridge:

```bash
# Create EventBridge rule (runs daily at 2 AM UTC)
aws events put-rule \
  --name batch-inference-daily \
  --schedule-expression "cron(0 2 * * ? *)" \
  --state ENABLED \
  --region us-west-2

# Add pipeline as target
aws events put-targets \
  --rule batch-inference-daily \
  --targets '[
    {
      "Id": "batch-inference-target",
      "Arn": "arn:aws:sagemaker:us-west-2:ACCOUNT_ID:pipeline/classification-model-batch-inference-pipeline",
      "RoleArn": "arn:aws:iam::ACCOUNT_ID:role/EventBridgeSageMakerStartPipelineRole",
      "Input": "{\"PipelineExecutionDisplayName\":\"scheduled-daily-batch-inference\"}"
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
2. Select `classification-model-batch-inference-pipeline`
3. View execution history and step status

**AWS CLI**:
```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name classification-model-batch-inference-pipeline \
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

### Check Predictions in S3

```bash
aws s3 ls s3://mlbucket-sagemaker/mlops/batch_inference/predictions/ --recursive
```

### Query Predictions in Athena

```sql
SELECT * FROM ml_metrics.mlops_poc_daily_inference
WHERE prediction = 1  -- Churn predictions
ORDER BY customer_id
LIMIT 100;
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
    "sagemaker:DescribeModelPackage",
    "sagemaker:DescribeModel",
    "sagemaker:ListModelPackages"
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
    "sagemaker:GetRecord"
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

## Output Format

### Predictions Format

Predictions are saved as CSV files with the following columns:

- `customer_id`: Customer identifier (from original data)
- `prediction`: Predicted class (0 or 1 for binary classification)
- `prediction_probability` (optional): Prediction probability if available

**Example**:
```csv
customer_id,prediction
CUST_000400,0
CUST_000401,1
CUST_000402,0
```

### Athena Table Schema

The Athena table `ml_metrics.mlops_poc_daily_inference` has the following schema:

- `customer_id` (STRING): Customer identifier
- `prediction` (INT): Predicted class (0 or 1)
- `prediction_probability` (DOUBLE, optional): Prediction probability
- `timestamp` (TIMESTAMP, optional): Prediction timestamp

### Evaluation Metrics (if available)

If ground-truth labels are present, evaluation metrics are saved to:

`s3://<bucket>/<prefix>/evaluation/evaluation.json`

**Format**:
```json
{
  "accuracy": 0.85,
  "precision": 0.82,
  "recall": 0.78,
  "f1_score": 0.80
}
```

---

## Troubleshooting

### No Approved Model Found

**Symptoms**: BatchInference step fails with "no approved model found" error.

**Solutions**:
1. Verify Model Package Group exists:
   ```bash
   aws sagemaker describe-model-package-group \
     --model-package-group-name classification-model-group \
     --region us-west-2
   ```
2. Check for approved models:
   ```bash
   aws sagemaker list-model-packages \
     --model-package-group-name classification-model-group \
     --model-approval-status Approved \
     --region us-west-2
   ```
3. Approve a model in the Model Registry if needed
4. Verify `model_package_group` in config matches training pipeline

### Preprocessing Mismatch

**Symptoms**: Predictions fail or are incorrect due to preprocessing errors.

**Solutions**:
1. Ensure preprocessing artifacts (label encoders, metadata) are available in model artifacts
2. Verify feature order matches training
3. Check for missing features in inference data
4. Review preprocessing logs in CloudWatch

### Feature Store Query Errors

**Symptoms**: DataSource step fails when querying Feature Store.

**Solutions**:
1. Verify feature group offline store is accessible
2. Check Glue database and table names match
3. Ensure Athena query results location is writable
4. Verify customer_id filtering format matches `CUST_XXXXXX` pattern
5. Ensure data exists for customer_id range 4000-5000

### Athena Table Creation Fails

**Symptoms**: PublishToAthena step fails when creating/updating table.

**Solutions**:
1. Verify Glue permissions for table creation
2. Check Athena database exists
3. Ensure S3 location for table is writable
4. Review CloudWatch logs for detailed error messages

### Predictions Not Queryable

**Symptoms**: Predictions are written but not queryable in Athena.

**Solutions**:
1. Verify Glue table metadata is updated
2. Check table partition configuration
3. Ensure S3 location is accessible
4. Wait a few minutes for Glue catalog to update

---

## Cost Considerations

### Processing Costs

- **Instance Type**: Use `ml.m5.xlarge` for processing (configurable)
- **Runtime**: Processing time depends on data volume
- **Storage**: S3 storage costs for predictions and intermediate data

### Optimization Tips

- Use appropriate instance types for your data volume
- Schedule batch inference during off-peak hours
- Monitor S3 storage and clean up old predictions if needed
- Use Parquet format for Athena tables (more efficient than CSV)

### Cost Comparison

- **Batch Inference**: More cost-effective for large volumes (thousands+ records)
- **Real-time Endpoint**: More cost-effective for small volumes or real-time requirements
- **Scheduled Execution**: Run during off-peak hours to reduce costs

---

## Clean Up

### Delete Pipeline

```bash
aws sagemaker delete-pipeline \
  --pipeline-name classification-model-batch-inference-pipeline \
  --region us-west-2
```

### Delete Athena Table

```bash
aws glue delete-table \
  --database-name ml_metrics \
  --name mlops_poc_daily_inference \
  --region us-west-2
```

### Delete S3 Artifacts

```bash
aws s3 rm s3://mlbucket-sagemaker/mlops/batch_inference/ --recursive
```

### Delete EventBridge Rule (if scheduled)

```bash
aws events remove-targets \
  --rule batch-inference-daily \
  --ids batch-inference-target \
  --region us-west-2

aws events delete-rule \
  --name batch-inference-daily \
  --region us-west-2
```

---

## Integration with Other Pipelines

### Feature Store Integration

This pipeline sources data from the Feature Store feature group created by `feature_store_poc`:
- **Feature Group**: `mlops_poc_churn_classification_feature`
- **Customer ID Range**: 4000-5000 (inference data)

### Model Registry Integration

This pipeline uses models trained by `training_poc`:
- **Model Package Group**: `classification-model-group`
- **Model Approval Status**: `Approved` (latest approved model)

### Data Flow Across Pipelines

```
feature_store_poc
    ↓ (ingests data)
Feature Store (mlops_poc_churn_classification_feature)
    ├─→ training_poc (customer_id 1-4000) → Model Registry
    │                                          ↓
    └─→ batch_inference_poc (customer_id 4000-5000) → Uses latest approved model
                                                      → Predictions in Athena
```

### Output Consumption

Predictions in the Athena table can be consumed by:
- **Analytics Tools**: Tableau, QuickSight, etc.
- **Reporting Systems**: Automated reports and dashboards
- **Downstream Applications**: Other ML pipelines or business applications
- **Data Warehouses**: ETL processes to data warehouses

---

## Additional Resources

- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [SageMaker Processing](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
- [Feature Store Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)
- [Athena Documentation](https://docs.aws.amazon.com/athena/latest/ug/what-is.html)
- [Glue Catalog Documentation](https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html)

---

## License

[Your License Here]
