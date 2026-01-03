# MLOps POC: End-to-End Machine Learning Pipeline on AWS SageMaker

## Summary
This is an end-to-end MLOps solution using AWS SageMaker Pipelines for churn prediction. The solution includes:
- Feature Store integration
- Model training (manual and AutoML)
- Model registry
- Real-time and batch inference
- Automated workflows

## Architecture Overview
The solution consists of four integrated pipelines:

### 1. **Feature Store POC (`feature_store_poc/`)**
- Ingests cleaned features from Athena into SageMaker Feature Store.
- Centralized feature storage for training and inference.
- Offline store (S3) for batch processing.
- Supports online store for real-time lookups.

### 2. **Training Pipeline (`training_poc/`)**
- Trains a RandomForest classifier using scikit-learn.
- Sources data from Feature Store (customer_id 1-4000).
- Automatic model evaluation and quality gating.
- Model Registry integration with auto-approval.
- Real-time endpoint deployment with autoscaling.

### 3. **AutoML Training Pipeline (`automl_training_poc/`)**
- Automated model selection using SageMaker AutoML.
- Explores multiple algorithms and hyperparameters.
- Timestamp-based model naming to prevent conflicts.
- Batch transform evaluation.
- Endpoint deployment with autoscaling.

### 4. **Batch Inference Pipeline (`batch_inference_poc/`)**
- Batch predictions on large datasets.
- Sources inference data from Feature Store (customer_id 4000-5000).
- Uses the latest approved model from Model Registry.
- Publishes predictions to Athena table for analytics.

## Key Features
- **Feature Store Integration**: Centralized feature management across pipelines.
- **Model Registry**: Versioning, approval workflows, and model tracking.
- **Quality Gates**: Conditional model registration based on accuracy thresholds.
- **Autoscaling**: Endpoint autoscaling for production workloads.
- **Failure Notifications**: SNS email notifications for pipeline failures.
- **EventBridge Scheduling**: Automated pipeline execution.
- **Data Preprocessing**: Consistent preprocessing (missing values, categorical encoding).
- **Customer ID Filtering**: Segregated data for training (1-4000) and inference (4000-5000).

## Technical Stack
- **AWS Services**: SageMaker Pipelines, Feature Store, Model Registry, Athena, Glue, S3, SNS, EventBridge.
- **ML Frameworks**: scikit-learn, SageMaker AutoML.
- **Languages**: Python 3.8+.
- **Infrastructure**: SageMaker Processing Jobs, Training Jobs, Endpoints.


## Documentation
Each pipeline includes detailed READMEs covering:
- Architecture and data flow
- Configuration and setup
- Pipeline steps
- Execution methods
- IAM permissions
- Troubleshooting
- Integration details


## Deployment
Each pipeline can be executed via:
- Python scripts (`execute_pipeline.py` or `pipeline.py`).
- AWS CLI.
- EventBridge (scheduled execution).

