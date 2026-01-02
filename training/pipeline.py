#!/usr/bin/env python
"""SageMaker Pipeline for Classification Model MLOps."""
import os
import argparse
import yaml
import json
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.properties import PropertyFile


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


def build_pipeline(config: dict) -> Pipeline:
    """Build the SageMaker Pipeline."""
    region = config["region"]
    bucket = config["bucket"]
    prefix = config["prefix"]
    role = config["role_arn"]
    pipeline_session = get_pipeline_session(region)
    
    # Parameters
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=config["processing_instance_type"]
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=config["training_instance_type"]
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value=config["approval_status"]
    )
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=float(config["accuracy_threshold"])
    )
    dataset_name = ParameterString(
        name="DatasetName",
        default_value=config.get("dataset_name", "customer_churn")
    )
    
    # Failure notification configuration
    sns_topic_arn = config.get("sns_topic_arn")
    notification_email = config.get("notification_email")

    # Step 0: Data Sourcing from Feature Store (replaces Athena table)
    # Use SKLearnProcessor for consistency (has boto3, pandas, numpy, etc.)
    data_source_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-data-sourcing",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )
    
    step_generate_data = ProcessingStep(
        name="DataSource",
        processor=data_source_processor,
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/raw",
                destination=f"s3://{bucket}/{prefix}/data/raw",
                output_name="raw_data",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "data_source.py"),
        job_arguments=[
            "--feature-group-name", config["feature_group_name"],
            "--customer-id-start", str(config["customer_id_start"]),
            "--customer-id-end", str(config["customer_id_end"]),
            "--region", region,
            "--output-path", "/opt/ml/processing/output/raw",
            "--athena-output", config["athena_query_results"],
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
    )
    
    # Step 1: Data Preprocessing
    # Use SKLearnProcessor for consistency (has scikit-learn, pandas, numpy pre-installed)
    preprocess_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-preprocess",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )
    
    step_preprocess = ProcessingStep(
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=step_generate_data.properties.ProcessingOutputConfig.Outputs["raw_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/train",
                destination=f"s3://{bucket}/{prefix}/data/processed/train",
                output_name="train",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/output/test",
                destination=f"s3://{bucket}/{prefix}/data/processed/test",
                output_name="test",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "preprocessing.py"),
        job_arguments=[
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_generate_data.name],
    )
    
    # Step 2: Model Training
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=os.path.join(os.path.dirname(__file__), "src"),
        framework_version="1.2-1",
        py_version="py3",
        instance_type=training_instance_type,
        role=role,
        sagemaker_session=pipeline_session,
        hyperparameters={
            "n-estimators": config["hyperparameters"]["n_estimators"],
            "max-depth": config["hyperparameters"]["max_depth"],
            "min-samples-split": config["hyperparameters"]["min_samples_split"],
            "min-samples-leaf": config["hyperparameters"]["min_samples_leaf"],
            "sns-topic-arn": sns_topic_arn,
            "notification-email": notification_email,
        },
        base_job_name=f"{config['project_name']}-train",
    )
    
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
            ),
        },
    )
    
    # Step 3: Model Evaluation
    # Use SKLearnProcessor to match scikit-learn version with training (1.2.1)
    eval_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-evaluate",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{prefix}/evaluation",
                output_name="evaluation",
            )
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "evaluate.py"),
        job_arguments=[
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        property_files=[evaluation_report],
        depends_on=[step_train.name],  # Explicitly depend on training step
    )
    
    # Step 4: Register Model (conditional on accuracy)
    # Use Join function to concatenate S3 URI with file path
    from sagemaker.workflow.functions import Join
    
    eval_output_uri = step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    metrics_uri = Join(
        on="/",
        values=[eval_output_uri, "evaluation.json"]
    )
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=metrics_uri,
            content_type="application/json",
        )
    )
    
    # Create a model for registration that uses the inference script
    sklearn_model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        entry_point="inference.py",
        source_dir=os.path.join(os.path.dirname(__file__), "src"),
        framework_version="1.2-1",
        py_version="py3",
        sagemaker_session=pipeline_session,
    )
    
    step_register = RegisterModel(
        name="RegisterModel",
        model=sklearn_model,
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=[config["inference_instance_type"]],
        transform_instances=[config["processing_instance_type"]],
        model_package_group_name=config["model_package_group"],
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # Step 5: Deploy Endpoint (with autoscaling)
    if config.get("enable_autoscaling", False):
        # Use SKLearnProcessor for deployment script (has boto3)
        deploy_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type="ml.t3.medium",  # Small instance for deployment script
            instance_count=1,
            base_job_name=f"{config['project_name']}-deploy",
            sagemaker_session=pipeline_session,
        volume_size_in_gb=10,
        )
        
        step_deploy = ProcessingStep(
            name="DeployEndpoint",
            processor=deploy_processor,
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output",
                    destination=f"s3://{bucket}/{prefix}/deployment",
                    output_name="deployment",
                )
            ],
            code=os.path.join(os.path.dirname(__file__), "src", "deploy_endpoint.py"),
            job_arguments=[
                "--model-package-group", config["model_package_group"],
                "--endpoint-name", config.get("endpoint_name", "classification-model-endpoint"),
                "--instance-type", config["inference_instance_type"],
                "--role-arn", config["role_arn"],
                "--region", config["region"],
                "--output-path", "/opt/ml/processing/output",
                "--min-capacity", str(config.get("autoscaling_min_capacity", 1)),
                "--max-capacity", str(config.get("autoscaling_max_capacity", 5)),
                "--sns-topic-arn", sns_topic_arn,
                "--notification-email", notification_email,
            ],
            depends_on=[step_register.name],
        )
        
        # Include deployment step in conditional
        if_steps_list = [step_register, step_deploy]
    else:
        if_steps_list = [step_register]
    
    # Conditional step: Only register and deploy if accuracy >= threshold
    step_cond = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_eval.name,
                    property_file=evaluation_report,
                    json_path="accuracy",
                ),
                right=accuracy_threshold,
            )
        ],
        if_steps=if_steps_list,
        else_steps=[],
    )
    
    # Build pipeline steps
    pipeline_steps = [step_generate_data, step_preprocess, step_train, step_eval, step_cond]
    
    # Create pipeline
    pipeline = Pipeline(
        name=config["pipeline_name"],
        parameters=[
            processing_instance_type,
            training_instance_type,
            model_approval_status,
            accuracy_threshold,
            dataset_name,
        ],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session,
    )
    
    return pipeline


def main():
    """Main entry point."""
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "configs", "pipeline_config.yaml")
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    pipeline = build_pipeline(config)
    
    # Upsert (create or update) pipeline
    pipeline.upsert(role_arn=config["role_arn"])
    print(f"Pipeline '{config['pipeline_name']}' created/updated successfully")
    
    # Start execution
    execution = pipeline.start()
    print(f"Pipeline execution started: {execution.arn}")
    print(f"View execution in SageMaker Studio or AWS Console")


if __name__ == "__main__":
    main()

