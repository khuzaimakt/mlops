#!/usr/bin/env python
"""SageMaker Pipeline for AutoML training."""
import os
import argparse
import yaml
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.automl_step import AutoMLStep
from sagemaker.automl.automl import AutoML, AutoMLInput
from sagemaker.workflow.properties import PropertyFile


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipeline_session(region: str) -> PipelineSession:
    boto_session = boto3.Session(region_name=region)
    return PipelineSession(boto_session=boto_session)


def build_pipeline(config: dict) -> Pipeline:
    region = config["region"]
    bucket = config["bucket"]
    prefix = config["prefix"]
    role = config["role_arn"]
    pipeline_session = get_pipeline_session(region)

    processing_instance_type = config["processing_instance_type"]
    sns_topic_arn = config.get("sns_topic_arn")
    notification_email = config.get("notification_email")
    
    # Parameters
    accuracy_threshold = ParameterFloat(
        name="AccuracyThreshold",
        default_value=float(config.get("accuracy_threshold", 0.5))
    )

    # Step 0: Data Sourcing
    data_source_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{config['project_name']}-data-sourcing",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=30,
    )

    step_data_source = ProcessingStep(
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

    # Step 1: Preprocess
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
                source=step_data_source.properties.ProcessingOutputConfig.Outputs["raw_data"].S3Output.S3Uri,
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
            ProcessingOutput(
                source="/opt/ml/processing/output/meta",
                destination=f"s3://{bucket}/{prefix}/data/processed/meta",
                output_name="meta",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "preprocessing.py"),
        job_arguments=[
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_data_source.name],
    )

    # Step 2: AutoML Training
    target_attr = config["target_attribute_name"]
    problem_type = config.get("problem_type", "BinaryClassification")
    max_runtime = int(config.get("max_runtime_sec", 3600))
    objective_metric = config.get("objective_metric_name")

    automl_kwargs = dict(
        role=role,
        target_attribute_name=target_attr,
        output_path=f"s3://{bucket}/{prefix}/automl/output",
        base_job_name=f"{config['project_name']}-automl",
        sagemaker_session=pipeline_session,
        mode="ENSEMBLING",  # Required by AutoMLStep
    )
    if problem_type:
        automl_kwargs["problem_type"] = problem_type
    if objective_metric:
        automl_kwargs["job_objective"] = {"MetricName": objective_metric}
    automl_kwargs["max_runtime_per_training_job_in_seconds"] = max_runtime
    # Note: current SDK version in this environment does not support max_parallel_training_jobs/max_candidates kwargs

    auto_ml = AutoML(**automl_kwargs)

    automl_input = AutoMLInput(
        inputs=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        target_attribute_name=target_attr,
    )

    automl_step_args = auto_ml.fit(
        inputs=automl_input,
    )

    step_automl = AutoMLStep(
        name="AutoMLTraining",
        step_args=automl_step_args,
    )

    # Step 3: Create Model from Best Candidate
    create_model_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",  # Small instance for model creation script
        instance_count=1,
        base_job_name=f"{config['project_name']}-create-model",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=10,
    )
    
    step_create_model = ProcessingStep(
        name="CreateModel",
        processor=create_model_processor,
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/model_info",
                output_name="model_info",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "create_model.py"),
        job_arguments=[
            "--pipeline-name", config["pipeline_name"],
            "--model-name-base", f"{config['project_name']}",
            "--role-arn", role,
            "--region", region,
            "--output-path", "/opt/ml/processing/output",
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_automl.name],
    )
    
    # Create a PropertyFile to read model name from create_model output
    model_info_file = PropertyFile(
        name="ModelInfo",
        output_name="model_info",
        path="model_info.json"
    )
    
    step_create_model.property_files = [model_info_file]

    # Step 4: Evaluate Model using Batch Transform
    evaluate_processor = SKLearnProcessor(
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
    
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        processor=evaluate_processor,
        inputs=[
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
            ProcessingInput(
                source=step_create_model.properties.ProcessingOutputConfig.Outputs["model_info"].S3Output.S3Uri,
                destination="/opt/ml/processing/model_info",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/{prefix}/evaluation",
                output_name="evaluation",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "evaluate_automl.py"),
        job_arguments=[
            "--model-info-path", "/opt/ml/processing/model_info/model_info.json",
            "--test-data", "/opt/ml/processing/test",
            "--evaluation-output", "/opt/ml/processing/evaluation",
            "--region", region,
            "--instance-type", processing_instance_type,
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        property_files=[evaluation_report],
        depends_on=[step_create_model.name],
    )

    # Step 5: Register Model (with conditional approval)
    register_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
        base_job_name=f"{config['project_name']}-register",
        sagemaker_session=pipeline_session,
        volume_size_in_gb=10,
    )
    
    step_register = ProcessingStep(
        name="RegisterModel",
        processor=register_processor,
        inputs=[
            ProcessingInput(
                source=step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                destination="/opt/ml/processing/evaluation",
            ),
            ProcessingInput(
                source=step_create_model.properties.ProcessingOutputConfig.Outputs["model_info"].S3Output.S3Uri,
                destination="/opt/ml/processing/model_info",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{bucket}/{prefix}/registration",
                output_name="registration",
            ),
        ],
        code=os.path.join(os.path.dirname(__file__), "src", "register_model.py"),
        job_arguments=[
            "--model-info-path", "/opt/ml/processing/model_info/model_info.json",
            "--model-package-group", config["model_package_group"],
            "--evaluation-s3-uri", step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
            "--accuracy-threshold", str(config.get("accuracy_threshold", 0.5)),
            "--region", region,
            "--output-path", "/opt/ml/processing/output",
            "--sns-topic-arn", sns_topic_arn,
            "--notification-email", notification_email,
        ],
        depends_on=[step_evaluate.name],
    )

    # Step 6: Deploy Endpoint (with autoscaling)
    if config.get("enable_autoscaling", False):
        deploy_processor = SKLearnProcessor(
            framework_version="1.2-1",
            role=role,
            instance_type="ml.t3.medium",
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
                "--endpoint-name", config.get("endpoint_name", f"{config['project_name']}-endpoint"),
                "--instance-type", config["inference_instance_type"],
                "--role-arn", role,
                "--region", region,
                "--output-path", "/opt/ml/processing/output",
                "--min-capacity", str(config.get("autoscaling_min_capacity", 1)),
                "--max-capacity", str(config.get("autoscaling_max_capacity", 5)),
                "--sns-topic-arn", sns_topic_arn,
                "--notification-email", notification_email,
            ],
            depends_on=[step_register.name],
        )
        
        if_steps_list = [step_register, step_deploy]
    else:
        if_steps_list = [step_register]

    # Conditional step: Only register and deploy if accuracy >= threshold
    step_cond = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
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
    pipeline_steps = [
        step_data_source,
        step_preprocess,
        step_automl,
        step_create_model,
        step_evaluate,
        step_cond,
    ]

    pipeline = Pipeline(
        name=config["pipeline_name"],
        parameters=[accuracy_threshold],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session,
    )

    return pipeline


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

