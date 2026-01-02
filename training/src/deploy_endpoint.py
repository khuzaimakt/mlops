#!/usr/bin/env python
"""Deploy SageMaker endpoint with autoscaling after model registration."""
import subprocess
import sys

# Install boto3 if not available (should be in container, but just in case)
try:
    import boto3
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "boto3"], check=True)
    import boto3

import os
import argparse
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SNS configuration (overridden in main() via CLI args when run in pipeline)
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")


def send_failure_notification(job_name: str, error_message: str) -> None:
    """Publish a failure notification to SNS."""
    try:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
        sns = boto3.client("sns", region_name=region)
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"[SageMaker Pipeline Failed] {job_name}",
            Message=(
                f"Step: {job_name}\n"
                f"Error: {error_message}\n"
                "See CloudWatch Logs for full traceback."
            ),
        )
        logger.info("Sent failure notification for %s", job_name)
    except Exception as notify_err:  # pragma: no cover - best-effort notification
        logger.error("Failed to send SNS notification: %s", notify_err)


def get_latest_approved_model(model_package_group_name, region='us-west-2'):
    """Get the latest approved model package ARN."""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    response = sagemaker_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response.get('ModelPackageSummaryList'):
        raise ValueError(f"No approved models found in {model_package_group_name}")
    
    return response['ModelPackageSummaryList'][0]['ModelPackageArn']


def create_or_update_endpoint(
    model_package_arn,
    endpoint_name,
    instance_type,
    initial_instance_count=1,
    role_arn=None,
    region='us-west-2'
):
    """Create or update a SageMaker endpoint."""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    # Check if endpoint exists
    endpoint_exists = False
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        logger.info(f"Endpoint {endpoint_name} already exists, will update it")
    except sagemaker_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] != 'ValidationException':
            raise
    
    # Create model from model package
    model_name = f"{endpoint_name}-model-{int(time.time())}"
    
    try:
        sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'ModelPackageName': model_package_arn,
            },
            ExecutionRoleArn=role_arn,
        )
        logger.info(f"Created model: {model_name}")
    except Exception as e:
        if 'AlreadyExistsException' not in str(type(e)):
            raise
        logger.warning(f"Model {model_name} already exists")
    
    # Create endpoint configuration
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"
    
    sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': initial_instance_count,
                'InitialVariantWeight': 1.0,
            }
        ],
    )
    logger.info(f"Created endpoint configuration: {endpoint_config_name}")
    
    # Create or update endpoint
    if endpoint_exists:
        # Update endpoint (blue/green deployment)
        logger.info(f"Updating endpoint {endpoint_name} with new configuration")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "updated"
    else:
        # Create new endpoint
        logger.info(f"Creating endpoint {endpoint_name}")
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
        action = "created"
    
    return {
        'endpoint_name': endpoint_name,
        'endpoint_config_name': endpoint_config_name,
        'model_name': model_name,
        'action': action
    }


def wait_for_endpoint_in_service(endpoint_name, region='us-west-2', max_wait_time=3600, check_interval=30):
    """Wait for endpoint to be in 'InService' status."""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    start_time = time.time()
    
    logger.info(f"Waiting for endpoint {endpoint_name} to be InService...")
    
    while True:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status == 'InService':
                logger.info(f"Endpoint {endpoint_name} is now InService")
                return True
            elif status in ['Failed', 'OutOfService']:
                raise RuntimeError(f"Endpoint {endpoint_name} is in {status} status")
            
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                raise TimeoutError(f"Endpoint {endpoint_name} did not reach InService status within {max_wait_time} seconds")
            
            logger.info(f"Endpoint status: {status} (waiting... {int(elapsed)}s elapsed)")
            time.sleep(check_interval)
            
        except Exception as e:
            if 'does not exist' in str(e).lower():
                # Endpoint doesn't exist yet (shouldn't happen, but handle gracefully)
                logger.warning(f"Endpoint {endpoint_name} does not exist yet")
                time.sleep(check_interval)
                continue
            raise


def get_autoscaling_service_role(region='us-west-2'):
    """Get the Application Auto Scaling service-linked role ARN for SageMaker."""
    sts_client = boto3.client('sts', region_name=region)
    iam_client = boto3.client('iam', region_name=region)
    account_id = sts_client.get_caller_identity()['Account']
    
    # The service-linked role name format for SageMaker Application Auto Scaling
    service_role_name = 'AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint'
    # Service-linked roles are in the root path, not under aws-service-role/
    role_arn = f'arn:aws:iam::{account_id}:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/{service_role_name}'
    
    try:
        # Try to get the role using the full ARN path
        # Service-linked roles use a specific path format
        role_path = f'aws-service-role/sagemaker.application-autoscaling.amazonaws.com/{service_role_name}'
        try:
            iam_client.get_role(RoleName=role_path)
            logger.info(f"Found service-linked role: {role_arn}")
            return role_arn
        except iam_client.exceptions.NoSuchEntityException:
            # Try without the path prefix (some accounts use different formats)
            try:
                iam_client.get_role(RoleName=service_role_name)
                # If found without path, construct ARN differently
                role_arn = f'arn:aws:iam::{account_id}:role/{service_role_name}'
                logger.info(f"Found service-linked role (alternative format): {role_arn}")
                return role_arn
            except iam_client.exceptions.NoSuchEntityException:
                # Role doesn't exist, try to create it
                try:
                    logger.info("Creating service-linked role for Application Auto Scaling...")
                    iam_client.create_service_linked_role(
                        AWSServiceName='sagemaker.application-autoscaling.amazonaws.com'
                    )
                    # Wait a bit for the role to be created
                    time.sleep(2)
                    logger.info(f"Created service-linked role: {role_arn}")
                    return role_arn
                except Exception as e:
                    if 'already exists' in str(e).lower() or 'InvalidInput' in str(type(e).__name__):
                        logger.info(f"Service-linked role already exists: {role_arn}")
                        return role_arn
                    else:
                        logger.warning(f"Could not create/get service-linked role: {e}")
                        return None
    except Exception as e:
        # If there's a validation error with the role name format, just return None
        # AWS will use the default service-linked role automatically
        logger.warning(f"Error checking service-linked role: {e}")
        logger.info("Will attempt autoscaling setup without explicit RoleARN (AWS will use default)")
        return None


def is_burstable_instance_type(instance_type):
    """Check if an instance type is burstable (t2, t3, t3a, t4g families).
    
    Burstable instance types cannot be used with autoscaling.
    """
    burstable_families = ['t2', 't3', 't3a', 't4g']
    instance_family = instance_type.split('.')[1] if '.' in instance_type else ''
    return any(instance_family.startswith(family) for family in burstable_families)


def get_endpoint_instance_type(endpoint_name, variant_name='AllTraffic', region='us-west-2'):
    """Get the instance type used by an endpoint variant."""
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = response['EndpointConfigName']
        
        config_response = sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        for variant in config_response.get('ProductionVariants', []):
            if variant.get('VariantName') == variant_name:
                return variant.get('InstanceType')
        
        # If variant not found, return first variant's instance type
        if config_response.get('ProductionVariants'):
            return config_response['ProductionVariants'][0].get('InstanceType')
        
        return None
    except Exception as e:
        logger.warning(f"Could not determine instance type: {e}")
        return None


def setup_autoscaling(endpoint_name, variant_name='AllTraffic', region='us-west-2', role_arn=None, min_capacity=1, max_capacity=5, instance_type=None):
    """Setup autoscaling for SageMaker endpoint."""
    # Check if instance type is burstable (autoscaling not supported)
    if instance_type is None:
        instance_type = get_endpoint_instance_type(endpoint_name, variant_name, region)
    
    if instance_type and is_burstable_instance_type(instance_type):
        logger.warning(f"Autoscaling cannot be configured for burstable instance type: {instance_type}")
        logger.warning("AWS does not support autoscaling on burstable instances (t2, t3, t3a, t4g families)")
        logger.warning("Please use a non-burstable instance type (e.g., ml.m5.xlarge, ml.c5.xlarge) to enable autoscaling")
        raise ValueError(f"Autoscaling not supported for burstable instance type: {instance_type}")
    
    application_autoscaling = boto3.client('application-autoscaling', region_name=region)
    
    resource_id = f'endpoint/{endpoint_name}/variant/{variant_name}'
    
    # Try to get service-linked role for Application Auto Scaling
    # This is the proper role to use, not the execution role
    autoscaling_role_arn = get_autoscaling_service_role(region)
    
    # If we couldn't get service-linked role, try without RoleARN (AWS will use default)
    use_role_arn = autoscaling_role_arn is not None
    
    # Register scalable target
    try:
        register_params = {
            'ServiceNamespace': 'sagemaker',
            'ResourceId': resource_id,
            'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount',
            'MinCapacity': min_capacity,
            'MaxCapacity': max_capacity,
        }
        
        # Only add RoleARN if we have the service-linked role
        if use_role_arn:
            register_params['RoleARN'] = autoscaling_role_arn
        
        application_autoscaling.register_scalable_target(**register_params)
        logger.info(f"Registered scalable target: {resource_id} (min={min_capacity}, max={max_capacity})")
    except Exception as e:
        error_msg = str(e).lower()
        error_code = e.response.get('Error', {}).get('Code', '') if hasattr(e, 'response') else ''
        
        if 'already registered' in error_msg or 'already exists' in error_msg:
            logger.info(f"Scalable target already registered: {resource_id}, updating...")
            # Update existing scalable target
            register_params = {
                'ServiceNamespace': 'sagemaker',
                'ResourceId': resource_id,
                'ScalableDimension': 'sagemaker:variant:DesiredInstanceCount',
                'MinCapacity': min_capacity,
                'MaxCapacity': max_capacity,
            }
            if use_role_arn:
                register_params['RoleARN'] = autoscaling_role_arn
            try:
                application_autoscaling.register_scalable_target(**register_params)
                logger.info("Updated scalable target")
            except Exception as e2:
                logger.warning(f"Could not update scalable target: {e2}")
        elif 'passrole' in error_msg or 'accessdenied' in error_msg or error_code == 'AccessDeniedException':
            # Permission issue - try without RoleARN (will use default service-linked role)
            logger.warning(f"Role permission issue ({error_msg}), trying without RoleARN...")
            try:
                application_autoscaling.register_scalable_target(
                    ServiceNamespace='sagemaker',
                    ResourceId=resource_id,
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                    MinCapacity=min_capacity,
                    MaxCapacity=max_capacity,
                    # Don't pass RoleARN - AWS will use the default service-linked role
                )
                logger.info(f"Registered scalable target without RoleARN: {resource_id}")
            except Exception as e2:
                logger.error(f"Failed to register scalable target even without RoleARN: {e2}")
                logger.warning("Autoscaling setup failed. Endpoint will be created without autoscaling.")
                logger.warning("You can manually set up autoscaling later or add IAM permissions.")
                raise
        else:
            logger.error(f"Failed to register scalable target: {e}")
            raise
    
    # Create scaling policy (target tracking)
    policy_name = f'{endpoint_name}-scaling-policy-{int(time.time())}'
    
    try:
        application_autoscaling.put_scaling_policy(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyName=policy_name,
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 1000.0,  # Target 1000 invocations per instance per minute
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance',
                },
                'ScaleInCooldown': 300,  # 5 minutes
                'ScaleOutCooldown': 60,  # 1 minute
            },
        )
        logger.info(f"Created scaling policy: {policy_name}")
    except Exception as e:
        if 'already exists' not in str(e).lower():
            raise
        logger.warning(f"Scaling policy might already exist")
    
    return {
        'resource_id': resource_id,
        'policy_name': policy_name
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-package-group", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--instance-type", type=str, default="ml.t2.medium")
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--output-path", type=str, default="/opt/ml/processing/output")
    parser.add_argument("--min-capacity", type=int, default=1)
    parser.add_argument("--max-capacity", type=int, default=5)
    parser.add_argument("--sns-topic-arn", type=str, default=os.environ.get("SNS_TOPIC_ARN"))
    parser.add_argument("--notification-email", type=str, default=os.environ.get("NOTIFICATION_EMAIL"))
    
    args = parser.parse_args()

    # Use SNS configuration from arguments or environment
    sns_topic_arn = args.sns_topic_arn or SNS_TOPIC_ARN
    notification_email = args.notification_email or NOTIFICATION_EMAIL

    try:
        logger.info("Getting latest approved model...")
        model_package_arn = get_latest_approved_model(args.model_package_group, args.region)
        logger.info(f"Using model package: {model_package_arn}")
        
        logger.info("Creating/updating endpoint...")
        deployment_result = create_or_update_endpoint(
            model_package_arn=model_package_arn,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            initial_instance_count=1,
            role_arn=args.role_arn,
            region=args.region
        )
        
        # Wait for endpoint to be InService before setting up autoscaling
        # This is required because autoscaling can only be registered when endpoint is InService
        try:
            wait_for_endpoint_in_service(args.endpoint_name, region=args.region)
        except Exception as e:
            logger.warning(f"Could not wait for endpoint to be InService: {e}")
            logger.warning("Will attempt autoscaling setup anyway (may fail if endpoint is still updating)")
        
        # Setup autoscaling (non-blocking - endpoint will still be created if this fails)
        autoscaling_result = None
        try:
            logger.info("Setting up autoscaling...")
            autoscaling_result = setup_autoscaling(
                endpoint_name=args.endpoint_name,
                variant_name='AllTraffic',
                region=args.region,
                role_arn=args.role_arn,
                min_capacity=args.min_capacity,
                max_capacity=args.max_capacity,
                instance_type=args.instance_type
            )
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Autoscaling setup failed: {error_msg}")
            
            # Provide specific guidance based on error type
            if 'burstable' in error_msg.lower() or 't2' in error_msg.lower() or 't3' in error_msg.lower():
                logger.warning("Endpoint was created successfully, but autoscaling was not configured.")
                logger.warning("Reason: Burstable instance types (t2, t3, t3a, t4g) do not support autoscaling.")
                logger.warning("Solution: Change inference_instance_type in pipeline_config.yaml to a non-burstable type (e.g., ml.m5.xlarge, ml.c5.xlarge)")
            else:
                logger.warning("Endpoint was created successfully, but autoscaling was not configured.")
                logger.warning("You can manually configure autoscaling later or fix IAM permissions and re-run deployment.")
            
            autoscaling_result = {
                'status': 'failed',
                'error': error_msg,
                'note': 'Endpoint created but autoscaling not configured'
            }
        
        # Save deployment results
        results = {
            'status': 'success',
            'model_package_arn': model_package_arn,
            'deployment': deployment_result,
            'autoscaling': autoscaling_result,
            'endpoint_name': args.endpoint_name
        }
        
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, "deployment.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Deployment successful! Results saved to {output_file}")
        logger.info(f"Endpoint: {args.endpoint_name}")
        logger.info(f"Action: {deployment_result['action']}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        send_failure_notification("Deploy Endpoint - Classification Model Training Pipeline", str(e))
        raise

