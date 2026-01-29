# AWS Infrastructure (Conceptual)

This directory contains Terraform definitions that model how the Blackwall system would be deployed on AWS.

These resources are not applied due to cost constraints.
The goal is to demonstrate infrastructure design and ML lifecycle alignment, not live cloud deployment.


## Architecture Mapping

- Control Plane: ECS Fargate service running `main.py`
- Training Jobs: SageMaker Training Jobs (XGBoost)
- Model Artifacts: S3
- Triggers: EventBridge (not shown)
- Logging: CloudWatch (not shown)
- Resources: We allocate 0.5 vCPU and 1GB of RAM
These definitions are intentionally minimal and illustrative. 
