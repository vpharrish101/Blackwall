terraform {
  required_version=">=1.5.0"
  required_providers {
    aws={
      source="hashicorp/aws"
      version="~>5.0"
    }
  }
}

provider "aws" {region="us-east-1"}
resource "aws_s3_bucket" "model_artifacts" {bucket="blackwall-model-artifacts"}
resource "aws_ecs_cluster" "blackwall_cluster"{name="blackwall-cluster"}

resource "aws_ecs_task_definition" "control_plane" {
  family="blackwall-control-plane"
  requires_compatibilities=["FARGATE"]
  network_mode="awsvpc"
  cpu="512"
  memory="1024"
  container_definitions=jsonencode([
    {
      name="blackwall"
      image="blackwall:latest"
      essential=true
      command=["python","run_blackwall.py"]
    }
  ])
}

resource "aws_sagemaker_training_job" "xgboost_child_training" {
  name="blackwall-child-training"
  algorithm_specification {
    training_image="xgboost"
    training_input_mode="File"
  }
  role_arn="arn:aws:iam::<account-id>:role/sagemaker-role"
}
