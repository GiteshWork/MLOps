terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.45.0" # Pinning to a specific, known-good version
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.23.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "4.0.5"
    }
  }
}

provider "aws" {
  region = "us-east-1" # You can change this to your preferred region
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.21.0" # Using the stable module version

  cluster_name    = "my-eks-cluster"
  cluster_version = "1.29"

  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    main = {
      min_size     = 1
      max_size     = 3
      desired_size = 2

      instance_types = ["t3.medium"]
    }
  }
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "region" {
  value = "us-east-1" # Must match the provider region
}