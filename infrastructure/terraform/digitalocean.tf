# DigitalOcean Infrastructure for Autonomous Financial Risk Management System
terraform {
  required_version = ">= 1.0"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Configure DigitalOcean Provider
provider "digitalocean" {
  token = var.digitalocean_token
}

# Data sources
data "digitalocean_kubernetes_versions" "main" {
  version_prefix = "1.28."
}

data "digitalocean_regions" "available" {}

# VPC
resource "digitalocean_vpc" "fintech_vpc" {
  name     = "${var.project_name}-vpc"
  region   = var.region
  ip_range = "10.10.0.0/16"

  tags = var.common_tags
}

# Kubernetes Cluster
resource "digitalocean_kubernetes_cluster" "fintech_cluster" {
  name    = "${var.project_name}-cluster"
  region  = var.region
  version = data.digitalocean_kubernetes_versions.main.latest_version
  vpc_uuid = digitalocean_vpc.fintech_vpc.id

  # Maintenance window
  maintenance_policy {
    start_time = "04:00"
    day        = "sunday"
  }

  # Node pool configuration
  node_pool {
    name       = "worker-pool"
    size       = var.node_size
    node_count = var.min_nodes
    auto_scale = true
    min_nodes  = var.min_nodes
    max_nodes  = var.max_nodes

    tags = var.common_tags

    # Node labels
    labels = {
      "node-type" = "worker"
      "app"       = var.project_name
    }
  }

  tags = var.common_tags
}

# Additional node pool for ML workloads
resource "digitalocean_kubernetes_node_pool" "ml_pool" {
  cluster_id = digitalocean_kubernetes_cluster.fintech_cluster.id
  name       = "ml-pool"
  size       = var.ml_node_size
  node_count = var.ml_min_nodes
  auto_scale = true
  min_nodes  = var.ml_min_nodes
  max_nodes  = var.ml_max_nodes

  labels = {
    "node-type" = "ml-worker"
    "workload"  = "machine-learning"
  }

  taint {
    key    = "ml-workload"
    value  = "true"
    effect = "NoSchedule"
  }

  tags = var.common_tags
}

# Managed Database - PostgreSQL
resource "digitalocean_database_cluster" "postgres" {
  name       = "${var.project_name}-postgres"
  engine     = "pg"
  version    = "14"
  size       = var.database_size
  region     = var.region
  node_count = var.database_nodes
  private_network_uuid = digitalocean_vpc.fintech_vpc.id

  tags = var.common_tags

  maintenance_window {
    day  = "sunday"
    hour = "05:00:00"
  }
}

# Database
resource "digitalocean_database_db" "fintech_db" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "fintech_risk"
}

# Database User
resource "digitalocean_database_user" "fintech_user" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = "fintech_user"
}

# Managed Database - Redis
resource "digitalocean_database_cluster" "redis" {
  name       = "${var.project_name}-redis"
  engine     = "redis"
  version    = "7"
  size       = var.redis_size
  region     = var.region
  node_count = 1
  private_network_uuid = digitalocean_vpc.fintech_vpc.id

  tags = var.common_tags

  maintenance_window {
    day  = "sunday"
    hour = "05:30:00"
  }
}

# Load Balancer
resource "digitalocean_loadbalancer" "fintech_lb" {
  name   = "${var.project_name}-lb"
  region = var.region
  vpc_uuid = digitalocean_vpc.fintech_vpc.id

  forwarding_rule {
    entry_protocol  = "http"
    entry_port      = 80
    target_protocol = "http"
    target_port     = 80
  }

  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 80
    certificate_name = digitalocean_certificate.fintech_cert.name
  }

  healthcheck {
    protocol   = "http"
    port       = 80
    path       = "/api/v1/health"
    check_interval_seconds   = 10
    response_timeout_seconds = 5
    unhealthy_threshold     = 3
    healthy_threshold       = 5
  }

  droplet_tag = "fintech-web"
  tags = var.common_tags
}

# SSL Certificate
resource "digitalocean_certificate" "fintech_cert" {
  name    = "${var.project_name}-cert"
  type    = "lets_encrypt"
  domains = [var.domain_name]

  lifecycle {
    create_before_destroy = true
  }
}

# Container Registry
resource "digitalocean_container_registry" "fintech_registry" {
  name                   = "${var.project_name}registry"
  subscription_tier_slug = var.registry_tier
  region                 = var.region
}

# Spaces (Object Storage)
resource "digitalocean_spaces_bucket" "fintech_storage" {
  name   = "${var.project_name}-storage"
  region = var.region
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "delete-old-versions"
    enabled = true

    noncurrent_version_expiration {
      days = 30
    }
  }
}

# Spaces bucket for ML models
resource "digitalocean_spaces_bucket" "ml_models" {
  name   = "${var.project_name}-ml-models"
  region = var.region
  acl    = "private"

  versioning {
    enabled = true
  }

  lifecycle_rule {
