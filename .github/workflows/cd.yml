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
    id      = "archive-old-models"
    enabled = true

    expiration {
      days = 90
    }

    noncurrent_version_expiration {
      days = 60
    }
  }
}

# Firewall for Kubernetes cluster
resource "digitalocean_firewall" "k8s_firewall" {
  name = "${var.project_name}-k8s-firewall"

  droplet_ids = []
  tags        = ["k8s:${digitalocean_kubernetes_cluster.fintech_cluster.id}"]

  # Inbound rules
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.allowed_ssh_ips
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "6443"
    source_addresses = var.allowed_k8s_api_ips
  }

  # Outbound rules
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}

# Monitoring Droplet (optional)
resource "digitalocean_droplet" "monitoring" {
  count = var.enable_monitoring_droplet ? 1 : 0

  image    = "ubuntu-22-04-x64"
  name     = "${var.project_name}-monitoring"
  region   = var.region
  size     = var.monitoring_droplet_size
  vpc_uuid = digitalocean_vpc.fintech_vpc.id
  
  ssh_keys = var.ssh_key_fingerprints
  tags     = concat(var.common_tags, ["monitoring"])

  user_data = templatefile("${path.module}/scripts/monitoring-init.sh", {
    grafana_admin_password = var.grafana_admin_password
    domain_name           = var.domain_name
  })
}

# Volume for monitoring data
resource "digitalocean_volume" "monitoring_volume" {
  count = var.enable_monitoring_droplet ? 1 : 0

  region                  = var.region
  name                    = "${var.project_name}-monitoring-volume"
  size                    = var.monitoring_volume_size
  initial_filesystem_type = "ext4"
  tags                    = var.common_tags
}

# Attach volume to monitoring droplet
resource "digitalocean_volume_attachment" "monitoring_volume_attachment" {
  count = var.enable_monitoring_droplet ? 1 : 0

  droplet_id = digitalocean_droplet.monitoring[0].id
  volume_id  = digitalocean_volume.monitoring_volume[0].id
}

# Configure Kubernetes provider
provider "kubernetes" {
  host  = digitalocean_kubernetes_cluster.fintech_cluster.endpoint
  token = digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].token
  cluster_ca_certificate = base64decode(
    digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].cluster_ca_certificate
  )
}

# Configure Helm provider
provider "helm" {
  kubernetes {
    host  = digitalocean_kubernetes_cluster.fintech_cluster.endpoint
    token = digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].token
    cluster_ca_certificate = base64decode(
      digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].cluster_ca_certificate
    )
  }
}

# Kubernetes namespace
resource "kubernetes_namespace" "fintech" {
  metadata {
    name = var.project_name

    labels = {
      app     = var.project_name
      version = var.app_version
    }
  }
}

# Kubernetes secrets
resource "kubernetes_secret" "database_credentials" {
  metadata {
    name      = "database-credentials"
    namespace = kubernetes_namespace.fintech.metadata[0].name
  }

  data = {
    host     = digitalocean_database_cluster.postgres.private_host
    port     = digitalocean_database_cluster.postgres.port
    database = digitalocean_database_db.fintech_db.name
    username = digitalocean_database_user.fintech_user.name
    password = digitalocean_database_user.fintech_user.password
    uri      = digitalocean_database_cluster.postgres.private_uri
  }

  type = "Opaque"
}

resource "kubernetes_secret" "redis_credentials" {
  metadata {
    name      = "redis-credentials"
    namespace = kubernetes_namespace.fintech.metadata[0].name
  }

  data = {
    host     = digitalocean_database_cluster.redis.private_host
    port     = digitalocean_database_cluster.redis.port
    password = digitalocean_database_cluster.redis.password
    uri      = digitalocean_database_cluster.redis.private_uri
  }

  type = "Opaque"
}

resource "kubernetes_secret" "api_keys" {
  metadata {
    name      = "api-keys"
    namespace = kubernetes_namespace.fintech.metadata[0].name
  }

  data = {
    alpha_vantage_key    = var.alpha_vantage_api_key
    polygon_key          = var.polygon_api_key
    news_api_key         = var.news_api_key
    twitter_bearer_token = var.twitter_bearer_token
    jwt_secret_key       = var.jwt_secret_key
  }

  type = "Opaque"
}

# ConfigMap for application configuration
resource "kubernetes_config_map" "app_config" {
  metadata {
    name      = "app-config"
    namespace = kubernetes_namespace.fintech.metadata[0].name
  }

  data = {
    "config.yaml" = templatefile("${path.module}/templates/app-config.yaml", {
      environment           = var.environment
      log_level            = var.log_level
      redis_host           = digitalocean_database_cluster.redis.private_host
      postgres_host        = digitalocean_database_cluster.postgres.private_host
      enable_debug         = var.enable_debug
      max_workers          = var.max_workers
      model_storage_bucket = digitalocean_spaces_bucket.ml_models.name
    })
  }
}

# Helm chart for NGINX Ingress Controller
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.7.1"
  namespace  = "ingress-nginx"
  create_namespace = true

  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }

  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/do-loadbalancer-name"
    value = "${var.project_name}-nginx-lb"
  }

  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/do-loadbalancer-protocol"
    value = "http"
  }

  set {
    name  = "controller.metrics.enabled"
    value = "true"
  }

  depends_on = [digitalocean_kubernetes_cluster.fintech_cluster]
}

# Helm chart for Cert-Manager
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = "v1.12.0"
  namespace  = "cert-manager"
  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }

  set {
    name  = "global.leaderElection.namespace"
    value = "cert-manager"
  }

  depends_on = [digitalocean_kubernetes_cluster.fintech_cluster]
}

# Helm chart for Prometheus Monitoring
resource "helm_release" "prometheus" {
  count = var.enable_prometheus ? 1 : 0

  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  version    = "48.3.1"
  namespace  = "monitoring"
  create_namespace = true

  values = [
    templatefile("${path.module}/templates/prometheus-values.yaml", {
      grafana_admin_password = var.grafana_admin_password
      storage_class         = "do-block-storage"
      retention_days        = var.metrics_retention_days
    })
  ]

  depends_on = [digitalocean_kubernetes_cluster.fintech_cluster]
}

# Project outputs
output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = digitalocean_kubernetes_cluster.fintech_cluster.endpoint
  sensitive   = true
}

output "cluster_token" {
  description = "Kubernetes cluster token"
  value       = digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].token
  sensitive   = true
}

output "database_host" {
  description = "Database host"
  value       = digitalocean_database_cluster.postgres.private_host
  sensitive   = true
}

output "database_connection_string" {
  description = "Database connection string"
  value       = digitalocean_database_cluster.postgres.private_uri
  sensitive   = true
}

output "redis_host" {
  description = "Redis host"
  value       = digitalocean_database_cluster.redis.private_host
  sensitive   = true
}

output "container_registry_endpoint" {
  description = "Container registry endpoint"
  value       = digitalocean_container_registry.fintech_registry.endpoint
}

output "load_balancer_ip" {
  description = "Load balancer IP address"
  value       = digitalocean_loadbalancer.fintech_lb.ip
}

output "spaces_bucket_endpoint" {
  description = "Spaces bucket endpoint"
  value       = "https://${digitalocean_spaces_bucket.fintech_storage.bucket_domain_name}"
}

output "ml_models_bucket_endpoint" {
  description = "ML models bucket endpoint"
  value       = "https://${digitalocean_spaces_bucket.ml_models.bucket_domain_name}"
}

output "monitoring_droplet_ip" {
  description = "Monitoring droplet IP address"
  value       = var.enable_monitoring_droplet ? digitalocean_droplet.monitoring[0].ipv4_address : null
}

output "kubeconfig" {
  description = "Kubernetes config file content"
  value       = digitalocean_kubernetes_cluster.fintech_cluster.kube_config[0].raw_config
  sensitive   = true
}
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
