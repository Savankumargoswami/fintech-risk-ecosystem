#!/bin/bash

# Autonomous Financial Risk Management System - Deployment Script
# This script automates the complete deployment process to DigitalOcean

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fintech-risk-ecosystem"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure/terraform"
KUBERNETES_DIR="$PROJECT_ROOT/infrastructure/kubernetes"

# Default values
ENVIRONMENT="production"
FORCE_REBUILD=false
SKIP_TESTS=false
DEPLOY_INFRASTRUCTURE=true
DEPLOY_APPLICATION=true
ENABLE_MONITORING=true

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deployment script for Autonomous Financial Risk Management System

OPTIONS:
    -e, --environment ENV       Deployment environment (default: production)
    -f, --force-rebuild        Force rebuild of Docker images
    -s, --skip-tests           Skip running tests before deployment
    --no-infrastructure        Skip infrastructure deployment
    --no-application          Skip application deployment
    --no-monitoring           Skip monitoring setup
    -h, --help                Show this help message

ENVIRONMENT VARIABLES:
    DIGITALOCEAN_TOKEN         DigitalOcean API token (required)
    SPACES_ACCESS_KEY         DigitalOcean Spaces access key (required)
    SPACES_SECRET_KEY         DigitalOcean Spaces secret key (required)
    ALPHA_VANTAGE_API_KEY     Alpha Vantage API key (required)
    POLYGON_API_KEY           Polygon API key (required)
    NEWS_API_KEY              News API key (required)
    TWITTER_BEARER_TOKEN      Twitter Bearer token (required)
    JWT_SECRET_KEY            JWT secret key (required)
    GRAFANA_ADMIN_PASSWORD    Grafana admin password (required)
    DOMAIN_NAME               Domain name for the application (required)

EXAMPLES:
    # Full deployment
    $0

    # Skip tests and force rebuild
    $0 --skip-tests --force-rebuild

    # Deploy only application (infrastructure already exists)
    $0 --no-infrastructure

    # Deploy to staging environment
    $0 --environment staging

EOF
}

check_dependencies() {
    log_success "All dependencies are installed"
}

check_environment_variables() {
    log_info "Checking required environment variables..."
    
    local required_vars=(
        "DIGITALOCEAN_TOKEN"
        "SPACES_ACCESS_KEY"
        "SPACES_SECRET_KEY"
        "ALPHA_VANTAGE_API_KEY"
        "POLYGON_API_KEY"
        "NEWS_API_KEY"
        "TWITTER_BEARER_TOKEN"
        "JWT_SECRET_KEY"
        "GRAFANA_ADMIN_PASSWORD"
        "DOMAIN_NAME"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing environment variables: ${missing_vars[*]}"
        log_info "Please set all required environment variables and try again."
        exit 1
    fi
    
    log_success "All required environment variables are set"
}

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements/testing.txt
    
    # Run linting
    log_info "Running code linting..."
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    black --check .
    isort --check-only .
    
    # Run unit tests
    log_info "Running unit tests..."
    pytest tests/unit/ -v --cov=. --cov-report=term-missing
    
    # Run integration tests
    log_info "Running integration tests..."
    pytest tests/integration/ -v
    
    log_success "All tests passed"
}

build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Configure DigitalOcean Container Registry
    log_info "Configuring DigitalOcean Container Registry..."
    doctl registry login
    
    # Get registry name
    REGISTRY_NAME=$(doctl registry get --format Name --no-header 2>/dev/null || echo "")
    if [ -z "$REGISTRY_NAME" ]; then
        log_info "Creating container registry..."
        doctl registry create "$PROJECT_NAME-registry" --subscription-tier starter
        REGISTRY_NAME="$PROJECT_NAME-registry"
    fi
    
    # Build main application image
    log_info "Building main application image..."
    IMAGE_TAG="registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME:$(git rev-parse --short HEAD)"
    
    docker build \
        --target production \
        --tag "$IMAGE_TAG" \
        --tag "registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME:latest" \
        --file infrastructure/docker/Dockerfile \
        .
    
    # Push images
    log_info "Pushing images to registry..."
    docker push "$IMAGE_TAG"
    docker push "registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME:latest"
    
    # Build ML worker image
    log_info "Building ML worker image..."
    WORKER_IMAGE_TAG="registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME-worker:$(git rev-parse --short HEAD)"
    
    docker build \
        --tag "$WORKER_IMAGE_TAG" \
        --tag "registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME-worker:latest" \
        --file infrastructure/docker/Dockerfile.worker \
        .
    
    docker push "$WORKER_IMAGE_TAG"
    docker push "registry.digitalocean.com/$REGISTRY_NAME/$PROJECT_NAME-worker:latest"
    
    log_success "Docker images built and pushed successfully"
    
    # Export variables for later use
    export IMAGE_TAG
    export WORKER_IMAGE_TAG
    export REGISTRY_NAME
}

deploy_infrastructure() {
    if [ "$DEPLOY_INFRASTRUCTURE" = false ]; then
        log_warning "Skipping infrastructure deployment as requested"
        return 0
    fi
    
    log_info "Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    log_info "Initializing Terraform..."
    terraform init \
        -backend-config="access_key=$SPACES_ACCESS_KEY" \
        -backend-config="secret_key=$SPACES_SECRET_KEY"
    
    # Create terraform.tfvars
    cat > terraform.tfvars << EOF
digitalocean_token = "$DIGITALOCEAN_TOKEN"
spaces_access_key = "$SPACES_ACCESS_KEY"
spaces_secret_key = "$SPACES_SECRET_KEY"
alpha_vantage_api_key = "$ALPHA_VANTAGE_API_KEY"
polygon_api_key = "$POLYGON_API_KEY"
news_api_key = "$NEWS_API_KEY"
twitter_bearer_token = "$TWITTER_BEARER_TOKEN"
jwt_secret_key = "$JWT_SECRET_KEY"
grafana_admin_password = "$GRAFANA_ADMIN_PASSWORD"
domain_name = "$DOMAIN_NAME"
environment = "$ENVIRONMENT"
project_name = "$PROJECT_NAME"
enable_monitoring_droplet = $ENABLE_MONITORING
EOF
    
    # Plan Terraform changes
    log_info "Planning Terraform changes..."
    terraform plan -out=tfplan
    
    # Apply Terraform changes
    log_info "Applying Terraform changes..."
    terraform apply -auto-approve tfplan
    
    # Save outputs
    terraform output -json > terraform_outputs.json
    
    log_success "Infrastructure deployment completed"
}

setup_kubernetes() {
    log_info "Setting up Kubernetes configuration..."
    
    cd "$TERRAFORM_DIR"
    
    # Get cluster name from Terraform outputs
    CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "$PROJECT_NAME-cluster")
    
    # Configure kubectl
    log_info "Configuring kubectl..."
    doctl kubernetes cluster kubeconfig save "$CLUSTER_NAME"
    
    # Verify cluster connectivity
    kubectl cluster-info
    
    log_success "Kubernetes configuration completed"
}

deploy_application() {
    if [ "$DEPLOY_APPLICATION" = false ]; then
        log_warning "Skipping application deployment as requested"
        return 0
    fi
    
    log_info "Deploying application to Kubernetes..."
    
    cd "$KUBERNETES_DIR"
    
    # Create namespace
    kubectl apply -f namespace.yaml
    
    # Create secrets
    log_info "Creating Kubernetes secrets..."
    kubectl create secret generic database-credentials \
        --from-literal=host="$(cd "$TERRAFORM_DIR" && terraform output -raw database_host)" \
        --from-literal=port="$(cd "$TERRAFORM_DIR" && terraform output -raw database_port)" \
        --from-literal=database="fintech_risk" \
        --from-literal=username="$(cd "$TERRAFORM_DIR" && terraform output -raw database_username)" \
        --from-literal=password="$(cd "$TERRAFORM_DIR" && terraform output -raw database_password)" \
        --namespace="$PROJECT_NAME" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic redis-credentials \
        --from-literal=host="$(cd "$TERRAFORM_DIR" && terraform output -raw redis_host)" \
        --from-literal=port="$(cd "$TERRAFORM_DIR" && terraform output -raw redis_port)" \
        --from-literal=password="$(cd "$TERRAFORM_DIR" && terraform output -raw redis_password)" \
        --namespace="$PROJECT_NAME" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic api-keys \
        --from-literal=alpha_vantage_key="$ALPHA_VANTAGE_API_KEY" \
        --from-literal=polygon_key="$POLYGON_API_KEY" \
        --from-literal=news_api_key="$NEWS_API_KEY" \
        --from-literal=twitter_bearer_token="$TWITTER_BEARER_TOKEN" \
        --from-literal=jwt_secret_key="$JWT_SECRET_KEY" \
        --namespace="$PROJECT_NAME" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image tags in deployment manifests
    log_info "Updating image tags in manifests..."
    sed -i.bak "s|{{IMAGE_TAG}}|${IMAGE_TAG}|g" deployment.yaml
    sed -i.bak "s|{{WORKER_IMAGE_TAG}}|${WORKER_IMAGE_TAG}|g" deployment.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f configmap.yaml
    kubectl apply -f deployment.yaml
    kubectl apply -f service.yaml
    kubectl apply -f ingress.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/fintech-risk-api -n "$PROJECT_NAME" --timeout=600s
    
    # Run database migrations
    log_info "Running database migrations..."
    kubectl run migration-job \
        --image="$IMAGE_TAG" \
        --rm -i --restart=Never \
        --namespace="$PROJECT_NAME" \
        -- python -m alembic upgrade head
    
    log_success "Application deployment completed"
}

setup_monitoring() {
    if [ "$ENABLE_MONITORING" = false ]; then
        log_warning "Skipping monitoring setup as requested"
        return 0
    fi
    
    log_info "Setting up monitoring and observability..."
    
    # Install Prometheus and Grafana using Helm
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install kube-prometheus-stack
    helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set grafana.adminPassword="$GRAFANA_ADMIN_PASSWORD" \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --wait
    
    # Install ELK stack for logging
    helm repo add elastic https://helm.elastic.co
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace logging \
        --create-namespace \
        --set replicas=1 \
        --set minimumMasterNodes=1 \
        --set resources.requests.memory=1Gi \
        --set resources.limits.memory=2Gi \
        --wait
    
    helm upgrade --install kibana elastic/kibana \
        --namespace logging \
        --set service.type=ClusterIP \
        --wait
    
    log_success "Monitoring setup completed"
}

run_health_checks() {
    log_info "Running post-deployment health checks..."
    
    # Wait a bit for services to stabilize
    sleep 30
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$PROJECT_NAME" -o wide
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n "$PROJECT_NAME"
    
    # Check ingress status
    log_info "Checking ingress status..."
    kubectl get ingress -n "$PROJECT_NAME"
    
    # Get application logs
    log_info "Checking application logs..."
    kubectl logs -l app=fintech-risk-api -n "$PROJECT_NAME" --tail=20
    
    # Test API health endpoint
    LOAD_BALANCER_IP=$(cd "$TERRAFORM_DIR" && terraform output -raw load_balancer_ip)
    log_info "Testing API health endpoint..."
    
    for i in {1..10}; do
        if curl -f "http://$LOAD_BALANCER_IP/api/v1/health" > /dev/null 2>&1; then
            log_success "API health check passed"
            break
        fi
        
        if [ $i -eq 10 ]; then
            log_error "API health check failed after 10 attempts"
            return 1
        fi
        
        log_info "Health check attempt $i failed, retrying in 10 seconds..."
        sleep 10
    done
    
    log_success "All health checks passed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    cd "$TERRAFORM_DIR"
    rm -f tfplan terraform.tfvars
    cd "$KUBERNETES_DIR"
    rm -f *.bak
}

main() {
    log_info "Starting deployment of $PROJECT_NAME to $ENVIRONMENT environment"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --no-infrastructure)
                DEPLOY_INFRASTRUCTURE=false
                shift
                ;;
            --no-application)
                DEPLOY_APPLICATION=false
                shift
                ;;
            --no-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Validation
    check_dependencies
    check_environment_variables
    
    # Pre-deployment steps
    run_tests
    build_and_push_images
    
    # Infrastructure deployment
    deploy_infrastructure
    setup_kubernetes
    
    # Application deployment
    deploy_application
    setup_monitoring
    
    # Post-deployment verification
    run_health_checks
    cleanup
    
    # Final success message
    log_success "🎉 Deployment completed successfully!"
    log_info "Access your application at: https://$DOMAIN_NAME"
    log_info "Grafana dashboard: https://$DOMAIN_NAME/grafana"
    log_info "Kibana dashboard: https://$DOMAIN_NAME/kibana"
    log_info "API documentation: https://$DOMAIN_NAME/api/docs"
    
    # Display useful information
    echo ""
    echo "=== Deployment Summary ==="
    echo "Environment: $ENVIRONMENT"
    echo "Image Tag: $IMAGE_TAG"
    echo "Cluster: $CLUSTER_NAME"
    echo "Load Balancer IP: $LOAD_BALANCER_IP"
    echo "=========================="
    echo ""
    echo "Next steps:"
    echo "1. Configure DNS to point $DOMAIN_NAME to $LOAD_BALANCER_IP"
    echo "2. Set up SSL certificate (handled automatically by cert-manager)"
    echo "3. Configure monitoring alerts"
    echo "4. Set up backup schedules"
    echo "5. Review and tune resource limits"
}

# Error handling
trap cleanup EXIT
set -e

# Run main function
main "$@"
info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required tools
    for tool in docker terraform kubectl doctl helm; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing tools and try again."
        exit 1
    fi
    
    log_
