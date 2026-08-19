# Build Docker Images and Deploy to Minikube

Quick guide to build images and deploy for testing.

## 1. Build Docker Images

**Note:** Each service uses its own requirements file:
- `Dockerfile.api` → uses `requirements-api.txt`
- `Dockerfile.mock-llm` → uses `requirements-mock-llm.txt`
- `Dockerfile.llm` → uses `requirements-llm.txt` (for GPU-enabled LLM service)

The Dockerfiles handle requirements installation automatically.

### ⚠️ Important: Docker Build vs Minikube Docker Daemon

**Current approach (Docker Desktop + Minikube):**
```powershell
# Build with Docker Desktop
docker build -f Dockerfile.api -t insightaid-api:test .
docker build -f Dockerfile.mock-llm -t mock-llm-service:test .

# Load into Minikube
minikube image load insightaid-api:test
minikube image load mock-llm-service:test
```

**Alternative approach (Minikube Docker daemon - don't mix with above):**
```powershell
# Point to Minikube's Docker daemon
minikube docker-env | Invoke-Expression
# Or for PowerShell:
& minikube -p minikube docker-env --shell powershell | Invoke-Expression

# Build directly into Minikube (no need for minikube image load)
docker build -f Dockerfile.api -t insightaid-api:test .
docker build -f Dockerfile.mock-llm -t mock-llm-service:test .
```

👉 **Pick one method and stick with it. Don't mix both approaches.**

### Build API Image
```powershell
docker build -f Dockerfile.api -t insightaid-api:test .
```

### Build Mock LLM Image
```powershell
docker build -f Dockerfile.mock-llm -t mock-llm-service:test .
```

## 2. Load Images into Minikube
# Start Minikube
minikube start --driver=docker

```powershell
# Make sure minikube is running
# minikube start

# Load images (only if using Docker Desktop approach above)
minikube image load insightaid-api:test
minikube image load mock-llm-service:test

# Verify
minikube image ls | Select-String "insightaid-api|mock-llm"
```

## 3. Update Kubernetes Deployments

Update image names in deployment files:

**kubernetes/api-deployment.yaml:**
```yaml
image: insightaid-api:test  # Change from :latest
imagePullPolicy: Never  # Use local image for Minikube testing
```

**kubernetes/llm-deployment-mock.yaml:**
```yaml
image: mock-llm-service:test  # Change from :latest
imagePullPolicy: Never  # Use local image for Minikube testing
```

**⚠️ imagePullPolicy Notes:**
- `Never`: Use for Minikube local testing (current setup)
- `IfNotPresent`: Use for CI/shared clusters where images may be in a registry
- Change to `IfNotPresent` when moving to production/CI environments

## 4. Deploy to Minikube

### Option A: Use the Script (Recommended)
```powershell
.\deploy-minikube-test.ps1
```

### Option B: Manual Deployment
```powershell
# Create namespace 
# Option A: single line
kubectl create namespace insightaid

#Option B: (idempotent - safe to run multiple times)
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: insightaid
EOF

# Deploy Qdrant (deployment + service)
kubectl apply -f kubernetes/qdrant-deployment.yaml -n insightaid
kubectl apply -f kubernetes/qdrant-service.yaml -n insightaid

kubectl get pods -n insightaid -w

# Deploy Mock LLM
kubectl apply -f kubernetes/llm-deployment-mock.yaml -n insightaid
kubectl apply -f kubernetes/llm-service.yaml -n insightaid

kubectl get pods -n insightaid -w

# Create ConfigMap (used by api-deployment.yaml via envFrom)

kubectl delete configmap api-config -n insightaid --ignore-not-found


kubectl create configmap api-config -n insightaid `
    --from-literal=EMBEDDING_MODEL=all-MiniLM-L6-v2 `
    --from-literal=EMBEDDING_DIM=384 `
    --from-literal=LLM_SERVICE_URL=http://llm-service:8001 `
    --from-literal=QDRANT_HOST=qdrant `
    --dry-run=client -o yaml | kubectl apply -f -

# Deploy API (ConfigMap is automatically loaded via envFrom in deployment)
kubectl apply -f kubernetes/api-deployment.yaml -n insightaid
kubectl apply -f kubernetes/api-service.yaml -n insightaid

kubectl get pods -n insightaid -w
```

## Verify inside cluster API - Qdrant
kubectl exec -it deployment/rag-api -n insightaid -- python -c "
from qdrant_client import QdrantClient
c = QdrantClient(host='qdrant', port=6333)
print(c.get_collections())
"

## Verify inside cluster API - LLM
kubectl exec -it deployment/rag-api -n insightaid -- python -c "
import httpx
print(httpx.get('http://llm-service:8001/health').json())
"

**Note:** Environment variables are loaded from ConfigMap via `envFrom` in `api-deployment.yaml`. No need for `kubectl set env`.

## 5. Access Services

### Port Forward
```powershell
# API Service
# Service port: 80 -> Container port: 3000 (see api-service.yaml)
kubectl port-forward svc/rag-api 3000:80 -n insightaid

# In another terminal, test:
Invoke-RestMethod http://localhost:3000/api/health
```

**Service Port Configuration:**
- `api-service.yaml` exposes port `80` (external)
- Targets container port `3000` (where FastAPI runs)
- Port forwarding: `3000:80` maps localhost:3000 → service:80 → container:3000

### Check Status
```powershell
# Pods
kubectl get pods -n insightaid

# Services
kubectl get svc -n insightaid

# Logs
kubectl logs -l app=rag-api -n insightaid --tail=50 -f
```

## 6. Test the Combined Endpoint

```powershell
# With files
$form = @{
    files = Get-Item "test.pdf"
    query = "What is this document about?"
}
Invoke-RestMethod -Uri "http://localhost:3000/api/upload-and-query" -Method Post -Form $form

# Without files (query existing session)
$form = @{
    query = "What is the damage limit?"
    session_id = "your-session-id"
}
Invoke-RestMethod -Uri "http://localhost:3000/api/upload-and-query" -Method Post -Form $form
```

## Troubleshooting

1. **Image not found:**
   - Check images are loaded: `minikube image ls`
   - Verify imagePullPolicy: Never
   - Rebuild and reload images

2. **Pods not starting:**
   - Check logs: `kubectl describe pod <pod-name> -n insightaid`
   - Verify ConfigMap exists: `kubectl get configmap -n insightaid`

3. **Connection errors:**
   - Verify service names match
   - Check pods are ready: `kubectl get pods -n insightaid`
   - **Ensure Qdrant Service exists:** `kubectl get svc qdrant -n insightaid`
   - Test service connectivity: `kubectl exec -it <pod-name> -n insightaid -- curl http://qdrant:6333`
   - If Qdrant DNS fails, verify `qdrant-service.yaml` is deployed (required for DNS resolution)