# GraphRAG++ Cloud Run Deployment Script
# Project: graphrag-plus-plus
# Run from: c:\Users\ethan\OneDrive\Desktop\GraphRAG\
# Requirements: gcloud CLI installed + logged in

$PROJECT  = "graphrag--plus-plus"
$BUCKET   = "graphrag--plus-plus-models"
$REGION   = "us-central1"
$IMAGE    = "gcr.io/$PROJECT/graphrag-inference"
$SERVICE  = "graphrag-inference"

Write-Host "=== GraphRAG++ Cloud Run Deployment ===" -ForegroundColor Cyan
Write-Host "Project : $PROJECT"
Write-Host "Region  : $REGION"
Write-Host "Image   : $IMAGE"

# Step 1: Set active project
Write-Host "`n[1/5] Setting GCP project..." -ForegroundColor Yellow
gcloud config set project $PROJECT

# Step 2: Enable required APIs
Write-Host "`n[2/5] Enabling APIs..." -ForegroundColor Yellow
gcloud services enable `
    cloudbuild.googleapis.com `
    run.googleapis.com `
    containerregistry.googleapis.com `
    storage.googleapis.com

# Step 3: Create GCS bucket (model will be uploaded from Colab)
Write-Host "`n[3/5] Creating GCS bucket gs://$BUCKET ..." -ForegroundColor Yellow
gcloud storage buckets create gs://$BUCKET `
    --location=$REGION `
    --project=$PROJECT
Write-Host "Bucket created. Upload your model from Colab before continuing."
Write-Host "Press Enter when model is uploaded to gs://$BUCKET/graphrag-model/ ..."
Read-Host

# Step 4: Build container image
Write-Host "`n[4/5] Building container image..." -ForegroundColor Yellow
gcloud builds submit inference/ `
    --tag $IMAGE `
    --project $PROJECT `
    --timeout 20m

# Step 5: Deploy to Cloud Run with L4 GPU
Write-Host "`n[5/5] Deploying to Cloud Run with L4 GPU..." -ForegroundColor Yellow
gcloud run deploy $SERVICE `
    --image $IMAGE `
    --region $REGION `
    --gpu 1 `
    --gpu-type nvidia-l4 `
    --memory 24Gi `
    --cpu 4 `
    --min-instances 0 `
    --max-instances 1 `
    --timeout 600 `
    --concurrency 1 `
    --set-env-vars "GCS_MODEL_PATH=gs://$BUCKET/graphrag-model" `
    --allow-unauthenticated `
    --project $PROJECT

# Get the service URL
Write-Host "`n=== Getting service URL ===" -ForegroundColor Cyan
$URL = gcloud run services describe $SERVICE `
    --region $REGION `
    --project $PROJECT `
    --format "value(status.url)"

Write-Host "`n✅ Deployment complete!" -ForegroundColor Green
Write-Host "Service URL: $URL"
Write-Host "`nAdd this to backend/.env:"
Write-Host "CLOUD_RUN_URL=$URL" -ForegroundColor Green
Write-Host "`nTest health endpoint:"
Write-Host "curl $URL/health"
