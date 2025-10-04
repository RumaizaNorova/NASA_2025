#!/bin/bash
# GCP Deployment Script for Shark Habitat Prediction

set -e

# Configuration
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="shark-habitat-prediction"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting GCP deployment for Shark Habitat Prediction..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Set project
echo "📋 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com

# Create service account if it doesn't exist
echo "👤 Creating service account..."
gcloud iam service-accounts create shark-habitat-service \
    --display-name="Shark Habitat Prediction Service" \
    --description="Service account for shark habitat prediction model" \
    || echo "Service account already exists"

# Grant necessary permissions
echo "🔐 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:shark-habitat-service@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:shark-habitat-service@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:shark-habitat-service@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create Cloud Storage buckets
echo "🪣 Creating Cloud Storage buckets..."
gsutil mb gs://$PROJECT_ID-shark-data || echo "Bucket already exists"
gsutil mb gs://$PROJECT_ID-shark-models || echo "Bucket already exists"

# Build and push Docker image
echo "🐳 Building and pushing Docker image..."
cd gcp_deployment
gcloud builds submit --tag $IMAGE_NAME .

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --service-account shark-habitat-service@$PROJECT_ID.iam.gserviceaccount.com

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo "✅ Service deployed successfully!"
echo "🌐 Service URL: $SERVICE_URL"

# Deploy to Vertex AI for training
echo "🤖 Setting up Vertex AI training..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="shark-habitat-training" \
    --config=vertex_ai_config.yaml

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Next steps:"
echo "1. Update environment variables in the deployed service"
echo "2. Upload your NASA Earthdata credentials"
echo "3. Test the model training pipeline"
echo "4. Monitor training progress in Vertex AI console"
