# GCP Deployment for Shark Habitat Prediction

This directory contains all the necessary files to deploy the Shark Habitat Prediction model to Google Cloud Platform.

## 🚀 Quick Start

1. **Set up GCP project and credentials**
   ```bash
   # Install gcloud CLI
   curl https://sdk.cloud.google.com | bash
   
   # Authenticate
   gcloud auth login
   gcloud auth application-default login
   
   # Set project
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   gcloud config set project $GOOGLE_CLOUD_PROJECT
   ```

2. **Update configuration**
   ```bash
   # Edit deploy.sh and update PROJECT_ID
   nano gcp_deployment/deploy.sh
   
   # Update vertex_ai_config.yaml with your project ID
   nano gcp_deployment/vertex_ai_config.yaml
   ```

3. **Deploy to GCP**
   ```bash
   chmod +x gcp_deployment/deploy.sh
   ./gcp_deployment/deploy.sh
   ```

## 📁 File Structure

```
gcp_deployment/
├── Dockerfile                 # Container configuration
├── cloudbuild.yaml           # Cloud Build configuration
├── vertex_ai_config.yaml     # Vertex AI training configuration
├── deploy.sh                 # Deployment script
├── gcp_train.py              # GCP-optimized training script
├── requirements_gcp.txt      # GCP-specific dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

Set these environment variables before deployment:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GCS_BUCKET="your-project-id-shark-models"
export GCP_REGION="us-central1"
export EARTHDATA_USERNAME="your-nasa-username"
export EARTHDATA_PASSWORD="your-nasa-password"
```

### Service Account Permissions

The deployment script creates a service account with these roles:
- `roles/storage.admin` - For Cloud Storage access
- `roles/bigquery.dataEditor` - For BigQuery access
- `roles/aiplatform.user` - For Vertex AI access

## 🏗️ Architecture

### Cloud Run
- **Purpose**: Web API for real-time predictions
- **Memory**: 4GB
- **CPU**: 2 cores
- **Timeout**: 1 hour
- **Max Instances**: 10

### Vertex AI
- **Purpose**: Model training with GPUs
- **Machine Type**: n1-standard-8
- **GPU**: NVIDIA_TESLA_T4
- **Memory**: 32GB
- **Timeout**: 2 hours

### Cloud Storage
- **Data Bucket**: `your-project-id-shark-data`
- **Models Bucket**: `your-project-id-shark-models`

## 📊 Training Pipeline

The GCP training pipeline:

1. **Download Data**: Downloads training data from Cloud Storage
2. **Load Features**: Loads NASA oceanographic features
3. **Prepare Features**: Creates feature matrix with temporal validation
4. **Train Model**: Trains Random Forest with temporal cross-validation
5. **Save Model**: Uploads trained model to Cloud Storage

### Key Features

- ✅ **100% Real NASA Data**: Uses actual satellite data
- ✅ **Temporal Validation**: Prevents data leakage
- ✅ **Scalable Training**: Runs on Vertex AI with GPUs
- ✅ **Model Versioning**: Stores models in Cloud Storage
- ✅ **Monitoring**: Integrated with GCP monitoring

## 🔍 Monitoring

### Cloud Logging
- Training logs are automatically captured
- Search for `shark-habitat-prediction` in Cloud Logging

### Cloud Monitoring
- Model performance metrics
- Training job status
- Resource utilization

### Vertex AI Console
- Training job progress
- Model evaluation metrics
- Hyperparameter tuning results

## 💰 Cost Estimation

### Development Phase (3 months)
- **Cloud Run**: ~$50/month
- **Vertex AI**: ~$200/month
- **Cloud Storage**: ~$20/month
- **Total**: ~$270/month

### Production Phase (1 year)
- **Cloud Run**: ~$500/month
- **Vertex AI**: ~$2,000/month
- **Cloud Storage**: ~$100/month
- **Total**: ~$2,600/month

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   gcloud auth application-default login
   ```

2. **Permission Denied**
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="user:your-email@domain.com" \
     --role="roles/owner"
   ```

3. **Out of Memory**
   - Increase memory in `vertex_ai_config.yaml`
   - Use smaller batch sizes in training

4. **Training Timeout**
   - Increase timeout in `vertex_ai_config.yaml`
   - Optimize feature computation

### Debug Commands

```bash
# Check service status
gcloud run services describe shark-habitat-prediction --region=us-central1

# View logs
gcloud logs read "resource.type=cloud_run_revision" --limit=50

# Check Vertex AI jobs
gcloud ai custom-jobs list --region=us-central1
```

## 📈 Performance Optimization

### Model Training
- Use GPU acceleration for large datasets
- Implement distributed training for multiple models
- Use hyperparameter tuning for optimal performance

### Inference
- Enable Cloud CDN for global distribution
- Use model caching for frequent predictions
- Implement batch prediction for efficiency

## 🔐 Security

### Data Protection
- All data encrypted in transit and at rest
- Service account with minimal permissions
- VPC for network isolation

### Access Control
- IAM-based access control
- API keys for external access
- Audit logging for compliance

## 📞 Support

For issues with GCP deployment:
1. Check Cloud Logging for error messages
2. Review service account permissions
3. Verify environment variables
4. Check resource quotas

For model-specific issues:
1. Review training logs
2. Check data quality
3. Validate feature engineering
4. Monitor model performance

## 🎯 Next Steps

After successful deployment:

1. **Test the API**: Make test predictions
2. **Monitor Performance**: Set up alerts
3. **Scale as Needed**: Adjust resources
4. **Update Models**: Retrain with new data
5. **Deploy Globally**: Use multiple regions

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Storage Documentation](https://cloud.google.com/storage/docs)
- [Cloud Build Documentation](https://cloud.google.com/build/docs)
