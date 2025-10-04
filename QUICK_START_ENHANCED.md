# 🚀 Quick Start Guide - Sharks from Space Enhanced

Get the enhanced shark habitat prediction pipeline running in **10 minutes**!

## 🎯 Prerequisites

- **Python 3.10+** with conda/miniconda
- **8GB RAM** minimum (16GB recommended)
- **20GB disk space** for data and outputs
- **Internet connection** for NASA data download

## ⚡ Lightning Setup (5 minutes)

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd sharks-from-space

# Create environment (this may take 3-5 minutes)
conda env create -f environment.yml
conda activate sharks-from-space

# Verify installation
python -c "import xgboost, lightgbm, sklearn; print('✅ All dependencies installed!')"
```

### 2. Configure Credentials

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials (optional for demo)
nano .env
```

**Required for full functionality:**
```bash
# NASA Earthdata (free account at https://urs.earthdata.nasa.gov/)
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# Mapbox (free token at https://account.mapbox.com/)
MAPBOX_ACCESS_TOKEN=your_mapbox_token
```

**Note**: The pipeline will work without credentials using demo data, but with limited functionality.

## 🚀 One-Command Launch

```bash
# Complete enhanced pipeline (4-6 hours)
make all-enhanced

# Watch the magic happen! ✨
```

**What happens:**
1. 📡 Downloads NASA satellite data (6 datasets)
2. 🔬 Computes 37 oceanographic features
3. 🦈 Processes shark tracking data
4. 🤖 Trains 3 ML models (XGBoost, LightGBM, Random Forest)
5. 🗺️ Generates interactive habitat maps
6. 📊 Creates performance dashboard

## 🎮 Explore Results (Instant)

### Interactive Map
```bash
# Open in browser
open web/index.html

# Or manually navigate to:
# file:///path/to/sharks-from-space/web/index.html
```

**Features to explore:**
- ▶️ **Play Animation**: Watch 365-day habitat evolution
- 🔄 **Model Switching**: Compare XGBoost vs LightGBM vs Random Forest
- 📊 **Performance Dashboard**: Live model metrics
- 🎨 **Display Options**: Toggle shark tracks, environmental data
- 📱 **Mobile Friendly**: Works on phones and tablets

### Performance Summary
```bash
# Check model performance
cat data/interim/training_summary.txt

# View detailed metrics
python -c "
import json
with open('data/interim/training_metrics.json') as f:
    metrics = json.load(f)
for model, data in metrics.items():
    roc_auc = data['aggregated_metrics'].get('roc_auc', 0)
    print(f'{model.upper()}: ROC-AUC = {roc_auc:.3f}')
"
```

## 🔧 Troubleshooting

### Common Issues

**1. Environment Creation Fails**
```bash
# Try with mamba (faster)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

**2. Memory Issues**
```bash
# Reduce memory usage
export MEMORY_LIMIT_GB=2.0
make all-enhanced
```

**3. NASA Data Download Fails**
```bash
# Use demo mode with existing data
make all-enhanced PARALLEL=false
```

**4. Mapbox Token Issues**
```bash
# The map will still work with fallback visualization
# Just won't have advanced Mapbox features
```

### Performance Optimization

**For Faster Processing:**
```bash
# Use fewer workers on slower systems
export MAX_WORKERS=4
make all-enhanced

# Skip time series generation
make data features labels train predict map
```

**For Better Performance:**
```bash
# Use more workers on powerful systems
export MAX_WORKERS=16
make all-enhanced
```

## 📊 Expected Results

### Performance Targets
- **ROC-AUC**: > 0.70 (vs baseline 0.55)
- **PR-AUC**: > 0.40 (vs baseline 0.22)
- **TSS**: > 0.30 (vs baseline 0.01)
- **Processing Time**: < 6 hours (vs baseline 20+ hours)

### Output Files
```
web/data/
├── habitat_prob_xgboost_*.png     # XGBoost predictions
├── habitat_prob_lightgbm_*.png    # LightGBM predictions
├── habitat_prob_random_forest_*.png # Random Forest predictions
├── *.tif                          # GeoTIFF rasters
├── *.json                         # Metadata files
└── prediction_metadata.json       # Comprehensive metadata

data/interim/
├── training_metrics.json          # Model performance
├── feature_importance.json        # Feature rankings
├── *.pkl                          # Trained models
└── training_summary.txt           # Human-readable summary
```

## 🎯 Next Steps

### Immediate Actions
1. **Explore the Interactive Map**: Navigate to `web/index.html`
2. **Check Performance**: Review `data/interim/training_summary.txt`
3. **Validate Results**: Run `python src/validate_delivery_enhanced.py`

### Advanced Usage
```bash
# Train specific models only
make train ALGORITHMS=xgboost lightgbm

# Generate predictions for specific date range
python src/predict_grid_enhanced.py --date-range 2014-01-01,2014-01-31

# Create custom visualization
python src/make_maps_enhanced.py --multi-model --time-series
```

### Conservation Applications
1. **Marine Protected Areas**: Use habitat maps to identify critical areas
2. **Fishing Optimization**: Avoid high-probability shark habitats
3. **Climate Impact**: Analyze habitat changes over time
4. **Policy Support**: Provide evidence for conservation decisions

## 📚 Learn More

- **Full Documentation**: [README_ENHANCED.md](README_ENHANCED.md)
- **Configuration Guide**: [config/README.md](config/README.md)
- **API Reference**: [docs/API.md](docs/API.md)
- **Troubleshooting**: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 🆘 Need Help?

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Discussions**: [Join the conversation](https://github.com/your-repo/discussions)
- **Email**: your-email@example.com

## 🎉 Success!

If you see this message, congratulations! You've successfully launched the enhanced Sharks from Space pipeline:

```
🎉 SHARKS FROM SPACE - MISSION ACCOMPLISHED!
🌊 Open web/index.html to explore your interactive shark habitat maps!
📊 Check data/interim/training_summary.txt for model performance!
🗺️ Your enhanced visualization is ready for conservation impact!
```

**Ready to make waves in shark conservation! 🦈🌊🚀**

---

*Quick Start Guide v2.0 - Enhanced Pipeline*
