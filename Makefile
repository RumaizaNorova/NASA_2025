PYTHON ?= python
CONFIG := config/params_enhanced.yaml

# Default targets
.PHONY: help install clean data features labels train predict map all demo test validate all-enhanced map-enhanced predict-all predict-timeseries validate-integrity export

# Show help
help:
	@echo "Sharks from Space - NASA Space Apps Challenge"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies and setup environment"
	@echo "  clean       - Clean intermediate files and outputs"
	@echo "  data        - Fetch NASA satellite data (requires credentials)"
	@echo "  features    - Compute advanced oceanographic features"
	@echo "  labels      - Process shark data and generate pseudo-absences"
	@echo "  train       - Train multiple ML models with advanced evaluation"
	@echo "  predict     - Generate habitat probability predictions"
	@echo "  predict-all - Generate predictions for all models"
	@echo "  predict-timeseries - Generate 365-day time series predictions"
	@echo "  map         - Create interactive web visualization"
	@echo "  map-enhanced - Create enhanced interactive map with time series"
	@echo "  map-timeseries - Create time series visualization"
	@echo "  all         - Run complete pipeline end-to-end"
	@echo "  all-enhanced - Run enhanced pipeline with advanced visualization"
	@echo "  test        - Run unit tests and validation"
	@echo "  validate    - Validate environment and data integrity"
	@echo "  validate-integrity - Comprehensive data integrity validation"
	@echo "  export      - Export results for sharing"
	@echo ""
	@echo "Advanced options:"
	@echo "  make data PARALLEL=true    - Enable parallel data downloads"
	@echo "  make features USE_DASK=true - Use Dask for parallel processing"
	@echo "  make train ALGORITHMS=xgboost lightgbm - Train specific models"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG=path/to/config.yaml - Use custom configuration"

# Install dependencies and setup environment
install:
	@echo "[*] Setting up environment..."
	conda env create -f environment.yml
	@echo "[*] Environment created. Activate with: conda activate sharks-from-space"
	@echo "[*] Copy .env.example to .env and configure your credentials"

# Clean intermediate files
clean:
	@echo "[*] Cleaning intermediate files..."
	rm -rf data/raw/*
	rm -rf data/interim/*
	rm -rf logs/*
	rm -rf web/data/*
	@echo "[*] Cleanup complete"

# Fetch raw satellite data with enhanced NASA API integration
data:
	@echo "[*] Fetching NASA satellite data..."
	$(PYTHON) src/fetch_data.py --config $(CONFIG) $(if $(PARALLEL),--parallel)

# Compute advanced oceanographic features
features:
	@echo "[*] Computing advanced features..."
	$(PYTHON) src/compute_features.py --config $(CONFIG) $(if $(USE_DASK),--use-dask) $(if $(OUTPUT_FORMAT),--output-format $(OUTPUT_FORMAT))

# Process shark data and generate pseudo-absences
labels:
	@echo "[*] Processing shark data and generating pseudo-absences..."
	$(PYTHON) src/label_join.py --config $(CONFIG) $(if $(OUTPUT_FORMAT),--output-format $(OUTPUT_FORMAT))

# Train multiple ML models with advanced evaluation
train:
	@echo "[*] Training ML models..."
	$(PYTHON) src/train_model.py --config $(CONFIG) $(if $(ALGORITHMS),--algorithms $(ALGORITHMS))

# Predict habitat probability over the full grid
predict:
	@echo "[*] Generating habitat predictions..."
	$(PYTHON) src/predict_grid.py --config $(CONFIG) --model xgboost --output-format both

# Generate predictions for all models
predict-all: train
	@echo "[*] Generating predictions for all models..."
	$(PYTHON) src/predict_grid_enhanced.py --config $(CONFIG) --models xgboost lightgbm random_forest --output-format all
	@echo "[*] All model predictions saved to web/data/"

# Generate time series predictions
predict-timeseries: data features labels train
	@echo "[*] Generating time series predictions..."
	$(PYTHON) src/predict_grid.py --config $(CONFIG) --date-range 2024-01-01,2024-01-31 --model xgboost --output-format both --parallel
	@echo "[*] Time series predictions saved to web/data/"

# Build enhanced interactive web map
map:
	@echo "[*] Creating interactive web visualization..."
	$(PYTHON) src/make_maps.py --config $(CONFIG)

# Build enhanced interactive map with time series and multi-model support
map-enhanced: predict-all
	@echo "[*] Creating enhanced interactive map with time series..."
	$(PYTHON) src/make_maps_enhanced.py --config $(CONFIG) --time-series --multi-model
	@echo "[*] Enhanced interactive map saved to web/index.html"

# Build time series visualization
map-timeseries: predict-timeseries
	@echo "[*] Creating time series map..."
	$(PYTHON) src/make_maps.py --config $(CONFIG) --time-series
	@echo "[*] Time series map saved to web/index.html"

# Run the entire enhanced pipeline end-to-end
all: data features labels train predict map
	@echo "[*] Complete pipeline finished successfully!"

# Run the enhanced pipeline with advanced visualization
all-enhanced: data features labels train predict-all map-enhanced
	@echo "[*] Enhanced pipeline with advanced visualization finished successfully!"
	@echo "🎉 SHARKS FROM SPACE - MISSION ACCOMPLISHED!"
	@echo "🌊 Open web/index.html to explore your interactive shark habitat maps!"
	@echo "📊 Check data/interim/training_summary.txt for model performance!"
	@echo "🗺️ Your enhanced visualization is ready for conservation impact!"

# Run the complete pipeline with validation
all-validated: data features labels train predict map validate-integrity
	@echo "[*] Complete validated pipeline finished successfully!"
	@echo "[*] All results verified to be based on real NASA and shark data"

# Quick mode for testing (smaller dataset, faster processing)
quick-test: clean
	@echo "[*] Running quick test mode..."
	$(PYTHON) src/label_join.py --config config/params_quick.yaml
	@echo "[*] Quick test completed"

# Quick full pipeline for testing
all-quick: data features labels train predict map
	@echo "[*] Quick pipeline finished successfully!"


# Run tests and validation
test:
	@echo "[*] Running tests..."
	pytest tests/ -v
	@echo "[*] Running data validation..."
	$(PYTHON) -c "from src.utils import validate_environment; print(validate_environment())"

# Validate environment and data integrity
validate:
	@echo "[*] Validating environment..."
	$(PYTHON) -c "from src.utils import validate_environment; import json; print(json.dumps(validate_environment(), indent=2))"
	@echo "[*] Checking data files..."
	@if [ -f data/interim/features.nc ] || [ -f data/interim/features.zarr ]; then \
		echo "✓ Feature data found"; \
	else \
		echo "✗ Feature data missing - run 'make features'"; \
	fi
	@if [ -f data/interim/training_data.csv ] || [ -f data/interim/training_data.parquet ]; then \
		echo "✓ Training data found"; \
	else \
		echo "✗ Training data missing - run 'make labels'"; \
	fi
	@if [ -f data/interim/training_metrics.json ]; then \
		echo "✓ Model metrics found"; \
	else \
		echo "✗ Model metrics missing - run 'make train'"; \
	fi

# Comprehensive data integrity validation
validate-integrity:
	@echo "[*] Running comprehensive data integrity validation..."
	$(PYTHON) src/validate_data_integrity.py --config $(CONFIG)
	@echo "[*] Data integrity validation complete"

# Development targets
dev-setup: install
	@echo "[*] Setting up development environment..."
	pip install -e .
	pre-commit install

# Performance testing
benchmark:
	@echo "[*] Running performance benchmarks..."
	$(PYTHON) scripts/benchmark.py --config $(CONFIG)

# Generate documentation
docs:
	@echo "[*] Generating documentation..."
	$(PYTHON) scripts/generate_docs.py

# Export results for sharing
export:
	@echo "[*] Exporting results..."
	mkdir -p exports/$(shell date +%Y%m%d_%H%M%S)
	cp -r data/interim/* exports/$(shell date +%Y%m%d_%H%M%S)/
	cp -r web/* exports/$(shell date +%Y%m%d_%H%M%S)/
	@echo "[*] Results exported to exports/"

# Quick start for new users
quickstart: install
	@echo "[*] Setting up credentials..."
	@echo "[*] Please copy .env.example to .env and configure your credentials"
	@echo "[*] Then run 'make all' to start the complete pipeline"