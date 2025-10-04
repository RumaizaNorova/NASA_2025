
# REAL DATA PIPELINE VALIDATION REPORT

## Overall Status: FAIL

## Validation Summary:
- Total Checks: 6
- Passed: 3
- Failed: 3

## Detailed Results:

### Nasa Data Download
**Status:** PASS
**Details:**
- download_script: EXISTS
- data_directory: EXISTS
- netcdf_files: 1248

### Data Processing
**Status:** WARNING
**Issues:**
- Synthetic data method found: 'synthetic_data'
- Processed features file not found
**Details:**
- processing_script: EXISTS

### Negative Sampling
**Status:** FAIL
**Issues:**
- Random generation found: 'np.random'
- Random generation found: 'random.choice'
- Random generation found: 'random.uniform'
- Random generation found: 'random.normal'
**Details:**
- sampling_script: EXISTS
- negative_samples_file: EXISTS
- negative_samples_count: 8838
- data_sources: {'oceanographic_survey': 4696, 'fishing_vessel': 3000, 'environmental_sampling': 921, 'marine_protected_area': 221}

### Balanced Dataset
**Status:** PASS
**Details:**
- balanced_dataset_file: EXISTS
- total_samples: 403596
- label_distribution: {1: 394758, 0: 8838}
- balance_ratio: 0.022388399981760978

### Model Training
**Status:** FAIL
**Issues:**
- Synthetic data references found in training script
**Details:**
- training_script: EXISTS
- model_file: EXISTS
- model_metadata: {'model_type': 'RandomForestClassifier', 'data_source': '100% REAL NASA satellite data', 'training_data': {'total_samples': 'Real dataset size', 'positive_samples': 'Real shark observations', 'negative_samples': 'Real background locations', 'synthetic_data': 'NONE - 100% REAL DATA'}, 'performance': {'test_roc_auc': nan, 'cv_roc_auc_mean': nan, 'cv_roc_auc_std': nan}, 'model_parameters': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'random_state': 42, 'n_jobs': -1}, 'created_at': '2025-10-04T04:33:08.558878'}

### File Integrity
**Status:** PASS
**Details:**
- existing_files: ['src/download_nasa_data.py', 'src/process_nasa_data.py', 'src/create_real_negative_sampling.py', 'src/train_real_data_model.py', 'src/validate_real_data_pipeline.py']
- missing_files: []
- data/raw_exists: True
- data/interim_exists: True
- models_exists: True
