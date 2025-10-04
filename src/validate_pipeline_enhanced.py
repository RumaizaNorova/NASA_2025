"""
Enhanced pipeline validation script for Sharks from Space project.

This script performs comprehensive validation of the entire pipeline including:
- Data integrity checks
- Model training validation
- Prediction output validation
- Web visualization validation
- Performance metrics validation
- End-to-end pipeline testing

Features:
- Real data validation (no synthetic data)
- Comprehensive file existence checks
- Data format consistency validation
- Model performance thresholds
- Web visualization completeness
- Pipeline dependency validation
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import pickle
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

try:
    from .utils import load_config, ensure_dir, setup_logging, validate_environment
except ImportError:
    from utils import load_config, ensure_dir, setup_logging, validate_environment

# Load environment variables
load_dotenv()


class EnhancedPipelineValidator:
    """Enhanced pipeline validator with comprehensive checks."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Configuration
        self.roi_cfg = config.get('roi', {})
        self.time_cfg = config.get('time', {})
        self.model_cfg = config.get('model', {})
        self.output_cfg = config.get('output', {})
        
        # Validation results
        self.validation_results = {
            'overall_status': 'PASS',
            'checks': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_roc_auc': 0.65,
            'min_pr_auc': 0.35,
            'min_tss': 0.20,
            'min_f1': 0.30
        }
    
    def validate_environment(self) -> bool:
        """Validate environment setup and dependencies."""
        self.logger.info("Validating environment setup...")
        
        try:
            env_status = validate_environment()
            
            # Check critical environment variables
            critical_vars = ['SHARK_CSV', 'NASA_EARTHDATA_USERNAME', 'NASA_EARTHDATA_PASSWORD']
            missing_vars = []
            
            for var in critical_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                self.validation_results['errors'].append(f"Missing environment variables: {missing_vars}")
                self.validation_results['checks']['environment'] = 'FAIL'
                return False
            
            # Check shark data availability
            if not env_status.get('shark_csv_available', False):
                self.validation_results['errors'].append("Shark CSV data not available")
                self.validation_results['checks']['environment'] = 'FAIL'
                return False
            
            self.validation_results['checks']['environment'] = 'PASS'
            self.logger.info("Environment validation passed")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Environment validation failed: {e}")
            self.validation_results['checks']['environment'] = 'FAIL'
            return False
    
    def validate_data_files(self) -> bool:
        """Validate data files existence and integrity."""
        self.logger.info("Validating data files...")
        
        required_files = [
            'data/interim/features.nc',
            'data/interim/training_data.csv',
            'data/interim/label_join_stats.json'
        ]
        
        missing_files = []
        corrupted_files = []
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
                continue
            
            # Check file integrity
            try:
                if file_path.endswith('.nc'):
                    # Check NetCDF file
                    with xr.open_dataset(file_path) as ds:
                        if len(ds.data_vars) == 0:
                            corrupted_files.append(file_path)
                
                elif file_path.endswith('.csv'):
                    # Check CSV file
                    df = pd.read_csv(file_path)
                    if len(df) == 0:
                        corrupted_files.append(file_path)
                
                elif file_path.endswith('.json'):
                    # Check JSON file
                    with open(file_path, 'r') as f:
                        json.load(f)
                        
            except Exception as e:
                corrupted_files.append(f"{file_path}: {e}")
        
        if missing_files:
            self.validation_results['errors'].extend([f"Missing file: {f}" for f in missing_files])
        
        if corrupted_files:
            self.validation_results['errors'].extend([f"Corrupted file: {f}" for f in corrupted_files])
        
        if missing_files or corrupted_files:
            self.validation_results['checks']['data_files'] = 'FAIL'
            return False
        
        self.validation_results['checks']['data_files'] = 'PASS'
        self.logger.info("Data files validation passed")
        return True
    
    def validate_training_data(self) -> bool:
        """Validate training data quality and consistency."""
        self.logger.info("Validating training data...")
        
        try:
            # Load training data
            df = pd.read_csv('data/interim/training_data.csv')
            
            # Check data size
            if len(df) < 100:
                self.validation_results['warnings'].append(f"Training data is small: {len(df)} samples")
            
            # Check for required columns
            required_cols = ['latitude', 'longitude', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.validation_results['errors'].append(f"Missing columns in training data: {missing_cols}")
                self.validation_results['checks']['training_data'] = 'FAIL'
                return False
            
            # Check label distribution
            label_counts = df['label'].value_counts()
            if len(label_counts) < 2:
                self.validation_results['errors'].append("Training data has only one class")
                self.validation_results['checks']['training_data'] = 'FAIL'
                return False
            
            # Check for class imbalance
            pos_ratio = label_counts.get(1, 0) / len(df)
            if pos_ratio < 0.05 or pos_ratio > 0.95:
                self.validation_results['warnings'].append(f"Severe class imbalance: {pos_ratio:.3f} positive ratio")
            
            # Check coordinate ranges
            if not (self.roi_cfg['lat_min'] <= df['latitude'].min() <= df['latitude'].max() <= self.roi_cfg['lat_max']):
                self.validation_results['warnings'].append("Training data coordinates outside ROI")
            
            if not (self.roi_cfg['lon_min'] <= df['longitude'].min() <= df['longitude'].max() <= self.roi_cfg['lon_max']):
                self.validation_results['warnings'].append("Training data coordinates outside ROI")
            
            # Check for missing values in features
            feature_cols = [col for col in df.columns if col not in ['latitude', 'longitude', 'timestamp', 'date', 'label', 'shark_id', 'species']]
            missing_feature_data = df[feature_cols].isnull().sum().sum()
            
            if missing_feature_data > 0:
                self.validation_results['warnings'].append(f"Missing feature values: {missing_feature_data}")
            
            self.validation_results['checks']['training_data'] = 'PASS'
            self.logger.info(f"Training data validation passed: {len(df)} samples, {len(feature_cols)} features")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Training data validation failed: {e}")
            self.validation_results['checks']['training_data'] = 'FAIL'
            return False
    
    def validate_models(self) -> bool:
        """Validate trained models and performance."""
        self.logger.info("Validating trained models...")
        
        try:
            # Check model files exist
            algorithms = self.model_cfg.get('algorithms', ['xgboost'])
            missing_models = []
            
            for algorithm in algorithms:
                model_path = f'data/interim/{algorithm}_model.pkl'
                if not os.path.exists(model_path):
                    missing_models.append(algorithm)
                    continue
                
                # Check model can be loaded
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Check model has required methods
                    if not (hasattr(model, 'predict') and hasattr(model, 'predict_proba')):
                        self.validation_results['errors'].append(f"Model {algorithm} missing required methods")
                        
                except Exception as e:
                    self.validation_results['errors'].append(f"Failed to load model {algorithm}: {e}")
            
            if missing_models:
                self.validation_results['errors'].extend([f"Missing model: {m}" for m in missing_models])
                self.validation_results['checks']['models'] = 'FAIL'
                return False
            
            # Check performance metrics
            metrics_path = 'data/interim/training_metrics.json'
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                performance_issues = []
                for algorithm in algorithms:
                    if algorithm in metrics:
                        algo_metrics = metrics[algorithm].get('aggregated_metrics', {})
                        
                        # Check performance thresholds
                        roc_auc = algo_metrics.get('roc_auc', 0)
                        pr_auc = algo_metrics.get('pr_auc', 0)
                        tss = algo_metrics.get('tss', 0)
                        f1 = algo_metrics.get('f1', 0)
                        
                        if roc_auc < self.performance_thresholds['min_roc_auc']:
                            performance_issues.append(f"{algorithm} ROC-AUC below threshold: {roc_auc:.3f} < {self.performance_thresholds['min_roc_auc']}")
                        
                        if pr_auc < self.performance_thresholds['min_pr_auc']:
                            performance_issues.append(f"{algorithm} PR-AUC below threshold: {pr_auc:.3f} < {self.performance_thresholds['min_pr_auc']}")
                        
                        if tss < self.performance_thresholds['min_tss']:
                            performance_issues.append(f"{algorithm} TSS below threshold: {tss:.3f} < {self.performance_thresholds['min_tss']}")
                        
                        if f1 < self.performance_thresholds['min_f1']:
                            performance_issues.append(f"{algorithm} F1 below threshold: {f1:.3f} < {self.performance_thresholds['min_f1']}")
                
                if performance_issues:
                    self.validation_results['warnings'].extend(performance_issues)
            
            self.validation_results['checks']['models'] = 'PASS'
            self.logger.info("Models validation passed")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Models validation failed: {e}")
            self.validation_results['checks']['models'] = 'FAIL'
            return False
    
    def validate_predictions(self) -> bool:
        """Validate prediction outputs."""
        self.logger.info("Validating prediction outputs...")
        
        try:
            # Check prediction files exist
            web_data_dir = 'web/data'
            if not os.path.exists(web_data_dir):
                self.validation_results['errors'].append("Web data directory not found")
                self.validation_results['checks']['predictions'] = 'FAIL'
                return False
            
            # Check for prediction files
            png_files = [f for f in os.listdir(web_data_dir) if f.endswith('.png') and 'habitat_prob' in f]
            tif_files = [f for f in os.listdir(web_data_dir) if f.endswith('.tif') and 'habitat_prob' in f]
            json_files = [f for f in os.listdir(web_data_dir) if f.endswith('.json') and 'habitat_prob' in f]
            
            if not png_files:
                self.validation_results['errors'].append("No prediction PNG files found")
                self.validation_results['checks']['predictions'] = 'FAIL'
                return False
            
            if not tif_files:
                self.validation_results['warnings'].append("No prediction GeoTIFF files found")
            
            if not json_files:
                self.validation_results['warnings'].append("No prediction metadata files found")
            
            # Check metadata file
            metadata_path = os.path.join(web_data_dir, 'prediction_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate metadata structure
                required_keys = ['models_used', 'features_used', 'roi_config']
                missing_keys = [key for key in required_keys if key not in metadata]
                
                if missing_keys:
                    self.validation_results['warnings'].append(f"Missing metadata keys: {missing_keys}")
            
            self.validation_results['checks']['predictions'] = 'PASS'
            self.logger.info(f"Predictions validation passed: {len(png_files)} PNG files")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Predictions validation failed: {e}")
            self.validation_results['checks']['predictions'] = 'FAIL'
            return False
    
    def validate_web_visualization(self) -> bool:
        """Validate web visualization files."""
        self.logger.info("Validating web visualization...")
        
        try:
            # Check HTML file exists
            html_path = 'web/index.html'
            if not os.path.exists(html_path):
                self.validation_results['errors'].append("Web visualization HTML file not found")
                self.validation_results['checks']['web_visualization'] = 'FAIL'
                return False
            
            # Check HTML file content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for required elements
            required_elements = [
                'Sharks from Space',
                'mapboxgl',
                'habitat',
                'control-panel'
            ]
            
            missing_elements = [elem for elem in required_elements if elem not in html_content]
            
            if missing_elements:
                self.validation_results['warnings'].append(f"Missing HTML elements: {missing_elements}")
            
            # Check for data references
            if 'habitat_prob_' not in html_content:
                self.validation_results['warnings'].append("HTML does not reference prediction data")
            
            self.validation_results['checks']['web_visualization'] = 'PASS'
            self.logger.info("Web visualization validation passed")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Web visualization validation failed: {e}")
            self.validation_results['checks']['web_visualization'] = 'FAIL'
            return False
    
    def validate_pipeline_dependencies(self) -> bool:
        """Validate pipeline dependency chain."""
        self.logger.info("Validating pipeline dependencies...")
        
        try:
            # Check that each step produces required outputs for next step
            dependencies = [
                ('data', 'data/raw'),
                ('features', 'data/interim/features.nc'),
                ('labels', 'data/interim/training_data.csv'),
                ('train', 'data/interim/training_metrics.json'),
                ('predict-all', 'web/data'),
                ('map-enhanced', 'web/index.html')
            ]
            
            missing_dependencies = []
            for step, output in dependencies:
                if not os.path.exists(output):
                    missing_dependencies.append(f"{step} -> {output}")
            
            if missing_dependencies:
                self.validation_results['warnings'].extend([f"Missing dependency: {d}" for d in missing_dependencies])
            
            self.validation_results['checks']['pipeline_dependencies'] = 'PASS'
            self.logger.info("Pipeline dependencies validation passed")
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Pipeline dependencies validation failed: {e}")
            self.validation_results['checks']['pipeline_dependencies'] = 'FAIL'
            return False
    
    def generate_recommendations(self) -> None:
        """Generate improvement recommendations."""
        self.logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Check performance metrics
        metrics_path = 'data/interim/training_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            for algorithm, algo_metrics in metrics.items():
                aggregated = algo_metrics.get('aggregated_metrics', {})
                roc_auc = aggregated.get('roc_auc', 0)
                
                if roc_auc < 0.70:
                    recommendations.append(f"Consider hyperparameter tuning for {algorithm} (ROC-AUC: {roc_auc:.3f})")
        
        # Check data quality
        if 'training_data' in self.validation_results['checks']:
            if self.validation_results['checks']['training_data'] == 'PASS':
                try:
                    df = pd.read_csv('data/interim/training_data.csv')
                    if len(df) < 1000:
                        recommendations.append("Consider collecting more training data for better model performance")
                except:
                    pass
        
        # Check feature engineering
        try:
            with xr.open_dataset('data/interim/features.nc') as ds:
                n_features = len(ds.data_vars)
                if n_features < 20:
                    recommendations.append("Consider adding more oceanographic features for better predictions")
        except:
            pass
        
        self.validation_results['recommendations'] = recommendations
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete pipeline validation."""
        self.logger.info("Starting enhanced pipeline validation...")
        
        # Run all validation checks
        checks = [
            ('environment', self.validate_environment),
            ('data_files', self.validate_data_files),
            ('training_data', self.validate_training_data),
            ('models', self.validate_models),
            ('predictions', self.validate_predictions),
            ('web_visualization', self.validate_web_visualization),
            ('pipeline_dependencies', self.validate_pipeline_dependencies)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                result = check_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.validation_results['errors'].append(f"{check_name} validation crashed: {e}")
                self.validation_results['checks'][check_name] = 'CRASH'
                all_passed = False
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Determine overall status
        if self.validation_results['errors']:
            self.validation_results['overall_status'] = 'FAIL'
        elif self.validation_results['warnings']:
            self.validation_results['overall_status'] = 'WARN'
        else:
            self.validation_results['overall_status'] = 'PASS'
        
        # Save validation results
        results_path = 'data/interim/validation_results.json'
        ensure_dir('data/interim')
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Validation complete. Overall status: {self.validation_results['overall_status']}")
        return self.validation_results
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "="*60)
        print("SHARKS FROM SPACE - ENHANCED PIPELINE VALIDATION")
        print("="*60)
        
        print(f"\nOverall Status: {self.validation_results['overall_status']}")
        
        print(f"\nValidation Checks:")
        for check_name, status in self.validation_results['checks'].items():
            status_icon = "✓" if status == 'PASS' else "✗" if status == 'FAIL' else "⚠"
            print(f"  {status_icon} {check_name.replace('_', ' ').title()}: {status}")
        
        if self.validation_results['errors']:
            print(f"\nErrors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                print(f"  ✗ {error}")
        
        if self.validation_results['warnings']:
            print(f"\nWarnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                print(f"  ⚠ {warning}")
        
        if self.validation_results['recommendations']:
            print(f"\nRecommendations ({len(self.validation_results['recommendations'])}):")
            for rec in self.validation_results['recommendations']:
                print(f"  💡 {rec}")
        
        print("\n" + "="*60)
        
        if self.validation_results['overall_status'] == 'PASS':
            print("🎉 PIPELINE VALIDATION PASSED!")
            print("🌊 Your Sharks from Space pipeline is ready for production!")
        elif self.validation_results['overall_status'] == 'WARN':
            print("⚠️  PIPELINE VALIDATION PASSED WITH WARNINGS")
            print("🔧 Consider addressing the warnings for optimal performance")
        else:
            print("❌ PIPELINE VALIDATION FAILED")
            print("🛠️  Please fix the errors before running the pipeline")
        
        print("="*60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate enhanced Sharks from Space pipeline.")
    parser.add_argument("--config", default="config/params_enhanced.yaml", 
                       help="Path to YAML configuration file")
    parser.add_argument("--output", default="data/interim/validation_results.json",
                       help="Output file for validation results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize validator
    validator = EnhancedPipelineValidator(config, args)
    
    # Run validation
    results = validator.run_validation()
    
    # Print summary
    validator.print_summary()
    
    # Exit with appropriate code
    if results['overall_status'] == 'FAIL':
        sys.exit(1)
    elif results['overall_status'] == 'WARN':
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
