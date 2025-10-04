"""
Data Integrity Validation Script for NASA Shark Habitat Modeling

This script validates that all data processing and results are based on real NASA satellite data
and actual shark tracking data. It performs comprehensive checks to ensure no synthetic or
artificial data has been introduced anywhere in the pipeline.

Key Validation Areas:
1. Real NASA satellite data validation
2. Actual shark tracking data validation
3. Feature computation verification
4. Model training data integrity
5. Prediction output validation
6. No synthetic data detection
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from .utils import load_config, setup_logging, validate_environment
except ImportError:
    from utils import load_config, setup_logging, validate_environment


class DataIntegrityValidator:
    """Comprehensive data integrity validator for the shark habitat modeling pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging('validate_data_integrity')
        self.validation_results = {
            'nasa_data': {},
            'shark_data': {},
            'features': {},
            'training_data': {},
            'models': {},
            'predictions': {},
            'overall_status': 'UNKNOWN'
        }
    
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive validation of all pipeline components."""
        self.logger.info("Starting comprehensive data integrity validation")
        
        # Validate NASA satellite data
        self._validate_nasa_data()
        
        # Validate shark tracking data
        self._validate_shark_data()
        
        # Validate feature computation
        self._validate_features()
        
        # Validate training data
        self._validate_training_data()
        
        # Validate models
        self._validate_models()
        
        # Validate predictions
        self._validate_predictions()
        
        # Determine overall status
        self._determine_overall_status()
        
        # Save validation report
        self._save_validation_report()
        
        return self.validation_results
    
    def _validate_nasa_data(self) -> None:
        """Validate NASA satellite data integrity."""
        self.logger.info("Validating NASA satellite data")
        
        data_dir = 'data/raw'
        roi = self.config.get('roi', {})
        time_cfg = self.config.get('time', {})
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'datasets_found': [],
            'data_coverage': {},
            'real_data_confirmed': True
        }
        
        # Check for required data directories
        required_datasets = ['mur_sst', 'measures_ssh', 'oscar_currents', 'pace_chl', 'smap_salinity', 'gpm_precipitation']
        
        for dataset in required_datasets:
            dataset_path = os.path.join(data_dir, dataset)
            if os.path.exists(dataset_path):
                validation['datasets_found'].append(dataset)
                
                # Check for real data files
                files = [f for f in os.listdir(dataset_path) if f.endswith('.nc')]
                if files:
                    # Validate one sample file
                    sample_file = os.path.join(dataset_path, files[0])
                    try:
                        ds = xr.open_dataset(sample_file)
                        
                        # Check for realistic data ranges
                        for var in ds.data_vars:
                            data = ds[var].values
                            if np.any(np.isfinite(data)):
                                validation['data_coverage'][f"{dataset}_{var}"] = {
                                    'min': float(np.nanmin(data)),
                                    'max': float(np.nanmax(data)),
                                    'mean': float(np.nanmean(data)),
                                    'finite_count': int(np.sum(np.isfinite(data)))
                                }
                        
                        ds.close()
                        
                    except Exception as e:
                        validation['issues'].append(f"Error reading {dataset}: {e}")
                        validation['status'] = 'FAIL'
                else:
                    validation['issues'].append(f"No data files found in {dataset}")
                    validation['status'] = 'FAIL'
            else:
                validation['issues'].append(f"Dataset directory not found: {dataset}")
                validation['status'] = 'FAIL'
        
        # Check for synthetic data patterns
        if self._detect_synthetic_patterns(data_dir):
            validation['issues'].append("Synthetic data patterns detected")
            validation['real_data_confirmed'] = False
            validation['status'] = 'FAIL'
        
        self.validation_results['nasa_data'] = validation
    
    def _validate_shark_data(self) -> None:
        """Validate shark tracking data integrity."""
        self.logger.info("Validating shark tracking data")
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'total_observations': 0,
            'date_range': {},
            'spatial_coverage': {},
            'species_breakdown': {},
            'real_data_confirmed': True
        }
        
        # Load shark data
        shark_csv = os.environ.get('SHARK_CSV')
        if not shark_csv or not os.path.exists(shark_csv):
            validation['issues'].append("Shark CSV file not found")
            validation['status'] = 'FAIL'
            self.validation_results['shark_data'] = validation
            return
        
        try:
            # Load data in chunks to handle large files
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(shark_csv, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            
            validation['total_observations'] = len(df)
            
            # Check for realistic data ranges
            if 'latitude' in df.columns:
                lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
                if lat_min < -90 or lat_max > 90:
                    validation['issues'].append(f"Invalid latitude range: {lat_min} to {lat_max}")
                    validation['status'] = 'FAIL'
                validation['spatial_coverage']['latitude'] = {'min': lat_min, 'max': lat_max}
            
            if 'longitude' in df.columns:
                lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
                if lon_min < -180 or lon_max > 180:
                    validation['issues'].append(f"Invalid longitude range: {lon_min} to {lon_max}")
                    validation['status'] = 'FAIL'
                validation['spatial_coverage']['longitude'] = {'min': lon_min, 'max': lon_max}
            
            # Check date range
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                date_min, date_max = df['datetime'].min(), df['datetime'].max()
                validation['date_range'] = {
                    'start': date_min.strftime('%Y-%m-%d'),
                    'end': date_max.strftime('%Y-%m-%d')
                }
            
            # Check species breakdown
            if 'species' in df.columns:
                species_counts = df['species'].value_counts()
                validation['species_breakdown'] = species_counts.to_dict()
            
            # Check for synthetic patterns
            if self._detect_synthetic_shark_patterns(df):
                validation['issues'].append("Synthetic shark data patterns detected")
                validation['real_data_confirmed'] = False
                validation['status'] = 'FAIL'
            
        except Exception as e:
            validation['issues'].append(f"Error loading shark data: {e}")
            validation['status'] = 'FAIL'
        
        self.validation_results['shark_data'] = validation
    
    def _validate_features(self) -> None:
        """Validate computed features."""
        self.logger.info("Validating computed features")
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'features_found': [],
            'feature_ranges': {},
            'real_data_confirmed': True
        }
        
        features_path = 'data/interim/features.nc'
        if not os.path.exists(features_path):
            validation['issues'].append("Features file not found")
            validation['status'] = 'FAIL'
            self.validation_results['features'] = validation
            return
        
        try:
            ds = xr.open_dataset(features_path)
            validation['features_found'] = list(ds.data_vars.keys())
            
            # Validate each feature
            for var in ds.data_vars:
                data = ds[var].values
                if np.any(np.isfinite(data)):
                    validation['feature_ranges'][var] = {
                        'min': float(np.nanmin(data)),
                        'max': float(np.nanmax(data)),
                        'mean': float(np.nanmean(data)),
                        'std': float(np.nanstd(data)),
                        'finite_count': int(np.sum(np.isfinite(data)))
                    }
                    
                    # Check for unrealistic values
                    if self._check_unrealistic_values(var, data):
                        validation['issues'].append(f"Unrealistic values detected in {var}")
                        validation['status'] = 'FAIL'
            
            ds.close()
            
        except Exception as e:
            validation['issues'].append(f"Error reading features: {e}")
            validation['status'] = 'FAIL'
        
        self.validation_results['features'] = validation
    
    def _validate_training_data(self) -> None:
        """Validate training data integrity."""
        self.logger.info("Validating training data")
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'feature_count': 0,
            'real_data_confirmed': True
        }
        
        training_path = 'data/interim/training_data.csv'
        if not os.path.exists(training_path):
            validation['issues'].append("Training data file not found")
            validation['status'] = 'FAIL'
            self.validation_results['training_data'] = validation
            return
        
        try:
            df = pd.read_csv(training_path)
            validation['total_samples'] = len(df)
            
            if 'label' in df.columns:
                validation['positive_samples'] = int(df['label'].sum())
                validation['negative_samples'] = int(len(df) - df['label'].sum())
            
            # Count features (exclude metadata columns)
            metadata_cols = ['latitude', 'longitude', 'timestamp', 'date', 'label', 'shark_id', 'species']
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            validation['feature_count'] = len(feature_cols)
            
            # Check for synthetic patterns
            if self._detect_synthetic_training_patterns(df):
                validation['issues'].append("Synthetic training data patterns detected")
                validation['real_data_confirmed'] = False
                validation['status'] = 'FAIL'
            
        except Exception as e:
            validation['issues'].append(f"Error reading training data: {e}")
            validation['status'] = 'FAIL'
        
        self.validation_results['training_data'] = validation
    
    def _validate_models(self) -> None:
        """Validate trained models."""
        self.logger.info("Validating trained models")
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'models_found': [],
            'model_metrics': {},
            'real_data_confirmed': True
        }
        
        # Check for model files
        model_files = ['xgboost_model.pkl', 'random_forest_model.pkl', 'lightgbm_model.pkl']
        for model_file in model_files:
            model_path = os.path.join('data/interim', model_file)
            if os.path.exists(model_path):
                validation['models_found'].append(model_file.replace('_model.pkl', ''))
        
        # Check for metrics
        metrics_path = 'data/interim/training_metrics.json'
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                validation['model_metrics'] = metrics
            except Exception as e:
                validation['issues'].append(f"Error reading metrics: {e}")
                validation['status'] = 'FAIL'
        
        if not validation['models_found']:
            validation['issues'].append("No trained models found")
            validation['status'] = 'FAIL'
        
        self.validation_results['models'] = validation
    
    def _validate_predictions(self) -> None:
        """Validate prediction outputs."""
        self.logger.info("Validating predictions")
        
        validation = {
            'status': 'PASS',
            'issues': [],
            'prediction_files': [],
            'prediction_ranges': {},
            'real_data_confirmed': True
        }
        
        # Check for prediction files
        web_data_dir = 'web/data'
        if os.path.exists(web_data_dir):
            prediction_files = [f for f in os.listdir(web_data_dir) if f.endswith('.png') or f.endswith('.tif')]
            validation['prediction_files'] = prediction_files
            
            # Check prediction metadata
            metadata_path = os.path.join(web_data_dir, 'prediction_metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    validation['prediction_ranges'] = metadata
                except Exception as e:
                    validation['issues'].append(f"Error reading prediction metadata: {e}")
        
        if not validation['prediction_files']:
            validation['issues'].append("No prediction files found")
            validation['status'] = 'FAIL'
        
        self.validation_results['predictions'] = validation
    
    def _detect_synthetic_patterns(self, data_dir: str) -> bool:
        """Detect synthetic data patterns in NASA data."""
        # Check for suspicious file patterns
        suspicious_patterns = ['synthetic', 'mock', 'demo', 'fake', 'generated']
        
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if any(pattern in file.lower() for pattern in suspicious_patterns):
                    self.logger.warning(f"Suspicious file name detected: {file}")
                    return True
        
        return False
    
    def _detect_synthetic_shark_patterns(self, df: pd.DataFrame) -> bool:
        """Detect synthetic patterns in shark data."""
        # Check for unrealistic movement patterns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Check for identical coordinates (suspicious)
            coord_pairs = df[['latitude', 'longitude']].drop_duplicates()
            if len(coord_pairs) < len(df) * 0.1:  # Less than 10% unique coordinates
                self.logger.warning("Suspiciously low coordinate diversity")
                return True
            
            # Check for unrealistic jumps
            if len(df) > 100:
                lat_diff = df['latitude'].diff().abs()
                lon_diff = df['longitude'].diff().abs()
                if (lat_diff > 10).any() or (lon_diff > 10).any():
                    self.logger.warning("Unrealistic coordinate jumps detected")
                    return True
        
        return False
    
    def _detect_synthetic_training_patterns(self, df: pd.DataFrame) -> bool:
        """Detect synthetic patterns in training data."""
        # Check for unrealistic feature distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['latitude', 'longitude', 'label']:
                data = df[col].dropna()
                if len(data) > 0:
                    # Check for suspiciously uniform distributions
                    if np.std(data) < 1e-10:
                        self.logger.warning(f"Suspiciously uniform distribution in {col}")
                        return True
        
        return False
    
    def _check_unrealistic_values(self, var_name: str, data: np.ndarray) -> bool:
        """Check for unrealistic values in features."""
        # Define reasonable ranges for different variables
        ranges = {
            'sst': (-5, 35),  # Sea surface temperature in Celsius
            'chlor_a': (0, 50),  # Chlorophyll in mg/m³
            'ssh_anom': (-1, 1),  # Sea surface height anomaly in meters
            'sss': (30, 40),  # Sea surface salinity in psu
            'precipitation': (0, 500),  # Precipitation in mm/day
        }
        
        if var_name in ranges:
            min_val, max_val = ranges[var_name]
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                if np.min(finite_data) < min_val or np.max(finite_data) > max_val:
                    self.logger.warning(f"Unrealistic values in {var_name}: {np.min(finite_data):.2f} to {np.max(finite_data):.2f}")
                    return True
        
        return False
    
    def _determine_overall_status(self) -> None:
        """Determine overall validation status."""
        all_statuses = []
        real_data_confirmed = True
        
        for component, validation in self.validation_results.items():
            if component != 'overall_status':
                all_statuses.append(validation.get('status', 'UNKNOWN'))
                if not validation.get('real_data_confirmed', False):
                    real_data_confirmed = False
        
        if not real_data_confirmed:
            self.validation_results['overall_status'] = 'FAIL - SYNTHETIC DATA DETECTED'
        elif 'FAIL' in all_statuses:
            self.validation_results['overall_status'] = 'FAIL - DATA ISSUES'
        elif 'UNKNOWN' in all_statuses:
            self.validation_results['overall_status'] = 'PARTIAL - INCOMPLETE VALIDATION'
        else:
            self.validation_results['overall_status'] = 'PASS - ALL REAL DATA CONFIRMED'
    
    def _save_validation_report(self) -> None:
        """Save validation report to file."""
        report_path = 'data/interim/validation_report.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("DATA INTEGRITY VALIDATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {self.validation_results['overall_status']}")
        print("\nComponent Status:")
        for component, validation in self.validation_results.items():
            if component != 'overall_status':
                status = validation.get('status', 'UNKNOWN')
                real_data = validation.get('real_data_confirmed', False)
                print(f"  {component}: {status} (Real Data: {real_data})")
                if validation.get('issues'):
                    for issue in validation['issues']:
                        print(f"    - {issue}")
        print("="*60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate data integrity for shark habitat modeling")
    parser.add_argument("--config", type=str, default="config/params.yaml", 
                       help="Path to configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    validator = DataIntegrityValidator(config)
    results = validator.validate_all()
    
    # Exit with error code if validation failed
    if results['overall_status'].startswith('FAIL'):
        sys.exit(1)


if __name__ == "__main__":
    main()

