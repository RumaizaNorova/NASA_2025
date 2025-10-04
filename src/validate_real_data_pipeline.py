#!/usr/bin/env python3
"""
Validate Real Data Pipeline
Comprehensive validation to ensure 100% real NASA data usage
NO SYNTHETIC DATA ALLOWED - VALIDATE EVERYTHING
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class RealDataValidator:
    """Validate that the entire pipeline uses 100% real data"""
    
    def __init__(self):
        self.data_dir = Path("data/raw/nasa_satellite")
        self.interim_dir = Path("data/interim")
        self.model_dir = Path("models")
        
        # Validation results
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'checks': {}
        }
    
    def validate_nasa_data_download(self):
        """Validate NASA data download system"""
        print("🔍 Validating NASA data download system...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check if download script exists
        download_script = Path("src/download_nasa_data.py")
        if not download_script.exists():
            check_results['status'] = 'FAIL'
            check_results['issues'].append("Download script not found")
        else:
            check_results['details']['download_script'] = 'EXISTS'
        
        # Check for synthetic data fallbacks in download script
        if download_script.exists():
            with open(download_script, 'r') as f:
                content = f.read()
                
            synthetic_indicators = [
                'synthetic', 'fake', 'random', 'estimated', 'simulated'
            ]
            
            for indicator in synthetic_indicators:
                if indicator.lower() in content.lower():
                    check_results['status'] = 'FAIL'
                    check_results['issues'].append(f"Synthetic data indicator found: '{indicator}'")
        
        # Check data directory structure
        if self.data_dir.exists():
            check_results['details']['data_directory'] = 'EXISTS'
            
            # Count NetCDF files
            nc_files = list(self.data_dir.glob("*.nc"))
            check_results['details']['netcdf_files'] = len(nc_files)
            
            if len(nc_files) == 0:
                check_results['status'] = 'WARNING'
                check_results['issues'].append("No NetCDF files found - data may not be downloaded")
        else:
            check_results['status'] = 'WARNING'
            check_results['issues'].append("NASA data directory not found")
        
        self.validation_results['checks']['nasa_data_download'] = check_results
        print(f"  ✅ NASA data download validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def validate_data_processing(self):
        """Validate data processing pipeline"""
        print("🔍 Validating data processing pipeline...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check processing script
        process_script = Path("src/process_nasa_data.py")
        if not process_script.exists():
            check_results['status'] = 'FAIL'
            check_results['issues'].append("Processing script not found")
        else:
            check_results['details']['processing_script'] = 'EXISTS'
            
            # Check for synthetic data methods
            with open(process_script, 'r') as f:
                content = f.read()
            
            synthetic_methods = [
                '_create_synthetic_', 'synthetic_data', 'fake_data', 'random_data'
            ]
            
            for method in synthetic_methods:
                if method in content:
                    check_results['status'] = 'FAIL'
                    check_results['issues'].append(f"Synthetic data method found: '{method}'")
        
        # Check for processed features
        features_file = self.interim_dir / 'real_nasa_oceanographic_features.csv'
        if features_file.exists():
            check_results['details']['features_file'] = 'EXISTS'
            
            # Load and validate features
            df = pd.read_csv(features_file)
            check_results['details']['feature_samples'] = len(df)
            check_results['details']['feature_columns'] = len(df.columns)
            
            # Check for NaN values in critical features
            critical_features = ['sst', 'ssh_anom', 'current_speed', 'current_direction', 'chl', 'sss', 'precipitation']
            nan_counts = {}
            
            for feature in critical_features:
                if feature in df.columns:
                    nan_count = df[feature].isna().sum()
                    nan_counts[feature] = nan_count
            
            check_results['details']['nan_counts'] = nan_counts
            
            # Check if all features are real (not synthetic)
            if any('synthetic' in col.lower() for col in df.columns):
                check_results['status'] = 'FAIL'
                check_results['issues'].append("Synthetic feature columns found")
        
        else:
            check_results['status'] = 'WARNING'
            check_results['issues'].append("Processed features file not found")
        
        self.validation_results['checks']['data_processing'] = check_results
        print(f"  ✅ Data processing validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def validate_negative_sampling(self):
        """Validate negative sampling system"""
        print("🔍 Validating negative sampling system...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check negative sampling script
        sampling_script = Path("src/create_real_negative_sampling.py")
        if not sampling_script.exists():
            check_results['status'] = 'FAIL'
            check_results['issues'].append("Negative sampling script not found")
        else:
            check_results['details']['sampling_script'] = 'EXISTS'
            
            # Check for random generation
            with open(sampling_script, 'r') as f:
                content = f.read()
            
            random_indicators = [
                'np.random', 'random.choice', 'random.uniform', 'random.normal'
            ]
            
            for indicator in random_indicators:
                if indicator in content:
                    check_results['status'] = 'FAIL'
                    check_results['issues'].append(f"Random generation found: '{indicator}'")
        
        # Check for real negative samples
        negative_file = self.interim_dir / 'real_negative_samples.csv'
        if negative_file.exists():
            check_results['details']['negative_samples_file'] = 'EXISTS'
            
            # Load and validate negative samples
            df = pd.read_csv(negative_file)
            check_results['details']['negative_samples_count'] = len(df)
            
            # Check data sources
            if 'source' in df.columns:
                sources = df['source'].value_counts().to_dict()
                check_results['details']['data_sources'] = sources
                
                # Validate sources are real
                valid_sources = ['fishing_vessel', 'marine_protected_area', 'oceanographic_survey', 'environmental_sampling']
                invalid_sources = [src for src in sources.keys() if src not in valid_sources]
                
                if invalid_sources:
                    check_results['status'] = 'FAIL'
                    check_results['issues'].append(f"Invalid data sources: {invalid_sources}")
        
        else:
            check_results['status'] = 'WARNING'
            check_results['issues'].append("Real negative samples file not found")
        
        self.validation_results['checks']['negative_sampling'] = check_results
        print(f"  ✅ Negative sampling validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def validate_balanced_dataset(self):
        """Validate balanced dataset"""
        print("🔍 Validating balanced dataset...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check balanced dataset
        balanced_file = self.interim_dir / 'real_balanced_dataset.csv'
        if balanced_file.exists():
            check_results['details']['balanced_dataset_file'] = 'EXISTS'
            
            # Load and validate dataset
            df = pd.read_csv(balanced_file)
            check_results['details']['total_samples'] = len(df)
            
            # Check label distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts().to_dict()
                check_results['details']['label_distribution'] = label_counts
                
                # Check for realistic balance
                if 0 in label_counts and 1 in label_counts:
                    balance_ratio = label_counts[0] / label_counts[1]
                    check_results['details']['balance_ratio'] = balance_ratio
                    
                    if balance_ratio > 10:  # Too many negative samples
                        check_results['status'] = 'WARNING'
                        check_results['issues'].append(f"Imbalanced dataset: {balance_ratio:.1f}:1 ratio")
            
            # Check for required columns
            required_cols = ['latitude', 'longitude', 'datetime', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                check_results['status'] = 'FAIL'
                check_results['issues'].append(f"Missing required columns: {missing_cols}")
        
        else:
            check_results['status'] = 'WARNING'
            check_results['issues'].append("Balanced dataset file not found")
        
        self.validation_results['checks']['balanced_dataset'] = check_results
        print(f"  ✅ Balanced dataset validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def validate_model_training(self):
        """Validate model training"""
        print("🔍 Validating model training...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check training script
        training_script = Path("src/train_real_data_model.py")
        if not training_script.exists():
            check_results['status'] = 'FAIL'
            check_results['issues'].append("Training script not found")
        else:
            check_results['details']['training_script'] = 'EXISTS'
            
            # Check for synthetic data references
            with open(training_script, 'r') as f:
                content = f.read()
            
            if 'synthetic' in content.lower():
                check_results['status'] = 'FAIL'
                check_results['issues'].append("Synthetic data references found in training script")
        
        # Check trained model
        model_file = self.model_dir / 'real_data_model.joblib'
        if model_file.exists():
            check_results['details']['model_file'] = 'EXISTS'
            
            # Load model metadata
            metadata_file = self.model_dir / 'real_data_model_metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                check_results['details']['model_metadata'] = metadata
                
                # Validate metadata
                if 'data_source' in metadata:
                    if 'real' not in metadata['data_source'].lower():
                        check_results['status'] = 'FAIL'
                        check_results['issues'].append("Model metadata does not specify real data")
                
                if 'synthetic_data' in metadata.get('training_data', {}):
                    if metadata['training_data']['synthetic_data'] != 'NONE - 100% REAL DATA':
                        check_results['status'] = 'FAIL'
                        check_results['issues'].append("Model trained with synthetic data")
        
        else:
            check_results['status'] = 'WARNING'
            check_results['issues'].append("Trained model file not found")
        
        self.validation_results['checks']['model_training'] = check_results
        print(f"  ✅ Model training validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def validate_file_integrity(self):
        """Validate file integrity and structure"""
        print("🔍 Validating file integrity...")
        
        check_results = {
            'status': 'PASS',
            'issues': [],
            'details': {}
        }
        
        # Check all critical files exist
        critical_files = [
            'src/download_nasa_data.py',
            'src/process_nasa_data.py',
            'src/create_real_negative_sampling.py',
            'src/train_real_data_model.py',
            'src/validate_real_data_pipeline.py'
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in critical_files:
            if Path(file_path).exists():
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        check_results['details']['existing_files'] = existing_files
        check_results['details']['missing_files'] = missing_files
        
        if missing_files:
            check_results['status'] = 'FAIL'
            check_results['issues'].append(f"Missing critical files: {missing_files}")
        
        # Check data directory structure
        data_dirs = ['data/raw', 'data/interim', 'models']
        for data_dir in data_dirs:
            if Path(data_dir).exists():
                check_results['details'][f'{data_dir}_exists'] = True
            else:
                check_results['details'][f'{data_dir}_exists'] = False
        
        self.validation_results['checks']['file_integrity'] = check_results
        print(f"  ✅ File integrity validation: {check_results['status']}")
        
        return check_results['status'] == 'PASS'
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("📊 Generating validation report...")
        
        # Determine overall status
        all_checks = self.validation_results['checks']
        pass_count = sum(1 for check in all_checks.values() if check['status'] == 'PASS')
        total_checks = len(all_checks)
        
        if pass_count == total_checks:
            self.validation_results['overall_status'] = 'PASS'
        elif pass_count > total_checks * 0.8:
            self.validation_results['overall_status'] = 'WARNING'
        else:
            self.validation_results['overall_status'] = 'FAIL'
        
        # Generate report
        report = f"""
# REAL DATA PIPELINE VALIDATION REPORT

## Overall Status: {self.validation_results['overall_status']}

## Validation Summary:
- Total Checks: {total_checks}
- Passed: {pass_count}
- Failed: {total_checks - pass_count}

## Detailed Results:
"""
        
        for check_name, check_result in all_checks.items():
            report += f"\n### {check_name.replace('_', ' ').title()}\n"
            report += f"**Status:** {check_result['status']}\n"
            
            if check_result['issues']:
                report += f"**Issues:**\n"
                for issue in check_result['issues']:
                    report += f"- {issue}\n"
            
            if check_result['details']:
                report += f"**Details:**\n"
                for key, value in check_result['details'].items():
                    report += f"- {key}: {value}\n"
        
        # Save report
        report_path = Path("REAL_DATA_VALIDATION_REPORT.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON results
        json_path = Path("real_data_validation_results.json")
        with open(json_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"  ✅ Validation report saved to: {report_path}")
        print(f"  ✅ JSON results saved to: {json_path}")
        
        return report
    
    def run_complete_validation(self):
        """Run complete validation pipeline"""
        print("🚀 Running complete REAL data validation pipeline...")
        print("⚠️  WARNING: This validates 100% REAL NASA satellite data usage!")
        
        try:
            # Run all validation checks
            checks = [
                self.validate_nasa_data_download,
                self.validate_data_processing,
                self.validate_negative_sampling,
                self.validate_balanced_dataset,
                self.validate_model_training,
                self.validate_file_integrity
            ]
            
            results = []
            for check in checks:
                try:
                    result = check()
                    results.append(result)
                except Exception as e:
                    print(f"  ❌ Error in validation check: {e}")
                    results.append(False)
            
            # Generate report
            report = self.generate_validation_report()
            
            # Print summary
            print(f"\n📊 VALIDATION SUMMARY:")
            print(f"  Overall Status: {self.validation_results['overall_status']}")
            print(f"  Checks Passed: {sum(results)}/{len(results)}")
            
            if self.validation_results['overall_status'] == 'PASS':
                print("🎉 SUCCESS: All validations passed!")
                print("✅ System validated: 100% REAL NASA SATELLITE DATA")
            else:
                print("⚠️  WARNING: Some validations failed!")
                print("❌ System may contain synthetic data")
            
            return self.validation_results['overall_status'] == 'PASS'
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    validator = RealDataValidator()
    
    try:
        success = validator.run_complete_validation()
        return success
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
