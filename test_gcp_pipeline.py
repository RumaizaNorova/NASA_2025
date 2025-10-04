#!/usr/bin/env python3
"""
Test GCP Pipeline Locally
Test the complete pipeline before deploying to GCP
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

def test_data_pipeline():
    """Test the data processing pipeline"""
    print("🧪 Testing data processing pipeline...")
    
    try:
        # Test data loading
        from train_real_data_model import RealDataModelTrainer
        
        trainer = RealDataModelTrainer()
        
        # Test data loading
        print("  📊 Testing data loading...")
        dataset_df = trainer.load_real_data()
        print(f"    ✅ Loaded {len(dataset_df):,} samples")
        
        # Test feature loading
        print("  🌊 Testing feature loading...")
        features_df = trainer.load_real_oceanographic_features()
        print(f"    ✅ Loaded {len(features_df):,} feature samples")
        
        # Test data merging
        print("  🔗 Testing data merging...")
        merged_df = trainer.merge_real_data_and_features(dataset_df, features_df)
        print(f"    ✅ Merged {len(merged_df):,} samples")
        
        # Test feature preparation
        print("  🔧 Testing feature preparation...")
        X, y = trainer.prepare_features(merged_df)
        print(f"    ✅ Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
        
        print("  ✅ Data pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Data pipeline test failed: {e}")
        return False

def test_model_training():
    """Test model training pipeline"""
    print("🧪 Testing model training pipeline...")
    
    try:
        from train_real_data_model import RealDataModelTrainer
        
        trainer = RealDataModelTrainer()
        
        # Load data
        dataset_df = trainer.load_real_data()
        features_df = trainer.load_real_oceanographic_features()
        merged_df = trainer.merge_real_data_and_features(dataset_df, features_df)
        X, y = trainer.prepare_features(merged_df)
        
        # Test model training (with smaller dataset for speed)
        print("  🚀 Testing model training...")
        sample_size = min(10000, len(X))
        X_sample = X.iloc[:sample_size]
        y_sample = y.iloc[:sample_size]
        merged_sample = merged_df.iloc[:sample_size]
        
        model, scaler, X_test, y_test, y_pred, y_pred_proba, auc_score, cv_scores = trainer.train_real_data_model(
            X_sample, y_sample, merged_sample
        )
        
        print(f"    ✅ Model trained with ROC-AUC: {auc_score:.4f}")
        print(f"    ✅ CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        print("  ✅ Model training test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model training test failed: {e}")
        return False

def test_gcp_components():
    """Test GCP-specific components"""
    print("🧪 Testing GCP components...")
    
    try:
        # Test GCP training script imports
        print("  📦 Testing GCP imports...")
        sys.path.append('gcp_deployment')
        from gcp_train import GCPModelTrainer
        print("    ✅ GCP imports successful")
        
        # Test GCP trainer initialization
        print("  🔧 Testing GCP trainer initialization...")
        trainer = GCPModelTrainer("test-project", "test-bucket")
        print("    ✅ GCP trainer initialized")
        
        # Test feature preparation
        print("  🔧 Testing GCP feature preparation...")
        
        # Create test data
        test_data = pd.DataFrame({
            'latitude': np.random.uniform(30, 50, 1000),
            'longitude': np.random.uniform(-80, -60, 1000),
            'datetime': pd.date_range('2012-01-01', periods=1000, freq='D'),
            'label': np.random.choice([0, 1], 1000)
        })
        
        # Add spatial features
        test_data['ocean_region'] = 'north_atlantic'
        test_data['distance_to_coast'] = np.abs(test_data['latitude']) * 111
        test_data['depth'] = 1000 + np.abs(test_data['latitude']) * 50
        test_data['continental_shelf'] = (test_data['depth'] < 200).astype(int)
        test_data['open_ocean'] = (test_data['depth'] > 1000).astype(int)
        
        X, y, merged_df = trainer.prepare_features(test_data, None)
        
        print(f"    ✅ Prepared {X.shape[0]:,} samples with {X.shape[1]} features")
        
        print("  ✅ GCP components test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ GCP components test failed: {e}")
        return False

def test_data_validation():
    """Test data validation pipeline"""
    print("🧪 Testing data validation pipeline...")
    
    try:
        from validate_real_data_pipeline import RealDataValidator
        
        validator = RealDataValidator()
        
        # Test individual validation checks
        print("  🔍 Testing NASA data validation...")
        nasa_valid = validator.validate_nasa_data_download()
        print(f"    NASA data validation: {'✅ PASS' if nasa_valid else '❌ FAIL'}")
        
        print("  🔍 Testing data processing validation...")
        processing_valid = validator.validate_data_processing()
        print(f"    Data processing validation: {'✅ PASS' if processing_valid else '❌ FAIL'}")
        
        print("  🔍 Testing model training validation...")
        training_valid = validator.validate_model_training()
        print(f"    Model training validation: {'✅ PASS' if training_valid else '❌ FAIL'}")
        
        print("  ✅ Data validation test completed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Data validation test failed: {e}")
        return False

def test_file_structure():
    """Test file structure and dependencies"""
    print("🧪 Testing file structure...")
    
    required_files = [
        'src/download_nasa_data.py',
        'src/process_nasa_data.py',
        'src/train_real_data_model.py',
        'src/validate_real_data_pipeline.py',
        'gcp_deployment/Dockerfile',
        'gcp_deployment/cloudbuild.yaml',
        'gcp_deployment/vertex_ai_config.yaml',
        'gcp_deployment/deploy.sh',
        'gcp_deployment/gcp_train.py',
        'gcp_deployment/requirements_gcp.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False
    else:
        print("  ✅ All required files present!")
        return True

def main():
    """Run all tests"""
    print("🚀 Testing GCP Pipeline Components")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Pipeline", test_data_pipeline),
        ("Model Training", test_model_training),
        ("GCP Components", test_gcp_components),
        ("Data Validation", test_data_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for GCP deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
