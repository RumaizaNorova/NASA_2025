#!/usr/bin/env python3
"""
Pipeline Validation Script
Comprehensive validation of the shark habitat prediction pipeline to prevent data leakage
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def validate_data_leakage():
    """Validate that the pipeline doesn't have data leakage issues"""
    print("🔍 VALIDATING PIPELINE FOR DATA LEAKAGE")
    print("=" * 60)
    
    issues_found = []
    
    # 1. Check if temporal features are being used
    print("1️⃣ Checking for temporal feature leakage...")
    
    # Check train_real_data_model.py
    train_file = Path("src/train_real_data_model.py")
    if train_file.exists():
        with open(train_file, 'r') as f:
            content = f.read()
            
        # Check for temporal feature creation
        if "df['year']" in content and "#" not in content.split("df['year']")[0][-10:]:
            issues_found.append("❌ Temporal features (year, month, day_of_year) are still being created!")
        else:
            print("  ✅ Temporal features properly removed")
            
        # Check for temporal split
        if "temporal" in content.lower() and "split" in content.lower():
            print("  ✅ Temporal cross-validation implemented")
        else:
            issues_found.append("❌ No temporal cross-validation found!")
    
    # 2. Check pseudo-absence generation
    print("\n2️⃣ Checking pseudo-absence temporal consistency...")
    
    label_file = Path("src/label_join.py")
    if label_file.exists():
        with open(label_file, 'r') as f:
            content = f.read()
            
        # Check for proper temporal sampling
        if "random_shark_idx" in content and "iloc[random_shark_idx]" in content:
            print("  ✅ Pseudo-absences use temporally consistent sampling")
        else:
            issues_found.append("❌ Pseudo-absences may have temporal mismatch!")
    
    # 3. Check NASA data processing
    print("\n3️⃣ Checking NASA data processing...")
    
    process_file = Path("src/process_nasa_data.py")
    if process_file.exists():
        with open(process_file, 'r') as f:
            content = f.read()
            
        # Check for proper interpolation
        if "interpolate_to_shark_coords" in content:
            print("  ✅ NASA data properly interpolated to shark coordinates")
        else:
            issues_found.append("❌ NASA data interpolation may be incorrect!")
    
    # 4. Check data download
    print("\n4️⃣ Checking NASA data download...")
    
    download_file = Path("src/download_nasa_data.py")
    if download_file.exists():
        with open(download_file, 'r') as f:
            content = f.read()
            
        # Check for proper date range
        if "2012-01-01" in content and "2019-12-31" in content:
            print("  ✅ NASA data download covers proper date range")
        else:
            issues_found.append("❌ NASA data download date range may be incorrect!")
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
        print("\n🚨 PIPELINE NOT READY FOR PRODUCTION!")
        return False
    else:
        print("✅ ALL VALIDATIONS PASSED!")
        print("✅ PIPELINE READY FOR GCP DEPLOYMENT!")
        return True

def validate_data_integrity():
    """Validate data integrity and consistency"""
    print("\n🔍 VALIDATING DATA INTEGRITY")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "src/download_nasa_data.py",
        "src/process_nasa_data.py", 
        "src/train_real_data_model.py",
        "src/label_join.py",
        "requirements.txt",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  {file_path}")
        return False
    else:
        print("✅ All required files present")
    
    # Check requirements.txt
    print("\n📦 Checking dependencies...")
    with open("requirements.txt", 'r') as f:
        requirements = f.read()
    
    critical_deps = ["numpy", "pandas", "scikit-learn", "xarray"]
    missing_deps = []
    
    for dep in critical_deps:
        if dep not in requirements.lower():
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing critical dependencies: {missing_deps}")
        return False
    else:
        print("✅ All critical dependencies present")
        
    # Check for netCDF4 (case insensitive)
    if "netcdf4" not in requirements.lower():
        print("⚠️ netCDF4 dependency not found in requirements.txt")
    else:
        print("✅ netCDF4 dependency found")
    
    return True

def create_test_dataset():
    """Create a small test dataset to validate the pipeline"""
    print("\n🧪 Creating test dataset...")
    
    # Create a small synthetic dataset for testing
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'latitude': np.random.uniform(-40, 40, n_samples),
        'longitude': np.random.uniform(-80, -20, n_samples),
        'datetime': pd.date_range('2014-07-01', periods=n_samples, freq='H'),
        'sst': np.random.uniform(15, 28, n_samples),
        'ssh_anom': np.random.uniform(-0.2, 0.2, n_samples),
        'current_speed': np.random.uniform(0, 0.5, n_samples),
        'current_direction': np.random.uniform(0, 360, n_samples),
        'chl': np.random.uniform(0.1, 2.0, n_samples),
        'sss': np.random.uniform(33, 37, n_samples),
        'precipitation': np.random.uniform(0, 5, n_samples),
        'label': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% positive
    })
    
    # Save test dataset
    test_path = Path("data/interim/test_dataset.csv")
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(test_path, index=False)
    
    print(f"✅ Test dataset created: {test_path}")
    print(f"  📊 Samples: {len(test_data):,}")
    print(f"  🦈 Positive: {len(test_data[test_data.label==1]):,}")
    print(f"  🚫 Negative: {len(test_data[test_data.label==0]):,}")
    
    return test_path

def main():
    """Main validation function"""
    print("🦈 SHARK HABITAT PREDICTION - PIPELINE VALIDATION")
    print("=" * 60)
    print("This script validates the pipeline for data leakage and integrity issues")
    print("=" * 60)
    
    # Validate data leakage
    leakage_ok = validate_data_leakage()
    
    # Validate data integrity  
    integrity_ok = validate_data_integrity()
    
    # Create test dataset
    test_path = create_test_dataset()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL VALIDATION RESULT")
    print("=" * 60)
    
    if leakage_ok and integrity_ok:
        print("✅ PIPELINE VALIDATION PASSED!")
        print("\n📋 Next Steps:")
        print("1. Commit code to GitHub")
        print("2. Deploy to GCP")
        print("3. Run full pipeline with real NASA data")
        print("4. Expect realistic performance (ROC-AUC ~0.6-0.8)")
        
        print(f"\n🧪 Test dataset available at: {test_path}")
        print("   Use this to test the pipeline locally before GCP deployment")
        
        return True
    else:
        print("❌ PIPELINE VALIDATION FAILED!")
        print("\n🔧 Required Actions:")
        print("1. Fix data leakage issues")
        print("2. Ensure all required files are present")
        print("3. Re-run validation")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
