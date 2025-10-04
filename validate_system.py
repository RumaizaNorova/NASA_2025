#!/usr/bin/env python3
"""
System Validation Script for AI-Enhanced Shark Habitat Prediction
Validates all critical components before heavy model training
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def validate_dependencies():
    """Validate all critical dependencies"""
    print("🔍 Validating dependencies...")
    
    critical_deps = [
        'numpy', 'pandas', 'sklearn', 'xgboost', 'lightgbm', 
        'openai', 'optuna', 'imblearn', 'shap', 'matplotlib', 'seaborn'
    ]
    
    missing_deps = []
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        return False
    
    print("✅ All dependencies validated")
    return True

def validate_shark_data():
    """Validate shark observation data"""
    print("\n🔍 Validating shark data...")
    
    try:
        df = pd.read_csv('../sharks_cleaned.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"  📊 Total observations: {len(df):,}")
        print(f"  📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  🦈 Unique sharks: {df['id'].nunique()}")
        print(f"  🐟 Species: {df['species'].nunique()}")
        print(f"  📈 Years covered: {df['datetime'].dt.year.nunique()}")
        
        # Check data quality
        missing_coords = df[['latitude', 'longitude']].isnull().sum().sum()
        if missing_coords > 0:
            print(f"  ⚠️  Missing coordinates: {missing_coords}")
        
        # Temporal distribution
        yearly_counts = df['datetime'].dt.year.value_counts().sort_index()
        print(f"  📊 Yearly distribution: {dict(yearly_counts)}")
        
        return True, df
        
    except Exception as e:
        print(f"  ❌ Error reading shark data: {e}")
        return False, None

def validate_training_data():
    """Validate current training data"""
    print("\n🔍 Validating training data...")
    
    try:
        # Check for expanded data first
        if os.path.exists('data/interim/training_data_expanded.csv'):
            df = pd.read_csv('data/interim/training_data_expanded.csv')
            print("  📊 Using expanded training data")
        else:
            df = pd.read_csv('data/interim/training_data.csv')
            print("  📊 Using original training data")
        
        print(f"  📊 Training samples: {len(df):,}")
        
        # Check if using expanded data
        if 'target' in df.columns:
            print(f"  🎯 Target distribution: {df['target'].value_counts().to_dict()}")
            pos_ratio = df['target'].mean()
            print(f"  ⚖️  Positive class ratio: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
            if pos_ratio >= 0.1:
                print(f"  ✅ GOOD CLASS BALANCE!")
                return True, df
            else:
                print(f"  ⚠️  Class imbalance detected")
                return False, df
        else:
            print(f"  🎯 Label distribution: {df['label'].value_counts().to_dict()}")
            pos_ratio = df['label'].mean()
            print(f"  ⚖️  Positive class ratio: {pos_ratio:.4f} ({pos_ratio*100:.2f}%)")
            
            if pos_ratio < 0.01:
                print(f"  ⚠️  EXTREME CLASS IMBALANCE DETECTED!")
                return False, df
            
            return True, df
        
    except Exception as e:
        print(f"  ❌ Error reading training data: {e}")
        return False, None

def validate_satellite_data():
    """Validate NASA satellite data availability"""
    print("\n🔍 Validating satellite data...")
    
    satellite_dirs = [
        'data/raw/gpm_precipitation',
        'data/raw/measures_ssh', 
        'data/raw/mur_sst',
        'data/raw/oscar_currents',
        'data/raw/pace_chl',
        'data/raw/smap_salinity'
    ]
    
    total_files = 0
    for dir_path in satellite_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.nc')]
            print(f"  📡 {os.path.basename(dir_path)}: {len(files)} files")
            total_files += len(files)
        else:
            print(f"  ❌ {dir_path}: Directory not found")
    
    print(f"  📊 Total satellite files: {total_files}")
    
    if total_files < 10:
        print(f"  ⚠️  LIMITED SATELLITE DATA - Only {total_files} files available")
        return False
    
    return True

def validate_configuration():
    """Validate configuration files"""
    print("\n🔍 Validating configuration...")
    
    config_files = [
        'config/params_ai_enhanced.yaml',
        'config/params.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"  ✅ {config_file}")
            except Exception as e:
                print(f"  ❌ {config_file}: {e}")
                return False
        else:
            print(f"  ❌ {config_file}: Not found")
            return False
    
    return True

def validate_environment():
    """Validate environment setup"""
    print("\n🔍 Validating environment...")
    
    # Check .env file
    if os.path.exists('.env'):
        print("  ✅ .env file exists")
        
        # Check for placeholder values
        with open('.env', 'r') as f:
            content = f.read()
            if 'your_openai_api_key_here' in content:
                print("  ⚠️  OpenAI API key not configured")
            if 'your_earthdata_token_here' in content:
                print("  ⚠️  Earthdata token not configured")
    else:
        print("  ❌ .env file not found")
        return False
    
    # Check directories
    required_dirs = ['data/raw', 'data/interim', 'logs', 'config']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}: Not found")
            return False
    
    return True

def generate_recommendations(shark_df, training_df):
    """Generate recommendations based on validation results"""
    print("\n📋 RECOMMENDATIONS:")
    
    if shark_df is not None and training_df is not None:
        shark_count = len(shark_df)
        training_count = len(training_df)
        
        if shark_count > training_count * 10:
            print(f"  🚀 EXPAND TRAINING DATA: Use full {shark_count:,} observations instead of {training_count:,}")
        
        # Check temporal coverage
        shark_years = shark_df['datetime'].dt.year.nunique()
        if shark_years > 1:
            print(f"  📅 EXPAND TEMPORAL COVERAGE: Use {shark_years} years of data instead of 6 days")
        
        # Check class imbalance
        pos_ratio = training_df['label'].mean()
        if pos_ratio < 0.01:
            print(f"  ⚖️  FIX CLASS IMBALANCE: Implement SMOTE, ADASYN, or cost-sensitive learning")
    
    print("  🔧 INSTALL MISSING DEPENDENCIES: Run 'pip install -r requirements_ai_enhanced.txt'")
    print("  🔑 CONFIGURE API KEYS: Update .env file with real credentials")
    print("  📡 EXPAND SATELLITE DATA: Download multiple years of NASA data")
    print("  🛡️  IMPLEMENT OVERFITTING PREVENTION: Add regularization, cross-validation")

def main():
    """Main validation function"""
    print("🚀 AI-Enhanced Shark Habitat Prediction System Validation")
    print("=" * 60)
    
    # Run all validations
    deps_ok = validate_dependencies()
    shark_ok, shark_df = validate_shark_data()
    training_ok, training_df = validate_training_data()
    satellite_ok = validate_satellite_data()
    config_ok = validate_configuration()
    env_ok = validate_environment()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY:")
    
    validations = [
        ("Dependencies", deps_ok),
        ("Shark Data", shark_ok),
        ("Training Data", training_ok),
        ("Satellite Data", satellite_ok),
        ("Configuration", config_ok),
        ("Environment", env_ok)
    ]
    
    all_passed = True
    for name, status in validations:
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {name}")
        if not status:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED - System ready for training!")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED - Fix before training")
        generate_recommendations(shark_df, training_df)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
