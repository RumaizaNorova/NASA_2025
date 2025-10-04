#!/usr/bin/env python3
"""
Performance Target Validation for AI-Enhanced Shark Habitat Prediction
Validates that all performance targets can be achieved with expanded data
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

class PerformanceValidator:
    """Validates performance targets with expanded data"""
    
    def __init__(self):
        self.config_path = 'config/params_ai_enhanced.yaml'
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self):
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_expanded_data(self):
        """Load expanded training data"""
        print("🔍 Loading expanded training data...")
        
        try:
            # Load expanded data
            df = pd.read_csv('data/interim/training_data_expanded.csv')
            
            print(f"  📊 Total samples: {len(df):,}")
            print(f"  🎯 Target distribution: {df['target'].value_counts().to_dict()}")
            print(f"  📅 Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Check data quality
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                print(f"  ⚠️  Missing values: {missing_values}")
            
            return df
            
        except Exception as e:
            print(f"  ❌ Error loading expanded data: {e}")
            return None
    
    def create_synthetic_features(self, df):
        """Create synthetic features for demonstration"""
        print("🔧 Creating synthetic features...")
        
        # Create synthetic oceanographic features
        np.random.seed(42)
        
        # Generate realistic oceanographic features
        n_samples = len(df)
        
        # Sea Surface Temperature (SST) features
        sst_base = np.random.normal(20, 5, n_samples)  # 20°C ± 5°C
        sst_grad = np.random.normal(0, 0.5, n_samples)  # SST gradient
        sst_front = np.random.exponential(0.1, n_samples)  # SST front strength
        
        # Chlorophyll features
        chl_base = np.random.lognormal(0, 1, n_samples)  # Log-normal distribution
        chl_grad = np.random.normal(0, 0.3, n_samples)  # Chlorophyll gradient
        chl_front = np.random.exponential(0.05, n_samples)  # Chlorophyll front
        
        # Sea Surface Height (SSH) features
        ssh_anom = np.random.normal(0, 0.1, n_samples)  # SSH anomaly
        
        # Current features
        u_current = np.random.normal(0, 0.5, n_samples)  # U-component
        v_current = np.random.normal(0, 0.5, n_samples)  # V-component
        current_speed = np.sqrt(u_current**2 + v_current**2)  # Current speed
        
        # Oceanographic dynamics
        divergence = np.random.normal(0, 0.01, n_samples)  # Divergence
        vorticity = np.random.normal(0, 0.01, n_samples)  # Vorticity
        ow = np.random.normal(0, 0.005, n_samples)  # Okubo-Weiss parameter
        
        # Eddy features
        eddy_flag = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Eddy presence
        eddy_cyc = np.random.exponential(0.1, n_samples)  # Cyclonic eddy
        eddy_anti = np.random.exponential(0.1, n_samples)  # Anticyclonic eddy
        eddy_intensity = np.random.exponential(0.05, n_samples)  # Eddy intensity
        
        # Additional features
        current_divergence = np.random.normal(0, 0.01, n_samples)
        current_vorticity = np.random.normal(0, 0.01, n_samples)
        normal_strain = np.random.normal(0, 0.01, n_samples)
        shear_strain = np.random.normal(0, 0.01, n_samples)
        strain_rate = np.random.exponential(0.01, n_samples)
        current_direction = np.random.uniform(0, 360, n_samples)
        current_persistence = np.random.exponential(0.1, n_samples)
        
        # Salinity and precipitation
        sss = np.random.normal(35, 2, n_samples)  # Sea surface salinity
        precipitation = np.random.exponential(0.5, n_samples)  # Precipitation
        
        # Temporal features
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['seasonal_cycle'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['seasonal_cycle_2'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Bathymetry features (simplified)
        depth = np.random.exponential(1000, n_samples)  # Depth
        depth_grad = np.random.normal(0, 50, n_samples)  # Depth gradient
        distance_to_coast = np.random.exponential(100, n_samples)  # Distance to coast
        
        # Add all features to dataframe
        feature_data = {
            'sst': sst_base,
            'sst_grad': sst_grad,
            'sst_front': sst_front,
            'chl_log': np.log(chl_base + 1),
            'chl_grad': chl_grad,
            'chl_front': chl_front,
            'ssh_anom': ssh_anom,
            'u_current': u_current,
            'v_current': v_current,
            'current_speed': current_speed,
            'divergence': divergence,
            'vorticity': vorticity,
            'ow': ow,
            'eddy_flag': eddy_flag,
            'eddy_cyc': eddy_cyc,
            'eddy_anti': eddy_anti,
            'eddy_intensity': eddy_intensity,
            'current_divergence': current_divergence,
            'current_vorticity': current_vorticity,
            'normal_strain': normal_strain,
            'shear_strain': shear_strain,
            'strain_rate': strain_rate,
            'current_direction': current_direction,
            'current_persistence': current_persistence,
            'sss': sss,
            'precipitation': precipitation,
            'depth': depth,
            'depth_grad': depth_grad,
            'distance_to_coast': distance_to_coast
        }
        
        # Add features to dataframe
        for feature_name, feature_values in feature_data.items():
            df[feature_name] = feature_values
        
        print(f"  ✅ Created {len(feature_data)} synthetic features")
        print(f"  📊 Total features: {len(df.columns)}")
        
        return df
    
    def calculate_performance_targets(self, y_true, y_pred, y_pred_proba):
        """Calculate all performance targets"""
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # PR-AUC
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # F1-Score
        f1 = f1_score(y_true, y_pred)
        
        # Precision and Recall
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # True Skill Statistic (TSS)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tss = sensitivity + specificity - 1
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'tss': tss,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def validate_with_baseline_models(self, df):
        """Validate performance targets with baseline models"""
        print("🔧 Validating with baseline models...")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in [
            'latitude', 'longitude', 'datetime', 'target', 'shark_id', 
            'species', 'timestamp', 'date', 'ocean_region'  # Exclude categorical
        ]]
        
        X = df[feature_cols]
        y = df['target']
        
        # Handle categorical features
        if 'ocean_region' in df.columns:
            # One-hot encode ocean region
            ocean_dummies = pd.get_dummies(df['ocean_region'], prefix='ocean')
            X = pd.concat([X, ocean_dummies], axis=1)
        
        print(f"  📊 Features: {len(feature_cols)}")
        print(f"  📊 Samples: {len(X)}")
        print(f"  🎯 Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=5  # Handle class imbalance
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"  🔧 Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate performance
                performance = self.calculate_performance_targets(y_test, y_pred, y_pred_proba)
                results[model_name] = performance
                
                print(f"    📊 ROC-AUC: {performance['roc_auc']:.3f}")
                print(f"    📊 PR-AUC: {performance['pr_auc']:.3f}")
                print(f"    📊 F1-Score: {performance['f1_score']:.3f}")
                print(f"    📊 TSS: {performance['tss']:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def validate_performance_targets(self, results):
        """Validate that performance targets can be achieved"""
        print("\n🎯 Validating performance targets...")
        
        # Get targets from configuration
        targets = self.config['evaluation']['targets']
        
        target_validation = {
            'roc_auc': {
                'target': targets['roc_auc'],
                'achieved': False,
                'best_score': 0
            },
            'pr_auc': {
                'target': targets['pr_auc'],
                'achieved': False,
                'best_score': 0
            },
            'f1': {
                'target': targets['f1'],
                'achieved': False,
                'best_score': 0
            },
            'tss': {
                'target': targets['tss'],
                'achieved': False,
                'best_score': 0
            }
        }
        
        # Check each model's performance
        for model_name, model_results in results.items():
            if model_results is None:
                continue
                
            for metric, validation in target_validation.items():
                # Map metric names
                metric_key = metric
                if metric == 'f1':
                    metric_key = 'f1_score'
                
                if metric_key in model_results:
                    score = model_results[metric_key]
                    if score > validation['best_score']:
                        validation['best_score'] = score
                    if score >= validation['target']:
                        validation['achieved'] = True
        
        # Print results
        print("  📊 Performance Target Validation:")
        for metric, validation in target_validation.items():
            status = "✅ ACHIEVED" if validation['achieved'] else "❌ NOT ACHIEVED"
            print(f"    {metric.upper()}: {validation['best_score']:.3f} / {validation['target']:.3f} {status}")
        
        return target_validation
    
    def generate_performance_report(self, results, target_validation):
        """Generate comprehensive performance report"""
        print("\n📋 Generating performance report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(pd.read_csv('data/interim/training_data_expanded.csv')),
                'features_count': len([col for col in pd.read_csv('data/interim/training_data_expanded.csv').columns 
                                     if col not in ['latitude', 'longitude', 'datetime', 'target', 'shark_id', 'species', 'timestamp', 'date']]),
                'class_balance': pd.read_csv('data/interim/training_data_expanded.csv')['target'].mean()
            },
            'model_results': results,
            'target_validation': target_validation,
            'recommendations': []
        }
        
        # Generate recommendations
        if not target_validation['roc_auc']['achieved']:
            report['recommendations'].append("Increase model complexity or feature engineering for ROC-AUC")
        if not target_validation['pr_auc']['achieved']:
            report['recommendations'].append("Implement better class imbalance handling for PR-AUC")
        if not target_validation['f1']['achieved']:
            report['recommendations'].append("Optimize precision-recall balance for F1-Score")
        if not target_validation['tss']['achieved']:
            report['recommendations'].append("Improve sensitivity and specificity for TSS")
        
        # Save report
        report_path = 'data/interim/performance_validation_report.json'
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  ✅ Report saved to: {report_path}")
        
        return report
    
    def run_validation(self):
        """Run complete performance validation"""
        print("🚀 Performance Target Validation")
        print("=" * 50)
        
        try:
            # Load realistic features data
            if os.path.exists('data/interim/training_data_realistic_features.csv'):
                print("🔍 Loading realistic features data...")
                df = pd.read_csv('data/interim/training_data_realistic_features.csv')
                df['datetime'] = pd.to_datetime(df['datetime'])
                print(f"  📊 Total samples: {len(df):,}")
                print(f"  🔢 Total features: {len(df.columns)}")
            else:
                # Fallback to expanded data
                df = self.load_expanded_data()
                if df is None:
                    return False
                # Create synthetic features
                df = self.create_synthetic_features(df)
            
            # Validate with baseline models
            results = self.validate_with_baseline_models(df)
            
            # Validate performance targets
            target_validation = self.validate_performance_targets(results)
            
            # Generate report
            report = self.generate_performance_report(results, target_validation)
            
            print("\n" + "=" * 50)
            print("🎉 PERFORMANCE VALIDATION COMPLETED!")
            
            # Summary
            achieved_targets = sum(1 for v in target_validation.values() if v['achieved'])
            total_targets = len(target_validation)
            
            print(f"📊 Targets achieved: {achieved_targets}/{total_targets}")
            
            if achieved_targets == total_targets:
                print("🎉 ALL PERFORMANCE TARGETS CAN BE ACHIEVED!")
            else:
                print("⚠️  Some targets need improvement")
                print("📋 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
            
            return achieved_targets == total_targets
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    validator = PerformanceValidator()
    return validator.run_validation()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
