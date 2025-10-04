#!/usr/bin/env python3
"""
Overfitting Prevention Strategies for AI-Enhanced Shark Habitat Prediction
Implements robust techniques to prevent overfitting in heavy model training
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

from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold, 
    cross_val_score, validation_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

class OverfittingPrevention:
    """Comprehensive overfitting prevention strategies"""
    
    def __init__(self, config_path='config/params_ai_enhanced.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {}
        
    def load_config(self):
        """Load configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def implement_cross_validation_strategies(self):
        """Implement robust cross-validation strategies"""
        print("🔧 Implementing cross-validation strategies...")
        
        cv_strategies = {
            'stratified_kfold': {
                'description': 'Stratified K-Fold for class balance',
                'implementation': StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            },
            'temporal_split': {
                'description': 'Temporal split for time series data',
                'implementation': TimeSeriesSplit(n_splits=5)
            },
            'group_kfold': {
                'description': 'Group K-Fold by shark individual',
                'implementation': GroupKFold(n_splits=5)
            }
        }
        
        # Update configuration
        self.config['cv']['strategies'] = cv_strategies
        self.config['cv']['primary_strategy'] = 'stratified_kfold'
        self.config['cv']['validation_splits'] = 5
        
        print("  ✅ Cross-validation strategies configured")
        return cv_strategies
    
    def implement_regularization_techniques(self):
        """Implement regularization techniques"""
        print("🔧 Implementing regularization techniques...")
        
        regularization_config = {
            'xgboost': {
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'max_depth': 6,  # Limit tree depth
                'min_child_weight': 3,  # Minimum samples per leaf
                'subsample': 0.8,  # Row sampling
                'colsample_bytree': 0.8,  # Column sampling
                'colsample_bylevel': 0.8,  # Level-wise sampling
                'colsample_bynode': 0.8,  # Node-wise sampling
                'gamma': 0.1,  # Minimum loss reduction
                'learning_rate': 0.05,  # Lower learning rate
                'n_estimators': 500  # Moderate number of trees
            },
            'lightgbm': {
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'max_depth': 6,  # Limit tree depth
                'min_child_samples': 20,  # Minimum samples per leaf
                'subsample': 0.8,  # Row sampling
                'colsample_bytree': 0.8,  # Column sampling
                'feature_fraction': 0.8,  # Feature sampling
                'bagging_fraction': 0.8,  # Bagging fraction
                'bagging_freq': 5,  # Bagging frequency
                'min_gain_to_split': 0.1,  # Minimum gain to split
                'learning_rate': 0.05,  # Lower learning rate
                'num_leaves': 31,  # Limit number of leaves
                'n_estimators': 500  # Moderate number of trees
            },
            'random_forest': {
                'max_depth': 10,  # Limit tree depth
                'min_samples_split': 10,  # Minimum samples to split
                'min_samples_leaf': 5,  # Minimum samples per leaf
                'max_features': 'sqrt',  # Feature sampling
                'bootstrap': True,  # Bootstrap sampling
                'max_samples': 0.8,  # Sample fraction
                'n_estimators': 300,  # Moderate number of trees
                'random_state': 42
            }
        }
        
        # Update configuration
        self.config['regularization'] = regularization_config
        
        print("  ✅ Regularization techniques configured")
        return regularization_config
    
    def implement_early_stopping(self):
        """Implement early stopping strategies"""
        print("🔧 Implementing early stopping strategies...")
        
        early_stopping_config = {
            'enable': True,
            'patience': 10,  # Stop if no improvement for 10 rounds
            'min_delta': 0.001,  # Minimum improvement threshold
            'monitor_metric': 'roc_auc',  # Metric to monitor
            'mode': 'max',  # Maximize the metric
            'restore_best_weights': True,  # Restore best weights
            'validation_fraction': 0.2,  # Validation set fraction
            'n_iter_no_change': 10,  # Iterations without change
            'tol': 1e-4  # Tolerance for improvement
        }
        
        # Update configuration
        self.config['early_stopping'] = early_stopping_config
        
        print("  ✅ Early stopping strategies configured")
        return early_stopping_config
    
    def implement_feature_selection(self):
        """Implement feature selection to prevent overfitting"""
        print("🔧 Implementing feature selection...")
        
        feature_selection_config = {
            'enable': True,
            'methods': ['variance_threshold', 'correlation_filter', 'mutual_info', 'recursive_elimination'],
            'variance_threshold': 0.01,  # Remove low variance features
            'correlation_threshold': 0.95,  # Remove highly correlated features
            'mutual_info_k': 10,  # Top K features by mutual information
            'recursive_elimination': {
                'cv': 3,  # Cross-validation folds
                'scoring': 'roc_auc',  # Scoring metric
                'n_features_to_select': 20  # Number of features to select
            },
            'feature_importance_threshold': 0.01  # Minimum importance threshold
        }
        
        # Update configuration
        self.config['feature_selection'] = feature_selection_config
        
        print("  ✅ Feature selection configured")
        return feature_selection_config
    
    def implement_ensemble_methods(self):
        """Implement ensemble methods to reduce overfitting"""
        print("🔧 Implementing ensemble methods...")
        
        ensemble_config = {
            'enable': True,
            'methods': ['voting', 'stacking', 'bagging'],
            'voting': {
                'voting_type': 'soft',  # Soft voting
                'weights': [0.4, 0.3, 0.3]  # Equal weights
            },
            'stacking': {
                'cv': 5,  # Cross-validation folds
                'final_estimator': 'logistic_regression',  # Final estimator
                'passthrough': False  # Don't pass original features
            },
            'bagging': {
                'n_estimators': 10,  # Number of base estimators
                'max_samples': 0.8,  # Sample fraction
                'max_features': 0.8,  # Feature fraction
                'bootstrap': True,  # Bootstrap sampling
                'bootstrap_features': False  # Don't bootstrap features
            }
        }
        
        # Update configuration
        self.config['ensemble'] = ensemble_config
        
        print("  ✅ Ensemble methods configured")
        return ensemble_config
    
    def implement_data_augmentation(self):
        """Implement data augmentation techniques"""
        print("🔧 Implementing data augmentation...")
        
        augmentation_config = {
            'enable': True,
            'methods': ['noise_injection', 'feature_perturbation', 'temporal_augmentation'],
            'noise_injection': {
                'noise_std': 0.01,  # Standard deviation of noise
                'augmentation_factor': 2  # Multiply samples by this factor
            },
            'feature_perturbation': {
                'perturbation_std': 0.05,  # Standard deviation of perturbation
                'augmentation_factor': 1.5  # Multiply samples by this factor
            },
            'temporal_augmentation': {
                'time_window': 7,  # Days to shift
                'augmentation_factor': 1.2  # Multiply samples by this factor
            }
        }
        
        # Update configuration
        self.config['data_augmentation'] = augmentation_config
        
        print("  ✅ Data augmentation configured")
        return augmentation_config
    
    def implement_validation_monitoring(self):
        """Implement validation monitoring"""
        print("🔧 Implementing validation monitoring...")
        
        monitoring_config = {
            'enable': True,
            'metrics': ['roc_auc', 'pr_auc', 'f1_score', 'precision', 'recall'],
            'validation_frequency': 10,  # Validate every N iterations
            'overfitting_threshold': 0.05,  # Max difference between train/val
            'early_stopping_patience': 15,  # Stop if overfitting detected
            'learning_rate_reduction': {
                'factor': 0.5,  # Reduce LR by this factor
                'patience': 5,  # Patience for LR reduction
                'min_lr': 1e-6  # Minimum learning rate
            },
            'model_checkpointing': {
                'save_best': True,  # Save best model
                'save_frequency': 50,  # Save every N iterations
                'max_checkpoints': 5  # Maximum number of checkpoints
            }
        }
        
        # Update configuration
        self.config['validation_monitoring'] = monitoring_config
        
        print("  ✅ Validation monitoring configured")
        return monitoring_config
    
    def create_overfitting_prevention_pipeline(self):
        """Create comprehensive overfitting prevention pipeline"""
        print("🔧 Creating overfitting prevention pipeline...")
        
        pipeline_config = {
            'steps': [
                'data_validation',
                'feature_selection',
                'cross_validation',
                'regularization',
                'early_stopping',
                'ensemble_training',
                'validation_monitoring'
            ],
            'data_validation': {
                'check_data_leakage': True,
                'check_temporal_leakage': True,
                'check_spatial_leakage': True,
                'check_feature_leakage': True
            },
            'training_strategy': {
                'use_validation_set': True,
                'validation_size': 0.2,
                'stratify_validation': True,
                'temporal_split': True
            }
        }
        
        # Update configuration
        self.config['overfitting_prevention'] = pipeline_config
        
        print("  ✅ Overfitting prevention pipeline configured")
        return pipeline_config
    
    def save_updated_config(self):
        """Save updated configuration"""
        print("💾 Saving updated configuration...")
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"  ✅ Configuration saved to: {self.config_path}")
    
    def create_validation_script(self):
        """Create validation script for overfitting detection"""
        print("🔧 Creating validation script...")
        
        script_content = '''#!/usr/bin/env python3
"""
Overfitting Detection and Validation Script
Monitors training progress and detects overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
import json
from pathlib import Path

def detect_overfitting(train_scores, val_scores, threshold=0.05):
    """Detect overfitting based on train/validation score difference"""
    
    if len(train_scores) != len(val_scores):
        raise ValueError("Train and validation scores must have same length")
    
    overfitting_detected = []
    for i, (train_score, val_score) in enumerate(zip(train_scores, val_scores)):
        difference = train_score - val_score
        overfitting_detected.append(difference > threshold)
    
    return overfitting_detected

def plot_training_curves(train_scores, val_scores, save_path=None):
    """Plot training curves to visualize overfitting"""
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_scores, label='Training Score', color='blue')
    plt.plot(val_scores, label='Validation Score', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Training vs Validation Scores')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def validate_model_performance(model, X_train, y_train, X_val, y_val):
    """Validate model performance and detect overfitting"""
    
    # Get predictions
    train_pred = model.predict_proba(X_train)[:, 1]
    val_pred = model.predict_proba(X_val)[:, 1]
    
    # Calculate scores
    train_score = roc_auc_score(y_train, train_pred)
    val_score = roc_auc_score(y_val, val_pred)
    
    # Detect overfitting
    overfitting = detect_overfitting([train_score], [val_score])
    
    results = {
        'train_score': train_score,
        'val_score': val_score,
        'overfitting_detected': overfitting[0],
        'score_difference': train_score - val_score
    }
    
    return results

if __name__ == "__main__":
    print("Overfitting Detection and Validation")
    print("=" * 40)
    
    # Example usage
    train_scores = [0.8, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99]
    val_scores = [0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.84, 0.83]
    
    overfitting = detect_overfitting(train_scores, val_scores)
    print(f"Overfitting detected: {overfitting}")
    
    plot_training_curves(train_scores, val_scores, 'training_curves.png')
'''
        
        script_path = Path('validate_overfitting.py')
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"  ✅ Validation script created: {script_path}")
    
    def run_overfitting_prevention(self):
        """Run comprehensive overfitting prevention setup"""
        print("🚀 Overfitting Prevention Setup")
        print("=" * 50)
        
        try:
            # Implement all strategies
            self.implement_cross_validation_strategies()
            self.implement_regularization_techniques()
            self.implement_early_stopping()
            self.implement_feature_selection()
            self.implement_ensemble_methods()
            self.implement_data_augmentation()
            self.implement_validation_monitoring()
            self.create_overfitting_prevention_pipeline()
            
            # Save updated configuration
            self.save_updated_config()
            
            # Create validation script
            self.create_validation_script()
            
            print("\n" + "=" * 50)
            print("🎉 OVERFITTING PREVENTION SETUP COMPLETED!")
            print("✅ Cross-validation strategies implemented")
            print("✅ Regularization techniques configured")
            print("✅ Early stopping strategies enabled")
            print("✅ Feature selection methods added")
            print("✅ Ensemble methods configured")
            print("✅ Data augmentation techniques added")
            print("✅ Validation monitoring enabled")
            print("✅ Configuration updated")
            print("✅ Validation script created")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    prevention = OverfittingPrevention()
    return prevention.run_overfitting_prevention()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
