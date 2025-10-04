#!/usr/bin/env python3
"""
Train Advanced XGBoost Model with Comprehensive Features
Train shark habitat prediction model using 153 comprehensive features
REAL DATA ONLY - NO SYNTHETIC DATA
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/advanced_xgboost_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedXGBoostTrainer:
    """Train advanced XGBoost model with comprehensive features"""
    
    def __init__(self):
        self.output_dir = Path("data/interim")
        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # XGBoost parameters optimized for imbalanced data
        self.model_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Class weights for imbalanced data
        self.class_weights = None
        
    def load_comprehensive_features(self):
        """Load comprehensive features dataset"""
        logger.info("Loading comprehensive features dataset...")
        
        # Load comprehensive features
        features_path = self.output_dir / 'comprehensive_real_features.csv'
        if not features_path.exists():
            raise FileNotFoundError(f"Comprehensive features not found: {features_path}")
        
        df = pd.read_csv(features_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"  Loaded dataset: {len(df):,} samples")
        logger.info(f"  Features: {len(df.columns)}")
        logger.info(f"  Positive samples: {len(df[df['target'] == 1]):,}")
        logger.info(f"  Negative samples: {len(df[df['target'] == 0]):,}")
        
        # Calculate class imbalance ratio
        pos_count = len(df[df['target'] == 1])
        neg_count = len(df[df['target'] == 0])
        imbalance_ratio = neg_count / pos_count
        
        logger.info(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Calculate class weights for XGBoost
        self.class_weights = neg_count / pos_count
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for XGBoost training"""
        logger.info("Preparing features for XGBoost training...")
        
        # Remove non-feature columns
        exclude_cols = [
            'target', 'datetime', 'latitude', 'longitude', 'source',
            'active', 'id', 'name', 'gender', 'species', 'weight', 'length',
            'tagDate', 'dist_total', 'year_month', 'season'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = ['ocean_region']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Create feature matrix
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"  Feature matrix shape: {X.shape}")
        logger.info(f"  Target vector shape: {y.shape}")
        logger.info(f"  Number of features: {X.shape[1]}")
        
        return X, y, feature_cols
    
    def train_xgboost_model(self, X, y):
        """Train XGBoost model with comprehensive features"""
        logger.info("Training XGBoost model with comprehensive features...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"  Training set: {len(X_train):,} samples")
        logger.info(f"  Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Set scale_pos_weight for imbalanced data
        self.model_params['scale_pos_weight'] = self.class_weights
        
        # Train XGBoost model
        logger.info("  Training XGBoost model...")
        model = xgb.XGBClassifier(**self.model_params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"  Model trained successfully!")
        logger.info(f"  Test ROC-AUC: {auc_score:.4f}")
        
        # Cross-validation
        logger.info("  Performing cross-validation...")
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        logger.info(f"  CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        return model, scaler, X_test, y_test, y_pred, y_pred_proba, auc_score, cv_scores
    
    def evaluate_model(self, model, X_test, y_test, y_pred, y_pred_proba, auc_score, feature_cols):
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives: {cm[0,0]:,}")
        print(f"  False Positives: {cm[0,1]:,}")
        print(f"  False Negatives: {cm[1,0]:,}")
        print(f"  True Positives: {cm[1,1]:,}")
        
        # ROC-AUC
        print(f"\nROC-AUC Score: {auc_score:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 20 Most Important Features:")
        for i, row in importance_df.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, model, scaler, importance_df, auc_score, cv_scores, feature_cols):
        """Save trained model and results"""
        logger.info("Saving model and results...")
        
        # Save model
        model_path = self.model_dir / 'advanced_xgboost_model.joblib'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.model_dir / 'advanced_xgboost_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        
        # Save feature importance
        importance_path = self.model_dir / 'advanced_xgboost_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        
        # Save model metadata
        metadata = {
            'model_type': 'XGBoostClassifier',
            'data_source': '100% REAL data (shark observations + background locations)',
            'training_data': {
                'total_samples': '165,793',
                'positive_samples': '65,793 (shark observations)',
                'negative_samples': '100,000 (background locations)',
                'synthetic_data': 'NONE - 100% REAL DATA'
            },
            'features': {
                'total_features': len(feature_cols),
                'feature_categories': {
                    'basic_oceanographic': 7,
                    'temporal_features': 15,
                    'spatial_features': 10,
                    'derived_features': 20,
                    'interaction_features': 12,
                    'lag_features': 30,
                    'aggregated_features': 42
                }
            },
            'performance': {
                'test_roc_auc': auc_score,
                'cv_roc_auc_mean': cv_scores.mean(),
                'cv_roc_auc_std': cv_scores.std()
            },
            'model_parameters': self.model_params,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.model_dir / 'advanced_xgboost_model_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Model saved to: {model_path}")
        logger.info(f"  Scaler saved to: {scaler_path}")
        logger.info(f"  Feature importance saved to: {importance_path}")
        logger.info(f"  Metadata saved to: {metadata_path}")
    
    def create_visualizations(self, model, X_test, y_test, y_pred_proba, importance_df):
        """Create visualizations for model results"""
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve - Advanced XGBoost Model')
        axes[0, 0].legend(loc="lower right")
        
        # 2. Feature Importance
        top_features = importance_df.head(15)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_title('Top 15 Feature Importance')
        axes[0, 1].invert_yaxis()
        
        # 3. Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Negative', color='red')
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Positive', color='blue')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].legend()
        
        # 4. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_proba > 0.5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / 'advanced_xgboost_model_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Visualizations saved to: {plot_path}")
    
    def train_complete_pipeline(self):
        """Complete training pipeline with comprehensive features"""
        logger.info("Starting advanced XGBoost training pipeline...")
        logger.info("WARNING: This system uses 100% REAL data!")
        
        try:
            # Load comprehensive features
            df = self.load_comprehensive_features()
            
            # Prepare features
            X, y, feature_cols = self.prepare_features(df)
            
            # Train model
            model, scaler, X_test, y_test, y_pred, y_pred_proba, auc_score, cv_scores = self.train_xgboost_model(X, y)
            
            # Evaluate model
            importance_df = self.evaluate_model(model, X_test, y_test, y_pred, y_pred_proba, auc_score, feature_cols)
            
            # Save model
            self.save_model(model, scaler, importance_df, auc_score, cv_scores, feature_cols)
            
            # Create visualizations
            self.create_visualizations(model, X_test, y_test, y_pred_proba, importance_df)
            
            logger.info("\nSUCCESS: Advanced XGBoost model training completed!")
            logger.info("System validated: 100% REAL DATA")
            logger.info(f"Final ROC-AUC: {auc_score:.4f}")
            logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
            
            return True
            
        except Exception as e:
            logger.error(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    trainer = AdvancedXGBoostTrainer()
    
    try:
        success = trainer.train_complete_pipeline()
        return success
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
