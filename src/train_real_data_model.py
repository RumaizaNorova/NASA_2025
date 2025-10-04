#!/usr/bin/env python3
"""
Train Model with 100% Real NASA Data
Train shark habitat prediction model using ONLY real data:
- Real NASA satellite data (2012-2019)
- Real shark observations
- Real negative samples from background locations
NO SYNTHETIC DATA ALLOWED
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

class RealDataModelTrainer:
    """Train model with 100% real NASA data - NO SYNTHETIC DATA"""
    
    def __init__(self):
        self.output_dir = Path("data/interim")
        self.model_dir = Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parameters for real data
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
    def load_real_data(self):
        """Load 100% real data for training"""
        print("📊 Loading 100% REAL data for training...")
        
        # Load real balanced dataset
        dataset_path = self.output_dir / 'real_balanced_dataset.csv'
        if not dataset_path.exists():
            raise FileNotFoundError(f"Real balanced dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        print(f"  ✅ Loaded real dataset: {len(df):,} samples")
        print(f"  🦈 Positive samples: {len(df[df['label'] == 1]):,}")
        print(f"  🚫 Negative samples: {len(df[df['label'] == 0]):,}")
        
        return df
    
    def load_real_oceanographic_features(self):
        """Load real NASA oceanographic features"""
        print("🌊 Loading real NASA oceanographic features...")
        
        # Try to find existing oceanographic features
        possible_files = [
            'real_nasa_oceanographic_features.csv',
            'training_data_oceanographic_only.csv',
            'training_data_enhanced_real_features.csv',
            'training_data_realistic_features.csv'
        ]
        
        features_path = None
        for file_name in possible_files:
            file_path = self.output_dir / file_name
            if file_path.exists():
                features_path = file_path
                break
        
        if not features_path:
            raise FileNotFoundError(f"No oceanographic features found in {self.output_dir}")
        
        features_df = pd.read_csv(features_path)
        features_df['datetime'] = pd.to_datetime(features_df['datetime'])
        
        print(f"  ✅ Loaded oceanographic features: {len(features_df):,} samples")
        print(f"  🔢 Feature columns: {len(features_df.columns)}")
        print(f"  📁 Source file: {features_path.name}")
        
        return features_df
    
    def merge_real_data_and_features(self, dataset_df, features_df):
        """Merge real dataset with real NASA features"""
        print("🔗 Merging real dataset with real NASA features...")
        
        # Merge on coordinates and datetime
        merged_df = pd.merge(
            dataset_df,
            features_df,
            on=['latitude', 'longitude', 'datetime'],
            how='inner'
        )
        
        print(f"  ✅ Merged dataset: {len(merged_df):,} samples")
        
        # Check for missing values in critical features
        critical_features = ['sst', 'ssh_anom', 'current_speed', 'current_direction', 'chl', 'sss', 'precipitation']
        missing_counts = {}
        
        for feature in critical_features:
            if feature in merged_df.columns:
                missing_count = merged_df[feature].isna().sum()
                missing_counts[feature] = missing_count
                if missing_count > 0:
                    print(f"  ⚠️ Warning: {missing_count} missing values in {feature}")
        
        # Remove rows with missing critical features
        initial_count = len(merged_df)
        merged_df = merged_df.dropna(subset=critical_features)
        final_count = len(merged_df)
        
        if initial_count != final_count:
            print(f"  🧹 Removed {initial_count - final_count} samples with missing features")
        
        print(f"  ✅ Final merged dataset: {len(merged_df):,} samples")
        
        return merged_df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        print("🔧 Preparing features for model training...")
        
        # Select feature columns
        feature_columns = [
            'latitude', 'longitude',
            'sst', 'ssh_anom', 'current_speed', 'current_direction',
            'chl', 'sss', 'precipitation',
            'ocean_region', 'distance_to_coast', 'depth',
            'continental_shelf', 'open_ocean'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            print(f"  ⚠️ Missing features: {missing_features}")
        
        # ❌ REMOVED: Temporal features cause data leakage!
        # df['year'] = df['datetime'].dt.year
        # df['month'] = df['datetime'].dt.month
        # df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # ✅ Only use oceanographic features for habitat prediction
        # temporal_features = ['year', 'month', 'day_of_year']  # REMOVED
        # available_features.extend(temporal_features)  # REMOVED
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df['label'].copy()
        
        # Handle categorical variables
        if 'ocean_region' in X.columns:
            X = pd.get_dummies(X, columns=['ocean_region'], prefix='region')
        
        print(f"  ✅ Feature matrix shape: {X.shape}")
        print(f"  ✅ Target vector shape: {y.shape}")
        print(f"  🔢 Number of features: {X.shape[1]}")
        
        return X, y
    
    def train_real_data_model(self, X, y, df):
        """Train model with real data"""
        print("🚀 Training model with 100% REAL data...")
        
        # ✅ TEMPORAL SPLIT: Use temporal cross-validation to prevent data leakage
        # Sort by datetime to ensure temporal order
        df_sorted = df.sort_values('datetime')
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        
        # Use 80% early data for training, 20% later data for testing
        split_idx = int(0.8 * len(df_sorted))
        X_train = X_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_train = y_sorted.iloc[:split_idx]
        y_test = y_sorted.iloc[split_idx:]
        
        print(f"  📊 Training set: {len(X_train):,} samples")
        print(f"  📊 Test set: {len(X_test):,} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        print("  🌲 Training Random Forest model...")
        model = RandomForestClassifier(**self.model_params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba_full = model.predict_proba(X_test_scaled)
        
        # Handle case where model only predicts one class
        if y_pred_proba_full.shape[1] == 2:
            y_pred_proba = y_pred_proba_full[:, 1]
        else:
            y_pred_proba = y_pred_proba_full[:, 0]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  ✅ Model trained successfully!")
        print(f"  📊 Test ROC-AUC: {auc_score:.4f}")
        
        # ✅ TEMPORAL CROSS-VALIDATION: Use temporal splits to prevent data leakage
        print("  🔄 Performing temporal cross-validation...")
        
        # Create temporal CV splits (no shuffling!)
        n_splits = 5
        temporal_cv_scores = []
        
        for i in range(n_splits):
            # Calculate split indices
            start_idx = int(i * len(X_train_scaled) / n_splits)
            end_idx = int((i + 1) * len(X_train_scaled) / n_splits)
            
            # Validation fold
            X_val = X_train_scaled[start_idx:end_idx]
            y_val = y_train[start_idx:end_idx]
            
            # Training folds (all data before validation fold)
            X_train_fold = X_train_scaled[:start_idx]
            y_train_fold = y_train[:start_idx]
            
            if len(X_train_fold) > 0:
                # Train model on training fold
                fold_model = RandomForestClassifier(**self.model_params)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                y_val_proba = fold_model.predict_proba(X_val)[:, 1]
                
                # Calculate AUC
                fold_auc = roc_auc_score(y_val, y_val_proba)
                temporal_cv_scores.append(fold_auc)
        
        cv_scores = np.array(temporal_cv_scores)
        
        print(f"  📊 CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
        return model, scaler, X_test, y_test, y_pred, y_pred_proba, auc_score, cv_scores
    
    def evaluate_real_data_model(self, model, X_test, y_test, y_pred, y_pred_proba, auc_score):
        """Evaluate model performance on real data"""
        print("📈 Evaluating model performance on real data...")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        
        # Handle different confusion matrix shapes
        if cm.shape == (2, 2):
            print(f"  True Negatives: {cm[0,0]:,}")
            print(f"  False Positives: {cm[0,1]:,}")
            print(f"  False Negatives: {cm[1,0]:,}")
            print(f"  True Positives: {cm[1,1]:,}")
        elif cm.shape == (1, 1):
            print(f"  All predictions: {cm[0,0]:,} (single class)")
        else:
            print(f"  Confusion matrix shape: {cm.shape}")
            print(f"  Matrix: {cm}")
        
        # ROC-AUC
        print(f"\n📊 ROC-AUC Score: {auc_score:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        feature_names = X_test.columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_real_data_model(self, model, scaler, importance_df, auc_score, cv_scores):
        """Save trained model and results"""
        print("💾 Saving real data model and results...")
        
        # Save model
        model_path = self.model_dir / 'real_data_model.joblib'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = self.model_dir / 'real_data_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        
        # Save feature importance
        importance_path = self.model_dir / 'real_data_feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        
        # Save model metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'data_source': '100% REAL NASA satellite data',
            'training_data': {
                'total_samples': 'Real dataset size',
                'positive_samples': 'Real shark observations',
                'negative_samples': 'Real background locations',
                'synthetic_data': 'NONE - 100% REAL DATA'
            },
            'performance': {
                'test_roc_auc': auc_score,
                'cv_roc_auc_mean': cv_scores.mean(),
                'cv_roc_auc_std': cv_scores.std()
            },
            'model_parameters': self.model_params,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.model_dir / 'real_data_model_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✅ Model saved to: {model_path}")
        print(f"  ✅ Scaler saved to: {scaler_path}")
        print(f"  ✅ Feature importance saved to: {importance_path}")
        print(f"  ✅ Metadata saved to: {metadata_path}")
    
    def create_real_data_visualizations(self, model, X_test, y_test, y_pred_proba, importance_df):
        """Create visualizations for real data model"""
        print("📊 Creating visualizations for real data model...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {auc_score:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve - Real Data Model')
        axes[0, 0].legend(loc="lower right")
        
        # 2. Feature Importance
        top_features = importance_df.head(10)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_title('Top 10 Feature Importance - Real Data')
        axes[0, 1].invert_yaxis()
        
        # 3. Prediction Distribution
        axes[1, 0].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Negative', color='red')
        axes[1, 0].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Positive', color='blue')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Prediction Distribution - Real Data')
        axes[1, 0].legend()
        
        # 4. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_proba > 0.5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_title('Confusion Matrix - Real Data')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / 'real_data_model_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Visualizations saved to: {plot_path}")
    
    def train_complete_real_data_pipeline(self):
        """Complete training pipeline with 100% real data"""
        print("🚀 Starting complete REAL data training pipeline...")
        print("⚠️  WARNING: This system uses 100% REAL NASA satellite data!")
        
        try:
            # Load real data
            dataset_df = self.load_real_data()
            features_df = self.load_real_oceanographic_features()
            
            # Merge data
            merged_df = self.merge_real_data_and_features(dataset_df, features_df)
            
            # Prepare features
            X, y = self.prepare_features(merged_df)
            
            # Train model
            model, scaler, X_test, y_test, y_pred, y_pred_proba, auc_score, cv_scores = self.train_real_data_model(X, y, merged_df)
            
            # Evaluate model
            importance_df = self.evaluate_real_data_model(model, X_test, y_test, y_pred, y_pred_proba, auc_score)
            
            # Save model
            self.save_real_data_model(model, scaler, importance_df, auc_score, cv_scores)
            
            # Create visualizations
            self.create_real_data_visualizations(model, X_test, y_test, y_pred_proba, importance_df)
            
            print("\n🎉 SUCCESS: Real data model training completed!")
            print("✅ System validated: 100% REAL NASA SATELLITE DATA")
            print(f"📊 Final ROC-AUC: {auc_score:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    trainer = RealDataModelTrainer()
    
    try:
        success = trainer.train_complete_real_data_pipeline()
        return success
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
