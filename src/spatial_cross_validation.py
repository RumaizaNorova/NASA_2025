#!/usr/bin/env python3
"""
Spatial Cross-Validation for Shark Habitat Prediction
Implement spatial cross-validation to prevent data leakage and ensure realistic performance metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SpatialCrossValidator:
    """Implement spatial cross-validation for habitat prediction"""
    
    def __init__(self):
        self.data_dir = Path("data/interim")
        self.output_dir = Path("data/interim")
        
    def load_oceanographic_data(self):
        """Load oceanographic-only training data"""
        print("🔍 Loading oceanographic training data...")
        
        data_path = self.data_dir / 'training_data_oceanographic_only.csv'
        if not data_path.exists():
            print("  ❌ Oceanographic data not found. Please run create_oceanographic_features.py first")
            return None
        
        df = pd.read_csv(data_path)
        print(f"  ✅ Loaded data: {len(df):,} samples")
        print(f"  🦈 Shark observations: {len(df[df.target==1]):,}")
        print(f"  ❌ Negative samples: {len(df[df.target==0]):,}")
        
        return df
    
    def create_spatial_folds(self, df, n_folds=5):
        """Create spatial folds based on geographic regions"""
        print(f"🗺️ Creating {n_folds} spatial folds...")
        
        # Create spatial regions based on latitude/longitude
        df['spatial_region'] = self._create_spatial_regions(df['latitude'], df['longitude'], n_folds)
        
        # Create folds based on spatial regions
        spatial_folds = []
        for region in range(n_folds):
            region_mask = df['spatial_region'] == region
            spatial_folds.append(region_mask)
        
        # Check fold distribution
        print("  📊 Spatial fold distribution:")
        for i, fold_mask in enumerate(spatial_folds):
            fold_size = fold_mask.sum()
            fold_sharks = df[fold_mask]['target'].sum()
            print(f"    Fold {i+1}: {fold_size:5,} samples ({fold_sharks:4,} sharks)")
        
        return spatial_folds
    
    def _create_spatial_regions(self, lat, lon, n_regions):
        """Create spatial regions for cross-validation"""
        # Create a grid-based spatial partitioning
        # This ensures that nearby locations are in the same fold
        
        # Normalize coordinates to [0, 1]
        lat_norm = (lat - lat.min()) / (lat.max() - lat.min())
        lon_norm = (lon - lon.min()) / (lon.max() - lon.min())
        
        # Create grid cells
        grid_size = int(np.ceil(np.sqrt(n_regions)))
        
        # Assign regions based on grid position
        lat_grid = (lat_norm * grid_size).astype(int)
        lon_grid = (lon_norm * grid_size).astype(int)
        
        # Combine grid positions
        grid_pos = lat_grid * grid_size + lon_grid
        
        # Map to n_regions
        regions = grid_pos % n_regions
        
        return regions
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("🔧 Preparing features for modeling...")
        
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in [
            'target', 'latitude', 'longitude', 'datetime', 'shark_id', 
            'species', 'timestamp', 'date', 'spatial_region'
        ]]
        
        # Separate features and target
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"  🔤 Encoding categorical variables: {list(categorical_cols)}")
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        missing_cols = X.columns[X.isnull().any()].tolist()
        if missing_cols:
            print(f"  🔧 Filling missing values in: {missing_cols}")
            X[missing_cols] = X[missing_cols].fillna(X[missing_cols].median())
        
        print(f"  ✅ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        
        return X, y, feature_cols
    
    def spatial_cross_validate(self, X, y, spatial_folds, model_type='random_forest'):
        """Perform spatial cross-validation"""
        print(f"🔄 Performing spatial cross-validation with {model_type}...")
        
        n_folds = len(spatial_folds)
        fold_scores = []
        fold_predictions = []
        fold_targets = []
        
        for fold_idx, fold_mask in enumerate(spatial_folds):
            print(f"  📁 Fold {fold_idx + 1}/{n_folds}")
            
            # Split data
            X_test = X[fold_mask]
            y_test = y[fold_mask]
            X_train = X[~fold_mask]
            y_train = y[~fold_mask]
            
            print(f"    Train: {len(X_train):,} samples ({y_train.sum():,} sharks)")
            print(f"    Test:  {len(X_test):,} samples ({y_test.sum():,} sharks)")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            fold_scores.append(auc_score)
            
            print(f"    ROC-AUC: {auc_score:.4f}")
            
            # Store predictions for overall evaluation
            fold_predictions.extend(y_pred_proba)
            fold_targets.extend(y_test)
        
        return fold_scores, fold_predictions, fold_targets
    
    def evaluate_spatial_performance(self, fold_scores, fold_predictions, fold_targets):
        """Evaluate spatial cross-validation performance"""
        print("📊 Evaluating spatial cross-validation performance...")
        
        # Calculate overall metrics
        overall_auc = roc_auc_score(fold_targets, fold_predictions)
        
        # Calculate fold statistics
        mean_auc = np.mean(fold_scores)
        std_auc = np.std(fold_scores)
        min_auc = np.min(fold_scores)
        max_auc = np.max(fold_scores)
        
        print(f"  📈 Overall ROC-AUC: {overall_auc:.4f}")
        print(f"  📊 Fold Statistics:")
        print(f"    Mean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"    Min ROC-AUC:  {min_auc:.4f}")
        print(f"    Max ROC-AUC:  {max_auc:.4f}")
        
        # Check if performance is realistic
        is_realistic = 0.60 <= mean_auc <= 0.80
        
        if is_realistic:
            print("  ✅ Performance is realistic for habitat prediction")
        else:
            print("  ⚠️ Performance may be unrealistic - check for data leakage")
        
        return {
            'overall_auc': overall_auc,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'min_auc': min_auc,
            'max_auc': max_auc,
            'fold_scores': fold_scores,
            'is_realistic': is_realistic
        }
    
    def run_spatial_cross_validation(self):
        """Run spatial cross-validation"""
        print("🚀 Running Spatial Cross-Validation")
        print("=" * 50)
        
        try:
            # Load data
            df = self.load_oceanographic_data()
            if df is None:
                return False
            
            # Create spatial folds
            spatial_folds = self.create_spatial_folds(df, n_folds=5)
            
            # Prepare features
            X, y, feature_cols = self.prepare_features(df)
            
            # Perform spatial cross-validation
            fold_scores, fold_predictions, fold_targets = self.spatial_cross_validate(
                X, y, spatial_folds, model_type='random_forest'
            )
            
            # Evaluate performance
            results = self.evaluate_spatial_performance(fold_scores, fold_predictions, fold_targets)
            
            # Save results
            results_path = self.output_dir / 'spatial_cv_results.json'
            import json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"  ✅ Results saved to: {results_path}")
            
            print("\n" + "=" * 50)
            print("🎉 SPATIAL CROSS-VALIDATION COMPLETED!")
            print(f"📊 Mean ROC-AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
            print(f"📈 Overall ROC-AUC: {results['overall_auc']:.4f}")
            print(f"✅ Spatial validation prevents data leakage")
            print(f"✅ Performance is {'realistic' if results['is_realistic'] else 'unrealistic'}")
            print("✅ Ready for habitat prediction modeling")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    validator = SpatialCrossValidator()
    return validator.run_spatial_cross_validation()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
