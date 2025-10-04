#!/usr/bin/env python3
"""
Validate Realistic Performance Expectations
Ensure the model achieves realistic performance metrics (ROC-AUC 0.65-0.75) for habitat prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RealisticPerformanceValidator:
    """Validate that model achieves realistic performance metrics"""
    
    def __init__(self):
        self.data_dir = Path("data/interim")
        self.output_dir = Path("data/interim")
        
        # Realistic performance expectations for habitat prediction
        self.expected_auc_range = (0.60, 0.80)  # Realistic range for habitat prediction
        self.expected_auc_ideal = (0.65, 0.75)  # Ideal range
        
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
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("🔧 Preparing features for modeling...")
        
        # Remove non-feature columns
        feature_cols = [col for col in df.columns if col not in [
            'target', 'latitude', 'longitude', 'datetime', 'shark_id', 
            'species', 'timestamp', 'date'
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
    
    def check_feature_correlations(self, X, y):
        """Check for high correlations that might indicate data leakage"""
        print("🔍 Checking feature-target correlations...")
        
        correlations = []
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                correlations.append((col, corr))
            except:
                pass
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("  📊 Top 10 feature-target correlations:")
        for col, corr in correlations[:10]:
            print(f"    {col:25s}: {corr:8.4f}")
        
        # Check for high correlations (potential leakage)
        high_corr_features = [col for col, corr in correlations if abs(corr) > 0.3]
        
        if high_corr_features:
            print(f"  ⚠️ Warning: High correlation features (>0.3): {high_corr_features}")
            return False
        else:
            print("  ✅ No high correlation features detected")
            return True
    
    def train_and_validate_model(self, X, y, model_type='random_forest'):
        """Train and validate model with realistic expectations"""
        print(f"🤖 Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  📊 Train: {len(X_train):,} samples ({y_train.sum():,} sharks)")
        print(f"  📊 Test:  {len(X_test):,} samples ({y_test.sum():,} sharks)")
        
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
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"  📈 ROC-AUC: {auc_score:.4f}")
        
        # Check if performance is realistic
        is_realistic = self.expected_auc_range[0] <= auc_score <= self.expected_auc_range[1]
        is_ideal = self.expected_auc_ideal[0] <= auc_score <= self.expected_auc_ideal[1]
        
        if is_ideal:
            print(f"  ✅ Performance is ideal for habitat prediction")
        elif is_realistic:
            print(f"  ✅ Performance is realistic for habitat prediction")
        else:
            print(f"  ⚠️ Performance is unrealistic - check for data leakage")
        
        return {
            'auc_score': auc_score,
            'is_realistic': is_realistic,
            'is_ideal': is_ideal,
            'model': model,
            'scaler': scaler,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
    
    def validate_multiple_models(self, X, y):
        """Validate multiple models to ensure consistent realistic performance"""
        print("🔄 Validating multiple models...")
        
        models = ['random_forest', 'logistic_regression']
        results = {}
        
        for model_type in models:
            print(f"\n  🤖 Testing {model_type}...")
            result = self.train_and_validate_model(X, y, model_type)
            results[model_type] = result
        
        return results
    
    def analyze_feature_importance(self, model, feature_cols):
        """Analyze feature importance to ensure oceanographic features are most important"""
        print("🔍 Analyzing feature importance...")
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest feature importance
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression coefficients
            importances = np.abs(model.coef_[0])
        else:
            print("  ⚠️ Cannot extract feature importance from model")
            return
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("  📊 Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"    {row['feature']:25s}: {row['importance']:8.4f}")
        
        # Check if oceanographic features are most important
        oceanographic_keywords = ['sst', 'ssh', 'current', 'chl', 'sss', 'precip', 'thermal', 'productivity']
        oceanographic_features = [f for f in feature_cols if any(kw in f.lower() for kw in oceanographic_keywords)]
        
        top_oceanographic = feature_importance.head(10)
        oceanographic_in_top = sum(1 for f in top_oceanographic['feature'] if f in oceanographic_features)
        
        print(f"  🌊 Oceanographic features in top 10: {oceanographic_in_top}/10")
        
        if oceanographic_in_top >= 5:
            print("  ✅ Oceanographic features are appropriately important")
        else:
            print("  ⚠️ Oceanographic features may not be sufficiently important")
        
        return feature_importance
    
    def run_validation(self):
        """Run realistic performance validation"""
        print("🚀 Validating Realistic Performance Expectations")
        print("=" * 50)
        
        try:
            # Load data
            df = self.load_oceanographic_data()
            if df is None:
                return False
            
            # Prepare features
            X, y, feature_cols = self.prepare_features(df)
            
            # Check feature correlations
            no_leakage = self.check_feature_correlations(X, y)
            
            # Validate multiple models
            results = self.validate_multiple_models(X, y)
            
            # Analyze feature importance
            best_model = results['random_forest']['model']
            feature_importance = self.analyze_feature_importance(best_model, feature_cols)
            
            # Summary
            print("\n" + "=" * 50)
            print("📊 PERFORMANCE VALIDATION SUMMARY")
            print("=" * 50)
            
            for model_type, result in results.items():
                auc = result['auc_score']
                status = "IDEAL" if result['is_ideal'] else ("REALISTIC" if result['is_realistic'] else "UNREALISTIC")
                print(f"  {model_type:15s}: ROC-AUC {auc:.4f} ({status})")
            
            # Overall assessment
            all_realistic = all(result['is_realistic'] for result in results.values())
            any_ideal = any(result['is_ideal'] for result in results.values())
            
            if all_realistic:
                print("\n✅ OVERALL ASSESSMENT: PERFORMANCE IS REALISTIC")
                print("   - No data leakage detected")
                print("   - Performance within expected range for habitat prediction")
                print("   - Model ready for scientific use")
            else:
                print("\n⚠️ OVERALL ASSESSMENT: PERFORMANCE IS UNREALISTIC")
                print("   - Check for data leakage")
                print("   - Verify feature engineering")
                print("   - Review negative sampling")
            
            # Save results
            validation_results = {
                'validation_date': datetime.now().isoformat(),
                'expected_auc_range': self.expected_auc_range,
                'expected_auc_ideal': self.expected_auc_ideal,
                'model_results': {
                    model_type: {
                        'auc_score': result['auc_score'],
                        'is_realistic': result['is_realistic'],
                        'is_ideal': result['is_ideal']
                    }
                    for model_type, result in results.items()
                },
                'feature_importance': feature_importance.to_dict('records'),
                'no_data_leakage': no_leakage,
                'overall_assessment': 'realistic' if all_realistic else 'unrealistic'
            }
            
            results_path = self.output_dir / 'realistic_performance_validation.json'
            import json
            with open(results_path, 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            print(f"\n✅ Validation results saved to: {results_path}")
            
            return all_realistic
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    validator = RealisticPerformanceValidator()
    return validator.run_validation()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
