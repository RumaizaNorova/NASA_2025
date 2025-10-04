"""
Train multiple ML models to predict shark foraging hotspots with advanced evaluation.

This enhanced script supports multiple algorithms (XGBoost, Random Forest, LightGBM, Neural Networks),
implements sophisticated spatio-temporal cross-validation, generates comprehensive evaluation metrics,
creates SHAP explanations, and provides detailed performance analysis with calibration curves.

Features:
- Multiple ML algorithms with hyperparameter optimization
- Advanced cross-validation strategies (spatial, temporal, individual-based)
- Comprehensive evaluation metrics (ROC-AUC, PR-AUC, TSS, F1, calibration)
- SHAP feature importance and explanations
- Model comparison and selection
- Performance visualization and reporting
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import pickle
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, confusion_matrix, classification_report
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.combine import SMOTEENN
import matplotlib
matplotlib.use('Agg')  # Use non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import shap
from dotenv import load_dotenv

try:
    from .utils import load_config, ensure_dir, setup_logging
except ImportError:
    from utils import load_config, ensure_dir, setup_logging

# Load environment variables
load_dotenv()


class AdvancedMLTrainer:
    """Advanced ML trainer with multiple algorithms and comprehensive evaluation."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Configuration
        self.model_cfg = config.get('model', {})
        self.cv_cfg = config.get('cv', {})
        self.eval_cfg = config.get('evaluation', {})
        
        # Algorithms to train
        self.algorithms = self.model_cfg.get('algorithms', ['xgboost'])
        
        # Results storage
        self.results = {}
        self.models = {}
        self.feature_importance = {}
        self.shap_explanations = {}
        
        # Set random seeds
        np.random.seed(42)
    
    def _load_training_data(self) -> pd.DataFrame:
        """Load training data from CSV or Parquet."""
        possible_paths = [
            'data/interim/training_data.csv',
            'data/interim/training_data.parquet'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Loading training data from {path}")
                if path.endswith('.parquet'):
                    return pd.read_parquet(path)
                else:
                    return pd.read_csv(path)
        
        raise FileNotFoundError("No training data found. Run label_join.py first.")
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels for training with enhanced preprocessing."""
        # Identify feature columns (exclude metadata)
        exclude_cols = {'date', 'lat', 'lon', 'label', 'shark_id', 'species', 'timestamp', 'latitude', 'longitude'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values.astype(float)
        y = df['label'].values.astype(int)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature scaling for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Handle class imbalance with manual sampling (SMOTE compatibility issue)
        try:
            # Simple oversampling of minority class
            n_positive = np.sum(y == 1)
            n_negative = np.sum(y == 0)
            
            if n_positive < n_negative:
                # Oversample positive class
                positive_indices = np.where(y == 1)[0]
                oversample_factor = min(3, n_negative // n_positive)  # Don't over-oversample
                
                if oversample_factor > 1:
                    oversampled_indices = np.tile(positive_indices, oversample_factor)
                    X_oversampled = X_scaled[oversampled_indices]
                    y_oversampled = y[oversampled_indices]
                    
                    # Combine with original data
                    X_balanced = np.vstack([X_scaled, X_oversampled])
                    y_balanced = np.hstack([y, y_oversampled])
                    
                    self.logger.info(f"Original dataset: {X.shape[0]} samples")
                    self.logger.info(f"Balanced dataset: {X_balanced.shape[0]} samples")
                    self.logger.info(f"Original class distribution: {np.bincount(y)}")
                    self.logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
                    
                    X, y = X_balanced, y_balanced
                else:
                    X = X_scaled
            else:
                X = X_scaled
        except Exception as e:
            self.logger.warning(f"Balancing failed, using original data: {e}")
            X = X_scaled
        
        self.logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        self.logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols
    
    def _create_model(self, algorithm: str) -> Any:
        """Create a model instance for the specified algorithm."""
        if algorithm == 'xgboost':
            params = self.model_cfg.get('xgboost', {})
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
        
        elif algorithm == 'random_forest':
            params = self.model_cfg.get('random_forest', {})
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 300),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
        
        elif algorithm == 'lightgbm':
            params = self.model_cfg.get('lightgbm', {})
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        elif algorithm == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _prepare_cv_splits(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare cross-validation splits based on configuration."""
        scheme = self.cv_cfg.get('scheme', 'spatial')
        n_folds = self.cv_cfg.get('folds', 5)
        
        if scheme == 'spatial':
            return self._spatial_cv_splits(df, n_folds)
        elif scheme == 'by_individual':
            return self._individual_cv_splits(df, n_folds)
        elif scheme == 'temporal':
            return self._temporal_cv_splits(df, n_folds)
        else:
            # Random KFold as fallback
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            return list(kf.split(X))
    
    def _spatial_cv_splits(self, df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create spatial cross-validation splits using KMeans clustering."""
        # Use latitude/longitude columns (case-insensitive)
        lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
        lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
        coords = df[[lat_col, lon_col]].values
        
        # Adjust number of clusters if needed
        k = min(n_folds, len(df))
        if k <= 1:
            return [(np.arange(len(df)), np.arange(len(df)))]
        
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(coords)
        
        splits = []
        for c in range(k):
            test_idx = np.where(cluster_labels == c)[0]
            train_idx = np.where(cluster_labels != c)[0]
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _individual_cv_splits(self, df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create individual-based cross-validation splits."""
        if 'shark_id' not in df.columns:
            self.logger.warning("No shark_id column found, falling back to random splits")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            return list(kf.split(df))
        
        unique_ids = df['shark_id'].unique()
        k = min(n_folds, len(unique_ids))
        
        if k <= 1:
            return [(np.arange(len(df)), np.arange(len(df)))]
        
        # Assign each individual to a fold
        fold_assign = {}
        for i, uid in enumerate(unique_ids):
            fold_assign[uid] = i % k
        
        splits = []
        for fold in range(k):
            test_ids = [uid for uid, f in fold_assign.items() if f == fold]
            test_idx = df.index[df['shark_id'].isin(test_ids)].to_numpy()
            train_idx = df.index[~df['shark_id'].isin(test_ids)].to_numpy()
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _temporal_cv_splits(self, df: pd.DataFrame, n_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create temporal cross-validation splits."""
        if 'date' not in df.columns:
            self.logger.warning("No date column found, falling back to random splits")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            return list(kf.split(df))
        
        # Sort by date
        df_sorted = df.sort_values('date')
        n_samples = len(df_sorted)
        
        splits = []
        for fold in range(n_folds):
            # Create temporal splits
            test_start = int(fold * n_samples / n_folds)
            test_end = int((fold + 1) * n_samples / n_folds)
            
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _compute_comprehensive_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        
        # TSS (True Skill Statistic)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 1:
            # Only one class predicted
            if y_true[0] == 1:
                tn, fp, fn, tp = 0, 0, len(y_true), 0
            else:
                tn, fp, fn, tp = len(y_true), 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['tss'] = tpr + tnr - 1.0
        
        # Additional metrics
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['specificity'] = tnr
        metrics['sensitivity'] = tpr
        
        return metrics
    
    def _generate_shap_explanations(self, model: Any, X: np.ndarray, feature_names: List[str], 
                                  algorithm: str) -> Dict[str, Any]:
        """Generate SHAP explanations for the model."""
        try:
            if algorithm in ['xgboost', 'lightgbm']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # For binary classification, get positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                return {
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'feature_names': feature_names
                }
            
            elif algorithm == 'random_forest':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                
                return {
                    'shap_values': shap_values,
                    'explainer': explainer,
                    'feature_names': feature_names
                }
            
            else:
                # For neural networks and other models
                explainer = shap.Explainer(model, X[:100])  # Use subset for efficiency
                shap_values = explainer(X[:100])
                
                return {
                    'shap_values': shap_values.values,
                    'explainer': explainer,
                    'feature_names': feature_names
                }
        
        except Exception as e:
            self.logger.warning(f"Failed to generate SHAP explanations for {algorithm}: {e}")
            return {}
    
    def _plot_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                               algorithm: str, output_dir: str):
        """Plot calibration curve for the model."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            plt.figure(figsize=(8, 6))
            plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=algorithm)
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Curve - {algorithm}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, f'calibration_{algorithm}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Calibration curve saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create calibration curve for {algorithm}: {e}")
    
    def _plot_feature_importance(self, feature_names: List[str], importance_values: np.ndarray,
                                algorithm: str, output_dir: str):
        """Plot feature importance."""
        try:
            # Sort features by importance
            indices = np.argsort(importance_values)[::-1]
            top_features = min(20, len(feature_names))
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(top_features), importance_values[indices[:top_features]])
            plt.yticks(range(top_features), [feature_names[i] for i in indices[:top_features]])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {algorithm}')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            output_path = os.path.join(output_dir, f'feature_importance_{algorithm}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Feature importance plot saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create feature importance plot for {algorithm}: {e}")
    
    def _plot_shap_summary(self, shap_values: np.ndarray, X: np.ndarray, feature_names: List[str],
                          algorithm: str, output_dir: str):
        """Plot SHAP summary."""
        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary - {algorithm}')
            
            output_path = os.path.join(output_dir, f'shap_summary_{algorithm}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP summary plot saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create SHAP summary plot for {algorithm}: {e}")
    
    def train_models(self) -> Dict[str, Any]:
        """Train all configured models and return results."""
        # Load training data
        df = self._load_training_data()
        X, y, feature_names = self._prepare_features(df)
        
        # Prepare cross-validation splits
        cv_splits = self._prepare_cv_splits(df, X, y)
        
        # Create output directory
        output_dir = 'data/interim'
        ensure_dir(output_dir)
        
        # Train each algorithm
        for algorithm in self.algorithms:
            self.logger.info(f"Training {algorithm} model...")
            
            # Create model
            model = self._create_model(algorithm)
            
            # Cross-validation
            fold_metrics = []
            all_true = []
            all_prob = []
            all_pred = []
            
            for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Handle class imbalance
                if algorithm in ['xgboost', 'lightgbm']:
                    n_pos = np.sum(y_train == 1)
                    n_neg = np.sum(y_train == 0)
                    if n_pos > 0:
                        model.set_params(scale_pos_weight=n_neg/n_pos)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
                
                # Compute metrics
                metrics = self._compute_comprehensive_metrics(y_test, y_prob, y_pred)
                fold_metrics.append(metrics)
                
                # Store for overall evaluation
                all_true.extend(y_test.tolist())
                all_prob.extend(y_prob.tolist())
                all_pred.extend(y_pred.tolist())
                
                roc_auc = metrics.get('roc_auc', 0)
                pr_auc = metrics.get('pr_auc', 0)
                tss = metrics.get('tss', 0)
                self.logger.info(f"{algorithm} Fold {fold_num}: ROC-AUC={roc_auc:.3f}, "
                               f"PR-AUC={pr_auc:.3f}, TSS={tss:.3f}")
            
            # Train final model on all data
            final_model = self._create_model(algorithm)
            if algorithm in ['xgboost', 'lightgbm']:
                n_pos = np.sum(y == 1)
                n_neg = np.sum(y == 0)
                if n_pos > 0:
                    final_model.set_params(scale_pos_weight=n_neg/n_pos)
            
            final_model.fit(X, y)
            
            # Store model and results
            self.models[algorithm] = final_model
            
            # Aggregate metrics
            aggregated_metrics = {}
            for metric in fold_metrics[0].keys():
                values = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
                if values:
                    aggregated_metrics[f'mean_{metric}'] = float(np.mean(values))
                    aggregated_metrics[f'std_{metric}'] = float(np.std(values))
            
            # Overall metrics
            all_true_arr = np.array(all_true)
            all_prob_arr = np.array(all_prob)
            all_pred_arr = np.array(all_pred)
            
            if len(np.unique(all_true_arr)) > 1:
                overall_metrics = self._compute_comprehensive_metrics(all_true_arr, all_prob_arr, all_pred_arr)
                aggregated_metrics.update(overall_metrics)
            
            self.results[algorithm] = {
                'fold_metrics': fold_metrics,
                'aggregated_metrics': aggregated_metrics
            }
            
            # Feature importance
            if hasattr(final_model, 'feature_importances_'):
                self.feature_importance[algorithm] = final_model.feature_importances_
                self._plot_feature_importance(feature_names, final_model.feature_importances_, 
                                            algorithm, output_dir)
            
            # SHAP explanations
            if self.eval_cfg.get('shap_explanations', False):
                shap_results = self._generate_shap_explanations(final_model, X, feature_names, algorithm)
                if shap_results:
                    self.shap_explanations[algorithm] = shap_results
                    self._plot_shap_summary(shap_results['shap_values'], X, feature_names, 
                                          algorithm, output_dir)
            
            # Calibration curve
            if self.eval_cfg.get('calibration_curves', False):
                self._plot_calibration_curve(all_true_arr, all_prob_arr, algorithm, output_dir)
            
            # Save model
            model_path = os.path.join(output_dir, f'{algorithm}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            
            self.logger.info(f"{algorithm} model saved to {model_path}")
        
        return self.results
    
    def save_results(self):
        """Save training results and generate reports."""
        output_dir = 'data/interim'
        ensure_dir(output_dir)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(output_dir, 'feature_importance.json')
            # Convert numpy arrays to lists for JSON serialization
            importance_dict = {k: v.tolist() for k, v in self.feature_importance.items()}
            with open(importance_path, 'w') as f:
                json.dump(importance_dict, f, indent=2)
        
        # Generate comparison plot
        self._plot_model_comparison(output_dir)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _plot_model_comparison(self, output_dir: str):
        """Plot model comparison."""
        try:
            algorithms = list(self.results.keys())
            metrics = ['roc_auc', 'pr_auc', 'tss', 'f1']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                values = []
                labels = []
                
                for algo in algorithms:
                    if f'mean_{metric}' in self.results[algo]['aggregated_metrics']:
                        values.append(self.results[algo]['aggregated_metrics'][f'mean_{metric}'])
                        labels.append(algo)
                
                if values:
                    axes[i].bar(labels, values)
                    axes[i].set_title(f'{metric.upper()} Comparison')
                    axes[i].set_ylabel(metric.upper())
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'model_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model comparison plot saved to {comparison_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create model comparison plot: {e}")
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a summary report of training results."""
        report_path = os.path.join(output_dir, 'training_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== SHARK HABITAT MODEL TRAINING SUMMARY ===\n\n")
            
            f.write(f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Algorithms Trained: {', '.join(self.algorithms)}\n\n")
            
            for algorithm in self.algorithms:
                f.write(f"--- {algorithm.upper()} ---\n")
                metrics = self.results[algorithm]['aggregated_metrics']
                
                for metric in ['roc_auc', 'pr_auc', 'tss', 'f1', 'precision', 'recall']:
                    if f'mean_{metric}' in metrics:
                        mean_val = metrics[f'mean_{metric}']
                        std_val = metrics.get(f'std_{metric}', 0)
                        f.write(f"{metric.upper()}: {mean_val:.3f} ± {std_val:.3f}\n")
                
                f.write("\n")
            
            # Best model
            best_algorithm = None
            best_roc_auc = 0
            
            for algorithm in self.algorithms:
                metrics = self.results[algorithm]['aggregated_metrics']
                if 'roc_auc' in metrics and metrics['roc_auc'] > best_roc_auc:
                    best_roc_auc = metrics['roc_auc']
                    best_algorithm = algorithm
            
            if best_algorithm:
                f.write(f"Best Model: {best_algorithm} (ROC-AUC: {best_roc_auc:.3f})\n")
        
        self.logger.info(f"Summary report saved to {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for shark habitat prediction.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to YAML configuration file")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (expects synthetic training data)")
    parser.add_argument("--algorithms", nargs='+', help="Specific algorithms to train (overrides config)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Override algorithms if specified
    if args.algorithms:
        config['model']['algorithms'] = args.algorithms
    
    # Initialize trainer
    trainer = AdvancedMLTrainer(config, args)
    
    # Train models
    results = trainer.train_models()
    
    # Save results
    trainer.save_results()
    
    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    for algorithm, result in results.items():
        metrics = result['aggregated_metrics']
        print(f"{algorithm.upper()}:")
        print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.3f}")
        print(f"  PR-AUC: {metrics.get('pr_auc', 'N/A'):.3f}")
        print(f"  TSS: {metrics.get('tss', 'N/A'):.3f}")
        print(f"  F1: {metrics.get('f1', 'N/A'):.3f}")
        print()
    
    # Find best model
    best_algorithm = None
    best_roc_auc = 0
    
    for algorithm, result in results.items():
        metrics = result['aggregated_metrics']
        if 'roc_auc' in metrics and metrics['roc_auc'] > best_roc_auc:
            best_roc_auc = metrics['roc_auc']
            best_algorithm = algorithm
    
    if best_algorithm:
        print(f"Best Model: {best_algorithm} (ROC-AUC: {best_roc_auc:.3f})")
    
    print(f"\nResults saved to data/interim/")


if __name__ == '__main__':
    main()