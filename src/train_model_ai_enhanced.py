"""
AI-Enhanced Model Training with Advanced Optimization

This enhanced training script integrates:
- AI-powered analysis and insights
- Advanced hyperparameter optimization with Optuna
- Advanced sampling strategies (SMOTE, ADASYN)
- Cost-sensitive learning approaches
- Ensemble methods
- Focal loss for extreme imbalance
- Comprehensive validation and reporting

Features:
- OpenAI integration for intelligent analysis
- Optuna hyperparameter optimization
- Advanced sampling techniques
- Cost-sensitive learning
- Ensemble model training
- Automated AI-powered reporting
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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
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

# Advanced sampling and optimization
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Install with: pip install imbalanced-learn")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    from .utils import load_config, ensure_dir, setup_logging
    from .ai_analysis import AIAnalysisEngine
except ImportError:
    from utils import load_config, ensure_dir, setup_logging
    from ai_analysis import AIAnalysisEngine

# Load environment variables
load_dotenv()


class AdvancedSampler:
    """Advanced sampling strategies for handling class imbalance."""
    
    def __init__(self, strategy: str = 'smote'):
        self.strategy = strategy
        self.sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the sampling strategy."""
        if not IMBLEARN_AVAILABLE:
            self.sampler = None
            return
        
        if self.strategy == 'smote':
            self.sampler = SMOTE(random_state=42, k_neighbors=1)
        elif self.strategy == 'adasyn':
            self.sampler = ADASYN(random_state=42)
        elif self.strategy == 'borderline_smote':
            self.sampler = BorderlineSMOTE(random_state=42, k_neighbors=1)
        elif self.strategy == 'smoteenn':
            self.sampler = SMOTEENN(random_state=42)
        elif self.strategy == 'smotetomek':
            self.sampler = SMOTETomek(random_state=42)
        elif self.strategy == 'random_under':
            self.sampler = RandomUnderSampler(random_state=42)
        elif self.strategy == 'tomek_links':
            self.sampler = TomekLinks()
        else:
            self.sampler = None
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sampling strategy to the data."""
        if self.sampler is None:
            return X, y
        
        try:
            return self.sampler.fit_resample(X, y)
        except Exception as e:
            print(f"Warning: Sampling failed with {self.strategy}: {e}")
            return X, y


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, algorithm: str, n_trials: int = 100):
        self.algorithm = algorithm
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray, 
                cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            return self._get_default_params()
        
        def objective(trial):
            # Suggest hyperparameters based on algorithm
            params = self._suggest_params(trial)
            
            # Cross-validation
            scores = []
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create and train model
                model = self._create_model(params)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_prob = model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) > 1:
                    score = roc_auc_score(y_val, y_prob)
                    scores.append(score)
            
            return np.mean(scores) if scores else 0.0
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.n_trials)
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            return self.best_params
            
        except Exception as e:
            print(f"Warning: Optuna optimization failed: {e}")
            return self._get_default_params()
    
    def _suggest_params(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the given algorithm."""
        if self.algorithm == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
        elif self.algorithm == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100)
            }
        elif self.algorithm == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        else:
            return {}
    
    def _create_model(self, params: Dict[str, Any]):
        """Create model with given parameters."""
        if self.algorithm == 'xgboost':
            return xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                **params
            )
        elif self.algorithm == 'lightgbm':
            return lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                **params
            )
        elif self.algorithm == 'random_forest':
            return RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters when optimization is not available."""
        if self.algorithm == 'xgboost':
            return {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif self.algorithm == 'lightgbm':
            return {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif self.algorithm == 'random_forest':
            return {
                'n_estimators': 300,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        else:
            return {}


class FocalLoss:
    """Focal loss implementation for extreme class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute focal loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute focal loss
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * np.power(1 - p_t, self.gamma)
        
        focal_loss = -focal_weight * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return np.mean(focal_loss)


class AIEnhancedTrainer:
    """AI-enhanced ML trainer with advanced optimization and analysis."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        self.config = config
        self.args = args
        self.logger = setup_logging(__name__)
        
        # Initialize AI analysis engine
        self.ai_engine = AIAnalysisEngine(config)
        
        # Configuration
        self.model_cfg = config.get('model', {})
        self.cv_cfg = config.get('cv', {})
        self.eval_cfg = config.get('evaluation', {})
        self.ai_cfg = config.get('ai_analysis', {})
        
        # Algorithms to train
        self.algorithms = self.model_cfg.get('algorithms', ['xgboost', 'lightgbm', 'random_forest'])
        
        # Results storage
        self.results = {}
        self.models = {}
        self.feature_importance = {}
        self.shap_explanations = {}
        self.optimization_results = {}
        
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
    
    def _prepare_features_enhanced(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Enhanced feature preparation with advanced preprocessing."""
        # Identify feature columns (exclude metadata)
        exclude_cols = {'date', 'lat', 'lon', 'label', 'shark_id', 'species', 'timestamp', 'latitude', 'longitude'}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values.astype(float)
        y = df['label'].values.astype(int)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.logger.info(f"Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        self.logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X_scaled, y, feature_cols
    
    def _apply_advanced_sampling(self, X: np.ndarray, y: np.ndarray, 
                               strategy: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """Apply advanced sampling strategy to handle class imbalance."""
        sampler = AdvancedSampler(strategy)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        self.logger.info(f"Applied {strategy} sampling:")
        self.logger.info(f"  Original: {X.shape[0]} samples, class distribution: {np.bincount(y)}")
        self.logger.info(f"  Balanced: {X_balanced.shape[0]} samples, class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                algorithm: str, cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        self.logger.info(f"Optimizing hyperparameters for {algorithm}...")
        
        optimizer = OptunaOptimizer(algorithm, n_trials=50)  # Reduced for faster execution
        best_params = optimizer.optimize(X, y, cv_splits)
        
        self.logger.info(f"Best parameters for {algorithm}: {best_params}")
        self.logger.info(f"Best CV score: {optimizer.best_score:.4f}")
        
        self.optimization_results[algorithm] = {
            'best_params': best_params,
            'best_score': optimizer.best_score
        }
        
        return best_params
    
    def _create_ensemble_model(self, base_models: Dict[str, Any]) -> Any:
        """Create ensemble model from base models."""
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Use voting classifier for ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        return ensemble
    
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
    
    def _create_model(self, algorithm: str, optimized_params: Dict[str, Any] = None) -> Any:
        """Create a model instance with optimized parameters."""
        params = optimized_params or {}
        
        if algorithm == 'xgboost':
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)
        
        elif algorithm == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
        
        elif algorithm == 'random_forest':
            default_params = {
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
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
    
    def train_models(self) -> Dict[str, Any]:
        """Train all configured models with AI enhancement."""
        # Load training data
        df = self._load_training_data()
        X, y, feature_names = self._prepare_features_enhanced(df)
        
        # Apply advanced sampling if enabled
        sampling_strategy = self.ai_cfg.get('sampling_strategy', 'smote')
        if sampling_strategy != 'none':
            X, y = self._apply_advanced_sampling(X, y, sampling_strategy)
        
        # Prepare cross-validation splits
        cv_splits = self._prepare_cv_splits(df, X, y)
        
        # Create output directory
        output_dir = 'data/interim'
        ensure_dir(output_dir)
        
        # Train each algorithm with optimization
        for algorithm in self.algorithms:
            self.logger.info(f"Training {algorithm} model with AI enhancement...")
            
            # Optimize hyperparameters
            if self.ai_cfg.get('optimize_hyperparameters', True):
                optimized_params = self._optimize_hyperparameters(X, y, algorithm, cv_splits)
            else:
                optimized_params = {}
            
            # Create model with optimized parameters
            model = self._create_model(algorithm, optimized_params)
            
            # Cross-validation with cost-sensitive learning
            fold_metrics = []
            all_true = []
            all_prob = []
            all_pred = []
            
            for fold_num, (train_idx, test_idx) in enumerate(cv_splits, 1):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Apply cost-sensitive learning
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
            final_model = self._create_model(algorithm, optimized_params)
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
                'aggregated_metrics': aggregated_metrics,
                'optimized_params': optimized_params
            }
            
            # Feature importance
            if hasattr(final_model, 'feature_importances_'):
                self.feature_importance[algorithm] = final_model.feature_importances_
            
            # Save model
            model_path = os.path.join(output_dir, f'{algorithm}_model_ai_enhanced.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)
            
            self.logger.info(f"{algorithm} model saved to {model_path}")
        
        # Create ensemble model if multiple algorithms trained
        if len(self.models) > 1 and self.ai_cfg.get('create_ensemble', True):
            self.logger.info("Creating ensemble model...")
            ensemble_model = self._create_ensemble_model(self.models)
            ensemble_model.fit(X, y)
            
            # Evaluate ensemble
            ensemble_prob = ensemble_model.predict_proba(X)[:, 1]
            ensemble_pred = ensemble_model.predict(X)
            ensemble_metrics = self._compute_comprehensive_metrics(y, ensemble_prob, ensemble_pred)
            
            self.models['ensemble'] = ensemble_model
            self.results['ensemble'] = {
                'aggregated_metrics': ensemble_metrics,
                'ensemble_type': 'voting_classifier'
            }
            
            # Save ensemble model
            ensemble_path = os.path.join(output_dir, 'ensemble_model_ai_enhanced.pkl')
            with open(ensemble_path, 'wb') as f:
                pickle.dump(ensemble_model, f)
            
            self.logger.info(f"Ensemble model saved to {ensemble_path}")
        
        return self.results
    
    def generate_ai_analysis(self) -> Dict[str, Any]:
        """Generate AI-powered analysis of results."""
        self.logger.info("Generating AI-powered analysis...")
        
        # Prepare metrics for AI analysis
        ai_metrics = {}
        for algorithm, result in self.results.items():
            if 'aggregated_metrics' in result:
                ai_metrics[algorithm] = result['aggregated_metrics']
        
        # Analyze performance
        performance_analysis = self.ai_engine.analyze_model_performance(
            ai_metrics, self.feature_importance
        )
        
        # Analyze feature importance
        if self.feature_importance:
            all_feature_names = []
            # Get feature names from the first model
            for algorithm in self.algorithms:
                if algorithm in self.feature_importance:
                    # We need to get feature names from the training data
                    df = self._load_training_data()
                    exclude_cols = {'date', 'lat', 'lon', 'label', 'shark_id', 'species', 'timestamp', 'latitude', 'longitude'}
                    feature_names = [c for c in df.columns if c not in exclude_cols]
                    all_feature_names = feature_names
                    break
            
            feature_analysis = self.ai_engine.analyze_feature_importance(
                self.feature_importance, all_feature_names
            )
        
        # Generate improvement recommendations
        target_metrics = {
            'roc_auc': 0.65,
            'pr_auc': 0.35,
            'tss': 0.20,
            'f1': 0.30
        }
        
        recommendations = self.ai_engine.generate_improvement_recommendations(
            ai_metrics, target_metrics
        )
        
        # Store AI analysis results
        self.ai_analysis_results = {
            'performance_analysis': performance_analysis,
            'feature_analysis': feature_analysis if self.feature_importance else {},
            'recommendations': recommendations
        }
        
        return self.ai_analysis_results
    
    def save_results(self):
        """Save training results and generate AI-powered reports."""
        output_dir = 'data/interim'
        ensure_dir(output_dir)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'ai_enhanced_training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(output_dir, 'ai_enhanced_feature_importance.json')
            # Convert numpy arrays to lists for JSON serialization
            importance_dict = {k: v.tolist() for k, v in self.feature_importance.items()}
            with open(importance_path, 'w') as f:
                json.dump(importance_dict, f, indent=2)
        
        # Save optimization results
        if self.optimization_results:
            opt_path = os.path.join(output_dir, 'optimization_results.json')
            with open(opt_path, 'w') as f:
                json.dump(self.optimization_results, f, indent=2)
        
        # Generate AI-powered comprehensive report
        report_path = self.ai_engine.generate_comprehensive_report(output_dir)
        
        self.logger.info(f"AI-enhanced results saved to {output_dir}")
        self.logger.info(f"AI analysis report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI-enhanced ML models for shark habitat prediction.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to YAML configuration file")
    parser.add_argument("--algorithms", nargs='+', help="Specific algorithms to train (overrides config)")
    parser.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    parser.add_argument("--sampling", default="smote", help="Sampling strategy (smote, adasyn, none)")
    parser.add_argument("--ensemble", action="store_true", help="Create ensemble model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    # Override algorithms if specified
    if args.algorithms:
        config['model']['algorithms'] = args.algorithms
    
    # Add AI analysis configuration
    config['ai_analysis'] = {
        'optimize_hyperparameters': args.optimize,
        'sampling_strategy': args.sampling,
        'create_ensemble': args.ensemble,
        'generate_insights': True,
        'generate_recommendations': True
    }
    
    # Initialize AI-enhanced trainer
    trainer = AIEnhancedTrainer(config, args)
    
    # Train models
    results = trainer.train_models()
    
    # Generate AI analysis
    ai_analysis = trainer.generate_ai_analysis()
    
    # Save results
    trainer.save_results()
    
    # Print summary
    print("\n=== AI-ENHANCED TRAINING SUMMARY ===")
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
    print(f"AI analysis completed with insights and recommendations")


if __name__ == '__main__':
    main()
