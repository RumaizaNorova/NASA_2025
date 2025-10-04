#!/usr/bin/env python3
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
