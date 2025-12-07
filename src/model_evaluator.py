"""Model Evaluator module for evaluating classification models.

This module provides functionality to evaluate machine learning models using
cross-validation and compare multiple models for molecular activity prediction.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# Configure module logger
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluator for classification models using cross-validation.
    
    This class supports evaluating models using k-fold cross-validation
    and comparing multiple models based on AUC scores.
    """

    def __init__(self, n_folds: int = 5):
        """Initialize the model evaluator.
        
        Args:
            n_folds: Number of folds for cross-validation. Default is 5.
        """
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        
        self.n_folds = n_folds
        logger.info(f"Initialized ModelEvaluator with n_folds={n_folds}")

    def cross_validate(
        self, 
        model: Any, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict:
        """Perform k-fold cross-validation and return AUC metrics.
        
        Uses stratified k-fold cross-validation to maintain class balance
        across folds. Calculates AUC for each fold and returns mean and
        standard deviation.
        
        Args:
            model: A classifier with fit and predict_proba methods.
            X: Feature matrix of shape (n_samples, n_features).
            y: Label vector of shape (n_samples,).
            
        Returns:
            Dictionary containing:
                - 'auc_scores': List of AUC scores for each fold
                - 'auc_mean': Mean AUC across all folds
                - 'auc_std': Standard deviation of AUC across folds
                - 'n_folds': Number of folds used
                
        Raises:
            ValueError: If X and y have incompatible shapes.
        """
        # Validate input shapes
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have incompatible shapes: X has {X.shape[0]} samples, "
                f"y has {y.shape[0]} samples"
            )
        
        logger.info(
            f"Starting {self.n_folds}-fold cross-validation with "
            f"{X.shape[0]} samples and {X.shape[1]} features"
        )
        
        # Use stratified k-fold to maintain class balance
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        auc_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Split data for this fold
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone the model to avoid fitting the same instance multiple times
            from sklearn.base import clone
            fold_model = clone(model)
            
            # Train on training fold
            fold_model.fit(X_train, y_train)
            
            # Get probability predictions for validation fold
            y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
            
            # Calculate AUC for this fold
            fold_auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(fold_auc)
            
            logger.debug(f"Fold {fold_idx + 1}/{self.n_folds}: AUC = {fold_auc:.4f}")
        
        auc_mean = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        
        logger.info(f"Cross-validation complete: AUC = {auc_mean:.4f} ± {auc_std:.4f}")
        
        return {
            'auc_scores': auc_scores,
            'auc_mean': auc_mean,
            'auc_std': auc_std,
            'n_folds': self.n_folds
        }

    def compare_models(
        self, 
        models: List[Any], 
        X: np.ndarray, 
        y: np.ndarray
    ) -> pd.DataFrame:
        """Compare multiple models using cross-validation and rank by AUC.
        
        Evaluates each model using cross-validation and returns a DataFrame
        with models ranked from highest to lowest AUC.
        
        Args:
            models: List of classifiers to compare.
            X: Feature matrix of shape (n_samples, n_features).
            y: Label vector of shape (n_samples,).
            
        Returns:
            DataFrame with columns:
                - 'model_index': Index of the model in the input list
                - 'model_type': String representation of the model class
                - 'auc_mean': Mean AUC from cross-validation
                - 'auc_std': Standard deviation of AUC
            Sorted by auc_mean in descending order (best model first).
            
        Raises:
            ValueError: If models list is empty or X/y have incompatible shapes.
        """
        if not models:
            raise ValueError("models list cannot be empty")
        
        logger.info(f"Comparing {len(models)} models using {self.n_folds}-fold CV")
        
        results = []
        
        for idx, model in enumerate(models):
            model_type = type(model).__name__
            logger.info(f"Evaluating model {idx + 1}/{len(models)}: {model_type}")
            
            cv_results = self.cross_validate(model, X, y)
            
            results.append({
                'model_index': idx,
                'model_type': model_type,
                'auc_mean': cv_results['auc_mean'],
                'auc_std': cv_results['auc_std']
            })
        
        # Create DataFrame and sort by AUC (descending)
        df = pd.DataFrame(results)
        df = df.sort_values('auc_mean', ascending=False).reset_index(drop=True)
        
        logger.info(f"Model comparison complete. Best model: {df.iloc[0]['model_type']} "
                   f"with AUC = {df.iloc[0]['auc_mean']:.4f}")
        
        return df
