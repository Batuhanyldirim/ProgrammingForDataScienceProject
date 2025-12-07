"""Model Predictor module for generating predictions.

This module provides functionality to generate probability predictions
from trained models and save predictions to CSV files.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Predictor for generating probability predictions from trained models.
    
    This class handles prediction generation and output file creation
    for molecular activity prediction.
    """

    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions for the positive class.
        
        Args:
            model: A trained classifier with a predict_proba method.
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Array of probability scores in [0.0, 1.0] for each sample,
            representing the probability of the positive class (activity).
            
        Raises:
            ValueError: If X is not 2-dimensional.
            AttributeError: If model does not have predict_proba method.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if not hasattr(model, 'predict_proba'):
            raise AttributeError(
                "Model must have a predict_proba method for probability prediction"
            )
        
        logger.info(f"Generating predictions for {X.shape[0]} samples")
        
        # Get probability predictions - returns shape (n_samples, n_classes)
        # We want the probability of the positive class (index 1)
        probabilities = model.predict_proba(X)[:, 1]
        
        logger.info(f"Generated {len(probabilities)} probability predictions")
        return probabilities

    def save_predictions(
        self, 
        indices: np.ndarray, 
        probabilities: np.ndarray, 
        filepath: str
    ) -> None:
        """Save predictions to a CSV file.
        
        Creates a CSV file with INDEX and PROBABILITY columns.
        
        Args:
            indices: Array of compound indices.
            probabilities: Array of probability scores.
            filepath: Path to the output CSV file.
            
        Raises:
            ValueError: If indices and probabilities have different lengths.
        """
        if len(indices) != len(probabilities):
            raise ValueError(
                f"indices and probabilities must have the same length: "
                f"got {len(indices)} indices and {len(probabilities)} probabilities"
            )
        
        logger.info(f"Saving {len(indices)} predictions to {filepath}")
        
        # Create DataFrame with predictions
        df = pd.DataFrame({
            'INDEX': indices,
            'PROBABILITY': probabilities
        })
        
        # Save to CSV without row index
        df.to_csv(filepath, index=False)
        
        logger.info(f"Predictions saved successfully to {filepath}")
