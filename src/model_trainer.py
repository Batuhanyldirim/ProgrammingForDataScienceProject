"""Model Trainer module for training classification models.

This module provides functionality to train machine learning models for
molecular activity prediction, supporting multiple classification algorithms.
"""

import logging
from typing import Any, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Configure module logger
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for classification models.
    
    This class supports training multiple classification algorithms including
    Random Forest and Gradient Boosting for molecular activity prediction.
    """

    SUPPORTED_ALGORITHMS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
    }

    def __init__(self, algorithm: str = 'random_forest', **kwargs):
        """Initialize the model trainer.
        
        Args:
            algorithm: The classification algorithm to use. 
                Supported values: 'random_forest', 'gradient_boosting'.
            **kwargs: Additional keyword arguments passed to the classifier.
        """
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Supported algorithms: {list(self.SUPPORTED_ALGORITHMS.keys())}"
            )
        
        self.algorithm = algorithm
        self._model_class = self.SUPPORTED_ALGORITHMS[algorithm]
        self._model_kwargs = kwargs
        self._model = None
        
        logger.info(f"Initialized ModelTrainer with algorithm='{algorithm}'")

    def train(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model and return fitted estimator.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label vector of shape (n_samples,).
            
        Returns:
            The fitted estimator.
            
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
            f"Training {self.algorithm} model with {X.shape[0]} samples "
            f"and {X.shape[1]} features"
        )
        
        # Create and fit the model
        self._model = self._model_class(**self._model_kwargs)
        self._model.fit(X, y)
        
        logger.info(f"Model training complete")
        return self._model

    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        """Return list of supported algorithm names.
        
        Returns:
            List of algorithm name strings that can be passed to __init__.
        """
        return list(cls.SUPPORTED_ALGORITHMS.keys())
