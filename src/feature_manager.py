"""Feature Manager module for creating and preprocessing molecular features.

This module provides functionality to create different feature representations
(descriptors, fingerprints, or combined) and preprocess them for ML models.
"""

import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .descriptor_calculator import DescriptorCalculator
from .fingerprint_generator import FingerprintGenerator

# Configure module logger
logger = logging.getLogger(__name__)


class FeatureManager:
    """Manager for molecular feature creation and preprocessing.
    
    This class coordinates descriptor and fingerprint generation, and provides
    preprocessing capabilities including NaN imputation and feature scaling.
    """

    def __init__(
        self,
        descriptor_calc: Optional[DescriptorCalculator] = None,
        fingerprint_gen: Optional[FingerprintGenerator] = None
    ):
        """Initialize the feature manager.
        
        Args:
            descriptor_calc: A DescriptorCalculator instance. If None, a default
                instance is created.
            fingerprint_gen: A FingerprintGenerator instance. If None, a default
                instance is created.
        """
        self.descriptor_calc = descriptor_calc or DescriptorCalculator()
        self.fingerprint_gen = fingerprint_gen or FingerprintGenerator()
        
        # Preprocessing components (fitted during preprocess with fit=True)
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        
        logger.info("Initialized FeatureManager")

    def create_descriptor_features(self, mols: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Create descriptor-only feature matrix.
        
        Args:
            mols: A list of RDKit Mol objects (or None for invalid molecules).
            
        Returns:
            A 2D numpy array of shape (n_molecules, n_descriptors).
        """
        logger.info(f"Creating descriptor features for {len(mols)} molecules")
        return self.descriptor_calc.calculate_batch(mols)

    def create_fingerprint_features(self, mols: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Create fingerprint-only feature matrix.
        
        Args:
            mols: A list of RDKit Mol objects (or None for invalid molecules).
            
        Returns:
            A 2D numpy array of shape (n_molecules, n_bits).
        """
        logger.info(f"Creating fingerprint features for {len(mols)} molecules")
        return self.fingerprint_gen.generate_batch(mols)

    def create_combined_features(self, mols: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Create combined descriptor + fingerprint feature matrix.
        
        Args:
            mols: A list of RDKit Mol objects (or None for invalid molecules).
            
        Returns:
            A 2D numpy array of shape (n_molecules, n_descriptors + n_bits).
            Descriptors are placed first, followed by fingerprint bits.
        """
        logger.info(f"Creating combined features for {len(mols)} molecules")
        
        descriptors = self.create_descriptor_features(mols)
        fingerprints = self.create_fingerprint_features(mols)
        
        # Convert fingerprints to float64 for concatenation
        fingerprints_float = fingerprints.astype(np.float64)
        
        return np.hstack([descriptors, fingerprints_float])

    def preprocess(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Handle missing values and scale features.
        
        This method performs two preprocessing steps:
        1. Impute NaN values using median imputation
        2. Scale features to zero mean and unit variance
        
        Args:
            features: A 2D numpy array of features.
            fit: If True, fit the imputer and scaler on this data.
                If False, use previously fitted transformers.
                
        Returns:
            A 2D numpy array of preprocessed features with no NaN values.
            
        Raises:
            ValueError: If fit=False but transformers have not been fitted.
        """
        if features.size == 0:
            return features
            
        if fit:
            logger.info("Fitting and transforming features")
            self._imputer = SimpleImputer(strategy='median')
            self._scaler = StandardScaler()
            
            # Fit and transform
            imputed = self._imputer.fit_transform(features)
            scaled = self._scaler.fit_transform(imputed)
        else:
            if self._imputer is None or self._scaler is None:
                raise ValueError(
                    "Transformers not fitted. Call preprocess with fit=True first."
                )
            logger.info("Transforming features with fitted transformers")
            imputed = self._imputer.transform(features)
            scaled = self._scaler.transform(imputed)
        
        return scaled
