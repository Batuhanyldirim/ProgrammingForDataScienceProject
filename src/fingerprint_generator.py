"""Fingerprint Generator module for computing Morgan fingerprints.

This module provides functionality to generate Morgan fingerprints from RDKit
Mol objects for use as feature vectors in machine learning models.
"""

import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Configure module logger
logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """Generator for Morgan fingerprints using RDKit.
    
    This class generates Morgan (circular) fingerprints with configurable
    radius and bit length for molecular feature representation.
    """

    def __init__(self, radius: int = 2, n_bits: int = 1024):
        """Initialize the fingerprint generator.
        
        Args:
            radius: The radius of the Morgan fingerprint (default 2).
            n_bits: The length of the fingerprint bit vector (default 1024).
        """
        self.radius = radius
        self.n_bits = n_bits
        self._generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        logger.info(f"Initialized FingerprintGenerator with radius={radius}, n_bits={n_bits}")

    def generate(self, mol: Optional[Chem.Mol]) -> np.ndarray:
        """Generate Morgan fingerprint for a single molecule.
        
        Args:
            mol: An RDKit Mol object, or None for invalid molecules.
            
        Returns:
            A numpy array of the fingerprint. If mol is None or fingerprint
            generation fails, returns a zero vector of length n_bits.
        """
        # Handle None molecules
        if mol is None:
            logger.warning("Received None molecule, returning zero vector")
            return np.zeros(self.n_bits, dtype=np.int8)
        
        try:
            fp = self._generator.GetFingerprintAsNumPy(mol)
            return fp.astype(np.int8)
        except Exception as e:
            logger.warning(f"Failed to generate fingerprint: {e}")
            return np.zeros(self.n_bits, dtype=np.int8)

    def generate_batch(self, mols: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Generate fingerprint matrix for multiple molecules.
        
        Args:
            mols: A list of RDKit Mol objects (or None for invalid molecules).
            
        Returns:
            A 2D numpy array of shape (n_molecules, n_bits).
            Rows corresponding to None molecules contain zero vectors.
        """
        if not mols:
            return np.empty((0, self.n_bits), dtype=np.int8)
        
        results = []
        for mol in mols:
            fp_vec = self.generate(mol)
            results.append(fp_vec)
        
        return np.vstack(results)
