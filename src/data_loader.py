"""Data Loader module for loading and managing CSV datasets.

This module provides functionality to load training and test data from CSV files
containing SMILES strings and activity labels.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


class DataLoader:
    """Loader for CSV files containing molecular data.
    
    This class provides methods to load training data (with activity labels)
    and test data (without activity labels) from CSV files.
    """

    def load_training_data(self, filepath: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load training CSV with SMILES and activity labels.
        
        Args:
            filepath: Path to the training CSV file containing INDEX, SMILES,
                and ACTIVE columns.
                
        Returns:
            A tuple containing:
                - DataFrame with INDEX and SMILES columns
                - numpy array of activity labels (ACTIVE column)
                
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        logger.info(f"Loading training data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        required_columns = {'INDEX', 'SMILES', 'ACTIVE'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        labels = df['ACTIVE'].values
        features_df = df[['INDEX', 'SMILES']].copy()
        
        logger.info(f"Loaded {len(df)} training samples")
        return features_df, labels

    def load_test_data(self, filepath: str) -> pd.DataFrame:
        """Load test CSV with SMILES only (no activity labels required).
        
        Args:
            filepath: Path to the test CSV file containing INDEX and SMILES columns.
                
        Returns:
            DataFrame with INDEX and SMILES columns.
                
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        logger.info(f"Loading test data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        required_columns = {'INDEX', 'SMILES'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        result_df = df[['INDEX', 'SMILES']].copy()
        
        logger.info(f"Loaded {len(df)} test samples")
        return result_df
