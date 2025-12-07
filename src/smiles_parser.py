"""SMILES Parser module for converting SMILES strings to RDKit Mol objects.

This module provides functionality to parse SMILES (Simplified Molecular Input
Line Entry System) strings into RDKit Mol objects for further processing.
"""

import logging
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Mol

# Configure module logger
logger = logging.getLogger(__name__)


class SMILESParser:
    """Parser for converting SMILES strings to RDKit Mol objects.
    
    This class provides methods to parse individual SMILES strings or batches
    of SMILES strings, with logging for invalid entries.
    """

    def parse(self, smiles: str) -> Optional[Mol]:
        """Convert a SMILES string to an RDKit Mol object.
        
        Args:
            smiles: A SMILES string representing a molecular structure.
            
        Returns:
            An RDKit Mol object if the SMILES is valid, None otherwise.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES string: '{smiles}'")
        return mol

    def parse_batch(self, smiles_list: List[str]) -> List[Optional[Mol]]:
        """Parse multiple SMILES strings to RDKit Mol objects.
        
        Args:
            smiles_list: A list of SMILES strings to parse.
            
        Returns:
            A list of RDKit Mol objects (or None for invalid SMILES),
            maintaining the same order as the input list.
        """
        results = []
        for smiles in smiles_list:
            mol = self.parse(smiles)
            results.append(mol)
        return results
