"""Descriptor Calculator module for computing molecular descriptors.

This module provides functionality to compute molecular descriptors from RDKit
Mol objects, including rdMolDescriptors, Lipinski descriptors, and Fragment counts.
"""

import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Fragments
from rdkit.Chem.rdMolDescriptors import (
    CalcExactMolWt,
    CalcTPSA,
    CalcNumRotatableBonds,
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRings,
    CalcNumAromaticRings,
    CalcNumAliphaticRings,
    CalcNumHeteroatoms,
    CalcFractionCSP3,
    CalcNumAmideBonds,
    CalcNumHeavyAtoms,
    CalcLabuteASA,
    CalcNumAtomStereoCenters,
    CalcNumSaturatedRings,
)

# Configure module logger
logger = logging.getLogger(__name__)


class DescriptorCalculator:
    """Calculator for molecular descriptors using RDKit.
    
    This class computes a fixed-length vector of molecular descriptors including
    rdMolDescriptors, Lipinski descriptors, and Fragment counts.
    """

    # rdMolDescriptors functions
    RDMOL_DESCRIPTORS = [
        ('ExactMolWt', CalcExactMolWt),
        ('TPSA', CalcTPSA),
        ('NumRotatableBonds', CalcNumRotatableBonds),
        ('NumHBA', CalcNumHBA),
        ('NumHBD', CalcNumHBD),
        ('NumRings', CalcNumRings),
        ('NumAromaticRings', CalcNumAromaticRings),
        ('NumAliphaticRings', CalcNumAliphaticRings),
        ('NumHeteroatoms', CalcNumHeteroatoms),
        ('NumAmideBonds', CalcNumAmideBonds),
        ('NumHeavyAtoms', CalcNumHeavyAtoms),
        ('LabuteASA', CalcLabuteASA),
        ('NumAtomStereoCenters', CalcNumAtomStereoCenters),
        ('NumSaturatedRings', CalcNumSaturatedRings),
    ]

    # Lipinski descriptors
    LIPINSKI_DESCRIPTORS = [
        ('HeavyAtomCount', Lipinski.HeavyAtomCount),
        ('NumHDonors', Lipinski.NumHDonors),
        ('NumHAcceptors', Lipinski.NumHAcceptors),
        ('FractionCSP3', CalcFractionCSP3),
    ]

    # Fragment descriptors - common functional groups
    FRAGMENT_DESCRIPTORS = [
        ('fr_Al_COO', Fragments.fr_Al_COO),
        ('fr_Al_OH', Fragments.fr_Al_OH),
        ('fr_Al_OH_noTert', Fragments.fr_Al_OH_noTert),
        ('fr_ArN', Fragments.fr_ArN),
        ('fr_Ar_COO', Fragments.fr_Ar_COO),
        ('fr_Ar_N', Fragments.fr_Ar_N),
        ('fr_Ar_NH', Fragments.fr_Ar_NH),
        ('fr_Ar_OH', Fragments.fr_Ar_OH),
        ('fr_COO', Fragments.fr_COO),
        ('fr_COO2', Fragments.fr_COO2),
        ('fr_C_O', Fragments.fr_C_O),
        ('fr_C_O_noCOO', Fragments.fr_C_O_noCOO),
        ('fr_C_S', Fragments.fr_C_S),
        ('fr_HOCCN', Fragments.fr_HOCCN),
        ('fr_Imine', Fragments.fr_Imine),
        ('fr_NH0', Fragments.fr_NH0),
        ('fr_NH1', Fragments.fr_NH1),
        ('fr_NH2', Fragments.fr_NH2),
        ('fr_N_O', Fragments.fr_N_O),
        ('fr_Ndealkylation1', Fragments.fr_Ndealkylation1),
        ('fr_Ndealkylation2', Fragments.fr_Ndealkylation2),
        ('fr_Nhpyrrole', Fragments.fr_Nhpyrrole),
        ('fr_SH', Fragments.fr_SH),
        ('fr_aldehyde', Fragments.fr_aldehyde),
        ('fr_alkyl_carbamate', Fragments.fr_alkyl_carbamate),
        ('fr_alkyl_halide', Fragments.fr_alkyl_halide),
        ('fr_allylic_oxid', Fragments.fr_allylic_oxid),
        ('fr_amide', Fragments.fr_amide),
        ('fr_amidine', Fragments.fr_amidine),
        ('fr_aniline', Fragments.fr_aniline),
        ('fr_aryl_methyl', Fragments.fr_aryl_methyl),
        ('fr_azide', Fragments.fr_azide),
        ('fr_azo', Fragments.fr_azo),
        ('fr_barbitur', Fragments.fr_barbitur),
        ('fr_benzene', Fragments.fr_benzene),
        ('fr_benzodiazepine', Fragments.fr_benzodiazepine),
        ('fr_bicyclic', Fragments.fr_bicyclic),
        ('fr_diazo', Fragments.fr_diazo),
        ('fr_dihydropyridine', Fragments.fr_dihydropyridine),
        ('fr_epoxide', Fragments.fr_epoxide),
        ('fr_ester', Fragments.fr_ester),
        ('fr_ether', Fragments.fr_ether),
        ('fr_furan', Fragments.fr_furan),
        ('fr_guanido', Fragments.fr_guanido),
        ('fr_halogen', Fragments.fr_halogen),
        ('fr_hdrzine', Fragments.fr_hdrzine),
        ('fr_hdrzone', Fragments.fr_hdrzone),
        ('fr_imidazole', Fragments.fr_imidazole),
        ('fr_imide', Fragments.fr_imide),
        ('fr_isocyan', Fragments.fr_isocyan),
        ('fr_isothiocyan', Fragments.fr_isothiocyan),
        ('fr_ketone', Fragments.fr_ketone),
        ('fr_ketone_Topliss', Fragments.fr_ketone_Topliss),
        ('fr_lactam', Fragments.fr_lactam),
        ('fr_lactone', Fragments.fr_lactone),
        ('fr_methoxy', Fragments.fr_methoxy),
        ('fr_morpholine', Fragments.fr_morpholine),
        ('fr_nitrile', Fragments.fr_nitrile),
        ('fr_nitro', Fragments.fr_nitro),
        ('fr_nitro_arom', Fragments.fr_nitro_arom),
        ('fr_nitro_arom_nonortho', Fragments.fr_nitro_arom_nonortho),
        ('fr_nitroso', Fragments.fr_nitroso),
        ('fr_oxazole', Fragments.fr_oxazole),
        ('fr_oxime', Fragments.fr_oxime),
        ('fr_para_hydroxylation', Fragments.fr_para_hydroxylation),
        ('fr_phenol', Fragments.fr_phenol),
        ('fr_phenol_noOrthoHbond', Fragments.fr_phenol_noOrthoHbond),
        ('fr_phos_acid', Fragments.fr_phos_acid),
        ('fr_phos_ester', Fragments.fr_phos_ester),
        ('fr_piperdine', Fragments.fr_piperdine),
        ('fr_piperzine', Fragments.fr_piperzine),
        ('fr_priamide', Fragments.fr_priamide),
        ('fr_prisulfonamd', Fragments.fr_prisulfonamd),
        ('fr_pyridine', Fragments.fr_pyridine),
        ('fr_quatN', Fragments.fr_quatN),
        ('fr_sulfide', Fragments.fr_sulfide),
        ('fr_sulfonamd', Fragments.fr_sulfonamd),
        ('fr_sulfone', Fragments.fr_sulfone),
        ('fr_term_acetylene', Fragments.fr_term_acetylene),
        ('fr_tetrazole', Fragments.fr_tetrazole),
        ('fr_thiazole', Fragments.fr_thiazole),
        ('fr_thiocyan', Fragments.fr_thiocyan),
        ('fr_thiophene', Fragments.fr_thiophene),
        ('fr_unbrch_alkane', Fragments.fr_unbrch_alkane),
        ('fr_urea', Fragments.fr_urea),
    ]

    def __init__(self, descriptor_names: Optional[List[str]] = None):
        """Initialize the descriptor calculator.
        
        Args:
            descriptor_names: Optional list of descriptor names to compute.
                If None, all available descriptors are computed.
        """
        self._all_descriptors = (
            self.RDMOL_DESCRIPTORS + 
            self.LIPINSKI_DESCRIPTORS + 
            self.FRAGMENT_DESCRIPTORS
        )
        
        if descriptor_names is not None:
            # Filter to only requested descriptors
            name_set = set(descriptor_names)
            self._descriptors = [
                (name, func) for name, func in self._all_descriptors
                if name in name_set
            ]
        else:
            self._descriptors = self._all_descriptors
        
        self._descriptor_names = [name for name, _ in self._descriptors]
        logger.info(f"Initialized DescriptorCalculator with {len(self._descriptors)} descriptors")


    def get_descriptor_names(self) -> List[str]:
        """Return names of computed descriptors.
        
        Returns:
            List of descriptor names in the order they appear in the output vector.
        """
        return self._descriptor_names.copy()

    def calculate(self, mol: Optional[Chem.Mol]) -> np.ndarray:
        """Compute descriptor vector for a single molecule.
        
        Args:
            mol: An RDKit Mol object, or None for invalid molecules.
            
        Returns:
            A numpy array of descriptor values. If mol is None or descriptor
            calculation fails, returns a vector of NaN values.
        """
        n_descriptors = len(self._descriptors)
        
        # Handle None molecules
        if mol is None:
            logger.warning("Received None molecule, returning NaN vector")
            return np.full(n_descriptors, np.nan)
        
        values = []
        for name, func in self._descriptors:
            try:
                value = func(mol)
                values.append(float(value))
            except Exception as e:
                logger.warning(f"Failed to compute descriptor '{name}': {e}")
                values.append(np.nan)
        
        return np.array(values, dtype=np.float64)

    def calculate_batch(self, mols: List[Optional[Chem.Mol]]) -> np.ndarray:
        """Compute descriptor matrix for multiple molecules.
        
        Args:
            mols: A list of RDKit Mol objects (or None for invalid molecules).
            
        Returns:
            A 2D numpy array of shape (n_molecules, n_descriptors).
            Rows corresponding to None molecules contain NaN values.
        """
        if not mols:
            return np.empty((0, len(self._descriptors)), dtype=np.float64)
        
        results = []
        for i, mol in enumerate(mols):
            descriptor_vec = self.calculate(mol)
            results.append(descriptor_vec)
        
        return np.vstack(results)
