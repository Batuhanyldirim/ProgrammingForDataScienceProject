"""Main Pipeline module for end-to-end molecular activity prediction.

This module orchestrates the complete workflow from data loading through
model training, evaluation, and prediction generation.
"""

import logging
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .data_loader import DataLoader
from .smiles_parser import SMILESParser
from .descriptor_calculator import DescriptorCalculator
from .fingerprint_generator import FingerprintGenerator
from .feature_manager import FeatureManager
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end pipeline for molecular activity prediction.
    
    This class coordinates all components to:
    1. Load and parse training data
    2. Generate multiple feature representations
    3. Train and evaluate multiple model/feature combinations
    4. Select the best model based on cross-validation AUC
    5. Generate predictions for test data
    """

    FEATURE_TYPES = ['descriptors', 'fingerprints', 'combined']
    ALGORITHMS = ['random_forest', 'gradient_boosting']

    def __init__(self, n_folds: int = 5):
        """Initialize the pipeline.
        
        Args:
            n_folds: Number of folds for cross-validation.
        """
        self.n_folds = n_folds
        
        # Initialize components
        self.data_loader = DataLoader()
        self.smiles_parser = SMILESParser()
        self.descriptor_calc = DescriptorCalculator()
        self.fingerprint_gen = FingerprintGenerator()
        self.evaluator = ModelEvaluator(n_folds=n_folds)
        self.predictor = ModelPredictor()
        
        # Feature managers for each feature type (to maintain separate preprocessing)
        self.feature_managers: Dict[str, FeatureManager] = {}
        
        # Store results
        self.results: List[Dict] = []
        self.best_model = None
        self.best_feature_type = None
        self.best_cv_auc = 0.0
        
        logger.info(f"Initialized Pipeline with {n_folds}-fold cross-validation")


    def _create_feature_manager(self) -> FeatureManager:
        """Create a new FeatureManager instance.
        
        Returns:
            A new FeatureManager with fresh preprocessing state.
        """
        return FeatureManager(
            descriptor_calc=self.descriptor_calc,
            fingerprint_gen=self.fingerprint_gen
        )

    def _generate_features(
        self, 
        mols: List, 
        feature_type: str, 
        feature_manager: FeatureManager,
        fit: bool = True
    ) -> np.ndarray:
        """Generate features of the specified type.
        
        Args:
            mols: List of RDKit Mol objects.
            feature_type: One of 'descriptors', 'fingerprints', 'combined'.
            feature_manager: FeatureManager instance to use.
            fit: Whether to fit the preprocessor.
            
        Returns:
            Preprocessed feature matrix.
        """
        if feature_type == 'descriptors':
            features = feature_manager.create_descriptor_features(mols)
        elif feature_type == 'fingerprints':
            features = feature_manager.create_fingerprint_features(mols)
        elif feature_type == 'combined':
            features = feature_manager.create_combined_features(mols)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        return feature_manager.preprocess(features, fit=fit)

    def _create_model(self, algorithm: str) -> Any:
        """Create a model instance for the specified algorithm.
        
        Args:
            algorithm: One of 'random_forest', 'gradient_boosting'.
            
        Returns:
            An unfitted classifier instance.
        """
        if algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def load_and_parse_training_data(
        self, 
        filepath: str
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """Load training data and parse SMILES to molecules.
        
        Args:
            filepath: Path to training CSV file.
            
        Returns:
            Tuple of (molecules, labels, indices).
        """
        logger.info(f"Loading training data from {filepath}")
        df, labels = self.data_loader.load_training_data(filepath)
        
        logger.info("Parsing SMILES strings to molecules")
        smiles_list = df['SMILES'].tolist()
        mols = self.smiles_parser.parse_batch(smiles_list)
        indices = df['INDEX'].values
        
        # Count valid molecules
        valid_count = sum(1 for m in mols if m is not None)
        logger.info(f"Parsed {valid_count}/{len(mols)} valid molecules")
        
        return mols, labels, indices


    def train_and_evaluate(
        self, 
        mols: List, 
        labels: np.ndarray
    ) -> pd.DataFrame:
        """Train and evaluate all model/feature combinations.
        
        Args:
            mols: List of RDKit Mol objects.
            labels: Activity labels.
            
        Returns:
            DataFrame with evaluation results for all combinations.
        """
        logger.info("Starting model training and evaluation")
        logger.info(f"Feature types: {self.FEATURE_TYPES}")
        logger.info(f"Algorithms: {self.ALGORITHMS}")
        
        self.results = []
        total_combinations = len(self.FEATURE_TYPES) * len(self.ALGORITHMS)
        current = 0
        
        for feature_type in self.FEATURE_TYPES:
            # Create a fresh feature manager for each feature type
            feature_manager = self._create_feature_manager()
            self.feature_managers[feature_type] = feature_manager
            
            logger.info(f"Generating {feature_type} features...")
            X = self._generate_features(mols, feature_type, feature_manager, fit=True)
            logger.info(f"Feature matrix shape: {X.shape}")
            
            for algorithm in self.ALGORITHMS:
                current += 1
                logger.info(
                    f"[{current}/{total_combinations}] Evaluating {algorithm} "
                    f"with {feature_type} features"
                )
                
                model = self._create_model(algorithm)
                cv_results = self.evaluator.cross_validate(model, X, labels)
                
                result = {
                    'feature_type': feature_type,
                    'algorithm': algorithm,
                    'auc_mean': cv_results['auc_mean'],
                    'auc_std': cv_results['auc_std'],
                    'auc_scores': cv_results['auc_scores']
                }
                self.results.append(result)
                
                logger.info(
                    f"  AUC: {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}"
                )
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('auc_mean', ascending=False).reset_index(drop=True)
        
        return results_df

    def select_best_model(self, results_df: pd.DataFrame, mols: List, labels: np.ndarray):
        """Select and train the best model based on CV results.
        
        Args:
            results_df: DataFrame with evaluation results.
            mols: List of RDKit Mol objects.
            labels: Activity labels.
        """
        best_row = results_df.iloc[0]
        self.best_feature_type = best_row['feature_type']
        best_algorithm = best_row['algorithm']
        self.best_cv_auc = best_row['auc_mean']
        
        logger.info(
            f"Best model: {best_algorithm} with {self.best_feature_type} features "
            f"(CV AUC: {self.best_cv_auc:.4f})"
        )
        
        # Retrain on full training data
        logger.info("Retraining best model on full training data")
        feature_manager = self.feature_managers[self.best_feature_type]
        X = self._generate_features(mols, self.best_feature_type, feature_manager, fit=True)
        
        trainer = ModelTrainer(algorithm=best_algorithm)
        self.best_model = trainer.train(X, labels)
        
        logger.info("Best model trained successfully")


    def report_results(self, results_df: pd.DataFrame):
        """Print evaluation results and estimated test AUC.
        
        Args:
            results_df: DataFrame with evaluation results.
        """
        print("\n" + "=" * 70)
        print("MODEL EVALUATION RESULTS")
        print("=" * 70)
        
        print("\nAll Model/Feature Combinations (ranked by CV AUC):")
        print("-" * 70)
        
        for idx, row in results_df.iterrows():
            print(
                f"{idx + 1}. {row['algorithm']:20s} + {row['feature_type']:12s} | "
                f"AUC: {row['auc_mean']:.4f} ± {row['auc_std']:.4f}"
            )
        
        print("\n" + "-" * 70)
        print(f"BEST MODEL: {results_df.iloc[0]['algorithm']} "
              f"with {results_df.iloc[0]['feature_type']} features")
        print(f"ESTIMATED TEST AUC: {self.best_cv_auc:.4f}")
        print("=" * 70 + "\n")

    def run_training_pipeline(self, training_filepath: str) -> pd.DataFrame:
        """Run the complete training pipeline.
        
        Args:
            training_filepath: Path to training CSV file.
            
        Returns:
            DataFrame with evaluation results.
        """
        logger.info("Starting training pipeline")
        
        # Load and parse data
        mols, labels, indices = self.load_and_parse_training_data(training_filepath)
        
        # Train and evaluate all combinations
        results_df = self.train_and_evaluate(mols, labels)
        
        # Select best model
        self.select_best_model(results_df, mols, labels)
        
        # Report results
        self.report_results(results_df)
        
        return results_df

    def load_and_parse_test_data(self, filepath: str) -> Tuple[List, np.ndarray]:
        """Load test data and parse SMILES to molecules.
        
        Args:
            filepath: Path to test CSV file.
            
        Returns:
            Tuple of (molecules, indices).
        """
        logger.info(f"Loading test data from {filepath}")
        df = self.data_loader.load_test_data(filepath)
        
        logger.info("Parsing SMILES strings to molecules")
        smiles_list = df['SMILES'].tolist()
        mols = self.smiles_parser.parse_batch(smiles_list)
        indices = df['INDEX'].values
        
        # Count valid molecules
        valid_count = sum(1 for m in mols if m is not None)
        logger.info(f"Parsed {valid_count}/{len(mols)} valid molecules")
        
        return mols, indices

    def predict_test_set(
        self, 
        test_filepath: str, 
        output_filepath: str
    ) -> np.ndarray:
        """Generate predictions for the test set and save to CSV.
        
        Uses the best model selected during training to generate predictions.
        Features are generated using the same pipeline as training (with fit=False
        to use the fitted preprocessor).
        
        Args:
            test_filepath: Path to test CSV file.
            output_filepath: Path to save predictions CSV.
            
        Returns:
            Array of probability predictions.
            
        Raises:
            ValueError: If no model has been trained yet.
        """
        if self.best_model is None:
            raise ValueError(
                "No model has been trained. Run run_training_pipeline first."
            )
        
        if self.best_feature_type is None:
            raise ValueError(
                "No feature type selected. Run run_training_pipeline first."
            )
        
        logger.info("Starting test set prediction")
        
        # Load and parse test data
        mols, indices = self.load_and_parse_test_data(test_filepath)
        
        # Generate features using the same pipeline as training
        feature_manager = self.feature_managers[self.best_feature_type]
        logger.info(f"Generating {self.best_feature_type} features for test set")
        X_test = self._generate_features(
            mols, 
            self.best_feature_type, 
            feature_manager, 
            fit=False  # Use fitted preprocessor from training
        )
        logger.info(f"Test feature matrix shape: {X_test.shape}")
        
        # Generate predictions
        probabilities = self.predictor.predict_proba(self.best_model, X_test)
        
        # Save predictions to CSV
        self.predictor.save_predictions(indices, probabilities, output_filepath)
        
        # Report
        print("\n" + "=" * 70)
        print("TEST SET PREDICTIONS")
        print("=" * 70)
        print(f"Test samples: {len(indices)}")
        print(f"Predictions saved to: {output_filepath}")
        print(f"Estimated AUC (from CV): {self.best_cv_auc:.4f}")
        print("=" * 70 + "\n")
        
        logger.info(f"Test set prediction complete. Saved to {output_filepath}")
        
        return probabilities

    def run_full_pipeline(
        self, 
        training_filepath: str, 
        test_filepath: str, 
        output_filepath: str
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Run the complete pipeline: training, evaluation, and test prediction.
        
        Args:
            training_filepath: Path to training CSV file.
            test_filepath: Path to test CSV file.
            output_filepath: Path to save predictions CSV.
            
        Returns:
            Tuple of (evaluation results DataFrame, test predictions array).
        """
        # Run training pipeline
        results_df = self.run_training_pipeline(training_filepath)
        
        # Generate test predictions
        predictions = self.predict_test_set(test_filepath, output_filepath)
        
        return results_df, predictions


def main():
    """Main entry point for the pipeline."""
    # Default file paths
    training_file = 'training_smiles.csv'
    test_file = 'test_smiles.csv'
    output_file = 'predictions.csv'
    
    # Create and run pipeline
    pipeline = Pipeline(n_folds=5)
    
    # Run full pipeline (training + test prediction)
    results_df, predictions = pipeline.run_full_pipeline(
        training_file, 
        test_file, 
        output_file
    )
    
    logger.info("Full pipeline complete")
    
    return pipeline, results_df, predictions


if __name__ == '__main__':
    main()
