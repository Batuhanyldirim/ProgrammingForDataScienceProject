"""Main Pipeline module for end-to-end molecular activity prediction.

This module orchestrates the complete workflow from data loading through
model training, evaluation, and prediction generation.
"""

import logging
import sys
import json
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import xgboost as xgb

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
    3. Train and evaluate multiple model/feature combinations (with param sweeps)
    4. Select the best model based on cross-validation AUC
    5. Produce a hold-out AUC estimate to reduce selection bias
    6. Generate predictions for test data
    """

    FEATURE_TYPES = ['descriptors', 'fingerprints', 'combined']
    MODEL_CONFIGS = {
        'random_forest': [
            {'n_estimators': 200, 'max_depth': None, 'max_features': None, 'min_samples_leaf': 1},
            {'n_estimators': 400, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1},
            {'n_estimators': 600, 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1},
            {'n_estimators': 300, 'max_depth': 25, 'max_features': None, 'min_samples_leaf': 2},
            {'n_estimators': 400, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2},
            {'n_estimators': 500, 'max_depth': 40, 'max_features': None, 'min_samples_leaf': 1},
        ],
        'xgboost': [
            {'n_estimators': 400, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 600, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.9, 'colsample_bytree': 0.9},
            {'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 8, 'subsample': 0.8, 'colsample_bytree': 0.8},
            {'n_estimators': 500, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9},
            {'n_estimators': 700, 'learning_rate': 0.07, 'max_depth': 6, 'subsample': 0.85, 'colsample_bytree': 0.85},
            {'n_estimators': 900, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        ],
    }
    QUICK_MODEL_CONFIGS = {
        'random_forest': [
            {'n_estimators': 200, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1},
        ],
        'xgboost': [
            {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9},
        ],
    }

    @staticmethod
    def _combo_key(feature_type: str, algorithm: str, params: Dict[str, Any]) -> str:
        """Create a stable key for a model/feature/param combination."""
        return f"{feature_type}::{algorithm}::{json.dumps(params, sort_keys=True)}"

    def __init__(
        self, 
        n_folds: int = 5, 
        show_progress: bool = True,
        model_configs: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        allowed_combinations: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None
    ):
        """Initialize the pipeline.
        
        Args:
            n_folds: Number of folds for cross-validation.
            show_progress: Whether to show progress bars during training.
            model_configs: Optional override for model hyperparameter grids.
            allowed_combinations: Optional list of (feature_type, algorithm, params) tuples
                to restrict which combos are evaluated.
        """
        self.n_folds = n_folds
        self.show_progress = show_progress
        self.model_configs = model_configs or self.MODEL_CONFIGS
        self.allowed_combinations = allowed_combinations
        self.allowed_combo_keys: Optional[Set[str]] = None
        self.allowed_feature_types: Optional[Set[str]] = None
        if allowed_combinations:
            self.allowed_combo_keys = {
                self._combo_key(ft, alg, params) for ft, alg, params in allowed_combinations
            }
            self.allowed_feature_types = {ft for ft, _, _ in allowed_combinations}
        
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
        self.best_algorithm = None
        self.best_params: Dict[str, Any] = {}
        self.holdout_auc = None
        
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
        fit: bool = True,
        preprocess: bool = True
    ) -> np.ndarray:
        """Generate features of the specified type.
        
        Args:
            mols: List of RDKit Mol objects.
            feature_type: One of 'descriptors', 'fingerprints', 'combined'.
            feature_manager: FeatureManager instance to use.
            fit: Whether to fit the preprocessor.
            preprocess: Whether to apply imputation/scaling via FeatureManager.
            
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
        
        if not preprocess:
            return features
        
        return feature_manager.preprocess(features, fit=fit)

    def _create_model(
        self, 
        algorithm: str, 
        params: Dict[str, Any], 
        with_preprocessing: bool = False
    ) -> Any:
        """Create a model instance for the specified algorithm.
        
        Args:
            algorithm: One of 'random_forest', 'gradient_boosting'.
            params: Hyperparameter dictionary for the estimator.
            with_preprocessing: If True, wrap the classifier with an imputer and scaler
                so preprocessing is fitted inside each CV fold.
            
        Returns:
            An unfitted classifier instance.
        """
        if algorithm == 'random_forest':
            model = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                **params
            )
        elif algorithm == 'xgboost':
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                **params
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        if not with_preprocessing:
            return model

        return SklearnPipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model),
        ])

    def load_and_parse_training_data(
        self, 
        filepath: str,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ) -> Tuple[List, np.ndarray, np.ndarray]:
        """Load training data and parse SMILES to molecules.
        
        Args:
            filepath: Path to training CSV file.
            sample_size: Optional number of samples to randomly select for a quick run.
            random_state: Random seed for sampling.
            
        Returns:
            Tuple of (molecules, labels, indices).
        """
        logger.info(f"Loading training data from {filepath}")
        df, labels = self.data_loader.load_training_data(filepath)

        if sample_size is not None and sample_size < len(df):
            logger.info(f"Subsampling training data to {sample_size} rows for quick run")
            all_idx = np.arange(len(df))
            _, sample_idx = train_test_split(
                all_idx,
                train_size=sample_size,
                stratify=labels,
                random_state=random_state
            )
            df = df.iloc[sample_idx].reset_index(drop=True)
            labels = labels[sample_idx]
        
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
        logger.info(f"Algorithms: {list(self.model_configs.keys())}")
        
        # Build combination list honoring allowed filters
        combos_by_feature = {}
        for feature_type in self.FEATURE_TYPES:
            if self.allowed_feature_types and feature_type not in self.allowed_feature_types:
                continue
            alg_list = []
            for algorithm, configs in self.model_configs.items():
                for params in configs:
                    key = self._combo_key(feature_type, algorithm, params)
                    if self.allowed_combo_keys and key not in self.allowed_combo_keys:
                        continue
                    alg_list.append((algorithm, params))
            if alg_list:
                combos_by_feature[feature_type] = alg_list

        self.results = []
        total_combinations = sum(len(v) for v in combos_by_feature.values())
        current = 0
        progress_bar = None
        if self.show_progress:
            progress_bar = tqdm(
                total=total_combinations,
                desc="Model/feature evaluations",
                unit="combo"
            )
        
        for feature_type, alg_param_list in combos_by_feature.items():
            # Create a fresh feature manager for each feature type
            feature_manager = self._create_feature_manager()
            self.feature_managers[feature_type] = feature_manager
            
            logger.info(f"Generating {feature_type} features...")
            # Generate raw features; preprocessing happens inside CV to avoid leakage
            X = self._generate_features(
                mols, feature_type, feature_manager, fit=False, preprocess=False
            )
            logger.info(f"Feature matrix shape: {X.shape}")
            
            for algorithm, params in alg_param_list:
                current += 1
                logger.info(
                    f"[{current}/{total_combinations}] Evaluating {algorithm} "
                    f"with {feature_type} features | params={params}"
                )
                
                model = self._create_model(
                    algorithm, params, with_preprocessing=True
                )
                cv_results = self.evaluator.cross_validate(
                    model, X, labels, show_progress=self.show_progress
                )
                
                result = {
                    'feature_type': feature_type,
                    'algorithm': algorithm,
                    'params': params,
                    'auc_mean': cv_results['auc_mean'],
                    'auc_std': cv_results['auc_std'],
                    'auc_scores': cv_results['auc_scores']
                }
                self.results.append(result)
                
                logger.info(
                    f"  AUC: {cv_results['auc_mean']:.4f} ± {cv_results['auc_std']:.4f}"
                )

                if progress_bar:
                    progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()

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
        self.best_algorithm = best_row['algorithm']
        self.best_params = best_row.get('params', {})
        self.best_cv_auc = best_row['auc_mean']
        
        logger.info(
            f"Best model: {self.best_algorithm} with {self.best_feature_type} features "
            f"(CV AUC: {self.best_cv_auc:.4f}, params={self.best_params})"
        )
        
        # Retrain on full training data
        logger.info("Retraining best model on full training data")
        feature_manager = self.feature_managers[self.best_feature_type]
        X = self._generate_features(
            mols, self.best_feature_type, feature_manager, fit=False, preprocess=False
        )
        
        self.best_model = self._create_model(
            self.best_algorithm, self.best_params, with_preprocessing=True
        )
        self.best_model.fit(X, labels)
        
        logger.info("Best model trained successfully")

    def evaluate_holdout(
        self,
        mols: List,
        labels: np.ndarray,
        test_size: float = 0.1,
        random_state: int = 42
    ) -> float:
        """Estimate AUC on a hold-out split to reduce model-selection bias."""
        feature_manager = self.feature_managers[self.best_feature_type]
        X = self._generate_features(
            mols, self.best_feature_type, feature_manager, fit=False, preprocess=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=test_size, stratify=labels, random_state=random_state
        )

        holdout_model = self._create_model(
            self.best_algorithm, self.best_params, with_preprocessing=True
        )
        holdout_model.fit(X_train, y_train)
        y_val_proba = holdout_model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        self.holdout_auc = auc

        logger.info(f"Hold-out AUC (test_size={test_size}): {auc:.4f}")
        return auc


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
        if self.holdout_auc is not None:
            print(f"HOLD-OUT AUC (bias-reduced): {self.holdout_auc:.4f}")
        print("=" * 70 + "\n")

    def run_training_pipeline(
        self, 
        training_filepath: str,
        cv_results_path: Optional[str] = None,
        best_summary_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Run the complete training pipeline.
        
        Args:
            training_filepath: Path to training CSV file.
            cv_results_path: Optional path to save CV leaderboard as CSV.
            best_summary_path: Optional path to save best model summary as JSON.
            
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

        # Hold-out evaluation for unbiased estimate
        self.evaluate_holdout(mols, labels)
        
        # Report results
        self.report_results(results_df)

        # Persist results for reporting
        if cv_results_path:
            results_df.to_csv(cv_results_path, index=False)
            logger.info(f"Saved CV results to {cv_results_path}")

        if best_summary_path:
            summary = {
                "best_algorithm": self.best_algorithm,
                "best_feature_type": self.best_feature_type,
                "best_params": self.best_params,
                "cv_auc_mean": self.best_cv_auc,
                "holdout_auc": self.holdout_auc,
                "n_folds": self.n_folds,
            }
            with open(best_summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Saved best model summary to {best_summary_path}")
        
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
        output_filepath: str,
        submission_filepath: Optional[str] = None,
        auc_estimate: Optional[float] = None
    ) -> np.ndarray:
        """Generate predictions for the test set and save to CSV.
        
        Uses the best model selected during training to generate predictions.
        Features are generated using the same pipeline as training (with fit=False
        to use the fitted preprocessor).
        
        Args:
            test_filepath: Path to test CSV file.
            output_filepath: Path to save predictions CSV.
            submission_filepath: Optional path to save competition-style submission
                with AUC estimate on first line followed by probabilities.
            auc_estimate: Optional AUC estimate to place on first line. If None,
                uses holdout AUC if available, else best CV AUC.
            
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
            fit=False,
            preprocess=False  # Preprocessing is inside the trained model pipeline
        )
        logger.info(f"Test feature matrix shape: {X_test.shape}")
        
        # Generate predictions
        probabilities = self.predictor.predict_proba(self.best_model, X_test)
        
        # Save predictions to CSV
        self.predictor.save_predictions(indices, probabilities, output_filepath)

        if submission_filepath:
            # Use holdout AUC if available, otherwise CV mean
            estimate = auc_estimate
            if estimate is None:
                estimate = self.holdout_auc if self.holdout_auc is not None else self.best_cv_auc
            if estimate is None:
                raise ValueError("No AUC estimate available to write submission file.")
            submission_values = np.concatenate([[estimate], probabilities])
            np.savetxt(submission_filepath, submission_values, fmt="%.10f")
            logger.info(
                f"Saved submission file with AUC estimate to {submission_filepath} "
                f"(entries: {len(submission_values)})"
            )
        
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


def run_quick_demo(
    training_file: str = 'training_smiles.csv',
    test_file: str = 'test_smiles.csv',
    output_file: str = 'predictions_quick.csv',
    sample_size: int = 5000,
    n_folds: int = 2
):
    """Run a small proof-of-concept pipeline on a subsample for speed."""
    pipeline = Pipeline(
        n_folds=n_folds,
        show_progress=True,
        model_configs=Pipeline.QUICK_MODEL_CONFIGS
    )

    logger.info(f"Starting quick demo with sample_size={sample_size}, n_folds={n_folds}")

    # Load subsampled data
    mols, labels, _ = pipeline.load_and_parse_training_data(
        training_file, sample_size=sample_size
    )

    # Train/evaluate and select best model
    results_df = pipeline.train_and_evaluate(mols, labels)
    pipeline.select_best_model(results_df, mols, labels)
    pipeline.evaluate_holdout(mols, labels, test_size=0.2)
    pipeline.report_results(results_df)

    # Save artifacts for reporting
    results_df.to_csv("cv_results_quick.csv", index=False)
    summary = {
        "best_algorithm": pipeline.best_algorithm,
        "best_feature_type": pipeline.best_feature_type,
        "best_params": pipeline.best_params,
        "cv_auc_mean": pipeline.best_cv_auc,
        "holdout_auc": pipeline.holdout_auc,
        "n_folds": pipeline.n_folds,
        "sample_size": sample_size,
    }
    with open("best_model_summary_quick.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Predict test set with the trained quick model
    pipeline.predict_test_set(test_file, output_file)

    logger.info("Quick demo pipeline complete")
    return pipeline, results_df


def run_two_phase(
    training_file: str = 'training_smiles.csv',
    test_file: str = 'test_smiles.csv',
    output_file: str = 'predictions_two_phase.csv',
    phase_a_sample_size: int = 20000,
    phase_a_folds: int = 2,
    phase_b_folds: int = 5,
    top_k: int = 3,
    phase_a_cv_path: str = "cv_results_phase_a.csv",
    phase_b_cv_path: str = "cv_results_phase_b.csv",
    phase_a_summary_path: str = "phase_a_summary.json",
    phase_b_summary_path: str = "best_model_summary_phase_b.json"
):
    """Two-phase training: fast exploratory pass, then focused refinement."""
    # Phase A: exploratory on subsample with fewer folds
    pipeline_a = Pipeline(
        n_folds=phase_a_folds,
        show_progress=True,
        model_configs=Pipeline.MODEL_CONFIGS
    )
    logger.info(
        f"Starting Phase A (exploratory): sample_size={phase_a_sample_size}, "
        f"n_folds={phase_a_folds}, evaluating full config grid"
    )
    mols_a, labels_a, _ = pipeline_a.load_and_parse_training_data(
        training_file, sample_size=phase_a_sample_size
    )
    results_a = pipeline_a.train_and_evaluate(mols_a, labels_a)
    results_a.to_csv(phase_a_cv_path, index=False)
    top_df = results_a.head(top_k)
    top_records = top_df.to_dict(orient="records")
    with open(phase_a_summary_path, "w") as f:
        json.dump(
            {
                "top_k": top_k,
                "top_models": top_records,
                "n_folds": phase_a_folds,
                "sample_size": phase_a_sample_size,
            },
            f,
            indent=2
        )
    logger.info(f"Phase A complete. Saved results to {phase_a_cv_path} and {phase_a_summary_path}")

    # Build allowed combinations for Phase B
    allowed_combos = [
        (row['feature_type'], row['algorithm'], row['params'])
        for _, row in top_df.iterrows()
    ]

    # Phase B: refinement on full data with restricted combos and full folds
    pipeline_b = Pipeline(
        n_folds=phase_b_folds,
        show_progress=True,
        model_configs=Pipeline.MODEL_CONFIGS,
        allowed_combinations=allowed_combos
    )
    logger.info(
        f"Starting Phase B (refinement): using top {top_k} combos from Phase A, "
        f"n_folds={phase_b_folds}"
    )
    mols_b, labels_b, _ = pipeline_b.load_and_parse_training_data(training_file)
    results_b = pipeline_b.train_and_evaluate(mols_b, labels_b)
    pipeline_b.select_best_model(results_b, mols_b, labels_b)
    pipeline_b.evaluate_holdout(mols_b, labels_b)
    pipeline_b.report_results(results_b)

    results_b.to_csv(phase_b_cv_path, index=False)
    with open(phase_b_summary_path, "w") as f:
        json.dump(
            {
                "best_algorithm": pipeline_b.best_algorithm,
                "best_feature_type": pipeline_b.best_feature_type,
                "best_params": pipeline_b.best_params,
                "cv_auc_mean": pipeline_b.best_cv_auc,
                "holdout_auc": pipeline_b.holdout_auc,
                "n_folds": pipeline_b.n_folds,
                "top_k_source": top_records,
            },
            f,
            indent=2
        )
    logger.info(f"Phase B complete. Saved results to {phase_b_cv_path} and {phase_b_summary_path}")

    # Predict test set with final winner
    pipeline_b.predict_test_set(test_file, output_file)
    logger.info("Two-phase pipeline complete")
    return pipeline_b, results_a, results_b


def main():
    """Main entry point for the pipeline."""
    # Default file paths
    training_file = 'training_smiles.csv'
    test_file = 'test_smiles.csv'
    output_file = 'predictions.csv'
    submission_file = 'submission.txt'  # override with group number if desired
    
    # Quick demo mode: run on a subsample with fewer folds/configs
    if len(sys.argv) > 1 and sys.argv[1] == 'quick_demo':
        run_quick_demo(training_file, test_file, output_file='predictions_quick.csv')
        return
    # Two-phase mode: exploratory + refinement
    if len(sys.argv) > 1 and sys.argv[1] == 'two_phase':
        run_two_phase(
            training_file=training_file,
            test_file=test_file,
            output_file='predictions_two_phase.csv'
        )
        return

    # Create and run full pipeline
    pipeline = Pipeline(n_folds=5)

    # Run full pipeline (training + test prediction) and save artifacts
    results_df = pipeline.run_training_pipeline(
        training_file,
        cv_results_path="cv_results.csv",
        best_summary_path="best_model_summary.json"
    )
    predictions = pipeline.predict_test_set(
        test_file, output_file, submission_filepath=submission_file
    )
    
    logger.info("Full pipeline complete")
    
    return pipeline, results_df, predictions


if __name__ == '__main__':
    main()
