"""
Evaluation module for Spaceship Titanic project.
Handles model evaluation, prediction, and submission file generation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import VotingClassifier
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def evaluate_baseline_models(
    X_train: pd.DataFrame, y_train: pd.Series, models: dict
) -> None:
    """
    Evaluate baseline models without hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training target
        models: Dictionary of model instances
    """
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0, stratify=y_train
    )

    logger.info("=" * 50)
    logger.info("Baseline Model Evaluation")
    logger.info("=" * 50)

    for name, model in models.items():
        model.fit(X_train_split, y_train_split)
        train_score = model.score(X_train_split, y_train_split)
        test_score = model.score(X_test_split, y_test_split)
        logger.info(f"{name.upper()}")
        logger.info(f"  Train accuracy: {train_score:.4f}")
        logger.info(f"  Test accuracy: {test_score:.4f}")


def evaluate_optimized_models(
    X_train: pd.DataFrame, y_train: pd.Series, models: dict, cv_splits: int = 5
) -> dict:
    """
    Evaluate optimized models using cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        models: Dictionary of model instances with optimized parameters
        cv_splits: Number of cross-validation splits

    Returns:
        Dictionary of cross-validation scores for each model
    """
    from sklearn.model_selection import StratifiedKFold

    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    cv_scores = {}

    logger.info("=" * 50)
    logger.info("Optimized Model Cross-Validation Results")
    logger.info("=" * 50)

    for name, model in models.items():
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        cv_scores[name] = scores["test_score"]
        mean_score = scores["test_score"].mean()
        std_score = scores["test_score"].std()
        logger.info(f"{name.upper()}")
        logger.info(f"  Mean: {mean_score:.4f}, Std: {std_score:.4f}")

    return cv_scores


def evaluate_voting_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, models: dict, cv_splits: int = 5
) -> dict:
    """
    Evaluate a voting classifier ensemble.

    Args:
        X_train: Training features
        y_train: Training target
        models: Dictionary of model instances
        cv_splits: Number of cross-validation splits

    Returns:
        Cross-validation scores
    """
    from sklearn.model_selection import StratifiedKFold

    estimators = [(name, model) for name, model in models.items()]
    voting = VotingClassifier(estimators)

    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    scores = cross_validate(voting, X=X_train, y=y_train, cv=kf)

    logger.info("=" * 50)
    logger.info("Voting Classifier Results")
    logger.info("=" * 50)
    logger.info(
        f"Mean: {scores['test_score'].mean():.4f}, Std: {scores['test_score'].std():.4f}"
    )

    return scores


def train_final_models(X_train: pd.DataFrame, y_train: pd.Series, models: dict) -> dict:
    """
    Train final models on the full training set.

    Args:
        X_train: Training features
        y_train: Training target
        models: Dictionary of model instances

    Returns:
        Dictionary of trained models
    """
    trained_models = {}

    logger.info("Training final models on full dataset...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"  {name.upper()} trained")

    return trained_models


def generate_predictions(trained_models: dict, X_test: pd.DataFrame) -> dict:
    """
    Generate predictions for test data using trained models.

    Args:
        trained_models: Dictionary of trained model instances
        X_test: Test features

    Returns:
        Dictionary mapping model names to predictions
    """
    predictions = {}

    logger.info("Generating predictions...")
    for name, model in trained_models.items():
        predictions[name] = model.predict(X_test)
        logger.info(f"  {name.upper()} predictions generated")

    return predictions


def save_submissions(
    predictions: dict,
    sample_submission: pd.DataFrame,
    output_dir: str = "../submissions",
) -> None:
    """
    Save prediction submissions to CSV files.

    Args:
        predictions: Dictionary mapping model names to predictions
        sample_submission: Sample submission DataFrame with PassengerId
        output_dir: Directory to save submission files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving submissions to {output_dir}...")
    for name, pred in predictions.items():
        submission = pd.DataFrame(
            {"PassengerId": sample_submission["PassengerId"], "Transported": pred}
        )
        output_file = output_path / f"output_{name}.csv"
        submission.to_csv(output_file, index=False)
        logger.info(f"  {name.upper()} submission saved to {output_file}")
