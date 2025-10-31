"""
Model training module for Spaceship Titanic project.
Handles model initialization, hyperparameter optimization with Optuna, and training.
"""

import logging

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Get logger for this module
logger = logging.getLogger(__name__)


def create_stratified_kfold(
    n_splits: int = 5, random_state: int = 0
) -> StratifiedKFold:
    """
    Create a StratifiedKFold cross-validator.

    Args:
        n_splits: Number of folds
        random_state: Random state for reproducibility

    Returns:
        StratifiedKFold cross-validator
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def optimize_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 100
) -> dict:
    """
    Optimize RandomForest hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "min_samples_split": trial.suggest_int("min_samples_split", 7, 15),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_features": trial.suggest_int("max_features", 3, 10),
            "random_state": 0,
        }
        model = RandomForestClassifier(**params)
        kf = create_stratified_kfold()
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        return scores["test_score"].mean()

    logger.info("Optimizing RandomForest hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best RandomForest params: {study.best_params}")
    logger.info(f"Best RandomForest score: {study.best_value:.4f}")
    return study.best_params


def optimize_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 100
) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.1, 1.0, step=0.1),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.5, 1.0, step=0.1
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": 0,
        }
        model = XGBClassifier(**params)
        kf = create_stratified_kfold()
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        return scores["test_score"].mean()

    logger.info("Optimizing XGBoost hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best XGBoost params: {study.best_params}")
    logger.info(f"Best XGBoost score: {study.best_value:.4f}")
    return study.best_params


def optimize_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 100
) -> dict:
    """
    Optimize LightGBM hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": 0,
        }
        model = LGBMClassifier(**params)
        kf = create_stratified_kfold()
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        return scores["test_score"].mean()

    logger.info("Optimizing LightGBM hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best LightGBM params: {study.best_params}")
    logger.info(f"Best LightGBM score: {study.best_value:.4f}")
    return study.best_params


def optimize_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 10
) -> dict:
    """
    Optimize Logistic Regression hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "C": trial.suggest_int("C", 1, 100),
            "random_state": 0,
        }
        model = LogisticRegression(**params)
        kf = create_stratified_kfold()
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        return scores["test_score"].mean()

    logger.info("Optimizing Logistic Regression hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best Logistic Regression params: {study.best_params}")
    logger.info(f"Best Logistic Regression score: {study.best_value:.4f}")
    return study.best_params


def optimize_svc(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 10) -> dict:
    """
    Optimize SVC hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials

    Returns:
        Dictionary of best hyperparameters
    """

    def objective(trial):
        params = {
            "C": trial.suggest_int("C", 50, 200),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "random_state": 0,
            "kernel": "rbf",
        }
        model = SVC(**params)
        kf = create_stratified_kfold()
        scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
        return scores["test_score"].mean()

    logger.info("Optimizing SVC hyperparameters...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best SVC params: {study.best_params}")
    logger.info(f"Best SVC score: {study.best_value:.4f}")
    return study.best_params


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Train all models with optimized hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Dictionary mapping model names to their best parameters
    """
    best_params = {}

    # Optimize each model
    best_params["rfc"] = optimize_random_forest(X_train, y_train, n_trials=100)
    best_params["xgb"] = optimize_xgboost(X_train, y_train, n_trials=100)
    best_params["lgb"] = optimize_lightgbm(X_train, y_train, n_trials=100)
    best_params["lr"] = optimize_logistic_regression(X_train, y_train, n_trials=10)
    best_params["svc"] = optimize_svc(X_train, y_train, n_trials=10)

    return best_params


def create_models_with_params(best_params: dict) -> dict:
    """
    Create model instances with best parameters.

    Args:
        best_params: Dictionary mapping model names to their best parameters

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "rfc": RandomForestClassifier(**best_params["rfc"]),
        "xgb": XGBClassifier(**best_params["xgb"]),
        "lgb": LGBMClassifier(**best_params["lgb"]),
        "lr": LogisticRegression(**best_params["lr"]),
        "svc": SVC(**best_params["svc"]),
    }
    return models
