"""
Main entry point for Spaceship Titanic prediction pipeline.
This script orchestrates the complete machine learning workflow:
1. Data loading
2. Preprocessing
3. Model training and hyperparameter optimization
4. Evaluation
5. Prediction and submission generation
"""

import logging

from data_loader import load_data, load_sample_submission
from evaluation import (
    evaluate_baseline_models,
    evaluate_optimized_models,
    evaluate_voting_classifier,
    generate_predictions,
    save_submissions,
    train_final_models,
)
from lightgbm import LGBMClassifier
from model_trainer import create_models_with_params, train_all_models
from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger.info("Starting Spaceship Titanic prediction pipeline")

    # Step 1: Load data
    logger.info("Loading data...")
    train_df, test_df = load_data()
    sample_submission = load_sample_submission()

    # Step 2: Preprocess data
    logger.info("Preprocessing data...")
    train_features, test_features, train_target = preprocess_data(train_df, test_df)
    logger.info(f"Training set shape: {train_features.shape}")
    logger.info(f"Test set shape: {test_features.shape}")

    # Step 3: Evaluate baseline models
    logger.info("Evaluating baseline models...")
    baseline_models = {
        "rfc": RandomForestClassifier(random_state=0),
        "xgb": XGBClassifier(random_state=0),
        "lgb": LGBMClassifier(random_state=0),
        "lr": LogisticRegression(random_state=0),
        "svc": SVC(random_state=0),
    }
    evaluate_baseline_models(train_features, train_target, baseline_models)

    # Step 4: Hyperparameter optimization
    logger.info("Starting hyperparameter optimization...")
    best_params = train_all_models(train_features, train_target)

    # Step 5: Create models with best parameters
    logger.info("Creating models with optimized parameters...")
    optimized_models = create_models_with_params(best_params)

    # Step 6: Evaluate optimized models
    logger.info("Evaluating optimized models with cross-validation...")
    evaluate_optimized_models(train_features, train_target, optimized_models)

    # Step 7: Evaluate voting classifier (optional)
    logger.info("Evaluating voting classifier ensemble...")
    evaluate_voting_classifier(train_features, train_target, optimized_models)

    # Step 8: Train final models on full dataset
    logger.info("Training final models on full dataset...")
    final_models = train_final_models(train_features, train_target, optimized_models)

    # Step 9: Generate predictions
    logger.info("Generating predictions on test set...")
    predictions = generate_predictions(final_models, test_features)

    # Step 10: Save submissions
    logger.info("Saving submission files...")
    save_submissions(predictions, sample_submission)

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
