# Source Code Structure

This directory contains the refactored, modular implementation of the Spaceship Titanic prediction pipeline.

## Module Overview

### `data_loader.py`
Handles data loading operations:
- `load_data()`: Loads train and test CSV files
- `load_sample_submission()`: Loads the sample submission file

### `preprocessing.py`
Data preprocessing and feature engineering:
- `remove_unnecessary_columns()`: Removes Name and PassengerId columns
- `fill_missing_values()`: Handles missing value imputation
- `engineer_cabin_features()`: Extracts deck, num, and side from Cabin
- `encode_categorical_features()`: Applies one-hot encoding
- `preprocess_data()`: Complete preprocessing pipeline

### `model_trainer.py`
Model training and hyperparameter optimization using Optuna:
- `optimize_random_forest()`: Hyperparameter tuning for RandomForest
- `optimize_xgboost()`: Hyperparameter tuning for XGBoost
- `optimize_lightgbm()`: Hyperparameter tuning for LightGBM
- `optimize_logistic_regression()`: Hyperparameter tuning for Logistic Regression
- `optimize_svc()`: Hyperparameter tuning for SVC
- `train_all_models()`: Orchestrates optimization for all models
- `create_models_with_params()`: Creates model instances with optimized parameters

### `evaluation.py`
Model evaluation and prediction generation:
- `evaluate_baseline_models()`: Evaluates models with default parameters
- `evaluate_optimized_models()`: Cross-validation evaluation of optimized models
- `evaluate_voting_classifier()`: Ensemble evaluation
- `train_final_models()`: Trains final models on full dataset
- `generate_predictions()`: Generates predictions on test set
- `save_submissions()`: Saves predictions to CSV files

### `main.py`
Main entry point that orchestrates the complete pipeline:
1. Data loading
2. Preprocessing
3. Baseline model evaluation
4. Hyperparameter optimization
5. Optimized model evaluation
6. Voting classifier evaluation
7. Final model training
8. Prediction generation
9. Submission file creation

## Usage

### Running the complete pipeline:
```bash
cd source
python main.py
```

### Using individual modules:
```python
from data_loader import load_data
from preprocessing import preprocess_data

# Load data
train_df, test_df = load_data()

# Preprocess
train_features, test_features, train_target = preprocess_data(train_df, test_df)
```

## Key Improvements

1. **Modularity**: Code is separated into logical modules with clear responsibilities
2. **Reusability**: Functions can be imported and reused in other scripts
3. **Maintainability**: Changes to one component don't affect others
4. **Readability**: Clear function names and documentation
5. **Testability**: Each module can be tested independently
6. **Logging**: Proper logging instead of print statements
7. **Type hints**: Better code documentation and IDE support
8. **DRY principle**: Eliminated code repetition (e.g., filling missing values)

## Output

The pipeline generates submission files in `../submissions/`:
- `output_rfc.csv`: RandomForest predictions
- `output_xgb.csv`: XGBoost predictions
- `output_lgb.csv`: LightGBM predictions
- `output_lr.csv`: Logistic Regression predictions
- `output_svc.csv`: SVC predictions
