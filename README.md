# spaceship-titanic
![License](https://img.shields.io/github/license/rmuraix/spaceship-titanic)
![issues](https://img.shields.io/github/issues/rmuraix/spaceship-titanic)
[![DeepSource](https://deepsource.io/gh/rmuraix/spaceship-titanic.svg/?label=active+issues&show_trend=true&token=FMZNa7VRDrQ21QysKY44wMmw)](https://deepsource.io/gh/rmuraix/spaceship-titanic/?ref=repository-badge)  
## About
My solution for [spaceship-titanic competition](https://www.kaggle.com/competitions/spaceship-titanic/) in Kaggle.
## Features
- Modular, maintainable code structure
- Multiple ML models: RandomForest, XGBoost, LightGBM, Logistic Regression, SVC
- Hyperparameter optimization with Optuna
- Comprehensive logging and evaluation
- Easy-to-use pipeline
## Score
![Kaggle Badge](https://img.shields.io/badge/Score:0.80804-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)  

![score](https://user-images.githubusercontent.com/35632215/187970779-96661d09-1618-44af-838a-ac9e239a39ac.png)
## Download/Usage
1. Clone the repository
2. Install [uv](https://docs.astral.sh/uv/) (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Run the pipeline:
   ```bash
   cd source
   uv run python main.py
   ```
5. Find submission files in `submissions/` directory

For more details on the code structure, see [source/README.md](source/README.md).

## Development
This project uses [uv](https://docs.astral.sh/uv/) for dependency management and [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

### Linting and Formatting
```bash
# Check code style
uv run ruff check source/

# Fix linting issues automatically
uv run ruff check --fix source/

# Format code
uv run ruff format source/
```

## Code Structure
The codebase has been refactored into modular components:
- `data_loader.py` - Data loading utilities
- `preprocessing.py` - Data preprocessing and feature engineering
- `model_trainer.py` - Model training and hyperparameter optimization
- `evaluation.py` - Model evaluation and prediction
- `main.py` - Main pipeline orchestration

## Contributing  
Please read [contributing guide](.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/rmuraix/.github/blob/main/.github/CODE_OF_CONDUCT.md).   
## License
'rmuraix/spaceship-titanic' is under [Apache License 2.0](/LICENSE).
