"""
Data loading module for Spaceship Titanic project.
Handles loading train and test data from CSV files.
"""
import pandas as pd
from pathlib import Path


def load_data(data_dir: str = "../data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (train_df, test_df)
    """
    data_path = Path(data_dir)
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    return train_df, test_df


def load_sample_submission(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load sample submission file.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Sample submission DataFrame
    """
    data_path = Path(data_dir)
    return pd.read_csv(data_path / "sample_submission.csv")
