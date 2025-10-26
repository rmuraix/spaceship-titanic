"""
Data preprocessing module for Spaceship Titanic project.
Handles missing value imputation, feature engineering, and encoding.
"""
import pandas as pd


def remove_unnecessary_columns(df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove columns that don't contribute to predictions.
    
    Args:
        df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    columns_to_drop = ["Name", "PassengerId"]
    df = df.drop(columns_to_drop, axis=1)
    test_df = test_df.drop(columns_to_drop, axis=1)
    return df, test_df


def fill_missing_values(df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values in both train and test datasets.
    
    Args:
        df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    # Define columns and their fill strategies
    mode_columns = [
        "CryoSleep", "Cabin", "Destination", "HomePlanet", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
    ]
    median_columns = ["Age"]
    
    # Fill mode columns
    for col in mode_columns:
        if col in df.columns:
            fill_value = df[col].mode()[0]
            df[col] = df[col].fillna(fill_value).infer_objects(copy=False)
        if col in test_df.columns:
            fill_value = test_df[col].mode()[0]
            test_df[col] = test_df[col].fillna(fill_value).infer_objects(copy=False)
    
    # Fill median columns
    for col in median_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(test_df[col].median())
    
    return df, test_df


def engineer_cabin_features(df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split cabin column into deck, num, and side features.
    
    Args:
        df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    # Process training data
    df["deck"] = df["Cabin"].str.split("/", expand=True)[0]
    df["num"] = df["Cabin"].str.split("/", expand=True)[1].astype("float")
    df["side"] = df["Cabin"].str.split("/", expand=True)[2]
    df = df.drop("Cabin", axis=1)
    
    # Process test data
    test_df["deck"] = test_df["Cabin"].str.split("/", expand=True)[0]
    test_df["num"] = test_df["Cabin"].str.split("/", expand=True)[1].astype("float")
    test_df["side"] = test_df["Cabin"].str.split("/", expand=True)[2]
    test_df = test_df.drop("Cabin", axis=1)
    
    return df, test_df


def encode_categorical_features(df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply one-hot encoding to categorical features.
    
    Args:
        df: Training DataFrame
        test_df: Test DataFrame
        
    Returns:
        Tuple of (encoded_train_df, encoded_test_df)
    """
    categorical_columns = ["HomePlanet", "CryoSleep", "VIP", "Destination", "deck", "side"]
    
    df = pd.get_dummies(df, columns=categorical_columns, sparse=True)
    test_df = pd.get_dummies(test_df, columns=categorical_columns, sparse=True)
    
    return df, test_df


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline for train and test data.
    
    Args:
        train_df: Raw training DataFrame
        test_df: Raw test DataFrame
        
    Returns:
        Tuple of (train_features, test_features, train_target)
    """
    # Remove unnecessary columns
    train_df, test_df = remove_unnecessary_columns(train_df, test_df)
    
    # Fill missing values
    train_df, test_df = fill_missing_values(train_df, test_df)
    
    # Engineer cabin features
    train_df, test_df = engineer_cabin_features(train_df, test_df)
    
    # Encode categorical features
    train_df, test_df = encode_categorical_features(train_df, test_df)
    
    # Separate features and target
    train_target = train_df["Transported"]
    train_features = train_df.drop("Transported", axis=1)
    test_features = test_df
    
    return train_features, test_features, train_target
