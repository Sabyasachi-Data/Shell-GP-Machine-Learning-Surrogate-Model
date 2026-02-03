# utils/data_loader.py
"""
Data Loading and Preprocessing Functions

This module handles:
- Loading data from CSV
- Stratified sampling
- Train/CV/Test splitting
- Feature/target extraction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath, sample_fraction=0.5, stratify_column='Strat_cat', random_state=42):
    """
    Load data from CSV file with stratified sampling
    
    Args:
        filepath (str): Path to CSV file
        sample_fraction (float): Fraction of data to sample per strata
        stratify_column (str): Column name for stratification
        random_state (int): Random seed
    
    Returns:
        pd.DataFrame: Sampled dataframe
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")
    
    # Stratified sampling
    sample_df = df.groupby(stratify_column).sample(
        frac=sample_fraction, 
        random_state=random_state
    )
    print(f"  Sampled shape: {sample_df.shape}")
    print(f"  Stratification counts:")
    print(sample_df[stratify_column].value_counts())
    
    return sample_df


def create_train_cv_test_split(df, stratify_column='Strat_cat', 
                                train_size=0.5, cv_test_split=0.5, 
                                random_state=42):
    """
    Create stratified train/CV/test splits
    
    Args:
        df (pd.DataFrame): Input dataframe
        stratify_column (str): Column for stratification
        train_size (float): Fraction for training set
        cv_test_split (float): How to split remainder into CV and test
        random_state (int): Random seed
    
    Returns:
        tuple: (train_set, cv_set, test_set) as DataFrames
    """
    print("\nCreating train/CV/test splits...")
    
    # First split: train vs (CV + test)
    train_set, hold_set = train_test_split(
        df, 
        test_size=(1 - train_size),
        shuffle=True,
        stratify=df[stratify_column],
        random_state=random_state
    )
    
    # Second split: CV vs test
    cv_set, test_set = train_test_split(
        hold_set,
        test_size=cv_test_split,
        shuffle=True,
        stratify=hold_set[stratify_column],
        random_state=random_state
    )
    
    # Drop stratification column
    for dataset in (train_set, cv_set, test_set):
        dataset.drop(columns=[stratify_column], inplace=True)
    
    print(f"  Train set: {train_set.shape}")
    print(f"  CV set: {cv_set.shape}")
    print(f"  Test set: {test_set.shape}")
    
    return train_set, cv_set, test_set


def extract_features_target(train_set, cv_set, test_set, 
                            n_features=10, target_col_idx=10):
    """
    Extract features (X) and target (y) from datasets
    
    Args:
        train_set (pd.DataFrame): Training set
        cv_set (pd.DataFrame): Cross-validation set
        test_set (pd.DataFrame): Test set
        n_features (int): Number of feature columns
        target_col_idx (int): Index of target column
    
    Returns:
        tuple: (X_train, y_train, X_cv, y_cv, X_test, y_test)
    """
    print("\nExtracting features and targets...")
    
    X_train = train_set.iloc[:, :n_features].values
    y_train = train_set.iloc[:, target_col_idx].values
    
    X_cv = cv_set.iloc[:, :n_features].values
    y_cv = cv_set.iloc[:, target_col_idx].values
    
    X_test = test_set.iloc[:, :n_features].values
    y_test = test_set.iloc[:, target_col_idx].values
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_cv: {X_cv.shape}, y_cv: {y_cv.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    return X_train, y_train, X_cv, y_cv, X_test, y_test


def load_and_prepare_data(config):
    """
    Complete data loading and preparation pipeline
    
    Args:
        config (dict): Configuration dictionary with:
            - data_path
            - sample_fraction
            - stratify_column
            - train_size
            - cv_test_split
            - n_features
            - target_column_index
            - random_state
    
    Returns:
        dict: Dictionary containing all data splits
    """
    # Load data
    df = load_data(
        filepath=config['data_path'],
        sample_fraction=config['sample_fraction'],
        stratify_column=config['stratify_column'],
        random_state=config['random_state']
    )
    
    # Create splits
    train_set, cv_set, test_set = create_train_cv_test_split(
        df,
        stratify_column=config['stratify_column'],
        train_size=config['train_size'],
        cv_test_split=config['cv_test_split'],
        random_state=config['random_state']
    )
    
    # Extract features and targets
    X_train, y_train, X_cv, y_cv, X_test, y_test = extract_features_target(
        train_set, cv_set, test_set,
        n_features=config['n_features'],
        target_col_idx=config['target_column_index']
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'X_test': X_test,
        'y_test': y_test,
        'train_set': train_set,
        'cv_set': cv_set,
        'test_set': test_set
    }
