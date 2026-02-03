# utils/model_training.py
"""
GP Model Training and Evaluation Functions

This module handles:
- Pipeline creation
- Model training
- Model evaluation
- Model persistence
"""

import numpy as np
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def create_pipeline(kernel, transformer_config, gp_config, gp_regressor_class):
    """
    Create scikit-learn pipeline with QuantileTransformer and GP regressor
    
    Args:
        kernel: Kernel object for GP
        transformer_config (dict): Configuration for QuantileTransformer
        gp_config (dict): Configuration for GP model
        gp_regressor_class: GPRegressorWithStd class
    
    Returns:
        Pipeline: Configured pipeline
    """
    return Pipeline([
        ('quantile_transform', QuantileTransformer(
            n_quantiles=transformer_config['n_quantiles'],
            output_distribution=transformer_config['output_distribution'],
            random_state=gp_config['random_state']
        )),
        ('gp_regressor', gp_regressor_class(
            kernel=kernel,
            alpha=gp_config['alpha'],
            n_restarts_optimizer=gp_config['n_restarts_optimizer'],
            normalize_y=gp_config['normalize_y'],
            random_state=gp_config['random_state']
        ))
    ])


def train_model(pipeline, X_train, y_train, log_transformer):
    """
    Train GP model on training data
    
    Args:
        pipeline: sklearn Pipeline
        X_train: Training features
        y_train: Training targets
        log_transformer: LogTransformer instance
    
    Returns:
        Fitted pipeline
    """
    print("\nTraining GP model...")
    
    # Transform target
    y_train_scaled = log_transformer.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # Fit pipeline
    pipeline.fit(X_train, y_train_scaled)
    
    print("✓ Model training complete!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    
    return pipeline


def evaluate_model(pipeline, X, y, log_transformer, dataset_name="Test"):
    """
    Evaluate model performance on a dataset
    
    Args:
        pipeline: Fitted pipeline
        X: Features
        y: True target values
        log_transformer: LogTransformer instance
        dataset_name: Name of dataset for printing
    
    Returns:
        dict: Dictionary containing predictions and metrics
    """
    print(f"\nEvaluating on {dataset_name} set...")
    
    # Get GP model from pipeline
    gp_model = pipeline.named_steps['gp_regressor']
    X_transformed = pipeline.named_steps['quantile_transform'].transform(X)
    
    # Predictions with uncertainty
    y_pred_scaled = gp_model.predict(X_transformed, return_std=True)
    y_std_scaled = gp_model.y_std_
    
    # Transform target
    y_scaled = log_transformer.transform(y.reshape(-1, 1)).flatten()
    
    # Metrics in scaled space
    metrics_scaled = {
        'R2': r2_score(y_scaled, y_pred_scaled),
        'MAE': mean_absolute_error(y_scaled, y_pred_scaled),
        'RMSE': root_mean_squared_error(y_scaled, y_pred_scaled)
    }
    
    # Transform back to original space
    y_pred = log_transformer.inverse_transform(y_pred_scaled)
    y_lower = log_transformer.inverse_transform(y_pred_scaled - 2 * y_std_scaled)
    y_upper = log_transformer.inverse_transform(y_pred_scaled + 2 * y_std_scaled)
    
    # Metrics in original space
    metrics_original = {
        'R2': r2_score(y, y_pred),
        'MAE': mean_absolute_error(y, y_pred),
        'RMSE': root_mean_squared_error(y, y_pred)
    }
    
    # Print metrics
    print(f"\nMetrics (Scaled Space - Log-transformed):")
    for metric, value in metrics_scaled.items():
        print(f"  {metric:5s}: {value:.6f}")
    
    print(f"\nMetrics (Original Space):")
    for metric, value in metrics_original.items():
        print(f"  {metric:5s}: {value:.6f}")
    
    return {
        'y_pred': y_pred,
        'y_pred_scaled': y_pred_scaled,
        'y_std_scaled': y_std_scaled,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'metrics_scaled': metrics_scaled,
        'metrics_original': metrics_original
    }


def save_model(pipeline, log_transformer, filepath):
    """
    Save trained model and transformer
    
    Args:
        pipeline: Fitted pipeline
        log_transformer: LogTransformer instance
        filepath: Path to save model
    """
    print(f"\nSaving model to: {filepath}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save artifacts
    artifacts = {
        'pipeline': pipeline,
        'qt_y': log_transformer
    }
    
    joblib.dump(artifacts, filepath)
    print("✓ Model saved successfully!")


def load_model(filepath):
    """
    Load saved model and transformer
    
    Args:
        filepath: Path to saved model
    
    Returns:
        dict: Dictionary with 'pipeline' and 'qt_y'
    """
    print(f"\nLoading model from: {filepath}")
    artifacts = joblib.load(filepath)
    print("✓ Model loaded successfully!")
    return artifacts


def retrain_on_combined_data(pipeline, X_train, y_train, X_cv, y_cv, log_transformer):
    """
    Retrain model on combined train+CV data for final model
    
    Args:
        pipeline: Pipeline to retrain
        X_train: Training features
        y_train: Training targets
        X_cv: CV features
        y_cv: CV targets
        log_transformer: LogTransformer instance
    
    Returns:
        Fitted pipeline
    """
    print("\n" + "="*70)
    print("Retraining on combined train+CV set...")
    print("="*70)
    
    # Combine data
    X_combined = np.vstack([X_train, X_cv])
    y_combined_scaled = np.concatenate([
        log_transformer.transform(y_train.reshape(-1, 1)).flatten(),
        log_transformer.transform(y_cv.reshape(-1, 1)).flatten()
    ])
    
    # Retrain
    pipeline.fit(X_combined, y_combined_scaled)
    
    print("✓ Final model trained!")
    print(f"  Combined training samples: {len(X_combined)}")
    print(f"  Features: {X_combined.shape[1]}")
    print("="*70)
    
    return pipeline
