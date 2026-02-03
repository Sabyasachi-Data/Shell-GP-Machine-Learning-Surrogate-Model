# utils/__init__.py
"""
Utility modules for Gaussian Process Pipeline

This package contains modular utility functions for:
- Data loading and preprocessing
- Custom transformers
- GP kernel definitions
- Model training and evaluation
- Visualization
- Sensitivity analysis
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import key functions for convenient access
from .data_loader import load_and_prepare_data
from .transformers import LogTransformer, GPRegressorWithStd
from .kernels import get_kernel_options, get_kernel_by_name
from .model_training import create_pipeline, train_model, evaluate_model, save_model, load_model
from .plotting import plot_predictions_with_uncertainty, plot_parity, plot_sobol_indices
from .sensitivity_analysis import (
    define_sensitivity_problem, 
    generate_sobol_samples,
    compute_sobol_indices
)

__all__ = [
    # Data
    'load_and_prepare_data',
    
    # Transformers
    'LogTransformer',
    'GPRegressorWithStd',
    
    # Kernels
    'get_kernel_options',
    'get_kernel_by_name',
    
    # Training
    'create_pipeline',
    'train_model',
    'evaluate_model',
    'save_model',
    'load_model',
    
    # Plotting
    'plot_predictions_with_uncertainty',
    'plot_parity',
    'plot_sobol_indices',
    
    # Sensitivity Analysis
    'define_sensitivity_problem',
    'generate_sobol_samples',
    'compute_sobol_indices',
]
