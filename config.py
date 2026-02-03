# config.py
"""
Configuration parameters for Gaussian Process Pipeline with Sensitivity Analysis

This module contains all configuration parameters for:
- Data preprocessing
- GP model hyperparameters
- Training/validation splits
- Visualization settings
- Sensitivity analysis parameters
"""

import os

# ============================================================================
# RANDOM SEED
# ============================================================================
RANDOM_STATE = 42

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_CONFIG = {
    'data_path': 'data/sph.csv',
    'stratify_column': 'Strat_cat',
    'target_column_index': 10,  # Column index for target variable (Omega)
    'n_features': 10,  # Number of input features
    'sample_fraction': 0.5,  # Fraction of data to sample per strategy
    'train_size': 0.5,  # Train set size (rest splits into CV and Test)
    'cv_test_split': 0.5,  # Split of remaining data into CV and Test
}

# ============================================================================
# TRANSFORMER CONFIGURATION
# ============================================================================
TRANSFORMER_CONFIG = {
    'n_quantiles': 100,
    'output_distribution': 'normal',  # 'normal' or 'uniform'
}

# ============================================================================
# GAUSSIAN PROCESS MODEL CONFIGURATION
# ============================================================================
GP_MODEL_CONFIG = {
    'alpha': 1e-10,  # Noise level in training data
    'n_restarts_optimizer': 10,  # Number of restarts for kernel hyperparameter optimization
    'normalize_y': False,  # Whether to normalize target values
    'random_state': RANDOM_STATE,
}

# ============================================================================
# KERNEL CONFIGURATION
# ============================================================================
# Kernel bounds for hyperparameter optimization
KERNEL_BOUNDS = {
    'length_scale': (1e-2, 1e2),
    'constant_value': (1e-3, 1e3),
    'noise_level': (1e-10, 1e0),
}

# Default kernel to use: 
DEFAULT_KERNEL = 'matern_2.5'

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'save_model': False,
    'model_filename': 'gp_model_optimized.pkl',
    'models_dir': 'outputs/models',
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
PLOT_CONFIG = {
    'plot_samples': 50,  # Number of samples to plot
    'figures_dir': 'outputs/figures',
    'dpi': 300,
    'figure_format': 'png',  # 'png', 'pdf', 'svg'
    'style': 'seaborn-v0_8-darkgrid',
}

# ========================================================================
# SENSITIVITY ANALYSIS CONFIGURATION
# ========================================================================
SENSITIVITY_CONFIG = {
    # Feature names for sensitivity analysis
    'feature_names': [
        'Fiber_Angle',
        'No_of_Plies',
        'E_1/E_2',
        'Winkler',
        'Pasternak',
        'Rx',
        'Ry',
        'a_h',
        'Skew_Angle',
        'Poisson_Ratio',
    ],

    # Bounds for each parameter (min, max)
    'bounds': [
        [15.000, 90.000],        # Fiber_Angle
        [2.000, 16.000],         # No_of_Plies
        [5.000, 90.000],         # E_1/E_2
        [0.0000, 10000.0000],    # Winkler
        [0.0000, 10000.0000],    # Pasternak
        [0.1000, 1.0000],        # Rx
        [0.1000, 1.0000],        # Ry
        [5.000, 100.000],        # a_h
        [0.000, 75.000],         # Skew_Angle
        [0.20, 0.4500],          # Poisson_Ratio
    ],

    # Saltelli sampling parameters
    # N in SALib is the base sample size; total evals are N*(2D+2) when
    # second-order indices are computed (D = number of inputs). [web:19][web:21]
    'n_samples': 4096,
    'calc_second_order': True,

    # Batch size for evaluating model on Sobol samples
    'batch_size': 2000,

    # Output directory
    'results_dir': 'outputs/results',
}

# ============================================================================
# OUTPUT PATHS
# ============================================================================
def ensure_directories():
    """Create all necessary output directories if they don't exist"""
    directories = [
        TRAINING_CONFIG['models_dir'],
        PLOT_CONFIG['figures_dir'],
        SENSITIVITY_CONFIG['results_dir'],
        os.path.dirname(DATA_CONFIG['data_path']),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration parameters"""
    assert DATA_CONFIG['sample_fraction'] > 0 and DATA_CONFIG['sample_fraction'] <= 1
    assert DATA_CONFIG['train_size'] > 0 and DATA_CONFIG['train_size'] < 1
    assert GP_MODEL_CONFIG['alpha'] > 0
    assert GP_MODEL_CONFIG['n_restarts_optimizer'] > 0
    assert len(SENSITIVITY_CONFIG['feature_names']) == len(SENSITIVITY_CONFIG['bounds'])
    print("✓ Configuration validated successfully!")

if __name__ == "__main__":
    validate_config()
    ensure_directories()
    print("Configuration loaded and directories created!")
