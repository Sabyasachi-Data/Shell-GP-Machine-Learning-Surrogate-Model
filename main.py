# main.py
"""
Main Script for Gaussian Process Pipeline with Sensitivity Analysis

This script orchestrates the complete workflow:
1. Load and prepare data
2. Train GP model with optimal kernel
3. Evaluate model performance
4. Perform Sobol sensitivity analysis
5. Generate visualizations and save results

Author: [Your Name]
Date: [Date]
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Import configuration
import config
from config import (
    DATA_CONFIG, TRANSFORMER_CONFIG, GP_MODEL_CONFIG, KERNEL_BOUNDS,
    DEFAULT_KERNEL, TRAINING_CONFIG, PLOT_CONFIG, SENSITIVITY_CONFIG, RANDOM_STATE
)

# Import utility modules
from utils.data_loader import load_and_prepare_data
from utils.transformers import LogTransformer, GPRegressorWithStd
from utils.kernels import get_kernel_by_name
from utils.model_training import (
    create_pipeline, train_model, evaluate_model, 
    save_model, load_model, retrain_on_combined_data
)
from utils.plotting import (
    plot_predictions_with_uncertainty, plot_parity, 
    plot_sobol_indices, plot_sobol_heatmap
)
from utils.sensitivity_analysis import (
    define_sensitivity_problem, generate_sobol_samples,
    evaluate_model_on_samples, compute_sobol_indices, save_sobol_results
)

import os


def main():
    """
    Main execution function
    """
    print("\n" + "="*80)
    print(" GAUSSIAN PROCESS REGRESSION WITH UNCERTAINTY QUANTIFICATION")
    print(" AND GLOBAL SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Ensure output directories exist
    config.ensure_directories()
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("="*80)
    
    # Combine configuration
    data_config = {**DATA_CONFIG, 'random_state': RANDOM_STATE}
    data = load_and_prepare_data(data_config)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_cv = data['X_cv']
    y_cv = data['y_cv']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # ========================================================================
    # STEP 2: CREATE AND TRAIN GP MODEL
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: GP MODEL TRAINING")
    print("="*80)
    
    # Initialize transformers
    log_transformer = LogTransformer()
    
    # Get kernel
    print(f"\nUsing kernel: {DEFAULT_KERNEL}")
    kernel = get_kernel_by_name(DEFAULT_KERNEL, KERNEL_BOUNDS)
    
    # Create pipeline
    pipeline = create_pipeline(
        kernel=kernel,
        transformer_config=TRANSFORMER_CONFIG,
        gp_config=GP_MODEL_CONFIG,
        gp_regressor_class=GPRegressorWithStd
    )
    
    # Train model on train + CV (or just train if you want to validate on CV first)
    # For production, we train on combined data
    X_combined = np.vstack([X_train, X_cv])
    y_combined = np.concatenate([y_train, y_cv])
    
    pipeline = train_model(pipeline, X_combined, y_combined, log_transformer)
    
    # ========================================================================
    # STEP 3: EVALUATE MODEL ON TEST SET
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL EVALUATION")
    print("="*80)
    
    results = evaluate_model(
        pipeline=pipeline,
        X=X_test,
        y=y_test,
        log_transformer=log_transformer,
        dataset_name="Test"
    )
    
    # ========================================================================
    # STEP 4: SAVE MODEL
    # ========================================================================
    if TRAINING_CONFIG['save_model']:
        print("\n" + "="*80)
        print("STEP 4: MODEL PERSISTENCE")
        print("="*80)
        
        model_path = os.path.join(
            TRAINING_CONFIG['models_dir'],
            TRAINING_CONFIG['model_filename']
        )
        save_model(pipeline, log_transformer, model_path)
    
    # ========================================================================
    # STEP 5: GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Prediction plot with uncertainty
    pred_plot_path = os.path.join(
        PLOT_CONFIG['figures_dir'],
        f'predictions_uncertainty.{PLOT_CONFIG["figure_format"]}'
    )
    plot_predictions_with_uncertainty(
        y_true=y_test,
        y_pred=results['y_pred'],
        y_lower=results['y_lower'],
        y_upper=results['y_upper'],
        n_samples=PLOT_CONFIG['plot_samples'],
        save_path=pred_plot_path,
        dpi=PLOT_CONFIG['dpi']
    )
    
    # Parity plot
    parity_plot_path = os.path.join(
        PLOT_CONFIG['figures_dir'],
        f'parity_plot.{PLOT_CONFIG["figure_format"]}'
    )
    plot_parity(
        y_true=y_test,
        y_pred=results['y_pred'],
        save_path=parity_plot_path,
        dpi=PLOT_CONFIG['dpi']
    )
    
    # ========================================================================
    # STEP 6: SOBOL SENSITIVITY ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: SOBOL SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Define problem
    problem = define_sensitivity_problem(
        feature_names=SENSITIVITY_CONFIG['feature_names'],
        bounds=SENSITIVITY_CONFIG['bounds']
    )
    
    # Generate Sobol samples
    param_values = generate_sobol_samples(
        problem=problem,
        n_samples=SENSITIVITY_CONFIG['n_samples'],
        calc_second_order=SENSITIVITY_CONFIG['calc_second_order']
    )
    
    # Evaluate model on samples
    Y = evaluate_model_on_samples(
        model=pipeline,
        param_values=param_values,
        log_transformer=log_transformer
    )
    
    # Compute Sobol indices
    Si = compute_sobol_indices(
        problem=problem,
        Y=Y,
        calc_second_order=SENSITIVITY_CONFIG['calc_second_order']
    )
    
    # Save Sobol results
    sobol_results_path = os.path.join(
        SENSITIVITY_CONFIG['results_dir'],
        'sensitivity_indices.json'
    )
    save_sobol_results(Si, problem, sobol_results_path)
    
    # ========================================================================
    # STEP 7: SENSITIVITY ANALYSIS VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 7: SENSITIVITY ANALYSIS VISUALIZATIONS")
    print("="*80)
    
    # Sobol indices bar plot
    sobol_plot_path = os.path.join(
        PLOT_CONFIG['figures_dir'],
        f'sobol_indices.{PLOT_CONFIG["figure_format"]}'
    )
    plot_sobol_indices(
        Si=Si,
        feature_names=SENSITIVITY_CONFIG['feature_names'],
        save_path=sobol_plot_path,
        dpi=PLOT_CONFIG['dpi']
    )
    
    # Second-order indices heatmap
    if SENSITIVITY_CONFIG['calc_second_order']:
        heatmap_path = os.path.join(
            PLOT_CONFIG['figures_dir'],
            f'sobol_heatmap.{PLOT_CONFIG["figure_format"]}'
        )
        plot_sobol_heatmap(
            Si=Si,
            feature_names=SENSITIVITY_CONFIG['feature_names'],
            save_path=heatmap_path,
            dpi=PLOT_CONFIG['dpi']
        )
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nOutputs saved to:")
    print(f"  Models: {TRAINING_CONFIG['models_dir']}")
    print(f"  Figures: {PLOT_CONFIG['figures_dir']}")
    print(f"  Results: {SENSITIVITY_CONFIG['results_dir']}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
