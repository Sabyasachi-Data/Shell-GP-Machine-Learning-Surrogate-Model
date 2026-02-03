# utils/sensitivity_analysis.py
"""
Sensitivity Analysis using SALib (Sobol Method)

This module defines the problem for sensitivity analysis and provides
functions to perform Sobol global sensitivity analysis on the GP model.
"""

import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol


def define_sensitivity_problem(feature_names, bounds):
    """
    Define the problem for SALib sensitivity analysis
    
    Args:
        feature_names (list): Names of input parameters
        bounds (list): List of [min, max] bounds for each parameter
    
    Returns:
        dict: Problem definition for SALib
    """
    problem = {
        'num_vars': len(feature_names),
        'names': feature_names,
        'bounds': bounds
    }
    return problem


def generate_sobol_samples(problem, n_samples=1024, calc_second_order=True):
    """
    Generate samples using Saltelli sampling scheme
    
    Args:
        problem (dict): Problem definition from define_sensitivity_problem
        n_samples (int): Number of samples (actual samples will be N*(2D+2))
        calc_second_order (bool): Whether to calculate second-order indices
    
    Returns:
        np.ndarray: Sample matrix of shape (N*(2D+2), D)
    """
    print("\nGenerating Sobol samples using Saltelli scheme...")
    print(f"  Base samples: {n_samples}")
    print(f"  Parameters: {problem['num_vars']}")
    print(f"  Calculate second-order: {calc_second_order}")
    
    param_values = saltelli.sample(
        problem, 
        n_samples, 
        calc_second_order=calc_second_order
    )
    
    print(f"  Total samples generated: {param_values.shape[0]}")
    print(f"  Sample shape: {param_values.shape}")
    
    return param_values


def evaluate_model_on_samples(model, param_values, log_transformer):
    """
    Evaluate GP model on Sobol samples
    
    Args:
        model: Fitted pipeline with GP model
        param_values: Sobol sample matrix
        log_transformer: LogTransformer for inverse transform
    
    Returns:
        np.ndarray: Model predictions for all samples
    """
    print("\nEvaluating model on Sobol samples...")
    
    # Get GP model and transformer from pipeline
    gp_model = model.named_steps['gp_regressor']
    quantile_transform = model.named_steps['quantile_transform']
    
    # Transform features
    X_transformed = quantile_transform.transform(param_values)
    
    # Predict (return_std=False for point predictions)
    y_pred_scaled = gp_model.predict(X_transformed, return_std=False)
    
    # Transform back to original space
    y_pred = log_transformer.inverse_transform(y_pred_scaled)
    
    print(f"  Predictions computed: {len(y_pred)}")
    print(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    return y_pred


def compute_sobol_indices(problem, Y, calc_second_order=True, num_resamples=100):
    """
    Compute Sobol sensitivity indices
    
    Args:
        problem (dict): Problem definition
        Y (np.ndarray): Model outputs corresponding to Sobol samples
        calc_second_order (bool): Whether to compute second-order indices
        num_resamples (int): Number of bootstrap resamples for confidence intervals
    
    Returns:
        dict: Sobol indices (S1, ST, S2 if calc_second_order=True)
    """
    print("\nComputing Sobol sensitivity indices...")
    print(f"  Number of outputs: {len(Y)}")
    print(f"  Second-order indices: {calc_second_order}")
    
    Si = sobol.analyze(
        problem, 
        Y, 
        calc_second_order=calc_second_order,
        num_resamples=num_resamples,
        print_to_console=False
    )
    
    print("\n" + "="*70)
    print("SOBOL SENSITIVITY ANALYSIS RESULTS")
    print("="*70)
    
    # Print first-order indices
    print("\nFirst-order Sobol Indices (S1):")
    print("-" * 70)
    print(f"{'Parameter':<25} {'S1':>10} {'S1_conf':>10}")
    print("-" * 70)
    for name, s1, conf in zip(problem['names'], Si['S1'], Si['S1_conf']):
        print(f"{name:<25} {s1:>10.4f} {conf:>10.4f}")
    
    # Print total-order indices
    print("\nTotal-order Sobol Indices (ST):")
    print("-" * 70)
    print(f"{'Parameter':<25} {'ST':>10} {'ST_conf':>10}")
    print("-" * 70)
    for name, st, conf in zip(problem['names'], Si['ST'], Si['ST_conf']):
        print(f"{name:<25} {st:>10.4f} {conf:>10.4f}")
    
    # Print second-order indices if available
    if calc_second_order and 'S2' in Si:
        print("\nSecond-order Sobol Indices (S2) - Top interactions:")
        print("-" * 70)
        
        # Find top 10 interactions
        S2 = Si['S2']
        n_params = len(problem['names'])
        interactions = []
        
        for i in range(n_params):
            for j in range(i + 1, n_params):
                interactions.append((i, j, S2[i, j]))
        
        # Sort by magnitude
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"{'Parameter 1':<20} {'Parameter 2':<20} {'S2':>10}")
        print("-" * 70)
        for i, j, s2_val in interactions[:10]:
            print(f"{problem['names'][i]:<20} {problem['names'][j]:<20} {s2_val:>10.4f}")
    
    print("="*70)
    
    return Si


def save_sobol_results(Si, problem, filepath):
    """
    Save Sobol indices to file
    
    Args:
        Si (dict): Sobol indices
        problem (dict): Problem definition
        filepath (str): Path to save results
    """
    import os
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'parameters': problem['names'],
        'bounds': problem['bounds'],
        'first_order': {
            'S1': Si['S1'].tolist(),
            'S1_conf': Si['S1_conf'].tolist()
        },
        'total_order': {
            'ST': Si['ST'].tolist(),
            'ST_conf': Si['ST_conf'].tolist()
        }
    }
    
    if 'S2' in Si:
        results['second_order'] = {
            'S2': Si['S2'].tolist(),
            'S2_conf': Si['S2_conf'].tolist()
        }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Sobol results saved to: {filepath}")
