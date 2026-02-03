# utils/plotting.py
"""
Visualization Functions for GP Model Results

This module provides functions for:
- Prediction plots with uncertainty bounds
- Residual plots
- Sobol sensitivity analysis visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_predictions_with_uncertainty(y_true, y_pred, y_lower, y_upper, 
                                      n_samples=50, save_path=None, dpi=300):
    """
    Plot predictions with uncertainty bounds
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_lower: Lower bound (prediction - 2σ)
        y_upper: Upper bound (prediction + 2σ)
        n_samples: Number of samples to plot
        save_path: Path to save figure (None = don't save)
        dpi: Resolution for saving
    
    Returns:
        matplotlib Figure object
    """
    n_samples = min(n_samples, len(y_true))
    indices = np.arange(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Predictions with uncertainty bounds
    ax1.plot(indices, y_true[:n_samples], 'ko-', 
             label='True Values', markersize=6, linewidth=1.5, alpha=0.7)
    ax1.plot(indices, y_pred[:n_samples], 'bs-', 
             label='Predictions', markersize=5, linewidth=1.5, alpha=0.7)
    
    ax1.fill_between(indices, 
                     y_lower[:n_samples], 
                     y_upper[:n_samples],
                     alpha=0.3, color='blue', 
                     label='95% Confidence Interval (±2σ)')
    
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Target Value (Omega)', fontsize=12)
    ax1.set_title(f'GP Predictions with Uncertainty Bounds (First {n_samples} Samples)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals with uncertainty
    residuals = y_true[:n_samples] - y_pred[:n_samples]
    std_width = y_upper[:n_samples] - y_pred[:n_samples]
    
    ax2.scatter(indices, residuals, c='darkred', s=50, alpha=0.6, 
                label='Residuals')
    ax2.fill_between(indices, -std_width, std_width,
                     alpha=0.2, color='gray', 
                     label='±2σ Uncertainty')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    
    ax2.set_xlabel('Sample Index', fontsize=12)
    ax2.set_ylabel('Residual (True - Predicted)', fontsize=12)
    ax2.set_title('Residuals with Uncertainty Bounds', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_parity(y_true, y_pred, save_path=None, dpi=300):
    """
    Plot parity plot (predicted vs actual)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save figure
        dpi: Resolution for saving
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Parity Plot: Predicted vs Actual', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_sobol_indices(Si, feature_names, save_path=None, dpi=300):
    """
    Plot Sobol sensitivity indices
    
    Args:
        Si: Sobol analysis results from SALib
        feature_names: List of feature names
        save_path: Path to save figure
        dpi: Resolution for saving
    
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First-order indices
    S1 = Si['S1']
    S1_conf = Si['S1_conf']
    
    y_pos = np.arange(len(feature_names))
    ax1.barh(y_pos, S1, xerr=S1_conf, align='center', alpha=0.7, 
             color='steelblue', ecolor='black', capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('First-order Sobol Index (S1)', fontsize=12)
    ax1.set_title('First-order Sensitivity Indices', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Total-order indices
    ST = Si['ST']
    ST_conf = Si['ST_conf']
    
    ax2.barh(y_pos, ST, xerr=ST_conf, align='center', alpha=0.7, 
             color='coral', ecolor='black', capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Total-order Sobol Index (ST)', fontsize=12)
    ax2.set_title('Total-order Sensitivity Indices', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def plot_sobol_heatmap(Si, feature_names, save_path=None, dpi=300):
    """
    Plot heatmap of second-order Sobol indices
    
    Args:
        Si: Sobol analysis results from SALib
        feature_names: List of feature names
        save_path: Path to save figure
        dpi: Resolution for saving
    
    Returns:
        matplotlib Figure object
    """
    if 'S2' not in Si:
        print("Warning: Second-order indices not available")
        return None
    
    S2 = Si['S2']
    n_params = len(feature_names)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    im = ax.imshow(S2, cmap='YlOrRd', aspect='auto', vmin=0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Second-order Sobol Index (S2)', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(n_params):
        for j in range(n_params):
            if i != j:
                text = ax.text(j, i, f'{S2[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Second-order Sensitivity Indices Heatmap', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig
