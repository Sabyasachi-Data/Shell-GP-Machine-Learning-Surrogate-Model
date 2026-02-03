# utils/kernels.py
"""
Kernel Definitions for Gaussian Process Regression

This module provides various kernel configurations for GP models.
Includes standard kernels (RBF, Matern, RationalQuadratic) and combined kernels.
"""

from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, ConstantKernel as C, RationalQuadratic
)


def get_kernel_options(kernel_bounds):
    """
    Define various kernel options for GP regression
    
    Args:
        kernel_bounds (dict): Dictionary containing bounds for:
            - 'length_scale': Tuple (min, max) for length scale
            - 'constant_value': Tuple (min, max) for constant value
            - 'noise_level': Tuple (min, max) for noise level
    
    Returns:
        dict: Dictionary of kernel name -> kernel object
    
    Available kernels:
        - rbf_plus_matern: Additive combination (best for structural mechanics)
        - rbf_times_matern: Multiplicative combination (complex correlations)
        - matern_1.5: Matern kernel with nu=1.5
        - matern_2.5: Matern kernel with nu=2.5
        - rbf: Radial Basis Function kernel
        - rational_quadratic: RationalQuadratic kernel
    """
    
    ls_bounds = kernel_bounds['length_scale']
    c_bounds = kernel_bounds['constant_value']
    noise_bounds = kernel_bounds['noise_level']
    
    kernels = {
       
        # 'matern_1.5': C(1.0, c_bounds) * Matern(
        #     length_scale=1.0, length_scale_bounds=ls_bounds, nu=1.5
        # ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=noise_bounds),
        
        'matern_2.5': C(1.0, c_bounds) * Matern(
            length_scale=1.0, length_scale_bounds=ls_bounds, nu=2.5
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=noise_bounds),
        
    #     'rbf': C(1.0, c_bounds) * RBF(
    #         length_scale=1.0, length_scale_bounds=ls_bounds
    #     ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=noise_bounds),
        
    #     'rational_quadratic': C(1.0, c_bounds) * RationalQuadratic(
    #         length_scale=1.0, alpha=1.0,
    #         length_scale_bounds=ls_bounds, alpha_bounds=ls_bounds
    #     ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=noise_bounds),
        
    #     # Simple RBF without noise kernel
    #     'rbf_simple': C(1.0, c_bounds) * RBF(
    #         length_scale=1.0, length_scale_bounds=ls_bounds
    #     ),
        }
    
    return kernels


def get_kernel_by_name(kernel_name, kernel_bounds):
    """
    Get a specific kernel by name
    
    Args:
        kernel_name (str): Name of kernel to retrieve
        kernel_bounds (dict): Kernel hyperparameter bounds
    
    Returns:
        Kernel object
    
    Raises:
        ValueError: If kernel name not found
    """
    kernels = get_kernel_options(kernel_bounds)
    
    if kernel_name not in kernels:
        raise ValueError(
            f"Kernel '{kernel_name}' not found. "
            f"Available kernels: {list(kernels.keys())}"
        )
    
    return kernels[kernel_name]


def print_kernel_info(kernel_name, kernel):
    """
    Print information about a kernel
    
    Args:
        kernel_name (str): Name of the kernel
        kernel (Kernel): Kernel object
    """
    print(f"\nKernel: {kernel_name}")
    print(f"  Formula: {kernel}")
    print(f"  Hyperparameters: {kernel.n_dims}")
    print(f"  Bounds: {kernel.bounds}")
