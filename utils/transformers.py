# utils/transformers.py
"""
Custom Transformers for Gaussian Process Pipeline

This module contains custom transformer classes:
1. LogTransformer: Applies log transformation to target variable
2. GPRegressorWithStd: Wrapper for GaussianProcessRegressor that stores std predictions
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log transformation for target variable using log1p (log(1+x))
    
    This transformer is useful for target variables with skewed distributions.
    Uses log1p to avoid issues with zero values.
    
    Attributes:
        fitted_ (bool): Whether the transformer has been fitted
    
    Methods:
        fit(y, **fit_params): Fit the transformer
        transform(y): Apply log transformation
        inverse_transform(y): Reverse log transformation
    """
    
    def __init__(self):
        self.fitted_ = False
    
    def fit(self, y, **fit_params):
        """
        Fit the transformer (no actual fitting required for log transform)
        
        Args:
            y: Target variable
            **fit_params: Additional fit parameters
        
        Returns:
            self
        """
        self.fitted_ = True
        return self
    
    def transform(self, y):
        """
        Apply log1p transformation
        
        Args:
            y: Target variable to transform
        
        Returns:
            Transformed target variable
        
        Raises:
            RuntimeError: If transformer hasn't been fitted
        """
        if not self.fitted_:
            raise RuntimeError("Transformer must be fitted before transform")
        return np.log1p(y)
    
    def inverse_transform(self, y):
        """
        Reverse log1p transformation using expm1
        
        Args:
            y: Transformed target variable
        
        Returns:
            Original scale target variable
        """
        return np.expm1(y)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for output
        
        Args:
            input_features: Input feature names
        
        Returns:
            Output feature names with '_log' suffix
        """
        if input_features is None:
            return None
        return [f"{name}_log" for name in input_features]


class GPRegressorWithStd(BaseEstimator, RegressorMixin):
    """
    Wrapper for GaussianProcessRegressor that stores standard deviation predictions
    
    This wrapper extends sklearn's GaussianProcessRegressor to store std predictions
    as an attribute, making it easier to retrieve uncertainty estimates.
    
    Attributes:
        kernel: GP kernel function
        alpha: Noise level in training data
        n_restarts_optimizer: Number of restarts for hyperparameter optimization
        normalize_y: Whether to normalize target values
        random_state: Random seed
        gp_: Fitted GaussianProcessRegressor instance
        y_std_: Standard deviation of predictions
    
    Methods:
        fit(X, y): Fit the GP model
        predict(X, return_std): Make predictions with optional std
        score(X, y): Return R² score
    """
    
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=10,
                 normalize_y=False, random_state=None):
        """
        Initialize GP regressor wrapper
        
        Args:
            kernel: GP kernel (default: None, uses default RBF)
            alpha: Noise level in training data
            n_restarts_optimizer: Number of restarts for optimization
            normalize_y: Whether to normalize target values
            random_state: Random seed
        """
        self.kernel = kernel
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.gp_ = None
        self.y_std_ = None
    
    def fit(self, X, y):
        """
        Fit the Gaussian Process model
        
        Args:
            X: Training features
            y: Training target values
        
        Returns:
            self
        """
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state
        )
        self.gp_.fit(X, y)
        return self
    
    def predict(self, X, return_std=False):
        """
        Make predictions with optional standard deviation
        
        Args:
            X: Features for prediction
            return_std: Whether to return standard deviation
        
        Returns:
            Predictions (and std if return_std=True)
        """
        if return_std:
            y_pred, y_std = self.gp_.predict(X, return_std=True)
            self.y_std_ = y_std
            return y_pred
        else:
            return self.gp_.predict(X, return_std=False)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'kernel': self.kernel,
            'alpha': self.alpha,
            'n_restarts_optimizer': self.n_restarts_optimizer,
            'normalize_y': self.normalize_y,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def score(self, X, y):
        """
        Return R² score of predictions
        
        Args:
            X: Features
            y: True target values
        
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
