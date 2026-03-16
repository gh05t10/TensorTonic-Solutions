import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.shape[0] < 2 or X.ndim != 2:
        return None
    else:
        mu = np.mean(X, axis=0)
        X_centered = X - mu
        cov =  (X_centered.T @ X_centered) / (X.shape[0]-1)
        return cov 