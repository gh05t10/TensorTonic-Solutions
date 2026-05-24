import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns: dict with "total", "recon", and "kl" loss values as floats
    """
    # Your implementation here
    mse = np.sum((x-x_recon)**2, axis=1).mean()
    kl_div = -0.5 * np.sum(1+log_var-mu**2-np.exp(log_var), axis=1).mean()
    return {"total": float(mse + kl_div), "recon": float(mse), "kl": float(kl_div)}
