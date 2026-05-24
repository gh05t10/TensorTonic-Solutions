import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Returns: float scalar KL divergence averaged over the batch
    """
    # Your implementation here
    var = np.exp(log_var)
    kl_elementwise = -0.5 * (1 + log_var - mu**2 - var)
    kl_div = np.mean(np.sum(kl_elementwise, axis=1))
    return float(kl_div)
