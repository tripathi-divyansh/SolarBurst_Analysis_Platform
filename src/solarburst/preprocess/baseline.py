"""
Asymmetric Least Squares (AsLS) and iterative robust event-masked baseline estimation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from .noise import estimate_robust_noise


def asymmetric_least_squares(
    y: np.ndarray,
    mask_valid: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    n_iter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares (AsLS) baseline estimation.
    Downweights positive residuals (flares/excesses) with asymmetry parameter p.
    
    min_b sum_i w_i (y_i - b_i)^2 + lambda sum_i (Delta^2 b_i)^2
    """
    n = len(y)
    # Fill invalid/gap values temporarily with nearest valid for numerical stability
    y_filled = y.copy()
    valid_indices = np.where(mask_valid)[0]
    if len(valid_indices) == 0:
        return np.zeros(n)
    
    if len(valid_indices) < n:
        # 1D linear interpolation for gaps
        y_filled[~mask_valid] = np.interp(
            np.where(~mask_valid)[0],
            valid_indices,
            y[valid_indices]
        )

    # Second-order difference matrix D
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float)
    D_T_D = D.T @ D

    w = np.ones(n, dtype=np.float64)
    # Zero weight for completely unobserved points initially
    w[~mask_valid] = 0.01

    baseline = y_filled.copy()

    for _ in range(n_iter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        A = W + lam * D_T_D
        b = w * y_filled
        baseline = spsolve(A, b)
        # Asymmetric weights: downweight peaks above baseline
        w_new = np.where(y_filled > baseline, p, 1.0 - p)
        w_new[~mask_valid] = 0.001
        if np.max(np.abs(w_new - w)) < 1e-4:
            break
        w = w_new

    return baseline


def iterative_masked_baseline(
    y: np.ndarray,
    mask_valid: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    excess_sigma: float = 3.0,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Iterative event masking with robust baseline refitting.
    
    1. Initial AsLS baseline.
    2. Identify provisional excess regions.
    3. Expand masks to include tails.
    4. Refit baseline with quiet observations.
    5. Flag intervals with insufficient quiet support.
    """
    n = len(y)
    noise_sigma = estimate_robust_noise(y, mask_valid)
    
    # 1. Initial AsLS baseline
    b = asymmetric_least_squares(y, mask_valid, lam=lam, p=p, n_iter=8)

    event_mask = np.zeros(n, dtype=bool)

    for it in range(iterations):
        # 2. Identify provisional excesses: z > excess_sigma
        residual = y - b
        z = residual / max(noise_sigma, 1e-6)
        provisional = (z > excess_sigma) & mask_valid

        # 3. Expand masks to include tails: dilate by +- 15 bins or until z < 1.0
        expanded = provisional.copy()
        change = True
        radius = 15
        for i in np.where(provisional)[0]:
            left = max(0, i - radius)
            right = min(n, i + radius + 1)
            expanded[left:right] |= (z[left:right] > 1.0) & mask_valid[left:right]

        event_mask = expanded

        # 4. Refit with unmasked quiet points
        quiet_mask = mask_valid & (~event_mask)
        if np.sum(quiet_mask) < 20:
            # Insufficient quiet points, keep previous baseline
            break

        # Refit with higher weight on quiet points
        w = np.where(quiet_mask, 1.0, 0.001)
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float)
        D_T_D = D.T @ D
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        A = W + lam * D_T_D
        
        y_target = y.copy()
        if not np.all(quiet_mask):
            valid_q = np.where(quiet_mask)[0]
            if len(valid_q) > 0:
                y_target[~quiet_mask] = np.interp(
                    np.where(~quiet_mask)[0],
                    valid_q,
                    y[valid_q]
                )

        b = spsolve(A, w * y_target)

    # 5. Check quiet support
    quiet_support = np.sum(mask_valid & (~event_mask)) / max(np.sum(mask_valid), 1)
    insufficient_quiet_support = quiet_support < 0.20

    return {
        "baseline": b,
        "noise_sigma": noise_sigma,
        "event_mask": event_mask,
        "quiet_support_fraction": float(quiet_support),
        "insufficient_quiet_support": bool(insufficient_quiet_support)
    }
