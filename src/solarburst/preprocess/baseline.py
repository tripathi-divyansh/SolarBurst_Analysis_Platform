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

    # For large time series (>6000 points, e.g. 1 full day of 1-second data),
    # compute baseline on regularized sub-grid and interpolate back.
    # Solar quiet baseline varies slowly (minutes/hours), so this preserves full physical accuracy
    # while reducing matrix solve time from minutes to milliseconds.
    if n > 6000:
        stride = int(np.ceil(n / 4000))
        sub_idx = np.arange(0, n, stride)
        y_sub = y_filled[sub_idx]
        mask_sub = mask_valid[sub_idx]
        lam_sub = max(lam / (stride ** 4), 1.0)
        base_sub = asymmetric_least_squares(y_sub, mask_sub, lam=lam_sub, p=p, n_iter=n_iter)
        return np.interp(np.arange(n), sub_idx, base_sub)

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
    from scipy.ndimage import binary_dilation

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

        # 3. Vectorized mask expansion (include tails)
        expanded = binary_dilation(provisional, iterations=15) & mask_valid
        tail_candidates = binary_dilation(provisional, iterations=40)
        expanded |= (z > 1.0) & tail_candidates & mask_valid

        event_mask = expanded

        # 4. Refit with unmasked quiet points
        quiet_mask = mask_valid & (~event_mask)
        if np.sum(quiet_mask) < 20:
            # Insufficient quiet points, keep previous baseline
            break

        y_target = y.copy()
        if not np.all(quiet_mask):
            valid_q = np.where(quiet_mask)[0]
            if len(valid_q) > 0:
                y_target[~quiet_mask] = np.interp(
                    np.where(~quiet_mask)[0],
                    valid_q,
                    y[valid_q]
                )

        if n > 6000:
            stride = int(np.ceil(n / 4000))
            sub_idx = np.arange(0, n, stride)
            y_sub = y_target[sub_idx]
            q_sub = quiet_mask[sub_idx]
            lam_sub = max(lam / (stride ** 4), 1.0)
            
            n_sub = len(y_sub)
            w_sub = np.where(q_sub, 1.0, 0.001)
            D_sub = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_sub - 2, n_sub), format="csc", dtype=float)
            W_sub = sparse.diags(w_sub, 0, shape=(n_sub, n_sub), format="csc")
            A_sub = W_sub + lam_sub * (D_sub.T @ D_sub)
            b_sub = spsolve(A_sub, w_sub * y_sub)
            b = np.interp(np.arange(n), sub_idx, b_sub)
        else:
            w = np.where(quiet_mask, 1.0, 0.001)
            D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc", dtype=float)
            D_T_D = D.T @ D
            W = sparse.diags(w, 0, shape=(n, n), format="csc")
            A = W + lam * D_T_D
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
