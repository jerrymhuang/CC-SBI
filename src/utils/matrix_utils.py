import numpy as np
from scipy.linalg import svd, sqrtm

def orthonormalize_ts(
    t1s: np.ndarray,
    t2s: np.ndarray,
    lowdin: bool = False
) -> dict[str, np.ndarray]:
    """
    Orthogonalize t1 and t2 amplitude arrays using either SVD or Löwdin orthogonalization.

    Parameters
    ----------
    t1s : np.ndarray
        Array of t1 amplitudes, shape (n_samples, n_virt, n_occ).
    t2s : np.ndarray
        Array of t2 amplitudes, shape (n_samples, n_virt, n_virt, n_occ, n_occ).
    lowdin : bool, optional
        If True, use Löwdin orthogonalization; if False, use SVD (default: False).

    Returns
    -------
    tuple
        - t1s_new : np.ndarray
            Orthogonalized t1 amplitudes, shape (n_samples, n_virt, n_occ).
        - t2s_new : np.ndarray
            Orthogonalized t2 amplitudes, shape (n_samples, n_virt, n_virt, n_occ, n_occ).
        - coefs : np.ndarray
            Coefficient matrix to express original amplitudes in terms of new ones, shape (n_samples, n_samples).
    """
    if t1s.shape[0] != t2s.shape[0]:
        raise ValueError("t1s and t2s must have the same number of samples")

    n_samples, n_virt, n_occ = t1s.shape
    t2_shape = t2s.shape[1:]
    if t2_shape != (n_virt, n_virt, n_occ, n_occ):
        raise ValueError("t2s shape must match (n_samples, n_virt, n_virt, n_occ, n_occ)")

    # Flatten and concatenate t1 and t2 for each sample
    t_tot = np.concatenate(
        (t1s.reshape(n_samples, -1), t2s.reshape(n_samples, -1)),
        axis=1
    )  # Shape: (n_samples, n_virt*n_occ + n_virt*n_virt*n_occ*n_occ)

    # Perform orthogonalization
    t_tot_old = t_tot.copy()
    if lowdin:
        # Löwdin orthogonalization
        overlap = t_tot @ t_tot.T
        overlap_sqrtm = np.real(sqrtm(overlap))
        t_tot = np.linalg.inv(overlap_sqrtm) @ t_tot
    else:
        # SVD orthogonalization
        U, s, Vt = svd(t_tot, full_matrices=False)
        t_tot = U @ Vt

    t_tot = t_tot.astype(np.float32)  # Ensure consistent data type

    # Compute coefficients
    coefficients = t_tot_old @ t_tot.T  # Shape: (n_samples, n_samples)

    # Reshape back to t1 and t2
    t1_new = t_tot[:, :n_virt * n_occ].reshape(n_samples, n_virt, n_occ)
    t2_new = t_tot[:, n_virt * n_occ:].reshape(n_samples, n_virt, n_virt, n_occ, n_occ)

    return {
        "t1": t1_new,
        "t2": t2_new,
        "coefficients": coefficients
    }
