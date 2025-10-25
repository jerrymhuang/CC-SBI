import numpy as np
import GPy

from paramz.transformations import Logistic

from .matrix_utils import orthonormalize_ts

def _flatten_list(list_U):
    return np.asarray([np.ravel(U) for U in list_U], dtype=float)


def _build_model(U_list, y, sigma0, ell0, fix_noise=True, noise=1e-10):
    X = _flatten_list(U_list)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    kern = GPy.kern.RBF(input_dim=X.shape[1], ARD=False)
    kern.variance = float(sigma0 ** 2)  # GPy uses σ_f^2
    kern.lengthscale = float(ell0)

    m = GPy.models.GPRegression(X, y, kernel=kern)
    m.Gaussian_noise.variance = float(noise)
    if fix_noise:
        m.Gaussian_noise.variance.fix()
    return m


# ---------- 1) Kernel matrix (kept for API parity; now via GPy) ----------
def RBF_kernel(list_U1, list_U2, kernel_params=[1, 1]):
    """
    Returns the kernel matrix for a set of unitary matrices.
    kernel_params = [log(sigma), log(ell)]
    """
    sigma = float(np.exp(kernel_params[0]))
    ell = float(np.exp(kernel_params[1]))
    X1, X2 = _flatten_list(list_U1), _flatten_list(list_U2)

    kern = GPy.kern.RBF(input_dim=X1.shape[1], ARD=False)
    kern.variance = sigma ** 2
    kern.lengthscale = ell

    K = kern.K(X1, X2)
    # mimic original tiny jitter on the common diagonal
    if list_U1 is list_U2 or (len(list_U1) == len(list_U2) and np.allclose(X1, X2)):
        n = min(len(list_U1), len(list_U2))
        K[np.arange(n), np.arange(n)] += 1e-10
    return K


# ---------- 2) GP posterior (same signature; delegates to GPy) ----------
def GP(X1, y1, X2, kernel_func, kernel_params):
    """
    Returns posterior mean and covariance of f(X2) given (X1,y1) under RBF kernel.
    Matches your original shapes: (μ2, Σ2).
    """
    sigma = float(np.exp(kernel_params[0]))
    ell = float(np.exp(kernel_params[1]))
    m = _build_model(X1, y1, sigma, ell, fix_noise=True, noise=1e-10)

    X2f = _flatten_list(X2)
    mu, var = m.predict_noiseless(X2f)  # (M,1), (M,1) — diag only
    # Build full Σ2 the same way your code conceptually defines it.
    # GPy can give cross-covariances; for a true full Σ2:
    #   Σ2 = m.posterior_covariance_between_points(X2f)
    # However, not all GPy versions expose it; fallback = diagonal matrix:
    try:
        Sigma2 = m.posterior_covariance_between_points(X2f)
    except Exception:
        Sigma2 = np.diagflat(var.ravel())

    return mu.ravel(), Sigma2


# ---------- 3) Negative log-likelihood (keeps your semantics) ----------
def log_likelihood(kernel_params, data_X, y, kernel):
    """
    Return the NEGATIVE log marginal likelihood (so it can be minimized).
    """
    sigma = float(np.exp(kernel_params[0]))
    ell = float(np.exp(kernel_params[1]))
    m = _build_model(data_X, y, sigma, ell, fix_noise=True, noise=1e-10)
    return -m.log_likelihood()


# ---------- 4) Hyperparameter search via GPy optimize ----------
def find_best_model(U_list, y, kernel, start_params):
    """
    Minimize NLL over log-params starting at start_params = [log σ, log ℓ].
    Enforces ℓ >= exp(0.25) to mirror your original bound.
    """
    sigma0 = float(np.exp(start_params[0]))
    ell0 = float(np.exp(start_params[1]))
    m = _build_model(U_list, y, sigma0, ell0, fix_noise=True, noise=1e-10)

    lower = float(np.exp(0.25))
    upper = float(1e12)

    # Constrain ℓ >= e^{0.25} ≈ 1.284 (mirrors your bound on log-ℓ >= 0.25)
    m.kern.lengthscale.transform = Logistic(lower, upper)
    # Keep variance positive; optionally add a modest lower bound
    m.kern.variance.constrain_positive()

    m.optimize('bfgs', max_iters=200)

    sigma_hat = float(np.sqrt(m.kern.variance[0]))  # σ
    ell_hat = float(m.kern.lengthscale[0])  # ℓ
    # Return in your convention: **log-params** (to be exp()’d by get_model)
    return np.array([np.log(sigma_hat), np.log(ell_hat)], dtype=float)


# ---------- 5) End-to-end (drop-in) ----------
def get_model_gpy(U_list, y, kernel, U_list_target, start_params=[-2, 0]):
    """
    Train (GPy) + predict, returning:
      (mean, diag(Σ2), np.exp(best_sigma))  where best_sigma = [log σ, log ℓ]
    This matches your original return signature exactly.
    """
    best_sigma = find_best_model(U_list, y, kernel, start_params)
    mu, Sigma2 = GP(U_list, y, U_list_target, RBF_kernel_unitary_matrices, best_sigma)
    return mu, np.diag(Sigma2), np.exp(best_sigma)


if __name__ == '__main__':
    # Mocking the simulation, for now
    batch_size = 49
    num_virtual_orbitals = 53
    num_occupied_orbitals = 5
    t1s = np.zeros((batch_size, num_virtual_orbitals, num_occupied_orbitals))
    t2s = np.zeros((batch_size, num_virtual_orbitals, num_virtual_orbitals, num_occupied_orbitals, num_occupied_orbitals))
    energies = np.zeros(batch_size)

    orth_ts = orthonormalize_ts(t1s, t2s)

    t1_ml = []
    t2_ml = []
    params_ml = []  # This used to be ml_params
    predictions = []
    stds_gpy = np.zeros(6561)   # target geometries

    for i in range(len(25)):
        mean, std, params = get_model_gpy(
            U_list=sample_u,
            U_list_target=target_u,
            kernel=RBF_kernel,
            y=t_coeffs[i] - np.mean(t_coeffs[i])
        )
        predictions.append(mean + np.mean(t_coeffs[i]))
        stds_gpy += (std)
        params_ml.append(params)

    means = np.array(predictions)
    means.shape

    for i in range(len(target_geometries)):
        t1_temp = np.zeros_like(t1s[0])
        t2_temp = np.zeros_like(t2s[0])

        for j in range(len(t_coeffs)):
            # This value has to be real
            t1_temp += means[j, i] * np.real(t1s_orth[j])
            t2_temp += means[j, i] * np.real(t2s_orth[j])

        t1_ml.append(t1_temp)
        t2_ml.append(t2_temp)

    # evc.solve_from_initial_guess() follows...