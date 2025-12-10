import numpy as np

def khatri_rao(A, B):
    I, R = A.shape
    J, Rb = B.shape
    assert R == Rb
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, R)

def unfold(X, mode):
    n_dims = X.ndim
    order = (mode,) + tuple(i for i in range(n_dims) if i != mode)
    X_perm = np.transpose(X, order)
    return X_perm.reshape(X.shape[mode], -1)


def cp_als(
        X,
        rank,
        n_iter_max=500,
        tol=1e-6,
        verbose=False,
        random_state=None,
        A_init=None,
        B_init=None,
        C_init=None
):
    """
    Computes the canonical polyadic (CP) decomposition with alternating least squares (ALS).
    """
    rng = np.random.default_rng(random_state)
    I, J, K = X.shape

    if A_init is not None:
        A = A_init
    else:
        A = rng.standard_normal((I, rank))

    if B_init is not None:
        B = B_init
    else:
        B = rng.standard_normal((J, rank))

    if C_init is not None:
        C = C_init
    else:
        C = rng.standard_normal((K, rank))

    X1 = unfold(X, 0)
    X2 = unfold(X, 1)
    X3 = unfold(X, 2)

    prev_error = None

    for it in range(n_iter_max):
        BtB = B.T @ B  # Is this a bug???
        CtC = C.T @ C
        KR = khatri_rao(B, C)
        G = BtB * CtC
        RHS = X1 @ KR
        A = RHS @ np.linalg.pinv(G)

        AtA = A.T @ A
        KR = khatri_rao(A, C)
        G = AtA * CtC
        RHS = X2 @ KR
        B = RHS @ np.linalg.pinv(G)

        BtB = B.T @ B
        KR = khatri_rao(A, B)
        G = AtA * BtB
        RHS = X3 @ KR
        C = RHS @ np.linalg.pinv(G)

        X_hat = np.einsum('ir,jr,kr->ijk', A, B, C)
        error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)

        if verbose and it % 10 == 0:
            print(f"iter {it:4d}  rel_error = {error:.3e}")

        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error

    return A, B, C, error


def update_C(X, A, B):
    I, J, K = X.shape
    R = A.shape[1]

    X3 = X.transpose(2, 0, 1).reshape(K, I * J)

    KR = khatri_rao(A, B)

    C_t, *_ = np.linalg.lstsq(KR, X3.T, rcond=None)
    C = C_t.T

    return C