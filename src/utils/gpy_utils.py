import numpy as np
import GPy

from paramz.transformations import Logexp

# ----------------------------------------------------------------------
# Fixed Unitary RBF kernel – works with GPy ≥1.10
# ----------------------------------------------------------------------
class UnitaryRBF(GPy.kern.Kern):
    def __init__(self, input_dim, variance=1., lengthscale=1., active_dims=None, name='unitary_rbf'):
        super().__init__(input_dim=input_dim, active_dims=active_dims, name=name)
        self.variance = GPy.Param('variance', float(variance), Logexp())
        self.lengthscale = GPy.Param('lengthscale', float(lengthscale), Logexp())
        self.link_parameters(self.variance, self.lengthscale)

    def K(self, X, X2=None):
        X_3d = X.reshape(-1, int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1])))
        if X2 is not None:
            X2_3d = X2.reshape(-1, int(np.sqrt(X2.shape[1])), int(np.sqrt(X2.shape[1])))
        else:
            X2_3d = None

        sf2 = self.variance ** 2
        l2 = self.lengthscale ** 2

        diff = X_3d[:, None, :, :] - (X2_3d[None, :, :, :] if X2_3d is not None else X_3d[None, :, :, :])
        dist2 = np.sum(diff**2, axis=(2, 3))
        K = sf2 * np.exp(-0.5 * dist2 / l2)
        return K

    def Kdiag(self, X):
        return np.full(X.shape[0], float(self.variance ** 2))


def RBF_kernel_unitary(X, X2=None, log_params=(0.0, 0.0)):
    """
    RBF kernel for unitary matrices.
    """
    log_sf, log_l = log_params
    sf = np.exp(log_sf)
    l = np.exp(log_l)

    # Frobenius distance ||U_i - U_j||_F
    if X2 is None:
        X2 = X

    diff = X[:, None, :, :] - X2[None, :, :, :]  # (N, M, d, d)
    sq_norm = np.sum(diff ** 2, axis=(2, 3))  # (N, M)

    K = sf ** 2 * np.exp(-0.5 * sq_norm / l ** 2)
    return K


def get_model_gpy(U_list, y, U_list_target, kernel=None, start_params=None):
    N, d, _ = U_list.shape
    M, _, _ = U_list_target.shape

    X = U_list.reshape(N, -1)
    X_pred = U_list_target.reshape(M, -1)

    if kernel is None:
        variance = np.exp(start_params[0]) if start_params else 1.0
        lengthscale = np.exp(start_params[1]) if start_params else 1.0
        kern = UnitaryRBF(input_dim=d*d, variance=variance, lengthscale=lengthscale)
    else:
        kern = kernel

    model = GPy.models.GPRegression(X, y[:, None], kern, noise_var=1e-8)
    model.optimize(messages=False, max_iters=1000, optimizer='lbfgs')

    mean, var = model.predict(X_pred, full_cov=False)
    mean = mean.ravel()
    std = np.sqrt(var.ravel())

    params = [float(model.unitary_rbf.variance), float(model.unitary_rbf.lengthscale)]

    return mean, std, params


if __name__ == '__main__':
    pass
 #   Mocking the simulation, for now
    # batch_size = 49
    # num_virtual_orbitals = 53
    # num_occupied_orbitals = 5
    # t1s = np.zeros((batch_size, num_virtual_orbitals, num_occupied_orbitals))
    # t2s = np.zeros((batch_size, num_virtual_orbitals, num_virtual_orbitals, num_occupied_orbitals, num_occupied_orbitals))
    # energies = np.zeros(batch_size)
    #
    # orth_ts = orthonormalize_ts(t1s, t2s)
    #
    # t1_ml = []
    # t2_ml = []
    # params_ml = []  # This used to be ml_params
    # predictions = []
    # stds_gpy = np.zeros(6561)   # target geometries
    #
    # for i in range(len(25)):
    #     mean, std, params = get_model_gpy(
    #         U_list=sample_u,
    #         U_list_target=target_u,
    #         kernel=RBF_kernel,
    #         y=t_coeffs[i] - np.mean(t_coeffs[i])
    #     )
    #     predictions.append(mean + np.mean(t_coeffs[i]))
    #     stds_gpy += (std)
    #     params_ml.append(params)
    #
    # means = np.array(predictions)
    # means.shape
    #
    # for i in range(len(target_geometries)):
    #     t1_temp = np.zeros_like(t1s[0])
    #     t2_temp = np.zeros_like(t2s[0])
    #
    #     for j in range(len(t_coeffs)):
    #         # This value has to be real
    #         t1_temp += means[j, i] * np.real(t1s_orth[j])
    #         t2_temp += means[j, i] * np.real(t2s_orth[j])
    #
    #     t1_ml.append(t1_temp)
    #     t2_ml.append(t2_temp)

    # evc.solve_from_initial_guess() follows...