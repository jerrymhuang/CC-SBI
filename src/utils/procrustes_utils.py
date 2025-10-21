import numpy as np
from scipy.linalg import sqrtm, svd

def orthogonal_procrustes_overlap(
    reference_orbital,
    reference_overlap,
    target_orbital,
    target_overlap
):
    reference_orbital_sqrtm = np.real(sqrtm(reference_orbital))
    reference_overlap_sqrtm = np.real(sqrtm(reference_overlap))

    matrix = target_orbital.T @ reference_overlap_sqrtm @ reference_orbital_sqrtm @ target_overlap

    U, S, V = np.linalg.svd(matrix)
    return U @ V


def localized_procrustes_overlap(
    reference_orbital,
    reference_overlap,
    target_orbital,
    target_overlap,

)
