import numpy as np
from pyscf import gto, scf
from scipy.linalg import sqrtm, fractional_matrix_power

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
    mo_occ,
    mix_states=False,
    active_orbitals=None,
    nelec=None,
    return_R=False
):
    """
    Performs orthogonal Procrustes alignment on occupied and unoccupied molecular orbitals.
    """
    if active_orbitals is None:
        active_orbitals = np.arange(len(target_orbital))
    if nelec is None:
        nelec = int(np.sum(mo_occ))

    active_orbitals_occ = active_orbitals[:nelec // 2]
    active_orbitals_unocc = active_orbitals[nelec // 2:]
    target_orbital_new = target_orbital.copy()

    if not mix_states:
        # Align occupied orbitals
        mo = target_orbital[:, active_orbitals_occ]
        premo = reference_orbital[:, active_orbitals_occ]
        R1 = orthogonal_procrustes_overlap(premo, reference_overlap, mo, target_overlap)
        mo = mo @ R1
        target_orbital_new[:, active_orbitals_occ] = mo

        # Align unoccupied orbitals
        mo_unocc = target_orbital[:, active_orbitals_unocc]
        premo_unocc = reference_orbital[:, active_orbitals_unocc]
        R2 = orthogonal_procrustes_overlap(premo_unocc, reference_overlap, mo_unocc, target_overlap)
        mo_unocc = mo_unocc @ R2
        target_orbital_new[:, active_orbitals_unocc] = mo_unocc

        R = np.block([[R1, np.zeros((R1.shape[0], R2.shape[1]))],
                      [np.zeros((R2.shape[0], R1.shape[1])), R2]])
    else:
        # Align all active orbitals together
        mo = target_orbital[:, active_orbitals]
        premo = reference_orbital[:, active_orbitals]
        R = orthogonal_procrustes_overlap(premo, reference_overlap, mo, target_overlap)
        mo = mo @ R
        target_orbital_new[:, active_orbitals] = mo

    if return_R:
        return target_orbital_new, R
    return target_orbital_new

def align_orbitals():
    raise NotImplementedError

def compute_procrustes_matrices(
    geometries,
    molecule_fun,
    basis,
    reference_determinant,
    reference_overlap
):
    rotation_matrices = []
    procrustes_orbitals = []
    for geometry in geometries:
        mol = gto.Mole()
        if isinstance(geometry,tuple) or isinstance(geometry,np.ndarray) or isinstance(geometry,list):
        	mol.atom=molecule_fun(*geometry)
        else:
        	mol.atom=molecule_fun(geometry)
        mol.basis = basis
        mol.unit = "bohr"
        mol.build()
        hf=scf.RHF(mol)
        hf.kernel()
        orbital_coefficients = hf.mo_coeff
        orbital_occupancies = hf.mo_occ
        overlaps = mol.intor("int1e_ovlp")
        procrustes_orbital = localized_procrustes_overlap(
            mol,
            orbital_coefficients,
            orbital_occupancies,
            reference_determinant,
            overlaps,
            reference_overlap
        )

        rotation_matrix = np.real(fractional_matrix_power(overlaps,0.5)) @ procrustes_orbital
        rotation_matrices.append(rotation_matrix)
        procrustes_orbitals.append(procrustes_orbital)

    return {
        "rotation_matrices": rotation_matrices,
        "procrustes_orbitals": procrustes_orbitals
    }
