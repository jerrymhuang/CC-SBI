import numpy as np
from scipy.linalg import sqrtm, fractional_matrix_power

from utils.molecule_utils import (
    build_pyscf_molecule,
    compute_integrals,
    compute_hartree_fock
)

def orthogonal_procrustes_overlap(
    reference_determinant,
    reference_overlap,
    target_determinant,
    target_overlap
):
    reference_determinant_sqrtm = np.real(sqrtm(reference_determinant))
    reference_overlap_sqrtm = np.real(sqrtm(reference_overlap))

    matrix = target_determinant.T @ reference_overlap_sqrtm @ reference_determinant_sqrtm @ target_overlap

    U, S, V = np.linalg.svd(matrix)
    return U @ V


def localized_procrustes_overlap(
    reference_determinant,
    reference_overlap,
    target_determinant,
    target_overlap,
    occupancies,
    mix_states=False,
    active_orbitals=None,
    num_electrons=None,
):
    """
    Performs orthogonal Procrustes alignment on occupied and unoccupied molecular orbitals.
    """
    if active_orbitals is None:
        active_orbitals = np.arange(len(target_determinant))
    if num_electrons is None:
        num_electrons = int(np.sum(occupancies))

    occupied_active_orbitals = active_orbitals[:num_electrons // 2]
    unoccupied_active_orbitals = active_orbitals[num_electrons // 2:]
    target_determinant_new = target_determinant.copy()

    if not mix_states:
        # Align occupied orbitals
        mo = target_determinant[:, occupied_active_orbitals]
        premo = reference_determinant[:, occupied_active_orbitals]
        R1 = orthogonal_procrustes_overlap(premo, reference_overlap, mo, target_overlap)
        mo = mo @ R1
        target_determinant_new[:, occupied_active_orbitals] = mo

        # Align unoccupied orbitals
        mo_unocc = target_determinant[:, unoccupied_active_orbitals]
        premo_unocc = reference_determinant[:, unoccupied_active_orbitals]
        R2 = orthogonal_procrustes_overlap(premo_unocc, reference_overlap, mo_unocc, target_overlap)
        mo_unocc = mo_unocc @ R2
        target_determinant_new[:, unoccupied_active_orbitals] = mo_unocc

        # Construct block-diagonal matrix
        orthogonal_overlap = np.block([[R1, np.zeros((R1.shape[0], R2.shape[1]))],
                      [np.zeros((R2.shape[0], R1.shape[1])), R2]])
    else:
        # Align all active orbitals together
        mo = target_determinant[:, active_orbitals]
        premo = reference_determinant[:, active_orbitals]
        orthogonal_overlap = orthogonal_procrustes_overlap(premo, reference_overlap, mo, target_overlap)
        mo = mo @ orthogonal_overlap
        target_determinant_new[:, active_orbitals] = mo

    return {
        "target_determinant": target_determinant_new,
        "orthogonal_overlap": orthogonal_overlap
    }


def compute_procrustes_matrices(
    pyscf_atoms_group,
    basis,
    reference_determinant,
    reference_overlap
):
    rotation_matrices = []
    procrustes_orbitals = []
    for atoms in pyscf_atoms_group:
        # Build molecules
        molecule = build_pyscf_molecule(atoms=atoms)
        
        # Compute integrals
        integrals = compute_integrals(molecule)
        overlaps = integrals["overlaps"]
        
        # Compute RHF
        rhf = compute_hartree_fock(molecule)
        determinant = rhf["determinant"]
        occupancies = rhf["occupancies"]
        
        procrustes_orbital = localized_procrustes_overlap(
            target_determinant=determinant,
            target_occupancies=occupancies,
            target_overlap=overlaps,
            reference_determinant=reference_determinant,
            reference_overlap=reference_overlap,
        )

        rotation_matrix = np.real(fractional_matrix_power(overlaps,0.5)) @ procrustes_orbital
        rotation_matrices.append(rotation_matrix)
        procrustes_orbitals.append(procrustes_orbital)

    return {
        "rotation_matrices": rotation_matrices,
        "procrustes_orbitals": procrustes_orbitals
    }
