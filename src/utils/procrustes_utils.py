import numpy as np
from tqdm import tqdm
from pyscf import ao2mo
from scipy.linalg import sqrtm, fractional_matrix_power

from utils.molecule_utils import (
    build_pyscf_molecule,
    compute_hartree_fock,
    compute_cc,
)

def orthogonal_procrustes_overlap(
    reference_determinant,
    reference_overlap,
    target_determinant,
    target_overlap
):
    target_overlap_sqrtm = np.real(sqrtm(target_overlap))
    reference_overlap_sqrtm = np.real(sqrtm(reference_overlap))

    matrix = target_determinant.T @ target_overlap_sqrtm @ reference_overlap_sqrtm @ reference_determinant

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
        orthogonal_overlap = np.block([
            [R1, np.zeros((R1.shape[0], R2.shape[0]))],
            [np.zeros((R2.shape[0], R1.shape[0])), R2]
        ])

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

def compute_reference_hartree_fock(
    molecule,
    reference_determinant,
    reference_overlap,
    reference_state = None,
    mix_states = False
):
    target_overlap = molecule.intor("int1e_ovlp")
    rhf = compute_hartree_fock(molecule=molecule)

    if reference_state is not None:
        canonical_orbital = localized_procrustes_overlap(
            reference_determinant=reference_determinant,
            reference_overlap=reference_overlap,
            target_determinant=rhf["determinant"],
            target_overlap=target_overlap,
            occupancies=rhf["occupancies"],
            mix_states=mix_states
        )
    else:
        canonical_orbital = rhf["determinant"]

    num_orbitals = canonical_orbital.shape[1]
    mol_core_hamiltonian = canonical_orbital.T @ rhf.get_hcore() @ canonical_orbital
    mol_overlap_matrix = canonical_orbital.T @ target_overlap @ canonical_orbital

    u = ao2mo.kernel(molecule, canonical_orbital).reshape(num_orbitals, num_orbitals, num_orbitals, num_orbitals)
    fock_matrix = rhf.get_fock(mol_core_hamiltonian, u)
    num_electrons = molecule.nelectron // 2

    occupied_orbitals = slice(0, num_electrons)
    virtual_orbitals = slice(num_electrons, num_orbitals)

    return {
        "canonical_orbital": canonical_orbital,
        "core_hamiltonian": mol_core_hamiltonian,
        "overlap_matrix": mol_overlap_matrix,
        "fock_matrix": fock_matrix,
        "occupied_orbitals": occupied_orbitals,
        "virtual_orbitals": virtual_orbitals,
    }

def compute_procrustes_matrices(
    batched_atoms,
    batched_positions,
    reference_determinant,
    reference_overlap
):
    rotation_matrices = []
    procrustes_orbitals = []

    for atoms, positions in tqdm(
        zip(batched_atoms, batched_positions),
        desc="Computing procrustes",
        total=batched_atoms.shape[0]
    ):
        # Build molecules
        molecule = build_pyscf_molecule(atoms=atoms, positions=positions)
        
        # For integral, we only need overlap here
        overlaps = molecule.intor("int1e_ovlp").astype(np.float32)
        
        # Compute RHF
        rhf = compute_hartree_fock(molecule)
        determinant = rhf["determinant"]
        occupancies = rhf["occupancies"]
        
        procrustes_overlap = localized_procrustes_overlap(
            target_determinant=determinant,
            target_overlap=overlaps,
            reference_determinant=reference_determinant,
            reference_overlap=reference_overlap,
            occupancies=occupancies,
        )

        procrustes_orbital = procrustes_overlap["target_determinant"]

        rotation_matrix = np.real(fractional_matrix_power(overlaps,0.5)) @ procrustes_orbital
        rotation_matrices.append(rotation_matrix)
        procrustes_orbitals.append(procrustes_orbital)

    return {
        "rotation_matrices": np.array(rotation_matrices),
        "procrustes_orbitals": np.array(procrustes_orbitals)
    }



def compute_cc_with_procrustes(
    batched_atoms,
    batched_positions,
    reference_determinant: np.ndarray,
    reference_overlap: np.ndarray,
    mix_states: bool = False
):
    t1s = []
    t2s = []
    energies = []

    for atoms, positions in tqdm(
        zip(batched_atoms, batched_positions),
        desc="Computing CCSD",
        total=batched_atoms.shape[0]
    ):
        # Build molecules
        molecule = build_pyscf_molecule(atoms=atoms, positions=positions)

        # For integral, we only need overlap here
        overlaps = molecule.intor("int1e_ovlp").astype(np.float32)

        # Run Hartree-Fock
        rhf = compute_hartree_fock(molecule)

        procrustes_overlap = localized_procrustes_overlap(
            reference_determinant=reference_determinant,
            reference_overlap=reference_overlap,
            target_determinant=rhf["determinant"],
            target_overlap=overlaps,
            occupancies=rhf["occupancies"],
            mix_states=mix_states,
        )

        # Update RHF determinant
        rhf["determinant"] = procrustes_overlap["target_determinant"]

        # Run CC with the updated RHF
        cc = compute_cc(molecule=molecule)
        t1s.append(cc["t1"])
        t2s.append(cc["t2"])
        energies.append(cc["total_energy"])

    return {
        "t1": np.array(t1s),
        "t2": np.array(t2s),
        "energies": np.array(energies),
    }
