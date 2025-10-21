import numpy as np
from collections.abc import Sequence, Iterable, Callable
import pyscf
from pyscf import gto, scf, cc


def assemble_molecules(molecule_fun, molecule_kwargs: dict | None = None) -> dict[str, np.ndarray]:
    """
    Generate a single molecule or atom set from molecule_fun specification.
    """
    molecule_kwargs = molecule_kwargs or {}
    base = molecule_fun(**molecule_kwargs)

    # Minimal validation: ensure base is a list of valid tuples
    if not isinstance(base, list) or not all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        and isinstance(item[1], (list, tuple)) and len(item[1]) == 3
        for item in base
    ):
        raise TypeError("molecule_fun must return a list of (str, [float, float, float]) tuples")

    atoms = [atom for atom, _ in base]
    positions = [coord for _, coord in base]

    return {
        "atoms": np.array(atoms, dtype=object),
        "positions": np.array(positions, dtype=np.float32),
    }


def build_pyscf_molecules(
    atoms: np.ndarray,
    positions: np.ndarray,
    unit: str = "Angstrom",
    basis: str = "sto3g",
    cartesian: bool = True,
    verbose: bool = False,
    charge: int = 0
):
    """
    Run RHF → CCSD for closed-shell molecules and return features for machine learning.
    """
    atoms = np.asarray(atoms, dtype=object).reshape(-1)
    pos = np.asarray(positions, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != atoms.shape[0]:
        raise ValueError(
            f"pos must have shape (N_atoms, 3) and match atoms length; got {pos.shape} vs {atoms.shape[0]}"
        )

    pyscf_atoms = [(str(atoms[i]), pos[i].tolist()) for i in range(atoms.shape[0])]

    # Set up and build molecule based on geometries
    mol = gto.Mole()
    mol.unit = unit
    mol.atom = pyscf_atoms
    mol.basis = basis
    mol.cart = cartesian
    mol.verbose = verbose
    mol.charge = 0 if charge is None else int(charge)

    num_electrons = sum(gto.charge(atom[0]) for atom in pyscf_atoms) - mol.charge
    if num_electrons % 2 != 0:
        raise ValueError("Only closed-shell molecules are supported (even number of electrons required)")

    mol.spin = 0
    mol.build()

    return mol


def compute_geometries(
    molecule: gto.Mole,
    coordinate_scale: float = 1.0
):
    raise NotImplementedError


def compute_integrals(
    molecule: gto.Mole,
    full_matrices: bool = False,
):
    kinetic_energy = molecule.intor("int1e_kin").astype(np.float32)
    full_nuc_attraction = molecule.intor("int1e_nuc").astype(np.float32)
    full_overlap = molecule.intor("int1e_ovlp").astype(np.float32)
    eri = molecule.intor("int2e_sph", aosym=1).astype(np.float32)

    num_basis = full_nuc_attraction.shape[0]
    tril_idx = np.tril_indices(num_basis)
    nuc_attraction = full_nuc_attraction[tril_idx].astype(np.float32)
    overlap = full_overlap[tril_idx].astype(np.float32)

    return {
        "kinetic_energy": kinetic_energy,
        "nuc_attraction": full_nuc_attraction if full_matrices else nuc_attraction,
        "overlap": overlap if full_matrices else overlap,
        "eri": eri,
    }

def compute_hartree_fock(molecules):
    raise NotImplementedError

def compute_cc():
    raise NotImplementedError

def compute_ccsd(
    atoms: Iterable[str],
    pos: np.ndarray,
    unit: str = "angstrom",
    basis: str = "sto3g",
    cartesian: bool = False,
    coordinate_scale: float | None = 0.1,
    verbose: int = 0,
    return_amplitudes: bool = True,
    return_geometries: bool = False,
    charge: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run RHF → CCSD for closed-shell molecules and return features for machine learning.
    """
    atoms = np.asarray(atoms, dtype=object).reshape(-1)
    pos = np.asarray(pos, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] != atoms.shape[0]:
        raise ValueError(
            f"pos must have shape (N_atoms, 3) and match atoms length; got {pos.shape} vs {atoms.shape[0]}"
        )

    pyscf_atoms = [(str(atoms[i]), pos[i].tolist()) for i in range(atoms.shape[0])]

    coordinates = np.array(
        [coordinate for _, coordinate in pyscf_atoms], dtype=np.float32
    )
    if coordinate_scale is not None:
        coordinates = (coordinates.reshape(-1) * coordinate_scale).astype(np.float32)
    else:
        coordinates = coordinates.reshape(-1).astype(np.float32)

    # Set up and build molecule based on geometries
    mol = gto.Mole()
    mol.unit = unit
    mol.atom = pyscf_atoms
    mol.basis = basis
    mol.cart = cartesian
    mol.verbose = verbose
    mol.charge = 0 if charge is None else int(charge)

    num_electrons = sum(gto.charge(atom[0]) for atom in pyscf_atoms) - mol.charge
    if num_electrons % 2 != 0:
        raise ValueError("Only closed-shell molecules are supported (even number of electrons required)")

    mol.spin = 0  
    mol.build()

    kinetic = mol.intor("int1e_kin").astype(np.float32)
    eri = mol.intor("int2e_sph", aosym=1).astype(np.float32)
    full_potential = mol.intor("int1e_nuc").astype(np.float32)
    full_overlap = mol.intor("int1e_ovlp").astype(np.float32)

    n_basis = full_potential.shape[0]
    tril_idx = np.tril_indices(n_basis)
    nuc_potential = full_potential[tril_idx].astype(np.float32)
    overlap = full_overlap[tril_idx].astype(np.float32)

    # Run restricted Hartree-Fock (RHF) and get orbital coefficients (determinant)
    rhf = scf.RHF(mol).run()
    orbital_occupancy = rhf.mo_occ
    orbital_coefficients = rhf.mo_coeff
    ccsd = cc.CCSD(rhf).run()

    sim_data: dict[str, np.ndarray] = {
        "nuc_potential": nuc_potential,
        "overlap": overlap,
        "coordinates": coordinates,
        "orbital_coefficients": orbital_coefficients,
        "orbital_occupancy": orbital_occupancy,
    }

    cc_t1_full = ccsd.t1.astype(np.float32)
    cc_t2_full = ccsd.t2.astype(np.float32)
    energies = ccsd.e_tot.astype(np.float32)
    sim_data["t1"] = cc_t1_full.reshape(-1).astype(np.float32)
    sim_data["t2"] = cc_t2_full.reshape(-1).astype(np.float32)
    sim_data["energies"] = energies.reshape(-1).astype(np.float32)

    if return_amplitudes:
        sim_data.update(
            cc_t1_full=cc_t1_full,
            cc_t2_full=cc_t2_full,
            eri=eri,
            full_overlap=full_overlap,
            kinetic=kinetic,
            nuc_potential=full_potential,
        )

    if return_geometries:
        # TODO: add geometric data for the molecule
        pass

    return sim_data
