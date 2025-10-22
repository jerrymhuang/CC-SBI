import numpy as np
from pyscf import gto, scf, cc
from collections.abc import Iterable, Callable
from .procrustes_utils import localized_procrustes_overlap


def assemble_molecule(molecule_fun, molecule_kwargs: dict | None = None) -> dict[str, np.ndarray]:
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


def build_pyscf_atoms(atoms: np.ndarray, positions: np.ndarray):
    atoms = np.asarray(atoms, dtype=object).reshape(-1)
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != atoms.shape[0]:
        raise ValueError(
            f"pos must have shape (N_atoms, 3) and match atoms length; got {positions.shape} vs {atoms.shape[0]}"
        )

    pyscf_atoms = [(str(atoms[i]), positions[i].tolist()) for i in range(atoms.shape[0])]

    return pyscf_atoms


def build_pyscf_molecule(
    atoms: np.ndarray | None = None,
    positions: np.ndarray | None = None,
    pyscf_atoms: list[tuple[str, list[float]]] | None = None,
    unit: str = "bohr",
    basis: str = "cc-pVTZ",
    cartesian: bool = False,
    verbose: int = 0,
    charge: int = 0
):
    # Build atoms if none provided
    if pyscf_atoms is None:
        pyscf_atoms = build_pyscf_atoms(atoms, positions)

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
        raise ValueError("Only closed-shell molecule are supported (even number of electrons required)")

    mol.spin = 0
    mol.build()

    return mol


def compute_coordinates(
    atoms: Iterable[str],
    positions: np.ndarray,
    coordinate_scale: float = 1.0
):
    atoms = np.asarray(atoms, dtype=object).reshape(-1)
    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] != atoms.shape[0]:
        raise ValueError(
            f"pos must have shape (N_atoms, 3) and match atoms length; got {positions.shape} vs {atoms.shape[0]}"
        )

    pyscf_atoms = [(str(atoms[i]), positions[i].tolist()) for i in range(atoms.shape[0])]

    coordinates = np.array(
        [coordinate for _, coordinate in pyscf_atoms], dtype=np.float32
    )
    if coordinate_scale is not None:
        coordinates = (coordinates.reshape(-1) * coordinate_scale).astype(np.float32)
    else:
        coordinates = coordinates.reshape(-1).astype(np.float32)

    return coordinates


def compute_integrals(
    molecule: gto.Mole,
    full_matrices: bool = True,
):
    kinetic_energy = molecule.intor("int1e_kin").astype(np.float32)
    full_nuc_attraction = molecule.intor("int1e_nuc").astype(np.float32)
    full_overlaps = molecule.intor("int1e_ovlp").astype(np.float32)
    eri = molecule.intor("int2e_sph", aosym=1).astype(np.float32)

    num_basis = full_nuc_attraction.shape[0]
    tril_idx = np.tril_indices(num_basis)
    nuc_attraction = full_nuc_attraction[tril_idx].astype(np.float32)
    overlaps = full_overlaps[tril_idx].astype(np.float32)

    return {
        "kinetic_energy": kinetic_energy,
        "nuc_attraction": full_nuc_attraction if full_matrices else nuc_attraction,
        "overlaps": full_overlaps if full_matrices else overlaps,
        "eri": eri,
    }


def compute_hartree_fock(molecule: gto.Mole):

    rhf = scf.RHF(molecule).run()
    occupancies = rhf.mo_occ
    determinant = rhf.mo_coeff

    return {
        "occupancies": occupancies,
        "determinant": determinant
    }


def compute_cc(
    molecule: gto.Mole | None = None,
    rhf: scf.RHF | None = None,
    flatten: bool = False
):
    # Hartree-Fock
    if rhf is None:
        rhf = scf.RHF(molecule).run()

    # CCSD
    ccsd = cc.CCSD(rhf).run()
    t1 = ccsd.t1.astype(np.float32)
    t2 = ccsd.t2.astype(np.float32)
    energy = ccsd.e_tot.astype(np.float32)

    return {
        "t1": t1.reshape(-1) if flatten else t1,
        "t2": t2.reshape(-1) if flatten else t2,
        "total_energy": energy
    }


def compute_ccsd(
    molecule_fun: Callable,
    molecule_kwargs: dict,
    unit: str = "bohr",
    basis: str = "cc-pVTZ",
    cartesian: bool = False,
    verbose: int = 0,
    charge: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run RHF → CCSD for closed-shell molecule and return features for machine learning.
    """
    raw_molecule = assemble_molecule(molecule_fun, molecule_kwargs)
    pyscf_molecule = build_pyscf_molecule(
        **raw_molecule,
        unit=unit,
        basis=basis,
        cartesian=cartesian,
        charge=charge,
        verbose=verbose
    )
        
    integrals = compute_integrals(pyscf_molecule)
    cc = compute_cc(pyscf_molecule)

    # Gather all data
    sim_data = raw_molecule | integrals | cc

    return sim_data
