import numpy as np
from collections.abc import Sequence, Iterable, Callable
from pyscf import gto, scf, cc


def assemble_molecules(species, species_kwargs: dict | None = None) -> dict[str, np.ndarray]:
    """
    Generate a single molecule or atom set from species specification.
    """
    species_kwargs = species_kwargs or {}

    # Normalize species to list[tuple[str, list[float]]]
    if callable(species):
        base = species(**species_kwargs)
        if not isinstance(base, list) or not all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                and isinstance(item[1], (list, tuple)) and len(item[1]) == 3
                for item in base
        ):
            raise TypeError(
                "Callable species must return a list of (str, [float, float, float]) tuples"
            )
    else:
        if isinstance(species, str):
            base = [(species, [0.0, 0.0, 0.0])]
        elif isinstance(species, dict):
            base = list(species.items())
        elif isinstance(species, list):
            base = species
        else:
            raise TypeError(
                "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
            )

    atoms = [str(atom) for atom, coord in base]
    positions = [np.asarray(coord, dtype=float).tolist() for atom, coord in base]

    return {
        "species": np.array(atoms, dtype=object),
        "pos": np.array(positions, dtype=np.float32),
    }


def compute_ccsd(
    species: Iterable[str] | list[tuple[str, Sequence[float]]],
    pos: np.ndarray | None = None,
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
    if isinstance(species, list) and len(species) > 0 and isinstance(species[0], tuple):
        pyscf_atoms = [(str(atom), list(map(float, coord))) for atom, coord in species]
        if pos is not None:
            raise ValueError(
                "pos must be None when species is a list of (atom, coord) tuples"
            )
    else:
        sp = np.asarray(species, dtype=object).reshape(-1)
        if pos is None:
            raise ValueError(
                "pos must be provided when species is an iterable of symbols"
            )
        xyz = np.asarray(pos, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != sp.shape[0]:
            raise ValueError(
                f"pos must have shape (N_atoms, 3) and match species length; got {xyz.shape} vs {sp.shape[0]}"
            )
        pyscf_atoms = [(str(sp[i]), xyz[i].tolist()) for i in range(sp.shape[0])]

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

    # Run restricted Hartree-Fock (RHF) and get orbital coefficients (determinant)
    rhf = scf.RHF(mol).run()
    mol_coefficients = rhf.mo_coeff
    ccsd = cc.CCSD(rhf).run()

    kinetic = mol.intor("int1e_kin").astype(np.float32)
    eri = mol.intor("int2e_sph", aosym=1).astype(np.float32)
    full_potential = mol.intor("int1e_nuc").astype(np.float32)
    full_overlap = mol.intor("int1e_ovlp").astype(np.float32)

    coordinates = np.array(
        [coordinate for _, coordinate in pyscf_atoms], dtype=np.float32
    )
    if coordinate_scale is not None:
        coordinates = (coordinates.reshape(-1) * coordinate_scale).astype(np.float32)
    else:
        coordinates = coordinates.reshape(-1).astype(np.float32)

    n_basis = full_potential.shape[0]
    tril_idx = np.tril_indices(n_basis)
    nuc_potential = full_potential[tril_idx].astype(np.float32)
    overlap = full_overlap[tril_idx].astype(np.float32)

    sim_data: dict[str, np.ndarray] = {
        "nuc_potential": nuc_potential,
        "overlap": overlap,
        "coordinates": coordinates,
        "orbital_coefficients": mol_coefficients,
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