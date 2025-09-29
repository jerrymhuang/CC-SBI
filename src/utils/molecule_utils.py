import numpy as np
from collections.abc import Sequence, Iterable, Callable
from pyscf import gto, scf, cc


def assemble_molecules(
    species: str
    | list[tuple[str, Sequence[float]]]
    | list[tuple[str, np.ndarray]]
    | dict[str, Sequence[float]]
    | Callable = "H",
    species_kwargs: dict | None = None,
    num_molecules: int = 1,
    arrangement: str = "chain",
    cluster_size: float = 2.0,
    distance: float = 2.0,
    min_inter_dist: float | None = None,
    perturb: bool = True,
    position_noise: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Generate a set of atoms or molecular fragments in a specified arrangement.
    """
    species_kwargs = species_kwargs or {}
    species_kwargs["perturb"] = perturb
    species_kwargs["noise_std"] = position_noise  # Pass position_noise as noise_std for h_atom

    # Normalize non-callable species to a fixed base (computed once, no perturbation)
    if not callable(species):
        if isinstance(species, str):
            fixed_base = [(species, [0.0, 0.0, 0.0])]
        elif isinstance(species, dict):
            fixed_base = list(species.items())
        elif isinstance(species, list):
            fixed_base = species
        else:
            raise TypeError(
                "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
            )
    else:
        fixed_base = None

    atoms: list[str] = []
    positions: list[list[float]] = []

    for i in range(num_molecules):
        unit_center = np.array([i * distance, 0.0, 0.0], dtype=float)

        # Get base for this unit (callable gets called per unit for independent perturbations)
        if callable(species):
            base = species(**species_kwargs)
            if not isinstance(base, list) or not all(
                isinstance(item, tuple)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], (list, tuple))
                and len(item[1]) == 3
                for item in base
            ):
                raise TypeError(
                    "Callable species must return a list of (str, [float, float, float]) tuples"
                )
        else:
            base = fixed_base

        for atom, coord in base:
            xyz = np.asarray(coord, dtype=float) + unit_center
            atoms.append(str(atom))
            positions.append(xyz.tolist())

    return {
        "species": np.array(atoms, dtype=object),
        "pos": np.array(positions, dtype=np.float32),
    }


def compute_ccsd(
    species: Iterable[str] | list[tuple[str, Sequence[float]]],
    pos: np.ndarray | None = None,
    basis: str = "sto3g",
    coordinate_scale: float | None = 0.1,
    verbose: int = 0,
    return_amplitudes: bool = False,
    charge: int | None = None,
    spin: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Run mean-field → CC and return features for machine learning.
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

    mol = gto.Mole()
    mol.atom = pyscf_atoms
    mol.basis = basis
    mol.verbose = verbose
    mol.charge = 0 if charge is None else int(charge)

    n_electrons = sum(gto.charge(atom[0]) for atom in pyscf_atoms) - mol.charge
    if spin is None:
        mol.spin = int(n_electrons % 2)
    else:
        mol.spin = int(spin)
    mol.build()

    open_shell = mol.spin != 0
    kinetic = mol.intor("int1e_kin").astype(np.float32)
    full_nuc_potential = mol.intor("int1e_nuc").astype(np.float32)
    eri = mol.intor("int2e_sph", aosym=1).astype(np.float32)
    full_overlap = mol.intor("int1e_ovlp").astype(np.float32)

    # Run unrestricted CCSD if open shell, CCSD otherwise
    if open_shell:
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf).run()
        is_uccsd = True
    else:
        mf = scf.RHF(mol).run()
        mycc = cc.CCSD(mf).run()
        is_uccsd = False

    coordinates = np.array(
        [coordinate for _, coordinate in pyscf_atoms], dtype=np.float32
    )
    if coordinate_scale is not None:
        coordinates = (coordinates.reshape(-1) * coordinate_scale).astype(np.float32)
    else:
        coordinates = coordinates.reshape(-1).astype(np.float32)

    n_basis = full_nuc_potential.shape[0]
    tril_idx = np.tril_indices(n_basis)
    nuc_potential = full_nuc_potential[tril_idx].astype(np.float32)
    overlap = full_overlap[tril_idx].astype(np.float32)

    sim_data: dict[str, np.ndarray] = {
        "nuc_potential": nuc_potential,
        "overlap": overlap,
        "coordinates": coordinates,
    }

    if is_uccsd:
        t1a, t1b = mycc.t1
        t2aa, t2ab, t2bb = mycc.t2
        cc_t1 = np.concatenate([t1a.ravel(), t1b.ravel()]).astype(np.float32)
        sim_data["t1"] = cc_t1
        if return_amplitudes:
            sim_data.update(
                cc_t1_full=cc_t1,
                cc_t2_full=np.concatenate(
                    [t2aa.ravel(), t2ab.ravel(), t2bb.ravel()]
                ).astype(np.float32),
                eri=eri,
                full_overlap=full_overlap,
                kinetic=kinetic,
                nuc_potential=full_nuc_potential,
            )
    else:
        cc_t1_full = mycc.t1.astype(np.float32)
        cc_t2_full = mycc.t2.astype(np.float32)
        sim_data["t1"] = cc_t1_full.reshape(-1).astype(np.float32)
        if return_amplitudes:
            sim_data.update(
                cc_t1_full=cc_t1_full,
                cc_t2_full=cc_t2_full,
                eri=eri,
                full_overlap=full_overlap,
                kinetic=kinetic,
                nuc_potential=full_nuc_potential,
            )

    return sim_data
