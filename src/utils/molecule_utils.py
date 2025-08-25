import numpy as np
from collections.abc import Sequence, Iterable, Callable
from pyscf import gto, scf, cc


def assemble_molecules(
    num_molecules: int = 1,
    bond_distance: float = 2.0,
    species: str
    | list[tuple[str, Sequence[float]]]
    | dict[str, Sequence[float]]
    | Callable = "H",
    perturb: bool = True,
    seed: int | None = None,
    species_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate a set of atoms or molecular fragments in a chain.

    Parameters
    ----------
    num_molecules : int, optional
        Number of repeats of the base unit (atom or fragment), default is 1.
    bond_distance : float, optional
        Spacing between consecutive units along +x (Å), default is 2.0.
    species : str, list of (atom, coord), dict, or callable, optional
        Defines the base unit to repeat:
        - str: Single atom symbol (e.g., "H").
        - list of (str, [float, float, float]): Fragment as list of (atom, coordinates).
        - dict {str: [float, float, float]}: Fragment as dictionary of atom to coordinates.
        - callable: Function returning a list of (atom, coord) tuples (e.g., a molecule generator).
        Default is "H".
    perturb : bool, optional
        If True, adds random displacements (±0.125 Å) per unit to break symmetry, default is True.
    seed : int, optional
        Random number generator seed for perturbations, default is None.
    species_kwargs : dict, optional
        Keyword arguments to pass to the callable `species`, if provided, default is None.

    Returns
    -------
    dict
        Dictionary with keys:
        - "species": np.ndarray of shape (N_atoms,), atom symbols.
        - "pos": np.ndarray of shape (N_atoms, 3), Cartesian coordinates in Å.

    Raises
    ------
    TypeError
        If `species` is not a string, list, dict, or callable, or if callable returns invalid output.
    """
    rng = np.random.default_rng(seed)
    species_kwargs = species_kwargs or {}

    # Normalize species to a list[(atom, coord)] at the origin
    if isinstance(species, str):
        base = [(species, [0.0, 0.0, 0.0])]
    elif isinstance(species, dict):
        base = list(species.items())
    elif isinstance(species, list):
        base = species
    elif callable(species):
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
        raise TypeError(
            "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
        )

    atoms: list[str] = []
    positions: list[list[float]] = []

    for i in range(num_molecules):
        offset = np.array([i * bond_distance, 0.0, 0.0], dtype=float)
        noise = (0.25 * rng.random(3) - 0.125) if perturb else np.zeros(3)
        for atom, coord in base:
            xyz = np.asarray(coord, dtype=float) + offset + noise
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

    Automatically chooses RHF/CCSD for closed-shell systems and UHF/UCCSD for
    open-shell (e.g., odd-electron hydrogen chains).

    Parameters
    ----------
    species : iterable of str or list of (atom, coord)
        Atom symbols or list of (atom, [x, y, z]) tuples defining the molecule.
        If a list of tuples, `pos` is ignored.
    pos : array-like of shape (N_atoms, 3), optional
        Cartesian coordinates (Å) for `species` if `species` is an iterable of symbols.
    basis : str, optional
        Basis set name, default "sto3g".
    coordinate_scale : float or None, optional
        If set, multiply flattened coordinates by this factor (default 0.1).
    verbose : int, optional
        PySCF verbosity.
    return_amplitudes : bool, optional
        If True, include full amplitudes and integral blocks.
    charge : int or None, optional
        Total molecular charge. If None, assumed 0.
    spin : int or None, optional
        2*S = (#alpha - #beta). If None, inferred from electron parity (0 for even, 1 for odd).

    Returns
    -------
    dict
        {
          "nuc_potential": lower-triangular nuclear attraction (Å^-1),
          "overlap": lower-triangular overlap,
          "coordinates": flattened (scaled if coordinate_scale not None),
          "cc_t1": flattened singles (RCCSD: t1; UCCSD: concat[t1a, t1b]),
          # If return_amplitudes:
          "cc_t1_full": (RCCSD) t1   OR (UCCSD) np.concatenate([t1a, t1b], axis=None),
          "cc_t2_full": (RCCSD) t2   OR (UCCSD) np.concatenate([t2aa, t2ab, t2bb], axis=None),
          "uccsd_blocks": dict with t1a,t1b,t2aa,t2ab,t2bb (only for UCCSD),
          "eri", "full_overlap", "kinetic", "nuc_potential": full matrices
        }
    """
    # Handle input: either species as list[(atom, coord)] or species+pos
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

    # Build PySCF molecule
    mol = gto.Mole()
    mol.atom = pyscf_atoms
    mol.basis = basis
    mol.verbose = verbose
    mol.charge = 0 if charge is None else int(charge)

    # Infer electron count and spin if not provided
    n_electrons = sum(gto.charge(atom[0]) for atom in pyscf_atoms) - mol.charge
    if spin is None:
        # closed-shell for even electrons, doublet for odd (e.g., H-atom systems)
        mol.spin = int(n_electrons % 2)
    else:
        mol.spin = int(spin)
    mol.build()

    # One-electron / open-shell handling:
    open_shell = mol.spin != 0

    # Compute integrals
    kinetic = mol.intor("int1e_kin").astype(np.float32)
    full_nuc_potential = mol.intor("int1e_nuc").astype(np.float32)
    eri = mol.intor("int2e_sph", aosym=1).astype(np.float32)
    full_overlap = mol.intor("int1e_ovlp").astype(np.float32)

    # Mean-field
    if open_shell:
        mf = scf.UHF(mol).run()
        mycc = cc.UCCSD(mf).run()
        is_uccsd = True
    else:
        mf = scf.RHF(mol).run()
        mycc = cc.CCSD(mf).run()
        is_uccsd = False

    # Coordinates
    coordinates = np.array(
        [coordinate for _, coordinate in pyscf_atoms], dtype=np.float32
    )
    if coordinate_scale is not None:
        coordinates = (coordinates.reshape(-1) * coordinate_scale).astype(np.float32)
    else:
        coordinates = coordinates.reshape(-1).astype(np.float32)

    # Pack integrals to lower-tri
    n_basis = full_nuc_potential.shape[0]
    tril_idx = np.tril_indices(n_basis)
    nuc_potential = full_nuc_potential[tril_idx].astype(np.float32)
    overlap = full_overlap[tril_idx].astype(np.float32)

    # Amplitudes (unified output)
    out: dict[str, np.ndarray] = {
        "nuc_potential": nuc_potential,
        "overlap": overlap,
        "coordinates": coordinates,
    }

    if is_uccsd:
        # PySCF UCCSD stores spin-resolved blocks
        t1a, t1b = mycc.t1
        t2aa, t2ab, t2bb = mycc.t2

        # Flattened singles for ML
        cc_t1 = np.concatenate([t1a.ravel(), t1b.ravel()]).astype(np.float32)
        out["cc_t1"] = cc_t1

        if return_amplitudes:
            out.update(
                cc_t1_full=cc_t1,  # keep same key as RCCSD; still provide blocks below
                cc_t2_full=np.concatenate(
                    [t2aa.ravel(), t2ab.ravel(), t2bb.ravel()]
                ).astype(np.float32),
                eri=eri,
                full_overlap=full_overlap,
                kinetic=kinetic,
                nuc_potential=full_nuc_potential,
                # Spin-resolved blocks for maximum fidelity:
                # uccsd_blocks=dict(
                #     t1a=t1a.astype(np.float32),
                #     t1b=t1b.astype(np.float32),
                #     t2aa=t2aa.astype(np.float32),
                #     t2ab=t2ab.astype(np.float32),
                #     t2bb=t2bb.astype(np.float32),
                # )
            )
    else:
        # Restricted CCSD: single arrays
        cc_t1_full = mycc.t1.astype(np.float32)
        cc_t2_full = mycc.t2.astype(np.float32)
        out["cc_t1"] = cc_t1_full.reshape(-1).astype(np.float32)

        if return_amplitudes:
            out.update(
                cc_t1_full=cc_t1_full,
                cc_t2_full=cc_t2_full,
                eri=eri,
                full_overlap=full_overlap,
                kinetic=kinetic,
                nuc_potential=full_nuc_potential,
            )

    return out
