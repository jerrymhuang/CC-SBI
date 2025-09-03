import numpy as np
from collections.abc import Sequence, Iterable, Callable
from pyscf import gto, scf, cc


def assemble_molecules(
    species: str
    | list[tuple[str, Sequence[float]]]
    | dict[str, Sequence[float]]
    | Callable = "H",
    species_kwargs: dict | None = None,
    num_molecules: int = 1,
    arrangement: str = "chain",
    cluster_size: float = 2.0,
    bond_distance: float = 2.0,
    min_inter_dist: float | None = None,
    perturb: bool = True,
    position_noise: float = 0.1,
    temperature: float = 300,
    use_fixed_noise: bool = False,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Generate a set of atoms or molecular fragments in a specified arrangement.

    Parameters
    ----------
    num_molecules : int, optional
        Number of repeats of the base unit (atom or fragment), default is 1.
    bond_distance : float, optional
        Ideal center-to-center distance between consecutive units along +x (Å) for chains,
        or reference scale for cluster radius, default is 2.0. Represents approximate
        equilibrium intermolecular spacing (e.g., van der Waals or hydrogen bond distance).
    species : str, list of (atom, coord), dict, or callable, optional
        Defines the base unit to repeat:
        - str: Single atom symbol (e.g., "H").
        - list of (str, [float, float, float]): Fragment as list of (atom, coordinates).
        - dict {str: [float, float, float]}: Fragment as dictionary of atom to coordinates.
        - callable: Function returning a list of (atom, coord) tuples (e.g., a molecule generator).
        Default is "H".
    arrangement : {"chain", "cluster"}, optional
        Arrangement type: "chain" (linear along +x) or "cluster" (random 3D within a sphere).
        Default is "chain".
    cluster_size : float, optional
        Scaling factor for cluster sphere radius (radius ~ num_molecules^(1/3) * bond_distance * cluster_size / 2).
        Larger values create sparser clusters, default is 2.0.
    min_inter_dist : float, optional
        Minimum center-to-center distance between units in cluster (Å), default is bond_distance / 3.
        Ensures no unphysical overlaps, mimicking repulsive intermolecular forces.
    perturb : bool, optional
        If True, adds Gaussian translational noise to unit positions and enables internal perturbations
        (e.g., bond/angle sampling) for callable species, default is True.
    position_noise : float, optional
        Standard deviation of translational noise (Å) at 300 K if use_fixed_noise=False,
        or fixed standard deviation if use_fixed_noise=True, default is 0.1.
    temperature : float, optional
        Temperature (K) for scaling noise amplitude (if use_fixed_noise=False), default is 300.
    use_fixed_noise : bool, optional
        If True, use position_noise directly as noise standard deviation; if False, scale by sqrt(temperature/300).
        Default is False.
    seed : int, optional
        Random number generator seed for perturbations, default is None.
    species_kwargs : dict, optional
        Keyword arguments to pass to the callable `species`, default is None.

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
    ValueError
        If `arrangement` is invalid or if cluster placement fails due to overlaps.

    Notes
    -----
    To support external geometries (e.g., from ASE/OpenMM), future versions can add support for `species`
    as an external object (e.g., ase.Atoms) by checking isinstance(species, Atoms) and extracting
    symbols/positions.
    """
    rng = np.random.default_rng(seed)
    species_kwargs = species_kwargs or {}
    species_kwargs["rng"] = rng if perturb else None
    species_kwargs["perturb"] = perturb
    species_kwargs["temperature"] = temperature

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
    unit_centers = []  # Track centers for cluster mode

    # Compute noise standard deviation
    noise_std = position_noise if use_fixed_noise else position_noise * np.sqrt(temperature / 300)

    if arrangement == "chain":
        for i in range(num_molecules):
            offset = np.array([i * bond_distance, 0.0, 0.0], dtype=float)
            noise = rng.normal(0, noise_std, 3) if perturb else np.zeros(3)
            center = offset + noise
            unit_centers.append(center)
            for atom, coord in base:
                xyz = np.asarray(coord, dtype=float) + center
                atoms.append(str(atom))
                positions.append(xyz.tolist())

    elif arrangement == "cluster":
        cluster_radius = (num_molecules ** (1 / 3)) * bond_distance * cluster_size / 2
        min_inter_dist = min_inter_dist or (bond_distance / 3)

        for _ in range(num_molecules):
            attempts = 0
            while attempts < 100:
                # Uniform spherical sampling: Generate unit vector and scale
                vec = rng.normal(0, 1, 3)
                vec /= np.linalg.norm(vec)
                r = rng.uniform(0, cluster_radius ** 3) ** (1 / 3)  # For uniform density in sphere
                pos = vec * r
                if all(np.linalg.norm(pos - c) >= min_inter_dist for c in unit_centers):
                    unit_centers.append(pos)
                    noise = rng.normal(0, noise_std, 3) if perturb else np.zeros(3)
                    center = pos + noise
                    for atom, coord in base:
                        xyz = np.asarray(coord, dtype=float) + center
                        atoms.append(str(atom))
                        positions.append(xyz.tolist())
                    break
                attempts += 1
            else:
                raise ValueError(
                    "Could not place units without overlaps; try larger cluster_size or smaller min_inter_dist"
                )

    else:
        raise ValueError("arrangement must be 'chain' or 'cluster'")

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
    [Unchanged from previous version]
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

    out: dict[str, np.ndarray] = {
        "nuc_potential": nuc_potential,
        "overlap": overlap,
        "coordinates": coordinates,
    }

    if is_uccsd:
        t1a, t1b = mycc.t1
        t2aa, t2ab, t2bb = mycc.t2
        cc_t1 = np.concatenate([t1a.ravel(), t1b.ravel()]).astype(np.float32)
        out["t1"] = cc_t1
        if return_amplitudes:
            out.update(
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
        out["t1"] = cc_t1_full.reshape(-1).astype(np.float32)
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