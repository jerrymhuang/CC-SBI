import numpy as np
from collections.abc import Iterable, Sequence
from pyscf import gto, scf, cc


def make_water(
    bond_length: float = 0.9572,
    angle_deg: float = 104.5,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2O fragment centered near the origin.

    Parameters
    ----------
    bond_length : float
        O–H bond length in Å.
    angle_deg : float
        H–O–H angle in degrees.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that oxygen is at ``center``.
    plane : {"xy", "xz", "yz"}
        Plane in which to place the molecule. Useful if you want to
        stack fragments without overlapping in z.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    theta = np.deg2rad(angle_deg)

    # Place oxygen at the origin; two hydrogens symmetric about +x axis.
    # Default orientation: molecule lies in the chosen plane, with the bisector
    # of the H–O–H angle along +x.
    h_offset = bond_length * np.sin(theta / 2)
    x = bond_length * np.cos(theta / 2)

    if plane == "xy":
        h1 = [x, +h_offset, 0.0]
        h2 = [x, -h_offset, 0.0]
        o = [0.0, 0.0, 0.0]
    elif plane == "xz":
        h1 = [x, 0.0, +h_offset]
        h2 = [x, 0.0, -h_offset]
        o = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h1 = [0.0, x, +h_offset]
        h2 = [0.0, x, -h_offset]
        o = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("O", o), ("H", h1), ("H", h2)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        frag = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


def make_molecules(
    N: int = 1,
    bond_distance: float = 2.0,
    species: str | list[tuple[str, Sequence[float]]] | dict[str, Sequence[float]] = "H",
    perturb: bool = True,
    seed: int | None = None,
):
    """G
    enerate a set of atoms (in chains) or molecular fragments.

    This function now **natively supports 'H2O'** as a ``species`` value.

    Parameters
    ----------
    N : int
        Number of repeats of the base unit (atom or fragment).
    bond_distance : float
        Spacing between consecutive units along +x (Å).
    species : str or list of (atom, coord) or dict
        - ``"H"`` or any atomic symbol: repeats that single atom.
        - ``"H2O"``: uses a canonical water fragment from :func:`make_water`.
        - list of ``(atom, [x, y, z])``: the fragment you provide will be tiled.
        - dict ``{atom: [x, y, z]}``: same as list but unordered.
    perturb : bool
        If True, adds small random displacements (±0.125 Å) per unit to break symmetry.
    seed : int, optional
        RNG seed.

    Returns
    -------
    dict
        ``{"species": np.ndarray[str], "pos": np.ndarray[float32]}`` with shape (N_atoms, 3).
    """
    rng = np.random.default_rng(seed)

    # Normalize a fragment description to a list[(atom, coord)] at the origin.
    if isinstance(species, str):
        if species.upper() == "H2O":
            base = make_water()
        else:
            base = [(species, [0.0, 0.0, 0.0])]
    elif isinstance(species, dict):
        base = list(species.items())
    elif isinstance(species, list):
        base = species
    else:
        raise TypeError("species must be a string, list[(atom, coord)], or dict{atom: coord}")

    atoms: list[str] = []
    positions: list[list[float]] = []

    for i in range(N):
        offset = np.array([i * bond_distance, 0.0, 0.0], dtype=float)
        noise = (0.25 * rng.random(3) - 0.125) if perturb else np.zeros(3)
        for atom, coord in base:
            xyz = np.asarray(coord, dtype=float) + offset + noise
            atoms.append(str(atom))
            positions.append(xyz.tolist())

    return {"species": np.array(atoms, dtype=object), "pos": np.array(positions, dtype=np.float32)}


def compute_ccsd(
    species: Iterable[str] | list[tuple[str, Sequence[float]]] | None = None,
    pos: np.ndarray | None = None,
    basis: str = "sto3g",
    verbose: int = 0,
    return_amplitudes: bool = False,
):
    """
    Run RHF → CCSD and return ML-friendly features.

    Parameters
    ----------
    species : iterable of str or list of (atom, coord), optional
        If ``None``, defaults to a single H2O molecule.
        If a list of ``(atom, coord)``, ``pos`` is ignored and coords are taken
        from the list.
    pos : (N, 3) array-like, optional
        Cartesian coordinates (Å) for ``species`` if ``species`` is an array of symbols.
    basis : str
        Basis set name (e.g., "sto3g", "6-31g").
    verbose : int
        PySCF verbosity level.
    return_amplitudes : bool
        If True, include full ``t1`` and ``t2`` arrays in the output dict.

    Returns
    -------
    dict
        Keys include ``V_mat`` (lower-triangular nuclear attraction), ``s_flat`` (overlap),
        ``atoms_flat`` (flattened coordinates /10), ``T1_flat`` (flattened t1). If
        ``return_amplitudes`` is True, includes ``t1`` and ``t2`` as np.float32.
    """
    # Default: a single water molecule
    if species is None and pos is None:
        fragment = make_water()
        species = [a for a, _ in fragment]
        pos = np.array([r for _, r in fragment], dtype=np.float32)

    # Allow the user to pass a fragment directly as list[(atom, coord)]
    if isinstance(species, list) and len(species) > 0 and isinstance(species[0], tuple):
        pyscf_atoms = [(str(a), list(map(float, r))) for a, r in species]
    else:
        sp = np.asarray(species, dtype=object).reshape(-1)
        xyz = np.asarray(pos, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != sp.shape[0]:
            raise ValueError(
                f"pos must have shape (N_atoms, 3) and match species length; got {xyz.shape} vs {sp.shape[0]}"
            )
        pyscf_atoms = [(str(sp[i]), xyz[i].tolist()) for i in range(sp.shape[0])]

    # Build molecule
    mol = gto.Mole()
    mol.atom = pyscf_atoms
    mol.basis = basis
    mol.verbose = verbose
    mol.charge = 0
    n_electrons = sum(gto.charge(a[0]) for a in pyscf_atoms)
    mol.spin = n_electrons % 2  # closed shell if even
    mol.build()

    # One- and two-electron integrals
    T = mol.intor("int1e_kin").astype(np.float32)
    V_e_nuc = mol.intor("int1e_nuc").astype(np.float32)
    eri = mol.intor("int2e_sph", aosym=1).astype(np.float32)
    S = mol.intor("int1e_ovlp").astype(np.float32)

    # SCF and CCSD
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()

    atoms_mat = np.array([coord for _, coord in pyscf_atoms], dtype=np.float32)
    t1 = mycc.t1.astype(np.float32)
    t2 = mycc.t2.astype(np.float32)

    # Lower-triangular packing for ML
    B = V_e_nuc.shape[0]
    x_idx, y_idx = np.tril_indices(B)
    V_mat = V_e_nuc[x_idx, y_idx]
    s_flat = S[x_idx, y_idx]
    atoms_flat = (atoms_mat.reshape(-1) / 10.0).astype(np.float32)
    t1_flat = t1.reshape(-1)

    out = dict(V_mat=V_mat, s_flat=s_flat, atoms_flat=atoms_flat, T1_flat=t1_flat)
    if return_amplitudes:
        out.update(t1=t1, t2=t2, eri=eri, S=S, T=T, V_e_nuc=V_e_nuc)
    return out


if __name__ == "__main__":
    # Quick self-test: single H2O and a chain of 3 waters
    w1 = compute_ccsd()  # default: one water
    chain = make_molecules(N=3, bond_distance=2.8, species="H2O", seed=0)
    w3 = compute_ccsd(species=chain["species"], pos=chain["pos"], basis="sto3g", verbose=0)
    print({k: v.shape for k, v in w3.items()})
