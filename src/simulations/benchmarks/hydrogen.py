import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def h_atom(
    center: Sequence[float] | None = None,

) -> list[tuple[str, list[float]]]:
    """
    Return a single hydrogen atom fragment centered near the origin.

    Parameters
    ----------
    center : Sequence[float], optional
        If provided, translate the hydrogen atom to `center`.

    Returns
    -------
    list of (str, [float, float, float])
        List containing a single (atom symbol, coordinates in Å) tuple for the hydrogen atom.

    Notes
    -----
    - A single hydrogen atom has no internal degrees of freedom (bonds or angles), so
      perturbations are applied by MoleculeSimulator in chains or clusters.
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting
      precomputed coordinates.
    """
    h = [0.0, 0.0, 0.0]
    fragment = [("H", h)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


def h2(
    bond_length: float = 0.7414,
    bond_std: float = 0.05,
    perturb: bool = False,
    rng: np.random.Generator | None = None,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2 molecule fragment centered near the origin with optional perturbations.

    Parameters
    ----------
    bond_length : float, optional
        H-H bond length in Å, default is 0.7414.
    bond_std : float, optional
        Standard deviation for bond length sampling (Å), default is 0.05.
    perturb : bool, optional
        If True, apply random perturbations to bond length, default is False.
    rng : np.random.Generator, optional
        Random number generator for perturbations, default is None.
    center : Sequence[float], optional
        If provided, translate the fragment so the first hydrogen is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the H2 molecule.

    Notes
    -----
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting precomputed coordinates.
    """
    if perturb and rng is not None:
        bond_length = np.clip(bond_length + rng.normal(0, bond_std), 0.5, 1.5)

    if plane == "xy":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [bond_length, 0.0, 0.0]
    elif plane == "xz":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [0.0, 0.0, bond_length]
    elif plane == "yz":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [0.0, bond_length, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("H", h1), ("H", h2_pos)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of H atoms and H2 molecules
    h_atoms_simulator = MoleculeSimulator(
        species=h_atom,
        distance=1.0,
        basis="sto3g",
        perturb=True,
        position_noise=0.1,
        coord_scale=0.1,
        verbose=0,
    )

    h2_simulator = MoleculeSimulator(
        species=h2,
        distance=2.8,
        basis="sto3g",
        perturb=True,
        position_noise=0.1,
        coord_scale=0.1,
        cache_integrals=True,
    )

    h_atoms_sim = h_atoms_simulator.simulate(num_molecules=7)
    h2_sim = h2_simulator.simulate(num_molecules=3)

    print("H atoms (as chain):", {k: v.shape for k, v in h_atoms_sim.items()})
    print(h_atoms_sim["coordinates"])
    print("H2 molecules:", {k: v.shape for k, v in h2_sim.items()})
    print(h2_sim["coordinates"])
