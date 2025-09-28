import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def h_atom(
    center: Sequence[float] | None = None,
    perturb: bool = True,
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
    h = np.zeros(3, dtype=np.float32)

    if perturb:
        h = h + np.random.normal(size=(3,))

    atoms = [("H", h)]

    print(atoms)

    if center is not None:
        c = np.asarray(center, dtype=np.float32)
        atoms = [(a, (np.asarray(r, float) + c).tolist()) for a, r in atoms]

    return atoms


def h_mole(
    bond_distance: float = 0.74,
    bond_std: float = 0.01,
    perturb: bool = False,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2 molecule fragment centered near the origin with optional perturbations.

    Parameters
    ----------
    bond_distance : float, optional
        H-H bond length in Å, default is 0.7414.
    bond_std : float, optional
        Standard deviation for bond length sampling (Å), default is 0.05.
    perturb : bool, optional
        If True, apply random perturbations to bond length, default is False.
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

    # Precompute bond distance
    if perturb:
        bond_distance += np.random.normal(0, bond_std)

    if plane == "xy" or "xz":
        h1 = [0.0, 0.0, 0.0]
        h2 = [bond_distance, 0.0, 0.0]
    elif plane == "yz":
        h1 = [0.0, 0.0, 0.0]
        h2 = [0.0, bond_distance, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    molecule = [("H", h1), ("H", h2)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in molecule]

    return molecule


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
        species=h_mole,
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
