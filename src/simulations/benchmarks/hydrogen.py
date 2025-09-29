import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def h_atom(
    center: Sequence[float] | None = None,
    perturb: bool = True,
    noise: float = 0.05
) -> list[tuple[str, list[float]]]:
    """
    Return a single hydrogen atom centered near the origin.
    """
    # Base position
    h = np.zeros(3, dtype=np.float32)

    if perturb:
        h = h + np.random.normal(0, noise, 3)

    atoms = [("H", h.tolist())]

    if center is not None:
        c = np.asarray(center, dtype=np.float32)
        atoms = [(a, (np.asarray(r, float) + c).tolist()) for a, r in atoms]

    return atoms


def h_mole(
    bond_distance: float = 0.74,
    perturb: bool = False,
    noise: float = 0.05,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2 molecule fragment centered near the origin with optional perturbations.
    """

    # Precompute bond distance
    if perturb:
        bond_distance += np.random.normal(0, noise)

    if plane in ["xy", "xz"]:
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
        noise=0.1,
        coord_scale=0.1,
        verbose=0,
    )

    h2_simulator = MoleculeSimulator(
        species=h_mole,
        distance=2.8,
        basis="sto3g",
        perturb=True,
        noise=0.1,
        coord_scale=0.1,
        cache_integrals=True,
    )

    h_atoms_sim = h_atoms_simulator.simulate(num_molecules=7)
    h2_sim = h2_simulator.simulate(num_molecules=3)

    print("H atoms (as chain):", {k: v.shape for k, v in h_atoms_sim.items()})
    print(h_atoms_sim["coordinates"])
    print("H2 molecules:", {k: v.shape for k, v in h2_sim.items()})
    print(h2_sim["coordinates"])
