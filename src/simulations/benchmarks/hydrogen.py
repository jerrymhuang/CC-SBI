import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def h_atom(center: Sequence[float] | None = None) -> list[tuple[str, list[float]]]:
    """
    Return a single hydrogen atom near the origin (as a fragment).

    Parameters
    ----------
    center : sequence of 3 floats, optional
        If provided, translate H to this position (Å).

    Returns
    -------
    list of (atom, [x, y, z])
        A one-atom "fragment" compatible with MoleculeSimulator/assemble_molecules.
    """
    r = np.array([0.0, 0.0, 0.0], dtype=float)
    if center is not None:
        r = r + np.asarray(center, dtype=float)
    return [("H", r.tolist())]


def h2(
    bond_length: float = 0.74,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2 molecule fragment centered near the origin.

    Parameters
    ----------
    bond_length : float
        H–H bond length in Å.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that the first hydrogen is at ``center``.
    plane : {"xy", "xz", "yz"}
        Direction to orient the bond axis. For "xy", bond along x; for "xz", along x; for "yz", along y.
        Useful if you want to stack fragments without overlapping.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    if plane == "xy":
        h2_pos = [bond_length, 0.0, 0.0]
        h1 = [0.0, 0.0, 0.0]
    elif plane == "xz":
        h2_pos = [bond_length, 0.0, 0.0]
        h1 = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h2_pos = [0.0, bond_length, 0.0]
        h1 = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("H", h1), ("H", h2_pos)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of H atoms with 7 units
    h_atoms_simulator = MoleculeSimulator(
        species=h_atom,  # one-atom fragment
        bond_distance=1.0,  # chain spacing Å
        basis="sto3g",
        seed=123,
        coord_scale=0.1,
        verbose=0,
    )

    h2_simulator = MoleculeSimulator(
        species=h2,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    h_atoms_sim = h_atoms_simulator.simulate(num_molecules=7)
    h2_sim = h2_simulator.simulate(num_molecules=3)

    print("H atoms (as chain):", {k: v.shape for k, v in h_atoms_sim.items()})
    print(h_atoms_sim["coordinates"])
    print("H2 molecules:", {k: v.shape for k, v in h2_sim.items()})
    print(h2_sim["coordinates"])
