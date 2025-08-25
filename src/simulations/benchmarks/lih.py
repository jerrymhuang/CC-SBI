import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def lih(
    bond_length: float = 1.595,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return a LiH fragment centered near the origin.

    Parameters
    ----------
    bond_length : float
        Li–H bond length in Å.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that lithium is at ``center``.
    plane : {"xy", "xz", "yz"}
        Direction to orient the bond axis. For "xy", bond along x; for "xz", along x; for "yz", along y.
        Useful if you want to stack fragments without overlapping.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    if plane == "xy":
        h = [bond_length, 0.0, 0.0]
        li = [0.0, 0.0, 0.0]
    elif plane == "xz":
        h = [bond_length, 0.0, 0.0]
        li = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h = [0.0, bond_length, 0.0]
        li = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("Li", li), ("H", h)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of LiH with 3 molecules
    simulator = MoleculeSimulator(
        species=lih,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=3)
    print("LiH molecules:", {k: v.shape for k, v in sim.items()})
