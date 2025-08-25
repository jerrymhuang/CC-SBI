import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def hf(
    bond_length: float = 0.917,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an HF fragment centered near the origin.

    Parameters
    ----------
    bond_length : float
        F–H bond length in Å.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that fluorine is at ``center``.
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
        f = [0.0, 0.0, 0.0]
    elif plane == "xz":
        h = [bond_length, 0.0, 0.0]
        f = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h = [0.0, bond_length, 0.0]
        f = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("F", f), ("H", h)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of HF with 3 molecules
    simulator = MoleculeSimulator(
        species=hf,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=3)
    print("HF molecules:", {k: v.shape for k, v in sim.items()})
