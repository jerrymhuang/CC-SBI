import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def methane(
    bond_length: float = 1.087,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return a CH4 fragment centered near the origin.

    Parameters
    ----------
    bond_length : float
        C–H bond length in Å.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that carbon is at ``center``.
    plane : {"xy", "xz", "yz"}
        Orientation of the molecule by permuting coordinates. Useful if you want to
        stack fragments without overlapping.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    s = bond_length / np.sqrt(3)
    h1 = s * np.array([1.0, 1.0, 1.0])
    h2 = s * np.array([1.0, -1.0, -1.0])
    h3 = s * np.array([-1.0, 1.0, -1.0])
    h4 = s * np.array([-1.0, -1.0, 1.0])
    c = np.array([0.0, 0.0, 0.0])

    # Permute coordinates based on plane
    def permute(coord: np.ndarray) -> list[float]:
        x, y, z = coord
        if plane == "xy":
            return [x, y, z]
        elif plane == "xz":
            return [x, z, y]
        elif plane == "yz":
            return [y, z, x]
        else:
            raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [
        ("C", permute(c)),
        ("H", permute(h1)),
        ("H", permute(h2)),
        ("H", permute(h3)),
        ("H", permute(h4)),
    ]

    if center is not None:
        cen = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + cen).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of CH4 with 3 molecules
    simulator = MoleculeSimulator(
        species=methane,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=3)
    print("Methane chain:", {k: v.shape for k, v in sim.items()})
