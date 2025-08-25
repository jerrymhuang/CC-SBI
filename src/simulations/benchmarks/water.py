import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def water(
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

    # Place oxygen at the origin; two hydrogen atoms symmetric about +x axis.
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
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of H2O with 3 water molecules
    simulator = MoleculeSimulator(
        species=water,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=3)
    print("Water molecules:", {k: v.shape for k, v in sim.items()})
