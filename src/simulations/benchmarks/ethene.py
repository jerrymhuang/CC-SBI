import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def ethene(
    cc_bond_length: float = 1.339,
    ch_bond_length: float = 1.085,
    hch_angle: float = 117.4,
    twist_angle: float = 0.0,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return a C2H4 fragment centered near the origin.

    Parameters
    ----------
    cc_bond_length : float
        C=C bond length in Å.
    ch_bond_length : float
        C–H bond length in Å.
    hch_angle : float
        H–C–H angle in degrees.
    twist_angle : float
        Dihedral twist angle between the two CH2 planes in degrees.
    center : sequence of 3 floats, optional
        If provided, translate the fragment so that the midpoint of C=C is near ``center``.
    plane : {"xy", "xz", "yz"}
        Plane in which to place the untwisted molecule. Useful if you want to
        stack fragments without overlapping in z.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    theta = np.deg2rad(hch_angle)
    phi = np.deg2rad(twist_angle)

    h_offset = ch_bond_length * np.sin(theta / 2)
    x_ch = ch_bond_length * np.cos(theta / 2)

    if plane == "xy":
        offset_idx = 1  # y offset
        perp_idx = 2  # z perp
        c1 = [0.0, 0.0, 0.0]
        c2 = [cc_bond_length, 0.0, 0.0]
        h11 = [x_ch, h_offset, 0.0]
        h12 = [x_ch, -h_offset, 0.0]
        h21_base = [cc_bond_length - x_ch, h_offset, 0.0]
        h22_base = [cc_bond_length - x_ch, -h_offset, 0.0]
    elif plane == "xz":
        offset_idx = 2
        perp_idx = 1
        c1 = [0.0, 0.0, 0.0]
        c2 = [cc_bond_length, 0.0, 0.0]
        h11 = [x_ch, 0.0, h_offset]
        h12 = [x_ch, 0.0, -h_offset]
        h21_base = [cc_bond_length - x_ch, 0.0, h_offset]
        h22_base = [cc_bond_length - x_ch, 0.0, -h_offset]
    elif plane == "yz":
        offset_idx = 2
        perp_idx = 0
        c1 = [0.0, 0.0, 0.0]
        c2 = [0.0, cc_bond_length, 0.0]
        h11 = [0.0, x_ch, h_offset]
        h12 = [0.0, x_ch, -h_offset]
        h21_base = [0.0, cc_bond_length - x_ch, h_offset]
        h22_base = [0.0, cc_bond_length - x_ch, -h_offset]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    # Apply twist to h21 and h22 around the bond axis
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    for h in [h21_base, h22_base]:
        offset = h[offset_idx]
        perp = h[perp_idx]
        h[offset_idx] = offset * cos_phi - perp * sin_phi
        h[perp_idx] = offset * sin_phi + perp * cos_phi

    fragment = [
        ("C", c1),
        ("C", c2),
        ("H", h11),
        ("H", h12),
        ("H", h21_base),
        ("H", h22_base),
    ]

    # Center at midpoint of C=C
    midpoint = np.mean(np.array([c1, c2]), axis=0)
    fragment = [(a, (np.asarray(r) - midpoint).tolist()) for a, r in fragment]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of C2H4 with 3 molecules
    simulator = MoleculeSimulator(
        species=ethene,
        bond_distance=2.8,
        basis="sto3g",
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=3)
    print("Ethene molecules:", {k: v.shape for k, v in sim.items()})
