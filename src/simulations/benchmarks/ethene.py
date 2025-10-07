import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def ethene(
    cc_bond_distance: float = 1.339,
    ch_bond_distance: float = 1.085,
    hch_angle: float = 117.4,
    twist_angle: float = 0.0,
    cc_bond_noise: float = 0.05,
    ch_bond_noise: float = 0.05,
    angle_noise: float = 5.0,
    perturb: bool = True,
    equal_bonds: bool = True,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return a C2H4 molecule centered near the origin with optional perturbations.

    Parameters
    ----------
    cc_bond_distance : float
        C=C bond length in Å (default: 1.339).
    ch_bond_distance : float
        C-H bond length in Å (default: 1.085).
    hch_angle : float
        H-C-H angle in degrees (default: 117.4).
    twist_angle : float
        Dihedral twist angle between the two CH2 planes in degrees (default: 0.0).
    cc_bond_noise : float
        Standard deviation of noise for C=C bond in Å (default: 0.05).
    ch_bond_noise : float
        Standard deviation of noise for C-H bonds in Å (default: 0.05).
    angle_noise : float
        Standard deviation of noise for H-C-H angle in degrees (default: 5.0).
    perturb : bool
        If True, apply random perturbations to bond lengths and H-C-H angle (default: True).
    equal_bonds : bool
        If True, all C-H bonds share the same perturbed length; if False, each C-H bond
        is perturbed independently (default: True).
    center : sequence of 3 floats, optional
        If provided, translate the molecule so that the midpoint of C=C is near ``center``.
    plane : {"xy", "xz", "yz"}
        Plane in which to place the untwisted molecule. Useful if you want to
        stack molecules without overlapping in z.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates are in Å.
    """
    # Apply perturbations if requested
    if perturb:
        cc_bond = cc_bond_distance + np.random.normal(0, cc_bond_noise)
        if equal_bonds:
            ch_bond = ch_bond_distance + np.random.normal(0, ch_bond_noise)
            ch_bonds = [ch_bond] * 4  # Same length for all four C-H bonds
        else:
            ch_bonds = [ch_bond_distance + np.random.normal(0, ch_bond_noise) for _ in range(4)]  # Independent lengths
        theta = np.deg2rad(hch_angle + np.random.normal(0, angle_noise))
    else:
        cc_bond = cc_bond_distance
        ch_bonds = [ch_bond_distance] * 4
        theta = np.deg2rad(hch_angle)

    phi = np.deg2rad(twist_angle)

    # Use the first C-H bond length for h11 and h12, second for h21, third for h22
    h_offset = [ch_bonds[i] * np.sin(theta / 2) for i in range(4)]
    x_ch = [ch_bonds[i] * np.cos(theta / 2) for i in range(4)]

    if plane == "xy":
        offset_idx = 1  # y offset
        perp_idx = 2  # z perp
        c1 = [0.0, 0.0, 0.0]
        c2 = [cc_bond, 0.0, 0.0]
        h11 = [x_ch[0], h_offset[0], 0.0]
        h12 = [x_ch[1], -h_offset[1], 0.0]
        h21 = [cc_bond - x_ch[2], h_offset[2], 0.0]
        h22 = [cc_bond - x_ch[3], -h_offset[3], 0.0]
    elif plane == "xz":
        offset_idx = 2
        perp_idx = 1
        c1 = [0.0, 0.0, 0.0]
        c2 = [cc_bond, 0.0, 0.0]
        h11 = [x_ch[0], 0.0, h_offset[0]]
        h12 = [x_ch[1], 0.0, -h_offset[1]]
        h21 = [cc_bond - x_ch[2], 0.0, h_offset[2]]
        h22 = [cc_bond - x_ch[3], 0.0, -h_offset[3]]
    elif plane == "yz":
        offset_idx = 2
        perp_idx = 0
        c1 = [0.0, 0.0, 0.0]
        c2 = [0.0, cc_bond, 0.0]
        h11 = [0.0, x_ch[0], h_offset[0]]
        h12 = [0.0, x_ch[1], -h_offset[1]]
        h21 = [0.0, cc_bond - x_ch[2], h_offset[2]]
        h22 = [0.0, cc_bond - x_ch[3], -h_offset[3]]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    # Apply twist to h21 and h22 around the bond axis
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    for h in [h21, h22]:
        offset = h[offset_idx]
        perp = h[perp_idx]
        h[offset_idx] = offset * cos_phi - perp * sin_phi
        h[perp_idx] = offset * sin_phi + perp * cos_phi

    molecule = [
        ("C", c1),
        ("C", c2),
        ("H", h11),
        ("H", h12),
        ("H", h21),
        ("H", h22),
    ]

    # Center at midpoint of C=C
    midpoint = np.mean(np.array([c1, c2]), axis=0)
    molecule = [(a, (np.asarray(r) - midpoint).tolist()) for a, r in molecule]

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in molecule]

    return molecule


if __name__ == "__main__":
    # Quick self-test: a chain of C2H4 with 3 molecules
    simulator = MoleculeSimulator(
        species=ethene,
        basis="cc-pVDZ",
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = simulator.simulate(num_molecules=1)
    print("Ethene molecules:", {k: v.shape for k, v in sim.items()})
