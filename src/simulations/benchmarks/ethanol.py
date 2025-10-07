import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def ethanol(
    cc_bond_distance: float = 1.54,  # C-C single bond
    co_bond_distance: float = 1.43,  # C-O bond
    oh_bond_distance: float = 0.96,  # O-H bond
    ch_bond_distance: float = 1.09,  # C-H bond
    hch_angle: float = 109.5,  # Tetrahedral H-C-H angle
    coh_angle: float = 108.0,  # C-O-H angle
    cc_bond_noise: float = 0.05,
    co_bond_noise: float = 0.05,
    oh_bond_noise: float = 0.05,
    ch_bond_noise: float = 0.05,
    angle_noise: float = 5.0,
    perturb: bool = True,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return a C2H5OH molecule centered near the origin with optional perturbations.

    Parameters
    ----------
    cc_bond_distance : float
        C-C bond length in Å (default: 1.54).
    co_bond_distance : float
        C-O bond length in Å (default: 1.43).
    oh_bond_distance : float
        O-H bond length in Å (default: 0.96).
    ch_bond_distance : float
        C-H bond length in Å (default: 1.09).
    hch_angle : float
        H-C-H angle in degrees (default: 109.5, tetrahedral).
    coh_angle : float
        C-O-H angle in degrees (default: 108.0).
    cc_bond_noise : float
        Standard deviation of noise for C-C bond in Å.
    co_bond_noise : float
        Standard deviation of noise for C-O bond in Å.
    oh_bond_noise : float
        Standard deviation of noise for O-H bond in Å.
    ch_bond_noise : float
        Standard deviation of noise for C-H bond in Å.
    angle_noise : float
        Standard deviation of noise for angles in degrees.
    perturb : bool
        If True, apply random perturbations to bond lengths and angles.
    center : sequence of 3 floats, optional
        If provided, translate the molecule so that the first carbon is near center.
    plane : {"xy", "xz", "yz"}
        Plane in which to place the C-C-O backbone.

    Returns
    -------
    list of (atom, [x, y, z])
        Coordinates of atoms in Å.
    """
    # Apply perturbations if requested
    if perturb:
        cc_bond = cc_bond_distance + np.random.normal(0, cc_bond_noise)
        co_bond = co_bond_distance + np.random.normal(0, co_bond_noise)
        oh_bond = oh_bond_distance + np.random.normal(0, oh_bond_noise)
        ch_bond = ch_bond_distance + np.random.normal(0, ch_bond_noise)
        hch_theta = np.deg2rad(hch_angle + np.random.normal(0, angle_noise))
        coh_theta = np.deg2rad(coh_angle + np.random.normal(0, angle_noise))
    else:
        cc_bond = cc_bond_distance
        co_bond = co_bond_distance
        oh_bond = oh_bond_distance
        ch_bond = ch_bond_distance
        hch_theta = np.deg2rad(hch_angle)
        coh_theta = np.deg2rad(coh_angle)

    # Define the backbone (C1-C2-O-H) along the x-axis initially
    c1 = [0.0, 0.0, 0.0]  # First carbon
    c2 = [cc_bond, 0.0, 0.0]  # Second carbon
    o = [cc_bond + co_bond * np.cos(coh_theta), co_bond * np.sin(coh_theta), 0.0]  # Oxygen
    h_oh = [
        o[0] + oh_bond * np.cos(coh_theta),
        o[1] + oh_bond * np.sin(coh_theta),
        0.0,
    ]  # Hydroxyl hydrogen

    # Define hydrogens on C1 and C2 (tetrahedral arrangement)
    # For simplicity, place C1 hydrogens in a plane perpendicular to C-C-O
    h_offset = ch_bond * np.sin(hch_theta / 2)
    x_ch = ch_bond * np.cos(hch_theta / 2)

    if plane == "xy":
        offset_idx = 1  # y offset
        perp_idx = 2  # z perpendicular
        h1_c1 = [x_ch, h_offset, 0.0]  # First hydrogen on C1
        h2_c1 = [x_ch, -h_offset, 0.0]  # Second hydrogen on C1
        h3_c1 = [x_ch, 0.0, h_offset]  # Third hydrogen on C1
        h_c2 = [cc_bond - x_ch, h_offset, 0.0]  # Hydrogen on C2
    elif plane == "xz":
        offset_idx = 2
        perp_idx = 1
        h1_c1 = [x_ch, 0.0, h_offset]
        h2_c1 = [x_ch, 0.0, -h_offset]
        h3_c1 = [x_ch, h_offset, 0.0]
        h_c2 = [cc_bond - x_ch, 0.0, h_offset]
    elif plane == "yz":
        offset_idx = 2
        perp_idx = 0
        h1_c1 = [0.0, x_ch, h_offset]
        h2_c1 = [0.0, x_ch, -h_offset]
        h3_c1 = [0.0, x_ch, h_offset]
        h_c2 = [0.0, cc_bond - x_ch, h_offset]
    else:
        raise ValueError("plane must be one of {'xy', 'xz', 'yz'}")

    molecule = [
        ("C", c1),  # C1 (CH3 group)
        ("C", c2),  # C2 (CH2OH group)
        ("O", o),   # Oxygen
        ("H", h_oh),  # Hydroxyl H
        ("H", h1_c1),  # H on C1
        ("H", h2_c1),  # H on C1
        ("H", h3_c1),  # H on C1
        ("H", h_c2),   # H on C2
    ]

    # Center the molecule at the first carbon (C1)
    midpoint = np.array(c1)
    molecule = [(a, (np.asarray(r) - midpoint).tolist()) for a, r in molecule]

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in molecule]

    return molecule


if __name__ == "__main__":
    # Quick self-test: a chain of C2H5OH with 3 molecules
    ethanol_simulator = MoleculeSimulator(
        species=ethanol,
        basis="sto3g",
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = ethanol_simulator.simulate(num_molecules=3)
    print("Ethanol molecules:", {k: v.shape for k, v in sim.items()})
