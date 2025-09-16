import numpy as np
from collections.abc import Sequence


def water(
    bond_length: float = 0.9572,
    angle_deg: float = 104.5,
    bond_std: float = 0.05,
    angle_std: float = 5.0,
    temperature: float = 300,
    use_fixed_noise: bool = False,
    perturb: bool = False,
    rng: np.random.Generator | None = None,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2O fragment centered near the origin with optional perturbations.

    Parameters
    ----------
    bond_length : float, optional
        Equilibrium O–H bond length in Å, default is 0.9572.
    angle_deg : float, optional
        Equilibrium H–O–H angle in degrees, default is 104.5.
    bond_std : float, optional
        Standard deviation for bond length sampling (Å) at 300 K, default is 0.05.
    angle_std : float, optional
        Standard deviation for angle sampling (degrees) at 300 K, default is 5.0.
    temperature : float, optional
        Temperature (K) for scaling perturbations, default is 300.
    use_fixed_noise : bool, optional
        If True, use bond_std/angle_std directly; if False, scale by sqrt(temperature/300), default is False.
    perturb : bool, optional
        If True, apply random perturbations to bond length/angle, default is False.
    rng : np.random.Generator, optional
        Random number generator for perturbations, default is None.
    center : Sequence[float], optional
        If provided, translate the fragment so oxygen is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the water molecule.

    Notes
    -----
    - The two O-H bonds are sampled independently with a correlation factor
      to mimic vibrational coupling (e.g., symmetric stretch dominance).
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting precomputed coordinates.
    """
    if perturb and rng is not None:
        noise_std = bond_std if use_fixed_noise else bond_std * np.sqrt(temperature / 300)
        angle_noise_std = angle_std if use_fixed_noise else angle_std * np.sqrt(temperature / 300)
        base_noise = rng.normal(0, noise_std)
        bond_length1 = np.clip(bond_length + base_noise + rng.normal(0, noise_std * 0.6), 0.5, 1.5)
        bond_length2 = np.clip(bond_length + base_noise + rng.normal(0, noise_std * 0.6), 0.5, 1.5)
        angle_deg = np.clip(angle_deg + rng.normal(0, angle_noise_std), 90, 120)
    else:
        bond_length1 = bond_length
        bond_length2 = bond_length

    avg_bond_length = (bond_length1 + bond_length2) / 2
    theta = np.deg2rad(angle_deg)
    h_offset = avg_bond_length * np.sin(theta / 2)
    x = avg_bond_length * np.cos(theta / 2)

    if plane == "xy":
        h1 = [x, h_offset, 0.0]
        h2 = [x, -h_offset, 0.0]
        o = [0.0, 0.0, 0.0]
    elif plane == "xz":
        h1 = [x, 0.0, h_offset]
        h2 = [x, 0.0, -h_offset]
        o = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h1 = [0.0, x, h_offset]
        h2 = [0.0, x, -h_offset]
        o = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("O", o), ("H", h1), ("H", h2)]
    molecule = fragment.copy()

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return molecule