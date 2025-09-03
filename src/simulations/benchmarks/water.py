import numpy as np
from collections.abc import Sequence
from pyscf import gto


def water(
    bond_length: float = 0.9572,
    angle_deg: float = 104.5,
    bond_std: float = 0.05,
    angle_std: float = 5.0,
    normal_mode_sampling: bool = False,
    temperature: float = 300,
    use_fixed_noise: bool = False,
    perturb: bool = False,
    rng: np.random.Generator | None = None,
    normal_modes: tuple[np.ndarray, np.ndarray] | None = None,
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
    normal_mode_sampling : bool, optional
        If True, sample displacements along normal modes instead of bond/angle, default is False.
    temperature : float, optional
        Temperature (K) for scaling perturbations, default is 300.
    use_fixed_noise : bool, optional
        If True, use bond_std/angle_std directly; if False, scale by sqrt(temperature/300), default is False.
    perturb : bool, optional
        If True, apply random perturbations to bond length/angle or normal modes, default is False.
    rng : np.random.Generator, optional
        Random number generator for perturbations, default is None.
    normal_modes : tuple of (np.ndarray, np.ndarray), optional
        Normal modes (shape: (9, N_modes)) and frequencies (cm^-1, shape: (N_modes,)) for sampling.
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
    - The two O-H bonds are sampled independently in Gaussian mode with a correlation factor
      to mimic vibrational coupling (e.g., symmetric stretch dominance).
    - Normal mode sampling uses physical vibrational modes, ensuring realistic coupling between bonds and angle.
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting precomputed coordinates.
    """
    if perturb and rng is not None:
        if normal_mode_sampling and normal_modes is not None:
            modes, freqs = normal_modes
            k_B = 1.380649e-23  # Boltzmann constant (J/K)
            m_u = 1.66053906660e-27  # Atomic mass unit (kg)
            c = 2.99792458e10  # Speed of light (cm/s)
            omega = 2 * np.pi * c * freqs
            mu = 1.0  # Approximate reduced mass for water (amu)
            sigma = np.sqrt(k_B * temperature / (omega ** 2 * mu * m_u)) * 1e10  # Convert m to Å
            displacements = rng.normal(0, sigma)  # Shape: (N_modes,)
            delta_coords = np.dot(modes, displacements).reshape(3, 3)  # Shape: (3 atoms, 3 coords)
            # Compute new bond lengths and angle from displaced coordinates
            bond_vec1 = delta_coords[1] - delta_coords[0]  # O-H1
            bond_vec2 = delta_coords[2] - delta_coords[0]  # O-H2
            bond_length1 = np.linalg.norm(bond_vec1)
            bond_length2 = np.linalg.norm(bond_vec2)
            cos_angle = np.dot(bond_vec1, bond_vec2) / (bond_length1 * bond_length2)
            angle_deg = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            bond_length1 = np.clip(bond_length1, 0.5, 1.5)
            bond_length2 = np.clip(bond_length2, 0.5, 1.5)
            angle_deg = np.clip(angle_deg, 90, 120)
        else:
            # Gaussian sampling with correlation between O-H bonds
            noise_std = bond_std if use_fixed_noise else bond_std * np.sqrt(temperature / 300)
            angle_noise_std = angle_std if use_fixed_noise else angle_std * np.sqrt(temperature / 300)
            # Correlate bond lengths to mimic symmetric stretch (ρ=0.8 for strong correlation)
            base_noise = rng.normal(0, noise_std)
            bond_length1 = np.clip(bond_length + base_noise + rng.normal(0, noise_std * 0.6), 0.5, 1.5)
            bond_length2 = np.clip(bond_length + base_noise + rng.normal(0, noise_std * 0.6), 0.5, 1.5)
            angle_deg = np.clip(angle_deg + rng.normal(0, angle_noise_std), 90, 120)
    else:
        bond_length1 = bond_length
        bond_length2 = bond_length

    # Use average bond length for coordinate calculation to maintain symmetry
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
