import numpy as np
from collections.abc import Sequence


def hf(
    bond_length: float = 0.917,
    bond_std: float = 0.05,
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
    Return an HF fragment centered near the origin with optional perturbations.

    Parameters
    ----------
    bond_length : float, optional
        H-F bond length in Å, default is 0.917.
    bond_std : float, optional
        Standard deviation for bond length sampling (Å) at 300 K, default is 0.05.
    normal_mode_sampling : bool, optional
        If True, sample displacements along normal modes, default is False.
    temperature : float, optional
        Temperature (K) for scaling perturbations, default is 300.
    use_fixed_noise : bool, optional
        If True, use bond_std directly; if False, scale by sqrt(temperature/300), default is False.
    perturb : bool, optional
        If True, apply random perturbations to bond length, default is False.
    rng : np.random.Generator, optional
        Random number generator for perturbations, default is None.
    normal_modes : tuple of (np.ndarray, np.ndarray), optional
        Normal modes (shape: (6, N_modes)) and frequencies (cm^-1, shape: (N_modes,)) for sampling.
    center : Sequence[float], optional
        If provided, translate the fragment so fluorine is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the HF molecule.

    Notes
    -----
    - HF is diatomic with one vibrational mode (H-F stretch). Normal mode sampling adjusts the bond length directly.
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting precomputed coordinates.
    """
    if perturb and rng is not None:
        if normal_mode_sampling and normal_modes is not None:
            modes, freqs = normal_modes
            k_B = 1.380649e-23  # Boltzmann constant (J/K)
            m_u = 1.66053906660e-27  # Atomic mass unit (kg)
            c = 2.99792458e10  # Speed of light (cm/s)
            omega = 2 * np.pi * c * freqs
            mu = 0.95  # Reduced mass for HF (amu, m_H * m_F / (m_H + m_F) ≈ 0.95)
            sigma = np.sqrt(k_B * temperature / (omega ** 2 * mu * m_u)) * 1e10  # Convert m to Å
            displacement = rng.normal(0, sigma[0])  # Single mode (H-F stretch)
            # Mode is along bond vector; adjust bond length
            bond_length = np.clip(bond_length + displacement, 0.5, 1.5)
        else:
            # Gaussian sampling for bond length
            noise_std = bond_std if use_fixed_noise else bond_std * np.sqrt(temperature / 300)
            bond_length = np.clip(bond_length + rng.normal(0, noise_std), 0.5, 1.5)

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
    molecule = fragment.copy()

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return molecule
