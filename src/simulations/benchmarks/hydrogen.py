import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def h_atom(
    normal_mode_sampling: bool = False,
    normal_modes: tuple[np.ndarray, np.ndarray] | None = None,
    center: Sequence[float] | None = None,
) -> list[tuple[str, list[float]]]:
    """
    Return a single hydrogen atom fragment centered near the origin.

    Parameters
    ----------
    normal_mode_sampling : bool, optional
        If True, attempt to sample displacements along normal modes, default is False.
        Ignored with a warning, as a single atom has no vibrational modes.
    normal_modes : tuple of (np.ndarray, np.ndarray), optional
        Normal modes and frequencies (cm^-1) for sampling. Ignored with a warning, as a single
        atom has no vibrational modes.
    center : Sequence[float], optional
        If provided, translate the hydrogen atom to `center`.

    Returns
    -------
    list of (str, [float, float, float])
        List containing a single (atom symbol, coordinates in Å) tuple for the hydrogen atom.

    Notes
    -----
    - A single hydrogen atom has no internal degrees of freedom (bonds or angles), so
      perturbations are applied by MoleculeSimulator in chains or clusters.
    - Normal mode sampling is irrelevant for a single atom (no vibrational modes) and is ignored.
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting
      precomputed coordinates.
    """
    if normal_mode_sampling and normal_modes is not None:
        print("Warning: normal_mode_sampling is ignored for a single hydrogen atom (no vibrational modes).")

    # Single hydrogen atom at origin
    h = [0.0, 0.0, 0.0]
    fragment = [("H", h)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


def h2(
    bond_length: float = 0.7414,
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
    Return an H2 molecule fragment centered near the origin with optional perturbations.

    Parameters
    ----------
    bond_length : float, optional
        H-H bond length in Å, default is 0.7414.
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
        If provided, translate the fragment so the first hydrogen is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the H2 molecule.

    Notes
    -----
    - H2 is diatomic with one vibrational mode (H-H stretch). Normal mode sampling adjusts the bond length directly.
    - Future versions may support external geometries (e.g., from ASE/OpenMM) by accepting precomputed coordinates.
    """
    if perturb and rng is not None:
        if normal_mode_sampling and normal_modes is not None:
            modes, freqs = normal_modes
            k_B = 1.380649e-23  # Boltzmann constant (J/K)
            m_u = 1.66053906660e-27  # Atomic mass unit (kg)
            c = 2.99792458e10  # Speed of light (cm/s)
            omega = 2 * np.pi * c * freqs
            mu = 0.504  # Reduced mass for H2 (amu, m_H * m_H / (m_H + m_H) ≈ 1.00794 / 2)
            sigma = np.sqrt(k_B * temperature / (omega ** 2 * mu * m_u)) * 1e10  # Convert m to Å
            displacement = rng.normal(0, sigma[0])  # Single mode (H-H stretch)
            # Mode is along bond vector; adjust bond length
            bond_length = np.clip(bond_length + displacement, 0.5, 1.5)
        else:
            # Gaussian sampling for bond length
            noise_std = bond_std if use_fixed_noise else bond_std * np.sqrt(temperature / 300)
            bond_length = np.clip(bond_length + rng.normal(0, noise_std), 0.5, 1.5)

    if plane == "xy":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [bond_length, 0.0, 0.0]
    elif plane == "xz":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [bond_length, 0.0, 0.0]
    elif plane == "yz":
        h1 = [0.0, 0.0, 0.0]
        h2_pos = [0.0, bond_length, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    fragment = [("H", h1), ("H", h2_pos)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        fragment = [(a, (np.asarray(r, float) + c).tolist()) for a, r in fragment]

    return fragment


if __name__ == "__main__":
    # Quick self-test: a chain of H atoms and H2 molecules
    h_atoms_simulator = MoleculeSimulator(
        species=h_atom,  # one-atom fragment
        bond_distance=1.0,  # chain spacing Å
        basis="sto3g",
        perturb=True,
        position_noise=0.1,
        temperature=300,
        use_fixed_noise=False,
        normal_mode_sampling=False,  # Ignored for single atoms
        seed=123,
        coord_scale=0.1,
        verbose=0,
    )

    h2_simulator = MoleculeSimulator(
        species=h2,
        bond_distance=2.8,
        basis="sto3g",
        perturb=True,
        position_noise=0.1,
        temperature=300,
        use_fixed_noise=False,
        normal_mode_sampling=True,
        seed=42,
        coord_scale=0.1,
        cache_integrals=True,
    )

    h_atoms_sim = h_atoms_simulator.simulate(num_molecules=7)
    h2_sim = h2_simulator.simulate(num_molecules=3)

    print("H atoms (as chain):", {k: v.shape for k, v in h_atoms_sim.items()})
    print(h_atoms_sim["coordinates"])
    print("H2 molecules:", {k: v.shape for k, v in h2_sim.items()})
    print(h2_sim["coordinates"])
