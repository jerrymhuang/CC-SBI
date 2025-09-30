import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def water(
    bond_distance: float = 0.9572,
    angle: float = 104.5,
    bond_noise: float = 0.25,
    angle_noise: float = 10,
    perturb: bool = True,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an H2O molecule centered near the origin with optional perturbations.
    """
    # Apply perturbations if requested
    if perturb:
        r1 = bond_distance + np.random.normal(0, bond_noise)
        r2 = bond_distance + np.random.normal(0, bond_noise)
        theta = np.deg2rad(angle + np.random.normal(0, angle_noise))
    else:
        r1 = r2 = bond_distance
        theta = np.deg2rad(angle)

    # Position hydrogen atoms in the specified plane
    if plane == "xy":
        o = [0.0, 0.0, 0.0]
        h1 = [r1 * np.cos(theta / 2), r1 * np.sin(theta / 2), 0.0]
        h2 = [r2 * np.cos(theta / 2), -r2 * np.sin(theta / 2), 0.0]
    elif plane == "xz":
        o = [0.0, 0.0, 0.0]
        h1 = [r1 * np.cos(theta / 2), 0.0, r1 * np.sin(theta / 2)]
        h2 = [r2 * np.cos(theta / 2), 0.0, -r2 * np.sin(theta / 2)]
    elif plane == "yz":
        o = [0.0, 0.0, 0.0]
        h1 = [0.0, r1 * np.cos(theta / 2), r1 * np.sin(theta / 2)]
        h2 = [0.0, r2 * np.cos(theta / 2), -r2 * np.sin(theta / 2)]
    else:
        raise ValueError("plane must be one of {'xy', 'xz', 'yz'}")

    molecule = [("O", o), ("H", h1), ("H", h2)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in molecule]

    return molecule


if __name__ == "__main__":

    # Quick self-test: a chain of H atoms and H2 molecules
    water_simulator = MoleculeSimulator(
        species=water,
        distance=1.0,
        basis="sto-3g",
        coord_scale=0.1,
        verbose=0,
    )

    water_sim = water_simulator.simulate(num_molecules=1)

    print("Water (as chain):", {k: v.shape for k, v in water_sim.items()})
    print(water_sim["coordinates"])