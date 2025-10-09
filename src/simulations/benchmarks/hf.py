import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def hf(
    bond_distance: float = 0.917,
    bond_noise: float = 0.25,
    perturb: bool = True,
    center: Sequence[float] | None = None,
    plane: str = "xy",
) -> list[tuple[str, list[float]]]:
    """
    Return an HF molecule centered near a given center with optional perturbations.

    Parameters
    ----------
    bond_distance : float, optional
        H-F bond length in Å, default is 0.917.
    bond_noise : float, optional
        Standard deviation for bond length sampling (Å), default is 0.25.
    perturb : bool, optional
        If True, apply random perturbations to bond length, default is False.
    center : Sequence[float], optional
        If provided, translate the fragment so fluorine is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the HF molecule.
    """
    #
    if perturb:
        bond_distance = np.random.normal(bond_distance, bond_noise)

    if plane in ["xy", "xz"]:
        h = [bond_distance, 0.0, 0.0]
        f = [0.0, 0.0, 0.0]
    elif plane == "yz":
        h = [0.0, bond_distance, 0.0]
        f = [0.0, 0.0, 0.0]
    else:
        raise ValueError("plane must be one of {'xy','xz','yz'}")

    molecule = [("F", f), ("H", h)]

    if center is not None:
        c = np.asarray(center, dtype=float)
        molecule = [(a, (np.asarray(r, float) + c).tolist()) for a, r in molecule]

    return molecule


if __name__ == "__main__":

    # Quick self-test: a chain of H atoms and H2 molecules
    water_simulator = MoleculeSimulator(
        species=hf,
        distance=1.0,
        basis="cc-pVDZ",
        coord_scale=0.1,
        verbose=0,
    )

    water_sim = water_simulator.simulate(num_molecules=1)

    print("HF (as chain):", {k: v.shape for k, v in water_sim.items()})
    print(water_sim["coordinates"])
