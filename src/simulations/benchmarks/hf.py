import numpy as np
from collections.abc import Sequence
from utils.molecule_utils import build_molecule_geometries
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
        If True, apply random perturbations to bond length, default is True.
    center : Sequence[float], optional
        If provided, translate the fragment so fluorine is at `center`.
    plane : {"xy", "xz", "yz"}, optional
        Plane for molecule orientation, default is "xy".

    Returns
    -------
    list of (str, [float, float, float])
        List of (atom symbol, coordinates in Å) for the HF molecule.
    """
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
    # Test the hf function
    try:
        # hf_mol = hf(perturb=False)
        # print(f"HF molecule (no perturbation): {hf_mol}")
        #
        # # Test with perturbation
        # hf_mol_perturbed = hf(perturb=True, bond_noise=0.1)
        # print(f"HF molecule (perturbed): {hf_mol_perturbed}")
        #
        # # Test with build_molecule_geometriess
        # hf_assembled = build_molecule_geometries(molecule_fun=hf, molecule_kwargs={"perturb": False})
        # print(f"Assembled HF molecule: {hf_assembled}")

        # Test with MoleculeSimulator
        hf_simulator = MoleculeSimulator(
            molecule_fun=hf,
            basis="sto-3g",
            coord_scale=0.1,
            verbose=0,
        )

        # hf_sim = hf_simulator.simulate(molecule_kwargs={"perturb": True})
        # print("HF simulation results:", {k: v.shape for k, v in hf_sim.items()})
        # print("HF coordinates:", hf_sim["coordinates"])

        hf_batch = hf_simulator.sample(num_samples=2, include_kwargs={"include_integrals": True, "include_cc": True})
        print("HF simulation results:", {k: v for k, v in hf_batch.items()})

    except ValueError as e:
        print(f"Error: {e}")