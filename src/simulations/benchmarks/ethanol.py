import numpy as np
from collections.abc import Sequence
from simulations.molecules import MoleculeSimulator


def ethanol():
    pass


if __name__ == "__main__":

    ethanol_simulator = MoleculeSimulator(
        species=ethanol,
        basis="sto3g",
        coord_scale=0.1,
        cache_integrals=True,
    )

    sim = ethanol_simulator.simulate(num_molecules=3)
    print("Ethene molecules:", {k: v.shape for k, v in sim.items()})