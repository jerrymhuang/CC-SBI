import numpy as np
from utils.molecule_utils import build_pyscf_molecule


def water_cluster(num_molecules=2, base_radius=3.0, seed=None):
    rng = np.random.default_rng(seed)
    atoms = []

    # Scale radius based on num_molecules to avoid crowding
    radius = base_radius * np.sqrt(num_molecules)
    print("radius:", radius)

    for _ in range(num_molecules):
        # Place oxygen atoms
        r = radius * np.sqrt(rng.random())
        theta = rng.random() * 2 * np.pi
        ox, oy = r * np.cos(theta), r * np.sin(theta)
        oz = 0.0

        # Orient randomly
        phi = rng.random() * 2 * np.pi

        # Vary bond distances and bond angles
        r_oh = 0.9575 + rng.normal(0.0, 0.05)
        angle_hoh = 104.5 + rng.normal(0.0, 15.0)  # degrees
        alpha = np.radians(angle_hoh / 2)

        # Position H-atoms w.r.t. O-atoms
        h1x = r_oh * np.sin(alpha)
        h1y = -r_oh * np.cos(alpha)
        h2x, h2y = -h1x, h1y

        # rotate in xy-plane
        rot = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi), np.cos(phi)]])
        h1 = rot @ np.array([h1x, h1y])
        h2 = rot @ np.array([h2x, h2y])

        # Gather info (as strings)
        atoms.append(("O", [ox, oy, oz]))
        atoms.append(("H", [ox + h1[0], oy + h1[1], oz]))
        atoms.append(("H", [ox + h2[0], oy + h2[1], oz]))

    return atoms


if __name__ == "__main__":
    cluster = water_cluster()
    print(cluster)

    molecules = build_pyscf_molecule(pyscf_atoms=cluster)
    print(molecules.atom)