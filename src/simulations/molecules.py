import numpy as np
from collections.abc import Sequence, Callable
from utils.molecule_utils import assemble_molecules, compute_ccsd
from pyscf import gto
from tqdm import tqdm


class MoleculeSimulator:
    """Simulate molecular systems and compute CCSD properties for ML datasets."""

    def __init__(
        self,
        species: str
        | list[tuple[str, list[float]]]
        | list[tuple[str, np.ndarray]]
        | dict[str, list[float]]
        | Callable = "H",
        species_kwargs: dict | None = None,
        num_molecules: int = 1,
        distance: float = 2.0,
        basis: str = "sto3g",
        perturb: bool = True,
        noise: float = 0.05,
        bond_noise: float = 0.05,
        angle_noise: float = 0.01,
        arrangement: str = "chain",
        cluster_size: float = 2.0,
        min_inter_dist: float | None = None,
        verbose: int = 0,
        return_amplitudes: bool = False,
        coord_scale: float | None = 0.1,
        cache_integrals: bool = False,
    ):
        """
        Initialize the molecule simulator.
        """
        self.species = species
        self.species_kwargs = species_kwargs or {}
        self.num_molecules = num_molecules
        self.distance = distance
        self.basis = basis
        self.perturb = perturb
        self.noise = noise
        self.bond_noise = bond_noise
        self.angle_noise = angle_noise
        self.arrangement = arrangement
        self.cluster_size = cluster_size
        self.min_inter_dist = min_inter_dist
        self.verbose = verbose
        self.return_amplitudes = return_amplitudes
        self.coord_scale = coord_scale
        self.cache_integrals = cache_integrals
        self._integral_cache = {}
        if cache_integrals:
            self._cache_base_integrals()

    def simulate(
        self,
        num_molecules: int | None = None,
        species: str
        | list[tuple[str, Sequence[float]]]
        | list[tuple[str, np.ndarray]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
        perturb: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Simulate a system and compute CCSD properties.
        """
        if num_molecules is None:
            num_molecules = self.num_molecules
        if num_molecules < 1:
            raise ValueError("N must be a positive integer")
        if species is None:
            species = self.species
            species_kwargs = species_kwargs or self.species_kwargs.copy()
        else:
            if not (isinstance(species, (str, list, dict)) or callable(species)):
                raise TypeError(
                    "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
                )
            species_kwargs = species_kwargs or {}

        molecules = assemble_molecules(
            num_molecules=num_molecules,
            distance=self.distance,
            species=species,
            perturb=perturb,
            noise=self.noise,
            bond_noise=self.bond_noise,
            angle_noise=self.angle_noise,
            species_kwargs=species_kwargs,
        )

        ccsd = compute_ccsd(
            species=molecules["species"],
            pos=molecules["pos"],
            basis=self.basis,
            verbose=self.verbose,
            return_amplitudes=self.return_amplitudes,
            coordinate_scale=self.coord_scale,
        )

        return ccsd

    def sample(
        self,
        samples: int | None = None,
        num_molecules: int | None = None,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
        show_progress: bool = True
    ) -> list[dict[str, np.ndarray]]:
        """
        Generate multiple CCSD samples.

        Parameters
        ----------
        samples : int, optional
            Number of samples to generate, default is None (uses num_molecules).
        num_molecules : int, optional
            Number of units per sample, defaults to self.num_molecules.
        species : str, list, dict, or callable, optional
            Base unit to simulate, defaults to self.species.
        species_kwargs : dict, optional
            Keyword arguments for callable species, defaults to self.species_kwargs.

        Returns
        -------
        list of dict
            List of CCSD outputs for each sample.
        """
        if samples is None:
            samples = self.num_molecules
        if samples < 1:
            raise ValueError("samples must be a positive integer")
        # Initialize lists to collect outputs
        all_data = []
        for _ in tqdm(range(samples), desc="Generating samples", disable=not show_progress):
            sample = self.simulate(num_molecules, species, species_kwargs)
            all_data.append(sample)

        # Combine samples into a single dictionary with stacked arrays
        combined_data = {}
        if all_data:
            for key in all_data[0].keys():
                combined_data[key] = np.stack([d[key] for d in all_data], axis=0)

        return combined_data

    def _cache_base_integrals(self):
        """Cache integrals for the base molecule."""
        base_molecule = assemble_molecules(
            num_molecules=1,
            distance=self.distance,
            species=self.species,
            perturb=self.perturb,
            species_kwargs=self.species_kwargs,
        )
        mol = gto.Mole()
        mol.atom = [
            (atom, pos.tolist())
            for atom, pos in zip(base_molecule["species"], base_molecule["pos"])
        ]
        mol.basis = self.basis
        mol.verbose = self.verbose
        mol.charge = 0
        n_electrons = sum(gto.charge(atom) for atom in base_molecule["species"]) - mol.charge
        mol.spin = n_electrons % 2
        mol.build()

        if self.cache_integrals:
            self._integral_cache = {
                "kinetic": mol.intor("int1e_kin").astype(np.float32),
                "nuc_potential": mol.intor("int1e_nuc").astype(np.float32),
                "eri": mol.intor("int2e_sph", aosym=1).astype(np.float32),
                "full_overlap": mol.intor("int1e_ovlp").astype(np.float32),
            }
