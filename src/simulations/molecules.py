import numpy as np
from collections.abc import Sequence, Callable
from utils.molecule_utils import assemble_molecules, compute_ccsd
from pyscf import gto
from tqdm import tqdm

class MoleculeSimulator:
    """Simulate closed-shell molecular systems and compute CCSD properties for ML datasets."""

    def __init__(
        self,
        species: str | list[tuple[str, list[float]]] | list[tuple[str, np.ndarray]] | dict[str, list[float]] | Callable,
        species_kwargs: dict | None = None,
        basis: str = "sto3g",
        verbose: int = 0,
        return_amplitudes: bool = False,
        coord_scale: float | None = 0.1,
        cache_integrals: bool = False,
    ):
        """
        Initialize the molecule simulator for closed-shell molecules.
        """
        self.species = species
        self.species_kwargs = species_kwargs or {}
        self.basis = basis
        self.verbose = verbose
        self.return_amplitudes = return_amplitudes
        self.coord_scale = coord_scale
        self.cache_integrals = cache_integrals
        self._integral_cache = {}
        if cache_integrals:
            self._cache_base_integrals()

    def simulate(
        self,
        species: str
        | list[tuple[str, Sequence[float]]]
        | list[tuple[str, np.ndarray]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a closed-shell system and compute CCSD properties.
        """
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
            species=species,
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
        samples: int,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
        show_progress: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Generate multiple CCSD samples for closed-shell molecules.
        """
        if samples < 1:
            raise ValueError("samples must be a positive integer")
        all_data = []
        for _ in tqdm(range(samples), desc="Generating samples", disable=not show_progress):
            sample = self.simulate(species, species_kwargs)
            all_data.append(sample)

        combined_data = {}
        if all_data:
            for key in all_data[0].keys():
                combined_data[key] = np.stack([d[key] for d in all_data], axis=0)

        return combined_data

    def _cache_base_integrals(self):
        """Cache integrals for the base closed-shell molecule."""
        base_molecule = assemble_molecules(
            species=self.species,
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
        if n_electrons % 2 != 0:
            raise ValueError("Only closed-shell molecules are supported (even number of electrons required)")
        mol.spin = 0
        mol.build()

        if self.cache_integrals:
            self._integral_cache = {
                "kinetic": mol.intor("int1e_kin").astype(np.float32),
                "nuc_potential": mol.intor("int1e_nuc").astype(np.float32),
                "eri": mol.intor("int2e_sph", aosym=1).astype(np.float32),
                "full_overlap": mol.intor("int1e_ovlp").astype(np.float32),
            }