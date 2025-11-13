import numpy as np
from collections.abc import Sequence, Callable
from utils.molecule_utils import (
    build_molecule_geometries,
    build_pyscf_molecule,
    compute_integrals,
    compute_hartree_fock,
    compute_cc,
    compute_coordinates
)
from tqdm import tqdm

class MoleculeSimulator:
    """Simulate closed-shell molecular systems and compute CCSD properties for ML datasets."""

    def __init__(
        self,
        molecule_fun: str | list[tuple[str, list[float]]] | dict[str, list[float]] | Callable,
        molecule_config: Callable | None = None,
        molecule_kwargs: dict | None = None,
        basis: str = "cc-pVTZ",
        unit: str = "bohr",
        verbose: int = 0,
        cartesian: bool = False,
        charge: int = 0,
        coord_scale: float | None = 0.1,
        cache_integrals: bool = False,
    ):
        """
        Initialize the molecule simulator for closed-shell molecules.

        Args:
            molecule_fun: Specification of the molecule (string, list of (atom, coord), dict, or callable).
            molecule_kwargs: Additional arguments for molecule_fun if callable.
            basis: Basis set for quantum chemistry calculations (default: "sto3g").
            verbose: Verbosity level for PySCF (default: 0).
            coord_scale: Scaling factor for coordinates (default: 0.1).
            cache_integrals: Whether to cache integrals for the base molecule (default: False).
        """
        self.molecule_fun = molecule_fun
        self.molecule_config = molecule_config
        self.molecule_kwargs = molecule_kwargs or {}
        self.unit = unit
        self.basis = basis
        self.charge = charge
        self.verbose = verbose
        self.cartesian = cartesian
        self.coord_scale = coord_scale
        self.cache_integrals = cache_integrals
        self._integral_cache = {}
        if cache_integrals:
            self._cache_base_integrals()

    def simulate(
        self,
        molecule_fun: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        molecule_kwargs: dict | None = None,
        molecule_config: dict | None = None,
        include_configs: bool = False,
        include_geometries: bool = False,
        include_integrals: bool = False,
        include_hartree_fock: bool = False,
        include_cc: bool = False,
        include_coordinates: bool = False,
        include_all: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a closed-shell system and compute CCSD properties.
        """
        if molecule_fun is None:
            molecule_fun = self.molecule_fun
            molecule_kwargs = molecule_kwargs or self.molecule_kwargs.copy()
            molecule_config = molecule_config or self.molecule_config.copy()
            kwargs = molecule_kwargs | molecule_config
        else:
            if not (isinstance(molecule_fun, (str, list, dict)) or callable(molecule_fun)):
                raise TypeError(
                    "molecule_fun must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
                )
            kwargs = molecule_kwargs | molecule_config or {}

        sim_data = {}

        # Assemble molecule using updated utility function
        geometries = build_molecule_geometries(
            molecule_fun=molecule_fun,
            molecule_kwargs=kwargs,
        )

        # Build PySCF molecule
        pyscf_molecule = build_pyscf_molecule(
            atoms=geometries["atoms"],
            positions=geometries["positions"],
            unit=self.unit,
            basis=self.basis,
            cartesian=self.cartesian,
            charge=self.charge,
            verbose=self.verbose
        )

        if include_all:
            include_integrals = True
            include_hartree_fock = True
            include_cc = True
            include_coordinates = True
            include_geometries = True
            include_configs = True

        # Initialize sim data
        if include_geometries:
            sim_data = sim_data | geometries

        if include_integrals:
            integrals = compute_integrals(pyscf_molecule)
            sim_data = sim_data | integrals

        if include_hartree_fock:
            rhf = compute_hartree_fock(pyscf_molecule)
            sim_data = sim_data | rhf

        if include_cc:
            cc = compute_cc(pyscf_molecule)
            sim_data = sim_data | cc

        if include_coordinates:
            coordinates = compute_coordinates(**geometries, coordinate_scale=self.coord_scale)
            sim_data = sim_data | {"coordinates": coordinates}

        if include_configs:
            sim_data = sim_data | molecule_config

        return sim_data

    def sample(
        self,
        batch_size: int,
        molecule_config: Callable | dict | None = None,
        molecule_kwargs: dict | None = None,
        include_kwargs: dict | None = None,
        show_progress: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Generate multiple CCSD samples for closed-shell molecules.
        """
        if batch_size < 1:
            raise ValueError("samples must be a positive integer")
        all_data = []
        for _ in tqdm(range(batch_size), desc="Generating samples", disable=not show_progress):
            # Allow customized priors for molecules
            config = molecule_config()
            sample = self.simulate(
                molecule_fun=self.molecule_fun,
                molecule_config=config,
                molecule_kwargs=molecule_kwargs,
                **include_kwargs
            )
            all_data.append(sample)

        combined_data = {}
        if all_data:
            for key in all_data[0].keys():
                combined_data[key] = np.stack([d[key] for d in all_data], axis=0)
                if combined_data[key].ndim == 1:
                    combined_data[key] = combined_data[key].reshape(-1, 1)

        return combined_data

    def _cache_base_integrals(self):
        """Cache integrals for the base closed-shell molecule."""
        # Assemble base molecule
        base_molecule = build_molecule_geometries(
            molecule_fun=self.molecule_fun,
            molecule_kwargs=self.molecule_kwargs,
        )

        # Build PySCF molecule using updated utility
        mol = build_pyscf_molecule(
            atoms=base_molecule["atoms"],
            positions=base_molecule["positions"],
            basis=self.basis,
            verbose=self.verbose,
            charge=0,
            unit="bohr",  # Consistent with compute_ccsd default
            cartesian=False,
        )

        # Compute and cache integrals
        if self.cache_integrals:
            self._integral_cache = compute_integrals(mol, full_matrices=False)
