import numpy as np
from tqdm import tqdm
from pyscf import gto
from collections.abc import Sequence, Callable
from utils.molecule_utils import assemble_molecules, compute_ccsd


class MoleculeSimulator:
    """
    A class for simulating chains of molecules and computing CCSD properties.

    Stores configuration for molecule generation and CCSD computation, including
    the base molecule type (species), allowing reusable simulations with consistent
    settings. Supports any molecule or atom as the base unit.

    Attributes
    ----------
    species : str, list of (atom, coord), dict, or callable
        The base molecular unit (e.g., "H", list of (atom, coord), or make_water).
    species_kwargs : dict
        Keyword arguments for callable species.
    bond_distance : float
        Spacing between consecutive molecular units along +x (Å).
    basis : str
        Basis set for CCSD calculation (e.g., "sto3g", "6-31g").
    perturb : bool
        If True, adds random displacements (±0.125 Å) per unit.
    seed : int or None
        Random number generator seed for perturbations.
    verbose : int
        PySCF verbosity level.
    return_amplitudes : bool
        If True, include full CCSD t1 and t2 amplitudes in output.
    coord_scale : float or None
        Scaling factor for coordinates in output (Å). If None, no scaling.
    cache_integrals : bool
        If True, cache integrals for the base molecule to accelerate simulations.
    _integral_cache : dict
        Internally cached integrals for the base molecule (if cache_integrals=True).
    """

    def __init__(
        self,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable = "H",
        species_kwargs: dict | None = None,
        num_molecules: int = 1,
        bond_distance: float = 2.0,
        basis: str = "sto3g",
        perturb: bool = True,
        seed: int | None = None,
        verbose: int = 0,
        return_amplitudes: bool = False,
        coord_scale: float | None = 0.1,
        cache_integrals: bool = False,
    ):
        if not (isinstance(species, (str, list, dict)) or callable(species)):
            raise TypeError(
                "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
            )
        self.species = species
        self.species_kwargs = species_kwargs or {}
        self.num_molecules = num_molecules
        self.bond_distance = bond_distance
        self.basis = basis
        self.perturb = perturb
        self.seed = seed
        self.verbose = verbose
        self.return_amplitudes = return_amplitudes
        self.coord_scale = coord_scale
        self.cache_integrals = cache_integrals
        self._integral_cache = {}

        # Optionally precompute integrals for the base molecule
        if cache_integrals:
            self._cache_base_integrals()

    def simulate(
        self,
        num_molecules: int,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a molecules of N molecules and compute CCSD properties.

        Parameters
        ----------
        num_molecules : int
            Number of molecular units in the molecules.
        species : str, list of (atom, coord), dict, or callable
            Defines the base molecular unit:
            - str: Single atom symbol (e.g., "H").
            - list of (str, [float, float, float]): Fragment as list of (atom, coordinates).
            - dict {str: [float, float, float]}: Fragment as dictionary of atom to coordinates.
            - callable: Function returning a list of (atom, coord) tuples.
        species_kwargs : dict, optional
            Keyword arguments for callable `species`, default is None.

        Returns
        -------
        dict
            Dictionary with keys:
            - nuc_attr: Lower-triangular nuclear attraction matrix (Å^-1).
            - overlap: Flattened overlap matrix.
            - coords: Flattened atomic coordinates (Å, scaled by coord_scale).
            - cc_t1: Flattened CCSD t1 amplitudes.
            - If return_amplitudes=True, also includes:
              - cc_t1_full: Full t1 amplitudes.
              - cc_t2_full: Full t2 amplitudes.
              - eri: Two-electron integrals.
              - full_overlap: Full overlap matrix.
              - kinetic: Kinetic energy integrals.
              - nuc_potential: Full nuclear potential matrix.

        Raises
        ------
        TypeError
            If `species` is not a string, list, dict, or callable.
        ValueError
            If `num_molecules` is not positive.
        """
        if num_molecules is None:
            num_molecules = self.num_molecules
        if num_molecules < 1:
            raise ValueError("N must be a positive integer")
        if species is None:
            species = self.species
            species_kwargs = species_kwargs or self.species_kwargs
        else:
            if not (isinstance(species, (str, list, dict)) or callable(species)):
                raise TypeError(
                    "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
                )

        molecules = assemble_molecules(
            num_molecules=num_molecules,
            bond_distance=self.bond_distance,
            species=species,
            perturb=self.perturb,
            seed=self.seed,
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
        batch_size: int,
        num_molecules: int | None = None,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Generate a batch of CCSD simulations by repeatedly calling `simulate()`,
        and return the results stacked into a dictionary.

        Parameters
        ----------
        batch_size : int
            Number of batches to simulate.
        num_molecules : int, optional
            Number of molecular units per simulation. Defaults to the instance's value.
        species : str | list of (atom, coord) | dict | callable, optional
            Base molecular unit (overrides instance's value if provided).
        species_kwargs : dict, optional
            Extra kwargs for callable `species`.
        show_progress : bool, optional
            If True, display a tqdm progress bar.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with the same keys as a single `simulate()` call,
            with each value stacked to shape (N, *orig_shape).
        """
        if batch_size < 1:
            raise ValueError("Batch size must be a positive integer")

        iterator = range(batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling CCSD dataset", unit="sim")

        # Run first simulation to initialize structure
        first = self.simulate(
            num_molecules=num_molecules,
            species=species,
            species_kwargs=species_kwargs,
        )
        keys = list(first.keys())
        buffers = {k: [first[k]] for k in keys}

        # Remaining simulations
        for _ in iterator if show_progress else iterator:
            out = self.simulate(
                num_molecules=num_molecules,
                species=species,
                species_kwargs=species_kwargs,
            )
            for k in keys:
                if out[k].shape != buffers[k][0].shape:
                    raise ValueError(
                        f"Shape mismatch for key '{k}': expected {buffers[k][0].shape}, got {out[k].shape}"
                    )
                buffers[k].append(out[k])

        # Stack along leading batch axis
        return {k: np.stack(v, axis=0) for k, v in buffers.items()}

    def _cache_base_integrals(self):
        """Precompute and cache integrals for a single unit of the base molecule."""
        base_molecule = assemble_molecules(
            num_molecules=1,
            bond_distance=self.bond_distance,
            species=self.species,
            perturb=False,  # No perturbation for base integrals
            seed=self.seed,
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
        mol.spin = sum(gto.charge(atom) for atom in base_molecule["species"]) % 2
        mol.build()

        self._integral_cache = {
            "kinetic": mol.intor("int1e_kin").astype(np.float32),
            "nuc_potential": mol.intor("int1e_nuc").astype(np.float32),
            "eri": mol.intor("int2e_sph", aosym=1).astype(np.float32),
            "full_overlap": mol.intor("int1e_ovlp").astype(np.float32),
        }
