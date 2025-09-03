import numpy as np
from collections.abc import Sequence, Callable
from utils.molecule_utils import assemble_molecules, compute_ccsd
from pyscf import gto, scf, hessian


class MoleculeSimulator:
    """Simulate molecular systems and compute CCSD properties for ML datasets."""

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
        position_noise: float = 0.1,
        temperature: float = 300,
        use_fixed_noise: bool = False,
        normal_mode_sampling: bool = False,
        arrangement: str = "chain",
        cluster_size: float = 2.0,
        min_inter_dist: float | None = None,
        seed: int | None = None,
        verbose: int = 0,
        return_amplitudes: bool = False,
        coord_scale: float | None = 0.1,
        cache_integrals: bool = False,
    ):
        """
        Initialize the molecule simulator.

        Parameters
        ----------
        species : str, list of (atom, coord), dict, or callable, optional
            Defines the base unit to repeat (atom or molecule), default is "H".
            - str: Single atom symbol (e.g., "H").
            - list of (str, [float, float, float]): Fragment as list of (atom, coordinates).
            - dict {str: [float, float, float]}: Fragment as dictionary of atom to coordinates.
            - callable: Function returning a list of (atom, coord) tuples (e.g., molecule generator).
        species_kwargs : dict, optional
            Keyword arguments for callable `species`, default is None.
        num_molecules : int, optional
            Number of base units to simulate, default is 1.
        bond_distance : float, optional
            Ideal center-to-center distance between units in chains (Å), or scale for clusters, default is 2.0.
        basis : str, optional
            Basis set for quantum chemistry calculations, default is "sto3g".
        perturb : bool, optional
            If True, apply random translational noise and internal perturbations (e.g., bond/angle sampling),
            default is True.
        position_noise : float, optional
            Standard deviation of translational noise (Å) at 300 K (if use_fixed_noise=False) or fixed,
            default is 0.1.
        temperature : float, optional
            Temperature (K) for scaling noise amplitude, default is 300.
        use_fixed_noise : bool, optional
            If True, use position_noise directly; if False, scale by sqrt(temperature/300), default is False.
        normal_mode_sampling : bool, optional
            If True, sample internal coordinates along normal modes for callable species, default is False.
        arrangement : {"chain", "cluster"}, optional
            Arrangement type: "chain" (linear along +x) or "cluster" (random 3D in a sphere), default is "chain".
        cluster_size : float, optional
            Scaling factor for cluster sphere radius, default is 2.0.
        min_inter_dist : float, optional
            Minimum center-to-center distance in clusters (Å), default is bond_distance / 3.
        seed : int, optional
            Random seed for perturbations, default is None.
        verbose : int, optional
            Verbosity level for PySCF calculations, default is 0.
        return_amplitudes : bool, optional
            If True, return full CCSD amplitudes, default is False.
        coord_scale : float, optional
            Scaling factor for coordinates in output, default is 0.1.
        cache_integrals : bool, optional
            If True, cache one- and two-electron integrals for the base molecule, default is False.

        Notes
        -----
        Future versions may support external geometries (e.g., from ASE/OpenMM) by allowing `species`
        to be an external object (e.g., ase.Atoms).
        """
        self.species = species
        self.species_kwargs = species_kwargs or {}
        self.num_molecules = num_molecules
        self.bond_distance = bond_distance
        self.basis = basis
        self.perturb = perturb
        self.position_noise = position_noise
        self.temperature = temperature
        self.use_fixed_noise = use_fixed_noise
        self.normal_mode_sampling = normal_mode_sampling
        self.arrangement = arrangement
        self.cluster_size = cluster_size
        self.min_inter_dist = min_inter_dist
        self.seed = seed
        self.verbose = verbose
        self.return_amplitudes = return_amplitudes
        self.coord_scale = coord_scale
        self.cache_integrals = cache_integrals
        self._integral_cache = {}
        self._normal_modes_cache = None

        if cache_integrals or normal_mode_sampling:
            self._cache_base_integrals()

    def _cache_base_integrals(self):
        """Cache integrals and normal modes for the base molecule."""
        base_molecule = assemble_molecules(
            num_molecules=1,
            bond_distance=self.bond_distance,
            species=self.species,
            perturb=False,  # No perturbations for base geometry
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

        if self.cache_integrals:
            self._integral_cache = {
                "kinetic": mol.intor("int1e_kin").astype(np.float32),
                "nuc_potential": mol.intor("int1e_nuc").astype(np.float32),
                "eri": mol.intor("int2e_sph", aosym=1).astype(np.float32),
                "full_overlap": mol.intor("int1e_ovlp").astype(np.float32),
            }

        if self.normal_mode_sampling:
            # Compute normal modes and frequencies
            mf = scf.RHF(mol).run() if mol.spin == 0 else scf.UHF(mol).run()
            hess = hessian.RHF(mf).kernel() if mol.spin == 0 else hessian.UHF(mf).kernel()
            # Simplified: Needs mass-weighting and mode filtering
            masses = np.array([mol.mass(atom) for atom in base_molecule["species"]])
            mass_weights = np.repeat(1.0 / np.sqrt(masses), 3)
            mass_weighted_hess = hess * mass_weights[:, None, :, None] * mass_weights[None, :, None, :]
            mass_weighted_hess = mass_weighted_hess.reshape(3 * len(mol.atom), 3 * len(mol.atom))
            eigenvalues, eigenvectors = np.linalg.eigh(mass_weighted_hess)
            freqs = np.sqrt(np.abs(eigenvalues)) * 2.194746e5  # Convert to cm^-1 (approx)
            modes = eigenvectors  # Normal modes
            # Filter out translational/rotational modes (simplified)
            valid_modes = freqs > 1e-2  # Remove near-zero frequencies
            self._normal_modes_cache = (modes[:, valid_modes], freqs[valid_modes])

    def simulate(
        self,
        num_molecules: int | None = None,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Simulate a system and compute CCSD properties.

        Parameters
        ----------
        num_molecules : int, optional
            Number of units to simulate, defaults to self.num_molecules.
        species : str, list, dict, or callable, optional
            Base unit to simulate, defaults to self.species.
        species_kwargs : dict, optional
            Keyword arguments for callable species, defaults to self.species_kwargs.

        Returns
        -------
        dict
            CCSD outputs (e.g., t1, coordinates, integrals).
        """
        if num_molecules is None:
            num_molecules = self.num_molecules
        if num_molecules < 1:
            raise ValueError("N must be a positive integer")
        if species is None:
            species = self.species
            species_kwargs = species_kwargs or self.species_kwargs.copy()
            species_kwargs["normal_modes"] = self._normal_modes_cache if self.normal_mode_sampling else None
        else:
            if not (isinstance(species, (str, list, dict)) or callable(species)):
                raise TypeError(
                    "species must be a string, list[(atom, coord)], dict{atom: coord}, or callable"
                )
            species_kwargs = species_kwargs or {}

        molecules = assemble_molecules(
            num_molecules=num_molecules,
            bond_distance=self.bond_distance,
            species=species,
            arrangement=self.arrangement,
            cluster_size=self.cluster_size,
            min_inter_dist=self.min_inter_dist,
            perturb=self.perturb,
            position_noise=self.position_noise,
            temperature=self.temperature,
            use_fixed_noise=self.use_fixed_noise,
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
        samples: int | None = None,
        num_molecules: int | None = None,
        species: str
        | list[tuple[str, Sequence[float]]]
        | dict[str, Sequence[float]]
        | Callable
        | None = None,
        species_kwargs: dict | None = None,
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
        return [
            self.simulate(num_molecules, species, species_kwargs)
            for _ in range(samples)
        ]
