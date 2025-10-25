import numpy as np
from pyscf import scf, cc
from opt_einsum import contract
from utils.molecule_utils import build_pyscf_molecule, compute_hartree_fock
from utils.procrustes_utils import localized_procrustes_overlap

class EVCSolver:
    """
    Class to solve CCSD equations using a starting guess for amplitudes, leveraging PySCF and Procrustes alignment.

    Parameters
    ----------
    all_x : list
        Geometry parameters for the molecule.
    molecule_func : callable
        Function returning molecular geometry (string or list of (atom, [x, y, z]) tuples).
    basis : str
        Basis set for the molecule (e.g., 'cc-pVTZ').
    reference_determinant : np.ndarray
        Reference Hartree-Fock MO coefficients for Procrustes alignment.
    t1s : list of np.ndarray
        List of t1 amplitude arrays for starting guesses.
    t2s : list of np.ndarray
        List of t2 amplitude arrays for starting guesses.
    reference_overlap : np.ndarray, optional
        Overlap matrix for the reference geometry.
    mix_states : bool, optional
        Whether to mix occupied and virtual orbitals in Procrustes alignment (default: False).
    """

    def __init__(
        self,
        all_x,
        molecule_func,
        basis,
        reference_determinant,
        t1s,
        t2s,
        reference_overlap=None,
        mix_states=False
    ):
        self.all_x = all_x
        self.molecule_func = molecule_func
        self.basis = basis
        self.reference_determinant = np.array(reference_determinant, dtype=np.float32)
        self.t1s = [np.array(t1, dtype=np.float32) for t1 in t1s]
        self.t2s = [np.array(t2, dtype=np.float32) for t2 in t2s]
        self.reference_overlap = (np.array(reference_overlap, dtype=np.float32)
                                  if reference_overlap is not None else None)
        self.mix_states = mix_states
        self.num_iterations = []
    
    @staticmethod
    def basis_change_cluster_operator(rotation_matrix, t1, t2):
        """
        Transform t1 and t2 amplitudes to a new orbital basis using rotation matrix U.

        Parameters
        ----------
        rotation_matrix : np.ndarray
            Rotation matrix from Procrustes alignment.
        t1 : np.ndarray
            t1 amplitudes (shape: num_virtual_orbitals, num_occupied_orbitals).
        t2 : np.ndarray
            t2 amplitudes (shape: num_virtual_orbitals, num_virtual_orbitals, num_occupied_orbitals, num_occupied_orbitals).

        Returns
        -------
        tuple
            Transformed t1 and t2 amplitudes.
        """
        num_virtual_orbitals, num_occupied_orbitals = t1.shape
        occupied_orbital_rotation = rotation_matrix[:num_occupied_orbitals, :num_occupied_orbitals]
        virtual_orbital_rotation = rotation_matrix[num_occupied_orbitals:, num_occupied_orbitals:]
        new_t1 = contract(
            'ij,ai,ab->bj',
            occupied_orbital_rotation,
            t1,
            virtual_orbital_rotation
        )
        new_t2 = contract(
            "ik,jl,abij,ac,bd->cdkl",
            occupied_orbital_rotation,
            occupied_orbital_rotation,
            t2,
            virtual_orbital_rotation,
            virtual_orbital_rotation
        )
        return new_t1, new_t2

    def solve_with_initial_guess(self, tolerance=1e-8, start_guess_indices=None):
        """
        Solve CCSD equations for each geometry using a starting guess for amplitudes.

        Parameters
        ----------
        tolerance : float, optional
            Convergence tolerance for CCSD (default: 1e-8).
        start_guess_indices : list of int, optional
            Indices of t1s/t2s to use as starting guesses for each geometry.
            If None, uses the first set of amplitudes for all geometries.

        Returns
        -------
        tuple
            Lists of HF energies, CCSD energies, and number of iterations per geometry.
        """
        rhf_energies = []
        ccsd_energies = []
        self.num_iterations = []

        # Default to first amplitude set if no indices provided
        if start_guess_indices is None:
            start_guess_indices = [0] * len(self.all_x)

        for k, (x_alpha, guess_idx) in enumerate(zip(self.all_x, start_guess_indices)):
            # Build molecule
            molecule = build_pyscf_molecule(
                pyscf_atoms=self.molecule_func(*x_alpha),
                basis=self.basis,
                unit="bohr",
                cartesian=False,
                verbose=0,
                charge=0
            )

            # Compute Hartree-Fock
            rhf_data = compute_hartree_fock(molecule)
            rhf = scf.RHF(molecule)
            rhf.mo_coeff = rhf_data["determinant"]
            rhf.mo_occ = rhf_data["occupancies"]
            rhf_energies.append(rhf.e_tot)

            # Compute Procrustes alignment
            target_overlap = molecule.intor("int1e_ovlp").astype(np.float32)
            procrustes_data = localized_procrustes_overlap(
                reference_determinant=self.reference_determinant,
                reference_overlap=self.reference_overlap,
                target_determinant=rhf_data["determinant"],
                target_overlap=target_overlap,
                occupancies=rhf_data["occupancies"],
                mix_states=self.mix_states
            )
            u = procrustes_data["orthogonal_overlap"]

            # Transform starting guess amplitudes
            t1_guess, t2_guess = self.t1s[guess_idx], self.t2s[guess_idx]
            t1_new, t2_new = self.basis_change_cluster_operator(u, t1_guess, t2_guess)

            # Run CCSD with transformed amplitudes as initial guess
            ccsd = cc.CCSD(rhf)
            ccsd.conv_tol = tolerance
            ccsd.t1, ccsd.t2 = t1_new, t2_new
            ccsd.run()
            ccsd_energies.append(ccsd.e_tot)
            self.num_iterations.append(ccsd.niter)

        return {
            "rhf_energies": rhf_energies,
            "ccsd_energies": ccsd_energies,
            "num_iterations": self.num_iterations
        }