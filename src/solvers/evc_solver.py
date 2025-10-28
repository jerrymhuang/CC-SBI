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
        geometries,
        t1s: np.ndarray,
        t2s: np.ndarray,
        reference_determinant: np.ndarray = None,
        reference_overlap=None,
        basis: str = "cc-pVTZ",
        mix_states: bool = False
    ):
        self.geometries = geometries
        self.basis = basis
        self.reference_determinant = reference_determinant
        self.t1s = t1s
        self.t2s = t2s
        self.reference_overlap = reference_overlap
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

    def solve_with_initial_guess(self, tolerance=1e-8):
        """
        Solve CCSD equations for each geometry using a starting guess for amplitudes.
        """
        rhf_energies = []
        ccsd_energies = []
        num_iterations = []

        num_molecules = self.geometries["atoms"].shape[0]

        for i in range(num_molecules):
            atoms = self.geometries["atoms"][i]
            positions = self.geometries["positions"][i]
            # Build molecule
            molecule = build_pyscf_molecule(
                atoms=atoms,
                positions=positions,
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
            t1_guess, t2_guess = self.t1s[i], self.t2s[i]
            t1_new, t2_new = self.basis_change_cluster_operator(u, t1_guess, t2_guess)

            # Run CCSD with transformed amplitudes as initial guess
            ccsd = cc.CCSD(rhf)
            ccsd.conv_tol = tolerance
            ccsd.t1, ccsd.t2 = t1_new, t2_new
            ccsd.run()
            ccsd_energies.append(ccsd.e_tot)
            num_iterations.append(ccsd.niter)

        return {
            "rhf_energies": rhf_energies,
            "ccsd_energies": ccsd_energies,
            "num_iterations": num_iterations
        }
