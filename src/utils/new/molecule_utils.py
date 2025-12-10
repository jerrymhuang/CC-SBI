import numpy as np
from pyscf import gto, dft, scf, mp, cc, df
from collections.abc import Callable
from .tensor_utils import cp_als

def build_pyscf_molecule(
    geometry: Callable | str,
    basis: str = "sto3g"
):
    mol = gto.Mole()
    mol.atom = geometry() if callable(geometry) else geometry
    mol.basis = basis
    mol.build()
    return mol

def build_grids(mol, level=7):
    grids = dft.gen_grid.Grids(mol=mol)
    grids.level = level
    grids.build()
    return grids

def build_numint():
    """Returns the numerical integration engine from DFT. """
    return dft.numint.NumInt()


def compute_dft(mol, grid_level=7, xc="PBE", tolerance=1e-10, density_fit=True):
    """
    mol   : pyscf.gto.Mole
    grids : pre-built pyscf.dft.gen_grid.Grids object
    xc    : XC functional string
    ni    : optional pre-constructed NumInt; if None, a new one is created
    """

    # DFT object using the *given* grids
    if density_fit:
        print("Using density fit")
        mf = dft.RKS(mol).density_fit().run()
        mf.with_df.auxbasis = "def2-universal-jfit"
    else:
        print("Not using density fit")
        mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = tolerance
    e_pbe = mf.kernel()
    print("Generating density matrices")

    # Density matrices
    dm_ao = mf.make_rdm1()
    C = mf.mo_coeff
    dm_mo = C.T @ dm_ao @ C

    grids = mf.grids
    ni = mf._numint
    # Real-space density on the same grids
    rho = ni.get_rho(mol, dm_ao, grids)
    weights = grids.weights
    nelec = np.einsum("g,g->", rho, weights)

    return {
        "dm_dft_ao": dm_ao,
        "C_dft": C,
        "dm_dft_mo": dm_mo,
        "rho_dft": rho,
        "n_electrons_dft": nelec,
        "e_dft": mf.e_tot,
        "e_pbe": e_pbe,
        "converged": mf.converged,
        "grids": grids,
        "ni": ni,
    }

def compute_rhf(mol, grids, tolerance=1e-10, ni=None, density_fit=True):
    """
    mol    : pyscf.gto.Mole
    grids  : pre-built pyscf.dft.gen_grid.Grids object
    ni     : optional pyscf.dft.numint.NumInt; if None, a new one is created
    """
    if ni is None:
        ni = dft.numint.NumInt()

    mf = scf.RHF(mol).density_fit(auxbasis="weigend").run()
    mf.conv_tol = tolerance
    mf.kernel()

    # Density matrices (AO/MO)
    dm_ao = mf.make_rdm1()
    C = mf.mo_coeff
    dm_mo = C.T @ dm_ao @ C

    # Real-space density on the provided grids
    rho = ni.get_rho(mol, dm_ao, grids)
    nelec = np.einsum("g,g->", rho, grids.weights)

    return {
        "e_rhf": mf.e_tot,
        "dm_rhf_ao": dm_ao,
        "C_rhf": C,
        "dm_rhf_mo": dm_mo,
        "rho_rhf": rho,
        "n_electrons_rhf": nelec,
        "converged": mf.converged,
        "rhf": mf
    }

def factorize_t2(t2: np.ndarray):
    o, v = t2.shape[0], t2.shape[2]
    t2_mat = t2.transpose(0, 2, 1, 3).reshape(o * v, o * v)
    print("max |T - T^T| =", np.max(np.abs(t2_mat - t2_mat.T)))

    eigvals, eigvecs = np.linalg.eigh(t2_mat)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    R = len(eigvals)
    eigvals_R = eigvals[:R]
    eigvecs_R = eigvecs[:, :R]

    t2_fact = eigvecs_R.reshape(o, v, R)

    t2_rec = np.einsum('k,iak,jbk->ijab', eigvals_R, t2_fact, t2_fact)
    return {
        "t2_mat": t2_mat,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "R": R,
        "eigvals_R": eigvals_R,
        "eigvecs_R": eigvecs_R,
        "t2_fact": t2_fact,
        "t2_rec": t2_rec
    }


def compute_mp2(mol, rhf: scf.RHF, grids, tolerance=1e-10, ni=None):
    mp2 = mp.MP2(rhf).density_fit(auxbasis="weigend")
    mp2.kernel()

    dm_mo = mp2.make_rdm1()
    C = rhf.mo_coeff
    dm_ao = C @ dm_mo @ C.T
    rho_mp2 = ni.get_rho(mol, dm_ao, grids)
    nelec_mp2 = np.einsum("g,g->", rho_mp2, grids.weights)

    t2 = mp2.t2
    factors = factorize_t2(t2)
    t2_fact = factors["t2_fact"]
    t2_rec = factors["t2_rec"]

    t2_mp2_A, t2_mp2_B, t2_mp2_C, err = cp_als(t2_fact, 100)
    t2_fact_rec = np.einsum('ir,ar,kr->iak', t2_mp2_A, t2_mp2_B, t2_mp2_C)
    print("max |T2_factor - T2_rec_factor_mat| =", np.max(np.abs(t2_fact - t2_fact_rec)))

    outputs = {
        "e_tot_mp2": mp2.e_tot,
        "e_corr_mp2": mp2.e_corr,
        "dm_mp2_ao": dm_ao,
        "C_mp2": C,
        "dm_mp2_mo": dm_mo,
        "rho_mp2": rho_mp2,
        "n_electrons_mp2": nelec_mp2,
        "t2_a": t2_mp2_A,
        "t2_b": t2_mp2_B,
        "t2_c": t2_mp2_C,
        "t2_fact_rec": t2_fact_rec,
        "max_t2_t2_rec": np.max(np.abs(t2 - t2_rec)),
        "max_t2_fact_rec": np.max(np.abs(t2_fact - t2_fact_rec))
    }

    return outputs | factors


def compute_cc(mol, rhf: scf.RHF, grids, ni=None):
    mycc = cc.CCSD(rhf).density_fit(auxbasis="weigend")
    mycc.kernel()
    et = mycc.ccsd_t()

    dm_mo = mycc.make_rdm1()
    C = rhf.mo_coeff
    dm_ao = C @ dm_mo @ C.T
    rho_cc = ni.get_rho(mol, dm_ao, grids)
    nelec_cc = np.einsum("g,g->", rho_cc, grids.weights)

    t1 = mycc.t1
    t2 = mycc.t2

    factors = factorize_t2(t2)

    outputs = {
        "e_tot_cc": mycc.e_tot,
        "e_corr_cc": mycc.e_corr,
        "dm_cc_ao": dm_ao,
        "C_cc": C,
        "dm_cc_mo": dm_mo,
        "rho_cc": rho_cc,
        "n_electrons_cc": nelec_cc,
        "t1": t1,
        "t2": t2,
    }

    return outputs | factors
