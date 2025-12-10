from pyscf import gto, scf, mp, dft, cc
import numpy as np

def khatri_rao(A, B):
    I, R = A.shape
    J, Rb = B.shape
    assert R == Rb
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, R)

def unfold(X, mode):
    n_dims = X.ndim
    order = (mode,) + tuple(i for i in range(n_dims) if i != mode)
    X_perm = np.transpose(X, order)
    return X_perm.reshape(X.shape[mode], -1)

def cp_als(
    X,
    rank,
    n_iter_max=500,
    tol=1e-6,
    verbose=False,
    random_state=None,
    A_init = None,
    B_init = None,
    C_init = None
):
    """
    Computes the canonical polyadic (CP) decomposition with alternating least squares (ALS).
    """
    rng = np.random.default_rng(random_state)
    I, J, K = X.shape

    if A_init is not None:
        A = A_init
    else:
        A = rng.standard_normal((I, rank))
    
    if B_init is not None:
        B = B_init
    else:
        B = rng.standard_normal((J, rank))
    
    if C_init is not None:
        C = C_init
    else:
        C = rng.standard_normal((K, rank))

    X1 = unfold(X, 0)  
    X2 = unfold(X, 1) 
    X3 = unfold(X, 2)

    prev_error = None

    for it in range(n_iter_max):
        BtB = B.T @ B               # Is this a bug???
        CtC = C.T @ C              
        KR = khatri_rao(B, C)     
        G = BtB * CtC            
        RHS = X1 @ KR           
        A = RHS @ np.linalg.pinv(G)

        AtA = A.T @ A
        KR = khatri_rao(A, C)       
        G = AtA * CtC
        RHS = X2 @ KR
        B = RHS @ np.linalg.pinv(G)

        BtB = B.T @ B
        KR = khatri_rao(A, B)       
        G = AtA * BtB
        RHS = X3 @ KR
        C = RHS @ np.linalg.pinv(G)

        X_hat = np.einsum('ir,jr,kr->ijk', A, B, C)
        error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)

        if verbose and it % 10 == 0:
            print(f"iter {it:4d}  rel_error = {error:.3e}")

        if prev_error is not None and abs(prev_error - error) < tol:
            break
        prev_error = error

    return A, B, C, error

def update_C(X, A, B):
    I, J, K = X.shape
    R = A.shape[1]

    X3 = X.transpose(2, 0, 1).reshape(K, I * J)

    KR = khatri_rao(A, B)

    C_t, *_ = np.linalg.lstsq(KR, X3.T, rcond=None)
    C = C_t.T 

    return C

if __name__ == "__main__":

    mol = gto.Mole()
    mol.atom = """
    O  0.000000  0.000000  0.000000
    H  0.000000 -0.757000  0.587000
    H  0.000000  0.757000  0.587000
    """
    mol.basis = "sto-3g"      
    mol.build()

    # DFT PBE
    mfdft = dft.RKS(mol)
    mfdft.xc = "PBE"          
    mfdft.grids.level = 7   
    mfdft.conv_tol = 1e-10    
    
    e_pbe = mfdft.kernel()
    print(f"PBE total energy: {e_pbe:.10f} Ha")
    print(f"Converged: {mfdft.converged}")

    dm_dft_ao = mfdft.make_rdm1()
    C = mfdft.mo_coeff
    dm_dft_mo = C.T @ dm_dft_ao @ C

    grids = mfdft.grids
    ni = mfdft._numint 

    rho_dft = ni.get_rho(mol, dm_dft_ao, grids)
    print("Number of grid points:", rho_dft.size)

    weights = grids.weights
    nelec_from_rho = np.einsum("g,g->", rho_dft, weights)
    print("∫ρ(r) d^3r =", nelec_from_rho, "(should be ~", mol.nelectron, ")")

    # RHF
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    print(f"Converged: {mf.converged}")
    print(f"HF energy: {e_hf:.10f} Ha")

    dm_rhf_ao = mf.make_rdm1()
    C = mf.mo_coeff
    dm_rhf_mo = C.T @ dm_rhf_ao @ C
    rho_rhf = ni.get_rho(mol, dm_rhf_ao, grids)
    nelec_hf = np.einsum("g,g->", rho_rhf, grids.weights)
    print("HF ∫ρ(r) d^3r =", nelec_hf)  

    #MP2
    mymp = mp.MP2(mf)
    mymp.kernel()
 
    print(f"MP2 energy   : {mymp.e_tot:.10f} Ha ({mymp.e_corr:.10f} Ha)")
    
    dm_mp_mo = mymp.make_rdm1()
    C = mf.mo_coeff
    dm_mp_ao = C @ dm_mp_mo @ C.T
    rho_mp = ni.get_rho(mol, dm_mp_ao, grids)
    nelec_mp = np.einsum("g,g->", rho_mp, grids.weights)
    print("MP2 ∫ρ(r) d^3r =", nelec_mp)

    print("Factorizing T2 from MP2")
    t2 = mymp.t2

    o, v = t2.shape[0], t2.shape[2]
    t2_mat = t2.transpose(0, 2, 1, 3).reshape(o*v, o*v)
    print("max |T - T^T| =", np.max(np.abs(t2_mat - t2_mat.T)))
    
    eigvals, eigvecs = np.linalg.eigh(t2_mat)
    idx = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

   # Add truncation here
    R = len(eigvals) 
    eigvals_R = eigvals[:R]   
    eigvecs_R = eigvecs[:, :R]

    t2_fact = eigvecs_R.reshape(o, v, R)

    t2_rec = np.einsum('k,iak,jbk->ijab', eigvals_R, t2_fact, t2_fact)    
    print("max |T2 - T2_rec| =", np.max(np.abs(t2 - t2_rec)))
    
    t2_mp2_A, t2_mp2_B, t2_mp2_C, err = cp_als(t2_fact, 100)
    t2_fact_rec = np.einsum('ir,ar,kr->iak', t2_mp2_A, t2_mp2_B, t2_mp2_C)    
    print("max |T2_factor - T2_rec_factor_mat| =", np.max(np.abs(t2_fact - t2_fact_rec)))

    # CCSD(T)
    mycc = cc.CCSD(mf)
    mycc.kernel()
    et = mycc.ccsd_t()
    
    print(f"Converged: {mycc.converged}")
    print(f"CCSD energy   : {mycc.e_tot:.10f} Ha ({mycc.e_corr:.10f} Ha)")
    print(f"CCSD(T) energy: {mycc.e_tot + et:.10f} Ha ({mycc.e_corr + et:.10f} Ha)")

    dm_cc_mo = mycc.make_rdm1()
    C = mf.mo_coeff
    dm_cc_ao = C @ dm_cc_mo @ C.T
    rho_cc = ni.get_rho(mol, dm_cc_ao, grids)
    nelec_cc = np.einsum("g,g->", rho_cc, grids.weights)
    print("CCSD ∫ρ(r) d^3r =", nelec_cc)

    print("Factorizing T2 from CCSD")
    t1 = mycc.t1
    t2 = mycc.t2

    o, v = t2.shape[0], t2.shape[2]
    t2_mat = t2.transpose(0, 2, 1, 3).reshape(o*v, o*v)
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
    print("max |T2 - T2_rec| =", np.max(np.abs(t2 - t2_rec)))
    
    #t2_A, t2_B, t2_C, err = cp_als(t2_fact, 100)
    # MP2 initialization
    #t2_A, t2_B, t2_C, err = cp_als(t2_fact, 100, A_init = t2_mp2_A, B_init = t2_mp2_B, C_init = t2_mp2_C)
    
    t2_A = t2_mp2_A
    t2_B = t2_mp2_B
    t2_C = update_C(t2_fact, t2_mp2_A, t2_mp2_B)
         
    t2_fact_rec = np.einsum('ir,ar,kr->iak', t2_A, t2_B, t2_C)    
    print("max |T2_factor - T2_rec_factor_mat| =", np.max(np.abs(t2_fact - t2_fact_rec)))

    print(f'Factor matrix differences (CCSD - MP2):\n Occ Factor: {np.linalg.norm(t2_A - t2_mp2_A)}\n Vir Factor: {np.linalg.norm(t2_B - t2_mp2_B)}\n Cor Factor: {np.linalg.norm(t2_C - t2_mp2_C)}')
    breakpoint()


