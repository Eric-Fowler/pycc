"""Live B0 check: the ef T1-residual program vs an actual PyCC ``ccwfn`` state.

Requires the full PyCC stack (psi4). It replaces ``reference.py`` with a real
converged PyCC state and PyCC's own ``r_T1`` / update, completing the B0
numerical-comparison contract against the production code rather than a numpy
transcription. Not run in the lightweight test environment (no psi4); kept here as
the runnable hookup for a psi4-equipped machine.

Sketch of the intended flow (fill in against the current ccwfn API):

    import psi4, pycc
    import numpy as np
    from devtools.ef_integration.b0 import residual_t1 as R

    # 1. a fixed PyCC CCSD state
    psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5"); psi4.set_options({"basis": "cc-pVDZ"})
    _, wfn = psi4.energy("SCF", return_wfn=True)
    cc = pycc.CCwfn(wfn, model="CCSD")
    cc.solve_cc(e_conv=1e-8, r_conv=1e-7, maxiter=2)   # any fixed, non-trivial (t1, t2)

    o, v = cc.o, cc.v
    F, ERI, L = cc.H.F, cc.H.ERI, cc.H.L
    Fae = cc.build_Fae(...); Fmi = cc.build_Fmi(...); Fme = cc.build_Fme(...)
    inp = {
        "t1": cc.t1, "t2": cc.t2,
        "f_ai": np.asarray(F[v, o]).swapaxes(0, 1),
        "Fae": Fae, "Fmi": Fmi, "Fme": Fme,
        "L_ovvo": np.asarray(L[o, v, v, o]),
        "ERI_ovvv": np.asarray(ERI[o, v, v, v]),
        "L_oovo": np.asarray(L[o, o, v, o]),
        "eps_o": np.asarray(cc.H.eps[o]), "eps_v": np.asarray(cc.H.eps[v]),
    }

    # 2. PyCC's own residual + update for the SAME state (the ground truth)
    r1_pycc, _r2 = cc.residuals(F, cc.t1, cc.t2)     # extract the T1 residual
    t1_pycc = cc.t1 + r1_pycc / cc.Dia

    # 3. the ef program on the same inputs
    ef_slice = R.build(int(cc.no), int(cc.nv))
    r1_ef, t1_ef = ef_slice.run(inp, ef.runner(1 << 32, device="cpu"), search=False)

    # 4. compare residual and post-Jacobi amplitude SEPARATELY (no DIIS)
    assert np.allclose(r1_ef, r1_pycc) and np.allclose(t1_ef, t1_pycc)

The exact intermediate-builder call signatures must be read off the current ccwfn;
the leaf shapes/labels above match ``residual_t1.build``.
"""
