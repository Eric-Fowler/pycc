# Track B (B0) — program-native PyCC slice

The program-native experiment from `../../../EF_INTEGRATION_PLAN.md` §4, kept in a
separate directory/commit series from Track A so the two experiments don't
contaminate each other.

## Scope of this first slice
PyCC's **CCSD T1 residual + fused Jacobi update**, expressed as **one composed
`ef.program`** (residual cut `R1`, reciprocal-denominator `Map`, update root
`t1 + R1 * Dinv1`), driven with **planner search disabled** (`search=False`, the
frozen B0 baseline). This is the plan's explicitly-labeled **plumbing milestone**,
not a planner-performance verdict — the CCSD intermediates (`Fae`/`Fmi`/`Fme`) and
integral blocks are input leaves so the slice stays isolated. The **T2 residual +
`t2 += r2/Dijab`** slice (the "preferred meaningful B0") is the next step.

The equations are PyCC's own (`reference.py`, transcribed from `pycc/ccwfn.py`), not
ehrenfest's DF-CCSD example.

## What is and is not verified
**Verified here:** the ef composed-program plumbing is validated against an
**independent numpy transcription** of PyCC's T1 equations — for one fixed state,
residual and post-Jacobi amplitude compared **separately**, no DIIS/convergence,
both the `ef.evaluate` oracle and the composed `ef.program` matching to machine
precision (ehrenfest `claude/tiling-from-scratch` @ da4d2d9).

**Not yet verified: production-PyCC equivalence.** `reference.py` and
`residual_t1.py` are both hand-translations of the same source equations, so their
agreement cannot catch a transcription mistake common to both. The check that can —
against PyCC's *production* residual path — is authored in `against_pycc.py`
(API-complete: PyCC's own `build_Fae/Fmi/Fme`, `cc.residuals`, and `cc.Dia`, with
the eps-derived denominator asserted equal to `cc.Dia`) but needs psi4, so it is
**not yet executed**.

- `reference.py` — independent numpy `r_T1` + Jacobi update (the numpy transcription).
- `residual_t1.py` — the composed `ef.program` builder (`build(o, v)`, `Slice.run(...)`).
- `against_pycc.py` — the live-`ccwfn` contract (needs psi4); replaces the numpy
  reference with production PyCC. `compare_against_pycc()` + a `pytest.importorskip`
  test.
- `tests/test_b0_t1.py` — plumbing contract vs the numpy transcription (numpy + ehrenfest).
- `tests/test_b0_against_pycc.py` — the production comparison (skips without psi4).

## Run
```bash
# plumbing vs numpy transcription (numpy + ehrenfest)
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_t1.py -q
# production comparison (needs psi4; skips otherwise)
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_against_pycc.py -q
```

## Not yet done (next B0/B1 steps)
- Execute `against_pycc.py` on a psi4 machine — the production-PyCC equivalence
  check (currently authored but unrun).
- T2 residual slice with its intermediates.
- B1a cold/warm + peak-memory benchmark of the `search=False` baseline (broken out:
  IR build, plan, compile, first exec, warm exec, whole-iteration, memory).
- B1b optional `search=True` experiment, reported separately.
