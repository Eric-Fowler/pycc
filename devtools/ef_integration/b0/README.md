# Track B (B0) — program-native PyCC slice

The program-native experiment from `../../../EF_INTEGRATION_PLAN.md` §4, kept in a
separate directory/commit series from Track A so the two experiments don't
contaminate each other.

## Two slices
**T1 (plumbing milestone)** — `residual_t1.py`: PyCC's CCSD T1 residual + fused
Jacobi update as one composed `ef.program`, with the intermediates as input leaves
so it stays isolated. A correctness/plumbing proof, **not** a performance verdict.

**T2 (the meaningful slice)** — `residual_t2.py`: PyCC's CCSD T2 residual with all
**seven intermediates built INSIDE the single program** as cuts
(`Fae`/`Fmi`/`Fme`/`Wmnij`/`Wmbej`/`Wmbje`/`Zmbij`), the `P(ij)(ab)` symmetrization,
and the fused `t2 + R2/Dijab` update. Only `R2` and `t2_trial` are host-read; the
intermediates stay internal so ehrenfest sees the reuse/materialization choices.
Because it includes the intermediate-production region (not pre-materialized
intermediates), **this is the first slice suitable for B1a performance work.**

Both use `search=False` (frozen B0 baseline) and PyCC's own equations (transcribed
from `pycc/ccwfn.py`), not ehrenfest's DF-CCSD example. The T2 EF slice uses clean
relabelled einsums (no `.T`/`.swapaxes`); `reference_t2.py` is a faithful mirror of
PyCC's exact strings, so a relabelling error shows up as a mismatch against it.

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

- `reference.py` / `reference_t2.py` — numpy transcription (T1) / faithful mirror (T2).
- `residual_t1.py` / `residual_t2.py` — the composed `ef.program` builders.
- `against_pycc.py` / `against_pycc_t2.py` — the live-`ccwfn` contracts (need psi4);
  replace the numpy side with production PyCC. Each has `compare_against_pycc()` + a
  `pytest.importorskip` test. The T2 one is **staged**: each EF intermediate vs
  PyCC's own `build_*`, then composed `r2` vs `cc.residuals(...)[1]`, then `t2_trial`
  vs `cc.t2 + r2_pycc/cc.Dijab`.
- `tests/test_b0_t1.py`, `tests/test_b0_t2.py` — local contracts vs the numpy
  side (numpy + ehrenfest); T2 includes per-intermediate unit checks.
- `tests/test_b0_against_pycc.py`, `tests/test_b0_t2_against_pycc.py` — the production
  comparisons (skip without psi4).

## Run
```bash
# plumbing vs numpy transcription (numpy + ehrenfest)
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_t1.py -q
# production comparison (needs psi4; skips otherwise)
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_against_pycc.py -q
```

## Not yet done (next B0/B1 steps)
- Execute the production comparisons (`against_pycc.py`, `against_pycc_t2.py`) on a
  psi4 machine — currently authored + staged but unrun here.
- B1a cold/warm + peak-memory benchmark of the **T2** `search=False` baseline
  (broken out: IR build, plan, compile, first exec, warm exec, whole-iteration,
  memory) — the T2 slice is the first one suitable for this since it produces its
  own intermediates.
- B1b optional `search=True` experiment, reported separately.
