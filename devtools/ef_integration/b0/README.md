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
intermediates stay internal. Because it includes the intermediate-production region
(not pre-materialized intermediates), **this is the first slice suitable for B1a
performance work.**

What the `search=False` baseline actually exposes (be precise — the
materialization choice is NOT being searched here): one composed forest;
cross-expression CSE / shared work; stable program order; intermediate reuse;
residency/preload opportunities; and the **declared-cut materialization baseline**
(the default value of the materialize-vs-fuse DoF). Testing *alternate*
materialize-vs-fuse decisions is `search=True`, which is **B1b**, reported
separately — so a good/bad B1a result must not be attributed to "EF's
materialization optimizer", which is deliberately off in B1a.

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
# production comparisons + benchmark smoke (need psi4; skip otherwise)
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_against_pycc.py \
    devtools/ef_integration/b0/tests/test_b0_t2_against_pycc.py -q
# B1a benchmark (gated on the production comparison; needs psi4)
PYTHONPATH=/path/to/ehrenfest python -m devtools.ef_integration.b0.bench_t2
```

## B1a benchmark (`bench_t2.py`) — authored, gated
The harness is written but its numbers must **not** be interpreted until the psi4
production gate (`against_pycc_t2.py`) is green (`bench_t2.main` enforces this order:
it runs the correctness comparison first and refuses to report timings if it fails).

Two contract points it gets right:
- **Apples-to-apples comparator:** a T2-ONLY production region
  (`build_* → cc.r_T2 → t2 + r2/cc.Dijab` on `cc.contract`/opt_einsum), **not**
  `cc.residuals()` (which also computes `r_T1` and would over-charge PyCC). `cc.r_T2`
  already symmetrizes, so this equals the correctness authority `cc.residuals(...)[1]`
  while doing only T2 work. The authority stays `cc.residuals(...)[1]`; the
  *comparator* is this T2-only region.
- **Lifecycle split, not one opaque number:** `Slice.precompile()` and
  `Slice.execute()` are separate, so the harness times EF IR/program build, runner
  construction, `precompile(search=False)`, first (cold) execute, and a **warm loop
  that reuses the same Slice/program/runner/precompiled plan** (no rebuild, no
  precompile). Only the warm distribution is compared to the PyCC T2 region.

## GPU / B2 note — do not canonize the B0 host boundary
For B0 correctness, host-reading `R2` is useful (it lets the residual be compared
separately). For a serious GPU/B2 regional handoff, revisit that: PyCC ultimately
needs the updated amplitudes and convergence info, not necessarily the full `R2`
crossing back — EF could form the Jacobi increment / RMS contribution internally.
`host-read full R2 + host-read t2_trial` is a B0 correctness convenience, **not** the
intended final GPU integration contract.

## Not yet done (next B0/B1 steps)
- Execute the production comparisons (`against_pycc.py`, `against_pycc_t2.py`) on a
  psi4 machine — the gate before any B1a interpretation.
- Run `bench_t2.py` (after the gate) and report the lifecycle-split numbers.
- B1b optional `search=True` experiment, reported separately.
