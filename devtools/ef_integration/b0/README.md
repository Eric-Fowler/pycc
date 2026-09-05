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

## Numerical-comparison contract (implemented)
For one fixed random state: compare the ef program's **residual** and
**post-Jacobi amplitude separately** to the PyCC equations, with **no DIIS,
convergence, or second iteration**. `tests/test_b0_t1.py` runs it two ways — the
`ef.evaluate` oracle and the composed `ef.program` — both matching the numpy
reference to machine precision (verified on ehrenfest `claude/tiling-from-scratch`
@ da4d2d9).

- `reference.py` — independent numpy `r_T1` + Jacobi update (the "existing PyCC" side here).
- `residual_t1.py` — the composed `ef.program` builder (`build(o, v)`, `Slice.run(...)`).
- `against_pycc.py` — the live-`ccwfn` version of the contract (needs psi4); the
  final hookup that replaces the numpy reference with production PyCC.
- `tests/test_b0_t1.py` — the B0 contract (numpy + ehrenfest only).

## Run
```bash
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/b0/tests/test_b0_t1.py -q
```

## Not yet done (next B0/B1 steps)
- Swap `reference.py` for a live PyCC `ccwfn` (`against_pycc.py`) on a psi4 machine.
- T2 residual slice with its intermediates.
- B1a cold/warm + peak-memory benchmark of the `search=False` baseline (broken out:
  IR build, plan, compile, first exec, warm exec, whole-iteration, memory).
- B1b optional `search=True` experiment, reported separately.
