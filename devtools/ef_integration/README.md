# ehrenfest (`ef`) integration tooling — Track A (A0)

Offline tooling for the compatibility/validation seam described in
`../../EF_INTEGRATION_PLAN.md`. Depends only on numpy and (for replay/adapter)
`ehrenfest`; it does **not** import the `pycc` package or require psi4, so it runs
in a lightweight environment.

## Pieces
- `../../pycc/ef_capture.py` — canonical signature definitions + the runtime
  recorder used by the `device.py` capture hook (owns `signature`/`layout_class`).
- `static_corpus.py` — parses the literal `contract('...')` specs from the sources
  and answers the implicit-output question without a live run.
- `adapter.py` — the reference `EfContractionAdapter` (CPU/DP/numpy compatibility
  path): whitespace-normalized, case-preserving, node memoized on
  `(clean_spec, shapes, per_operand_dtypes)`, implicit-output rejected loudly.
- `replay.py` — builds representative operands (values random, layout matching the
  record) and compares `ef.evaluate` / `ef.runner` to `np.einsum`.
- `tests/test_a0.py` — self-tests (numpy + ehrenfest only).

## Run
```bash
# 1. static inventory (no deps beyond stdlib)
python -m devtools.ef_integration.static_corpus

# 2. offline replay against ef over the real spec corpus (synthetic extent)
PYTHONPATH=/path/to/ehrenfest python -m devtools.ef_integration.replay --from-corpus --extent 3 --runner

# 3. self-tests
PYTHONPATH=/path/to/ehrenfest python -m pytest devtools/ef_integration/tests/test_a0.py -q
```

## Live capture (requires psi4 + the full pycc stack)
Run any CCSD calculation with the capture env var set to a JSONL path, then replay
the *real* shapes/dtypes/layouts:
```bash
PYCC_EF_CAPTURE=/tmp/ccsd_sigs.jsonl python -m pytest pycc/tests/test_002_ccsd_energy.py
PYTHONPATH=/path/to/ehrenfest python -m devtools.ef_integration.replay --jsonl /tmp/ccsd_sigs.jsonl --runner
```
The capture hook (`pycc/device.py`, `ContractionBackend.__call__`) is off unless
`PYCC_EF_CAPTURE` is set and never perturbs the returned value.

## Observed so far (against ehrenfest `claude/tiling-from-scratch` @ da4d2d9)
Static inventory of the corpus: **1907 literal einsum specs, 858 distinct**;
arity {1: 11, 2: 1846, 3: 49, 4: 1}; 66 specs use uppercase indices; 4 carry
whitespace; **every literal spec is explicit-output** (implicit-output invariant
holds → the adapter's loud rejection is safe).

Offline replay (`--from-corpus --extent 3 --runner`, C + non-contiguous passes):
**all 858 distinct specs OK through both `ef.evaluate` and `ef.runner`.** Caveats:
uniform synthetic extent 3, DP real only — this covers the *index patterns* of the
corpus, not real extents, mixed precision, or complex (RT-CC) dtypes. Those need
the live capture above (and complex is a known `ef.runner` gap, §6 of the plan).
