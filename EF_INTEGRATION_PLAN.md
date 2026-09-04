# PyCC × ehrenfest (`ef`) integration plan

**Status:** living document — first iteration. Expect many revisions.
**Goal:** route PyCC's tensor contractions through the `ehrenfest` front-end (`import ehrenfest as ef`)
so that (a) PyCC gains a real contraction *planner* (path search, tiling, capacity/placement-aware
execution) and (b) PyCC's real coupled-cluster equations become a stress test that surfaces gaps and
priorities for `ef`. `ef` is a work in progress, so this starts deliberately small (single method,
CPU, double precision, one contraction at a time) and widens over many iterations toward the whole repo.

This plan is grounded in an actual read of both codebases and a working proof-of-concept adapter that
was run against representative PyCC contractions (see [§6 Feasibility evidence](#6-feasibility-evidence)).

---

## 1. Why this is tractable: PyCC has exactly one contraction seam

Every heavy contraction in the non-local paths goes through a single callable, `self.contract`, an
instance of `ContractionBackend` defined in `pycc/device.py:38`. Its call signature is already the
einsum surface `ef` expects:

```python
contract(subscripts, *operands) -> ndarray   # e.g. contract('ijef,abef->ijab', tau, ERI[v,v,v,v])
```

- Created once by `DeviceManager.__init__` (`pycc/device.py:154`), exposed as `self.contract` on the
  `Wavefunction` base (`pycc/wavefunction.py:159`).
- Every sub-object reuses the same callable: `cclambda`, `ccdensity`, `ccresponse`, `cceom`, `cphf`,
  `correlatedderivs`, `rtcc` all do `self.contract = self.ccwfn.contract`; the per-method idiom is a
  local rebind `contract = self.contract` at the top of each routine (~200 sites).
- The `(T)` kernels take it as an explicit argument (`pycc/cctriples.py:192, 274, 321`).

There are **~1776 `contract(` call sites across 22 files**, all funnelling through that one object:

| file | calls | file | calls | file | calls |
|---|---|---|---|---|---|
| `ccresponse.py` | 479 | `cctriples.py` | 342 | `ccwfn.py` | 189 |
| `ccdensity.py` | 182 | `cclambda.py` | 134 | `cchbar.py` | 107 |
| `cceom.py` | 99 | `lccwfn.py` | 73 | `local.py` | 45 |
| `ciwfn.py` | 39 | `cphf.py` | 23 | `rt/rtcc.py` | 23 |

**Consequence:** we can swap the contraction engine for the entire non-local codebase by
re-implementing one method — `ContractionBackend.__call__` — without touching a single equation.

### Seams that bypass `self.contract` (handle separately, later)

- `pycc/local.py:8` and `pycc/lccwfn.py:8` — `from opt_einsum import contract` at module level
  (device-unaware). The **local (PNO/PAO)** path also stores amplitudes as *Python lists of small
  per-pair tensors* in per-pair truncated subspaces (`lccwfn.py:265-333`), so its shapes vary per
  pair — a different integration problem, deferred.
- `pycc/rt/rtcc.py:469` — one direct `opt_einsum.contract` call. RT-CC is **complex-valued** (see the
  `ef` gap in §5).

---

## 2. What `ef` can do today (the target API)

The old `ef.report`/`ef.simulate`/`ef.GTArray`/`auto_plan` API described in `ehrenfest`'s
`docs/reference/DESIGN_OVERVIEW.md` **no longer exists** — that stack was deleted and rebuilt around a
"tiling" engine that actually executes. Trust the code (`frontend.py`, `runner.py`, `oracle.py`) and
the live example under `ehrenfest/examples/chemistry/coupledCluster/ccsd/`, not the reference prose.

The relevant public surface (`ehrenfest/__init__.py`):

- `ef.array(shape, name, *, invariant=False)` — a named leaf tensor. `.at(letters)` labels its axes;
  `.base.uid` is the key data is passed under.
- `ef.einsum(spec, *operands) -> expr` — an einsum-spelled contraction chain over arrays/exprs
  (n-ary; the planner picks the binary order).
- `expr.node(out) -> Sum/Contract` — lower to one IR node; `out` names each free letter once.
- **Two ways to get numbers back**, both taking `{base.uid: ndarray}`:
  - `ef.evaluate(node, arrays)` — the **oracle**: `np.einsum` under the hood. Always available,
    pure-numpy, matches `np.einsum` to machine precision, handles complex. This is the safe drop-in.
  - `ef.runner(capacity, *, device="cpu"|"gpu")(node, arrays)` — the **real planner/executor**: does
    `path → binarize → fit_chain → execute`, caches the fitted plan per Contract identity. Returns a
    numpy array on the CPU lane; compiles cuTensor on GPU.
- `ef.program([...roots...], host=..., boundaries=...)` + `run.precompile/begin_pass/execute` — plan a
  *whole iteration* at once (residency/preloads across many contractions). This is the long-term prize.

**Axis order caveat:** a node's free-axis order is a realization detail (identity is layout-free), so
results must be transposed back to the caller's requested `->` letters by name (the example's
`_transposed` helper does exactly this). Our adapter handles it.

---

## 3. The adapter: mapping `contract(subscripts, *operands)` → `ef`

A minimal, validated adapter (this exact code was run against PyCC patterns in §6):

```python
import numpy as np
import ehrenfest as ef

def ef_contract(subscripts, *operands, run=None):
    """Drop-in for ContractionBackend.__call__ using ef.

    run=None  -> ef.evaluate (oracle/numpy);  run=<ef.runner(...)> -> real tiling executor.
    """
    _, out = subscripts.split("->")
    arrs = [ef.array(tuple(np.shape(o)), f"op{i}") for i, o in enumerate(operands)]
    data = {arrs[i].base.uid: np.asarray(operands[i]) for i in range(len(operands))}
    node = ef.einsum(subscripts, *arrs).node(out)
    val = run(node, data) if run is not None else ef.evaluate(node, data)
    if out:                                    # recover caller's requested axis order by name
        names = [ix.name for ix in node.free]
        val = np.asarray(val).transpose(tuple(names.index(L) for L in out))
    return np.asarray(val)
```

Notes / decisions to make as we go:

- **Plan caching.** The *same subscript strings recur every iteration*. A production adapter should
  cache the built `node` (and, for the runner, the fitted plan) keyed on `(subscripts, shapes, dtype)`.
  `ef.runner` already caches per Contract identity, so reusing one runner instance across iterations
  gets most of this for free; caching `node` construction avoids rebuilding IR each call.
- **Device/precision.** Today `ContractionBackend` also moves operands to GPU and upcasts real↔complex
  per contraction. The `ef` adapter must preserve that policy (or delegate it to `ef`'s placement).
  First integration is **CPU/DP only** — no device or precision behaviour to preserve.
- **Fresh names per call** avoids any operand-aliasing subtlety (e.g. `contract('ia,ia->', x, x)`).

---

## 4. Phased rollout

Each phase is independently shippable and reversible via a runtime switch. Order chosen so risk is low
first and the "planner actually plans" payoff comes only once correctness is nailed.

### Phase 0 — Shadow / validation mode (**start here; zero behavioural risk**)
Add an optional mode to `ContractionBackend` where, for each call, it runs **both** `opt_einsum` (the
real result, still returned) **and** `ef_contract` (via `ef.evaluate`), compares with `np.allclose`,
and logs any mismatch or `ef` exception with the subscript + shapes. Run the existing CCSD test suite
(`test_002`, `test_017`, `test_020`, `test_003`, `test_004`, `test_005`) under this mode.
- **Output:** a coverage report — which of PyCC's real contraction patterns `ef` already computes
  correctly, and a ranked list of the ones it can't (the `ef` backlog, §5). This is the single most
  valuable first artifact and directly serves "show positives and opportunities in `ef`."
- No PyCC numbers change; opt_einsum stays authoritative.

### Phase 1 — `ef.evaluate` as the real backend, CCSD energy, CPU/DP, behind a flag
Add `contraction_backend='ef'` (or an env switch) selecting `ef.evaluate` as the value actually
returned, for `model='CCSD'`, `device='CPU'`, `precision='DP'` only. Gate: `test_002_ccsd_energy`
passes to its existing tolerance. Keep opt_einsum the default.

### Phase 2 — `ef.runner` (the real tiling planner) for the same slice
Swap `ef.evaluate` → a shared `ef.runner(capacity, device="cpu")` instance reused across iterations.
Now PyCC is genuinely *planning* its contractions. Gate: same CCSD tests green; capture per-iteration
timing vs opt_einsum (`pycc/timing.py` already exists) and the runner's plan/`budget()` for the
ladder/ring terms. Expect this to expose path-search and caching behaviour worth reporting upstream.

### Phase 3 — Widen the method surface (still non-local, CPU/DP)
Extend the flag to CCD/CC2, then the post-CCSD chain that reuses the same seam: `cchbar` → `cclambda`
→ `ccdensity`. Each is the same `contract(...)` idiom; the gate is the corresponding numbered test.
`ccresponse`/`cceom` (the two largest call counts) come after, once the core is trusted.

### Phase 4 — Whole-iteration planning via `ef.program`
Replace per-call planning with `ef.program([...])` + `precompile`/`begin_pass`/`execute` so `ef` plans
residency and reuse across a full residual evaluation (the CCSD example already drives it this way).
This is where the planner's cross-contraction wins (peak-live, preloads, shared intermediates) actually
land — the real objective, not just a faster einsum.

### Phase 5+ — The hard/bypass paths
GPU lane; mixed precision; RT-CC (blocked on complex-in-runner, §5); the local (PNO/PAO) per-pair path
(`local.py`/`lccwfn.py`) which needs its own design because shapes vary per pair; `(T)` triples
(`cctriples.py`, threaded `contract` argument — straightforward once the core works).

---

## 5. `ef` improvement backlog (surfaced so far)

Ranked opportunities for the `ehrenfest` side, from the API read and the probes:

1. **Complex dtype through `ef.runner`.** `ef.evaluate` handles complex (it's numpy); the tiling
   runner raises `UFuncTypeError: Cannot cast ufunc 'add' output from complex128 to float64` on a
   complex contraction. **Blocks the entire RT-CC path (`rt/`) from using the real planner.** (Verified
   in §6.)
2. **No symmetry surface on the front-end.** `ehrenfest/symmetry.py` exists but is not re-exported and
   `array`/`einsum`/`runner` take no symmetry annotation. CC is dominated by permutational symmetry
   (`r_T2 += r_T2.swapaxes(0,1).swapaxes(2,3)`, packed `<ab|ef>` ladder); exposing it would be a large
   planner win. Design decision for the `ef` owner (a D*/G*), not something to hand-fix.
3. **No batching concept.** CC3/(T) build per-`ijk` batched `t3` tensors in a triple loop — a batching
   surface would let `ef` plan those as one decision instead of per-iteration Python.
4. **`combine` is sumprod-only.** Fine for standard einsum CC; note it forecloses any max/logsumexp-
   style contraction if that ever arises.
5. **Local/per-pair tensors** have no home in the current IR (fixed-extent `Index` only). The PNO/PAO
   path needs either symbolic extents or a per-pair program story — a genuine design question.
6. **Docs drift.** `ehrenfest`'s reference docs describe the deleted API; anyone integrating will be
   misled. Worth a note upstream (already tracked in their `docs/in-progress/NAMING_AND_SCOPE_DEBT.md`).

(Items 2, 4, 5 are theory/design decisions for the `ehrenfest` owner — this repo will *report* them via
the shadow-mode findings, not attempt them here.)

---

## 6. Feasibility evidence

An adapter identical to §3 was run against representative PyCC contractions (small random arrays,
compared to `np.einsum` with `np.allclose`). Results:

| pattern | source | `ef.evaluate` | `ef.runner` (cpu) |
|---|---|---|---|
| `ia,ia->` (RMS/energy scalar) | `ccwfn.py:278` | ✅ | ✅ |
| `ijab,ijab->` (scalar reduction) | `ccwfn.py` energy | ✅ | ✅ |
| `mnaf,mnef->ae` (Fae build) | `ccwfn.py:492` | ✅ | ✅ |
| `ijcd,acQ,bdQ->abij` (3-operand DF chain) | ef CCSD example | ✅ | ✅ |
| `mf,mafe->ae` | `ccwfn.py:496` | ✅ | ✅ |
| `abc,dbc->ad` (CC3 triples) | `ccwfn.py:423` | ✅ | ✅ |
| `ia,jb->ijab` (outer product, no sum) | `build_tau` | ✅ | ✅ |
| `ijab->ab`, `ijab->ijba` (single-op reduce/transpose) | ubiquitous | ✅ | ✅ |
| `ijij->` (single-op trace) / `iijj->ij` (diagonal) | — | ✅ | ✅ |
| complex-valued `mnaf,mnef->ae` (RT-CC) | `rt/rtcc.py` | ✅ | ❌ `UFuncTypeError` |

**Takeaway:** every real-valued PyCC contraction pattern tested already computes correctly through
both `ef.evaluate` and `ef.runner`, including scalar reductions, chains, outer products, transposes,
and diagonals. `ef.evaluate` additionally handles complex; the runner does not yet. So a CPU/DP CCSD
integration is feasible *today* with `ef.runner`, and `ef.evaluate` is a correctness oracle for shadow
mode.

Probe scripts (kept out of the repo) live in the session scratchpad; they can be promoted into
`pycc/tests/` as a differential `test_0xx_ef_backend.py` in Phase 0.

---

## 7. Concrete first step (next iteration)

1. Add `ef` as an **optional** dependency (CPU path needs only `numpy` + `threadpoolctl`; no psi4, no
   torch). Decide: vendored path vs. installed package vs. git submodule of `ehrenfest`.
2. Implement Phase 0 shadow mode in `pycc/device.py` (`ContractionBackend.__call__`), guarded by an env
   var / constructor kwarg, defaulting **off**.
3. Add `pycc/tests/test_0xx_ef_backend.py`: run `test_002`'s water/cc-pVDZ CCSD under shadow mode and
   assert the mismatch log is empty (skipped if `ehrenfest` not importable).
4. Produce the first coverage report → feed §5 upstream to `ehrenfest`.

---

## 8. Open questions for the maintainers

- **Dependency shape:** how should PyCC depend on `ehrenfest` (still pre-release)? Optional extra +
  runtime `ImportError` guard is assumed here.
- **Scope of ambition per release:** confirm the crawl order (CCSD energy → lambda/density →
  response/EOM) matches your priorities.
- **RT-CC / complex:** is unblocking complex in the `ef` runner worth prioritising on the `ef` side, or
  do we keep RT-CC on opt_einsum indefinitely?
- **Local CC:** treat the PNO/PAO per-pair path as a separate, later design project (it needs `ef`
  features that don't exist yet)?
