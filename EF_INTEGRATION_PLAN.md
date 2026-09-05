# PyCC × ehrenfest (`ef`) integration plan

**Status:** living document — iteration 2 (folds in a critical review pass with empirical timings).
Expect further revisions.
**Constraint:** `ef` (the `ehrenfest` package) has ongoing work and is **not modified by this effort** —
we integrate against `ef` *as it currently exists*. Gaps we hit are reported to the `ef` team, not
patched here.

## 0. What this can and cannot deliver (read this first)

With `ef` frozen at its current state, this integration **cannot make PyCC faster or more capable in the
near term**. Measured on realistic CCSD sizes (occupied `o=5`, virtual `v=20`, CPU/DP), `ef.runner` is
**5–11× slower per contraction** than the `np.einsum(optimize)`-class path PyCC already uses via
`opt_einsum` (see [§6](#6-feasibility-and-performance-evidence)). `ef`'s planner wins (peak-live,
capacity-aware placement, cross-contraction reuse) are asymptotic and memory-bound; they do not appear
at the sizes the CPU/DP test suite exercises, and they only materialize through whole-iteration planning
(Phase 4), which is a substantial re-expression of the equations, not a drop-in.

So the honest, near-term value of this work is:

1. **A real coupled-cluster harness for `ef`.** PyCC's production equations become a stress test that
   surfaces concrete, ranked requirements for the `ef` team (§5).
2. **Establishing and regression-guarding the seam**, so PyCC is ready to benefit the moment `ef`
   matures — the switch is built, tested, and off by default.
3. **Differential validation** of `ef` against equations it was not written against.

If the goal were near-term PyCC benefit, this plan cannot provide it with `ef` as-is; that expectation
should be set explicitly. What follows is scoped to items 1–3.

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

There are **~1776 `contract(` call sites across 22 files**, all funnelling through that one object
(`ccresponse.py` 479, `cctriples.py` 342, `ccwfn.py` 189, `ccdensity.py` 182, `cclambda.py` 134,
`cchbar.py` 107, `cceom.py` 99, …). **Consequence:** the engine for the entire non-local codebase can be
swapped by editing one method, `ContractionBackend.__call__`, without touching a single equation.

### The ef path is a *branch inside* `__call__`, not a replacement

`ContractionBackend.__call__` today also handles the GPU/torch route (moving operands to the compute
device, real↔complex upcasts). The `ef` adapter is numpy-only, so it must be an **added branch** guarded
by the flag and the CPU/DP/numpy precondition — the existing opt_einsum and GPU paths stay intact and
default.

### Seams that bypass `self.contract` (handle separately, later)

- `pycc/local.py:8` and `pycc/lccwfn.py:8` — module-level `from opt_einsum import contract`
  (device-unaware). The **local (PNO/PAO)** path also stores amplitudes as *Python lists of small
  per-pair tensors* in per-pair truncated subspaces (`lccwfn.py:265-333`); shapes vary per pair — a
  different integration problem that needs `ef` features it does not yet expose. Deferred.
- `pycc/rt/rtcc.py:469` — one direct `opt_einsum.contract`. RT-CC is **complex-valued**; blocked on the
  `ef.runner` complex gap (§5).

---

## 2. What `ef` can do today (the target API)

The old `ef.report`/`ef.simulate`/`ef.GTArray`/`auto_plan` API described in `ehrenfest`'s reference docs
**no longer exists** — that stack was deleted and rebuilt around a "tiling" engine that actually
executes. Trust the code (`frontend.py`, `runner.py`, `oracle.py`) and the live example under
`ehrenfest/examples/chemistry/coupledCluster/ccsd/`, not the reference prose.

Relevant public surface (`ehrenfest/__init__.py`):

- `ef.array(shape, name, *, invariant=False)` — a named leaf tensor. `.base.uid` keys the data dict.
- `ef.einsum(spec, *operands) -> expr` — an einsum-spelled contraction chain (n-ary; planner picks the
  binary order). **Whitespace-sensitive and case-sensitive** — see §3.
- `expr.node(out) -> Sum/Contract` — lower to one IR node; `out` names each free letter once.
- **Two ways to get numbers back**, both taking `{base.uid: ndarray}`:
  - `ef.evaluate(node, arrays)` — the **oracle**: `np.einsum` under the hood. Pure-numpy, handles
    complex, matches `np.einsum` to machine precision. Its *only* role here is as a validation oracle;
    as a "backend" it is strictly worse than PyCC's existing `opt_einsum` (no path optimization).
  - `ef.runner(capacity, *, device="cpu"|"gpu")(node, arrays)` — the **real planner/executor**:
    `path → binarize → fit_chain → execute`, caching the fitted plan **per node-object identity**
    (critical — see §3). Returns a numpy array on the CPU lane.
- `ef.program([...roots...], host=..., boundaries=...)` + `run.precompile/begin_pass/execute` — plan a
  *whole iteration* at once. This is the only route to the planner's real wins, and the hardest (Phase 4).

**Axis order:** a node's free-axis order is a realization detail (identity is layout-free), so results
must be transposed back to the caller's `out` letters by name. **`capacity`:** set it at or above the
working-set size (e.g. total RAM budget in bytes) so it does not force tiling/spill; a too-small value
changes execution behaviour. The adapter below uses a large default.

---

## 3. The adapter: mapping `contract(subscripts, *operands)` → `ef`

This version folds in three defects found while reviewing the naïve adapter (whitespace, cache-defeat,
GPU/torch preconditions). **It is not yet a drop-in — it is the reference the Phase-0/Phase-2 code
should implement.**

```python
import numpy as np
import ehrenfest as ef

class EfContractionAdapter:
    """Routes contract(subscripts, *operands) through ef, for the CPU/DP/numpy case only.

    Correctness- and cache-critical details, all verified against ef (see §6):
      * ef.einsum is whitespace-sensitive: '-> be' would create an index named ' '.
        Strip ALL spaces from the spec before ef.einsum. (numpy tolerates them; ef does not.)
      * ef.einsum is case-sensitive: 'E' and 'e' are distinct indices (pycc uses both). Do NOT
        lowercase.
      * ef.runner caches the fitted plan per NODE-OBJECT identity, not structurally. Building a fresh
        ef.array/node every call replans every call (measured ~1.7 ms vs ~0.46 ms reused). So the
        built node MUST be memoized on (clean_spec, shapes, dtype) and the same object reused across
        iterations; reusing the runner alone is not enough.
    """

    def __init__(self, capacity=1 << 34, device="cpu", oracle=False):
        self._run = None if oracle else ef.runner(capacity, device=device)
        self._cache = {}                       # (clean_spec, shapes, dtype) -> (node, arrays_by_pos)

    def __call__(self, subscripts, *operands):
        spec = subscripts.replace(" ", "")     # ef is whitespace-sensitive; numpy is not
        out = spec.split("->")[1]
        shapes = tuple(tuple(np.shape(o)) for o in operands)
        dtype = np.result_type(*[np.asarray(o).dtype for o in operands])
        key = (spec, shapes, dtype)
        node, arrs = self._cache.get(key, (None, None))
        if node is None:                        # build once, reuse the node object forever after
            arrs = [ef.array(shapes[i], f"op{i}") for i in range(len(operands))]
            node = ef.einsum(spec, *arrs).node(out)
            self._cache[key] = (node, arrs)
        data = {arrs[i].base.uid: np.asarray(operands[i]) for i in range(len(operands))}
        val = self._run(node, data) if self._run is not None else ef.evaluate(node, data)
        if out:                                 # recover caller's requested axis order by name
            names = [ix.name for ix in node.free]
            val = np.asarray(val).transpose(tuple(names.index(L) for L in out))
        return np.asarray(val)
```

Remaining known overheads (inherent, not bugs): the out-order **transpose adds a copy** per non-scalar
contraction (opt_einsum already returns in `out` order), on top of the planning/execution gap in §0.

Open decisions: whether to cache per-`ContractionBackend` or globally; how to key when the same spec
recurs at genuinely different dtypes (RT complex vs DP); how large `capacity` should be relative to the
PyCC problem.

---

## 4. Phased rollout

Each phase is independently shippable and reversible via a runtime switch, **off by default**.

### Phase 0 — Signature capture + offline coverage (**start here; zero behavioural risk**)
Do **not** run both engines inline across the whole suite — that doubles ~1776 sites × iterations ×
`ef`'s replan overhead and is impractical. Instead:
1. Add a thin logging shim in `ContractionBackend.__call__` (flag-guarded, off by default) that records
   each unique `(clean_spec, shapes, dtype)` from **one small** CCSD run (water/cc-pVDZ, `test_002`).
2. Replay those captured signatures **offline** against `ef` in a standalone test, comparing
   `ef.evaluate` (and, where it runs, `ef.runner`) to `opt_einsum`/`np.einsum` with a stated tolerance.
- **Output:** a coverage report — which real PyCC contraction patterns `ef` computes correctly, and a
  ranked list of the ones it cannot (feeds §5). This is the single most valuable first artifact and
  decouples `ef` coverage from the live CC loop. `pycc/tests/test_024_contract_cpu.py` already exists as
  a backend-equivalence test and is the natural pattern/home.
- No PyCC numbers change; `opt_einsum` stays authoritative.

### Phase 1 — `ef.runner` as the real backend, CCSD energy, CPU/DP, behind a flag
(The earlier draft had a separate `ef.evaluate`-as-backend phase; dropped — `ef.evaluate` is `np.einsum`
and strictly worse than the current `opt_einsum`, so it earns no milestone. `ef.evaluate` is the Phase-0
oracle only.) Add `contraction_backend='ef'` (or an env switch) selecting the `EfContractionAdapter`
(runner mode) as the value actually returned, for `model='CCSD'`, `device='CPU'`, `precision='DP'` only.
Gate: `test_002_ccsd_energy` passes to its existing tolerance. Record per-iteration timing vs
`opt_einsum` (expected slower — that is a reported datapoint, not a failure) and the runner's
`plan`/`budget()` for the ladder/ring terms.

### Phase 2 — Widen the method surface (still non-local, CPU/DP)
Extend the flag to CCD/CC2, then the post-CCSD chain that reuses the same seam: `cchbar` → `cclambda` →
`ccdensity`. Each is the same `contract(...)` idiom; the gate is the corresponding numbered test.
`ccresponse`/`cceom` (the two largest call counts) come after the core is trusted.

### Phase 3 — Whole-iteration planning via `ef.program` (**a different kind of work — flagged**)
This is **not** a seam swap. It requires re-expressing the residual equations *symbolically* (building
`ef` exprs the way `ehrenfest`'s own `equations.py` does) and declaring host/device boundaries, which
touches the CC routines themselves, not `device.py`. It also depends on `ef` program features maturing.
It is, however, the **only** phase where the planner's cross-contraction wins (peak-live, preloads,
shared intermediates) actually land. Treat it as a separate project gated on both the earlier phases and
`ef` readiness — not one more rung on the same ladder.

### Phase 4+ — The hard/bypass paths
GPU lane; mixed precision; RT-CC (blocked on complex-in-runner, §5); the local (PNO/PAO) per-pair path
(`local.py`/`lccwfn.py`), which needs `ef` features that don't exist yet; `(T)` triples (`cctriples.py`,
threaded `contract` argument — straightforward once the core works).

---

## 5. `ef` requirements surfaced (reported upstream, not fixed here)

Ranked, from the API read and the probes. These are inputs to the `ef` team's own roadmap.

1. **Performance at CC sizes / caching ergonomics.** `ef.runner` is 5–11× slower than `np.einsum(opt)`
   at `o=5,v=20` (§6), and its plan cache keys on node-object identity, so a naïve adapter replans every
   call. Both are the dominant blockers to any real PyCC use. (We route around the second with the
   memoizing adapter; the first is inherent at these sizes.)
2. **Complex dtype through `ef.runner`.** `ef.evaluate` handles complex; the runner raises
   `UFuncTypeError: Cannot cast ufunc 'add' output from complex128 to float64`. **Blocks the entire
   RT-CC path (`rt/`).** (Verified in §6.)
3. **No symmetry surface on the front-end.** `ehrenfest/symmetry.py` exists but is not re-exported and
   `array`/`einsum`/`runner` take no symmetry annotation. CC is dominated by permutational symmetry
   (`r_T2 += r_T2.swapaxes(0,1).swapaxes(2,3)`, packed `<ab|ef>` ladder); exposing it is a large planner
   win.
4. **No batching concept.** CC3/(T) build per-`ijk` batched `t3` tensors in a triple loop; a batching
   surface would let `ef` plan those as one decision.
5. **`combine` is sumprod-only** (fine for standard einsum CC; forecloses max/logsumexp-style).
6. **Local/per-pair tensors** have no home in the current fixed-extent `Index` IR — the PNO/PAO path
   needs symbolic extents or a per-pair program story.
7. **Docs drift** — the reference docs describe the deleted API; already tracked in their
   `docs/in-progress/NAMING_AND_SCOPE_DEBT.md`.

---

## 6. Feasibility and performance evidence

Adapters identical to §3 were run against representative PyCC contractions (random arrays, compared to
`np.einsum` with `np.allclose`).

**Correctness** — every real-valued PyCC pattern passes through both entry points:

| pattern | source | `ef.evaluate` | `ef.runner` |
|---|---|---|---|
| `ia,ia->`, `ijab,ijab->` (scalar reductions) | `ccwfn.py` RMS/energy | ✅ | ✅ |
| `mnaf,mnef->ae`, `mf,mafe->ae` (intermediates) | `ccwfn.py:492/496` | ✅ | ✅ |
| `ijcd,acQ,bdQ->abij` (3-operand DF chain) | ef CCSD example | ✅ | ✅ |
| `abc,dbc->ad` (CC3 triples) | `ccwfn.py:423` | ✅ | ✅ |
| `ia,jb->ijab` (outer product), `ijab->ab`/`ijba` (reduce/transpose) | ubiquitous | ✅ | ✅ |
| `ijij->`, `iijj->ij` (trace / diagonal) | — | ✅ | ✅ |
| `E,F,mEF->m` (uppercase indices) | `ccresponse.py` | ✅ | — |
| non-contiguous operand (a `swapaxes` view) | ubiquitous | ✅ | — |
| complex-valued `mnaf,mnef->ae` (RT-CC) | `rt/rtcc.py` | ✅ | ❌ `UFuncTypeError` |
| `f,b,ef-> be` (whitespace in spec) | `ccresponse.py` | ✅ *after stripping spaces* | — |

**Performance** (per call, warm; `ef.runner` reused; CPU):

| contraction (`o=5,v=20`) | `ef.runner` | `np.einsum(optimize)` | ratio |
|---|---|---|---|
| `ijef,abef->ijab` (v⁴ ladder) | 1.78 ms | 0.15 ms | **11.5× slower** |
| `abef,ie,jf->ijab` (3-op chain) | 2.13 ms | 0.43 ms | **5.0× slower** |

**Caching** (`ijef,abef->ijab`, reused runner): fresh node per call **1.67 ms** vs reused node object
**0.46 ms** (node-build alone 0.07 ms) — confirming the runner replans unless the *node object* is
memoized, and that even perfectly-cached it stays ~3× slower than numpy at this size.

**Numerical agreement:** on the ladder term `ef.runner` matched `np.einsum(optimize)` to `0.0e0` max abs
diff (exact). So tolerance is likely a non-issue, but Phase 0 should confirm it across patterns rather
than assume it.

Probe scripts live in the session scratchpad; they can be promoted into a differential
`pycc/tests/test_0xx_ef_backend.py` in Phase 0.

---

## 7. Concrete first step (next iteration)

1. Add `ef` as an **optional** dependency (CPU path needs only `numpy` + `threadpoolctl`; no psi4, no
   torch). Decide: installed package vs. git submodule of `ehrenfest`; guard with a lazy `ImportError`.
2. Implement Phase 0: a flag-guarded signature-capture shim in `pycc/device.py`, defaulting **off**,
   plus an offline replay test that produces the first coverage report.
3. Feed the coverage report + §5 to the `ef` team.

---

## 8. Open questions for the maintainers

- **Dependency shape:** how should PyCC depend on `ehrenfest` (still pre-release)? Optional extra +
  runtime `ImportError` guard is assumed.
- **Is the harness worth it now,** given `ef` is currently slower and unmodifiable — i.e. do you value
  items 0.1–0.3 enough to carry a permanently-off-by-default backend until `ef` matures?
- **Crawl order:** confirm CCSD energy → lambda/density → response/EOM matches your priorities.
- **RT-CC / local CC:** keep both on `opt_einsum` indefinitely until `ef` grows complex-runner support
  and per-pair/symbolic-extent tensors?
