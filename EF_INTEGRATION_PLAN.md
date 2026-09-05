# PyCC × ehrenfest (`ef`) integration plan

**Status:** living document — iteration 3. Plan-only; no code beyond the reference adapter in §3.
**Constraint:** `ef` (the `ehrenfest` package) has ongoing work and is **not modified by this effort** —
we integrate against `ef` *as it currently exists*. Gaps we hit are reported to the `ef` team, not
patched here.
**EF baseline for this iteration:** `ehrenfest` branch `claude/tiling-from-scratch` at commit
`da4d2d9` — every capability and failure mode cited below was checked against that exact tree.

---

## 0. Two different questions — do not conflate them

This plan must keep two questions apart, because the evidence answers them very differently:

1. **Is EF a competitive drop-in einsum implementation?** i.e. can a per-contraction adapter that
   forwards each `contract(subscripts, *operands)` call to `ef.runner` beat PyCC's current
   `opt_einsum` path? **On the tested small CPU/DP contractions, no** — a per-contraction CPU/DP adapter
   around `ef.runner` is 5–11× slower than `np.einsum(optimize)` at `o=5, v=20`
   (see [§7](#7-evidence)). The single-contraction seam should be understood as a
   **compatibility / validation path** and is *expected* to lose on small CPU contractions, because it
   shows EF one isolated contraction at a time with none of the cross-contraction context its planner
   exists to exploit.

2. **Can EF's graph/program execution model benefit PyCC?** i.e. when EF sees a whole
   equation/iteration — multiple contractions, shared subexpressions, cuts, accumulation, host
   boundaries, residency and reuse — can it do better than PyCC term-by-term? **This is an open
   empirical question, and it is the interesting one.** The per-contraction timings say essentially
   nothing about it. EF's current branch already *implements and runs* this program-native path for its
   own CCSD and CCSD(T): `examples/chemistry/coupledCluster/ccsd/` expresses dressing, residuals,
   denominator construction, the Jacobi update and much of DIIS as EF graphs and runs the main iteration
   as **one composed program** (`assemble.py`), and its README reports real GPU cc-pVDZ/pVTZ/pVQZ runs;
   `examples/chemistry/coupledCluster/ccsdpT/` has a working spin-free `(T)` graph with automatic
   CSE/fusion of its permutation-heavy expression.

**What we may and may not claim.** The EF example timings are **not** an apples-to-apples PyCC
benchmark — different algorithm, representation, and hardware — so they must never be reported as a PyCC
speedup. What they *do* establish is that whole-program optimization is a real, exercised capability at
chemistry scale. Whether *PyCC* benefits therefore requires an apples-to-apples PyCC vertical slice
(Track B, §4), not an extrapolation from EF's own examples.

So the earlier blanket statement — "with EF frozen this integration cannot make PyCC faster or more
capable" — is **withdrawn as too categorical**. The accurate statement is: the *drop-in single-
contraction seam* is currently unattractive on small CPU contractions; the *program-native path* is a
present, implemented EF capability whose value to PyCC is untested and worth an early experiment.

---

## 1. The single contraction seam (the compatibility path)

Every heavy contraction in the non-local PyCC paths goes through one callable, `self.contract`, an
instance of `ContractionBackend` (`pycc/device.py:38`). Its signature is already the einsum surface EF
consumes:

```python
contract(subscripts, *operands) -> ndarray   # e.g. contract('ijef,abef->ijab', tau, ERI[v,v,v,v])
```

- Created by `DeviceManager.__init__` (`pycc/device.py:154`), exposed as `self.contract` on the
  `Wavefunction` base (`pycc/wavefunction.py:159`); every sub-object rebinds it
  (`self.contract = self.ccwfn.contract`), ~200 sites, and the `(T)` kernels take it as an argument
  (`pycc/cctriples.py:192, 274, 321`). ~1776 `contract(` call sites across 22 files funnel through it.

This is the ideal seam **for the compatibility path only**: swapping the engine for the whole non-local
codebase is one method edit, no equation touched. It is **not** the route to EF's planner value — see §4
Track B and the semantic-identity caveat in §3. The EF path must be an **added branch** inside
`__call__`, guarded by the flag and a CPU/DP/numpy precondition; the existing `opt_einsum` and GPU/torch
routes stay intact and default.

Bypass seams to handle later: `pycc/local.py:8` and `pycc/lccwfn.py:8` (module-level
`from opt_einsum import contract`; the PNO/PAO path also stores per-pair lists of small tensors,
`lccwfn.py:265-333`), and `pycc/rt/rtcc.py:469` (one direct call; RT-CC is complex — §6).

---

## 2. What EF actually exposes today (three levels of use)

The reference docs describe a deleted API; trust `frontend.py`, `runner.py`, the `planning` module,
`oracle.py`, and the CCSD example. The current public surface is substantially richer than
"array/einsum/evaluate/runner/program", and the **reference integration architecture for program-native
coupled cluster is `examples/chemistry/coupledCluster/ccsd/assemble.py`** — the plan should be read
alongside it.

### 2a. Frontend / IR construction
- `ef.array(shape, name, *, invariant=False)` — a named leaf; `invariant` tags a cross-iteration
  lifetime (e.g. integrals that don't change between amplitude updates). `.base.uid` keys the data dict.
- `ef.einsum(spec, *operands) -> expr` — n-ary einsum chain (planner picks the binary order).
  Whitespace- and case-sensitive (§3). Has an **automatic blow-up/CSE guard** for products of multi-term
  expressions (it auto-cuts to keep an expanded product from exploding, and shares repeats).
- `expr.node(out) -> Sum` — lower to a complete `Sum` IR node; `out` names each free letter once.
- `ef.cut(e, out, name)` / `ef.cuts_of(node)` — a **cut** is a named, reusable computed value; declaring
  one lets the planner decide *materialize vs. fuse* for that value rather than baking the choice in.
- `ef.map(operand, fn)` — elementwise maps (`exp, reciprocal, square, relu, sqrt, neg`).
- `ef.broadcast(e, extents)`, `ef.relabel(e, pairs)`.
- `ef.perm_sum(e, group)` — a **chemistry-facing permutation surface** (the `P(ij)`/`P(ab)` antisymmetric
  sums that pervade CC residuals), expressed directly rather than as hand-written `swapaxes` adds.
- The frontend **canonicalizes α-equivalent monomials**, so permutation partners (P-partners) and other
  structurally identical terms **share the same contraction node** automatically — the basis of EF's CSE.

### 2b. Getting numbers back — three levels, not two
1. **Per-node runner / cache** — `ef.runner(capacity, *, device="cpu"|"gpu")(node, arrays)`: one node
   through `path → binarize → fit_chain → execute`, caching the fitted plan **per node-object identity**
   (§3). `ef.evaluate(node, arrays)` is the numpy oracle — **validation only**, never a backend (it is
   `np.einsum`, strictly worse than PyCC's `opt_einsum`).
2. **Ordered precompiled node set** — `run.precompile(items, …)` gives *program order* significance:
   input residency and scheduled GPU preloads are planned across multiple runnable nodes, not per call.
3. **Composed `ef.program` forest** — `ef.program([...roots...], host=..., boundaries=...)` builds a
   forest of cuts + roots, finds **explicit and implicit shared work** across the whole forest, models
   caller **host reads** and **host-availability boundaries**, and supports **composed execution** via
   `run.begin_pass(arrays)` / `run.execute(through=…)`. This is where *materialize-vs-fuse*, cross-term
   CSE, and residency decisions are actually made — the level PyCC would target for real benefit.

`ef.evaluate` handles complex and matches `np.einsum` to machine precision; `capacity` should be set at
or above the working set (e.g. a RAM/VRAM budget in bytes) so it does not force tiling/spill.

---

## 3. The adapter (compatibility path) — reference, with limits stated

Folds in the iteration-2 fixes (whitespace, case, node-object reuse, CPU/DP restriction) and corrects
one earlier over-claim. **This is the reference the Track-A code implements, not a drop-in yet.**

```python
import numpy as np
import ehrenfest as ef

class EfContractionAdapter:
    """Route contract(subscripts, *operands) through ef, CPU/DP/numpy only (compatibility path).

    Verified against ef @ da4d2d9 (see §7):
      * ef.einsum is whitespace-sensitive ('-> be' makes an index named ' '); normalize ALL
        whitespace, not just ASCII spaces: ''.join(subscripts.split()). numpy tolerates spaces; ef does
        not. (If you deliberately scope to the exact PyCC corpus, a literal-space strip suffices — but
        robust normalization costs nothing.)
      * ef.einsum is case-sensitive: 'E' != 'e' (pycc uses both). Do NOT lowercase.
      * ef.runner caches the fitted plan per NODE-OBJECT identity, not structurally. Rebuilding a fresh
        node every call replans every call (measured ~1.7 ms vs ~0.46 ms reused). Memoize the node on
        (clean_spec, shapes, dtype) and reuse the SAME object across iterations; reusing the runner is
        not enough.
    """

    def __init__(self, capacity=1 << 34, device="cpu", oracle=False):
        self._run = None if oracle else ef.runner(capacity, device=device)
        self._cache = {}

    def __call__(self, subscripts, *operands):
        spec = "".join(subscripts.split())          # robust whitespace normalization; case preserved
        out = spec.split("->")[1]
        shapes = tuple(tuple(np.shape(o)) for o in operands)
        dtype = np.result_type(*[np.asarray(o).dtype for o in operands])
        key = (spec, shapes, dtype)
        node, arrs = self._cache.get(key, (None, None))
        if node is None:
            arrs = [ef.array(shapes[i], f"op{i}") for i in range(len(operands))]
            node = ef.einsum(spec, *arrs).node(out)
            self._cache[key] = (node, arrs)
        data = {arrs[i].base.uid: np.asarray(operands[i]) for i in range(len(operands))}
        val = self._run(node, data) if self._run is not None else ef.evaluate(node, data)
        if out:
            names = [ix.name for ix in node.free]
            val = np.asarray(val).transpose(tuple(names.index(L) for L in out))  # normally a zero-copy view
        return np.asarray(val)
```

**Two limitations to state plainly:**

1. **This seam does not expose EF's CSE/planner potential, and must not be described as if it did.** The
   `(clean_spec, shapes, dtype)` memoization stabilizes node identity for *that one PyCC call
   signature*. It does **not** reproduce the semantic leaf identity and whole-expression
   canonicalization EF gets when an equation is built **once** as a frontend graph (where shared leaves
   and α-equivalent monomials across *different* terms collapse to shared nodes). Cross-term CSE,
   materialize-vs-fuse, and residency are program-level properties this per-call seam structurally
   cannot see. Getting them is Track B, not this adapter.
2. The out-order transpose is **normally a zero-copy view** (`np.transpose`), not an inherent copy —
   EF's own alignment APIs rely on zero-copy transposes. A later consumer forcing contiguity may copy,
   but the transpose itself does not. (The earlier "adds a copy per contraction" claim is withdrawn.)

Keep the CPU/DP restriction throughout.

---

## 4. Two tracks (run in parallel; they answer different questions)

### Track A — drop-in compatibility seam (harness / regression)
- **A0.** Capture real PyCC contraction signatures `(clean_spec, shapes, dtype)` from one small CCSD run
  (water/cc-pVDZ, `test_002`) via a flag-guarded logging shim (off by default), then **replay them
  offline** against `ef.evaluate` and `ef.runner`, comparing to `opt_einsum`/`np.einsum` at a stated
  tolerance. Output: a coverage report of what EF computes correctly + a ranked gap list (§6).
  `pycc/tests/test_024_contract_cpu.py` is the natural pattern/home.
- **A1.** Optional CPU/DP `ef.runner` backend behind a flag for *one* CCSD energy test (`test_002`),
  off by default. Its benchmark stays explicitly labeled **"per-contraction adapter performance"** — it
  is *not* a benchmark of EF program-native execution.
- **A2.** Widen to more methods **only if maintaining the seam itself proves valuable** (as a
  differential oracle / regression guard). Not a goal in its own right.

### Track B — program-native PyCC integration (the real experiment)
Do **not** make Track B wait on Track A widening to lambda/density/response/EOM — they answer different
questions. The EF CCSD implementation proves the APIs needed to prototype B0 already exist, so this is
**not** "pending EF program features maturing"; the composed-program mechanism is present and exercised
(there may still be performance/integration gaps, which is exactly what B measures).
- **B0.** Build **one narrow PyCC vertical slice at an equation/iteration boundary** — e.g. one residual
  plus the denominator/Jacobi update — expressed with **stable `ef.array` leaves** and **one composed
  `ef.program`**, numerically checked against existing PyCC. This is dozens of leaves and a handful of
  roots, **not** 1776 individual calls. `assemble.py` is the architectural template.
- **B1.** Measure **cold planning/compile separately from warm iteration wall time and peak memory** —
  break out IR/node construction, planning, compilation, first execution, warm execution, whole-
  iteration wall time, and peak memory/transfers where available. (EF's own note: search "costs minutes
  per program today — TZ 239 s for ~8 s saved over a 16-iteration run," so cold-vs-warm and decision
  persistence dominate the verdict.)
- **B2.** Only after correctness, try an **EF-owned GPU execution path** for that region (§5).

---

## 5. GPU ownership boundary (explicit)

PyCC's existing GPU path is **Torch-based**: `DeviceManager` stores/places tensors and
`ContractionBackend` moves operands to the compute device and handles real↔complex promotion. EF's GPU
runner wants to **own** its planning, transfers, residency, preloads and cuTensor execution.

Therefore a serious EF GPU integration must **not** be modeled as another branch inside PyCC's
per-contraction Torch `ContractionBackend.__call__` — that would defeat EF's residency/program logic and
create two competing device managers fighting over placement. The likely GPU seam is the
**residual/update or iteration boundary** (Track B), where EF owns the program *and* its data movement
for that region. Keep the existing Torch backend untouched for the normal PyCC path; the EF GPU path is
a *regional handoff*, not a per-call substitution.

---

## 6. EF gaps relevant to PyCC (reported upstream, verified against `da4d2d9`)

Written narrowly against the current branch — several iteration-2 items were too broad.

1. **Per-contraction runner cost + cache ergonomics.** `ef.runner` is 5–11× slower than
   `np.einsum(opt)` at `o=5,v=20` on isolated contractions (§7), and its plan cache keys on node-object
   identity. Both bite the *compatibility* seam specifically; Track B sidesteps them by construction.
2. **Complex dtype through `ef.runner`.** *Verified* failure on this branch:
   `UFuncTypeError: Cannot cast ufunc 'add' output from complex128 to float64` — the runner still has
   hard-coded float64 paths. `ef.evaluate` handles complex. This is a **blocker for RT-CC** (`rt/`).
   (Stated as the verified failure mode, distinct from architectural inference — general complex
   support should not be assumed either way.)
3. **SP / general numerical-dtype *execution* through the runner** (distinct from planner-side precision
   *costing*). PyCC's single/mixed-precision paths need the runner to actually execute in the requested
   dtype; confirm what the runner supports before relying on it.
4. **Declared input-tensor symmetry.** EF already has `relabel`/`perm_sum`, automatic derivation of
   structural permutation invariances, and orbit-domain reduction in the tiler. What
   `ehrenfest/symmetry.py` explicitly notes as *not yet implemented* is **declared input symmetry** —
   e.g. letting a `t2` leaf's coupled axis swap participate as an extra allowed leaf-axis permutation in
   the matcher. That specific declared-leaf-symmetry surface is the gap, **not** "no symmetry."
5. **CUDA leaf-view coverage.** The EF CCSD code calls out a specific limit: a **diagonal operand view
   is not yet emitted on the CUDA lane** (`equations.py:133-135`, deferred to its Phase 3), so a
   diagonal is taken host-side. PyCC uses diagonal/sliced/`swapaxes` views heavily; confirm which
   leaf-view forms the CUDA lane emits before a GPU Track-B slice depends on them.
6. **Persistence of expensive planner decisions across runs.** Program search and materialize-vs-fuse
   scoring cost far more than they save over a single short solve (EF's own TZ note above). For PyCC use,
   amortizing/persisting those decisions across runs matters.
7. **Local / PNO per-pair varying extents** (fixed-extent `Index` IR) — deferred; re-audit against
   current code before treating as firm.
8. *(Demoted)* **`combine` is sumprod-only** — not a ranked blocker: standard PyCC CC needs no other
   reduction algebra. Note it only if a concrete PyCC operation ever needs one.
9. **Docs drift** — reference docs describe the deleted API (tracked in EF's
   `docs/in-progress/NAMING_AND_SCOPE_DEBT.md`).

Batching note: do **not** claim EF has "no batching / no triples story" — a working spin-free CCSD(T)
graph with CSE/fusion exists (`ccsdpT/`). Any real gap is narrower — e.g. PyCC/CC3-style **dynamic
per-`ijk` program families** — and only if it survives mapping the equations onto EF's surface.

---

## 7. Evidence

### 7a. Drop-in CPU per-contraction seam benchmark — *not* a benchmark of EF program-native execution
Adapter per §3, random arrays, compared to `np.einsum` with `np.allclose`.

**Correctness** — every real-valued PyCC pattern passes through both entry points; complex passes
through `evaluate` only:

| pattern | source | `ef.evaluate` | `ef.runner` |
|---|---|---|---|
| `ia,ia->`, `ijab,ijab->` (scalar reductions) | `ccwfn.py` RMS/energy | ✅ | ✅ |
| `mnaf,mnef->ae`, `mf,mafe->ae` (intermediates) | `ccwfn.py:492/496` | ✅ | ✅ |
| `ijcd,acQ,bdQ->abij` (3-operand DF chain) | EF CCSD example | ✅ | ✅ |
| `abc,dbc->ad` (CC3 triples) | `ccwfn.py:423` | ✅ | ✅ |
| `ia,jb->ijab` (outer), `ijab->ab`/`ijba` (reduce/transpose) | ubiquitous | ✅ | ✅ |
| `ijij->`, `iijj->ij` (trace / diagonal) | — | ✅ | ✅ |
| `E,F,mEF->m` (uppercase indices) | `ccresponse.py` | ✅ | — |
| non-contiguous operand (`swapaxes` view) | ubiquitous | ✅ | — |
| `f,b,ef-> be` (whitespace in spec) | `ccresponse.py` | ✅ *after normalizing spaces* | — |
| complex `mnaf,mnef->ae` (RT-CC) | `rt/rtcc.py` | ✅ | ❌ `UFuncTypeError` (verified @ `da4d2d9`) |

**Per-contraction performance** (per call, warm, `ef.runner` reused, CPU, `o=5,v=20`):

| contraction | `ef.runner` | `np.einsum(optimize)` | ratio |
|---|---|---|---|
| `ijef,abef->ijab` (v⁴ ladder) | 1.78 ms | 0.15 ms | 11.5× slower |
| `abef,ie,jf->ijab` (3-op chain) | 2.13 ms | 0.43 ms | 5.0× slower |

**Caching:** fresh node per call 1.67 ms vs reused node object 0.46 ms (node-build alone 0.07 ms) —
confirming per-node-identity caching; even perfectly cached it stays ~3× slower than numpy at this size.
**Numerics:** `ef.runner` matched `np.einsum(optimize)` to `0.0e0` on the ladder; Track A should still
confirm tolerance across patterns rather than assume it.

*Future performance tables (Track B especially) must break out these components separately:* IR/node
construction · planning · compilation · first (cold) execution · warm execution · whole-iteration wall
time · peak memory / transfers.

### 7b. Current EF capability evidence (feasibility of the program-native model — *not* a PyCC speedup)
- `examples/chemistry/coupledCluster/ccsd/` runs the CCSD iteration as **one composed `ef.program`**
  (dressing + residuals + denominators + Jacobi update + much of DIIS); README reports real GPU runs,
  e.g. benzene/cc-pVDZ (`o=21, v=93, naux=420`) CCSD 13 iterations in ~9.7 s (~0.45 s/iter), energy
  Δ ≈ 6.4e-9 vs reference, plus cc-pVTZ/pVQZ sections.
- `examples/chemistry/coupledCluster/ccsdpT/` — a working **spin-free `(T)`** graph with automatic
  CSE/fusion of the permutation-heavy expression.
These demonstrate the program-native execution model is real and exercised at chemistry scale. They are
EF-vs-reference on EF's own representation/hardware — **not** an apples-to-apples PyCC comparison, and
must not be cited as a PyCC speedup. They justify running Track B; they do not pre-judge its result.

---

## 8. Dependency handling
- Keep `ef` **optional and lazy-imported**; the default PyCC install and test run must not require it.
- **No git submodule.** Until there is a stable `ehrenfest` package/API release, pin the integration
  CI/dev environment to a **specific known-good EF commit/tag** (this iteration's baseline: `da4d2d9` on
  `claude/tiling-from-scratch`) rather than a moving branch.
- CPU path needs only `numpy` + `threadpoolctl` (no psi4, no torch).

---

## 9. Open questions for the maintainers
- **Track priority:** is the Track-B vertical slice (B0) worth doing *now*, in parallel with Track A?
  (This plan argues yes — it is the only thing that answers the question that matters.)
- **Where to cut the B0 slice:** which residual + update boundary is cleanest to express as one EF
  program while remaining numerically checkable against current PyCC?
- **Dependency/pinning:** confirm pin-to-commit (vs. waiting for an EF release) and how the CI env
  obtains EF.
- **RT-CC / local CC:** keep both on `opt_einsum` until EF grows complex-runner support and per-pair /
  symbolic-extent tensors?
- **GPU:** confirm the regional-handoff model (§5) over any per-call GPU substitution.
