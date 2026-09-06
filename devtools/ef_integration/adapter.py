"""Reference ``ef`` contraction adapter (compatibility path, CPU/DP/numpy only).

This is the Track-A adapter described in EF_INTEGRATION_PLAN.md §3, kept in
devtools so the pycc package does not gain a hard ehrenfest dependency. It maps a
single ``contract(subscripts, *operands)`` call onto ef, caching the built node so
``ef.runner``'s per-node-identity plan cache actually hits across iterations.

It intentionally does NOT expose ef's cross-term CSE / materialize-vs-fuse /
residency — those are program-level (Track B). This seam is a compatibility and
validation path only.
"""

from __future__ import annotations

import numpy as np
import ehrenfest as ef


class EfContractionAdapter:
    """Callable with the same surface as ``ContractionBackend.__call__``.

    * whitespace-normalized, case-preserved spec (ef is sensitive to both);
    * implicit-output specs rejected loudly (never a bare IndexError);
    * node memoized on ``(clean_spec, shapes, per_operand_dtypes)`` and the SAME
      node object reused, so the runner's plan cache hits;
    * layout is deliberately NOT in the key (contraction identity is value/layout
      independent; pinned by test_a0.test_node_cache_is_layout_independent).
    """

    def __init__(self, capacity: int = 1 << 34, device: str = "cpu", oracle: bool = False):
        self._run = None if oracle else ef.runner(capacity, device=device)
        self._cache: dict = {}

    def __call__(self, subscripts: str, *operands):
        spec = "".join(subscripts.split())
        if "->" not in spec:
            raise ValueError(f"implicit-output einsum not supported by the ef adapter: {subscripts!r}")
        out = spec.split("->")[1]
        shapes = tuple(tuple(np.shape(o)) for o in operands)
        dtypes = tuple(np.asarray(o).dtype.str for o in operands)
        key = (spec, shapes, dtypes)
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
