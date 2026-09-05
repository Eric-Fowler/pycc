"""Static inventory of PyCC's ``contract('...')`` einsum corpus.

Complements the runtime capture (which needs a live CCSD run, hence psi4): this
parses string-literal specs straight from the sources, so the *index-pattern*
corpus and the implicit-output question can be answered without running anything.
~1900 of ~1950 ``contract(`` occurrences use a literal spec; the rest are
docstring prose or the backend's own ``opt_einsum.contract(subscripts, ...)`` and
are correctly skipped (no literal to parse).

Answers, per §4 A0:
  * does any real spec omit ``->`` (implicit output)? -> invariant to assert;
  * arity distribution, uppercase-index usage, whitespace-in-spec occurrences.

Run:  python -m devtools.ef_integration.static_corpus  [pycc_dir]
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

# contract('<spec>' ...  — a single-quoted first argument that looks like einsum
_LITERAL = re.compile(r"contract\(\s*'([^']*)'")
# a spec is einsum-ish if it is only letters/commas/arrow/spaces (excludes prose)
_EINSUMISH = re.compile(r"^[A-Za-z,\->\s]+$")


def iter_specs(pycc_dir: pathlib.Path):
    for path in sorted(pycc_dir.rglob("*.py")):
        if "ef_capture.py" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in _LITERAL.finditer(text):
            spec = m.group(1)
            if "," not in spec and "->" not in spec:
                continue  # not an einsum spec (e.g. a device string)
            if not _EINSUMISH.match(spec):
                continue
            line = text.count("\n", 0, m.start()) + 1
            yield spec, str(path.relative_to(pycc_dir.parent)), line


def inventory(pycc_dir: pathlib.Path) -> dict:
    specs = list(iter_specs(pycc_dir))
    implicit = [(s, f, ln) for (s, f, ln) in specs if "->" not in s]
    with_space = [(s, f, ln) for (s, f, ln) in specs if s != "".join(s.split())]
    uppercase = [(s, f, ln) for (s, f, ln) in specs if any(c.isupper() for c in s)]
    arity = collections.Counter(s.split("->")[0].count(",") + 1 for (s, _, _) in specs)
    distinct = sorted({"".join(s.split()) for (s, _, _) in specs})
    return {
        "n_occurrences": len(specs),
        "n_distinct": len(distinct),
        "implicit_output": implicit,
        "with_whitespace": with_space,
        "uppercase_index": uppercase,
        "arity_hist": dict(sorted(arity.items())),
        "distinct_specs": distinct,
    }


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    pycc_dir = pathlib.Path(argv[0]) if argv else pathlib.Path(__file__).resolve().parents[2] / "pycc"
    inv = inventory(pycc_dir)
    print(f"contract('...') literal einsum specs: {inv['n_occurrences']} occurrences, "
          f"{inv['n_distinct']} distinct")
    print(f"arity (operands -> count): {inv['arity_hist']}")
    print(f"specs using uppercase indices: {len(inv['uppercase_index'])} "
          f"(e.g. {inv['uppercase_index'][0][0] if inv['uppercase_index'] else '-'})")
    print(f"specs written with whitespace: {len(inv['with_whitespace'])} "
          f"(e.g. {inv['with_whitespace'][0][0] if inv['with_whitespace'] else '-'})")
    if inv["implicit_output"]:
        print(f"IMPLICIT-OUTPUT specs found: {len(inv['implicit_output'])} — "
              f"invariant DOES NOT HOLD; adapter must normalize implicit semantics:")
        for s, f, ln in inv["implicit_output"][:20]:
            print(f"    {f}:{ln}  {s!r}")
    else:
        print("INVARIANT HOLDS: every literal contract() spec spells '->' (explicit output). "
              "The adapter may reject implicit-output specs loudly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
