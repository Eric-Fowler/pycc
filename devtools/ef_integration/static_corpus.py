"""AST inventory of PyCC's ``contract(...)`` einsum corpus.

Complements the runtime capture (which needs a live CCSD run, hence psi4): this
walks the AST of every source file, finds every call whose callee is ``contract``
or ``*.contract``, and inspects argument zero. A spec counts as *literal* iff arg0
is a string constant (``ast.Constant`` with a ``str`` value) — which covers single
and double quotes and Python's implicit adjacent-string concatenation, and excludes
f-strings, ``+`` concatenation, and variables. Non-literal calls are reported with
source locations so they can be audited, not assumed away.

This is what justifies A0 making implicit-output a hard adapter rejection: the claim
"every literal spec spells ``->``" is only as strong as the scan behind it, so the
scan must inspect the real argument, not a regex.

Run:  python -m devtools.ef_integration.static_corpus  [pycc_dir]
"""

from __future__ import annotations

import ast
import collections
import pathlib
import sys


def _callee_name(func: ast.AST):
    """'contract' for ``contract(...)`` and ``obj.contract(...)``; else None."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _literal_str(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _looks_einsum(spec: str) -> bool:
    # an einsum spec is letters/commas/arrow/space only, and names >=1 operand term
    body = "".join(spec.split())
    if not body:
        return False
    if not all(c.isalpha() or c in ",->" for c in body):
        return False
    return ("," in body) or ("->" in body)


def iter_contract_calls(pycc_dir: pathlib.Path):
    """Yield (arg0_literal_or_None, is_einsum, spec_or_repr, file, line) per call."""
    for path in sorted(pycc_dir.rglob("*.py")):
        if path.name == "ef_capture.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = str(path.relative_to(pycc_dir.parent))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee_name(node.func) != "contract":
                continue
            if not node.args:
                # contract() with no positional args (e.g. the backend's own **-forwarding
                # or an unrelated method) — record as non-literal for auditing
                yield None, False, "<no-args>", rel, node.lineno
                continue
            lit = _literal_str(node.args[0])
            if lit is None:
                yield None, False, ast.dump(node.args[0])[:60], rel, node.lineno
            else:
                yield lit, _looks_einsum(lit), lit, rel, node.lineno


def inventory(pycc_dir: pathlib.Path) -> dict:
    calls = list(iter_contract_calls(pycc_dir))
    einsum_lits = [(s, f, ln) for (lit, isk, s, f, ln) in calls if lit is not None and isk]
    nonliteral = [(s, f, ln) for (lit, isk, s, f, ln) in calls if lit is None]
    nonein_lit = [(s, f, ln) for (lit, isk, s, f, ln) in calls
                  if lit is not None and not isk]  # literal but not einsum (e.g. a device str)
    implicit = [(s, f, ln) for (s, f, ln) in einsum_lits if "->" not in s]
    with_space = [(s, f, ln) for (s, f, ln) in einsum_lits if s != "".join(s.split())]
    uppercase = [(s, f, ln) for (s, f, ln) in einsum_lits if any(c.isupper() for c in s)]
    arity = collections.Counter(s.split("->")[0].count(",") + 1 for (s, _, _) in einsum_lits)
    distinct = sorted({"".join(s.split()) for (s, _, _) in einsum_lits})
    return {
        "n_contract_calls": len(calls),
        "n_einsum_literal": len(einsum_lits),
        "n_nonliteral": len(nonliteral),
        "n_literal_noneinsum": len(nonein_lit),
        "n_distinct": len(distinct),
        "nonliteral": nonliteral,
        "literal_noneinsum": nonein_lit,
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
    print(f"contract() call expressions        : {inv['n_contract_calls']}")
    print(f"  literal einsum-spec calls        : {inv['n_einsum_literal']}  ({inv['n_distinct']} distinct)")
    print(f"  literal non-einsum first arg      : {inv['n_literal_noneinsum']}")
    print(f"  NON-literal first arg (audit)    : {inv['n_nonliteral']}")
    print(f"arity (operands -> count)          : {inv['arity_hist']}")
    print(f"uppercase-index specs              : {len(inv['uppercase_index'])}")
    print(f"whitespace-in-spec specs           : {len(inv['with_whitespace'])}")
    if inv["nonliteral"]:
        print("non-literal contract() calls (audit these — not assumed prose/backend):")
        for s, f, ln in inv["nonliteral"]:
            print(f"    {f}:{ln}  {s}")
    if inv["implicit_output"]:
        print(f"IMPLICIT-OUTPUT literal specs: {len(inv['implicit_output'])} — invariant FAILS:")
        for s, f, ln in inv["implicit_output"][:20]:
            print(f"    {f}:{ln}  {s!r}")
    else:
        print("INVARIANT HOLDS: every literal einsum contract() spec spells '->' (explicit output).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
