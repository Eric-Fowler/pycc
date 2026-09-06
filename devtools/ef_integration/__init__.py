"""Offline tooling for the ehrenfest (``ef``) integration — Track A (A0).

Standalone, dependency-light utilities that inventory PyCC's contraction corpus
and replay it against ``ef``. Nothing here is imported by the ``pycc`` package;
these are developer tools run from a repo checkout. They depend only on numpy and
(for replay/adapter) ``ehrenfest`` — deliberately not on psi4/scipy/the pycc
package, so A0 can run in a lightweight environment.
"""
