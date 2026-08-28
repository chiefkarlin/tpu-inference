# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Phase 0 test runner, IN THE REPOSITORY so that a count can be reproduced.

WHY THIS FILE EXISTS

Round 1 review R10: five of the then-seven test files in this package had never
been collected, and the two counts that had been reported (37/37 and 24/24) were
produced by a hand-rolled collector kept in ``/tmp``. A count produced by an
instrument that is not in the repository cannot be reproduced by the next
reader, so it is not evidence -- it is an anecdote with a number in it. Two
successive agents then each declined to build another throwaway collector, which
was the right call and left the suite exactly as unrunnable as before.

The way out of that loop is not a better throwaway. It is to SHIP the collector.
This file is that collector. It is deliberately about two hundred lines, has no
dependencies outside the standard library, and is version-controlled next to the
tests it runs, so ``python3 tools/laguna_phase0/run_tests.py`` from the
repository root produces the same count for anybody who checks the tree out.

WHAT IT IS NOT

**It is not pytest and it must never be reported as pytest.** It supports
exactly the two pytest surfaces this suite actually uses -- ``pytest.raises``
and ``pytest.approx`` -- and nothing else. It has no fixtures, no
``parametrize``, no marks, no ``conftest.py``, no plugins, no assertion
rewriting. If real pytest is importable this runner **defers to it and refuses
to run BY DEFAULT**, so that the richer tool is never silently shadowed by the
poorer one. **That refusal can be overridden** -- see USAGE below, where the
override flag is named. It is stated in both places because the unqualified
version of this sentence is what made the override an undisclosed one.

Every line it prints names itself, and the summary line says
``laguna-phase0 fallback runner`` rather than ``passed``, because the whole
point of R10 is that a reader must be able to tell which instrument produced a
count without being told separately.

WHAT IT CANNOT DO -- THE EXHAUSTIVE LIST, AND WHAT A COUNT FROM IT MAY CLAIM

This list is here, and not only in the fix document, because a reader who runs
this file must be able to find its limits without knowing that any other
document exists.

Unsupported, and a test using any of these will ERROR or be silently skipped
rather than quietly "pass":

* fixtures of every kind, including ``@pytest.fixture``, and therefore all of
  the builtin fixtures -- ``tmp_path``, ``tmp_path_factory``, ``monkeypatch``,
  ``capsys``, ``capfd``, ``caplog``, ``recwarn``, ``request``
* ``@pytest.mark.parametrize``, and marks generally -- ``skip``, ``skipif``,
  ``xfail``, ``usefixtures``, and any custom mark
* ``conftest.py``: NOT read at all, at any level
* plugins, and anything from ``pytest.ini`` / ``pyproject.toml`` /
  ``setup.cfg`` -- no configuration file of any kind is consulted
* test CLASSES: ``class Test*`` is NOT collected. Only module-level functions
  named ``test_*`` in files named ``test_*.py`` are collected
* setup/teardown of any flavour -- ``setup_function``, ``setup_module``,
  ``setUp``, and ``unittest.TestCase`` entirely
* assertion introspection. A failing ``assert`` reports the exception, NOT
  pytest's expression rewriting showing the operand values
* ``pytest.approx`` on anything but a SCALAR (rel=1e-6, abs=1e-12); no
  sequences, dicts or numpy arrays
* ``pytest.raises`` other than as a CONTEXT MANAGER, and it takes no ``match=``

SO THE HONEST CLAIM A COUNT FROM THIS RUNNER SUPPORTS is: "these N module-level
test functions were collected and executed by the fallback runner, and none
raised." IT DOES NOT SUPPORT "the test suite passes", because a suite that uses
any construct above is not fully collected here, and a construct this runner
skips is INVISIBLE rather than red. **CHECK THE COLLECTED COUNT AGAINST THE
NUMBER OF ``def test_`` LINES IN THE TREE BEFORE BELIEVING A GREEN RESULT** --
use ``--list``. An honest untested beats an unreproducible 37-of-37.

That reconciliation was run on this tree rather than assumed, and it is stated
as a MEASUREMENT ON A DATE and not as a standing property of the suite: as of
this commit, ``def test_`` appears 156 times across the 8 test files, 156 are
collected, and no test uses any construct in the unsupported list above -- the
only occurrences of the word "fixture" in the suite are in string literals and
comments. There are **no indented test functions**, and **one** class
definition in the suite, ``approx`` in ``tests/test_m1_profile.py``, which is a
local comparator and not a ``class Test*``; it is therefore correctly not
collected. That is stated rather than reported as "no classes", because "no
classes" would have been a true-sounding summary of a tree that has one.

So for THIS suite, at THIS commit, the gap between "156 collected and executed"
and "the suite passes" is closed by inspection. **THE GAP REOPENS THE
MOMENT SOMEONE ADDS A TEST USING A CONSTRUCT ABOVE, AND IT WILL REOPEN
SILENTLY.** Re-run the reconciliation; do not inherit this paragraph's result.

USAGE

    python3 tools/laguna_phase0/run_tests.py
    python3 tools/laguna_phase0/run_tests.py --select closure
    python3 tools/laguna_phase0/run_tests.py --list
    python3 tools/laguna_phase0/run_tests.py --even-if-pytest-is-available

**THE REFUSAL WHEN REAL PYTEST IS PRESENT IS A DEFAULT, NOT A GUARANTEE, AND
``--even-if-pytest-is-available`` TURNS IT OFF.** Stated here because the first
version of this docstring said the runner "defers to it and refuses to run"
without qualification, and an undisclosed override makes a stated safety
property read as absolute when it is a default. What the refusal actually
prevents is ACCIDENTAL shadowing of a stronger tool by a weaker imitation.
Deliberate shadowing is a supported mode; it takes an explicit flag, and every
line this runner prints still says NOT pytest.

EXIT STATUS -- SIX STATES, EACH DISTINGUISHABLE, WHICH IS THE POINT

    0  every collected test passed, and at least one ran
    1  a collected test failed or errored -- it RAN and did not pass
    2  NOTHING WAS COLLECTED. An empty run is UNDETERMINED and never a pass,
       the same rule the instruments follow, and the rule R7 exists because M0
       broke it
    3  THE RUNNER DECLINED TO RUN: real pytest is importable and the override
       was not passed. Nothing was collected and nothing was executed
    4  ``--list`` only: tests were collected and DELIBERATELY NOT EXECUTED
    5  COLLECTION IS INCOMPLETE: a ``test_*.py`` on disk contributed no test.
       No count is printed at all, because the count would be true of a
       corpus nobody asked about

**WHY 5 EXISTS AND WHY IT IS NOT A TEST.** A mutant that made ``collect`` read
only the first test file SURVIVED THE ENTIRE SUITE, including the test written
to catch exactly that, because that test lives in a file the truncated
collector never reaches. The suite printed a smaller green number and no
failure. **A completeness check that the instrument can silently exclude is not
a check**, so the reconciliation runs inside ``run`` on every full run, before
any count is emitted. See ``uncollected_files``.

**WHY 3 AND 4 ARE NOT 2 AND NOT 0.** An undeclared-changes sweep by a
non-author found that the refusal used to return 2 -- the same code as "nothing
was collected" -- and that ``--list`` used to return 0, the same code as "every
test passed", after running nothing at all. **That is the P4 defect class,
which this package was in the middle of closing in M0 when a fresh instance of
it shipped here.** "Could not run", "found nothing to run" and "chose not to
run" are three different facts about the world, and a caller that cannot tell
them apart from the exit status is being handed a number that cannot come out
differently in the case it matters. Only 0 means a test passed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import pathlib
import sys
import traceback
import types
from typing import Any, Callable, List, Optional, Sequence, Tuple

PACKAGE = "tools.laguna_phase0.tests"
# Exit codes, named rather than written as bare integers at the return sites.
# The three-state convention this package works under is passed / ran and did
# not pass / could not run; 3 and 4 split "could not run" into its two distinct
# causes, because a caller that cannot tell them apart is reading a number that
# does not depend on which happened.
EXIT_ALL_PASSED = 0
EXIT_SOMETHING_FAILED = 1
EXIT_NOTHING_COLLECTED = 2
EXIT_DECLINED = 3
EXIT_LISTED_ONLY = 4
# 5 is not a variant of "something failed". Nothing failed: the runner never
# looked at part of the corpus, so it has no result for that part and refuses
# to publish a count that would be read as covering it.
EXIT_COLLECTION_INCOMPLETE = 5

TESTS_DIR = pathlib.Path(__file__).with_name("tests")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

BANNER = ("laguna-phase0 fallback runner -- NOT pytest. Supports raises/approx "
          "only. Counts from this runner are observations from this runner.")


# --------------------------------------------------------------------------
# The two pytest surfaces this suite uses, and no others.
# --------------------------------------------------------------------------


class ExceptionInfo:
    """What ``with raises(E) as info`` binds. ``.value`` and ``.type`` only."""

    def __init__(self) -> None:
        self.value: Optional[BaseException] = None
        self.type: Optional[type] = None


class _Raises:
    """Minimal ``pytest.raises``. Context-manager form only."""

    def __init__(self, expected: Any):
        self._expected = expected
        self._info = ExceptionInfo()

    def __enter__(self) -> ExceptionInfo:
        return self._info

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            name = getattr(self._expected, "__name__", str(self._expected))
            raise AssertionError(f"DID NOT RAISE {name}")
        if not issubclass(exc_type, self._expected):
            return False  # propagate: the wrong exception is a failure, not a pass
        self._info.value = exc
        self._info.type = exc_type
        return True


class _Approx:
    """Minimal ``pytest.approx`` for scalars, with pytest's default tolerances."""

    def __init__(self, expected: Any, rel: Optional[float] = None,
                 abs: Optional[float] = None):  # noqa: A002 - pytest's own name
        self._expected = expected
        self._rel = 1e-6 if rel is None else rel
        self._abs = 1e-12 if abs is None else abs

    def __eq__(self, other: Any) -> bool:
        if self._expected is None or other is None:
            return other is self._expected
        try:
            a, b = float(other), float(self._expected)
        except (TypeError, ValueError):
            return NotImplemented
        if math.isnan(a) or math.isnan(b):
            return False
        return abs(a - b) <= max(self._abs, self._rel * abs(b))

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def __repr__(self) -> str:
        return f"approx({self._expected!r}, rel={self._rel}, abs={self._abs})"


def install_pytest_stand_in() -> None:
    """Registers the stand-in as ``pytest`` ONLY when real pytest is absent.

    Shadowing an installed pytest with a two-function imitation would be a way
    to make a weaker check look like a stronger one, which is the failure mode
    this whole package is written against. So the caller checks first, and this
    function is only ever reached when the import genuinely failed.
    """
    module = types.ModuleType("pytest")
    module.raises = _Raises            # type: ignore[attr-defined]
    module.approx = _Approx            # type: ignore[attr-defined]
    module.ExceptionInfo = ExceptionInfo  # type: ignore[attr-defined]
    module.__doc__ = (
        "STAND-IN installed by tools/laguna_phase0/run_tests.py because real "
        "pytest is not importable. raises/approx only.")
    module.__laguna_phase0_stand_in__ = True  # type: ignore[attr-defined]
    sys.modules["pytest"] = module


def real_pytest_available() -> bool:
    """True only for a REAL pytest, and safe to call after the stand-in is in.

    TWO WAYS THIS USED TO BE WRONG, BOTH FOUND BY THE FIRST TEST EVER WRITTEN
    AGAINST THIS RUNNER, and both only reachable on a second call in one
    process -- which is to say, never during a normal run and always during a
    test of a normal run.

    1. ``find_spec("pytest")`` RAISES ``ValueError: pytest.__spec__ is None``
       once the stand-in is registered, because the stand-in is a plain module
       object with no spec. The function did not merely return the wrong
       answer, it blew up.
    2. Had it not raised, a spec-bearing stand-in would have made this return
       True and the runner would have reported real pytest as present on the
       strength of its own imitation.

    So the stand-in is recognised BY ITS MARKER FIRST, before any spec lookup.
    """
    installed = sys.modules.get("pytest")
    if getattr(installed, "__laguna_phase0_stand_in__", False):
        return False
    if installed is not None:
        return True
    try:
        return importlib.util.find_spec("pytest") is not None
    except (ImportError, ValueError):
        # A broken or spec-less `pytest` on the path is NOT a usable pytest.
        # Answering False here means the fallback runs, which is the safe
        # direction: the alternative is refusing to run over a tool that
        # cannot be imported anyway.
        return False


# --------------------------------------------------------------------------
# Collection and execution.
# --------------------------------------------------------------------------


def collect(select: Optional[str] = None) -> List[Tuple[str, str, Callable[[], Any]]]:
    """Returns ``(module_name, test_name, function)`` for every collected test.

    Collection failures are NOT swallowed. A module that cannot be imported is
    the single most likely way for this suite to report a clean run over
    nothing, so an import error propagates and the runner exits non-zero.
    """
    found: List[Tuple[str, str, Callable[[], Any]]] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module_name = f"{PACKAGE}.{path.stem}"
        module = importlib.import_module(module_name)
        for attr in sorted(vars(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn) or getattr(fn, "__module__", None) != module_name:
                continue
            if select and select not in attr and select not in path.stem:
                continue
            found.append((path.stem, attr, fn))
    return found


def uncollected_files(
        cases: Sequence[Tuple[str, str, Callable[[], Any]]]) -> List[str]:
    """Test files on disk that contributed no collected test.

    THE COMPLETENESS CHECK CANNOT LIVE IN A TEST, AND THIS IS THE WHOLE POINT.
    A test asserting "the collector found every file" is itself in a file the
    collector has to find. Truncate the collector and that test is not
    collected either, so it cannot fail -- the suite reports a clean run over a
    smaller corpus and the number goes down silently. It was measured: a mutant
    that made ``collect`` read only the first test file was caught by NOTHING,
    including the test written specifically to catch it.

    So the reconciliation is performed BY THE RUNNER, on every full run, before
    any count is printed. An instrument that can only be checked by the thing
    it is measuring is not checked.

    TWO PROPERTIES, BOTH DELIBERATE, NEITHER A BUG:

    * A ``test_*.py`` holding no test function at all is reported as
      uncollected. That is wanted. A test file contributing nothing is a hole
      of the same kind, and it is likelier to be a half-finished file than a
      deliberate one.
    * IT DOES NOT COVER ``--list``, which does not call ``run``. A truncated
      collector under ``--list`` prints a shorter list and says so. ``--list``
      never returns the code for a pass, so it cannot manufacture a green
      result, but it CAN under-report a count to a reader who trusts it.
    """
    seen = {module_name for module_name, _, _ in cases}
    return sorted(p.stem for p in TESTS_DIR.glob("test_*.py")
                  if p.stem not in seen)


def run(select: Optional[str] = None, verbose: bool = False) -> int:
    cases = collect(select)
    if select is None:
        missing = uncollected_files(cases)
        if missing:
            print(f"{BANNER}\nlaguna-phase0 fallback runner: COLLECTION IS "
                  f"INCOMPLETE. {len(missing)} test file(s) on disk "
                  f"contributed no test: {', '.join(missing)}.\n"
                  "NO COUNT IS REPORTED. A green result over part of the "
                  "corpus is worse than no result, because it looks like the "
                  "whole corpus.")
            return EXIT_COLLECTION_INCOMPLETE
    if not cases:
        print(f"{BANNER}\ncollected 0 tests"
              + (f" matching {select!r}" if select else "")
              + "\nlaguna-phase0 fallback runner: NOTHING COLLECTED -- "
                "UNDETERMINED, not a pass")
        return EXIT_NOTHING_COLLECTED

    print(BANNER)
    failures: List[Tuple[str, str, str]] = []
    current = None
    for module_name, test_name, fn in cases:
        if module_name != current:
            current = module_name
            print(f"\n{module_name}")
        try:
            fn()
        except Exception:  # noqa: BLE001 - a test may raise anything
            failures.append((module_name, test_name, traceback.format_exc()))
            print(f"  FAIL  {test_name}")
        else:
            if verbose:
                print(f"  ok    {test_name}")

    for module_name, test_name, tb in failures:
        print(f"\n{'=' * 70}\nFAIL {module_name}::{test_name}\n{'-' * 70}\n{tb}",
              end="")

    passed = len(cases) - len(failures)
    print(f"\nlaguna-phase0 fallback runner (NOT pytest): "
          f"{passed} passed, {len(failures)} failed, {len(cases)} collected "
          f"from {len(sorted(TESTS_DIR.glob('test_*.py')))} files")
    return EXIT_SOMETHING_FAILED if failures else EXIT_ALL_PASSED


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--select", default=None,
                        help="substring of a test name or module to run")
    parser.add_argument("--list", action="store_true",
                        help="collect and print, run nothing")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--even-if-pytest-is-available", action="store_true",
        help="run anyway; only for comparing the two collectors deliberately")
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    if real_pytest_available() and not args.even_if_pytest_is_available:
        print("real pytest IS importable in this environment.\n"
              "This runner exists only for environments where it is not, and it "
              "supports far less. Run:\n"
              "    python3 -m pytest tools/laguna_phase0/tests\n"
              "Pass --even-if-pytest-is-available to override deliberately.")
        # EXIT 3, not 2. Declining to run is not the same fact as collecting
        # nothing, and before this was separated a caller could not tell them
        # apart. See the EXIT STATUS block in the module docstring.
        return EXIT_DECLINED
    if not real_pytest_available():
        install_pytest_stand_in()

    if args.list:
        cases = collect(args.select)
        for module_name, test_name, _ in cases:
            print(f"{module_name}::{test_name}")
        print(f"collected {len(cases)} -- LISTED ONLY, NOTHING WAS EXECUTED")
        # EXIT 4, not 0. `--list` executed no test, so it has no business
        # returning the code that means every test passed.
        return EXIT_LISTED_ONLY if cases else EXIT_NOTHING_COLLECTED

    return run(args.select, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
