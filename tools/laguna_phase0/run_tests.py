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

Round 1 review R10: five of the seven test files in this package had never been
collected, and the two counts that had been reported (37/37 and 24/24) were
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
to run**, so that the richer tool is never silently shadowed by the poorer one.

Every line it prints names itself, and the summary line says
``laguna-phase0 fallback runner`` rather than ``passed``, because the whole
point of R10 is that a reader must be able to tell which instrument produced a
count without being told separately.

USAGE

    python3 tools/laguna_phase0/run_tests.py
    python3 tools/laguna_phase0/run_tests.py --select closure
    python3 tools/laguna_phase0/run_tests.py --list

Exit status is 0 when every collected test passed, 1 when any failed or errored,
and 2 when nothing was collected. **Zero collected tests is exit 2, not exit 0**
-- an empty run is UNDETERMINED and never a pass, which is the same rule the
instruments themselves follow, and the rule R7 exists because M0 broke.
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
    spec = importlib.util.find_spec("pytest")
    return spec is not None


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


def run(select: Optional[str] = None, verbose: bool = False) -> int:
    cases = collect(select)
    if not cases:
        print(f"{BANNER}\ncollected 0 tests"
              + (f" matching {select!r}" if select else "")
              + "\nlaguna-phase0 fallback runner: NOTHING COLLECTED -- "
                "UNDETERMINED, not a pass")
        return 2

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
    return 1 if failures else 0


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
        return 2
    if not real_pytest_available():
        install_pytest_stand_in()

    if args.list:
        cases = collect(args.select)
        for module_name, test_name, _ in cases:
            print(f"{module_name}::{test_name}")
        print(f"collected {len(cases)}")
        return 0 if cases else 2

    return run(args.select, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
