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
"""Tests for the fallback runner's OWN exit status.

THE RUNNER WAS THE ONE FILE IN THIS PACKAGE WITH NO TESTS, which is how it
shipped a fresh instance of the P4 defect class -- two different facts sharing
one exit code -- in the same three-commit slice that was closing P4 in M0. A
non-author's undeclared-changes sweep found it, not the author and not the
suite.

Runs under ``tools/laguna_phase0/run_tests.py``, a fallback runner and NOT
pytest, which means this file is a runner testing itself. That is a real limit
and it is stated rather than glossed: these tests check what ``main()``
RETURNS, which is a plain function call, and they do not check what the process
exits with. A defect in the ``sys.exit(main())`` line at the bottom of the
module would be invisible here.
"""

import contextlib
import io
import sys

from tools.laguna_phase0 import run_tests


def quietly(argv):
    """Runs `main(argv)` with stdout captured. Returns (rc, output).

    Captured deliberately: without it a nested `collected 0 -- LISTED ONLY`
    line lands in the middle of the real suite's transcript, where a reader
    can mistake it for the run's own result. A test that makes the transcript
    harder to read is a test that makes the next count harder to trust.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rc = run_tests.main(argv)
    return rc, buffer.getvalue()

# Selector that matches this file's own tests, so `--list` has something to
# collect without listing the whole suite.
SELF = "test_run_tests"
# Selector that matches nothing. Deliberately not a real test name.
MATCHES_NOTHING = "zzz_no_test_has_this_in_its_name"


def test_the_seven_exit_codes_are_distinct():
    """The whole point of the change. If two collapse, the fix is undone."""
    codes = [
        run_tests.EXIT_ALL_PASSED,
        run_tests.EXIT_SOMETHING_FAILED,
        run_tests.EXIT_NOTHING_COLLECTED,
        run_tests.EXIT_DECLINED,
        run_tests.EXIT_LISTED_ONLY,
        run_tests.EXIT_COLLECTION_INCOMPLETE,
        run_tests.EXIT_UNSUPPORTED_CONSTRUCT,
    ]
    assert len(set(codes)) == len(codes)
    assert run_tests.EXIT_ALL_PASSED == 0


def test_listing_does_not_return_the_all_passed_code():
    """`--list` executes nothing, so it may not return the code for success.

    It used to return 0 -- indistinguishable from a green run of the entire
    suite -- after running no test at all.
    """
    rc, out = quietly(["--list", "--select", SELF])
    assert rc == run_tests.EXIT_LISTED_ONLY
    assert rc != run_tests.EXIT_ALL_PASSED
    assert "NOTHING WAS EXECUTED" in out


def test_listing_something_and_listing_nothing_are_different_codes():
    """Otherwise `--list` cannot tell an empty selector from a full one."""
    listed, _ = quietly(["--list", "--select", SELF])
    empty, _ = quietly(["--list", "--select", MATCHES_NOTHING])
    assert listed == run_tests.EXIT_LISTED_ONLY
    assert empty == run_tests.EXIT_NOTHING_COLLECTED
    assert listed != empty


def test_a_selector_that_matches_nothing_is_undetermined_and_not_a_pass():
    rc, out = quietly(["--select", MATCHES_NOTHING])
    assert rc == run_tests.EXIT_NOTHING_COLLECTED
    assert "UNDETERMINED, not a pass" in out


def test_declining_because_pytest_is_present_is_its_own_code():
    """Exit 3, not 2. Declining to run is not the same as collecting nothing.

    Real pytest is absent in the environment this was written in, so the
    condition is forced rather than waited for. **A branch that cannot be
    reached in the environment at hand is exactly the branch that ships
    unexercised**, and this one shipped unexercised.
    """
    original = run_tests.real_pytest_available
    try:
        run_tests.real_pytest_available = lambda: True
        rc, out = quietly([])
    finally:
        run_tests.real_pytest_available = original
    assert rc == run_tests.EXIT_DECLINED
    assert "real pytest IS importable" in out
    assert rc != run_tests.EXIT_NOTHING_COLLECTED


def test_the_override_flag_defeats_the_refusal_and_the_flag_is_documented():
    """The bypass is real, so it is tested and named in the docstring.

    An undisclosed override makes a stated safety property read as absolute
    when it is a default. The remedy is disclosure, not removal: deliberate
    comparison of the two collectors is a legitimate thing to want.
    """
    original = run_tests.real_pytest_available
    try:
        run_tests.real_pytest_available = lambda: True
        rc, _ = quietly(["--even-if-pytest-is-available",
                         "--list", "--select", SELF])
    finally:
        run_tests.real_pytest_available = original
    assert rc == run_tests.EXIT_LISTED_ONLY
    assert "--even-if-pytest-is-available" in run_tests.__doc__
    assert "NOT A GUARANTEE" in run_tests.__doc__


def test_the_collector_finds_every_test_file_and_not_just_one():
    """A count from an instrument that saw one file cannot fail on the rest.

    KEPT, AND KNOWN TO BE INSUFFICIENT ON ITS OWN. Measured: a mutant that
    truncated ``collect`` to the first test file did NOT fail this test,
    because this test is in a file the truncated collector never reaches. The
    check that actually catches it is ``uncollected_files``, inside the runner.
    """
    cases = run_tests.collect()
    modules = {module_name for module_name, _, _ in cases}
    on_disk = {p.stem for p in run_tests.TESTS_DIR.glob("test_*.py")}
    assert modules == on_disk
    assert len(on_disk) > 1


def test_uncollected_files_names_the_files_that_contributed_nothing():
    """Driven directly, so it does not depend on being collected to be run.

    Empty input is the interesting case and it is asserted first: given no
    cases at all, EVERY file on disk is uncollected. A reconciler that returned
    an empty list for empty input would be the same defect one level up -- it
    would call the total truncation clean.
    """
    on_disk = run_tests.test_files_on_disk()
    assert run_tests.uncollected_files([]) == on_disk
    assert on_disk, "no test files on disk at all -- UNDETERMINED, not a pass"

    stems = [name[: -len(".py")] for name in on_disk]
    truncated = [(stems[0], "test_x", lambda: None)]
    assert run_tests.uncollected_files(truncated) == on_disk[1:]
    assert on_disk[0] not in run_tests.uncollected_files(truncated)

    full = [(stem, "test_x", lambda: None) for stem in stems]
    assert run_tests.uncollected_files(full) == []


def test_a_truncated_collector_gets_its_own_code_and_prints_no_count():
    """THE MUTANT THAT SURVIVED EVERYTHING, TURNED INTO A TEST.

    ``collect`` is replaced with one that returns only the first file's tests,
    which is the mutation that was measured as undetected. The runner must
    refuse: its own code, distinct from failure and from success, and NO COUNT
    in the output, because a count is what a reader would have believed.
    """
    original = run_tests.collect

    def only_the_first_file(select=None):
        cases = original(select)
        first = cases[0][0]
        return [case for case in cases if case[0] == first]

    try:
        run_tests.collect = only_the_first_file
        rc, out = quietly([])
    finally:
        run_tests.collect = original

    assert rc == run_tests.EXIT_COLLECTION_INCOMPLETE
    assert rc != run_tests.EXIT_ALL_PASSED
    assert rc != run_tests.EXIT_SOMETHING_FAILED
    assert "COLLECTION IS INCOMPLETE" in out
    assert "passed," not in out, "a count was printed over a truncated corpus"


def test_an_async_test_is_refused_and_never_counted_as_passed():
    """Sweep measured this one as a live false pass: collected, never awaited.

    The async function is defined INSIDE this test on purpose. At module level
    the collector would find it, the runner would refuse every run, and the
    suite could never report anything again -- which is the correct behaviour
    for a real async test and useless for a test about that behaviour.
    """
    async def test_would_fail_if_it_ever_ran():
        assert False, "if you see this, the runner awaited it"

    named = run_tests.unsupported_constructs(
        [("test_synthetic", "test_would_fail_if_it_ever_ran",
          test_would_fail_if_it_ever_ran)])
    assert len(named) == 1
    assert "async def" in named[0]
    assert not run_tests.unsupported_constructs(
        [("test_synthetic", "test_ordinary", lambda: None)])

    original = run_tests.collect

    def with_one_async(select=None):
        return list(original(select)) + [
            ("test_synthetic", "test_would_fail_if_it_ever_ran",
             test_would_fail_if_it_ever_ran)]

    try:
        run_tests.collect = with_one_async
        rc, out = quietly([])
    finally:
        run_tests.collect = original

    assert rc == run_tests.EXIT_UNSUPPORTED_CONSTRUCT
    assert rc != run_tests.EXIT_ALL_PASSED
    assert "UNSUPPORTED CONSTRUCT" in out
    assert "passed," not in out, "a count was printed alongside a never-run test"


def test_a_test_file_in_a_subdirectory_is_not_silently_ignored():
    """The collector's glob is one level deep. Silence about that was the bug.

    Writes a real file, because the reconciliation reads the disk and a mock of
    the disk would be testing the mock. ABSOLUTE PATH, built from
    ``TESTS_DIR``, never a relative write after a directory change.
    """
    nested_dir = run_tests.TESTS_DIR / "sub_probe"
    nested = nested_dir / "test_nested_probe.py"
    try:
        nested_dir.mkdir(exist_ok=True)
        nested.write_text("def test_nested():\n    assert True\n")

        assert "sub_probe/test_nested_probe.py" in run_tests.test_files_on_disk()
        collected = run_tests.collect()
        assert all(module != "test_nested_probe" for module, _, _ in collected)
        assert "sub_probe/test_nested_probe.py" in run_tests.uncollected_files(
            collected)

        rc, out = quietly([])
        assert rc == run_tests.EXIT_COLLECTION_INCOMPLETE
        assert "sub_probe/test_nested_probe.py" in out
    finally:
        if nested.exists():
            nested.unlink()
        if nested_dir.exists():
            nested_dir.rmdir()
    assert not nested_dir.exists(), "the probe directory outlived its test"
    assert run_tests.uncollected_files(run_tests.collect()) == []


def test_this_module_is_importable_under_the_name_the_collector_uses():
    """Guards the import path the runner depends on, which no test covered."""
    assert f"{run_tests.PACKAGE}.{SELF}" in sys.modules
