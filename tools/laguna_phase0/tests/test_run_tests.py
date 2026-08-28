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


def test_the_eight_exit_codes_are_distinct():
    """The whole point of the change. If two collapse, the fix is undone.

    EXIT_PARTIAL_PASSED joined this list with the --select fix. It is listed
    here rather than merely defined because a code that no test compares
    against the others is a code that can silently be given an existing value.
    """
    codes = [
        run_tests.EXIT_ALL_PASSED,
        run_tests.EXIT_SOMETHING_FAILED,
        run_tests.EXIT_NOTHING_COLLECTED,
        run_tests.EXIT_DECLINED,
        run_tests.EXIT_LISTED_ONLY,
        run_tests.EXIT_COLLECTION_INCOMPLETE,
        run_tests.EXIT_UNSUPPORTED_CONSTRUCT,
        run_tests.EXIT_PARTIAL_PASSED,
    ]
    assert len(set(codes)) == len(codes)
    assert run_tests.EXIT_ALL_PASSED == 0
    assert run_tests.EXIT_PARTIAL_PASSED != run_tests.EXIT_ALL_PASSED


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


# --------------------------------------------------------------------------
# THE `--select` SCOPE. Until these landed, `--select <one_module>` covered one
# module, printed the WHOLE TREE's file count as its denominator, and returned
# the code that means the suite passed. The check that would have caught the
# missing coverage was gated off by the one `if select is None:` in the file,
# so the narrowest run this runner offers was also the only one with no
# completeness check at all.
#
# EVERY TEST BELOW IS PAIRED. A test that only asserts the new number would
# pass against a runner that prints a constant, so each one also asserts that
# the OLD string is absent, or that the new code differs from the old code.
# --------------------------------------------------------------------------


def _synthetic_full_collection(select=None):
    """Cases covering every top-level test file, with trivial bodies.

    A FULL run has to be exercised from inside the suite, and a real nested
    ``quietly([])`` would collect THIS FILE and re-enter itself without bound.
    The existing tests get away with ``quietly([])`` only because each one
    forces an exit before a single test body is called. These cases let the
    whole-tree path run to its summary line instead.
    """
    cases = [(name[: -len(".py")], "test_synthetic_ok", lambda: None)
             for name in run_tests.test_files_on_disk()]
    if select:
        cases = [case for case in cases
                 if select in case[0] or select in case[1]]
    return cases


def _a_real_file_stem_other_than_this_one():
    """A selector that names a real file which is NOT the one we are running.

    Selecting this module's own tests from inside one of them would recurse.
    Derived from the disk rather than typed, so it cannot name a file that
    stopped existing (R20: derive every constant from the artefact).
    """
    stems = [name[: -len(".py")] for name in run_tests.test_files_on_disk()
             if "/" not in name and name != f"{SELF}.py"]
    assert stems, "no other test file on disk -- UNDETERMINED, not a pass"
    return stems[0]


def test_files_in_scope_is_the_whole_tree_only_when_nothing_was_selected():
    on_disk = run_tests.test_files_on_disk()
    assert run_tests.files_in_scope() == on_disk
    assert run_tests.files_in_scope(None) == on_disk

    stem = _a_real_file_stem_other_than_this_one()
    scoped = run_tests.files_in_scope(stem)
    assert scoped == [f"{stem}.py"]
    assert len(scoped) < len(on_disk), "the scope did not actually narrow"


def test_a_selector_naming_no_file_puts_no_file_in_scope():
    """The honest empty case, asserted rather than left to be discovered.

    An empty scope is the ABSENCE of a completeness check, not a clean one, so
    the runner is required to say so in the transcript. Both halves are checked
    here because the empty list on its own is indistinguishable from a scope
    function that always returns nothing.
    """
    assert run_tests.files_in_scope(MATCHES_NOTHING) == []
    assert run_tests.files_in_scope(
        _a_real_file_stem_other_than_this_one()) != []


def test_covered_files_counts_what_ran_and_never_what_is_on_disk():
    """The denominator's source. It must not be able to see the tree."""
    cases = [("test_alpha", "test_one", lambda: None),
             ("test_alpha", "test_two", lambda: None),
             ("test_beta", "test_three", lambda: None)]
    assert run_tests.covered_files(cases) == ["test_alpha.py", "test_beta.py"]
    assert run_tests.covered_files([]) == []
    assert run_tests.covered_files(cases) != run_tests.test_files_on_disk()


def test_a_partial_run_reports_the_selected_count_and_not_the_tree_count():
    """THE DEFECT, TURNED INTO A TEST.

    The old summary read ``N collected from 8 files`` after running one
    module. The whole-tree number is asserted ABSENT, not merely the new
    number present: a runner that printed both would satisfy the weaker
    assertion while still handing a reader the claim that was wrong.
    """
    stem = _a_real_file_stem_other_than_this_one()
    on_disk = len(run_tests.test_files_on_disk())
    assert on_disk > 1

    rc, out = quietly(["--select", stem])

    assert f"from {on_disk} files" not in out, "the whole-tree denominator survived"
    assert "from 1 selected file(s): " + stem + ".py" in out
    assert "NOT WHOLE-TREE COVERAGE" in out
    assert rc != run_tests.EXIT_ALL_PASSED


def test_a_partial_pass_does_not_return_the_code_that_means_the_suite_passed():
    """``rc == 0`` is the check every caller writes, so 0 must mean the tree.

    Measured before the fix: a one-module ``--select`` run returned 0. A caller
    could not distinguish it from a green run of the whole suite by any means
    the runner offered except reading the prose.
    """
    stem = _a_real_file_stem_other_than_this_one()
    rc, _ = quietly(["--select", stem])
    assert rc == run_tests.EXIT_PARTIAL_PASSED
    assert rc != run_tests.EXIT_ALL_PASSED

    original = run_tests.collect
    try:
        run_tests.collect = _synthetic_full_collection
        full_rc, _ = quietly([])
    finally:
        run_tests.collect = original
    assert full_rc == run_tests.EXIT_ALL_PASSED
    assert full_rc != rc, "the partial and whole-tree verdicts share a code"


def test_a_failure_inside_a_partial_run_still_returns_the_failure_code():
    """SCOPE MUST NEVER SOFTEN A FAILURE.

    The new code exists to stop a partial run claiming whole-tree success. If
    it also swallowed a partial FAILURE it would be a worse defect than the one
    it replaced -- a red run reporting a code nobody treats as red.
    """
    def planted():
        raise AssertionError("planted failure, and it must be reported")

    stem = _a_real_file_stem_other_than_this_one()
    original = run_tests.collect

    def one_failing(select=None):
        return [(stem, "test_planted_failure", planted)]

    try:
        run_tests.collect = one_failing
        rc, out = quietly(["--select", stem])
    finally:
        run_tests.collect = original

    assert rc == run_tests.EXIT_SOMETHING_FAILED
    assert rc != run_tests.EXIT_PARTIAL_PASSED
    assert rc != run_tests.EXIT_ALL_PASSED
    assert "1 failed" in out
    assert "planted failure" in out


def test_the_completeness_check_is_scoped_under_select_rather_than_skipped():
    """It used to be gated off entirely, so a partial run reconciled nothing.

    Driven through ``uncollected_files`` directly AND through a whole run, for
    the reason the module docstring gives: a reconciliation reachable only
    through the collector can be excluded by a defect in the collector.
    """
    stem = _a_real_file_stem_other_than_this_one()

    # Scoped: with no cases at all, the only file the selector names is the
    # only file reported -- not every file in the tree.
    scoped_missing = run_tests.uncollected_files([], stem)
    assert scoped_missing == [f"{stem}.py"]
    assert len(scoped_missing) < len(run_tests.uncollected_files([]))

    # And the unscoped call is untouched: it is still the whole tree.
    assert run_tests.uncollected_files([]) == run_tests.test_files_on_disk()


def test_a_selected_file_that_contributes_nothing_refuses_the_partial_run():
    """POSITIVE CONTROL for the scoped reconciliation, planted on disk.

    A real file, because the reconciliation reads the disk and a mock of the
    disk would be testing the mock. ABSOLUTE PATH built from ``TESTS_DIR``.
    Before the fix this selector produced exit 2 -- ``NOTHING COLLECTED`` --
    which says "no test matched", a true-sounding answer to a question nobody
    asked. The file was there; it contributed nothing; that is a hole.
    """
    probe = run_tests.TESTS_DIR / "test_dev5_scope_probe.py"
    try:
        probe.write_text("# a test file with no test in it, on purpose\n")
        assert "test_dev5_scope_probe.py" in run_tests.test_files_on_disk()

        rc, out = quietly(["--select", "dev5_scope_probe"])

        assert rc == run_tests.EXIT_COLLECTION_INCOMPLETE
        assert rc != run_tests.EXIT_ALL_PASSED
        assert rc != run_tests.EXIT_PARTIAL_PASSED
        assert rc != run_tests.EXIT_NOTHING_COLLECTED
        assert "COLLECTION IS INCOMPLETE" in out
        assert "test_dev5_scope_probe.py" in out
        assert "SCOPE: only the file(s) matching" in out
        assert "passed," not in out, "a count was printed over a hole"
    finally:
        if probe.exists():
            probe.unlink()
    assert not probe.exists(), "the probe file outlived its test"
    assert "test_dev5_scope_probe.py" not in run_tests.test_files_on_disk()


def test_the_whole_tree_run_still_prints_the_whole_tree_denominator():
    """The other half of the pair. The fix must not narrow a FULL run.

    A remedy that made every run report ``1 selected file`` would make the
    symptom disappear and destroy the number the suite exists to produce.

    ``collect`` is substituted for the reason ``_synthetic_full_collection``
    documents: a real nested full run would collect this file and re-enter
    itself without bound. What is under test here is the SUMMARY the whole-tree
    branch prints, and that branch reads ``test_files_on_disk()`` directly
    rather than anything the substitute supplies, so the denominator in the
    transcript is the real one off the real disk.
    """
    on_disk = run_tests.test_files_on_disk()
    assert len(on_disk) > 1, "one file on disk -- this asserts nothing"

    original = run_tests.collect
    try:
        run_tests.collect = _synthetic_full_collection
        rc, out = quietly([])
    finally:
        run_tests.collect = original

    assert rc == run_tests.EXIT_ALL_PASSED
    assert f"from {len(on_disk)} files" in out
    assert "selected file(s)" not in out
    assert "PARTIAL RUN" not in out


def test_this_module_is_importable_under_the_name_the_collector_uses():
    """Guards the import path the runner depends on, which no test covered."""
    assert f"{run_tests.PACKAGE}.{SELF}" in sys.modules
