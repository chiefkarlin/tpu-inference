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
"""Tests for the shared plumbing in ``common.py``.

Runs under ``tools/laguna_phase0/run_tests.py``, a fallback runner and NOT
pytest. Counts from it are observations from that runner.

The redaction tests here are written to PIN THE HOLE rather than to certify the
mechanism. A test that only shows `GITHUB_TOKEN` being redacted would read as
"credentials are protected", which is not what the code does.
"""

import os

import pytest

from tools.laguna_phase0 import common


def test_a_name_that_looks_like_a_credential_is_recorded_by_length_only():
    os.environ["LAGUNA_TEST_FAKE_TOKEN"] = "abcdefgh"
    try:
        snap = common.env_snapshot(["LAGUNA_TEST_FAKE_TOKEN"])
    finally:
        del os.environ["LAGUNA_TEST_FAKE_TOKEN"]
    assert snap["LAGUNA_TEST_FAKE_TOKEN"]["present"] is True
    assert snap["LAGUNA_TEST_FAKE_TOKEN"]["length"] == 8
    assert snap["LAGUNA_TEST_FAKE_TOKEN"]["value"] == "REDACTED-BY-POLICY"
    assert "abcdefgh" not in repr(snap)


def test_an_absent_credential_name_reports_absence_and_not_a_zero_length_value():
    snap = common.env_snapshot(["LAGUNA_TEST_ABSENT_SECRET"])
    assert snap["LAGUNA_TEST_ABSENT_SECRET"]["present"] is False
    assert snap["LAGUNA_TEST_ABSENT_SECRET"]["length"] == 0


def test_the_redaction_denylist_fails_open_and_this_test_says_so_out_loud():
    """NON-BLOCKING REVIEW ITEM, PINNED RATHER THAN CLOSED.

    `_SECRETISH` is a DENYLIST. A variable whose NAME does not contain one of
    its seven words has its VALUE written into the artifact in full. This test
    exists so that the hole is a documented, asserted property instead of a
    surprise, and so that anyone who later believes `env_snapshot` sanitises an
    arbitrary environment has to delete an assertion that says otherwise.

    THE FIRST LINE OF DEFENCE IS THE CALLER'S EXPLICIT `RECORDED_ENV` LIST, not
    this pattern. Do not pass `os.environ` to `env_snapshot`.
    """
    for name in ("LAGUNA_TEST_SESSION", "LAGUNA_TEST_COOKIE",
                 "LAGUNA_TEST_SIGNATURE", "LAGUNA_TEST_PRIVATE_PEM",
                 "LAGUNA_TEST_BEARER", "LAGUNA_TEST_SALT"):
        os.environ[name] = "not-redacted-by-this-code"
        try:
            snap = common.env_snapshot([name])
        finally:
            del os.environ[name]
        assert snap[name] == "not-redacted-by-this-code", (
            f"{name} is now redacted. If that was deliberate, good -- update "
            "this list and the _SECRETISH comment together. If it was not, "
            "the denylist has changed shape and the docstring is stale.")


def test_an_ordinary_variable_is_recorded_by_value():
    """The control: the redaction must not be unconditional either."""
    os.environ["LAGUNA_TEST_PLAIN"] = "c32"
    try:
        snap = common.env_snapshot(["LAGUNA_TEST_PLAIN"])
    finally:
        del os.environ["LAGUNA_TEST_PLAIN"]
    assert snap["LAGUNA_TEST_PLAIN"] == "c32"


# --------------------------------------------------------------------------
# Outcome / worst()
# --------------------------------------------------------------------------


def test_undetermined_outranks_passed_and_failed_outranks_everything():
    assert common.worst([common.Outcome.PASSED,
                         common.Outcome.UNDETERMINED]) is common.Outcome.UNDETERMINED
    assert common.worst([common.Outcome.UNDETERMINED,
                         common.Outcome.FAILED]) is common.Outcome.FAILED


def test_an_empty_payload_is_refused_rather_than_written():
    with pytest.raises(common.IntermediatesMissingError):
        common.write_artifact("/tmp/laguna-phase0-should-not-exist.json",
                              kind="TEST", payload={})
