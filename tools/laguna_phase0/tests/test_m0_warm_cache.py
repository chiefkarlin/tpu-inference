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
"""Unit tests for the M0 warm-cache protocol.

**THE "NOT EXECUTED" NOTICE THAT USED TO BE HERE WAS STALE AND IS REMOVED.**
It was written before this package had a runner and it survived several
commits in which these tests were in fact collected and executed. A stale
disclaimer is not a safe default: it invites a reader either to discount a real
result or to conclude the file's own notes cannot be trusted.

What is true as of `run_tests.py` landing: these tests are collected and
executed by `tools/laguna_phase0/run_tests.py`, WHICH IS NOT PYTEST and
supports far less. A count from it means "collected and executed by the
fallback runner, and none raised" -- it does not mean "the suite passes", and
NOTHING IN THIS FILE HAS EVER BEEN RUN AGAINST HARDWARE.
"""

import json
import tempfile
import pathlib

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m0_warm_cache as m0

BUCKETS = [m0.Bucket("decode_tokens", 32), m0.Bucket("decode_tokens", 64)]


def thresholds():
    return common.Thresholds.load()


def build_evidence(*, warmed, start, end, env_ok=True, plan=None):
    plan = plan or m0.WarmupPlan(window_buckets=list(BUCKETS))
    evidence = m0.WarmupEvidence(
        identity={"pod_id": "POD_UID:fixture", "configuration_fingerprint": "abc"},
        plan=plan,
        warmed=[{"bucket": b.to_dict(), "started_utc": "t0", "finished_utc": "t1"}
                for b in warmed],
        environment={"VLLM_XLA_CHECK_RECOMPILATION": "1" if env_ok else "0"})
    if start is not None:
        evidence.window_start_counter = m0.CounterReading(value=start, source="fixture")
    if end is not None:
        evidence.window_end_counter = m0.CounterReading(value=end, source="fixture")
    return evidence


def test_a_warm_window_with_a_still_counter_is_valid():
    checks = m0.evaluate_window(build_evidence(warmed=BUCKETS, start=7, end=7),
                                thresholds())
    assert m0.window_verdict(checks) is m0.WindowVerdict.VALID


def test_recompilation_inside_the_window_voids_it():
    checks = m0.evaluate_window(build_evidence(warmed=BUCKETS, start=7, end=9),
                                thresholds())
    assert m0.window_verdict(checks) is m0.WindowVerdict.VOID
    counter = next(c for c in checks if c.name == "m0.no_in_window_recompilation")
    assert counter.intermediates["events_in_window"] == 2
    assert "VOID, not adjusted" in counter.reason


def test_an_unreadable_counter_is_undetermined_not_zero_events():
    checks = m0.evaluate_window(build_evidence(warmed=BUCKETS, start=None, end=None),
                                thresholds())
    assert m0.window_verdict(checks) is m0.WindowVerdict.UNDETERMINED


def test_the_negative_control_leaves_a_bucket_unwarmed_and_the_window_is_void():
    """Review finding B7-ii: M0's assertion has to be able to fire."""
    plan = m0.WarmupPlan(window_buckets=list(BUCKETS), leave_unwarmed=[BUCKETS[1]])
    assert plan.buckets_to_warm() == [BUCKETS[0]]
    control = plan.negative_control()
    assert control is not None and control.executed is False
    checks = m0.evaluate_window(
        build_evidence(warmed=plan.buckets_to_warm(), start=7, end=7, plan=plan),
        thresholds())
    assert m0.window_verdict(checks) is m0.WindowVerdict.VOID
    warmed = next(c for c in checks if c.name == "m0.buckets_warmed")
    assert warmed.intermediates["unwarmed_buckets_in_window"] == ["decode_tokens=64"]


def test_the_required_environment_is_asserted_not_assumed():
    checks = m0.evaluate_window(
        build_evidence(warmed=BUCKETS, start=0, end=0, env_ok=False), thresholds())
    env = next(c for c in checks if c.name == "m0.required_env")
    assert env.outcome is common.Outcome.FAILED
    assert m0.window_verdict(checks) is m0.WindowVerdict.VOID


def test_warmth_identity_changes_with_the_configuration():
    """A pod warmed under one configuration is not warm under the next."""
    first = m0.warmth_identity("pod-a", {"enable_continue_decode": False})
    second = m0.warmth_identity("pod-a", {"enable_continue_decode": True})
    assert first["pod_id"] == second["pod_id"]
    assert first["configuration_fingerprint"] != second["configuration_fingerprint"]


def test_probe_without_a_source_reads_nothing_rather_than_zero():
    reading = m0.RecompilationProbe().read()
    assert reading.value is None


def test_probe_prefers_an_in_process_counter():
    reading = m0.RecompilationProbe(counter=lambda: 3).read()
    assert reading.value == 3
    assert reading.units == "recompilation events"


def test_ledger_records_warm_up_and_both_counter_readings():
    counter = iter([1, 1])
    ledger = m0.WarmupLedger(plan=m0.WarmupPlan(window_buckets=list(BUCKETS)),
                             probe=m0.RecompilationProbe(counter=lambda: next(counter)),
                             configuration={"cell": "B-c32"},
                             pod_id="pod-a")
    ledger.warm_all(lambda bucket: None)
    with ledger:
        pass
    payload = ledger.evidence.to_dict()
    assert len(payload["warmed_buckets"]) == len(BUCKETS)
    assert payload["recompilation_counter_at_window_start"]["value"] == 1
    assert payload["recompilation_counter_at_window_end"]["value"] == 1


# --------------------------------------------------------------------------
# P3: NegativeControl.executed was hardcoded False, so the field that answers
# "has this control ever fired?" could not record a firing even after one.
#
# These tests route through m0.main(), the REAL PRODUCER, and read the JSON it
# writes. That distinction matters: asserting on a descriptor built by hand
# would only show that the dataclass can hold True. It would show nothing about
# whether the producer will ever emit True, which is the whole defect.
# --------------------------------------------------------------------------


def _corrupted_evidence_document():
    """Evidence as a harness would write it, for a run whose control DID fire."""
    plan = m0.WarmupPlan(window_buckets=list(BUCKETS), leave_unwarmed=[BUCKETS[1]])
    evidence = build_evidence(warmed=plan.buckets_to_warm(), start=7, end=9, plan=plan)
    return evidence.to_dict()


def test_the_producer_records_a_fired_control_as_executed():
    """P3. Reverting `executed=executed` in m0 turns exactly this test red."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path, out_path = root / "evidence.json", root / "m0.json"
        evidence_path.write_text(json.dumps(_corrupted_evidence_document()),
                                 encoding="utf-8")
        rc = m0.main(["evaluate", "--evidence", str(evidence_path),
                      "--out", str(out_path)])
        record = json.loads(out_path.read_text(encoding="utf-8"))

    # The control did what it was built to do, so the window is VOID and the
    # run is not reportable -- but neither of those is the point here.
    assert rc == 1
    assert record["reportable"] is False
    control = record["negative_control"]
    assert control is not None
    assert control["executed"] is True, (
        "the evaluate path reads evidence from a run that really happened; a "
        "control present in it has fired, and the artifact has to say so")


def test_a_planned_control_is_not_recorded_as_executed():
    """The other half of the check: True must not be unconditional.

    A test that only ever asserts True cannot tell a producer that records
    firings from one that records True always. This is the empty-case leg.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = pathlib.Path(tmp) / "plan.json"
        rc = m0.main(["plan", "--bucket", "decode_tokens=32",
                      "--bucket", "decode_tokens=64",
                      "--nc-leave-bucket-unwarmed", "decode_tokens=64",
                      "--out", str(out_path)])
        record = json.loads(out_path.read_text(encoding="utf-8"))

    assert rc == 0
    control = record["negative_control"]
    assert control is not None
    assert control["executed"] is False, (
        "a plan is a document. Nothing has run, so nothing has fired")


def test_a_clean_run_carries_no_control_at_all():
    """Third leg: `executed` is only meaningful when a control is present."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path, out_path = root / "evidence.json", root / "m0.json"
        evidence_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=7, end=7).to_dict()),
            encoding="utf-8")
        m0.main(["evaluate", "--evidence", str(evidence_path),
                 "--out", str(out_path)])
        record = json.loads(out_path.read_text(encoding="utf-8"))

    assert record["negative_control"] is None
    assert record["reportable"] is True


# --------------------------------------------------------------------------
# P4-class, M0 instance. The three process states must be distinguishable:
# 0 passed, 1 ran and did not pass, 2 could not run.
# --------------------------------------------------------------------------


def test_a_valid_window_exits_zero():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=7, end=7).to_dict()),
            encoding="utf-8")
        assert m0.main(["evaluate", "--evidence", str(evidence_path),
                        "--out", str(root / "m0.json")]) == 0


def test_a_void_window_exits_one_because_it_ran_and_did_not_pass():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=7, end=9).to_dict()),
            encoding="utf-8")
        assert m0.main(["evaluate", "--evidence", str(evidence_path),
                        "--out", str(root / "m0.json")]) == 1


def test_an_undetermined_window_is_not_two_and_is_not_one_either():
    """UNDETERMINED is a verdict the instrument reached, so it is not a 2.

    THE OLD NAME OF THIS TEST WAS `..._exits_one_and_not_two`, AND THAT NAME IS
    THE WHOLE DEFECT. The author asked "1 or 2?", answered it correctly, and
    pinned the answer. Nobody asked the other question: a VOID window ALSO
    exited 1, so the code that meant "I checked and it is bad" and the code
    that meant "I COULD NOT CHECK" were the same integer. The second is the
    worse fact and it was wearing the first one's clothes.

    Both halves are asserted here so neither can rot: not 2 (the run happened),
    and not 1 (a verdict was NOT reached).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=None, end=None).to_dict()),
            encoding="utf-8")
        rc = m0.main(["evaluate", "--evidence", str(evidence_path),
                      "--out", str(root / "m0.json")])
        assert rc != common.EXIT_COULD_NOT_RUN
        assert rc != common.EXIT_DECIDED_NOT_CLEAN
        assert rc == common.EXIT_RAN_BUT_COULD_NOT_DECIDE


def test_void_and_undetermined_do_not_share_an_exit_code():
    """THE DEFECT ITSELF, PINNED AS ONE ASSERTION RATHER THAN TWO CONSTANTS.

    Asserting each code separately would still pass if a later edit collapsed
    them onto the same NEW value. This compares the two codes an operator
    actually receives, from two real runs, so the property survives renaming.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        void_path = root / "void.json"
        void_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=7, end=9).to_dict()),
            encoding="utf-8")
        undet_path = root / "undet.json"
        undet_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=None, end=None).to_dict()),
            encoding="utf-8")
        void_rc = m0.main(["evaluate", "--evidence", str(void_path),
                           "--out", str(root / "void-out.json")])
        undet_rc = m0.main(["evaluate", "--evidence", str(undet_path),
                            "--out", str(root / "undet-out.json")])
        # POSITIVE CONTROL ON THE FIXTURES: if a mistake made both documents
        # produce the SAME verdict, the inequality below would be measuring
        # nothing. Pin what each one actually is before comparing them.
        assert void_rc == common.EXIT_DECIDED_NOT_CLEAN
        assert undet_rc == common.EXIT_RAN_BUT_COULD_NOT_DECIDE
        assert void_rc != undet_rc
        # AND NEITHER MAY BE GREEN. This is the property that made the change
        # safe to land: a refinement of 1, never a promotion to 0.
        assert void_rc != 0 and undet_rc != 0


def test_a_missing_evidence_file_exits_two_and_not_one():
    """The defect: this used to escape as a traceback, which Python exits 1 on.

    An operator scripting M0 could not tell "the window is void" from "you gave
    me a path that is not there". Both were 1.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        assert m0.main(["evaluate", "--evidence", str(root / "absent.json"),
                        "--out", str(root / "m0.json")]) == 2


def test_evidence_that_is_not_json_exits_two_and_not_one():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path = root / "evidence.json"
        evidence_path.write_text("this is not json", encoding="utf-8")
        assert m0.main(["evaluate", "--evidence", str(evidence_path),
                        "--out", str(root / "m0.json")]) == 2


# --- C2 / R7 decidable half: absent is not empty, and empty is not a pass ---
#
# THE ENUMERATION THE BRIEF ASKED FOR, BEFORE THE COERCION MOVED. Across the
# whole owned tree there is exactly ONE definition of `evidence_from_dict`
# (m0_warm_cache.py:403) and exactly ONE call site (m0_warm_cache.py:487, the
# `evaluate` arm of main). Nothing else in the package or the tree calls it.
#
# THE MORE INTERESTING HALF OF THAT ENUMERATION IS WHAT DOES *NOT* APPEAR IN
# IT. Every test above builds a `WarmupEvidence` by calling the constructor
# directly through `build_evidence`. NOT ONE of them reaches the function
# whose coercion is the defect. The suite could therefore never have caught
# this, no matter how thorough it was about `evaluate_window`, because the
# coercion sits on a path the suite does not use. That is the shape to look
# for elsewhere: a parser that only production traffic exercises, guarded by
# tests that hand-build the parsed object.


def _document(**overrides):
    """A minimally valid on-disk evidence document, with keys removable.

    Built by round-tripping a real ledger-shaped object so the baseline is
    something the emitter could actually produce, rather than a hand-written
    dict that agrees with my reading of the format. Pass a key with value
    `_ABSENT` to delete it.
    """
    doc = build_evidence(warmed=BUCKETS, start=7, end=7).to_dict()
    for key, value in overrides.items():
        if value is _ABSENT:
            doc.pop(key, None)
        else:
            doc[key] = value
    return doc


_ABSENT = object()


def _named(checks, name):
    return next(c for c in checks if c.name == name)


def test_an_evidence_document_with_no_plan_does_not_vacuously_pass():
    """THE DEFECT. A document with no plan at all reported every bucket warmed.

    `data.get("plan", {})` made an absent plan into an empty plan, and an empty
    plan has no unwarmed buckets, so `missing` was empty and the check said
    PASSED with the words "every bucket the window uses was warmed first". The
    window could then be VALID and main could exit 0.

    ABSENCE OF EVIDENCE WAS BEING REPORTED AS EVIDENCE OF WARMTH, which is the
    flattering direction, and it is reachable from any harness that fails to
    write the plan -- exactly the harness most likely to have got other things
    wrong too.
    """
    evidence = m0.evidence_from_dict(_document(plan=_ABSENT))
    check = _named(m0.evaluate_window(evidence, thresholds()), "m0.buckets_warmed")
    assert check.outcome is common.Outcome.UNDETERMINED
    assert "plan" in check.intermediates.get("absent_keys", [])


def test_an_absent_window_bucket_list_is_not_an_empty_one():
    """The same coercion one level down, where the plan exists but is hollow."""
    evidence = m0.evidence_from_dict(_document(plan={"leave_unwarmed": []}))
    check = _named(m0.evaluate_window(evidence, thresholds()), "m0.buckets_warmed")
    assert check.outcome is common.Outcome.UNDETERMINED
    assert "plan.window_buckets" in check.intermediates.get("absent_keys", [])


def test_an_explicitly_empty_window_is_undetermined_and_not_a_pass():
    """Present-but-empty is a different fact from absent, and neither passes.

    This one is NOT a parse defect -- the document says what it means. It is
    still not a pass: "every bucket the window uses was warmed" is vacuously
    true of a window that uses none, and a vacuous truth is not a measurement.
    Kept separate from the absent case so the two cannot be conflated, and the
    reason text has to distinguish them.
    """
    evidence = m0.evidence_from_dict(
        _document(plan={"window_buckets": [], "leave_unwarmed": []}))
    check = _named(m0.evaluate_window(evidence, thresholds()), "m0.buckets_warmed")
    assert check.outcome is common.Outcome.UNDETERMINED
    assert "absent_keys" not in check.intermediates or not [
        k for k in check.intermediates["absent_keys"] if "window_buckets" in k]


def test_an_absent_warmed_list_is_undetermined_and_not_a_failure():
    """The same defect pointing the OTHER way, and it is still a defect.

    `data.get("warmed_buckets", [])` turned "the document does not say what was
    warmed" into "nothing was warmed", which reports FAILED. That direction is
    unflattering, so it is tempting to leave alone -- but a check that is wrong
    in the safe direction is still wrong, and an operator who gets VOID for a
    missing key will go looking for a cache problem that does not exist.

    Present-and-empty still FAILS, immediately below, so this does not weaken
    the check; it separates two facts that were being reported as one.
    """
    evidence = m0.evidence_from_dict(_document(warmed_buckets=_ABSENT))
    check = _named(m0.evaluate_window(evidence, thresholds()), "m0.buckets_warmed")
    assert check.outcome is common.Outcome.UNDETERMINED
    assert "warmed_buckets" in check.intermediates.get("absent_keys", [])


def test_an_explicitly_empty_warmed_list_still_fails():
    """THE NEGATIVE CONTROL ON THE FIX ABOVE.

    If this went UNDETERMINED too, the previous test would have bought its
    result by disabling the detector rather than by sharpening it. A document
    that positively asserts nothing was warmed, for a window that uses buckets,
    is a VOID window and must stay FAILED.
    """
    evidence = m0.evidence_from_dict(_document(warmed_buckets=[]))
    check = _named(m0.evaluate_window(evidence, thresholds()), "m0.buckets_warmed")
    assert check.outcome is common.Outcome.FAILED


def test_a_complete_document_still_passes_through_the_parser():
    """THE SECOND NEGATIVE CONTROL: the parser must not have become a wall.

    Every test above asserts something stopped being a pass. Without this one
    they are all satisfied by a parser that refuses everything, which is the
    constant-1 failure the P1 item was reopened over -- the same defect with a
    louder failure mode.
    """
    evidence = m0.evidence_from_dict(_document())
    checks = m0.evaluate_window(evidence, thresholds())
    assert _named(checks, "m0.buckets_warmed").outcome is common.Outcome.PASSED
    assert m0.window_verdict(checks) is m0.WindowVerdict.VALID


def test_a_document_with_no_plan_does_not_exit_zero():
    """The end-to-end consequence, at the exit code, through main.

    The unit assertions above are about a check object. This is about what an
    operator's shell sees, which is the thing that actually gates a run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        path = root / "evidence.json"
        path.write_text(json.dumps(_document(plan=_ABSENT)), encoding="utf-8")
        rc = m0.main(["evaluate", "--evidence", str(path),
                      "--out", str(root / "m0.json")])
        # THE NAMED CONTRACT FIRST, THE EXACT VALUE SECOND. This assertion used
        # to read `== 1`, which was NARROWER THAN THE CONTRACT THE NAME STATES
        # and pinned an answer the test was not written to be about. A missing
        # plan is not-decidable, not decided-bad.
        assert rc != 0
        assert rc == common.EXIT_RAN_BUT_COULD_NOT_DECIDE


def test_the_round_trip_a_real_harness_performs_is_unaffected():
    """Evidence written by the ledger and read back must be untouched by this.

    The fix adds a notion of "absent" that only the parser can produce. A
    ledger-built object has every field, so its absent-key list must be empty
    and its verdict identical to the direct-construction path. If this ever
    fails, the parser and the emitter have drifted apart.
    """
    direct = build_evidence(warmed=BUCKETS, start=7, end=7)
    round_tripped = m0.evidence_from_dict(json.loads(json.dumps(direct.to_dict())))
    assert round_tripped.absent_keys == []
    assert (m0.window_verdict(m0.evaluate_window(round_tripped, thresholds()))
            is m0.window_verdict(m0.evaluate_window(direct, thresholds())))
