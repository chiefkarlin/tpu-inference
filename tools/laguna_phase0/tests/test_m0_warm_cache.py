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

NOT EXECUTED. Written without run authorisation; the presence of this file is
not evidence that anything in it passes.
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


def test_an_undetermined_window_exits_one_and_not_two():
    """UNDETERMINED is a verdict the instrument reached, so it is not a 2."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(
            json.dumps(build_evidence(warmed=BUCKETS, start=None, end=None).to_dict()),
            encoding="utf-8")
        assert m0.main(["evaluate", "--evidence", str(evidence_path),
                        "--out", str(root / "m0.json")]) == 1


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
