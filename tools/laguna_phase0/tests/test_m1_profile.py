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
"""Unit tests for the M1 emitter and decider.

NOT EXECUTED. Written without run authorisation. The E6b fixture suite in
``m1_decider_fixtures.py`` is the part that was run, and it was run because the
decider is a pure function of emitted quantities and needs no TPU.
"""

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m1_profile as m1


def thresholds():
    return common.Thresholds.load()


def quantity(value, name="q", units="seconds", confirmed=True):
    return m1.Quantity(value,
                       common.CounterProvenance(name=name, units=units,
                                                source="test fixture",
                                                units_confirmed=confirmed))


def step(idle, buckets, step_time=1.0, idle_loose=None, overlap=0.0, index=0,
         run_id="run-a", unmapped=0.0):
    return m1.DecodeStepProfile(
        step=index,
        run_id=run_id,
        step_time=quantity(step_time, "step_time"),
        idle_strict=quantity(idle, "idle_strict"),
        idle_loose=quantity(idle if idle_loose is None else idle_loose, "idle_loose"),
        buckets={k: quantity(v, f"bucket.{k}") for k, v in buckets.items()},
        bucket_overlap=quantity(overlap, "bucket_overlap"),
        unmapped_event_seconds=unmapped)


# --- interval arithmetic -------------------------------------------------


def test_union_counts_an_overlap_once_and_the_sum_counts_it_twice():
    intervals = [m1.Interval(0.0, 2.0), m1.Interval(1.0, 3.0)]
    assert m1.union_duration(intervals) == 3.0
    assert m1.total_duration(intervals) == 4.0


def test_the_emitter_clips_to_the_step_window():
    profile = m1.step_profile_from_intervals(
        step=0, run_id="r", window=m1.Interval(10.0, 11.0),
        intervals_by_bucket={"grouped_matmul": [m1.Interval(9.0, 10.5)]},
        mapping=m1.EventMapping({}, {"grouped_matmul": m1.BucketKind.COMPUTE}))
    assert profile.buckets["grouped_matmul"].value == 0.5


def test_the_non_overlap_term_is_dma_in_flight_with_no_compute():
    mapping = m1.EventMapping({}, {"grouped_matmul": m1.BucketKind.COMPUTE,
                                   "dma": m1.BucketKind.DMA})
    profile = m1.step_profile_from_intervals(
        step=0, run_id="r", window=m1.Interval(0.0, 1.0),
        intervals_by_bucket={"grouped_matmul": [m1.Interval(0.0, 0.4)],
                             "dma": [m1.Interval(0.4, 0.7)]},
        mapping=mapping)
    derived = m1.derive(profile)
    assert derived.idle_strict_s == 0.3
    assert derived.idle_loose_s == 0.6
    assert abs(derived.non_overlap_term_s - 0.3) < 1e-12


# --- Rule 1 --------------------------------------------------------------


def test_the_cuts_are_inclusive_and_mixed_is_the_strict_interior():
    assert m1.classify_f_host(0.45, 0.45, 0.12) is m1.HostBoundVerdict.HOLDS
    assert m1.classify_f_host(0.12, 0.45, 0.12) is m1.HostBoundVerdict.REFUTED
    assert m1.classify_f_host(0.30, 0.45, 0.12) is m1.HostBoundVerdict.MIXED


def test_mixed_is_a_result_and_not_an_abstention():
    chk = m1.decide_host_bound([m1.derive(step(0.30, {"grouped_matmul": 0.70}))],
                               thresholds())
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["verdict"] == "MIXED"


def test_an_empty_run_is_undetermined_and_never_a_refutation():
    chk = m1.decide_host_bound([], thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["verdict"] == "UNDETERMINED"


def test_idle_exceeding_the_step_is_an_instrument_fault_not_a_verdict():
    derived = [m1.derive(step(1.5, {"grouped_matmul": 0.1}))]
    chk = m1.decide_host_bound(derived, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["verdict"] == "UNDETERMINED"


def test_steps_in_different_branches_do_not_borrow_the_name_mixed():
    derived = [m1.derive(step(0.60, {"a": 0.40}, index=0)),
               m1.derive(step(0.05, {"a": 0.95}, index=1))]
    chk = m1.decide_host_bound(derived, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["verdict"] == "UNDETERMINED"
    assert chk.intermediates["branches_present"] == ["HOLDS", "REFUTED"]


def test_a_branch_that_moves_with_the_idle_definition_fails_the_run():
    derived = [m1.derive(step(0.05, {"c": 0.35, "dma": 0.60}, idle_loose=0.65))]
    chk = m1.check_non_overlap_term(derived, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["branch_strict"] == "REFUTED"
    assert chk.intermediates["branch_loose"] == "HOLDS"


def test_a_stable_branch_publishes_the_distance_from_both_cuts():
    derived = [m1.derive(step(0.05, {"c": 0.90, "dma": 0.05}, idle_loose=0.08))]
    chk = m1.check_non_overlap_term(derived, thresholds())
    assert chk.outcome is common.Outcome.PASSED
    assert "distance_from_holds_cut" in chk.intermediates


# --- Rule 2 --------------------------------------------------------------


def test_a_low_reference_voids_the_cut_before_the_ratio_is_read():
    pair = m1.EfficiencyPair(e_dec=0.49, e_ref=0.40,
                             run_id_dec="r", run_id_ref="r")
    chk = m1.decide_efficiency(pair, thresholds())
    assert chk.intermediates["verdict"] == "VOID"
    assert "ratio_e_dec_over_e_ref" not in chk.intermediates


def test_a_reference_from_another_run_is_not_comparable():
    pair = m1.EfficiencyPair(e_dec=0.70, e_ref=0.75,
                             run_id_dec="r1", run_id_ref="r2")
    chk = m1.decide_efficiency(pair, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["same_profile_run"] is False


def test_the_three_efficiency_branches():
    def verdict(e_dec, e_ref=0.80):
        pair = m1.EfficiencyPair(e_dec, e_ref, "r", "r")
        return m1.decide_efficiency(pair, thresholds()).intermediates["verdict"]

    assert verdict(0.68) == "BYTE_BOUND"          # ratio 0.85 exactly
    assert verdict(0.48) == "EFFICIENCY_DEFICIT"  # ratio 0.60 exactly
    assert verdict(0.56) == "PARTIAL"             # ratio 0.70


# --- E6a -----------------------------------------------------------------


def test_terms_that_exceed_the_step_are_an_arithmetic_contradiction():
    derived = [m1.derive(step(0.20, {"a": 0.50, "b": 0.60}))]
    chk = m1.check_closure(derived, thresholds())
    assert chk.outcome is common.Outcome.FAILED


def test_a_residual_is_published_and_never_adjudicated():
    derived = [m1.derive(step(0.20, {"a": 0.30}))]
    chk = m1.check_closure(derived, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert abs(chk.intermediates["unattributed_fraction_by_step"][0] - 0.50) < 1e-12
    assert chk.intermediates["threshold"]["value"] is None


def test_terms_that_sum_to_the_step_while_overlapping_do_not_read_as_closed():
    derived = [m1.derive(step(0.20, {"a": 0.50, "b": 0.30}, overlap=0.25))]
    chk = m1.check_closure(derived, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert abs(chk.intermediates["unattributed_fraction_by_step"][0]) < 1e-12
    assert "double count" in chk.reason


def test_no_overlap_measurement_means_the_residual_cannot_be_read_as_closure():
    profile = m1.DecodeStepProfile(
        step=0, run_id="r", step_time=quantity(1.0), idle_strict=quantity(0.2),
        idle_loose=quantity(0.2), buckets={"a": quantity(0.8)},
        bucket_overlap=None)
    chk = m1.check_bucket_overlap([m1.derive(profile)], thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["steps_without_an_overlap_measurement"] == [0]


# --- counter provenance --------------------------------------------------


def test_an_unconfirmed_unit_is_emitted_as_unknown_and_flagged():
    profile = m1.DecodeStepProfile(
        step=0, run_id="r", step_time=quantity(1.0), idle_strict=quantity(0.2),
        idle_loose=quantity(0.2), buckets={}, bucket_overlap=None,
        hbm_bytes=quantity(1e9, "hbm_bytes", units="bytes", confirmed=False))
    assert profile.to_dict()["hbm_bytes"]["provenance"]["units"].startswith("UNKNOWN")
    chk = m1.check_counter_units([profile])
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "hbm_bytes" in chk.intermediates["unconfirmed_counters"]


def test_the_reference_step_times_are_never_read_by_a_verdict():
    derived = [m1.derive(step(0.20, {"a": 0.80}))]
    ts = thresholds()
    m1.assess([step(0.20, {"a": 0.80})], None, ts)
    assert not any(p["key"] == "m1.reference_step_ms" for p in ts.provenance())
    assert derived[0].step_time_s == 1.0
