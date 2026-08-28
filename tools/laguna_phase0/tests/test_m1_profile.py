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

RUN, as of round 1's fix pass, through ``tools/laguna_phase0/run_tests.py`` --
the in-repository fallback collector, which is NOT pytest and supports only
``raises`` and ``approx``. Before that collector was shipped this file had never
been collected at all (round 1 R10), and the count it now produces is an
observation from that collector rather than from pytest.

**The hand-built ``step()`` helper below tests the DECIDER and nothing else.**
Round 1 R2 found that every test of the closure and idle legs constructed
``DecodeStepProfile`` directly, supplying ``idle_strict`` as a free argument. In
a real profile ``idle_strict`` is DERIVED from the intervals and is constrained
by them, so a hand-built profile can express states the emitter cannot produce
-- and, worse, cannot express the emitter's own mistakes. That is why a green
suite missed C1 and R1. Tests of the EMITTER live in the section marked
"emitter-routed" and must call ``step_profile_from_intervals``.
"""

import random

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m1_profile as m1


def thresholds():
    return common.Thresholds.load()


class approx:
    """Tolerant float comparison, local so this file needs no test framework.

    Interval arithmetic here is a difference of sums, so exact equality is the
    wrong assertion even when the maths is exact: ``1.0 - 0.7`` is
    ``0.30000000000000004``. One assertion in this file used ``==`` and was
    never collected, so nobody found out.
    """

    def __init__(self, expected, tol=1e-12):
        self.expected, self.tol = expected, tol

    def __eq__(self, other):
        return abs(float(other) - float(self.expected)) <= self.tol

    def __repr__(self):
        return f"approx({self.expected!r}, tol={self.tol})"


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
    # Exact `== 0.3` here: 1.0 - 0.7 is 0.30000000000000004. This assertion had
    # never been collected, so the failure had never been seen. It is a FALSE
    # RED -- the emitter was right and the test was wrong -- which is why the
    # adjacent non-overlap assertion below already used a tolerance.
    assert derived.idle_strict_s == approx(0.3)
    assert derived.idle_loose_s == approx(0.6)
    assert derived.non_overlap_term_s == approx(0.3)


# --- emitter-routed (R2): these MUST go through step_profile_from_intervals --
#
# Every test in this section drives the real emitter. None of them may build a
# DecodeStepProfile by hand, because the defects they exist to catch are
# defects IN the emitter and a hand-built profile cannot exhibit them.


def emitted_step(*, mapped=None, unmapped=(), window=(0.0, 1.0), kinds=None,
                 index=0, run_id="run-a"):
    """One emitter-produced profile. The only profile factory this section uses."""
    mapped = {"grouped_matmul": [m1.Interval(0.0, 0.4)]} if mapped is None else mapped
    kinds = ({k: m1.BucketKind.COMPUTE for k in mapped} if kinds is None else kinds)
    return m1.step_profile_from_intervals(
        step=index, run_id=run_id, window=m1.Interval(*window),
        intervals_by_bucket=mapped,
        mapping=m1.EventMapping({}, kinds),
        unmapped=[(n, m1.Interval(*iv)) for n, iv in unmapped])


def test_unmapped_device_busy_time_is_not_booked_as_host_idle():
    """C1. 1.0 s step: 0.40 mapped compute, 0.30 unmapped BUSY, 0.30 truly idle.

    An unmapped kernel is a kernel executing. Strict idle is defined as a gap
    with no kernel executing and no DMA in flight, so the 0.30 s of unmapped
    device-busy time is not idle and must not be counted as such.
    """
    profile = emitted_step(unmapped=[("mystery_kernel", (0.4, 0.7))])
    assert profile.unmapped_event_seconds == approx(0.30)
    assert profile.unmapped_event_names == ("mystery_kernel",)

    derived = m1.derive(profile)
    assert derived.idle_strict_s == approx(0.30), (
        "unmapped device-busy time was booked as strict idle")
    assert derived.f_host_strict == approx(0.30)


def test_unmapped_busy_time_does_not_convert_mixed_into_holds():
    """C1, at the level that survives retelling: the published Rule 1 verdict.

    True f_host here is 0.30, which is MIXED. Booking the unmapped 0.30 s as
    idle emits 0.60, which clears the 0.45 cut and publishes HOLDS -- the
    string that travels into a summary detached from every caveat.
    """
    profile = emitted_step(unmapped=[("mystery_kernel", (0.4, 0.7))])
    chk = m1.decide_host_bound([m1.derive(profile)], thresholds())
    assert chk.intermediates["verdict"] == "MIXED", (
        "the emitter manufactured HOLDS out of unmapped device-busy time")
    assert chk.intermediates["aggregate_f_host"] == approx(0.30)


def test_the_closure_residual_can_be_positive_on_an_emitted_profile():
    """R1. Unmapped time is time the instrument did NOT attribute to a term.

    The residual is ``S - (I + every NAMED term)``. An unmapped event is by
    definition not a named term, so the 0.30 s belongs in the residual. If it
    is added to the attributed side instead, the residual reports perfect
    closure over time nobody accounted for.
    """
    profile = emitted_step(unmapped=[("mystery_kernel", (0.4, 0.7))])
    derived = m1.derive(profile)
    assert derived.unattributed_s > 0.0, (
        "the closure residual cannot go positive, so it has never been a check")
    assert derived.unattributed_s == approx(0.30)
    assert derived.unattributed_fraction == approx(0.30)


def test_the_closure_residual_is_not_non_positive_by_construction():
    """R1's control. The defect was that 2000 random profiles gave max 0.0.

    This is the test that must be watched failing: it is not an assertion about
    one profile, it is an assertion that the TERM CAN FIRE AT ALL. A quieter
    version of a check that cannot fire would still pass every single-profile
    test above.

    THE THRESHOLD IS 2000 OF 2000, NOT ONE. An earlier revision asserted
    ``positives > 0`` while the commit message claimed all 2000 fired. The
    assertion was therefore weaker than the claim made for it, and a
    regression that killed 1999 of the 2000 would have passed it silently. The
    construction guarantees a positive residual on every draw -- ``cuts`` is
    sorted, so the unmapped span is non-negative and is disjoint from the
    mapped one -- so anything less than 2000 is a real change and not noise.
    Measured min residual over this seed: 3.59e-05, seven orders of magnitude
    above the 1e-12 floor, so the count is not sitting on the threshold.
    """
    rng = random.Random(20260828)
    draws = 2000
    positives = 0
    for _ in range(draws):
        cuts = sorted(rng.uniform(0.0, 1.0) for _ in range(4))
        mapped = {"grouped_matmul": [m1.Interval(cuts[0], cuts[1])]}
        unmapped = [("mystery", (cuts[2], cuts[3]))]
        derived = m1.derive(emitted_step(mapped=mapped, unmapped=unmapped))
        if derived.unattributed_s > 1e-12:
            positives += 1
    assert positives == draws, (
        f"only {positives} of {draws} emitter-produced profiles yielded a "
        "positive residual; every draw is constructed to yield one, so the "
        "closure term has become unable to fire on some inputs")


def test_a_double_count_still_drives_the_residual_negative_from_the_emitter():
    """R1 must not be fixed by making the residual unable to go NEGATIVE either.

    Two named terms covering the same wall time is a double count, and the
    emitter's own overlap term should push the residual below zero.
    """
    profile = emitted_step(
        mapped={"a": [m1.Interval(0.0, 0.6)], "b": [m1.Interval(0.3, 0.9)]})
    derived = m1.derive(profile)
    assert derived.bucket_overlap_s == approx(0.30)
    assert derived.unattributed_s < 0.0
    chk = m1.check_closure([derived], thresholds())
    assert chk.outcome is common.Outcome.FAILED


def test_check_closure_reads_an_emitted_profile_and_not_a_hand_built_one():
    """R2. The closure leg's coverage must include real emitter output."""
    profile = emitted_step(unmapped=[("mystery_kernel", (0.4, 0.7))])
    chk = m1.check_closure([m1.derive(profile)], thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["unattributed_fraction_by_step"][0] == approx(0.30)
    assert chk.intermediates["unmapped_seconds_by_step"][0] == approx(0.30)


def test_the_emitter_reports_unmapped_time_it_clipped_to_the_window():
    """Unmapped events straddling the window contribute only the part inside."""
    profile = emitted_step(unmapped=[("straddler", (0.8, 1.5)),
                                     ("outside", (2.0, 3.0))])
    assert profile.unmapped_event_seconds == approx(0.20)
    assert profile.unmapped_event_names == ("straddler",)


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


def test_a_zero_reference_is_an_instrument_fault_and_not_an_orderly_void():
    """R3. Every non-positive e_ref is also below the 0.5 floor, so the floor
    test used to answer first and reported PASSED/VOID -- a legitimate-looking
    disposition -- for a reading that is physically impossible.

    VOID is a result about the MACHINE. FAILED is a result about the
    MEASUREMENT. Folding the second into the first is exactly the collapse the
    module's two-axis design exists to prevent.
    """
    pair = m1.EfficiencyPair(e_dec=0.60, e_ref=0.0,
                             run_id_dec="r", run_id_ref="r")
    chk = m1.decide_efficiency(pair, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["verdict"] == "UNDETERMINED"
    assert "ratio_e_dec_over_e_ref" not in chk.intermediates


def test_a_negative_reference_is_an_instrument_fault_too():
    pair = m1.EfficiencyPair(e_dec=0.60, e_ref=-0.25,
                             run_id_dec="r", run_id_ref="r")
    chk = m1.decide_efficiency(pair, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["verdict"] == "UNDETERMINED"


def test_a_small_but_positive_reference_still_voids_rather_than_failing():
    """The other side of the hoist: it must not swallow the VOID branch.

    A reference that is implausibly low but physically possible is still a
    statement about the machine, not about the instrument, and VOID is still
    the right disposition for it.
    """
    pair = m1.EfficiencyPair(e_dec=0.10, e_ref=1e-9,
                             run_id_dec="r", run_id_ref="r")
    chk = m1.decide_efficiency(pair, thresholds())
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["verdict"] == "VOID"


def test_a_cross_run_pair_is_still_answered_before_the_instrument_fault():
    """Ordering control. The hoist moves e_ref <= 0 above the FLOOR test only.

    It must not jump the same-run test: a non-positive reference imported from
    a different run is not comparable in the first place, and answering
    "instrument fault" would assert more than we know.
    """
    pair = m1.EfficiencyPair(e_dec=0.60, e_ref=0.0,
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
    """The absence leg. See the POSITIVE CONTROL below, which is what makes
    this one mean anything.

    Non-blocking review item: on its own this test asserts only that a key is
    NOT in a list, and it would pass just as happily against an empty list, a
    misspelled key, or a provenance() that had silently stopped recording. It
    is non-vacuous today because five keys are in fact recorded -- but "today"
    is doing load-bearing work in that sentence, so the control below pins it.
    """
    derived = [m1.derive(step(0.20, {"a": 0.80}))]
    ts = thresholds()
    m1.assess([step(0.20, {"a": 0.80})], None, ts)
    assert not any(p["key"] == "m1.reference_step_ms" for p in ts.provenance())
    assert derived[0].step_time_s == 1.0


def test_the_never_read_assertion_is_run_against_a_recording_provenance():
    """POSITIVE CONTROL for the test above. An absence proves nothing unless
    the instrument that reports it can also report a presence."""
    ts = thresholds()
    m1.assess([step(0.20, {"a": 0.80})], None, ts)
    recorded = [p["key"] for p in ts.provenance()]

    # 1. The recorder is recording at all -- not an empty list.
    assert recorded, ("provenance() returned nothing, so the absence of "
                      "m1.reference_step_ms above is vacuous")

    # 2. It records the keys the verdict path DOES read, by name. If a future
    #    change stops recording, this list empties and the control fires
    #    before the absence test can go quietly green.
    for key in ("m1.f_host_holds_at_or_above", "m1.f_host_refuted_at_or_below"):
        assert key in recorded, f"{key} should have been served and recorded"

    # 3. The key under test IS spellable and IS present in the file, so the
    #    absence above is a fact about reading and not a typo.
    assert ts.entry("m1.reference_step_ms").key == "m1.reference_step_ms"
