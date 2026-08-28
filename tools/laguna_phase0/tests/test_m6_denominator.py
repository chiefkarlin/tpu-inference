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
"""Unit tests for the M6 denominator assertion.

NOT EXECUTED. These were written without run authorisation and have never been
collected: pytest collection imports the package, and the presence of this file
is not evidence that anything in it passes.

The test that carries the review's finding B7-i is
:func:`test_consistent_redenomination_leaves_the_ridge_invariant`: it pins the
reason a wrong *count* is the least consequential corruption, and the reason the
pairing is what has to be guarded.
"""

import pytest

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m6_denominator as m6

DEVICES_PER_CHIP = 2


def thresholds():
    return common.Thresholds.load()


def chiplets(chip_coords):
    """Two JAX devices per chip, sharing coordinates, differing in core_on_chip."""
    views = []
    device_id = 0
    for coords in chip_coords:
        for core in range(DEVICES_PER_CHIP):
            views.append(
                m6.DeviceView(device_id=device_id,
                              process_index=0,
                              slice_index=0,
                              coords=coords,
                              core_on_chip=core))
            device_id += 1
    return views


FOUR_CHIPS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]


def test_coords_route_counts_chips_not_chiplets():
    evidence = m6.chip_count_from_coords(chiplets(FOUR_CHIPS), DEVICES_PER_CHIP)
    assert evidence.chip_count == 4
    assert evidence.detail["device_count"] == 4 * DEVICES_PER_CHIP


def test_coords_route_abstains_when_chiplets_do_not_collapse():
    """One device per coordinate is the shape a banner-derived count has.

    The route must return nothing rather than return the device count under a
    chip's name.
    """
    views = [
        m6.DeviceView(device_id=i, coords=coords, core_on_chip=0)
        for i, coords in enumerate(FOUR_CHIPS)
    ]
    evidence = m6.chip_count_from_coords(views, DEVICES_PER_CHIP)
    assert evidence.chip_count is None
    assert "collapsed chiplets" in evidence.detail["why"]


def test_coords_route_abstains_without_coordinates():
    views = [m6.DeviceView(device_id=0, coords=None)]
    assert m6.chip_count_from_coords(views, DEVICES_PER_CHIP).chip_count is None


def test_gke_route_abstains_when_the_unit_is_not_declared():
    evidence = m6.chip_count_from_gke(4, None, DEVICES_PER_CHIP)
    assert evidence.chip_count is None
    assert "unit" in evidence.detail["why"]


def test_gke_route_converts_a_declared_device_denominated_allocation():
    assert m6.chip_count_from_gke(8, "devices", DEVICES_PER_CHIP).chip_count == 4
    assert m6.chip_count_from_gke(4, "chips", DEVICES_PER_CHIP).chip_count == 4


def test_single_route_is_undetermined_not_passed():
    _, chk = m6.reconcile_chip_count([
        m6.chip_count_from_coords(chiplets(FOUR_CHIPS), DEVICES_PER_CHIP),
        m6.chip_count_from_gke(None, None, DEVICES_PER_CHIP),
    ])
    assert chk.outcome is common.Outcome.UNDETERMINED


def test_routes_that_disagree_fail():
    count, chk = m6.reconcile_chip_count([
        m6.chip_count_from_coords(chiplets(FOUR_CHIPS), DEVICES_PER_CHIP),
        m6.chip_count_from_gke(8, "chips", DEVICES_PER_CHIP),
    ])
    assert count is None
    assert chk.outcome is common.Outcome.FAILED


def test_no_route_at_all_is_undetermined():
    _, chk = m6.reconcile_chip_count([
        m6.chip_count_from_coords([], DEVICES_PER_CHIP),
        m6.chip_count_from_gke(None, None, DEVICES_PER_CHIP),
    ])
    assert chk.outcome is common.Outcome.UNDETERMINED


def test_pinned_pair_passes_every_basis_check():
    th = thresholds()
    pinned = m6.pinned_basis(th)
    tol = float(th.require("m6.ridge_relative_tolerance"))
    assert m6.check_basis_labels(pinned).outcome is common.Outcome.PASSED
    assert m6.check_pinned_values(pinned, pinned).outcome is common.Outcome.PASSED
    assert m6.check_pairing(pinned, pinned, tol).outcome is common.Outcome.PASSED


def test_consistent_redenomination_leaves_the_ridge_invariant():
    """Review finding B7-i, in code.

    Halving both sides is a re-denomination, not a fault: the ridge point is
    their quotient and it does not move, so no arm flips. The labels are what
    catch it, and the pairing check correctly does not.
    """
    th = thresholds()
    pinned = m6.pinned_basis(th)
    tol = float(th.require("m6.ridge_relative_tolerance"))
    redenominated = m6.Injection(consistent_redenomination=True).apply_to_basis(pinned)
    assert redenominated.ridge_flops_per_byte == pytest.approx(
        pinned.ridge_flops_per_byte, rel=1e-12)
    assert m6.check_pairing(redenominated, pinned, tol).outcome is common.Outcome.PASSED
    assert m6.check_basis_labels(redenominated).outcome is common.Outcome.FAILED
    assert m6.check_pinned_values(redenominated, pinned).outcome is common.Outcome.FAILED


def test_inconsistent_pairing_fails_the_pairing_check():
    """Negative control leg 2 -- the fault that actually flips an arm."""
    th = thresholds()
    pinned = m6.pinned_basis(th)
    tol = float(th.require("m6.ridge_relative_tolerance"))
    mispaired = m6.Injection(inconsistent_pairing=True).apply_to_basis(pinned)
    chk = m6.check_pairing(mispaired, pinned, tol)
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["supplied_ridge_flops_per_byte"] == pytest.approx(
        pinned.ridge_flops_per_byte / 2.0, rel=1e-12)


def test_mislabelled_but_inconsistent_pair_is_still_caught_by_the_ridge():
    """A label can lie; the quotient cannot."""
    th = thresholds()
    pinned = m6.pinned_basis(th)
    tol = float(th.require("m6.ridge_relative_tolerance"))
    liar = m6.RooflineBasis(dense_bf16_tflops=pinned.dense_bf16_tflops / 2.0,
                            hbm_bandwidth_gbytes_per_s=pinned.hbm_bandwidth_gbytes_per_s,
                            flops_basis=m6.Basis.PER_CHIP,
                            bandwidth_basis=m6.Basis.PER_CHIP,
                            source_row="fixture: mislabelled per-device FLOPs")
    assert m6.check_basis_labels(liar).outcome is common.Outcome.PASSED
    assert m6.check_pairing(liar, pinned, tol).outcome is common.Outcome.FAILED


def test_assert_denominator_raises_on_leg_1_injection():
    th = thresholds()
    with pytest.raises(m6.DenominatorAssertionError):
        m6.assert_denominator(views=chiplets(FOUR_CHIPS),
                              thresholds=th,
                              gke_allocation=4,
                              gke_allocation_unit="chips",
                              tensor_parallel_size=8,
                              injection=m6.Injection(chip_count_scale=2.0))


def test_assert_denominator_raises_on_leg_2_injection():
    th = thresholds()
    with pytest.raises(m6.DenominatorAssertionError):
        m6.assert_denominator(views=chiplets(FOUR_CHIPS),
                              thresholds=th,
                              gke_allocation=4,
                              gke_allocation_unit="chips",
                              tensor_parallel_size=8,
                              injection=m6.Injection(inconsistent_pairing=True))


def test_assert_denominator_raises_when_nothing_could_be_checked():
    """UNDETERMINED stops the run too: an unasserted denominator is not reportable."""
    with pytest.raises(m6.DenominatorAssertionError):
        m6.assert_denominator(views=[], thresholds=thresholds())


def test_clean_denominator_passes():
    report = m6.assert_denominator(views=chiplets(FOUR_CHIPS),
                                   thresholds=thresholds(),
                                   gke_allocation=4,
                                   gke_allocation_unit="chips",
                                   tensor_parallel_size=8)
    assert report.outcome is common.Outcome.PASSED
    assert report.chip_count == 4


def test_tensor_parallel_ceiling_is_enforced():
    th = thresholds()
    ceiling = int(th.require("m6.max_tensor_parallel_size"))
    assert m6.check_tensor_parallel(ceiling * 2, ceiling).outcome is common.Outcome.FAILED
    assert m6.check_tensor_parallel(None, ceiling).outcome is common.Outcome.UNDETERMINED


# --------------------------------------------------------------------------
# P3, M6 half. See the note in test_m0_warm_cache.py: the M0 half is routed
# through the real producer. THIS HALF IS NOT, and the reason is worth stating
# rather than hiding. M6's producer is `m6.main()`, whose first act is
# `device_views_from_jax()`, which imports jax. There is no jax here and no
# package manager, so main() cannot be reached offline yet. P2 adds the offline
# path; the producer-routed leg for M6 lands there, and until it does the two
# tests below show only that the descriptor can carry the value -- NOT that the
# producer will ever emit it.
# --------------------------------------------------------------------------


def test_the_m6_control_descriptor_can_record_a_firing():
    injection = m6.Injection(inconsistent_pairing=True)
    control = injection.as_negative_control(executed=True)
    assert control is not None and control.executed is True


def test_the_m6_control_descriptor_defaults_to_not_executed():
    """The empty-case leg: True must not be unconditional."""
    injection = m6.Injection(inconsistent_pairing=True)
    control = injection.as_negative_control()
    assert control is not None and control.executed is False


def test_an_inactive_injection_has_no_control_even_when_told_it_executed():
    """`executed=True` must not conjure a control out of an uncorrupted run."""
    assert m6.Injection().as_negative_control(executed=True) is None


# --------------------------------------------------------------------------
# R5 -- LEG 1 MUST NOT WRITE ITS OWN VERDICT.
#
# Every test below fails against the pre-fix module, and the reasons differ,
# which is the point: one demonstrates that the old leg reported a failure with
# nothing corrupted, one that its verdict came from the injection rather than
# from the detector, and one that the blind-detector branch is reachable at all.
# --------------------------------------------------------------------------


def two_agreeing_routes(**kwargs):
    """A clean, PASSING reconciliation: both routes live and in agreement.

    The tensor-parallel size is read from the pinned ceiling rather than
    written here, so that this helper cannot drift into asserting a size the
    thresholds no longer permit.
    """
    th = thresholds()
    return dict(views=chiplets(FOUR_CHIPS),
                thresholds=th,
                gke_allocation=4,
                gke_allocation_unit="chips",
                tensor_parallel_size=int(th.require("m6.max_tensor_parallel_size")),
                **kwargs)


def test_leg_1_verdict_is_reached_by_the_reconciliation_not_by_the_injection():
    """The FAILED must be route disagreement, in the detector's own words.

    Pre-fix this reason read "negative control leg 1: the derived chip count
    was overridden by a scale factor of 2.0" -- the injection announcing its own
    result. The detector was never consulted and would not have been missed.
    """
    with pytest.raises(m6.DenominatorAssertionError):
        m6.assert_denominator(**two_agreeing_routes(
            injection=m6.Injection(chip_count_scale=2.0)))
    report = m6.assert_denominator(
        **two_agreeing_routes(injection=m6.Injection(chip_count_scale=2.0),
                              raise_on_failure=False))
    count_check = [c for c in report.checks if c.name == "m6.chip_count"][0]
    assert count_check.outcome is common.Outcome.FAILED
    assert "routes disagree" in count_check.reason
    assert "negative control" not in count_check.reason.lower()
    assert report.leg_1_control["response"] == m6.ControlResponse.DETECTED.value


def test_leg_1_that_corrupts_nothing_reports_nothing_detected():
    """A scale that does not move the value must not report a detection.

    THIS IS THE TEST THAT PINS THE OLD DEFECT MOST DIRECTLY. A scale of 1.0
    changes no input at all, yet the pre-fix leg still wrote FAILED and still
    stopped the run. An instrument that reports a caught corruption when there
    was no corruption is not a strict instrument, it is a broken one, and it is
    the same shape as an instrument that cannot report one.
    """
    report = m6.assert_denominator(**two_agreeing_routes(
        injection=m6.Injection(chip_count_scale=1.0)))
    assert report.outcome is common.Outcome.PASSED
    assert report.chip_count == 4
    assert report.leg_1_control["response"] == m6.ControlResponse.NOT_EXERCISED.value
    assert report.leg_1_control["routes_moved"] == []


def test_leg_1_rounding_back_to_the_same_count_is_also_not_exercised():
    """The near-miss of the case above: 1.1 x 4 rounds back to 4."""
    report = m6.assert_denominator(**two_agreeing_routes(
        injection=m6.Injection(chip_count_scale=1.1)))
    assert report.outcome is common.Outcome.PASSED
    assert report.leg_1_control["response"] == m6.ControlResponse.NOT_EXERCISED.value


def test_leg_1_corrupts_one_route_only_and_leaves_the_other_alone():
    """Corrupting both routes identically would be undetectable by design."""
    raw = [
        m6.chip_count_from_coords(chiplets(FOUR_CHIPS), DEVICES_PER_CHIP),
        m6.chip_count_from_gke(4, "chips", DEVICES_PER_CHIP),
    ]
    injected = m6.Injection(chip_count_scale=2.0).apply_to_evidence(raw)
    clean = {e.route: e.chip_count for e in raw}
    by_route = {e.route: e.chip_count for e in injected}
    # Written as a multiple of the clean value, never as a literal count: the
    # scale exists precisely so that no file here states a device count under a
    # chip count's name, and a test file is a file.
    assert by_route[m6.LEG_1_TARGET_ROUTE] == clean[m6.LEG_1_TARGET_ROUTE] * 2
    assert by_route["gke_allocation"] == clean["gke_allocation"]


def test_leg_1_does_not_invent_an_answer_for_a_route_that_abstained():
    """No value to scale is not a licence to supply one."""
    raw = [m6.chip_count_from_coords([], DEVICES_PER_CHIP)]
    injected = m6.Injection(chip_count_scale=2.0).apply_to_evidence(raw)
    assert injected[0].chip_count is None
    assert "negative_control_leg_1" not in injected[0].detail


def test_leg_1_on_a_single_route_is_undemonstrable_not_a_detection():
    """One live route: the clean run does not pass either, so nothing is shown.

    And the honest consequence is recorded rather than glossed -- on this
    topology the corrupted count is what the reconciliation carries forward.
    """
    report = m6.assert_denominator(views=chiplets(FOUR_CHIPS),
                                   thresholds=thresholds(),
                                   injection=m6.Injection(chip_count_scale=2.0),
                                   raise_on_failure=False)
    assert report.outcome is common.Outcome.UNDETERMINED
    assert report.leg_1_control["response"] == m6.ControlResponse.UNDEMONSTRABLE.value
    assert (report.leg_1_control["reconciliation_with_injection"]["chip_count"]
            == len(FOUR_CHIPS) * 2)


def test_no_leg_1_record_at_all_when_the_leg_is_not_active():
    """Absent must mean absent. A control record for an unrun control is a lie."""
    report = m6.assert_denominator(**two_agreeing_routes())
    assert report.leg_1_control is None
    assert report.payload()["negative_control_leg_1"] is None


def test_the_leg_1_adjudicator_reports_a_blind_detector():
    """The NOT_DETECTED branch, exercised directly.

    It is unreachable through `assert_denominator` today because two routes
    with no preference between them always disagree once one moves. That is a
    property of the routes, not of the adjudicator, and it stops holding the
    moment a third route or a tie-break is added. So the branch is driven here
    with a detector that passes both inputs -- a branch nobody has ever run is
    not a branch anybody should rely on.
    """
    clean_count = len(FOUR_CHIPS)
    dirty_count = clean_count * 2
    raw = [m6.ChipCountEvidence("jax_device_coords", clean_count, {})]
    injected = [m6.ChipCountEvidence("jax_device_coords", dirty_count, {})]
    passing = common.check("m6.chip_count", common.Outcome.PASSED, "a blind detector", {"routes": "hand-built"})
    record = m6.adjudicate_leg_1(scale=2.0,
                                 raw_evidence=raw,
                                 injected_evidence=injected,
                                 baseline=(clean_count, passing),
                                 corrupted=(dirty_count, passing))
    assert record["response"] == m6.ControlResponse.NOT_DETECTED.value
    assert "BLIND" in record["why"]


def test_the_leg_1_adjudicator_is_silent_when_the_leg_is_inactive():
    """The empty case of the adjudicator itself: no scale, no record."""
    raw = [m6.ChipCountEvidence("jax_device_coords", 4, {})]
    passing = common.check("m6.chip_count", common.Outcome.PASSED, "clean", {"routes": "hand-built"})
    assert m6.adjudicate_leg_1(scale=None,
                               raw_evidence=raw,
                               injected_evidence=raw,
                               baseline=(4, passing),
                               corrupted=(4, passing)) is None
