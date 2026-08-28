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
