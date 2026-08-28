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
"""Unit tests for the M4 router-output histogram.

NOT EXECUTED. Written without run authorisation.
"""

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m4_router_histogram as m4


def thresholds():
    return common.Thresholds.load()


def capture(concurrency, per_layer_counts, padding=True, steps=2):
    observations = []
    for step in range(steps):
        for layer, count in enumerate(per_layer_counts):
            observations.append(
                m4.RouterObservation(step=step, layer=layer, distinct_experts=count))
    return m4.Capture(concurrency=concurrency, shape="B",
                      observations=observations, padding_rows_included=padding)


def test_distinct_experts_are_counted_from_top_k_rows():
    observations = m4.observations_from_router_indices(
        step=0, indices_by_layer={0: [[3, 7], [7, 3]], 1: [[1, 2], [3, 4]]})
    assert [o.distinct_experts for o in observations] == [2, 4]
    assert observations[0].rows == 2


def test_the_per_layer_histogram_is_emitted_not_just_the_summary():
    cap = m4.Capture(concurrency=32, shape="B", padding_rows_included=True,
                     observations=[
                         m4.RouterObservation(step=0, layer=0, distinct_experts=180),
                         m4.RouterObservation(step=1, layer=0, distinct_experts=186),
                     ])
    assert cap.per_layer_histogram() == {0: {180: 1, 186: 1}}
    assert cap.per_layer_e() == {0: 183.0}
    assert cap.summary_e() == 183.0


def test_every_ladder_point_carries_its_basis_and_counter_provenance():
    point = capture(16, [100, 110]).ladder_point().to_dict()
    assert point["basis"] == "router_count"
    assert point["units"].startswith("distinct expert ids")
    assert point["detail"]["counter"]["units_confirmed"] is True


def test_e32_within_ten_percent_of_the_reference_passes():
    chk = m4.check_byte_model(capture(32, [183, 183]), thresholds())
    assert chk.outcome is common.Outcome.PASSED


def test_e32_outside_the_tolerance_fails_and_publishes_the_distance():
    chk = m4.check_byte_model(capture(32, [120, 120]), thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["relative_difference"] > 0.10


def test_an_empty_capture_is_undetermined_never_pass():
    empty = m4.Capture(concurrency=32, shape="B", observations=[],
                       padding_rows_included=True)
    assert m4.check_byte_model(empty, thresholds()).outcome is common.Outcome.UNDETERMINED


def test_e1_inside_the_stated_band_says_padding_rows_do_not_disperse():
    chk = m4.check_padding_dispersal(capture(1, [12, 14]), thresholds())
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["region"] == "NON_DISPERSAL_BAND"
    assert "do NOT disperse" in chk.reason


def test_e1_near_the_disperse_reference_is_pending_a_tolerance_not_unmeasurable():
    """'Near 119' has no tolerance from any named party, so the leg abstains.

    But it abstains under its own label. Pending a ruling is a different state
    from not being able to tell, and the region name says which one this is.
    """
    chk = m4.check_padding_dispersal(capture(1, [119, 119]), thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["region"] == "DISPERSAL_NEIGHBOURHOOD_PENDING_TOLERANCE"
    assert chk.intermediates["distance_from_disperse_reference"] == 0.0


def test_e1_in_neither_neighbourhood_is_its_own_finding_and_not_an_abstention():
    """E(1) = 60 means neither prediction in the design holds.

    Under the old two-way shape this returned the same label as 119.4, which
    filed 'the design is wrong' under a word meaning 'we could not tell'.
    """
    chk = m4.check_padding_dispersal(capture(1, [60, 60]), thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["region"] == "NEITHER_NEIGHBOURHOOD"
    assert "finding about the design" in chk.reason


def test_a_value_further_from_119_than_the_rival_hypothesis_is_also_neither():
    """Any tolerance admitting E(1) = 400 would swallow the band whole."""
    assert m4.classify_dispersal(400.0, [10.0, 20.0], 119.0, None) is (
        m4.DispersalRegion.NEITHER_NEIGHBOURHOOD)


def test_a_value_below_the_band_is_neither_rather_than_pending():
    assert m4.classify_dispersal(5.0, [10.0, 20.0], 119.0, None) is (
        m4.DispersalRegion.NEITHER_NEIGHBOURHOOD)


def test_the_partition_is_conservative_in_the_strip_between_the_hypotheses():
    """Documenting where it declines to convict rather than hiding it.

    E(1) = 70 is nearer 119 than the band and within the gap, so it comes back
    PENDING rather than NEITHER. That is deliberate: only a tolerance would
    settle it, and inventing one is the thing being avoided.
    """
    assert m4.classify_dispersal(70.0, [10.0, 20.0], 119.0, None) is (
        m4.DispersalRegion.DISPERSAL_PENDING_TOLERANCE)


def test_a_ruled_tolerance_resolves_the_pending_region_without_touching_the_rest():
    assert m4.classify_dispersal(119.4, [10.0, 20.0], 119.0, 0.05) is (
        m4.DispersalRegion.DISPERSAL_NEIGHBOURHOOD)
    assert m4.classify_dispersal(60.0, [10.0, 20.0], 119.0, 0.05) is (
        m4.DispersalRegion.NEITHER_NEIGHBOURHOOD)


def test_both_distances_are_always_published():
    """Distance to 119 alone cannot separate 'pending' from 'neither'."""
    chk = m4.check_padding_dispersal(capture(1, [60, 60]), thresholds())
    assert chk.intermediates["distance_from_disperse_reference"] == -59.0
    assert chk.intermediates["distance_from_no_disperse_band"] == 40.0
    assert chk.intermediates["gap_between_the_two_hypotheses"] == 99.0


def test_the_leg_declares_its_own_bias_in_the_output():
    chk = m4.check_padding_dispersal(capture(1, [12, 14]), thresholds())
    assert "confirm only NON-dispersal" in chk.intermediates["known_bias"]


def test_a_capture_that_does_not_record_padding_rows_cannot_answer_the_question():
    chk = m4.check_padding_dispersal(capture(1, [119, 119], padding=None), thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["region"] == "NOT_ESTABLISHED"
    assert "padding rows" in chk.reason


def test_a_missing_rung_makes_the_ladder_undetermined():
    checks = m4.assess([capture(32, [183])], thresholds())
    rung = next(c for c in checks if c.name == "m4.ladder_complete")
    assert rung.outcome is common.Outcome.UNDETERMINED
    assert rung.intermediates["missing"] == [1, 16]


def test_the_ladder_exposes_c1_and_c16_for_a_statistic_it_does_not_compute():
    points = m4.ladder([capture(1, [12]), capture(16, [90]), capture(32, [183])])
    assert [p["concurrency"] for p in points] == [1, 16, 32]
    assert all(p["basis"] == "router_count" for p in points)
