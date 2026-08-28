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

EXECUTION STATUS. These are pure in-process tests over synthetic inputs. They
touch no hardware, start no server and run no benchmark, which is the same class
as the E6b decider fixtures the engineering manager authorised. They WERE run;
the result is recorded in the dev notes. Nothing here constitutes a measurement.

THE ANCHOR TESTS ARE GONE, NOT DISABLED. The previous version of this file
passed 119.0 as an anchor argument in four assertions and asserted region names
from a three-way partition around it. That partition is retired: a test that
pins a retired anchor is the anchor, still live, wearing a different hat. What
replaces them is one test asserting the anchor CANNOT BE REACHED -- see
``test_the_retired_anchors_are_traps_and_not_merely_absent``.
"""

import pytest

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m4_router_histogram as m4


def thresholds():
    return common.Thresholds.load()


RUNTIME_TOPK = m4.TopKReading(
    value=10,
    source=m4.TopKSource.RUNNING_MODEL_EFFECTIVE_CONFIG,
    detail="model.config.num_experts_per_tok on the served model object")


def capture(concurrency, per_layer_counts, padding=True, steps=2,
            basis=m4.ExpertCountBasis.ROUTED_ROWS, top_k=RUNTIME_TOPK,
            window_steps=1):
    observations = []
    for step in range(steps):
        for layer, count in enumerate(per_layer_counts):
            observations.append(
                m4.RouterObservation(step=step, layer=layer,
                                     distinct_experts=count,
                                     window_steps=window_steps))
    return m4.Capture(concurrency=concurrency, shape="B",
                      observations=observations, count_basis=basis,
                      top_k=top_k, padding_rows_included=padding)


# -- counting ----------------------------------------------------------------


def test_distinct_experts_are_counted_from_top_k_rows():
    observations = m4.observations_from_router_indices(
        step=0, indices_by_layer={0: [[3, 7], [7, 3]], 1: [[1, 2], [3, 4]]})
    assert [o.distinct_experts for o in observations] == [2, 4]
    assert observations[0].rows == 2
    assert observations[0].window_steps == 1


def test_the_per_layer_histogram_is_emitted_not_just_the_summary():
    cap = capture(32, [], steps=0)
    cap.observations = [
        m4.RouterObservation(step=0, layer=0, distinct_experts=180),
        m4.RouterObservation(step=1, layer=0, distinct_experts=186),
    ]
    assert cap.per_layer_histogram() == {0: {180: 1, 186: 1}}
    assert cap.per_layer_e() == {0: 183.0}
    assert cap.summary_e() == 183.0


def test_every_ladder_point_carries_its_basis_and_its_count_declaration():
    point = capture(16, [100, 110]).ladder_point().to_dict()
    assert point["basis"] == "router_count"
    assert point["units"].startswith("distinct expert ids")
    assert point["detail"]["counter"]["units_confirmed"] is True
    declaration = point["detail"]["count_declaration"]
    assert declaration["counting_basis"] == "ROUTED_GMM_OR_MEGABLOX_ROWS"
    assert declaration["window_steps"] == 1
    assert declaration["cut"] == 10


# -- the count declaration: 11 is two different systems ----------------------


def test_the_cut_moves_by_one_with_the_instrumentation_point_not_the_model():
    routed = capture(1, [10], basis=m4.ExpertCountBasis.ROUTED_ROWS)
    weights = capture(1, [10], basis=m4.ExpertCountBasis.WEIGHTS_READ)
    assert routed.declaration().cut == 10
    assert weights.declaration().cut == 11


def test_eleven_is_a_clean_system_or_a_dispersing_one_and_the_basis_decides():
    """IDENTICAL OUTPUT, OPPOSITE VERDICTS. The whole reason for the delta."""
    counting_shared = m4.check_padding_dispersal(
        capture(1, [11, 11], basis=m4.ExpertCountBasis.WEIGHTS_READ),
        thresholds(), cut_established=True)
    not_counting_shared = m4.check_padding_dispersal(
        capture(1, [11, 11], basis=m4.ExpertCountBasis.ROUTED_ROWS),
        thresholds(), cut_established=True)
    assert counting_shared.intermediates["verdict"] == "NON_DISPERSAL_CONFIRMED"
    assert not_counting_shared.intermediates["verdict"] == "DISPERSAL_CONFIRMED"


def test_an_undeclared_basis_cannot_form_a_cut_and_does_not_default_to_zero():
    cap = capture(1, [11], basis=m4.ExpertCountBasis.NOT_DECLARED)
    assert cap.declaration().cut is None
    chk = m4.check_count_basis(cap, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "assuming 0" in chk.reason


def test_no_count_is_emitted_without_its_declaration():
    payload = capture(1, [10]).to_dict()
    assert payload["count_declaration"]["cut"] == 10
    assert payload["summary_e"]["provenance"]["units_confirmed"] is True
    assert all(o["window_steps"] == 1 for o in payload["observations"])


# -- top-k -------------------------------------------------------------------


def test_top_k_from_the_running_model_matching_ten_passes():
    assert m4.check_top_k(RUNTIME_TOPK, thresholds()).outcome is common.Outcome.PASSED


def test_top_k_from_a_config_file_cannot_establish_what_the_model_used():
    reading = m4.TopKReading(value=10, source=m4.TopKSource.CONFIG_FILE)
    chk = m4.check_top_k(reading, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "RUNNING model" in chk.reason


def test_a_hardcoded_top_k_is_a_defect_even_when_the_constant_is_right():
    reading = m4.TopKReading(value=10, source=m4.TopKSource.HARDCODED)
    chk = m4.check_top_k(reading, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert "stops being" in chk.reason


def test_the_serving_stacks_own_fallback_is_named_and_its_consequence_stated():
    reading = m4.TopKReading(value=8,
                             source=m4.TopKSource.RUNNING_MODEL_EFFECTIVE_CONFIG)
    chk = m4.check_top_k(reading, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert "vLLM config-class default" in chk.reason
    assert "INSTRUMENT FAULT" in chk.reason


def test_the_framework_default_fails_in_the_opposite_direction():
    reading = m4.TopKReading(value=16,
                             source=m4.TopKSource.RUNNING_MODEL_EFFECTIVE_CONFIG)
    chk = m4.check_top_k(reading, thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert "HF default" in chk.reason
    assert "DISPERSAL CONFIRMED" in chk.reason


def test_an_unread_top_k_is_undetermined_and_never_falls_back():
    chk = m4.check_top_k(m4.TopKReading(value=None), thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "falling back" in chk.reason


# -- window length -----------------------------------------------------------


def test_a_per_step_window_passes_and_is_recorded_on_every_observation():
    chk = m4.check_window_length(capture(1, [10, 10]), thresholds())
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["window_steps_seen"] == [1]


def test_a_pooled_count_is_a_different_quantity_and_is_refused():
    chk = m4.check_window_length(capture(1, [40, 40], window_steps=5),
                                 thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["window_steps_seen"] == [5]
    assert "different quantity" in chk.reason


def test_a_pooled_capture_cannot_reach_a_dispersal_verdict():
    """The pooled count crosses the cut with certainty; that is not dispersal."""
    cap = capture(1, [40, 40], window_steps=5)
    established = all(
        c.outcome is common.Outcome.PASSED
        for c in (m4.check_top_k(cap.top_k, thresholds()),
                  m4.check_count_basis(cap, thresholds()),
                  m4.check_window_length(cap, thresholds())))
    assert established is False
    chk = m4.check_padding_dispersal(cap, thresholds(), cut_established=established)
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["verdict"] == "NOT_CLASSIFIED"


# -- the structural dispersal rule -------------------------------------------


def test_every_observation_at_the_cut_is_non_dispersal_confirmed():
    chk = m4.check_padding_dispersal(capture(1, [10, 10]), thresholds(),
                                     cut_established=True)
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["verdict"] == "NON_DISPERSAL_CONFIRMED"


def test_one_observation_above_the_cut_confirms_dispersal():
    chk = m4.check_padding_dispersal(capture(1, [10, 11]), thresholds(),
                                     cut_established=True)
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["verdict"] == "DISPERSAL_CONFIRMED"
    assert chk.intermediates["evidence"]["above_cut_count"] == 2


def test_the_rule_runs_per_step_and_not_on_the_mean():
    """A mean sitting exactly on the cut hides steps above it.

    Layer 0 reads 9 then 11 across two steps: the per-layer mean is exactly 10,
    the cut, and a mean-based rule would report NON-DISPERSAL. The per-step rule
    sees a value below the cut and refuses to classify at all.
    """
    cap = capture(1, [], steps=0)
    cap.observations = [
        m4.RouterObservation(step=0, layer=0, distinct_experts=9),
        m4.RouterObservation(step=1, layer=0, distinct_experts=11),
    ]
    assert cap.per_layer_e() == {0: 10.0}
    chk = m4.check_padding_dispersal(cap, thresholds(), cut_established=True)
    assert chk.intermediates["verdict"] == "INSTRUMENT_FAULT_OR_CAPACITY_DROPPING"


def test_below_the_cut_is_a_defect_and_is_not_filed_as_could_not_be_checked():
    chk = m4.check_padding_dispersal(capture(1, [8, 8]), thresholds(),
                                     cut_established=True)
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["verdict"] == "INSTRUMENT_FAULT_OR_CAPACITY_DROPPING"
    assert "it was checked" in chk.reason


def test_an_unestablished_cut_is_not_a_weak_verdict_it_is_no_verdict():
    chk = m4.check_padding_dispersal(capture(1, [10, 10]), thresholds(),
                                     cut_established=False)
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["verdict"] == "NOT_CLASSIFIED"


def test_a_capture_that_does_not_record_padding_rows_cannot_answer_the_question():
    chk = m4.check_padding_dispersal(capture(1, [10], padding=None),
                                     thresholds(), cut_established=True)
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "padding rows" in chk.reason


def test_no_distance_to_any_retired_anchor_is_published():
    chk = m4.check_padding_dispersal(capture(1, [60, 60]), thresholds(),
                                     cut_established=True)
    blob = repr(chk.intermediates)
    assert "distance_from_disperse_reference" not in blob
    assert "gap_between_the_two_hypotheses" not in blob
    assert "RETIRED FROM THE DECISION PATH" in chk.intermediates["retired_anchors"]


# -- the retirement, enforced by the code and not only by prose --------------


def test_the_retired_anchors_are_traps_and_not_merely_absent():
    """Reaching for a retired anchor must die loudly, not return a float.

    A hole is what a future reader fills with the better number. OPT-1860: the
    corrected value is MORE attractive to reinstate than the wrong one was.
    """
    thr = thresholds()
    for key in ("m4.e1_disperse_reference", "m4.e1_no_disperse_band",
                "m4.e1_disperse_tolerance_fraction"):
        entry = thr.entry(key)
        assert entry.value is None
        assert entry.kind == "withdrawn"
        assert "DO NOT RE-POINT THIS AT 120.68" in entry.source
        with pytest.raises(common.ThresholdError):
            thr.require(key)


def test_the_uniform_models_are_discriminated_by_execution_not_by_assertion():
    """Model B cannot reproduce the one value that is certain by construction."""
    result = m4.discriminate_uniform_models(top_k=10, num_experts=256)
    assert result["model_a_reproduces_the_certain_value"] is True
    assert result["model_b_reproduces_the_certain_value"] is False
    assert round(result["model_b_at_one_row"], 3) == 9.826


def test_the_illustrative_expectation_is_model_a_and_says_it_is_a_placeholder():
    out = m4.uniform_distinct_expected(rows=16, top_k=10, num_experts=256)
    assert out["model"] == "MODEL_A_WITHOUT_REPLACEMENT"
    assert round(out["value"], 2) == 120.68
    assert out["status"].startswith("PLACEHOLDER")


# -- the byte-model leg ------------------------------------------------------


def test_e32_within_ten_percent_of_the_reference_passes():
    chk = m4.check_byte_model(capture(32, [183, 183]), thresholds())
    assert chk.outcome is common.Outcome.PASSED


def test_e32_outside_the_tolerance_fails_and_publishes_the_distance():
    chk = m4.check_byte_model(capture(32, [120, 120]), thresholds())
    assert chk.outcome is common.Outcome.FAILED
    assert chk.intermediates["relative_difference"] > 0.10


def test_the_byte_model_reference_carries_its_provenance_flag_and_is_not_moved():
    chk = m4.check_byte_model(capture(32, [183, 183]), thresholds())
    assert chk.intermediates["reference"] == 183
    assert "MODEL B" in chk.intermediates["provenance_flag_on_the_reference"]
    illustrative = chk.intermediates["uniform_expectation_illustrative_only"]
    assert illustrative["model"] == "MODEL_A_WITHOUT_REPLACEMENT"
    assert round(illustrative["value"], 2) == 184.47


def test_an_empty_capture_is_undetermined_never_pass():
    empty = capture(32, [], steps=0)
    assert m4.check_byte_model(empty, thresholds()).outcome is common.Outcome.UNDETERMINED


# -- the negative control ----------------------------------------------------


def test_the_control_case_has_the_distinct_count_it_claims():
    case = m4.default_control_cases(top_k=10)[0]
    rows = case.indices()
    assert len(rows) == 16
    assert len({e for row in rows for e in row}) == case.known_distinct


def test_the_control_runs_at_both_c1_and_c16():
    chk = m4.check_counter_controls(top_k=10, cut=10)
    assert chk.intermediates["concurrencies_controlled"] == [1, 16]
    assert chk.outcome is common.Outcome.PASSED


def test_c1_has_fifteen_padding_rows_and_c16_has_none():
    """Dispersal is a c1 phenomenon by construction; the block is 16 rows."""
    c1, c16 = m4.default_control_cases(top_k=10)
    assert (c1.real_rows, c1.padding_rows) == (1, 15)
    assert (c16.real_rows, c16.padding_rows) == (16, 0)


def test_a_counter_stuck_at_the_cut_is_detected_at_both_points():
    """The comfortable failure mode. The control has to reject it or it is décor."""
    for case in m4.default_control_cases(top_k=10):
        leg = m4.run_counter_control(case, cut=10,
                                     counter=m4.stuck_at_cut_counter)
        assert leg["passed"] is False
        assert leg["reported_the_cut_instead"] is True


# -- assembly ----------------------------------------------------------------


def test_a_missing_rung_makes_the_ladder_undetermined():
    checks = m4.assess([capture(32, [183])], thresholds())
    rung = next(c for c in checks if c.name == "m4.ladder_complete")
    assert rung.outcome is common.Outcome.UNDETERMINED
    assert rung.intermediates["missing"] == [1, 16]


def test_the_ladder_exposes_c1_and_c16_for_a_statistic_it_does_not_compute():
    points = m4.ladder([capture(1, [10]), capture(16, [90]), capture(32, [183])])
    assert [p["concurrency"] for p in points] == [1, 16, 32]
    assert all(p["basis"] == "router_count" for p in points)


def test_a_capture_round_trips_through_its_dict_form():
    original = capture(1, [10, 11])
    restored = m4.capture_from_dict({
        "concurrency": 1,
        "shape": "B",
        "count_basis": original.count_basis.value,
        "top_k": original.top_k.to_dict(),
        "padding_rows_included": True,
        "observations": [o.to_dict() for o in original.observations],
    })
    assert restored.declaration().cut == original.declaration().cut
    assert restored.per_layer_e() == original.per_layer_e()
