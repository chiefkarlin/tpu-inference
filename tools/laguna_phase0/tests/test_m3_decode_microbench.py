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
"""Unit tests for the M3 decode-only microbenchmark.

NOT EXECUTED. Written without run authorisation.
"""

import pytest

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m2_replication as m2
from tools.laguna_phase0 import m3_decode_microbench as m3
from tools.laguna_phase0 import basis as basis_mod

CONFIG = m3.MicrobenchConfig(batch_size=32, context_length=512, steps=5)


def thresholds():
    return common.Thresholds.load()


def timings(values=(18.0, 18.2, 17.9, 18.1, 18.0)):
    return m3.timings_from_recorded(values, "fixture sync")


def test_the_design_threshold_is_not_readable():
    """Review finding B8: the 1 ms threshold is overruled and held at null."""
    th = thresholds()
    assert th.entry("m3.design_stated_threshold_ms").value is None
    assert th.entry("m3.design_stated_threshold_ms").kind == "rejected"
    with pytest.raises(common.ThresholdError):
        th.require("m3.serving_stack_cost_threshold_ms")


def test_without_a_threshold_the_difference_is_published_and_undetermined():
    checks = m3.assess(timings(), CONFIG, served_tpot_ms=18.75, concurrency=32,
                       shape="B", thresholds=thresholds())
    cost = next(c for c in checks if c.name == "m3.serving_stack_cost")
    assert cost.outcome is common.Outcome.UNDETERMINED
    assert cost.intermediates["serving_stack_cost"]["difference_ms"] == pytest.approx(0.75)


def test_the_cross_basis_comparison_is_declared_rather_than_hidden():
    payload = m3.serving_stack_cost(18.0, 18.75, 32, "B")
    assert payload["basis"]["cross_basis_comparison"] is True
    assert payload["engine_step"]["basis"] == "engine_step_wall_clock"
    assert payload["served_tpot"]["basis"] == "tpot_derived"


def test_timing_without_a_device_sync_is_refused():
    with pytest.raises(m3.SynchronisationMissingError):
        m3.run_step_loop(step=lambda: None, sync=None, config=CONFIG)


def test_step_loop_times_each_step_and_publishes_them_raw():
    calls = []
    result = m3.run_step_loop(step=lambda: calls.append("step"),
                              sync=lambda x: x,
                              config=m3.MicrobenchConfig(batch_size=1,
                                                         context_length=8,
                                                         steps=3,
                                                         warmup_steps=1))
    assert len(calls) == 4
    assert len(result.step_ms) == 3
    assert result.to_dict()["units"] == "ms"


def test_uninitialised_kv_is_undetermined_with_the_bias_direction_stated():
    config = m3.MicrobenchConfig(batch_size=32, context_length=512, steps=5,
                                 kv_prefilled=False)
    checks = m3.assess(timings(), config, served_tpot_ms=18.75, concurrency=32,
                       shape="B", thresholds=thresholds())
    kv = next(c for c in checks if c.name == "m3.kv_prefilled")
    assert kv.outcome is common.Outcome.UNDETERMINED
    assert "UNKNOWN" in kv.reason


def test_http_or_prefill_inside_the_window_fails():
    config = m3.MicrobenchConfig(batch_size=32, context_length=512, steps=5,
                                 prefill_in_window=True)
    checks = m3.assess(timings(), config, served_tpot_ms=18.75, concurrency=32,
                       shape="B", thresholds=thresholds())
    window = next(c for c in checks if c.name == "m3.window_contained_no_http_or_prefill")
    assert window.outcome is common.Outcome.FAILED


def test_an_absent_served_tpot_is_undetermined_never_pass():
    checks = m3.assess(timings(), CONFIG, served_tpot_ms=None, concurrency=32,
                       shape="B", thresholds=thresholds())
    cost = next(c for c in checks if c.name == "m3.serving_stack_cost")
    assert cost.outcome is common.Outcome.UNDETERMINED


def test_a_threshold_finer_than_the_measured_spread_is_refused():
    """M2's rule, enforced at the point where it would be violated."""
    th = thresholds()
    th._data["m3"]["serving_stack_cost_threshold_ms"]["value"] = 0.1  # noqa: SLF001
    spread = m2.summarise_replicates(
        m2.Cell("B", 512, 256, 32), "tpot",
        [m2.Replicate(value=v, units="ms",
                      basis=basis_mod.MeasurementBasis.TPOT_DERIVED)
         for v in (18.0, 18.9, 17.6, 18.4)])
    checks = m3.assess(timings(), CONFIG, served_tpot_ms=18.75, concurrency=32,
                       shape="B", thresholds=th, spread=spread)
    cost = next(c for c in checks if c.name == "m3.serving_stack_cost")
    assert cost.outcome is common.Outcome.UNDETERMINED
    assert "finer than the measured spread" in cost.reason
