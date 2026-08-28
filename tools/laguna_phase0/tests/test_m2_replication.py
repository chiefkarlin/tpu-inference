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
"""Unit tests for M2 replication and for basis labelling.

NOT EXECUTED. Written without run authorisation.
"""

import pytest

from tools.laguna_phase0 import basis as basis_mod
from tools.laguna_phase0 import common
from tools.laguna_phase0 import m2_replication as m2

CELL = m2.Cell(shape="B", isl=512, osl=256, concurrency=32)
WALL = basis_mod.MeasurementBasis.WALL_CLOCK


def thresholds():
    return common.Thresholds.load()


def replicates(values, declared=WALL):
    return [m2.Replicate(value=v, units="s", basis=declared) for v in values]


def test_both_m2_cells_are_configured_and_include_c1():
    """Review finding B8: the replication treatment extends to c1."""
    concurrencies = {c.concurrency for c in m2.cells_from_thresholds(thresholds())}
    assert concurrencies == {1, 32}


def test_summary_always_ships_the_raw_values():
    estimate = m2.summarise_replicates(CELL, "wall_time", replicates([10.0, 10.5, 9.5, 10.2]))
    assert estimate.raw_values == [10.0, 10.5, 9.5, 10.2]
    assert estimate.coefficient_of_variation is not None
    assert estimate.n == 4


def test_too_few_replicates_is_undetermined_not_a_cov():
    estimate = m2.summarise_replicates(CELL, "wall_time", replicates([10.0, 10.4]))
    chk = m2.check_replication(estimate, thresholds())
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert chk.intermediates["raw_values"] == [10.0, 10.4]


def test_replicates_on_mixed_bases_are_refused():
    """Review finding B3, at the point where the mixing would happen."""
    mixed = replicates([10.0, 10.4]) + [
        m2.Replicate(value=10.1, units="s",
                     basis=basis_mod.MeasurementBasis.TPOT_DERIVED)
    ]
    with pytest.raises(basis_mod.BasisMismatchError):
        m2.summarise_replicates(CELL, "wall_time", mixed)


def test_an_unknown_basis_cannot_be_combined_with_anything():
    unknown = [m2.Replicate(value=1.0, units="s",
                            basis=basis_mod.MeasurementBasis.UNKNOWN)]
    with pytest.raises(basis_mod.BasisMismatchError):
        m2.summarise_replicates(CELL, "wall_time", unknown)


def point(value):
    return basis_mod.LadderPoint(concurrency=32, value=value, units="s", basis=WALL,
                                 instrument="M2", shape="B")


def test_a_difference_inside_the_spread_is_not_resolvable_rather_than_no_effect():
    estimate = m2.summarise_replicates(CELL, "wall_time",
                                       replicates([10.0, 10.5, 9.5, 10.2]))
    chk = m2.compare_with_spread("phase1", point(10.0), point(10.1), estimate)
    assert chk.outcome is common.Outcome.UNDETERMINED
    assert "NOT RESOLVABLE" in chk.reason
    assert chk.intermediates["spread"]["raw_values"]


def test_a_difference_outside_the_spread_is_reported_with_the_spread_beside_it():
    estimate = m2.summarise_replicates(CELL, "wall_time",
                                       replicates([10.0, 10.5, 9.5, 10.2]))
    chk = m2.compare_with_spread("phase1", point(10.0), point(7.0), estimate)
    assert chk.outcome is common.Outcome.PASSED
    assert chk.intermediates["resolution_floor"] == pytest.approx(estimate.stdev)


def test_a_comparison_without_a_spread_estimate_is_undetermined():
    chk = m2.compare_with_spread("phase1", point(10.0), point(7.0), None)
    assert chk.outcome is common.Outcome.UNDETERMINED


def test_the_plan_renders_runs_and_runs_nothing():
    plan = m2.plan_replicate_commands(CELL, 4, ["bench", "--shape=B"])
    assert len(plan) == 4
    assert plan[0]["argv"][-1] == "--max-concurrency=32"
    assert "VOID" in plan[0]["m0_requirement"]
