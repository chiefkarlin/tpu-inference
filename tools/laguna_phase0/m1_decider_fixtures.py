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
"""E6b -- synthetic fixtures for the M1 decider, with predictions written first.

WHY THIS ONE IS RUN WHEN NOTHING ELSE IN THIS PACKAGE IS

The decider is a pure function of emitted quantities. It needs no TPU, no
server, no profiler and no measurement: hand it numbers, it returns a branch.
So the branch logic can be checked now, in-process, against numbers chosen to
land on each side of every cut and exactly ON the cuts -- and it should be,
because a decider whose MIXED branch has never once been exercised is a branch
that exists in the prose and not in the machine.

THE PREDICTIONS ARE PART OF THE SOURCE, AND THEY WERE COMMITTED BEFORE THE
SUITE WAS RUN. That ordering is the whole method. A prediction written after
seeing the output is not a prediction, and a fixture suite that is adjusted
until it agrees with the code tests nothing except the author's patience. The
commit that introduced this file contains the predictions and no results; the
results are a separate artifact written afterwards.

Three buckets, as everywhere else in this package:

  PASSED                              every stated prediction matched
  FAILED                              a stated prediction did not match
  COULD_NOT_BE_CHECKED_MECHANICALLY   the prediction was not expressible as a
                                      mechanical comparison

A fixture that raises is FAILED, not an error to be swallowed: the decider
crashing on a legal input is a result about the decider.

WHAT THE FIXTURES DO NOT ESTABLISH. They exercise the decider's arithmetic on
supplied numbers. They say nothing about whether the emitter's numbers are the
right numbers, whether the counters carry the units they claim, or whether a
real trace looks anything like these. Passing here is necessary and nowhere
near sufficient.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from tools.laguna_phase0 import common
from tools.laguna_phase0 import m1_profile as m1

TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Fixture construction helpers. Deliberately thin: a fixture should be
# readable as "these numbers went in", not as a builder DSL.
# --------------------------------------------------------------------------


def _q(value: float, name: str, units: str = "seconds",
       confirmed: bool = True) -> m1.Quantity:
    return m1.Quantity(
        value,
        common.CounterProvenance(
            name=name, units=units,
            source="SYNTHETIC FIXTURE -- this number was chosen by the author "
                   "of tools/laguna_phase0/m1_decider_fixtures.py to land on a "
                   "particular side of a cut. It did not come from a machine.",
            units_confirmed=confirmed))


def synthetic_step(*, index: int = 0, step_time: float = 1.0,
                   idle_strict: float = 0.0, idle_loose: Optional[float] = None,
                   buckets: Optional[Mapping[str, float]] = None,
                   overlap: Optional[float] = 0.0,
                   unmapped: float = 0.0,
                   run_id: str = "fixture-run") -> m1.DecodeStepProfile:
    return m1.DecodeStepProfile(
        step=index,
        run_id=run_id,
        step_time=_q(step_time, "step_time"),
        idle_strict=_q(idle_strict, "idle_strict"),
        idle_loose=_q(idle_strict if idle_loose is None else idle_loose,
                      "idle_loose"),
        buckets={k: _q(v, f"bucket.{k}") for k, v in (buckets or {}).items()},
        bucket_overlap=None if overlap is None else _q(overlap, "bucket_overlap"),
        unmapped_event_seconds=unmapped)


def _rule1(steps: Sequence[m1.DecodeStepProfile]) -> Callable[[common.Thresholds],
                                                              common.Check]:
    return lambda t: m1.decide_host_bound([m1.derive(s) for s in steps], t)


def _non_overlap(steps: Sequence[m1.DecodeStepProfile]):
    return lambda t: m1.check_non_overlap_term([m1.derive(s) for s in steps], t)


def _rule2(pair: Optional[m1.EfficiencyPair]):
    return lambda t: m1.decide_efficiency(pair, t)


def _closure(steps: Sequence[m1.DecodeStepProfile]):
    return lambda t: m1.check_closure([m1.derive(s) for s in steps], t)


def _overlap_leg(steps: Sequence[m1.DecodeStepProfile]):
    return lambda t: m1.check_bucket_overlap([m1.derive(s) for s in steps], t)


@dataclasses.dataclass(frozen=True)
class Prediction:
    """What the author expected, recorded before the suite was ever run."""

    outcome: Optional[common.Outcome] = None
    verdict: Optional[str] = None
    intermediates: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    reason_contains: Optional[str] = None
    absent_intermediates: Sequence[str] = ()
    note: str = ""

    @property
    def is_mechanical(self) -> bool:
        return bool(self.outcome or self.verdict or self.intermediates
                    or self.reason_contains or self.absent_intermediates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value if self.outcome else None,
            "verdict": self.verdict,
            "intermediates": dict(self.intermediates),
            "reason_contains": self.reason_contains,
            "absent_intermediates": list(self.absent_intermediates),
            "note": self.note,
        }


@dataclasses.dataclass(frozen=True)
class Fixture:
    name: str
    family: str
    why: str
    run: Callable[[common.Thresholds], common.Check]
    prediction: Prediction


# --------------------------------------------------------------------------
# FAMILY 1 -- Rule 1, every branch and both boundaries.
#
# The cuts are inclusive, so the boundary values are the ones that distinguish
# a correct implementation from one written with the wrong comparison operator,
# and MIXED is the branch most likely to have been described but not built.
# --------------------------------------------------------------------------

FAMILY_RULE1: List[Fixture] = [
    Fixture(
        name="rule1.holds.clear",
        family="rule1_f_host",
        why="f_host well above the upper cut",
        run=_rule1([synthetic_step(idle_strict=0.60, buckets={"grouped_matmul": 0.40})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="HOLDS",
                              intermediates={"aggregate_f_host": 0.60})),
    Fixture(
        name="rule1.holds.exactly_on_the_cut",
        family="rule1_f_host",
        why="0.45 exactly. The design says HOLDS at or above, so the boundary "
            "belongs to HOLDS and a strict > would put it in MIXED",
        run=_rule1([synthetic_step(idle_strict=0.45, buckets={"grouped_matmul": 0.55})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="HOLDS",
                              intermediates={"aggregate_f_host": 0.45})),
    Fixture(
        name="rule1.mixed.middle",
        family="rule1_f_host",
        why="0.30, squarely between the cuts. MIXED is a defined branch and "
            "must come back as a verdict, not as an abstention",
        run=_rule1([synthetic_step(idle_strict=0.30, buckets={"grouped_matmul": 0.70})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="MIXED",
                              intermediates={"aggregate_f_host": 0.30})),
    Fixture(
        name="rule1.mixed.just_inside_the_lower_cut",
        family="rule1_f_host",
        why="0.13: one hundredth above REFUTED. The narrowest MIXED case",
        run=_rule1([synthetic_step(idle_strict=0.13, buckets={"grouped_matmul": 0.87})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="MIXED")),
    Fixture(
        name="rule1.refuted.exactly_on_the_cut",
        family="rule1_f_host",
        why="0.12 exactly. REFUTED at or below, so the boundary belongs to "
            "REFUTED and a strict < would put it in MIXED",
        run=_rule1([synthetic_step(idle_strict=0.12, buckets={"grouped_matmul": 0.88})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="REFUTED",
                              intermediates={"aggregate_f_host": 0.12})),
    Fixture(
        name="rule1.refuted.clear",
        family="rule1_f_host",
        why="f_host well below the lower cut",
        run=_rule1([synthetic_step(idle_strict=0.02, buckets={"grouped_matmul": 0.98})]),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="REFUTED")),
    Fixture(
        name="rule1.empty_run",
        family="rule1_f_host",
        why="No steps at all. An empty input must be UNDETERMINED and must "
            "never read as a refutation of the host-bound hypothesis",
        run=_rule1([]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              verdict="UNDETERMINED")),
    Fixture(
        name="rule1.idle_exceeds_the_step",
        family="rule1_f_host",
        why="Idle longer than the step that contains it. Arithmetically "
            "impossible, so it is an instrument fault and not a HOLDS",
        run=_rule1([synthetic_step(idle_strict=1.50, buckets={"grouped_matmul": 0.10})]),
        prediction=Prediction(outcome=common.Outcome.FAILED,
                              verdict="UNDETERMINED")),
    Fixture(
        name="rule1.steps_disagree",
        family="rule1_f_host",
        why="One step HOLDS, one REFUTED. The aggregate would land in the "
            "middle band, and reporting that as MIXED would be a lie with a "
            "true-sounding name: MIXED means f_host fell between the cuts, not "
            "that the steps disagreed",
        run=_rule1([
            synthetic_step(index=0, idle_strict=0.60, buckets={"grouped_matmul": 0.40}),
            synthetic_step(index=1, idle_strict=0.02, buckets={"grouped_matmul": 0.98}),
        ]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              verdict="UNDETERMINED",
                              intermediates={"branches_present": ["HOLDS", "REFUTED"]})),
]


# --------------------------------------------------------------------------
# FAMILY 2 -- the non-overlap term, which must not be able to decide Rule 1.
# --------------------------------------------------------------------------

FAMILY_NON_OVERLAP: List[Fixture] = [
    Fixture(
        name="non_overlap.branch_moves_with_the_definition",
        family="non_overlap_term",
        why="Strict idle 0.05 (REFUTED) but loose idle 0.65 (HOLDS): the DMA "
            "mask alone decides the verdict. Exactly the condition the design "
            "says must not obtain, so the leg must fail rather than quietly "
            "report the strict answer",
        run=_non_overlap([synthetic_step(idle_strict=0.05, idle_loose=0.65,
                                         buckets={"grouped_matmul": 0.35,
                                                  "dma": 0.60})]),
        prediction=Prediction(outcome=common.Outcome.FAILED,
                              intermediates={"branch_strict": "REFUTED",
                                             "branch_loose": "HOLDS"})),
    Fixture(
        name="non_overlap.branch_is_stable",
        family="non_overlap_term",
        why="Both definitions land in REFUTED, so the DMA mask is not "
            "load-bearing and the distance from each cut is published",
        run=_non_overlap([synthetic_step(idle_strict=0.05, idle_loose=0.08,
                                         buckets={"grouped_matmul": 0.90,
                                                  "dma": 0.05})]),
        prediction=Prediction(outcome=common.Outcome.PASSED,
                              intermediates={"branch_strict": "REFUTED",
                                             "branch_loose": "REFUTED"})),
]


# --------------------------------------------------------------------------
# FAMILY 3 -- Rule 2, including the void path that outranks the ratio.
#
# The boundary fixtures use e_ref = 1.0 so that the ratio is exactly
# representable in binary and the fixture tests the comparison operator rather
# than the floating-point unit. One fixture deliberately does NOT do that; see
# rule2.byte_bound.boundary_in_awkward_arithmetic.
# --------------------------------------------------------------------------

FAMILY_RULE2: List[Fixture] = [
    Fixture(
        name="rule2.void.reference_below_the_floor",
        family="rule2_efficiency",
        why="e_ref = 0.40 is below the plausibility floor. The reference "
            "kernel did not reach a credible efficiency, so the cut is VOID "
            "and the ratio must not be computed at all -- note e_dec/e_ref "
            "here is 1.225, which would otherwise read as a confident "
            "BYTE_BOUND off an untrustworthy denominator",
        run=_rule2(m1.EfficiencyPair(e_dec=0.49, e_ref=0.40,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="VOID",
                              absent_intermediates=["ratio_e_dec_over_e_ref"])),
    Fixture(
        name="rule2.void.reference_exactly_on_the_floor",
        family="rule2_efficiency",
        why="e_ref = 0.50 exactly. The design voids BELOW 0.5, so the floor "
            "itself is not void and the ratio is read normally",
        run=_rule2(m1.EfficiencyPair(e_dec=0.50, e_ref=0.50,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="BYTE_BOUND",
                              intermediates={"ratio_e_dec_over_e_ref": 1.0})),
    Fixture(
        name="rule2.byte_bound.exactly_on_the_cut",
        family="rule2_efficiency",
        why="ratio exactly 0.85, chosen with e_ref = 1.0 so the quotient is "
            "exact. At or above is BYTE_BOUND",
        run=_rule2(m1.EfficiencyPair(e_dec=0.85, e_ref=1.0,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="BYTE_BOUND",
                              intermediates={"ratio_e_dec_over_e_ref": 0.85})),
    Fixture(
        name="rule2.byte_bound.boundary_in_awkward_arithmetic",
        family="rule2_efficiency",
        why="0.68 / 0.80, which is 0.85 in decimal but neither operand is "
            "representable in binary. I predict BYTE_BOUND from an error "
            "analysis rather than from running it: the numerator's "
            "representation error is about +7.2e-17 relative, the "
            "denominator's about +5.6e-17, so the exact quotient is about "
            "0.85 x (1 + 1.6e-17) and the nearest double to that is 0.85 "
            "itself, the spacing there being 1.1e-16. If this comes back "
            "PARTIAL the implementation is fine and the finding is that the "
            "cut is not robust to how the two efficiencies were rounded -- "
            "which is worth knowing before a candidate is killed by it",
        run=_rule2(m1.EfficiencyPair(e_dec=0.68, e_ref=0.80,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="BYTE_BOUND")),
    Fixture(
        name="rule2.deficit.exactly_on_the_cut",
        family="rule2_efficiency",
        why="ratio exactly 0.60. At or below is EFFICIENCY_DEFICIT",
        run=_rule2(m1.EfficiencyPair(e_dec=0.60, e_ref=1.0,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED,
                              verdict="EFFICIENCY_DEFICIT",
                              intermediates={"ratio_e_dec_over_e_ref": 0.60})),
    Fixture(
        name="rule2.partial.between_the_cuts",
        family="rule2_efficiency",
        why="ratio 0.70: neither byte-bound nor a deficit, and PARTIAL is a "
            "defined answer rather than a shrug",
        run=_rule2(m1.EfficiencyPair(e_dec=0.70, e_ref=1.0,
                                     run_id_dec="r", run_id_ref="r")),
        prediction=Prediction(outcome=common.Outcome.PASSED, verdict="PARTIAL")),
    Fixture(
        name="rule2.reference_from_another_run",
        family="rule2_efficiency",
        why="The design requires e_ref in the SAME profile run. A reference "
            "imported from elsewhere gives a perfectly computable ratio that "
            "means something else, so the leg must abstain rather than divide",
        run=_rule2(m1.EfficiencyPair(e_dec=0.70, e_ref=0.75,
                                     run_id_dec="r1", run_id_ref="r2")),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              verdict="UNDETERMINED",
                              absent_intermediates=["ratio_e_dec_over_e_ref"])),
    Fixture(
        name="rule2.no_pair_emitted",
        family="rule2_efficiency",
        why="Rule 2 was not measured. Absence is UNDETERMINED, never a pass",
        run=_rule2(None),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              verdict="UNDETERMINED")),
]


# --------------------------------------------------------------------------
# FAMILY 4 -- closure arithmetic (E6a). The second fixture is the one that
# matters: terms that sum to S while double-counting a region.
# --------------------------------------------------------------------------

FAMILY_CLOSURE: List[Fixture] = [
    Fixture(
        name="closure.terms_do_not_sum_to_the_step",
        family="closure",
        why="I + terms = 0.50 of a 1.0 step. Half the step is unattributed. "
            "The residual must be published as 0.5 and the leg must NOT pass, "
            "because no party has set a threshold for how much is too much",
        run=_closure([synthetic_step(idle_strict=0.20,
                                     buckets={"grouped_matmul": 0.20,
                                              "collectives": 0.10},
                                     overlap=0.0)]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              intermediates={"unattributed_fraction_by_step": {0: 0.50}})),
    Fixture(
        name="closure.overlapping_terms_that_sum_to_the_step",
        family="closure",
        why="THE FIXTURE THIS FAMILY EXISTS FOR. I = 0.20, terms 0.50 and "
            "0.30, total exactly 1.00 -- a residual of zero, which reads as "
            "perfect attribution. But the terms overlap by 0.25, so a quarter "
            "of the step is counted twice and another quarter is not counted "
            "at all. A residual of zero is not evidence of closure and the "
            "output must say so",
        run=_closure([synthetic_step(idle_strict=0.20,
                                     buckets={"grouped_matmul": 0.50,
                                              "collectives": 0.30},
                                     overlap=0.25)]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              intermediates={"unattributed_fraction_by_step": {0: 0.0},
                                             "bucket_overlap_fraction_by_step": {0: 0.25}},
                              reason_contains="double count")),
    Fixture(
        name="closure.terms_exceed_the_step",
        family="closure",
        why="Terms summing past the step they sit in. Zero is not a tuning "
            "constant, it is the edge of arithmetic possibility, so this is a "
            "FAILED and not a threshold being crossed",
        run=_closure([synthetic_step(idle_strict=0.20,
                                     buckets={"grouped_matmul": 0.50,
                                              "collectives": 0.60},
                                     overlap=0.0)]),
        prediction=Prediction(outcome=common.Outcome.FAILED,
                              intermediates={"negative_steps": {0: -0.30}})),
    Fixture(
        name="closure.no_threshold_is_ever_borrowed",
        family="closure",
        why="A tiny residual of 0.001 is still not a pass. The leg publishes "
            "and abstains, and the null threshold travels in the output so a "
            "reader can see that the absence was deliberate",
        run=_closure([synthetic_step(idle_strict=0.20,
                                     buckets={"grouped_matmul": 0.799},
                                     overlap=0.0)]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              intermediates={"threshold": {"key": "m1.closure_residual_threshold",
                                                           "value": None,
                                                           "kind": "deliberately-absent"}})),
    Fixture(
        name="closure.overlap_never_measured",
        family="closure",
        why="Without an overlap measurement the residual cannot be read as a "
            "closure measure at all, and the overlap leg must say which steps "
            "are affected rather than reporting a clean zero",
        run=_overlap_leg([synthetic_step(idle_strict=0.20,
                                         buckets={"grouped_matmul": 0.80},
                                         overlap=None)]),
        prediction=Prediction(outcome=common.Outcome.UNDETERMINED,
                              intermediates={"steps_without_an_overlap_measurement": [0]})),
]


ALL_FIXTURES: List[Fixture] = (FAMILY_RULE1 + FAMILY_NON_OVERLAP + FAMILY_RULE2
                               + FAMILY_CLOSURE)


# --------------------------------------------------------------------------
# Comparison. Predictions in, outcomes out, side by side.
# --------------------------------------------------------------------------


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual),
                                rel_tol=TOLERANCE, abs_tol=TOLERANCE)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            any(_matches(k, ak) and _matches(v, actual[ak]) for ak in actual)
            for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return False
        return all(_matches(e, a) for e, a in zip(expected, actual))
    return expected == actual


def evaluate(fixture: Fixture, thresholds: common.Thresholds) -> Dict[str, Any]:
    """Runs one fixture and puts its prediction next to its outcome."""
    record: Dict[str, Any] = {
        "fixture": fixture.name,
        "family": fixture.family,
        "why": fixture.why,
        "prediction": fixture.prediction.to_dict(),
    }
    if not fixture.prediction.is_mechanical:
        record["result"] = common.Outcome.UNDETERMINED.value
        record["mismatches"] = ["the prediction states nothing comparable"]
        return record

    try:
        check = fixture.run(thresholds)
    except Exception as exc:  # noqa: BLE001 -- a crash is a result about the decider
        record["result"] = common.Outcome.FAILED.value
        record["raised"] = f"{type(exc).__name__}: {exc}"
        record["mismatches"] = ["the decider raised on a legal input"]
        return record

    actual = check.to_dict()
    record["actual"] = actual
    mismatches: List[str] = []

    p = fixture.prediction
    if p.outcome is not None and check.outcome is not p.outcome:
        mismatches.append(
            f"outcome: predicted {p.outcome.value}, got {check.outcome.value}")
    if p.verdict is not None:
        got = check.intermediates.get("verdict")
        if got != p.verdict:
            mismatches.append(f"verdict: predicted {p.verdict}, got {got}")
    for key, expected in p.intermediates.items():
        if key not in check.intermediates:
            mismatches.append(f"intermediate {key}: absent")
        elif not _matches(expected, check.intermediates[key]):
            mismatches.append(
                f"intermediate {key}: predicted {expected!r}, got "
                f"{check.intermediates[key]!r}")
    for key in p.absent_intermediates:
        if key in check.intermediates:
            mismatches.append(
                f"intermediate {key}: predicted absent, but it is present "
                f"with value {check.intermediates[key]!r}")
    if p.reason_contains and p.reason_contains not in check.reason:
        mismatches.append(
            f"reason: predicted it would contain {p.reason_contains!r}")

    record["mismatches"] = mismatches
    record["result"] = (common.Outcome.PASSED.value if not mismatches
                        else common.Outcome.FAILED.value)
    return record


def run_all(thresholds: common.Thresholds) -> List[Dict[str, Any]]:
    return [evaluate(f, thresholds) for f in ALL_FIXTURES]


def tally(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {o.value: 0 for o in common.Outcome}
    for record in records:
        counts[record["result"]] += 1
    return counts


# --------------------------------------------------------------------------
# Proving the harness can fail.
#
# Twenty-four green lines are not evidence that the decider is right. They are
# equally consistent with a comparison harness that returns PASSED whatever it
# is handed, and that failure mode is invisible precisely when everything is
# passing. So before the green run is reported, the decider is deliberately
# broken in ways the fixtures are supposed to catch, and the fixtures are
# required to go red. A mutation that the suite does not notice is a hole in
# the suite, reported as such.
#
# The mutants are the mistakes most likely to be made for real: an inclusive
# cut written as a strict one, a threshold borrowed for a quantity nobody set
# one for, and a same-run requirement quietly dropped.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Mutation:
    name: str
    why: str
    attribute: str
    build: Callable[[Any], Any]
    fixtures_that_must_fail: Sequence[str]


def _mutant_strict_cuts(_original):
    def classify(f_host, holds_at, refuted_at):
        if f_host > holds_at:
            return m1.HostBoundVerdict.HOLDS
        if f_host < refuted_at:
            return m1.HostBoundVerdict.REFUTED
        return m1.HostBoundVerdict.MIXED

    return classify


def _mutant_closure_borrows_a_threshold(_original):
    def check_closure(derivations, thresholds):
        worst_residual = max((abs(d.unattributed_fraction) for d in derivations),
                             default=0.0)
        return common.check(
            "m1.closure_residual",
            common.Outcome.PASSED if worst_residual < 0.6 else common.Outcome.FAILED,
            "residual within a threshold this mutant invented",
            {"worst_residual": worst_residual})

    return check_closure


def _mutant_rule2_ignores_the_run_id(original):
    def decide_efficiency(pair, thresholds):
        if pair is not None:
            pair = dataclasses.replace(pair, run_id_ref=pair.run_id_dec)
        return original(pair, thresholds)

    return decide_efficiency


MUTATIONS: List[Mutation] = [
    Mutation(
        name="inclusive_cuts_written_as_strict",
        why="The single most likely real mistake in Rule 1. Both boundary "
            "fixtures must go red; if they do not, the boundaries are "
            "decorative",
        attribute="classify_f_host",
        build=_mutant_strict_cuts,
        fixtures_that_must_fail=["rule1.holds.exactly_on_the_cut",
                                 "rule1.refuted.exactly_on_the_cut"]),
    Mutation(
        name="closure_leg_invents_a_threshold",
        why="A closure leg that passes anything under a made-up bound is the "
            "exact failure E6a exists to prevent, and it would look like a "
            "clean instrument",
        attribute="check_closure",
        build=_mutant_closure_borrows_a_threshold,
        fixtures_that_must_fail=["closure.terms_do_not_sum_to_the_step",
                                 "closure.overlapping_terms_that_sum_to_the_step",
                                 "closure.terms_exceed_the_step",
                                 "closure.no_threshold_is_ever_borrowed"]),
    Mutation(
        name="rule2_drops_the_same_run_requirement",
        why="Dropping it yields a perfectly computable ratio that means "
            "something else, which is the kind of break that never announces "
            "itself",
        attribute="decide_efficiency",
        build=_mutant_rule2_ignores_the_run_id,
        fixtures_that_must_fail=["rule2.reference_from_another_run"]),
]


def prove_the_harness_can_fail(
        thresholds: common.Thresholds) -> List[Dict[str, Any]]:
    """Breaks the decider on purpose and requires the fixtures to notice."""
    by_name = {f.name: f for f in ALL_FIXTURES}
    records: List[Dict[str, Any]] = []
    for mutation in MUTATIONS:
        original = getattr(m1, mutation.attribute)
        setattr(m1, mutation.attribute, mutation.build(original))
        try:
            observed = {name: evaluate(by_name[name], thresholds)["result"]
                        for name in mutation.fixtures_that_must_fail}
        finally:
            setattr(m1, mutation.attribute, original)
        blind = sorted(n for n, r in observed.items()
                       if r != common.Outcome.FAILED.value)
        records.append({
            "mutation": mutation.name,
            "why": mutation.why,
            "mutated": f"m1_profile.{mutation.attribute}",
            "fixtures_that_must_fail": list(mutation.fixtures_that_must_fail),
            "observed": observed,
            "fixtures_that_did_not_notice": blind,
            "result": (common.Outcome.PASSED.value if not blind
                       else common.Outcome.FAILED.value),
        })
    return records


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None,
                        help="artifact path for the prediction/outcome table")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args(argv)

    thresholds = common.Thresholds.load(args.thresholds)
    mutations = prove_the_harness_can_fail(thresholds)
    records = run_all(thresholds)
    counts = tally(records)

    print("PROVING THE HARNESS CAN FAIL (decider deliberately broken):")
    for mutation in mutations:
        print(f"  {mutation['result']:35s} {mutation['mutation']}")
        for name in mutation["fixtures_that_did_not_notice"]:
            print(f"    NOT NOTICED BY {name}")
    print()

    width = max(len(r["fixture"]) for r in records)
    for record in records:
        print(f"{record['result']:35s} {record['fixture']:{width}s}")
        for mismatch in record["mismatches"]:
            print(f"    {mismatch}")
    print(f"\n{counts}")

    blind_mutations = [m["mutation"] for m in mutations
                       if m["result"] != common.Outcome.PASSED.value]
    mutation_check = common.check(
        "e6b.the_suite_can_fail",
        common.Outcome.PASSED if not blind_mutations else common.Outcome.FAILED,
        "the decider was broken on purpose in each of the ways these fixtures "
        "exist to catch, and the fixtures were required to go red",
        {"mutations": [m["mutation"] for m in mutations],
         "mutations_not_noticed": blind_mutations})

    if args.out:
        checks = [
            common.check(
                "e6b.fixture_suite",
                common.Outcome.PASSED if counts[common.Outcome.FAILED.value] == 0
                else common.Outcome.FAILED,
                "predictions were written and committed before the suite was "
                "run; every one is recorded here next to its outcome",
                {"tally": counts, "fixtures": len(records)}),
        ]
        common.write_artifact(
            args.out,
            kind="e6b.decider_fixtures",
            payload={"records": records, "tally": counts,
                     "mutations": mutations,
                     "reading_note":
                         "The fixture tally is only meaningful alongside the "
                         "mutation records. A suite that cannot go red has "
                         "not checked anything, and a green tally is what "
                         "that looks like from outside."},
            checks=checks + [mutation_check],
            thresholds=thresholds)

    blind = [m for m in mutations if m["result"] != common.Outcome.PASSED.value]
    if counts[common.Outcome.FAILED.value] or blind:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
