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
"""M1 -- device profile emitter and the two-rule decider (E6, E6a).

WHAT THIS IS FOR

The campaign has two live explanations for the decode step time and they make
opposite predictions about where the time goes. M1 is the instrument that
separates them, and the design turns it into two rules:

  Rule 1 (host-bound):  f_host = I / S, where I is device-idle time inside a
      decode step and S is the step's wall time. At or above 0.45 the
      host-bound reading HOLDS; at or below 0.12 it is REFUTED; strictly
      between the two the answer is MIXED -- which is a defined branch of the
      rule and not a failure of it.

  Rule 2 (efficiency):  e_dec against e_ref, both measured IN THE SAME PROFILE
      RUN. At or above 0.85 x e_ref the step is BYTE-BOUND; at or below
      0.60 x e_ref there is an EFFICIENCY DEFICIT; between them, PARTIAL. If
      e_ref itself comes out below 0.5 the reference is not credible and the
      whole cut is VOID.

TWO AXES, DELIBERATELY NOT COLLAPSED

Every leg here reports a ``common.Outcome`` AND, where a rule was evaluated, a
rule verdict in its intermediates. They answer different questions:

  outcome  -- could this be decided mechanically from what was emitted?
              PASSED means a defined disposition was reached. FAILED means the
              inputs contradict themselves. COULD_NOT_BE_CHECKED_MECHANICALLY
              means the inputs were not enough.
  verdict  -- WHICH defined disposition. HOLDS / MIXED / REFUTED, or
              BYTE_BOUND / PARTIAL / EFFICIENCY_DEFICIT / VOID.

So a PASSED check can carry a REFUTED verdict. Folding the two would make
"the rule refuted the hypothesis" indistinguishable from "the rule could not be
run", and those two have to be told apart by anyone reading the result.

WHAT IS EMITTED AND NEVER FOLDED (E6a)

  the non-overlap term   I_loose - I_strict: time with a DMA in flight and no
                         compute running. Reported as its OWN channel. The
                         design asserts it is nowhere near the 0.45/0.12 cut;
                         this module does not assume that, it tests whether
                         swapping the idle definition changes the branch.
  the closure residual   S - (I + every named term). Emitted as a fraction of
                         S, per step, first-class. There is NO threshold for
                         it and none is invented here.
  the bucket overlap     wall time during which two or more named terms were
                         simultaneously active. It is what makes a residual of
                         zero uninformative, so it is emitted next to it.

NOTHING IN THIS MODULE RUNS A MODEL, A SERVER OR A PROFILER. It ingests an
already-captured trace and decides. That is what makes the decider a pure
function of emitted quantities, and therefore testable against synthetic
fixtures without a TPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tools.laguna_phase0 import common

# --------------------------------------------------------------------------
# Counter provenance.
#
# Units travel with values. Where this module computes a quantity itself out of
# interval arithmetic, the units are established by construction and are marked
# confirmed. Where a number is handed to us by a profiler, the CALLER declares
# the units and whether it could confirm them -- see TraceUnits. We never infer
# a unit from whether the resulting number looks sensible.
# --------------------------------------------------------------------------

DERIVED_SECONDS = "computed by tools/laguna_phase0/m1_profile.py from trace intervals already converted to seconds by the caller-declared TraceUnits"

COUNTER_STEP_TIME = common.CounterProvenance(
    name="step_time",
    units="seconds",
    source=DERIVED_SECONDS + "; step boundary supplied by the caller",
    units_confirmed=True)

COUNTER_IDLE_STRICT = common.CounterProvenance(
    name="idle_strict",
    units="seconds",
    source=DERIVED_SECONDS +
    "; S minus the union of ALL device-busy intervals, compute and DMA alike",
    units_confirmed=True,
    note="The strict definition. This is the I in Rule 1.")

COUNTER_IDLE_LOOSE = common.CounterProvenance(
    name="idle_loose",
    units="seconds",
    source=DERIVED_SECONDS +
    "; S minus the union of COMPUTE intervals only, ignoring DMA in flight",
    units_confirmed=True,
    note="Never substituted into Rule 1. Emitted so the non-overlap term can be.")

COUNTER_BUCKET = common.CounterProvenance(
    name="bucket_time",
    units="seconds",
    source=DERIVED_SECONDS + "; union of the intervals mapped to this bucket",
    units_confirmed=True)

COUNTER_BUCKET_OVERLAP = common.CounterProvenance(
    name="bucket_overlap",
    units="seconds",
    source=DERIVED_SECONDS +
    "; sum of per-bucket union durations minus the union across all buckets",
    units_confirmed=True,
    note="Wall time attributed to more than one named term at once.")

COUNTER_HBM_BYTES = common.CounterProvenance(
    name="hbm_bytes",
    units="bytes",
    source="NOT ESTABLISHED BY THIS MODULE -- supplied by the caller from the "
    "profiler's memory-bandwidth counter, with its own units declaration",
    units_confirmed=False,
    note="Left unconfirmed on purpose. A byte/kibibyte/element confusion here "
    "does not look implausible, it looks like a different verdict. The caller "
    "must confirm the unit against the profiler's own documentation and pass "
    "units_confirmed=True; until then the emitted unit reads UNKNOWN.")


@dataclasses.dataclass(frozen=True)
class Quantity:
    """One number that refuses to travel without its units and its source."""

    value: float
    provenance: common.CounterProvenance

    def to_dict(self) -> Dict[str, Any]:
        return common.counted(self.value, self.provenance)


def quantity_from_dict(payload: Mapping[str, Any]) -> Quantity:
    prov = payload["provenance"]
    return Quantity(value=float(payload["value"]),
                    provenance=common.CounterProvenance(
                        name=prov["name"],
                        units=prov["units"],
                        source=prov["source"],
                        units_confirmed=bool(prov.get("units_confirmed", False)),
                        note=prov.get("note", "")))


# --------------------------------------------------------------------------
# Interval arithmetic. The emitter's whole job is to turn a trace into these.
# --------------------------------------------------------------------------


class BucketKind(str, enum.Enum):
    """What a named term occupies the device with."""

    COMPUTE = "compute"
    DMA = "dma"


@dataclasses.dataclass(frozen=True)
class Interval:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"interval ends before it starts: {self}")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def clipped(self, lo: float, hi: float) -> Optional["Interval"]:
        start, end = max(self.start, lo), min(self.end, hi)
        return Interval(start, end) if end > start else None


def union_duration(intervals: Iterable[Interval]) -> float:
    """Wall time covered by at least one interval. Overlaps counted once."""
    ordered = sorted(intervals, key=lambda i: i.start)
    total, cursor = 0.0, None
    for interval in ordered:
        if cursor is None or interval.start > cursor[1]:
            if cursor is not None:
                total += cursor[1] - cursor[0]
            cursor = [interval.start, interval.end]
        else:
            cursor[1] = max(cursor[1], interval.end)
    if cursor is not None:
        total += cursor[1] - cursor[0]
    return total


def total_duration(intervals: Iterable[Interval]) -> float:
    """Sum of durations. Overlaps counted twice -- that is the point."""
    return sum(i.duration for i in intervals)


@dataclasses.dataclass(frozen=True)
class TraceUnits:
    """How the caller says the trace's timestamps convert to seconds.

    ``confirmed`` is not decoration. An unconfirmed conversion propagates into
    every counter this module derives, and the emitted provenance says so
    rather than presenting a plausible number as a known one.
    """

    seconds_per_tick: float
    source: str
    confirmed: bool

    def to_seconds(self, ticks: float) -> float:
        return float(ticks) * self.seconds_per_tick


@dataclasses.dataclass(frozen=True)
class EventMapping:
    """Which trace event names belong to which named term, and of what kind."""

    bucket_of_event: Mapping[str, str]
    kind_of_bucket: Mapping[str, BucketKind]

    def bucket(self, event_name: str) -> Optional[str]:
        return self.bucket_of_event.get(event_name)

    def kind(self, bucket: str) -> BucketKind:
        return self.kind_of_bucket[bucket]


# --------------------------------------------------------------------------
# The per-step profile: what the emitter emits and the decider consumes.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DecodeStepProfile:
    """One decode step, fully attributed as far as the trace allows.

    ``unmapped_event_seconds`` is kept rather than discarded: trace events that
    matched no bucket are exactly the material of the closure residual, and
    dropping them would make the residual look smaller than it is.
    """

    step: int
    run_id: str
    step_time: Quantity
    idle_strict: Quantity
    idle_loose: Quantity
    buckets: Mapping[str, Quantity]
    bucket_overlap: Optional[Quantity]
    hbm_bytes: Optional[Quantity] = None
    unmapped_event_seconds: float = 0.0
    unmapped_event_names: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "run_id": self.run_id,
            "step_time": self.step_time.to_dict(),
            "idle_strict": self.idle_strict.to_dict(),
            "idle_loose": self.idle_loose.to_dict(),
            "buckets": {k: v.to_dict() for k, v in self.buckets.items()},
            "bucket_overlap":
                self.bucket_overlap.to_dict() if self.bucket_overlap else None,
            "hbm_bytes": self.hbm_bytes.to_dict() if self.hbm_bytes else None,
            "unmapped_event_seconds": self.unmapped_event_seconds,
            "unmapped_event_names": list(self.unmapped_event_names),
        }


def step_from_dict(payload: Mapping[str, Any]) -> DecodeStepProfile:
    overlap = payload.get("bucket_overlap")
    hbm = payload.get("hbm_bytes")
    return DecodeStepProfile(
        step=int(payload["step"]),
        run_id=str(payload["run_id"]),
        step_time=quantity_from_dict(payload["step_time"]),
        idle_strict=quantity_from_dict(payload["idle_strict"]),
        idle_loose=quantity_from_dict(payload["idle_loose"]),
        buckets={k: quantity_from_dict(v)
                 for k, v in payload.get("buckets", {}).items()},
        bucket_overlap=quantity_from_dict(overlap) if overlap else None,
        hbm_bytes=quantity_from_dict(hbm) if hbm else None,
        unmapped_event_seconds=float(payload.get("unmapped_event_seconds", 0.0)),
        unmapped_event_names=tuple(payload.get("unmapped_event_names", ())))


def step_profile_from_intervals(
    *,
    step: int,
    run_id: str,
    window: Interval,
    intervals_by_bucket: Mapping[str, Sequence[Interval]],
    mapping: EventMapping,
    hbm_bytes: Optional[Quantity] = None,
    unmapped: Sequence[Tuple[str, Interval]] = (),
) -> DecodeStepProfile:
    """Turns one step's intervals into the emitted profile.

    Everything is clipped to the step window first: an interval that straddles
    the boundary contributes only the part inside, or the terms would sum past
    the step and the closure residual would go negative for a reason that has
    nothing to do with double counting.
    """
    clipped: Dict[str, List[Interval]] = {}
    for bucket, intervals in intervals_by_bucket.items():
        kept = [c for c in (i.clipped(window.start, window.end) for i in intervals)
                if c is not None]
        clipped[bucket] = kept

    per_bucket = {
        bucket: Quantity(union_duration(intervals),
                         dataclasses.replace(COUNTER_BUCKET, name=f"bucket.{bucket}"))
        for bucket, intervals in clipped.items()
    }

    unmapped_clipped = [(name, c)
                        for name, c in ((n, i.clipped(window.start, window.end))
                                        for n, i in unmapped) if c is not None]

    mapped_only: List[Interval] = []
    compute_only: List[Interval] = []
    for bucket, intervals in clipped.items():
        mapped_only.extend(intervals)
        if mapping.kind(bucket) is BucketKind.COMPUTE:
            compute_only.extend(intervals)

    # STRICT IDLE IS THE COMPLEMENT OF *ALL* DEVICE-BUSY TIME, MAPPED OR NOT.
    # An unmapped kernel is a kernel executing, and the spec defines strict idle
    # as a gap with no kernel executing and no DMA in flight. Omitting the
    # unmapped intervals here books device-busy time as host idle, which inflates
    # f_host and can only ever inflate it -- the error is one-signed toward the
    # host-bound reading. See round 1 C1.
    everything: List[Interval] = mapped_only + [c for _, c in unmapped_clipped]

    step_seconds = window.duration
    idle_strict = max(0.0, step_seconds - union_duration(everything))
    idle_loose = max(0.0, step_seconds - union_duration(compute_only))
    # The overlap is a statement about the NAMED terms only: how much wall time
    # two named terms both claim. It stays on the mapped union deliberately --
    # folding unmapped time in here would net an omission against a double count
    # and make both unreadable.
    overlap = sum(q.value for q in per_bucket.values()) - union_duration(mapped_only)

    return DecodeStepProfile(
        step=step,
        run_id=run_id,
        step_time=Quantity(step_seconds, COUNTER_STEP_TIME),
        idle_strict=Quantity(idle_strict, COUNTER_IDLE_STRICT),
        idle_loose=Quantity(idle_loose, COUNTER_IDLE_LOOSE),
        buckets=per_bucket,
        bucket_overlap=Quantity(max(0.0, overlap), COUNTER_BUCKET_OVERLAP),
        hbm_bytes=hbm_bytes,
        unmapped_event_seconds=total_duration([c for _, c in unmapped_clipped]),
        unmapped_event_names=tuple(sorted({n for n, _ in unmapped_clipped})))


# --------------------------------------------------------------------------
# Derivations. Named quantities, computed once, published in full.
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StepDerivation:
    step: int
    step_time_s: float
    idle_strict_s: float
    idle_loose_s: float
    f_host_strict: float
    f_host_loose: float
    non_overlap_term_s: float
    bucket_seconds: Mapping[str, float]
    attributed_s: float
    unattributed_s: float
    unattributed_fraction: float
    bucket_overlap_s: Optional[float]
    bucket_overlap_fraction: Optional[float]
    # Carried so the residual is DECOMPOSABLE by a reader. Post-R1 the residual
    # is (unmapped-exclusive busy time) - (bucket overlap), and those two can
    # cancel. A reader who sees only the net cannot tell "nothing wrong" from
    # "an omission netted against a double count", so both parts travel with it.
    unmapped_event_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["bucket_seconds"] = dict(self.bucket_seconds)
        return out


def derive(profile: DecodeStepProfile) -> StepDerivation:
    """Computes every derived quantity for one step. Decides nothing."""
    s = profile.step_time.value
    if s <= 0:
        raise ValueError(f"step {profile.step} has non-positive wall time {s}")
    idle_strict = profile.idle_strict.value
    idle_loose = profile.idle_loose.value
    buckets = {k: v.value for k, v in profile.buckets.items()}
    # THE ATTRIBUTED SIDE IS IDLE PLUS THE *NAMED* TERMS, AND NOTHING ELSE.
    #
    # Round 1 R1: unmapped_event_seconds used to be added here. That is backwards
    # by definition -- an unmapped event is precisely one the instrument could
    # NOT attribute to a named term, so adding it to the attributed side books
    # unattributed time as attributed and reports perfect closure over exactly
    # the material the residual exists to surface. The closure identity the
    # design states is
    #
    #     unattributed = S - (I + grouped_matmul + collective + dma_busy + ...)
    #
    # over the NAMED terms; unmapped is not among them.
    #
    # NOTE FOR ANYONE RE-DERIVING R1: fixing C1 alone does NOT make this term
    # able to fire, contrary to the round 1 review's suggested fix. With C1 fixed
    # and this line unchanged the residual is
    #     union(all) - union(mapped) - overlap - total(unmapped)  <= 0
    # because union(all) - union(mapped) <= total(unmapped) always. Measured:
    # still 0 positives in 2000 randomised emitter profiles, max exactly 0.0.
    # BOTH lines had to change. With both fixed the residual is
    #     (time covered ONLY by unmapped events) - bucket_overlap
    # which is positive on unattributed device time, negative on a double count,
    # and therefore a check that can actually fire in both directions.
    attributed = idle_strict + sum(buckets.values())
    unattributed = s - attributed
    overlap = profile.bucket_overlap.value if profile.bucket_overlap else None
    return StepDerivation(
        step=profile.step,
        step_time_s=s,
        idle_strict_s=idle_strict,
        idle_loose_s=idle_loose,
        f_host_strict=idle_strict / s,
        f_host_loose=idle_loose / s,
        non_overlap_term_s=idle_loose - idle_strict,
        bucket_seconds=buckets,
        attributed_s=attributed,
        unattributed_s=unattributed,
        unattributed_fraction=unattributed / s,
        bucket_overlap_s=overlap,
        bucket_overlap_fraction=None if overlap is None else overlap / s,
        unmapped_event_seconds=profile.unmapped_event_seconds)


# --------------------------------------------------------------------------
# Rule 1.
# --------------------------------------------------------------------------


class HostBoundVerdict(str, enum.Enum):
    HOLDS = "HOLDS"
    MIXED = "MIXED"
    REFUTED = "REFUTED"
    UNDETERMINED = "UNDETERMINED"


def classify_f_host(f_host: float, holds_at: float,
                    refuted_at: float) -> HostBoundVerdict:
    """The whole of Rule 1, on one number. Both cuts are inclusive.

    MIXED is the strict interior. It is a result, not a gap in the rule: the
    design defines it, so a value of 0.30 is answered, not deferred.
    """
    if f_host >= holds_at:
        return HostBoundVerdict.HOLDS
    if f_host <= refuted_at:
        return HostBoundVerdict.REFUTED
    return HostBoundVerdict.MIXED


def decide_host_bound(derivations: Sequence[StepDerivation],
                      thresholds: common.Thresholds) -> common.Check:
    """Rule 1 over a run, using the STRICT idle definition throughout."""
    name = "m1.rule1_host_bound"
    if not derivations:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "no decode steps were profiled, so f_host has no value; an empty "
            "run is undetermined and never a refutation",
            {"steps": 0, "verdict": HostBoundVerdict.UNDETERMINED.value})

    holds_at = thresholds.require("m1.f_host_holds_at_or_above")
    refuted_at = thresholds.require("m1.f_host_refuted_at_or_below")
    if thresholds.require("m1.idle_definition") != "strict":
        return common.check(
            name, common.Outcome.FAILED,
            "the pinned idle definition is not 'strict'; Rule 1 is defined on "
            "strict device idle and this module will not evaluate it otherwise",
            {"idle_definition": thresholds.require("m1.idle_definition"),
             "verdict": HostBoundVerdict.UNDETERMINED.value})

    per_step = {d.step: d.f_host_strict for d in derivations}
    impossible = {k: v for k, v in per_step.items() if v < 0.0 or v > 1.0}
    total_idle = sum(d.idle_strict_s for d in derivations)
    total_step = sum(d.step_time_s for d in derivations)
    aggregate = total_idle / total_step
    branches = {k: classify_f_host(v, holds_at, refuted_at).value
                for k, v in per_step.items()}
    intermediates = {
        "f_host_by_step": per_step,
        "branch_by_step": branches,
        "aggregate_f_host": aggregate,
        "aggregate_definition": "sum(idle_strict) / sum(step_time), time-weighted",
        "idle_seconds_total": total_idle,
        "step_seconds_total": total_step,
        "holds_at_or_above": holds_at,
        "refuted_at_or_below": refuted_at,
        "spread": common.summarise([d.f_host_strict for d in derivations]),
    }

    if impossible:
        intermediates["impossible_steps"] = impossible
        intermediates["verdict"] = HostBoundVerdict.UNDETERMINED.value
        return common.check(
            name, common.Outcome.FAILED,
            "f_host left [0, 1] on at least one step, so idle time exceeded "
            "the step that contains it; that is an instrument fault and no "
            "branch of Rule 1 may be read off it",
            intermediates)

    distinct = sorted(set(branches.values()))
    if len(distinct) > 1:
        intermediates["verdict"] = HostBoundVerdict.UNDETERMINED.value
        intermediates["branches_present"] = distinct
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "steps of one run landed in different branches of Rule 1 "
            f"({', '.join(distinct)}), so the run has no single answer. Note "
            "this is NOT the MIXED branch: MIXED means f_host fell in the "
            "defined middle band, and steps disagreeing is a different fact "
            "that is not allowed to borrow its name",
            intermediates)

    verdict = classify_f_host(aggregate, holds_at, refuted_at)
    intermediates["verdict"] = verdict.value
    return common.check(
        name, common.Outcome.PASSED,
        f"Rule 1 evaluated on strict device idle: f_host = {aggregate:.4f} "
        f"against holds >= {holds_at} / refuted <= {refuted_at}, verdict "
        f"{verdict.value}",
        intermediates)


def check_non_overlap_term(derivations: Sequence[StepDerivation],
                           thresholds: common.Thresholds) -> common.Check:
    """Is Rule 1's answer an artefact of which idle definition was used?

    The design asserts the non-overlap term sits nowhere near the cuts. "Near"
    has no tolerance from any named party and one is not invented here. What
    IS mechanical, and stronger, is the substitution test: recompute the branch
    with the loose definition and see whether it moves. If it does, the verdict
    is a property of a definitional choice rather than of the machine.
    """
    name = "m1.non_overlap_term"
    if not derivations:
        return common.check(name, common.Outcome.UNDETERMINED,
                            "no steps to compute the non-overlap term from",
                            {"steps": 0})

    holds_at = thresholds.require("m1.f_host_holds_at_or_above")
    refuted_at = thresholds.require("m1.f_host_refuted_at_or_below")
    strict_total = sum(d.idle_strict_s for d in derivations)
    loose_total = sum(d.idle_loose_s for d in derivations)
    step_total = sum(d.step_time_s for d in derivations)
    strict_branch = classify_f_host(strict_total / step_total, holds_at, refuted_at)
    loose_branch = classify_f_host(loose_total / step_total, holds_at, refuted_at)
    intermediates = {
        "non_overlap_term_seconds_by_step":
            {d.step: d.non_overlap_term_s for d in derivations},
        "non_overlap_term_fraction_of_step":
            {d.step: d.non_overlap_term_s / d.step_time_s for d in derivations},
        "definition": "I_loose - I_strict: DMA in flight with no compute running",
        "f_host_strict": strict_total / step_total,
        "f_host_loose": loose_total / step_total,
        "branch_strict": strict_branch.value,
        "branch_loose": loose_branch.value,
        "distance_from_holds_cut": abs(strict_total / step_total - holds_at),
        "distance_from_refuted_cut": abs(strict_total / step_total - refuted_at),
        "channel": "SEPARATE. This term is never added into I and never "
                   "substituted into Rule 1.",
    }
    if strict_branch is loose_branch:
        return common.check(
            name, common.Outcome.PASSED,
            "the branch of Rule 1 is unchanged by the idle definition, so the "
            "verdict does not rest on the DMA-in-flight mask",
            intermediates)
    return common.check(
        name, common.Outcome.FAILED,
        f"swapping strict idle for loose idle moves Rule 1 from "
        f"{strict_branch.value} to {loose_branch.value}. The non-overlap term "
        "straddles a cut, which is exactly the condition the design says must "
        "not obtain; the strict verdict must not be reported as if the choice "
        "were immaterial",
        intermediates)


# --------------------------------------------------------------------------
# Rule 2.
# --------------------------------------------------------------------------


class EfficiencyVerdict(str, enum.Enum):
    BYTE_BOUND = "BYTE_BOUND"
    PARTIAL = "PARTIAL"
    EFFICIENCY_DEFICIT = "EFFICIENCY_DEFICIT"
    VOID = "VOID"
    UNDETERMINED = "UNDETERMINED"


@dataclasses.dataclass(frozen=True)
class EfficiencyPair:
    """e_dec and e_ref, each carrying the run it was measured in.

    The run ids are not bookkeeping. The design requires the reference to be
    measured in the SAME profile run; a reference imported from another run
    silently changes what the ratio means, and the only way to catch that is to
    carry the provenance and compare it.
    """

    e_dec: float
    e_ref: float
    run_id_dec: str
    run_id_ref: str
    e_dec_source: str = ""
    e_ref_source: str = ""

    @property
    def same_run(self) -> bool:
        return self.run_id_dec == self.run_id_ref

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def decide_efficiency(pair: Optional[EfficiencyPair],
                      thresholds: common.Thresholds) -> common.Check:
    """Rule 2, including the void path that outranks it."""
    name = "m1.rule2_efficiency"
    if pair is None:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "no efficiency pair was emitted, so Rule 2 was not evaluated",
            {"pair": None, "verdict": EfficiencyVerdict.UNDETERMINED.value})

    byte_bound_ratio = thresholds.require("m1.e_dec_byte_bound_ratio")
    deficit_ratio = thresholds.require("m1.e_dec_efficiency_deficit_ratio")
    floor = thresholds.require("m1.e_ref_plausibility_floor")

    intermediates = {
        "e_dec": pair.e_dec,
        "e_ref": pair.e_ref,
        "run_id_dec": pair.run_id_dec,
        "run_id_ref": pair.run_id_ref,
        "same_profile_run": pair.same_run,
        "e_dec_source": pair.e_dec_source,
        "e_ref_source": pair.e_ref_source,
        "byte_bound_at_or_above": byte_bound_ratio,
        "deficit_at_or_below": deficit_ratio,
        "e_ref_plausibility_floor": floor,
    }

    if not pair.same_run:
        intermediates["verdict"] = EfficiencyVerdict.UNDETERMINED.value
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "e_dec and e_ref come from different profile runs; the design "
            "requires the reference in the same run and the ratio is not "
            "comparable across runs",
            intermediates)

    # R3. THIS TEST MUST STAY ABOVE THE FLOOR TEST BELOW, and the ordering is
    # the whole fix. Zero and every negative are also below the 0.5 floor, so
    # while the floor answered first this branch was UNREACHABLE and an
    # impossible reading was reported as PASSED with a VOID verdict -- an
    # orderly, legitimate-looking disposition.
    #
    # VOID is a result about the MACHINE: the reference kernel ran and did not
    # reach a credible efficiency. FAILED is a result about the MEASUREMENT:
    # the number we were handed cannot be a measurement of anything. Folding
    # the second into the first is the axis collapse this module's two-axis
    # design exists to prevent.
    #
    # It is hoisted above the FLOOR test only, and deliberately NOT above the
    # same-run test: a non-positive reference imported from a different run is
    # not comparable in the first place, and calling it an instrument fault
    # would assert more than we know.
    if pair.e_ref <= 0:
        intermediates["verdict"] = EfficiencyVerdict.UNDETERMINED.value
        return common.check(
            name, common.Outcome.FAILED,
            f"e_ref = {pair.e_ref} is non-positive, so the ratio is undefined. "
            "An efficiency cannot be zero or negative; this is an instrument "
            "fault and not a low reference, and it is NOT the VOID disposition",
            intermediates)

    if pair.e_ref < floor:
        intermediates["verdict"] = EfficiencyVerdict.VOID.value
        return common.check(
            name, common.Outcome.PASSED,
            f"e_ref = {pair.e_ref} is below the plausibility floor {floor}: "
            "the reference kernel did not reach a credible efficiency, so the "
            "cut is VOID and neither BYTE_BOUND nor EFFICIENCY_DEFICIT may be "
            "read off it. This is a defined disposition, not a failure to "
            "measure",
            intermediates)

    ratio = pair.e_dec / pair.e_ref
    intermediates["ratio_e_dec_over_e_ref"] = ratio
    if ratio >= byte_bound_ratio:
        verdict = EfficiencyVerdict.BYTE_BOUND
    elif ratio <= deficit_ratio:
        verdict = EfficiencyVerdict.EFFICIENCY_DEFICIT
    else:
        verdict = EfficiencyVerdict.PARTIAL
    intermediates["verdict"] = verdict.value
    return common.check(
        name, common.Outcome.PASSED,
        f"Rule 2 evaluated in one profile run: e_dec/e_ref = {ratio:.4f}, "
        f"verdict {verdict.value}",
        intermediates)


# --------------------------------------------------------------------------
# E6a: closure residual and bucket overlap. Emitted, not adjudicated.
# --------------------------------------------------------------------------


# The representable-noise floor for the sign of the closure residual. See the
# comment at its use site in check_closure: this is not a threshold anybody was
# asked to set, it is the precision of the arithmetic that produces the number.
_RESIDUAL_SIGN_EPSILON = 1e-12


def check_closure(derivations: Sequence[StepDerivation],
                  thresholds: common.Thresholds) -> common.Check:
    """The closure residual, per step, as a fraction of S.

    NO THRESHOLD EXISTS FOR THIS AND NONE IS INVENTED HERE. The entry in
    thresholds.json is null with kind 'deliberately-absent'. So the leg can
    return exactly two things:

      FAILED         when the residual is negative, meaning the named terms
                     sum to more wall time than the step contains. Zero is not
                     a tuning constant; it is the edge of arithmetic
                     possibility, and a negative residual is a double count.
      UNDETERMINED   in every other case. The residual is published and left
                     for a human to read. It never returns PASSED, because a
                     pass would assert a standard that no party has set.

    A residual near zero is NOT evidence of good attribution while the bucket
    overlap is non-zero: overlapping terms can sum to S while double-counting
    one region and omitting another. So the overlap travels with the residual
    and the reason says so out loud.
    """
    name = "m1.closure_residual"
    entry = thresholds.entry("m1.closure_residual_threshold")
    if not derivations:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "no steps, so there is no residual to publish",
            {"steps": 0, "threshold": entry.to_dict()})

    by_step = {d.step: d.unattributed_fraction for d in derivations}
    seconds = {d.step: d.unattributed_s for d in derivations}
    overlaps = {d.step: d.bucket_overlap_fraction for d in derivations}
    intermediates = {
        "definition": "S - (I_strict + every named term). Unmapped events are "
                      "NOT a named term and are NOT on the attributed side: "
                      "they are the material this residual exists to surface",
        "unattributed_fraction_by_step": by_step,
        "unattributed_seconds_by_step": seconds,
        "unmapped_seconds_by_step":
            {d.step: d.unmapped_event_seconds for d in derivations},
        "bucket_overlap_fraction_by_step": overlaps,
        "attributed_seconds_by_step": {d.step: d.attributed_s for d in derivations},
        "bucket_seconds_by_step": {d.step: dict(d.bucket_seconds) for d in derivations},
        "spread": common.summarise(list(by_step.values())),
        "threshold": entry.to_dict(),
        "sign_epsilon": _RESIDUAL_SIGN_EPSILON,
    }

    # Round 1 non-blocking: an exact `< 0.0` fires on -2.8e-17. The residual is a
    # difference of sums, so noise at machine precision is reachable and calling
    # it an arithmetic contradiction would be a false FAILED. This epsilon is NOT
    # a tuning constant and no party is being asked to set one: it is the
    # representable-noise floor of the arithmetic itself, and it is published in
    # the intermediates so a reader can see exactly what was treated as zero.
    negative = {k: v for k, v in by_step.items() if v < -_RESIDUAL_SIGN_EPSILON}
    if negative:
        intermediates["negative_steps"] = negative
        return common.check(
            name, common.Outcome.FAILED,
            "the named terms sum to more than the step they sit in, so at "
            "least one region of the timeline is attributed twice. This is an "
            "arithmetic contradiction, not a threshold being crossed",
            intermediates)

    # Round 1 non-blocking nit: `if v` conflated None (never measured) with 0.0
    # (measured, and zero) -- the exact conflation this module works hard to
    # avoid everywhere else. A step with NO overlap measurement is the stronger
    # reason to refuse to read the residual as closure, not a reason to skip it.
    unmeasured = [k for k, v in overlaps.items() if v is None]
    overlapping = {k: v for k, v in overlaps.items() if v is not None and v != 0.0}
    if unmeasured:
        intermediates["steps_without_an_overlap_measurement"] = unmeasured
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "residual published, but at least one step carries no bucket-overlap "
            "measurement at all. Without it the residual cannot be read as a "
            "closure measure: an omission and a double count net against each "
            "other and a small value would mean nothing",
            intermediates)

    if overlapping:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "residual published. It is NOT a closure measure here: the named "
            "terms overlap each other, so the residual nets a double count "
            "against an omission and a small value would mean nothing. Read "
            "it next to the bucket overlap. No threshold exists for either",
            intermediates)

    return common.check(
        name, common.Outcome.UNDETERMINED,
        "residual published with no verdict attached: no party has set a "
        "threshold for it, and this module does not invent one",
        intermediates)


def check_bucket_overlap(derivations: Sequence[StepDerivation],
                         thresholds: common.Thresholds) -> common.Check:
    """How much wall time two named terms both claim. Also unadjudicated."""
    name = "m1.bucket_overlap"
    entry = thresholds.entry("m1.bucket_overlap_threshold")
    if not derivations:
        return common.check(name, common.Outcome.UNDETERMINED,
                            "no steps, so there is no overlap to publish",
                            {"steps": 0, "threshold": entry.to_dict()})

    unknown = [d.step for d in derivations if d.bucket_overlap_s is None]
    fractions = {d.step: d.bucket_overlap_fraction
                 for d in derivations if d.bucket_overlap_fraction is not None}
    intermediates = {
        "definition": "sum of per-term durations minus the union across terms",
        "overlap_fraction_by_step": fractions,
        "overlap_seconds_by_step":
            {d.step: d.bucket_overlap_s for d in derivations},
        "steps_without_an_overlap_measurement": unknown,
        "threshold": entry.to_dict(),
    }
    if unknown:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "some steps carry no overlap measurement, so their closure "
            "residual cannot be read as a closure measure at all",
            intermediates)
    intermediates["spread"] = common.summarise(list(fractions.values()))
    return common.check(
        name, common.Outcome.UNDETERMINED,
        "overlap published with no verdict attached: no threshold exists for "
        "it and none is invented here",
        intermediates)


def check_counter_units(profiles: Sequence[DecodeStepProfile]) -> common.Check:
    """Which emitted counters carry units nobody has confirmed.

    Named for the QUESTION and not for one of its answers. It was
    ``m1.counter_units_confirmed``, which reads as an assertion that the units
    are confirmed while the check's whole job is to list the ones that are not
    -- so a reader skimming an artifact for check names saw a reassurance where
    the payload holds the opposite. A deliberate rename closing a named review
    nit, not tidying: the old name appears in no artifact in this repository
    and in no other module.
    """
    name = "m1.counter_units"
    if not profiles:
        return common.check(name, common.Outcome.UNDETERMINED,
                            "nothing was emitted, so no units were declared",
                            {"profiles": 0})
    unconfirmed: Dict[str, str] = {}
    confirmed: List[str] = []
    for profile in profiles:
        quantities: List[Quantity] = [
            profile.step_time, profile.idle_strict, profile.idle_loose
        ]
        quantities.extend(profile.buckets.values())
        if profile.bucket_overlap:
            quantities.append(profile.bucket_overlap)
        if profile.hbm_bytes:
            quantities.append(profile.hbm_bytes)
        for q in quantities:
            if q.provenance.units_confirmed:
                confirmed.append(q.provenance.name)
            else:
                unconfirmed[q.provenance.name] = q.provenance.units
    intermediates = {
        "unconfirmed_counters": unconfirmed,
        "confirmed_counters": sorted(set(confirmed)),
    }
    if unconfirmed:
        return common.check(
            name, common.Outcome.UNDETERMINED,
            "at least one emitted counter has units that were not established "
            "from the profiler's own output or documentation. Those values are "
            "emitted with units UNKNOWN and must not be compared against a "
            "reference that assumes a unit",
            intermediates)
    return common.check(name, common.Outcome.PASSED,
                        "every emitted counter carries confirmed units",
                        intermediates)


# --------------------------------------------------------------------------
# Assembly.
# --------------------------------------------------------------------------


def assess(profiles: Sequence[DecodeStepProfile],
           pair: Optional[EfficiencyPair],
           thresholds: common.Thresholds) -> Tuple[List[StepDerivation],
                                                   List[common.Check]]:
    derivations = [derive(p) for p in profiles]
    checks = [
        decide_host_bound(derivations, thresholds),
        check_non_overlap_term(derivations, thresholds),
        decide_efficiency(pair, thresholds),
        check_closure(derivations, thresholds),
        check_bucket_overlap(derivations, thresholds),
        check_counter_units(profiles),
    ]
    return derivations, checks


def payload(profiles: Sequence[DecodeStepProfile],
            derivations: Sequence[StepDerivation],
            pair: Optional[EfficiencyPair]) -> Dict[str, Any]:
    return {
        "steps": [p.to_dict() for p in profiles],
        "derived": [d.to_dict() for d in derivations],
        "efficiency_pair": pair.to_dict() if pair else None,
        "reference_step_times_note":
            "m1.reference_step_ms exists in thresholds.json for orientation "
            "only. No leg in this module reads it, and no verdict here is "
            "computed against it.",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True,
                        help="JSON emitted by the profile emitter")
    parser.add_argument("--out", required=True, help="artifact path to write")
    parser.add_argument("--thresholds", default=None)
    args = parser.parse_args(argv)

    thresholds = common.Thresholds.load(args.thresholds)
    document = common.load_json(args.profile)
    profiles = [step_from_dict(s) for s in document.get("steps", [])]
    raw_pair = document.get("efficiency_pair")
    pair = EfficiencyPair(**raw_pair) if raw_pair else None

    derivations, checks = assess(profiles, pair, thresholds)
    common.write_artifact(args.out,
                          kind="m1.profile_decision",
                          payload=payload(profiles, derivations, pair),
                          checks=checks,
                          thresholds=thresholds)
    overall = common.worst(c.outcome for c in checks)
    print(f"M1: {overall.value}; artifact written to {args.out}")
    for item in checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    # THE EXIT CODE TRACKS THE *OUTCOME* AXIS, NEVER THE VERDICT AXIS.
    #
    # This used to `return 0` unconditionally -- the only main() in the package
    # that did. An artifact recording overall_outcome FAILED still exited 0, so
    # any harness or operator wrapper gating on exit status read every M1 result
    # as success, and M1 is the instrument that decides between the two rival
    # explanations. Same convention as M0/M2/M3/M4/M6: 0 decided cleanly,
    # 1 something FAILED or could not be checked, 2 the run could not start.
    #
    # Deliberately NOT conditioned on the Rule 1 verdict: HOLDS, MIXED and
    # REFUTED are all successful measurements and all exit 0. An exit code that
    # went non-zero on REFUTED would make "the rule refuted the hypothesis"
    # indistinguishable from "the rule could not be run", which is precisely the
    # collapse this module's two-axis design exists to prevent.
    return 0 if overall is common.Outcome.PASSED else 1


if __name__ == "__main__":
    sys.exit(main())
