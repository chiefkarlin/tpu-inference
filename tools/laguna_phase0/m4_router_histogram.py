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
"""M4 -- the router-output histogram: distinct experts per layer per step.

Measured at c1, c16 and c32. It costs no timed run -- it is read off the router's
own output during a run that is happening anyway -- and per review finding B4 it
gates the whole ranking rather than one candidate, because its result selects the
Phase 1 candidate. It is the highest-leverage cheap item in Phase 0.

WHAT IS EMITTED. The per-layer histogram, not just the summary ``E``. The summary
is a verdict; the histogram is the intermediate, and a verdict nobody can
recompute is not a measurement. Every ladder point carries its basis ON THE
POINT.

WHAT IS DELIBERATELY NOT COMPUTED HERE. The Phase 1 naming statistic -- the
c1/c16 residual ratio -- and the name of the Phase 1 candidate. Those belong to
the engineering manager and are gated on M4 actually running. This module emits
``E`` at c1 and c16 in a form that can feed the statistic, and stops there.

ACCEPTANCE, FROM THE DESIGN. If E(32) is within 10% of 183 the byte model stands.
If E(1) is 10-20, padding rows do not disperse. If E(1) is near 119, they do --
but "near" has no tolerance in any document, and none is invented here. See
``m4.e1_disperse_tolerance_fraction`` in ``thresholds.json``, which is null with
the escalation recorded in its source field.

THE DISPERSAL LEG PARTITIONS THE WHOLE LINE INTO THREE NAMED REGIONS, because a
two-way test with an abstention bolted onto one arm files two different findings
under one label. E(1) = 119.4 means "the dispersal reading looks right and we
lack a tolerance". E(1) = 60 means "NEITHER prediction in the design holds",
which is a substantive result about the design and the one nobody would go
looking for if it were filed under a word meaning "we could not tell". Both
distances -- to the band and to 119 -- are published on every result, because
the distance to 119 alone cannot tell those two apart and the pair can, with no
tolerance ruled at all.

A KNOWN BIAS, STATED IN THE OUTPUT AND NOT ONLY IN THE NOTES. Only the
non-dispersal hypothesis was given an exact band, so this leg can decisively
confirm only non-dispersal -- the finding that leaves the ranking intact.
Dispersal, which would damage it, can at best come back as pending. The gap runs
in the direction the campaign would prefer, and it is escalated rather than
logged.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.laguna_phase0 import basis as basis_mod
from tools.laguna_phase0 import common

ROUTER_COUNTER = common.CounterProvenance(
    name="distinct_experts",
    units="distinct expert ids per layer per decode step",
    source=("counted by this module from the router's own top-k index tensor; "
            "the unit is established by construction rather than read out of a "
            "profiler"),
    units_confirmed=True)


@dataclasses.dataclass(frozen=True)
class RouterObservation:
    """Distinct experts activated in one layer during one decode step.

    Attributes:
      step: Decode step index within the capture.
      layer: Layer index.
      distinct_experts: Count of distinct expert ids the router selected.
      rows: Number of routed rows, padding included or not per
        ``padding_rows_included`` on the capture.
    """

    step: int
    layer: int
    distinct_experts: int
    rows: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def observations_from_router_indices(step: int,
                                     indices_by_layer: Dict[int, Sequence[Sequence[int]]]
                                     ) -> List[RouterObservation]:
    """Counts distinct experts per layer from top-k index rows.

    Args:
      step: The decode step these indices came from.
      indices_by_layer: Layer index -> rows of selected expert ids, one row per
        routed token (padding rows included or excluded by the caller, and
        declared on the capture).
    """
    out: List[RouterObservation] = []
    for layer, rows in sorted(indices_by_layer.items()):
        flat = [int(e) for row in rows for e in row]
        out.append(
            RouterObservation(step=step,
                              layer=layer,
                              distinct_experts=len(set(flat)),
                              rows=len(list(rows))))
    return out


@dataclasses.dataclass
class Capture:
    """One concurrency's worth of router observations, and what it means.

    Attributes:
      concurrency: The ladder point.
      shape: Benchmark shape, recorded on every emitted point.
      observations: Per layer, per step.
      padding_rows_included: Whether padding rows were routed and counted.
        ``None`` means it was not recorded, which makes the padding question
        unanswerable from this capture -- and that question is what M4 is for.
    """

    concurrency: int
    shape: str
    observations: List[RouterObservation]
    padding_rows_included: Optional[bool] = None

    def per_layer_histogram(self) -> Dict[int, Dict[int, int]]:
        """Layer -> {distinct-expert count -> how many steps showed it}."""
        hist: Dict[int, Dict[int, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        for obs in self.observations:
            hist[obs.layer][obs.distinct_experts] += 1
        return {layer: dict(sorted(counts.items())) for layer, counts in sorted(hist.items())}

    def per_layer_e(self) -> Dict[int, float]:
        """Layer -> mean distinct experts per step. ``E`` per layer."""
        by_layer: Dict[int, List[int]] = collections.defaultdict(list)
        for obs in self.observations:
            by_layer[obs.layer].append(obs.distinct_experts)
        return {layer: statistics.fmean(values) for layer, values in sorted(by_layer.items())}

    def summary_e(self) -> Optional[float]:
        """Mean over layers of the per-layer means. ``None`` on an empty capture."""
        per_layer = self.per_layer_e()
        if not per_layer:
            return None
        return statistics.fmean(per_layer.values())

    def ladder_point(self) -> basis_mod.LadderPoint:
        """``E`` at this concurrency, with its basis attached to the point."""
        return basis_mod.LadderPoint(
            concurrency=self.concurrency,
            value=self.summary_e(),
            units=ROUTER_COUNTER.units,
            basis=basis_mod.MeasurementBasis.ROUTER_COUNT,
            instrument="M4",
            shape=self.shape,
            detail={
                "counter": ROUTER_COUNTER.to_dict(),
                "padding_rows_included": self.padding_rows_included,
                "layers": len(self.per_layer_e()),
                "steps": len({o.step for o in self.observations}),
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "shape": self.shape,
            "padding_rows_included": self.padding_rows_included,
            "per_layer_histogram": {str(k): {str(kk): vv for kk, vv in v.items()}
                                    for k, v in self.per_layer_histogram().items()},
            "per_layer_e": {str(k): v for k, v in self.per_layer_e().items()},
            "summary_e": common.counted(self.summary_e(), ROUTER_COUNTER),
            "observations": [o.to_dict() for o in self.observations],
        }


def check_byte_model(capture: Capture, thresholds: common.Thresholds) -> common.Check:
    """E(32) within 10% of 183 => the byte model stands."""
    reference = float(thresholds.require("m4.e32_reference"))
    tolerance = float(thresholds.require("m4.e32_tolerance_fraction"))
    observed = capture.summary_e()
    intermediates = {
        "observed_e": common.counted(observed, ROUTER_COUNTER),
        "reference": reference,
        "tolerance_fraction": tolerance,
        "per_layer_e": {str(k): v for k, v in capture.per_layer_e().items()},
        "concurrency": capture.concurrency,
    }
    if observed is None:
        return common.check("m4.byte_model", common.Outcome.UNDETERMINED,
                            "the capture is empty; an empty input is UNDETERMINED, "
                            "never a pass", intermediates)
    relative = abs(observed - reference) / reference
    intermediates["relative_difference"] = relative
    if relative <= tolerance:
        return common.check(
            "m4.byte_model", common.Outcome.PASSED,
            f"E(32) = {observed:.2f} is within {tolerance:.0%} of {reference}; the "
            "byte model stands as written", intermediates)
    return common.check(
        "m4.byte_model", common.Outcome.FAILED,
        f"E(32) = {observed:.2f} differs from {reference} by {relative:.1%}, "
        f"outside {tolerance:.0%}; the byte model does not stand as written",
        intermediates)


class DispersalRegion(str, enum.Enum):
    """Where E(1) fell, as a name a reader can search for.

    The design offers two neighbourhoods and the line has three regions. The
    third one -- E(1) in NEITHER -- is a substantive result about the design
    rather than a failure to measure, and it does not get to share a label with
    "we lack a tolerance", or nobody will ever go looking for it.
    """

    NON_DISPERSAL_BAND = "NON_DISPERSAL_BAND"
    DISPERSAL_NEIGHBOURHOOD = "DISPERSAL_NEIGHBOURHOOD"
    DISPERSAL_PENDING_TOLERANCE = "DISPERSAL_NEIGHBOURHOOD_PENDING_TOLERANCE"
    NEITHER_NEIGHBOURHOOD = "NEITHER_NEIGHBOURHOOD"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"


#: The two hypotheses are 99 apart: 119 minus the band's upper edge of 20. That
#: gap is not a tolerance and is not treated as one. It is used only for the
#: one inference it genuinely supports: a tolerance around 119 wide enough to
#: admit a point further away than the rival hypothesis itself would swallow
#: the rival hypothesis whole, and the two predictions would stop being
#: distinguishable. So beyond that distance, no tolerance anyone could rule
#: would rescue the dispersal reading, and the region can be named without one.
def _hypothesis_gap(band: Sequence[float], reference: float) -> float:
    return abs(reference - max(band))


def classify_dispersal(observed: float, band: Sequence[float],
                       reference: float,
                       tolerance: Optional[float]) -> DispersalRegion:
    """The three-way partition of the whole line. No region is left unnamed.

    Both discriminators are derived from the two anchors the design itself
    states; no new number is introduced. A point outside the band is called
    NEITHER when it is nearer the non-dispersal band than the dispersal anchor
    -- it missed the only exactly stated criterion and is not even closer to
    the other one -- or when it is further from 119 than the two hypotheses are
    from each other.
    """
    if band[0] <= observed <= band[1]:
        return DispersalRegion.NON_DISPERSAL_BAND

    distance_to_reference = abs(observed - reference)
    distance_to_band = min(abs(observed - band[0]), abs(observed - band[1]))

    if (distance_to_band < distance_to_reference
            or distance_to_reference > _hypothesis_gap(band, reference)):
        return DispersalRegion.NEITHER_NEIGHBOURHOOD

    if tolerance is None:
        return DispersalRegion.DISPERSAL_PENDING_TOLERANCE
    if distance_to_reference / reference <= tolerance:
        return DispersalRegion.DISPERSAL_NEIGHBOURHOOD
    return DispersalRegion.NEITHER_NEIGHBOURHOOD


def check_padding_dispersal(capture: Capture,
                            thresholds: common.Thresholds) -> common.Check:
    """Do padding rows disperse across experts at c1?

    A three-way partition, not a two-way test with an abstention bolted onto
    one arm:

      NON_DISPERSAL_BAND            E(1) inside the exactly stated band 10-20.
                                    PASSED: padding rows do not disperse.
      DISPERSAL_..._PENDING_TOLERANCE
                                    E(1) in the neighbourhood of 119, which the
                                    design states without a tolerance. No named
                                    party has supplied one and none is invented,
                                    so COULD_NOT_BE_CHECKED_MECHANICALLY.
      NEITHER_NEIGHBOURHOOD         E(1) in neither. FAILED -- and note what
                                    that means: not "the instrument failed" but
                                    "neither hypothesis in the design holds",
                                    which is a finding about the design.

    THE DISTANCE TO BOTH ANCHORS IS ALWAYS PUBLISHED. Distance to 119 alone
    cannot tell the second region from the third; distance to both can, with no
    tolerance ruled at all.

    A KNOWN BIAS IN THIS LEG, STATED IN THE OUTPUT AND NOT ONLY IN THE NOTES.
    As the design specifies it, this leg can decisively CONFIRM only
    non-dispersal -- the hypothesis that leaves the ranking intact -- because
    only that hypothesis was given an exact band. Dispersal, the finding that
    would damage the ranking, can at best be reported as pending. The gap runs
    in the direction the campaign would prefer, and it is escalated rather than
    logged.
    """
    band = [float(x) for x in thresholds.require("m4.e1_no_disperse_band")]
    reference = float(thresholds.require("m4.e1_disperse_reference"))
    observed = capture.summary_e()
    tolerance_entry = thresholds.entry("m4.e1_disperse_tolerance_fraction")
    intermediates: Dict[str, Any] = {
        "observed_e": common.counted(observed, ROUTER_COUNTER),
        "no_disperse_band": band,
        "disperse_reference": reference,
        "disperse_tolerance": tolerance_entry.to_dict(),
        "per_layer_e": {str(k): v for k, v in capture.per_layer_e().items()},
        "padding_rows_included": capture.padding_rows_included,
        "concurrency": capture.concurrency,
        "known_bias":
            "This leg can decisively confirm only NON-dispersal, because only "
            "that hypothesis was given an exact band. Dispersal -- the finding "
            "that would damage the ranking -- can at best come back as "
            "pending. The asymmetry favours the comfortable answer.",
    }
    if observed is None:
        intermediates["region"] = DispersalRegion.NOT_ESTABLISHED.value
        return common.check("m4.padding_dispersal", common.Outcome.UNDETERMINED,
                            "the capture is empty, so E(1) has no value; an "
                            "empty input is undetermined and never a pass",
                            intermediates)

    intermediates["distance_from_disperse_reference"] = observed - reference
    intermediates["relative_distance_from_disperse_reference"] = (
        abs(observed - reference) / reference)
    intermediates["distance_from_no_disperse_band"] = (
        0.0 if band[0] <= observed <= band[1]
        else min(abs(observed - band[0]), abs(observed - band[1])))
    intermediates["gap_between_the_two_hypotheses"] = _hypothesis_gap(
        band, reference)

    if capture.padding_rows_included is not True:
        intermediates["region"] = DispersalRegion.NOT_ESTABLISHED.value
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            "the capture does not record that padding rows were routed and "
            "counted, and whether padding rows disperse is the question; both "
            "distances are published above but no region can be assigned",
            intermediates)

    tolerance = (None if tolerance_entry.value is None
                 else float(tolerance_entry.value))
    region = classify_dispersal(observed, band, reference, tolerance)
    intermediates["region"] = region.value

    if region is DispersalRegion.NON_DISPERSAL_BAND:
        return common.check(
            "m4.padding_dispersal", common.Outcome.PASSED,
            f"E(1) = {observed:.2f} lies in the stated band {band}: padding "
            "rows do NOT disperse", intermediates)

    if region is DispersalRegion.DISPERSAL_NEIGHBOURHOOD:
        return common.check(
            "m4.padding_dispersal", common.Outcome.PASSED,
            f"E(1) = {observed:.2f} is within {tolerance:.0%} of {reference}: "
            "padding rows DO disperse", intermediates)

    if region is DispersalRegion.DISPERSAL_PENDING_TOLERANCE:
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            f"E(1) = {observed:.2f} is outside the band {band} and sits in the "
            f"neighbourhood of {reference}, but 'near {reference}' has no "
            "tolerance from any named party and this module does not invent "
            "one. PENDING A TOLERANCE, NOT UNMEASURABLE: both distances are "
            "published and the ruling is outstanding with the architect",
            intermediates)

    return common.check(
        "m4.padding_dispersal", common.Outcome.FAILED,
        f"E(1) = {observed:.2f} is in NEITHER neighbourhood: it misses the "
        f"stated band {band} and is not in the neighbourhood of {reference} "
        "under any tolerance that would still tell the two hypotheses apart. "
        "This is a finding about the design, not a failure of the instrument "
        "-- neither prediction in section 7 holds -- and it is deliberately "
        "not filed under the label that means 'we could not tell'",
        intermediates)


def ladder(captures: Iterable[Capture]) -> List[Dict[str, Any]]:
    """Every ladder point, each carrying its own basis.

    The c1 and c16 points are here so that the Phase 1 naming statistic can be
    computed from them by whoever owns that decision. THIS MODULE DOES NOT
    COMPUTE IT AND DOES NOT NAME A CANDIDATE.
    """
    return [c.ladder_point().to_dict() for c in sorted(captures, key=lambda c: c.concurrency)]


def assess(captures: Sequence[Capture],
           thresholds: common.Thresholds) -> List[common.Check]:
    checks: List[common.Check] = []
    wanted = [int(c) for c in thresholds.require("m4.concurrencies")]
    present = sorted({c.concurrency for c in captures})
    missing = [c for c in wanted if c not in present]
    checks.append(
        common.check(
            "m4.ladder_complete",
            common.Outcome.PASSED if not missing else common.Outcome.UNDETERMINED,
            f"captures present at c{present}" if not missing else
            f"no capture at c{missing}; the ladder is incomplete",
            {"wanted": wanted, "present": present, "missing": missing}))
    for capture in captures:
        if capture.concurrency == 32:
            checks.append(check_byte_model(capture, thresholds))
        if capture.concurrency == 1:
            checks.append(check_padding_dispersal(capture, thresholds))
    return checks


def capture_from_dict(data: Dict[str, Any]) -> Capture:
    return Capture(concurrency=int(data["concurrency"]),
                   shape=data.get("shape", "unspecified"),
                   padding_rows_included=data.get("padding_rows_included"),
                   observations=[RouterObservation(**o) for o in data["observations"]])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Aggregates captures into histograms, ladder points and dispositions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("capture", nargs="+",
                        help="JSON captures, one per concurrency, each holding "
                             "raw router observations")
    args = parser.parse_args(argv)

    thresholds = common.Thresholds.load(args.thresholds)
    captures = [capture_from_dict(common.load_json(p)) for p in args.capture]
    checks = assess(captures, thresholds)
    common.write_artifact(
        args.out,
        kind="M4",
        payload={
            "ladder": ladder(captures),
            "captures": [c.to_dict() for c in captures],
            "not_computed_here": (
                "the c1/c16 residual ratio and the name of the Phase 1 candidate; "
                "both belong to the engineering manager and are gated on M4 having "
                "run"),
        },
        checks=checks,
        thresholds=thresholds)
    print(f"M4: artifact written to {args.out}")
    for item in checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    return 0 if common.worst(c.outcome for c in checks) is common.Outcome.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
