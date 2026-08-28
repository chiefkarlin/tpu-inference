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
but "near" has no tolerance in any document, so the dispersal leg reports
COULD_NOT_BE_CHECKED_MECHANICALLY and publishes the distance from 119 until a
named party supplies one. See ``m4.e1_disperse_tolerance_fraction`` in
``thresholds.json``.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
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


def check_padding_dispersal(capture: Capture,
                            thresholds: common.Thresholds) -> common.Check:
    """Do padding rows disperse across experts at c1?

    The "do not disperse" leg has an exact band from the design. The "disperse"
    leg says only "near 119", and no named party has supplied a tolerance for
    it, so that leg reports UNDETERMINED with the distance published rather than
    borrowing a tolerance that was designed for a different quantity.
    """
    band = list(thresholds.require("m4.e1_no_disperse_band"))
    reference = float(thresholds.require("m4.e1_disperse_reference"))
    observed = capture.summary_e()
    intermediates: Dict[str, Any] = {
        "observed_e": common.counted(observed, ROUTER_COUNTER),
        "no_disperse_band": band,
        "disperse_reference": reference,
        "per_layer_e": {str(k): v for k, v in capture.per_layer_e().items()},
        "padding_rows_included": capture.padding_rows_included,
        "concurrency": capture.concurrency,
    }
    if observed is None:
        return common.check("m4.padding_dispersal", common.Outcome.UNDETERMINED,
                            "the capture is empty", intermediates)
    intermediates["distance_from_disperse_reference"] = observed - reference
    intermediates["relative_distance_from_disperse_reference"] = (
        abs(observed - reference) / reference)
    if capture.padding_rows_included is not True:
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            "the capture does not record that padding rows were routed and "
            "counted, and whether padding rows disperse is the question; "
            "E is published above", intermediates)
    if band[0] <= observed <= band[1]:
        return common.check(
            "m4.padding_dispersal", common.Outcome.PASSED,
            f"E(1) = {observed:.2f} lies in the stated band {band}: padding rows "
            "do NOT disperse", intermediates)
    try:
        tolerance = float(thresholds.require("m4.e1_disperse_tolerance_fraction"))
    except common.ThresholdError as exc:
        intermediates["threshold_status"] = str(exc)
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            f"E(1) = {observed:.2f} is outside the stated band {band}, but 'near "
            f"{reference}' has no tolerance from any named party and this module "
            "does not invent one; the distance is published above", intermediates)
    intermediates["disperse_tolerance_fraction"] = tolerance
    if abs(observed - reference) / reference <= tolerance:
        return common.check(
            "m4.padding_dispersal", common.Outcome.PASSED,
            f"E(1) = {observed:.2f} is within {tolerance:.0%} of {reference}: "
            "padding rows disperse", intermediates)
    return common.check(
        "m4.padding_dispersal", common.Outcome.UNDETERMINED,
        f"E(1) = {observed:.2f} matches neither the band {band} nor "
        f"{reference} +/- {tolerance:.0%}; this is a reportable finding, not a "
        "value to be rounded to the nearer reference", intermediates)


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
