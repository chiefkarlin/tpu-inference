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
"""M2 -- replication of the TPU cell, at c32 AND at c1.

The deployed TPU baseline is n = 1 with no variance estimate, against a
four-replicate H200 arm. M2 replicates the Shape B (isl512/osl256) cell at c32
with at least four replicates, and -- review finding B8 -- extends the same
treatment to c1, which is cheap because c1 runs are short.

WHY c1 MATTERS AND IS NOT A NICETY: without a spread estimate at c1, two later
comparisons have one side with no precision basis at all, including the
criterion that distinguishes "the mechanism acted" from "something moved".

WHAT THIS MODULE PUBLISHES: the per-replicate raw values, always, alongside the
coefficient of variation. A CoV with the replicates withheld cannot be checked,
and this module has no mode that emits one without them.

WHAT IT REFUSES: to call a difference smaller than the measured spread an
effect, or to call it no effect. That case is "not resolvable at this
replication", which is UNDETERMINED -- see :func:`compare_with_spread`. It also
refuses to combine points on different bases; see ``basis.py``.

Nothing here executes a benchmark. :func:`plan_replicate_commands` renders the
argv a human or an authorised harness would run, and :func:`ingest_replicates`
reads the artifacts afterwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from tools.laguna_phase0 import basis as basis_mod
from tools.laguna_phase0 import common


@dataclasses.dataclass(frozen=True)
class Cell:
    """One benchmark cell. A result without its cell is not reportable."""

    shape: str
    isl: int
    osl: int
    concurrency: int

    def label(self) -> str:
        return f"{self.shape} isl{self.isl}/osl{self.osl} c{self.concurrency}"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cell":
        return cls(shape=data["shape"],
                   isl=int(data["isl"]),
                   osl=int(data["osl"]),
                   concurrency=int(data["concurrency"]))


def cells_from_thresholds(thresholds: common.Thresholds) -> List[Cell]:
    return [Cell.from_dict(c) for c in thresholds.require("m2.cells")]


@dataclasses.dataclass(frozen=True)
class Replicate:
    """One run of one cell, and where its number came from."""

    value: float
    units: str
    basis: basis_mod.MeasurementBasis
    artifact: Optional[str] = None
    started_utc: Optional[str] = None
    detail: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["basis"] = self.basis.value
        return out


@dataclasses.dataclass
class SpreadEstimate:
    """The measured spread of a cell. The input other instruments need.

    This is what M3's threshold is derived from and what G5's "no effect smaller
    than the spread" is measured against. It is a MEASUREMENT, not a setting:
    nothing in this package supplies a default for it.
    """

    cell: Cell
    metric: str
    units: str
    basis: basis_mod.MeasurementBasis
    n: int
    mean: Optional[float]
    stdev: Optional[float]
    coefficient_of_variation: Optional[float]
    raw_values: List[float]

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["cell"] = self.cell.to_dict()
        out["basis"] = self.basis.value
        return out


def _stdev(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return (sum((v - mean)**2 for v in values) / (len(values) - 1))**0.5


def summarise_replicates(cell: Cell, metric: str,
                         replicates: Sequence[Replicate]) -> SpreadEstimate:
    """Mean, stdev and CoV -- always shipped with the raw values.

    Raises:
      BasisMismatchError: if the replicates are not all on one declared basis.
    """
    points = [
        basis_mod.LadderPoint(concurrency=cell.concurrency,
                              value=r.value,
                              units=r.units,
                              basis=r.basis,
                              instrument="M2",
                              shape=cell.shape) for r in replicates
    ]
    declared = basis_mod.require_single_basis(points)
    values = [float(r.value) for r in replicates]
    return SpreadEstimate(cell=cell,
                          metric=metric,
                          units=replicates[0].units if replicates else "unknown",
                          basis=declared,
                          n=len(values),
                          mean=(sum(values) / len(values)) if values else None,
                          stdev=_stdev(values),
                          coefficient_of_variation=common.coefficient_of_variation(values),
                          raw_values=values)


def check_replication(estimate: SpreadEstimate,
                      thresholds: common.Thresholds) -> common.Check:
    """Enough replicates for a spread estimate to exist at all?"""
    minimum = int(thresholds.require("m2.min_replicates"))
    intermediates = estimate.to_dict()
    intermediates["min_replicates"] = minimum
    if estimate.n < minimum:
        return common.check(
            "m2.replication", common.Outcome.UNDETERMINED,
            f"{estimate.n} replicates at {estimate.cell.label()}; the treatment "
            f"calls for at least {minimum}, so this cell has no spread estimate "
            "and remains an unreplicated point estimate", intermediates)
    if estimate.coefficient_of_variation is None:
        return common.check(
            "m2.replication", common.Outcome.UNDETERMINED,
            "the coefficient of variation is undefined for these values",
            intermediates)
    return common.check(
        "m2.replication", common.Outcome.PASSED,
        f"{estimate.n} replicates at {estimate.cell.label()}; CoV "
        f"{estimate.coefficient_of_variation:.4f}, raw values published alongside",
        intermediates)


def compare_with_spread(label: str,
                        before: basis_mod.LadderPoint,
                        after: basis_mod.LadderPoint,
                        spread: Optional[SpreadEstimate]) -> common.Check:
    """Emits a difference WITH the spread beside it, and refuses to overclaim.

    G5 of the design: no effect smaller than the M2 spread is claimed. A
    difference inside the spread is "not resolvable at this replication" --
    which is UNDETERMINED. It is not "no effect", and it is certainly not "a
    small effect".
    """
    declared = basis_mod.require_single_basis([before, after])
    difference = (after.value or 0.0) - (before.value or 0.0)
    intermediates: Dict[str, Any] = {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "difference": difference,
        "basis": declared.value,
        "spread": spread.to_dict() if spread else None,
    }
    if spread is None or spread.stdev is None:
        return common.check(
            f"m2.comparison.{label}", common.Outcome.UNDETERMINED,
            "no spread estimate exists for this cell, so this difference has no "
            "precision basis and no criterion may be stated on it", intermediates)
    intermediates["resolution_floor"] = spread.stdev
    if abs(difference) < spread.stdev:
        return common.check(
            f"m2.comparison.{label}", common.Outcome.UNDETERMINED,
            f"difference {difference:+.4f} {before.units} is inside the measured "
            f"spread ({spread.stdev:.4f}); NOT RESOLVABLE AT THIS REPLICATION",
            intermediates)
    return common.check(
        f"m2.comparison.{label}", common.Outcome.PASSED,
        f"difference {difference:+.4f} {before.units} exceeds the measured spread "
        f"({spread.stdev:.4f})", intermediates)


def plan_replicate_commands(cell: Cell, replicates: int,
                            base_argv: Sequence[str]) -> List[Dict[str, Any]]:
    """Renders the runs, without running them.

    Each entry names its cell, its replicate index and the argv to invoke. The
    warm-up protocol (M0) is per pod and per configuration, so the plan also
    carries a reminder of which warm-up identity each replicate must run under:
    replicates that are not all warm under the same identity are not replicates
    of the same thing.
    """
    plan = []
    for index in range(replicates):
        plan.append({
            "cell": cell.to_dict(),
            "replicate_index": index,
            "argv": list(base_argv) + [f"--max-concurrency={cell.concurrency}"],
            "m0_requirement": ("warm this pod for this configuration before the "
                               "timed window; a replicate whose window is VOID is "
                               "discarded, not adjusted"),
        })
    return plan


def ingest_replicates(paths: Sequence[str], metric_key: str, units: str,
                      declared_basis: str) -> List[Replicate]:
    """Reads per-replicate results, requiring the basis to be declared.

    The basis is a parameter with no default on purpose. An ingest that guesses
    it reintroduces exactly the defect ``basis.py`` exists to prevent.
    """
    parsed = basis_mod.MeasurementBasis(declared_basis)
    out: List[Replicate] = []
    for path in paths:
        data = common.load_json(path)
        node: Any = data
        for part in metric_key.split("."):
            node = node[part]
        out.append(
            Replicate(value=float(node),
                      units=units,
                      basis=parsed,
                      artifact=str(path),
                      detail={"metric_key": metric_key}))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    plan_cmd = sub.add_parser("plan", help="render the replicate runs, unexecuted")
    plan_cmd.add_argument("--out", required=True)
    plan_cmd.add_argument("--thresholds", default=None)
    plan_cmd.add_argument("--replicates", type=int, default=None,
                          help="defaults to the minimum in thresholds.json")
    plan_cmd.add_argument("base_argv", nargs=argparse.REMAINDER,
                          help="the benchmark invocation, after --")

    sum_cmd = sub.add_parser("summarise", help="CoV plus every raw value")
    sum_cmd.add_argument("--out", required=True)
    sum_cmd.add_argument("--thresholds", default=None)
    sum_cmd.add_argument("--cell-concurrency", type=int, required=True)
    sum_cmd.add_argument("--metric", required=True,
                         help="name of the metric, for the record")
    sum_cmd.add_argument("--metric-key", required=True,
                         help="dotted key of the metric inside each result file")
    sum_cmd.add_argument("--units", required=True)
    sum_cmd.add_argument("--basis", required=True,
                         choices=[b.value for b in basis_mod.MeasurementBasis
                                  if b is not basis_mod.MeasurementBasis.UNKNOWN],
                         help="declared on the command line; never inferred")
    sum_cmd.add_argument("result", nargs="+")

    args = parser.parse_args(argv)
    thresholds = common.Thresholds.load(args.thresholds)

    try:
        cells = cells_from_thresholds(thresholds)
        if args.mode == "plan":
            replicates = args.replicates or int(thresholds.require("m2.min_replicates"))
            base_argv = [a for a in (args.base_argv or []) if a != "--"]
            payload = {
                "replicates_per_cell": replicates,
                "cells": [c.to_dict() for c in cells],
                "runs": [run for cell in cells
                         for run in plan_replicate_commands(cell, replicates, base_argv)],
            }
            common.write_artifact(args.out, kind="M2-plan", payload=payload,
                                  thresholds=thresholds)
            print(f"M2: plan for {len(payload['runs'])} runs written to {args.out}")
            return 0

        cell = next((c for c in cells if c.concurrency == args.cell_concurrency), None)
        if cell is None:
            print(f"M2: c{args.cell_concurrency} is not one of the M2 cells "
                  f"({[c.label() for c in cells]})", file=sys.stderr)
            return 2
        replicates = ingest_replicates(args.result, args.metric_key, args.units,
                                       args.basis)
        estimate = summarise_replicates(cell, args.metric, replicates)
        chk = check_replication(estimate, thresholds)
    except (common.ThresholdError, basis_mod.BasisMismatchError, KeyError) as exc:
        print(f"M2: {exc}", file=sys.stderr)
        return 2

    common.write_artifact(args.out,
                          kind="M2",
                          payload={"spread_estimate": estimate.to_dict()},
                          checks=[chk],
                          thresholds=thresholds)
    print(f"M2: {chk.outcome.value} -- {chk.reason}")
    print(f"    raw values: {json.dumps(estimate.raw_values)}")
    return 0 if chk.outcome is common.Outcome.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
