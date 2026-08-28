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
"""M3 -- the decode-only microbenchmark.

Fixed batch, pre-filled KV, NO HTTP AND NO PREFILL INSIDE THE TIMED WINDOW. The
point is to separate engine step time from serving-stack overhead: the
difference between this step time and the served TPOT is the serving stack.

THE THRESHOLD IS AN INPUT AND THIS MODULE HAS NO DEFAULT FOR IT. The design
states "if that difference is < 1 ms, the residual is inside the engine". Review
finding B8 rules that out: a 1 ms threshold on a TPOT-derived quantity, with no
spread estimate anywhere, violates M2's own rule. So the difference is emitted as
an intermediate and the verdict is computed from a threshold derived from M2's
measured spread. With no threshold supplied the outcome is
COULD_NOT_BE_CHECKED_MECHANICALLY. The author of this module did not pick a
number, and ``thresholds.json`` holds the design's 1 ms at null, kind
``rejected``, so that no code path can read it.

TWO PROPERTIES THIS MODULE ENFORCES RATHER THAN ASSUMES:

* **Synchronisation.** JAX dispatch is asynchronous, so a step loop timed
  without a device sync measures dispatch and not execution -- and it measures
  it as impossibly fast, which is the flattering direction. A sync callable is
  required; without one the result is UNDETERMINED rather than fast.
* **The comparison crosses bases on purpose.** Engine-step wall clock against
  served TPOT is the measurement, not a mistake, and it is recorded as a
  declared crossing so a reader can tell it apart from review finding B3.

THE ENGINE BINDING IS A NAMED SEAM. This module owns the protocol, the timing,
the statistics and the verdict; the caller supplies a callable that advances one
decode step. The in-tree binding intended for it is
``TPUModelRunner.execute_model`` driven from a decode-only scheduler output with
``num_computed_tokens`` already at the context length, but that call sequence has
never been exercised by the author, so it is supplied at the call site rather
than guessed at here. See ``README.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.laguna_phase0 import basis as basis_mod
from tools.laguna_phase0 import common
from tools.laguna_phase0 import m2_replication as m2


class SynchronisationMissingError(RuntimeError):
    """A step loop was about to be timed without a device sync."""


@dataclasses.dataclass
class MicrobenchConfig:
    """The declared shape of the microbenchmark. Recorded into the artifact.

    Attributes:
      batch_size: Fixed batch, held constant for every step.
      context_length: Tokens already in the KV cache per sequence.
      steps: Timed decode steps.
      warmup_steps: Steps issued before the window opens. Separate from M0's
        bucket warm-up, which is about compilation; these absorb first-call
        effects inside an already-warm process.
      kv_prefilled: Whether the KV cache was filled by a real prefill before the
        window. False is allowed and recorded, but it makes the result
        UNDETERMINED: MoE routing depends on the hidden states, so uninitialised
        KV can route differently from a real workload and the direction of that
        bias is unknown.
      http_in_window: Must be False. Present so the artifact states it.
      prefill_in_window: Must be False. Present so the artifact states it.
    """

    batch_size: int
    context_length: int
    steps: int
    warmup_steps: int = 0
    kv_prefilled: bool = True
    http_in_window: bool = False
    prefill_in_window: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class StepTimings:
    """Per-step wall-clock times, published raw."""

    step_ms: List[float]
    sync_source: str

    def to_dict(self) -> Dict[str, Any]:
        summary = common.summarise(self.step_ms)
        summary["units"] = "ms"
        summary["counter_source"] = (
            "time.perf_counter around one engine step, device-synchronised by "
            f"{self.sync_source}")
        summary["median"] = statistics.median(self.step_ms) if self.step_ms else None
        return summary


def run_step_loop(step: Callable[[], Any],
                  sync: Optional[Callable[[Any], Any]],
                  config: MicrobenchConfig,
                  sync_source: str = "caller-supplied sync callable") -> StepTimings:
    """Times ``config.steps`` decode steps. Nothing else happens in the window.

    Args:
      step: Advances exactly one decode step and returns whatever the engine
        returns. It must not issue a prefill and must not touch HTTP.
      sync: Blocks until the returned work has actually executed on device.
      config: The declared shape, recorded into the artifact.
      sync_source: What the sync callable is, for the counter provenance.

    Raises:
      SynchronisationMissingError: if ``sync`` is None.
    """
    if sync is None:
        raise SynchronisationMissingError(
            "a decode step loop timed without a device sync measures dispatch, "
            "not execution, and it does so in the flattering direction")
    for _ in range(config.warmup_steps):
        sync(step())
    times: List[float] = []
    for _ in range(config.steps):
        started = time.perf_counter()
        sync(step())
        times.append((time.perf_counter() - started) * 1e3)
    return StepTimings(step_ms=times, sync_source=sync_source)


def timings_from_recorded(step_ms: Sequence[float], sync_source: str) -> StepTimings:
    """Rebuilds timings emitted by a harness elsewhere, for offline assessment."""
    return StepTimings(step_ms=[float(v) for v in step_ms], sync_source=sync_source)


def serving_stack_cost(step_ms: float, served_tpot_ms: float,
                       concurrency: int, shape: str) -> Dict[str, Any]:
    """The intermediate: served TPOT minus engine step time, with both sides.

    Emitted whether or not a threshold exists to judge it against, because the
    quantity is the deliverable and the verdict is downstream of it.
    """
    micro_point = basis_mod.LadderPoint(
        concurrency=concurrency, value=step_ms, units="ms",
        basis=basis_mod.MeasurementBasis.ENGINE_STEP_WALL_CLOCK,
        instrument="M3", shape=shape)
    served_point = basis_mod.LadderPoint(
        concurrency=concurrency, value=served_tpot_ms, units="ms",
        basis=basis_mod.MeasurementBasis.TPOT_DERIVED,
        instrument="serving benchmark", shape=shape)
    crossing = basis_mod.declare_cross_basis(
        micro_point, served_point,
        "the serving-stack cost IS the difference between an engine-step wall "
        "clock and a served TPOT; the crossing is the measurement")
    return {
        "engine_step": micro_point.to_dict(),
        "served_tpot": served_point.to_dict(),
        "difference_ms": served_tpot_ms - step_ms,
        "basis": crossing,
    }


def assess(timings: StepTimings,
           config: MicrobenchConfig,
           served_tpot_ms: Optional[float],
           concurrency: int,
           shape: str,
           thresholds: common.Thresholds,
           spread: Optional[m2.SpreadEstimate] = None) -> List[common.Check]:
    """Dispositions for one microbenchmark run.

    The threshold is read from ``thresholds.json`` and is null there by
    construction, so the ordinary outcome of this function today is
    UNDETERMINED with the difference published. That is the intended state: the
    number has to come from M2's measured spread at the same cell, and until M2
    has run there is nothing legitimate to compare against.
    """
    checks: List[common.Check] = []
    step_summary = timings.to_dict()

    checks.append(
        common.check(
            "m3.window_contained_no_http_or_prefill",
            common.Outcome.PASSED if not (config.http_in_window or
                                          config.prefill_in_window)
            else common.Outcome.FAILED,
            "the timed window declares no HTTP and no prefill"
            if not (config.http_in_window or config.prefill_in_window) else
            "the timed window declares HTTP or prefill inside it",
            {"config": config.to_dict()}))

    checks.append(
        common.check(
            "m3.kv_prefilled",
            common.Outcome.PASSED if config.kv_prefilled else common.Outcome.UNDETERMINED,
            "the KV cache was filled by a real prefill before the window"
            if config.kv_prefilled else
            "the KV cache was not filled by a real prefill; MoE routing depends "
            "on the hidden states, so the expert distribution -- and with it the "
            "grouped-matmul time -- may differ from a real workload. Direction of "
            "the bias: UNKNOWN.",
            {"config": config.to_dict()}))

    if served_tpot_ms is None or not timings.step_ms:
        checks.append(
            common.check(
                "m3.serving_stack_cost", common.Outcome.UNDETERMINED,
                "no served TPOT was supplied, or no steps were timed; an absent "
                "input is UNDETERMINED, never a pass",
                {"step_time_summary": step_summary,
                 "served_tpot_ms": served_tpot_ms}))
        return checks

    median_step = statistics.median(timings.step_ms)
    intermediates: Dict[str, Any] = {
        "step_time_summary": step_summary,
        "serving_stack_cost": serving_stack_cost(median_step, served_tpot_ms,
                                                 concurrency, shape),
        "m2_spread": spread.to_dict() if spread else None,
    }
    try:
        threshold_ms = float(thresholds.require("m3.serving_stack_cost_threshold_ms"))
    except common.ThresholdError as exc:
        intermediates["threshold_status"] = str(exc)
        checks.append(
            common.check(
                "m3.serving_stack_cost", common.Outcome.UNDETERMINED,
                "the difference is published above; no threshold has been derived "
                "from M2's measured spread, and this module does not invent one",
                intermediates))
        return checks

    difference = intermediates["serving_stack_cost"]["difference_ms"]
    intermediates["threshold_ms"] = threshold_ms
    if spread is not None and spread.stdev is not None and threshold_ms < spread.stdev:
        checks.append(
            common.check(
                "m3.serving_stack_cost", common.Outcome.UNDETERMINED,
                f"the supplied threshold ({threshold_ms} ms) is finer than the "
                f"measured spread ({spread.stdev} ms) at this cell; M2's rule "
                "forbids stating a criterion there", intermediates))
        return checks
    inside = abs(difference) < threshold_ms
    checks.append(
        common.check(
            "m3.serving_stack_cost", common.Outcome.PASSED,
            f"serving-stack cost {difference:+.4f} ms against a threshold of "
            f"{threshold_ms} ms: the residual is "
            f"{'inside the engine' if inside else 'not inside the engine'}",
            intermediates))
    return checks


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Offline assessment of an already-emitted step-time series.

    The timed loop itself is driven by the harness that owns the engine, through
    :func:`run_step_loop`. This entry point exists so the assessment can be
    re-run when M2 lands and supplies a threshold, without re-running anything
    on hardware.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--step-times", required=True,
                        help="JSON file holding a list of per-step times in ms")
    parser.add_argument("--sync-source", required=True,
                        help="what synchronised each step, for the record")
    parser.add_argument("--served-tpot-ms", type=float, default=None)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--shape", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--kv-not-prefilled", action="store_true",
                        help="declare that the KV cache was not filled by a real "
                             "prefill; makes the result UNDETERMINED")
    args = parser.parse_args(argv)

    thresholds = common.Thresholds.load(args.thresholds)
    step_ms = common.load_json(args.step_times)
    timings = timings_from_recorded(step_ms, args.sync_source)
    config = MicrobenchConfig(batch_size=args.batch_size,
                              context_length=args.context_length,
                              steps=len(timings.step_ms),
                              kv_prefilled=not args.kv_not_prefilled)
    checks = assess(timings, config, args.served_tpot_ms, args.concurrency,
                    args.shape, thresholds)
    common.write_artifact(args.out,
                          kind="M3",
                          payload={"config": config.to_dict(),
                                   "step_times": timings.to_dict()},
                          checks=checks,
                          thresholds=thresholds)
    print(f"M3: artifact written to {args.out}")
    for item in checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    return 0 if common.worst(c.outcome for c in checks) is common.Outcome.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
