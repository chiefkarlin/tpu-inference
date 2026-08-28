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
"""M0 -- the warm-cache protocol and the in-window recompilation assertion.

Every timed window is preceded by a warm-up pass over the SAME padding buckets
the timed window will use, ``VLLM_XLA_CHECK_RECOMPILATION=1`` is set, and the
window must complete with ZERO recompilation events.

A RUN THAT RECOMPILES IN-WINDOW IS VOID, NOT ADJUSTED. There is no adjusted path
in this module. Nothing subtracts the compile time, annotates it, or carries the
window forward with a caveat, because a correction that makes the symptom go away
is indistinguishable from the check having stopped being checked.

WARMTH IS PER POD **AND PER CONFIGURATION**. The compile cache is rooted on an
emptyDir, so it does not survive a pod restart; and a candidate configuration
change alters the compiled graph set, so a pod warmed under one configuration is
not warm under the next. Both facts are carried in the evidence as a warmth
identity, and a window whose identity differs from the warm-up that preceded it
is void on that ground alone.

WARM-UP EVIDENCE IS A DELIVERABLE, NOT A CLAIM. Every window emits which buckets
were warmed and when, the warmth identity, the environment the protocol depends
on, and the recompilation counter's reading at window start and at window end, so
that a third party can see the window was warm without taking anyone's word for
it.

The negative control (review finding B7-ii) leaves a bucket deliberately unwarmed
so that the assertion can be made to fire. IT HAS NOT BEEN EXECUTED: inducing a
real recompilation needs a real pod and a real compile cache.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import json
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from tools.laguna_phase0 import common

# Environment the protocol depends on. Recorded into every window's evidence.
RECORDED_ENV = (
    "VLLM_XLA_CHECK_RECOMPILATION",
    "VLLM_XLA_CACHE_PATH",
    "JAX_COMPILATION_CACHE_DIR",
    "POD_NAME",
    "POD_UID",
    "HOSTNAME",
)


class WindowVerdict(str, enum.Enum):
    """A timed window is valid, void, or not assessable. Never adjusted."""

    VALID = "VALID"
    VOID = "VOID"
    UNDETERMINED = "COULD_NOT_BE_CHECKED_MECHANICALLY"


@dataclasses.dataclass(frozen=True, order=True)
class Bucket:
    """One padding bucket: the shape a compiled graph is keyed on.

    Attributes:
      kind: What is being padded, e.g. ``"decode_tokens"`` or ``"num_reqs"``.
      size: The padded size.
    """

    kind: str
    size: int

    def key(self) -> str:
        return f"{self.kind}={self.size}"

    @classmethod
    def parse(cls, text: str) -> "Bucket":
        kind, _, size = text.partition("=")
        if not size:
            raise ValueError(f"bucket {text!r} must be written kind=size")
        return cls(kind=kind, size=int(size))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def warmth_identity(pod_id: str, configuration: Any) -> Dict[str, str]:
    """Identity a warm cache is valid for: this pod, this configuration.

    The configuration is fingerprinted rather than stored so that two windows
    can be compared without a human deciding which differences matter. Any
    change to the launch arguments changes the fingerprint, which is the
    intended sensitivity: a configuration change alters the compiled graph set.
    """
    blob = json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")
    return {
        "pod_id": pod_id,
        "configuration_fingerprint": hashlib.sha256(blob).hexdigest(),
    }


def pod_identity() -> str:
    """Best available per-pod identity. Warmth does not cross a pod restart."""
    for name in ("POD_UID", "POD_NAME", "HOSTNAME"):
        value = os.environ.get(name)
        if value:
            return f"{name}:{value}"
    return "unknown-pod"


@dataclasses.dataclass
class CounterReading:
    """One reading of the recompilation counter, with where it came from.

    ``value is None`` means the counter could not be read. That is UNDETERMINED
    downstream; it is never treated as zero events, because "I saw no evidence"
    and "I have evidence of none" are different statements and only one of them
    supports a claim.
    """

    value: Optional[int]
    source: str
    units: str = "recompilation events"
    detail: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class RecompilationProbe:
    """Reads the recompilation event count.

    Two sources, in order of preference:

    * an in-process counter supplied by the harness (a callable returning an
      int), which is exact;
    * a scan of the server log for patterns supplied in ``thresholds.json``.

    The log patterns are an INSTRUMENT BINDING and they are unconfirmed: no log
    from a warmed v7x pod has been read by the author of this module. If neither
    source is available the probe returns ``None`` and the window is
    UNDETERMINED.
    """

    def __init__(self,
                 counter: Optional[Callable[[], int]] = None,
                 log_path: Optional[str] = None,
                 patterns: Optional[Sequence[str]] = None):
        self._counter = counter
        self._log_path = log_path
        self._patterns = [re.compile(p) for p in (patterns or [])]
        self._pattern_source = list(patterns or [])

    def read(self) -> CounterReading:
        if self._counter is not None:
            return CounterReading(value=int(self._counter()),
                                  source="in-process counter supplied by the harness")
        if self._log_path and self._patterns:
            try:
                with open(self._log_path, "r", encoding="utf-8", errors="replace") as handle:
                    hits = sum(1 for line in handle
                               if any(p.search(line) for p in self._patterns))
            except OSError as exc:
                return CounterReading(value=None,
                                      source="server log scan",
                                      detail={"error": str(exc),
                                              "log_path": self._log_path})
            return CounterReading(value=hits,
                                  source="server log scan",
                                  detail={"log_path": self._log_path,
                                          "patterns": self._pattern_source,
                                          "patterns_confirmed_against_a_real_log": False})
        return CounterReading(
            value=None,
            source="none available",
            detail={
                "why": ("no in-process counter was supplied and no confirmed log "
                        "pattern is configured; see thresholds.json "
                        "m0.recompilation_event_log_patterns")
            })


@dataclasses.dataclass
class WarmupPlan:
    """Which buckets to warm before a window, and any deliberate omission.

    Attributes:
      window_buckets: The buckets the timed window will use.
      leave_unwarmed: NEGATIVE CONTROL. Buckets the window will use that are
        deliberately not warmed, so that the assertion can be made to fire.
    """

    window_buckets: List[Bucket]
    leave_unwarmed: List[Bucket] = dataclasses.field(default_factory=list)

    def buckets_to_warm(self) -> List[Bucket]:
        omitted = set(self.leave_unwarmed)
        return [b for b in self.window_buckets if b not in omitted]

    def negative_control(self, *,
                         executed: bool = False) -> Optional[common.NegativeControl]:
        """The control descriptor. ``executed`` says whether it actually RAN.

        It used to be hardcoded ``False`` here, which made the field incapable
        of ever recording a fired control -- and "has this control ever fired?"
        is exactly the question round 1 R9/R11 turn on. A plan is rendered
        unexecuted; evidence evaluated from a real corrupted run is not.
        """
        if not self.leave_unwarmed:
            return None
        return common.NegativeControl(
            name="m0-unwarmed-bucket",
            description=("deliberately skips warm-up for "
                         f"{[b.key() for b in self.leave_unwarmed]} while the timed "
                         "window still uses them"),
            expected_effect=("the window compiles in-flight, the recompilation counter "
                             "advances, and the window is marked VOID"),
            executed=executed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_buckets": [b.to_dict() for b in self.window_buckets],
            "leave_unwarmed": [b.to_dict() for b in self.leave_unwarmed],
            "buckets_to_warm": [b.to_dict() for b in self.buckets_to_warm()],
        }


@dataclasses.dataclass
class WarmupEvidence:
    """Everything a third party needs in order to check the window was warm."""

    identity: Dict[str, str]
    plan: WarmupPlan
    warmed: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    window_start_counter: Optional[CounterReading] = None
    window_end_counter: Optional[CounterReading] = None
    window_started_utc: Optional[str] = None
    window_ended_utc: Optional[str] = None
    environment: Dict[str, Any] = dataclasses.field(default_factory=dict)
    absent_keys: List[str] = dataclasses.field(default_factory=list)
    """Keys that were MISSING from the source document, when there was one.

    Only :func:`evidence_from_dict` can populate this. An object built by
    :class:`WarmupLedger` during a real run has every field by construction, so
    its list is empty -- and empty here means "nothing was missing", never "we
    did not look".

    It exists because ``dict.get(key, default)`` destroys the difference
    between "the harness said zero" and "the harness did not say", and
    :func:`evaluate_window` needs that difference to decide whether it is
    looking at a measurement or at a hole. The list is published in the check's
    intermediates so a reader can see which distinction was drawn and why.
    """

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmth_identity": self.identity,
            "plan": self.plan.to_dict(),
            "warmed_buckets": self.warmed,
            "window_started_utc": self.window_started_utc,
            "window_ended_utc": self.window_ended_utc,
            "recompilation_counter_at_window_start":
                self.window_start_counter.to_dict() if self.window_start_counter else None,
            "recompilation_counter_at_window_end":
                self.window_end_counter.to_dict() if self.window_end_counter else None,
            "environment": self.environment,
        }


class WarmupLedger:
    """Runs the protocol around one timed window and records the evidence.

    The ledger does not time anything and does not decide anything; it records.
    The harness calls :meth:`warm` once per bucket with a callable that issues
    the warm-up pass, then wraps its timed window in :meth:`window`.
    """

    def __init__(self,
                 plan: WarmupPlan,
                 probe: RecompilationProbe,
                 configuration: Any,
                 pod_id: Optional[str] = None):
        self.plan = plan
        self.probe = probe
        self.evidence = WarmupEvidence(
            identity=warmth_identity(pod_id or pod_identity(), configuration),
            plan=plan,
            environment=common.env_snapshot(RECORDED_ENV))

    def warm(self, bucket: Bucket, warm_pass: Callable[[Bucket], Any]) -> Any:
        """Issues one warm-up pass and records that it happened, and when."""
        started = common.utc_now_iso()
        result = warm_pass(bucket)
        self.evidence.warmed.append({
            "bucket": bucket.to_dict(),
            "started_utc": started,
            "finished_utc": common.utc_now_iso(),
        })
        return result

    def warm_all(self, warm_pass: Callable[[Bucket], Any]) -> None:
        for bucket in self.plan.buckets_to_warm():
            self.warm(bucket, warm_pass)

    def open_window(self) -> None:
        self.evidence.window_started_utc = common.utc_now_iso()
        self.evidence.window_start_counter = self.probe.read()

    def close_window(self) -> None:
        self.evidence.window_ended_utc = common.utc_now_iso()
        self.evidence.window_end_counter = self.probe.read()

    def __enter__(self) -> "WarmupLedger":
        self.open_window()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close_window()
        return False


def evaluate_window(evidence: WarmupEvidence,
                    thresholds: common.Thresholds) -> List[common.Check]:
    """Turns warm-up evidence into dispositions. Void, valid, or unassessable.

    Three independent checks, all of which have to hold: the required
    environment was set, every bucket the window uses was warmed under the same
    warmth identity, and the recompilation counter did not advance.
    """
    allowed = int(thresholds.require("m0.recompilation_events_allowed_in_window"))
    required_env = dict(thresholds.require("m0.required_env"))
    checks: List[common.Check] = []

    env_seen = {k: evidence.environment.get(k) for k in required_env}
    env_bad = {k: (env_seen.get(k), v) for k, v in required_env.items()
               if str(env_seen.get(k)) != str(v)}
    checks.append(
        common.check(
            "m0.required_env",
            common.Outcome.FAILED if env_bad else common.Outcome.PASSED,
            ("the protocol's required environment was not set: "
             f"{env_bad}") if env_bad else "the required environment was set",
            {"required": required_env, "observed": env_seen}))

    warmed = {Bucket(**w["bucket"]) for w in evidence.warmed}
    missing = sorted(b.key() for b in evidence.plan.window_buckets if b not in warmed)
    warmth_intermediates = {
        "window_buckets": [b.key() for b in evidence.plan.window_buckets],
        "warmed_buckets": sorted(b.key() for b in warmed),
        "unwarmed_buckets_in_window": missing,
        "warmth_identity": evidence.identity,
        "absent_keys": list(evidence.absent_keys),
    }

    # THE VACUOUS PASS, AND THE TWO WAYS TO REACH IT. "Every bucket the window
    # uses was warmed" is trivially true when the window is not known to use
    # any bucket, and a trivial truth is not a measurement. Reaching it by a
    # missing key and reaching it by an empty list are different facts about
    # the harness, so they get different reasons, but neither is a pass.
    undecidable = None
    if "plan" in evidence.absent_keys:
        undecidable = ("the evidence document carries no plan at all, so there is no "
                       "statement of which buckets the timed window uses. This is not "
                       "a window that uses no buckets; it is a document that does not "
                       "say, and it cannot be read as warm")
    elif "plan.window_buckets" in evidence.absent_keys:
        undecidable = ("the plan does not list the buckets the timed window uses, so "
                       "there is nothing to check warmth against. Absent is not empty")
    elif not evidence.plan.window_buckets:
        undecidable = ("the plan declares that the timed window uses no buckets at "
                       "all. Nothing was checked, because there is nothing this check "
                       "could be about -- which is a statement about the plan, not a "
                       "clean warm-up")
    elif "warmed_buckets" in evidence.absent_keys:
        undecidable = ("the evidence document does not record what was warmed. That "
                       "is not the same as recording that nothing was warmed, and it "
                       "must not be reported as an unwarmed window")

    if undecidable is not None:
        checks.append(
            common.check("m0.buckets_warmed", common.Outcome.UNDETERMINED,
                         undecidable, warmth_intermediates))
    else:
        checks.append(
            common.check(
                "m0.buckets_warmed",
                common.Outcome.FAILED if missing else common.Outcome.PASSED,
                (f"the window uses buckets that were never warmed: {missing}")
                if missing else "every bucket the window uses was warmed first",
                warmth_intermediates))

    start = evidence.window_start_counter
    end = evidence.window_end_counter
    counter_intermediates = {
        "at_window_start": start.to_dict() if start else None,
        "at_window_end": end.to_dict() if end else None,
        "events_allowed_in_window": allowed,
    }
    if start is None or end is None or start.value is None or end.value is None:
        checks.append(
            common.check(
                "m0.no_in_window_recompilation", common.Outcome.UNDETERMINED,
                "the recompilation counter could not be read at both ends of the "
                "window; this is not zero events, it is no evidence",
                counter_intermediates))
    else:
        events = end.value - start.value
        counter_intermediates["events_in_window"] = events
        checks.append(
            common.check(
                "m0.no_in_window_recompilation",
                common.Outcome.PASSED if events <= allowed else common.Outcome.FAILED,
                (f"{events} recompilation events inside the timed window; the run is "
                 "VOID, not adjusted") if events > allowed else
                f"{events} recompilation events inside the timed window",
                counter_intermediates))
    return checks


def window_verdict(checks: Sequence[common.Check]) -> WindowVerdict:
    """VOID on any failure, UNDETERMINED when something could not be checked.

    Deliberately total: there is no fourth outcome in which a window is kept
    with a caveat.
    """
    outcome = common.worst(c.outcome for c in checks)
    if outcome is common.Outcome.PASSED:
        return WindowVerdict.VALID
    if outcome is common.Outcome.FAILED:
        return WindowVerdict.VOID
    return WindowVerdict.UNDETERMINED


def evidence_from_dict(data: Dict[str, Any]) -> WarmupEvidence:
    """Rebuilds evidence written by a harness, for offline evaluation.

    RECORDS WHAT WAS MISSING INSTEAD OF FILLING IT IN. The defaults below are
    still applied, because the rest of the module wants real lists to iterate;
    what changed is that supplying a default is no longer silent. Every key
    that had to be defaulted is named in :attr:`WarmupEvidence.absent_keys`,
    and :func:`evaluate_window` refuses to draw a conclusion from a defaulted
    one.

    The defect this closes: ``data.get("plan", {})`` turned an evidence
    document with no plan into a plan with no buckets, and a window that uses
    no buckets has no unwarmed buckets, so the warmth check reported PASSED
    with the words "every bucket the window uses was warmed first". A document
    that said nothing about warming was being read as proof of warming.

    Note the two directions, because only one of them is tempting to leave
    alone. Absent ``plan`` flattered the run; absent ``warmed_buckets``
    condemned it, by reading "the document does not say what was warmed" as
    "nothing was warmed". BOTH ARE THE SAME BUG and both are fixed here. A
    check that is wrong in the safe direction is still wrong, and it sends an
    operator hunting a cache fault that never happened.

    This function is NOT a validator and does not decide anything. It parses,
    and it reports what it could not find. Deciding is
    :func:`evaluate_window`'s job, which is why a missing key produces an
    UNDETERMINED verdict rather than an exception: the file is readable, the
    instrument ran, and "the evidence does not decide" is a result. Exit status
    2 stays reserved for could-not-run, per :func:`main`.
    """
    absent: List[str] = []
    if "plan" not in data:
        absent.append("plan")
    plan_data = data.get("plan", {}) or {}
    if "plan" not in absent and "window_buckets" not in plan_data:
        absent.append("plan.window_buckets")
    if "warmed_buckets" not in data:
        absent.append("warmed_buckets")

    plan = WarmupPlan(
        window_buckets=[Bucket(**b) for b in plan_data.get("window_buckets", [])],
        leave_unwarmed=[Bucket(**b) for b in plan_data.get("leave_unwarmed", [])])

    def reading(node: Optional[Dict[str, Any]]) -> Optional[CounterReading]:
        if node is None:
            return None
        return CounterReading(value=node.get("value"),
                              source=node.get("source", "unknown"),
                              units=node.get("units", "recompilation events"),
                              detail=node.get("detail", {}))

    return WarmupEvidence(
        identity=data.get("warmth_identity", {}),
        plan=plan,
        warmed=data.get("warmed_buckets", []),
        window_start_counter=reading(data.get("recompilation_counter_at_window_start")),
        window_end_counter=reading(data.get("recompilation_counter_at_window_end")),
        window_started_utc=data.get("window_started_utc"),
        window_ended_utc=data.get("window_ended_utc"),
        environment=data.get("environment", {}),
        absent_keys=absent)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Two offline modes: emit a warm-up plan, or evaluate a window's evidence.

    Neither mode runs a model. The warm-up pass itself is issued by the harness
    that owns the engine, through :class:`WarmupLedger`.

    EXIT STATUS. Three states, never two:

    * ``0`` -- ran, and the window is VALID.
    * ``1`` -- ran, and the window is not VALID (VOID or UNDETERMINED). A
      verdict was reached and it was not a pass.
    * ``2`` -- COULD NOT RUN. Thresholds unreadable, evidence file missing or
      not JSON. Nothing was assessed and no verdict exists.

    An operator who cannot tell "never ran" from "ran and stopped the run" has
    no instrument, so ``2`` is reserved for the former and is never a verdict.
    Note that UNDETERMINED is a verdict -- the instrument ran and reported that
    the evidence does not decide -- which is why it is ``1`` and not ``2``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    plan_cmd = sub.add_parser("plan", help="emit the warm-up plan for a window")
    plan_cmd.add_argument("--bucket", action="append", required=True,
                          help="a padding bucket the timed window will use, kind=size")
    plan_cmd.add_argument("--nc-leave-bucket-unwarmed", action="append", default=[],
                          help="NEGATIVE CONTROL: a bucket to leave unwarmed so the "
                               "in-window assertion can be made to fire")
    plan_cmd.add_argument("--out", required=True)

    eval_cmd = sub.add_parser("evaluate", help="evaluate a window's warm-up evidence")
    eval_cmd.add_argument("--evidence", required=True)
    eval_cmd.add_argument("--out", required=True)
    eval_cmd.add_argument("--thresholds", default=None)

    args = parser.parse_args(argv)

    if args.mode == "plan":
        plan = WarmupPlan(
            window_buckets=[Bucket.parse(b) for b in args.bucket],
            leave_unwarmed=[Bucket.parse(b) for b in args.nc_leave_bucket_unwarmed])
        common.write_artifact(args.out,
                              kind="M0-plan",
                              payload=plan.to_dict(),
                              negative_control=plan.negative_control())
        print(f"M0: warm-up plan written to {args.out}")
        return 0

    # P4-class, M0 instance. Both of these used to escape as tracebacks, and an
    # uncaught exception leaves Python with exit status 1 -- the same status a
    # VOID window leaves. "The evidence file is not there" and "the window is
    # void" are not the same fact and must not be the same code.
    try:
        thresholds = common.Thresholds.load(args.thresholds)
    except (OSError, ValueError, common.ThresholdError) as exc:
        print(f"M0: could not load thresholds: {exc}", file=sys.stderr)
        return 2
    try:
        evidence = evidence_from_dict(common.load_json(args.evidence))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"M0: could not read evidence {args.evidence!r}: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        checks = evaluate_window(evidence, thresholds)
    except common.ThresholdError as exc:
        print(f"M0: {exc}", file=sys.stderr)
        return 2
    verdict = window_verdict(checks)
    payload = dict(evidence.to_dict())
    payload["window_verdict"] = verdict.value
    common.write_artifact(args.out,
                          kind="M0",
                          payload=payload,
                          checks=checks,
                          thresholds=thresholds,
                          # This path evaluates evidence from a run that really
                          # happened, so a control present here DID fire.
                          negative_control=evidence.plan.negative_control(
                              executed=True))
    print(f"M0: window is {verdict.value}; artifact written to {args.out}")
    for item in checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    return 0 if verdict is WindowVerdict.VALID else 1


if __name__ == "__main__":
    raise SystemExit(main())
