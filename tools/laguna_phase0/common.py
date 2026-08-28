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
"""Shared primitives for the Phase 0 instruments.

Three standing rules of this campaign are enforced here, in code, rather than
left to each caller to remember:

1.  **Three buckets, never two.** :class:`Outcome` is PASSED, FAILED or
    COULD_NOT_BE_CHECKED_MECHANICALLY. An empty input is UNDETERMINED, never
    PASS. There is no boolean anywhere in this package that stands in for a
    verdict.
2.  **Publish intermediates, not verdicts.** :func:`check` refuses to build a
    result that does not carry the quantities it was computed from, and
    :func:`write_artifact` refuses to write one. A verdict nobody can recompute
    is not a measurement.
3.  **Thresholds are inputs with provenance.** :class:`Thresholds` reads them
    from a JSON file in which every leaf carries ``value``, ``kind`` and
    ``source``. A threshold that is absent, or present with a null value, is an
    error at the point of use -- it is never silently defaulted. Every artifact
    written through :func:`write_artifact` embeds the provenance of every
    threshold that was read to produce it.

Nothing in this module executes a benchmark, a server or a cluster call.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import math
import os
import pathlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA = "laguna-phase0/1"

DEFAULT_THRESHOLDS_PATH = pathlib.Path(__file__).with_name("thresholds.json")

# Names whose *values* must never be written into an artifact. Presence and
# length only. Matched case-insensitively against the environment variable
# name, not against its value.
_SECRETISH = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|PASSWD|CREDENTIAL|AUTH",
                        re.IGNORECASE)


class Outcome(str, enum.Enum):
    """The only three dispositions an instrument may report."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    UNDETERMINED = "COULD_NOT_BE_CHECKED_MECHANICALLY"


class ThresholdError(RuntimeError):
    """A threshold was required and was absent, null or malformed.

    Raised rather than defaulted, deliberately. A threshold picked by whoever
    is holding the keyboard is indistinguishable from a designed one six weeks
    later.
    """


class IntermediatesMissingError(ValueError):
    """A verdict was offered without the quantities it was computed from."""


@dataclasses.dataclass(frozen=True)
class Threshold:
    """One threshold, with the provenance that makes it auditable.

    Attributes:
      key: Dotted key as it appears in the thresholds file.
      value: The value. ``None`` means "must be supplied; there is no default".
      kind: ``"input"`` (expected to be supplied or derived from measurement)
        or ``"constant"`` (fixed by a named party and not run-dependent).
      source: The named party or document the value came from.
      note: Anything a reader needs in order to weigh it.
    """

    key: str
    value: Any
    kind: str
    source: str
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Thresholds:
    """A read-only view over the thresholds file that records what it served.

    The recording matters: :meth:`provenance` is embedded in every artifact, so
    a reader of a result can see every number that shaped it and where each one
    came from, without going back to the source tree.
    """

    def __init__(self, data: Mapping[str, Any], path: Optional[str] = None):
        self._data = data
        self._path = str(path) if path is not None else "<in-memory>"
        self._served: Dict[str, Threshold] = {}

    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> "Thresholds":
        p = pathlib.Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
        with open(p, "r", encoding="utf-8") as handle:
            return cls(json.load(handle), path=str(p))

    @property
    def path(self) -> str:
        return self._path

    def entry(self, dotted_key: str) -> Threshold:
        """Returns the threshold entry, without asserting it has a value."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise ThresholdError(
                    f"threshold {dotted_key!r} is not present in {self._path}; "
                    "it is not defaulted here -- add it with a source, or stop "
                    "and ask the party who owns it")
            node = node[part]
        if not isinstance(node, Mapping) or "value" not in node:
            raise ThresholdError(
                f"threshold {dotted_key!r} in {self._path} is malformed: every "
                "leaf must be an object with value/kind/source")
        entry = Threshold(key=dotted_key,
                          value=node["value"],
                          kind=node.get("kind", "unspecified"),
                          source=node.get("source", "UNSOURCED"),
                          note=node.get("note", ""))
        self._served[dotted_key] = entry
        return entry

    def require(self, dotted_key: str) -> Any:
        """Returns the value, or raises if it is absent or null.

        A null value in the file is deliberate: it marks a threshold that has
        to be supplied from a measurement (M2's spread, for instance) and has
        no defensible default. Callers catch :class:`ThresholdError` and report
        UNDETERMINED; they do not invent a number.
        """
        entry = self.entry(dotted_key)
        if entry.value is None:
            raise ThresholdError(
                f"threshold {dotted_key!r} has no value: {entry.note or entry.source}")
        return entry.value

    def provenance(self) -> List[Dict[str, Any]]:
        """Every threshold read through this object, in the order first read."""
        return [t.to_dict() for t in self._served.values()]


@dataclasses.dataclass(frozen=True)
class CounterProvenance:
    """Units and source, travelling next to a counter's value.

    A downstream gate compares measured tile visits against an expected group
    count near 183 or a row-tile count near 4 -- roughly a factor of 46 apart.
    A units error there does not produce an implausible number; it produces a
    confident wrong verdict that kills or resurrects a candidate. So every
    counter read out of a profiler carries its units and where they were
    established, IN THE OUTPUT, and ``units_confirmed=False`` when they could
    not be established from the profiler's own documentation or output.

    Units are never inferred from whether the resulting number looks
    reasonable: a wrong-unit reading of a plausible quantity is precisely what
    looks reasonable.
    """

    name: str
    units: str
    source: str
    units_confirmed: bool
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if not self.units_confirmed:
            out["units"] = f"UNKNOWN (claimed: {self.units})"
        return out


def counted(value: Any, provenance: CounterProvenance) -> Dict[str, Any]:
    """One counter value with its provenance attached, ready to emit."""
    return {"value": value, "provenance": provenance.to_dict()}


@dataclasses.dataclass(frozen=True)
class Check:
    """A single disposition plus everything needed to recompute it."""

    name: str
    outcome: Outcome
    reason: str
    intermediates: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "intermediates": self.intermediates,
        }


def check(name: str, outcome: Outcome, reason: str,
          intermediates: Mapping[str, Any]) -> Check:
    """Builds a :class:`Check`, refusing to build a bare verdict.

    Raises:
      IntermediatesMissingError: if ``intermediates`` is empty. Including for
        UNDETERMINED -- "what I looked at and why it was not enough" is itself
        the intermediate in that case.
    """
    if not intermediates:
        raise IntermediatesMissingError(
            f"check {name!r} carries no intermediates; publish the quantities "
            "the verdict was computed from")
    return Check(name=name,
                 outcome=outcome,
                 reason=reason,
                 intermediates=dict(intermediates))


def worst(outcomes: Iterable[Outcome]) -> Outcome:
    """Aggregates dispositions without letting an UNDETERMINED become a PASS.

    FAILED dominates UNDETERMINED, which dominates PASSED. An empty iterable is
    UNDETERMINED: nothing was checked, so nothing passed.
    """
    seen = list(outcomes)
    if not seen:
        return Outcome.UNDETERMINED
    if Outcome.FAILED in seen:
        return Outcome.FAILED
    if Outcome.UNDETERMINED in seen:
        return Outcome.UNDETERMINED
    return Outcome.PASSED


def utc_now_iso() -> str:
    return datetime.datetime.now(
        datetime.timezone.utc).replace(microsecond=0).isoformat()


def env_snapshot(names: Sequence[str]) -> Dict[str, Any]:
    """Records the named environment variables for an artifact.

    Values are recorded for ordinary variables. For any name that looks like a
    credential, presence and length only -- never the value. This is a campaign
    constraint and it is enforced here so that no caller has to remember it.
    """
    out: Dict[str, Any] = {}
    for name in names:
        raw = os.environ.get(name)
        if _SECRETISH.search(name):
            out[name] = {
                "present": raw is not None,
                "length": len(raw) if raw is not None else 0,
                "value": "REDACTED-BY-POLICY",
            }
        else:
            out[name] = raw
    return out


def coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    """Sample CoV (stdev with n-1), or ``None`` when it is not defined.

    ``None`` for fewer than two values, or for a mean of zero. Callers turn
    that into UNDETERMINED; none of them substitute a zero.
    """
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if mean == 0.0:
        return None
    var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
    return math.sqrt(var) / abs(mean)


def summarise(values: Sequence[float]) -> Dict[str, Any]:
    """Mean/spread summary that always ships the raw values beside it."""
    vals = [float(v) for v in values]
    cov = coefficient_of_variation(vals)
    return {
        "n": len(vals),
        "raw_values": vals,
        "mean": (sum(vals) / len(vals)) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "coefficient_of_variation": cov,
    }


@dataclasses.dataclass
class NegativeControl:
    """Marks an artifact as the product of a deliberately corrupted run.

    An artifact carrying one of these is never reportable as a measurement. The
    flag exists so that a control run can never be mistaken for a real one
    after the fact, by anybody, including us.
    """

    name: str
    description: str
    expected_effect: str
    executed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def write_artifact(path: os.PathLike,
                   *,
                   kind: str,
                   payload: Mapping[str, Any],
                   checks: Sequence[Check] = (),
                   thresholds: Optional[Thresholds] = None,
                   negative_control: Optional[NegativeControl] = None,
                   extra_provenance: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Writes one artifact as JSON and returns the record that was written.

    Args:
      path: Destination file. Parent directories are created.
      kind: Instrument identifier, e.g. ``"M4"``.
      payload: The measurement itself, intermediates included.
      checks: Dispositions computed from the payload.
      thresholds: The thresholds object the caller read from, so its
        provenance travels with the result.
      negative_control: Set when this run was deliberately corrupted. Forces
        ``reportable`` to false.
      extra_provenance: Anything else the reader needs (config fingerprint,
        pod identity, upstream artifact paths).

    Raises:
      IntermediatesMissingError: if a payload is empty.
    """
    if not payload:
        raise IntermediatesMissingError(
            f"artifact {kind!r} has an empty payload; a verdict without its "
            "intermediates is not a measurement")
    record: Dict[str, Any] = {
        "schema": SCHEMA,
        "kind": kind,
        "created_utc": utc_now_iso(),
        "reportable": negative_control is None,
        "negative_control": negative_control.to_dict() if negative_control else None,
        "payload": dict(payload),
        "checks": [c.to_dict() for c in checks],
        "overall_outcome": worst(c.outcome for c in checks).value if checks else
                           Outcome.UNDETERMINED.value,
        "thresholds_used": thresholds.provenance() if thresholds else [],
        "thresholds_file": thresholds.path if thresholds else None,
        "provenance": dict(extra_provenance or {}),
    }
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return record


def load_json(path: os.PathLike) -> Any:
    with open(pathlib.Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)
