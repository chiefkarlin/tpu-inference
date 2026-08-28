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
"""Ladder points that carry their own basis, and refuse to be mixed.

Review finding B3 of the design review: a residual curve was assembled from
points on two different bases -- some wall-clock, some TPOT-derived -- and the
mix understated a load-bearing term by about 57%. The defect is not that anyone
chose the wrong basis; it is that the basis was not attached to the point, so
mixing them looked like arithmetic.

So in this package a measured point is never a bare float. It is a
:class:`LadderPoint` carrying its concurrency, its units, its basis and the
instrument that produced it, and any operation that combines points calls
:func:`require_single_basis` first. Label the basis ON THE POINT, not in a
footnote.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


class MeasurementBasis(str, enum.Enum):
    """What a number was measured against. Never inferred, always declared."""

    WALL_CLOCK = "wall_clock"
    ENGINE_STEP_WALL_CLOCK = "engine_step_wall_clock"
    TPOT_DERIVED = "tpot_derived"
    DEVICE_TIMELINE = "device_timeline"
    ROUTER_COUNT = "router_count"
    UNKNOWN = "UNKNOWN"


class BasisMismatchError(ValueError):
    """Points on different bases were about to be combined."""


@dataclasses.dataclass(frozen=True)
class LadderPoint:
    """One point on the concurrency ladder, with its basis attached to it.

    Attributes:
      concurrency: The ``c`` of the ladder point.
      value: The measured quantity.
      units: Units of ``value``. Written out, e.g. ``"ms"``,
        ``"distinct experts per layer per step"``.
      basis: What the value was measured against.
      instrument: Which instrument produced it, e.g. ``"M4"``.
      shape: The benchmark shape, e.g. ``"B isl512/osl256"``.
      detail: Anything a reader needs to recompute it.
    """

    concurrency: int
    value: Optional[float]
    units: str
    basis: MeasurementBasis
    instrument: str
    shape: Optional[str] = None
    detail: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["basis"] = self.basis.value
        return out


def require_single_basis(points: Sequence[LadderPoint]) -> MeasurementBasis:
    """Returns the common basis, or refuses.

    Raises:
      BasisMismatchError: if the points are on more than one basis, or if any
        of them is on an UNKNOWN basis. An unknown basis is not a wildcard that
        matches everything; it is a point that cannot be combined with anything.
    """
    if not points:
        raise BasisMismatchError("no points supplied; there is no basis to agree on")
    bases = {p.basis for p in points}
    if MeasurementBasis.UNKNOWN in bases:
        raise BasisMismatchError(
            "at least one point declares an UNKNOWN basis and cannot be combined")
    if len(bases) > 1:
        raise BasisMismatchError(
            "points are on more than one basis: "
            f"{sorted(b.value for b in bases)}; mixing them is review finding B3")
    return bases.pop()


def declare_cross_basis(left: LadderPoint, right: LadderPoint,
                        why: str) -> Dict[str, Any]:
    """Records a comparison that crosses bases ON PURPOSE.

    Some comparisons are cross-basis by construction -- the serving-stack cost
    is precisely an engine-step wall-clock time subtracted from a TPOT-derived
    one, and the crossing is the measurement rather than a mistake. Those go
    through here, which writes the crossing into the artifact in words, so that
    the difference between a deliberate crossing and review finding B3 is
    visible to a reader instead of resting on the author's intent.
    """
    return {
        "cross_basis_comparison": True,
        "left_basis": left.basis.value,
        "right_basis": right.basis.value,
        "why_this_crossing_is_deliberate": why,
    }


def ladder_to_dicts(points: Iterable[LadderPoint]) -> List[Dict[str, Any]]:
    return [p.to_dict() for p in points]
