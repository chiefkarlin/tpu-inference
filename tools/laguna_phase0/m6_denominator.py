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
"""M6 -- the denominator assertion, in the harness itself.

A pin that no instrument reads is not a defence, it is a note. This module is
the instrument: it derives the chip count from the runtime rather than from a
banner, checks the per-chip roofline pair against a pinned table, and **fails
the run** when either is wrong.

THE UNIT TRAP THIS EXISTS FOR. A v7x chip is two chiplets, which is two
TensorCores, which is two JAX devices. The runtime device is the chiplet, so the
device count is TWICE the chip count, and the startup banner's chip-count field
reports devices. Chip count is therefore derived only from distinct JAX device
coordinates or from the GKE ``google.com/tpu`` allocation -- never from the
banner.

THE FAULT THE DESIGN'S OWN CONTROL DOES NOT CATCH, AND WHY THIS MODULE HAS TWO
LEGS. Injecting a wrong chip count exercises the least consequential input: a
*consistent* re-denomination halves FLOPs and bandwidth together, the ridge
point is their quotient, and it does not move -- 2307/7.38 and 1153.5/3.69 are
the same 312.6 FLOP/byte, so an arithmetic intensity of 279.6 stays below the
ridge and the memory-bound classification is unchanged under all four pairings.
What flips an arm is an INCONSISTENT PAIRING: per-device FLOPs read against
per-chip bandwidth. So this module guards the pairing, not the unit count, and
:func:`assert_denominator` checks the ridge as a first-class check that is
independent of the basis labels -- because a label can lie and a quotient
cannot.

Neither negative control has been executed. See ``README.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools.laguna_phase0 import common

# Environment variable through which a pod may publish its `google.com/tpu`
# allocation, via the downward API. There is no kubectl call anywhere in this
# package: cluster mutation and object inspection are both barred, and a
# harness that shells out to the control plane is a harness that cannot run
# unprivileged.
GKE_ALLOCATION_ENV = "LAGUNA_GKE_TPU_ALLOCATION"
GKE_ALLOCATION_UNIT_ENV = "LAGUNA_GKE_TPU_ALLOCATION_UNIT"

RECORDED_ENV = (
    GKE_ALLOCATION_ENV,
    GKE_ALLOCATION_UNIT_ENV,
    "TPU_WORKER_ID",
    "TPU_ACCELERATOR_TYPE",
)


# The route whose answer negative-control leg 1 corrupts. Leg 1 perturbs
# exactly ONE route, so that the reconciliation has something to disagree with.
# Corrupting both routes identically would be undetectable BY DESIGN -- no
# cross-check exists that could see it -- and a control that cannot be detected
# is not a stronger control, it is an untestable one. Which route is corrupted
# is therefore part of what leg 1 claims, and it is named here rather than left
# implicit at the injection site.
LEG_1_TARGET_ROUTE = "jax_device_coords"


class Basis(str, enum.Enum):
    """Which denominator a figure is expressed against."""

    PER_CHIP = "per_chip"
    PER_DEVICE = "per_device"


class ControlResponse(str, enum.Enum):
    """Whether a negative control's corruption changed the instrument's answer.

    THIS IS NOT AN OUTCOME AND IT IS NOT A VERDICT. It answers one question and
    only that question: *did the instrument say something different because of
    the corruption?* A control that cannot move the instrument's answer has not
    been shown to work, whatever answer the instrument happened to give.

    ``UNDEMONSTRABLE`` and ``NOT_EXERCISED`` are both "could not be checked
    mechanically" for this purpose. Neither is a pass, and neither may be
    reported as one.
    """

    DETECTED = "detected"
    NOT_DETECTED = "not-detected"
    NOT_EXERCISED = "not-exercised"
    UNDEMONSTRABLE = "undemonstrable"


class DenominatorAssertionError(RuntimeError):
    """The denominator was wrong, or could not be established. The run stops.

    Raised for UNDETERMINED as well as for FAILED: a run whose denominator was
    not asserted is not reportable, so "I could not check" stops the run in
    exactly the same way that "it is wrong" does.
    """


@dataclasses.dataclass(frozen=True)
class DeviceView:
    """A JAX device reduced to the fields the chip-count derivation needs.

    Kept as a plain record so the derivation can be exercised without a TPU
    attached, and so a profile artifact can carry the device set it was taken
    on.
    """

    device_id: int
    process_index: int = 0
    slice_index: Optional[int] = None
    coords: Optional[Tuple[int, ...]] = None
    core_on_chip: Optional[int] = None

    def chip_key(self) -> Optional[Tuple[Any, ...]]:
        """Identity of the physical chip this device (chiplet) sits on.

        ``core_on_chip`` is deliberately excluded: it is what distinguishes the
        two chiplets of one chip, so including it would count chiplets. Returns
        ``None`` when coordinates are unavailable, which the caller turns into
        UNDETERMINED rather than into a count.
        """
        if self.coords is None:
            return None
        return (self.process_index, self.slice_index, tuple(self.coords))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def device_views_from_jax(devices: Optional[Sequence[Any]] = None) -> List[DeviceView]:
    """Snapshots the live JAX devices into :class:`DeviceView` records.

    JAX is imported lazily so that importing this module costs nothing and
    initialises no backend.
    """
    if devices is None:
        import jax  # pylint: disable=import-outside-toplevel
        devices = jax.devices()
    views: List[DeviceView] = []
    for dev in devices:
        coords = getattr(dev, "coords", None)
        views.append(
            DeviceView(device_id=int(getattr(dev, "id", -1)),
                       process_index=int(getattr(dev, "process_index", 0)),
                       slice_index=_maybe_int(getattr(dev, "slice_index", None)),
                       coords=tuple(int(c) for c in coords) if coords is not None else None,
                       core_on_chip=_maybe_int(getattr(dev, "core_on_chip", None))))
    return views


def _maybe_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


@dataclasses.dataclass(frozen=True)
class RooflineBasis:
    """A FLOPs/bandwidth pair together with the basis each side is on.

    The basis labels are recorded because they are worth checking, and the
    ridge point is computed because it is worth checking *independently of the
    labels*.
    """

    dense_bf16_tflops: float
    hbm_bandwidth_gbytes_per_s: float
    flops_basis: Basis
    bandwidth_basis: Basis
    source_row: str

    @property
    def ridge_flops_per_byte(self) -> float:
        """FLOPs per byte at which the arms cross. Invariant under a
        *consistent* re-denomination; halved or doubled by an inconsistent
        pairing, which is exactly why it is the pairing detector."""
        return (self.dense_bf16_tflops * 1e12) / (self.hbm_bandwidth_gbytes_per_s * 1e9)

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["flops_basis"] = self.flops_basis.value
        out["bandwidth_basis"] = self.bandwidth_basis.value
        out["ridge_flops_per_byte"] = self.ridge_flops_per_byte
        return out


def pinned_basis(thresholds: common.Thresholds) -> RooflineBasis:
    """The pinned per-chip table. Both halves are read from the same row."""
    entry_f = thresholds.entry("m6.per_chip_dense_bf16_tflops")
    entry_b = thresholds.entry("m6.per_chip_hbm_bandwidth_gbytes_per_s")
    return RooflineBasis(dense_bf16_tflops=float(entry_f.value),
                         hbm_bandwidth_gbytes_per_s=float(entry_b.value),
                         flops_basis=Basis.PER_CHIP,
                         bandwidth_basis=Basis.PER_CHIP,
                         source_row=f"{entry_f.source} | {entry_b.source}")


@dataclasses.dataclass
class Injection:
    """Deliberate corruptions, for the negative controls. None are executed.

    Attributes:
      chip_count_scale: Leg 1. Multiplies ONE route's derived chip count --
        :data:`LEG_1_TARGET_ROUTE` -- and then leaves the reconciliation alone
        to reach whatever verdict it reaches. Expressed as a scale rather than
        as a literal so that no artifact in this repository ever contains a
        device count written as a chip count.

        BEFORE ROUND 1 THIS LEG WROTE ITS OWN FAILED VERDICT (review finding
        R5). It perturbed the value *and* asserted the answer, so it reported
        FAILED even for a scale of 1.0, and it would still have reported FAILED
        with the entire reconciliation deleted. It could not come out any other
        way, which is the definition of a control that has not been shown to
        work. It now corrupts an INPUT only; see :func:`adjudicate_leg_1` for
        how the response is measured rather than declared.
      inconsistent_pairing: Leg 2, the one that matters. Halves the FLOPs side
        only, leaving bandwidth per-chip: per-device FLOPs against per-chip
        bandwidth.
      consistent_redenomination: Not a control -- a demonstration. Halves both
        sides. The ridge does not move, the classification does not change, and
        that is the point: this is the fault class the design's own control was
        watching.
    """

    chip_count_scale: Optional[float] = None
    inconsistent_pairing: bool = False
    consistent_redenomination: bool = False

    def active(self) -> bool:
        return (self.chip_count_scale is not None or self.inconsistent_pairing
                or self.consistent_redenomination)

    def as_negative_control(self, *,
                            executed: bool = False) -> Optional[common.NegativeControl]:
        """The control descriptor. ``executed`` says whether it actually RAN.

        Hardcoded ``False`` until round 1: the field that answers "has this
        control ever fired?" -- R9's whole subject, and R11's -- could not
        record a firing even after one.
        """
        if not self.active():
            return None
        legs = []
        if self.chip_count_scale is not None:
            legs.append(f"leg 1: chip count scaled by {self.chip_count_scale}")
        if self.inconsistent_pairing:
            legs.append("leg 2: per-device FLOPs against per-chip bandwidth")
        if self.consistent_redenomination:
            legs.append("demonstration: consistent re-denomination of both sides")
        return common.NegativeControl(
            name="m6-negative-control",
            description="; ".join(legs),
            expected_effect=("M6 fails the run on legs 1 and 2. Under a consistent "
                             "re-denomination the ridge is unchanged and the pairing "
                             "check passes, which is the finding, not a defect."),
            executed=executed)

    def apply_to_basis(self, basis: RooflineBasis) -> RooflineBasis:
        flops, bandwidth = basis.dense_bf16_tflops, basis.hbm_bandwidth_gbytes_per_s
        flops_basis, bandwidth_basis = basis.flops_basis, basis.bandwidth_basis
        if self.inconsistent_pairing:
            flops = flops / 2.0
            flops_basis = Basis.PER_DEVICE
        if self.consistent_redenomination:
            flops = flops / 2.0
            bandwidth = bandwidth / 2.0
            flops_basis = Basis.PER_DEVICE
            bandwidth_basis = Basis.PER_DEVICE
        return dataclasses.replace(basis,
                                   dense_bf16_tflops=flops,
                                   hbm_bandwidth_gbytes_per_s=bandwidth,
                                   flops_basis=flops_basis,
                                   bandwidth_basis=bandwidth_basis)

    def apply_to_evidence(
            self,
            evidence: Sequence["ChipCountEvidence"]) -> List["ChipCountEvidence"]:
        """Leg 1: corrupt the INPUT to the reconciliation, never its verdict.

        The corruption stops here. Nothing downstream of this method is told
        that an injection happened, so every check that follows reaches its
        verdict from the numbers alone -- which is the only arrangement in
        which the verdict is evidence that the checks work.

        A route that produced no answer is left alone: there is no value to
        scale, and manufacturing one would be inventing the very thing the
        route refused to guess at.
        """
        if self.chip_count_scale is None:
            return list(evidence)
        out: List["ChipCountEvidence"] = []
        for item in evidence:
            if item.route != LEG_1_TARGET_ROUTE or item.chip_count is None:
                out.append(item)
                continue
            scaled = int(round(item.chip_count * self.chip_count_scale))
            detail = dict(item.detail)
            detail["negative_control_leg_1"] = {
                "route_corrupted": item.route,
                "scale": self.chip_count_scale,
                "chip_count_before_injection": item.chip_count,
                "chip_count_after_injection": scaled,
            }
            out.append(ChipCountEvidence(item.route, scaled, detail))
        return out


@dataclasses.dataclass
class ChipCountEvidence:
    """One route's answer, with enough detail to see how it got there."""

    route: str
    chip_count: Optional[int]
    detail: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def chip_count_from_coords(views: Sequence[DeviceView],
                           devices_per_chip: int) -> ChipCountEvidence:
    """Distinct chip coordinates, cross-checked against the device count.

    Two chiplets of one chip share coordinates and differ only in
    ``core_on_chip``, so distinct coordinates count chips. If that collapse does
    not happen -- if distinct coordinates equal the device count -- the route
    returns no answer instead of returning the device count under a chip's name.
    That refusal is the whole point of the route: the flattering reading of an
    ambiguous topology is the one that survives review, so it is the one that
    must be refused mechanically.
    """
    detail: Dict[str, Any] = {
        "device_count": len(views),
        "devices_per_chip_expected": devices_per_chip,
        "devices": [v.to_dict() for v in views],
    }
    if not views:
        detail["why"] = "no devices were visible"
        return ChipCountEvidence("jax_device_coords", None, detail)
    keys = [v.chip_key() for v in views]
    if any(k is None for k in keys):
        detail["why"] = "at least one device exposed no coordinates"
        return ChipCountEvidence("jax_device_coords", None, detail)
    distinct = sorted({tuple(k) for k in keys})  # type: ignore[arg-type]
    detail["distinct_chip_keys"] = [list(k) for k in distinct]
    detail["distinct_chip_key_count"] = len(distinct)
    if len(views) != len(distinct) * devices_per_chip:
        detail["why"] = (
            "device count is not the expected multiple of distinct coordinates; "
            "the coordinate route cannot be trusted to have collapsed chiplets "
            "on this topology")
        return ChipCountEvidence("jax_device_coords", None, detail)
    return ChipCountEvidence("jax_device_coords", len(distinct), detail)


def chip_count_from_gke(allocation: Optional[int],
                        unit: Optional[str],
                        devices_per_chip: int) -> ChipCountEvidence:
    """The GKE ``google.com/tpu`` allocation, converted only if its unit is declared.

    THE UNIT IS NOT ASSUMED. Whether that resource counts chips or chiplets on
    v7x has not been established by the author of this module, and guessing it
    is the factor-of-two hazard in its purest form. An undeclared unit yields no
    answer.
    """
    detail: Dict[str, Any] = {"raw_allocation": allocation, "declared_unit": unit}
    if allocation is None:
        detail["why"] = f"no allocation supplied (env {GKE_ALLOCATION_ENV} unset)"
        return ChipCountEvidence("gke_allocation", None, detail)
    if unit == "chips":
        return ChipCountEvidence("gke_allocation", int(allocation), detail)
    if unit == "devices":
        if int(allocation) % devices_per_chip:
            detail["why"] = "device-denominated allocation is not a whole number of chips"
            return ChipCountEvidence("gke_allocation", None, detail)
        return ChipCountEvidence("gke_allocation",
                                 int(allocation) // devices_per_chip, detail)
    detail["why"] = (
        "the unit of the allocation was not declared; this route reports nothing "
        "rather than guess between chips and chiplets")
    return ChipCountEvidence("gke_allocation", None, detail)


def reconcile_chip_count(evidence: Sequence[ChipCountEvidence]) -> Tuple[Optional[int], common.Check]:
    """Combines the routes. Disagreement is a failure, not a tie-break.

    Neither route is preferred over the other, because a preference is how a
    factor of two survives: whichever route is believed becomes the one that is
    never checked.
    """
    answers = {e.route: e.chip_count for e in evidence}
    values = sorted({v for v in answers.values() if v is not None})
    intermediates: Dict[str, Any] = {
        "routes": [e.to_dict() for e in evidence],
        "answers": answers,
    }
    if not values:
        return None, common.check(
            "m6.chip_count", common.Outcome.UNDETERMINED,
            "no route produced a chip count; the denominator is unestablished and "
            "the run is not reportable", intermediates)
    if len(values) > 1:
        return None, common.check(
            "m6.chip_count", common.Outcome.FAILED,
            f"routes disagree on the chip count: {answers}", intermediates)
    count = values[0]
    if len([v for v in answers.values() if v is not None]) == 1:
        return count, common.check(
            "m6.chip_count", common.Outcome.UNDETERMINED,
            f"only one route produced a chip count ({count}); it is unconfirmed by "
            "a second route, so it is recorded and not passed", intermediates)
    return count, common.check("m6.chip_count", common.Outcome.PASSED,
                               f"both routes agree on {count} chips", intermediates)


def adjudicate_leg_1(
        *,
        scale: Optional[float],
        raw_evidence: Sequence[ChipCountEvidence],
        injected_evidence: Sequence[ChipCountEvidence],
        baseline: Tuple[Optional[int], common.Check],
        corrupted: Tuple[Optional[int], common.Check]) -> Optional[Dict[str, Any]]:
    """Measures whether leg 1's corruption changed the reconciliation's answer.

    THE CONTROL'S RESULT IS A DIFFERENCE BETWEEN TWO REAL RUNS OF THE SAME
    DETECTOR, on the uncorrupted and the corrupted input. That is the whole
    repair for R5. A difference cannot be hand-written the way a verdict can:
    to report DETECTED, the detector has to have passed the clean input and
    refused the dirty one, and if the detector were deleted both runs would
    return the same thing and this would report NOT_DETECTED.

    The question it answers is the standing one: WHAT INPUT WOULD MAKE THIS SAY
    SOMETHING ELSE? All four responses below are answers to it.

    Args:
      scale: ``Injection.chip_count_scale``. ``None`` means the leg is not
        active and there is nothing to adjudicate.
      raw_evidence: The routes' answers before the injection.
      injected_evidence: The same routes after it.
      baseline: ``reconcile_chip_count(raw_evidence)``.
      corrupted: ``reconcile_chip_count(injected_evidence)``.

    Returns:
      ``None`` when the leg is inactive, otherwise a record carrying the
      response, the reason, and both reconciliations' outcomes.

    ON REACHABILITY, STATED RATHER THAN LEFT TO BE DISCOVERED. Through
    :func:`assert_denominator` as the routes stand today, ``NOT_DETECTED``
    cannot occur: any scale that moves the value makes the two routes disagree,
    and disagreement is FAILED. That is a property of there being exactly two
    routes and no preference between them, NOT a property of this function, and
    it would stop holding the moment a third route, a tie-break or a preferred
    route is added. It is reachable here, and it is exercised directly by
    ``test_the_leg_1_adjudicator_reports_a_blind_detector``, because a branch
    that has never been executed is not a branch anyone should rely on.
    """
    if scale is None:
        return None
    baseline_count, baseline_check = baseline
    corrupted_count, corrupted_check = corrupted
    moved = [{
        "route": before.route,
        "before": before.chip_count,
        "after": after.chip_count,
    } for before, after in zip(raw_evidence, injected_evidence)
             if before.chip_count != after.chip_count]

    record: Dict[str, Any] = {
        "leg": "leg 1: chip count",
        "scale": scale,
        "target_route": LEG_1_TARGET_ROUTE,
        "routes_moved": moved,
        "reconciliation_without_injection": {
            "outcome": baseline_check.outcome.value,
            "chip_count": baseline_count,
            "reason": baseline_check.reason,
        },
        "reconciliation_with_injection": {
            "outcome": corrupted_check.outcome.value,
            "chip_count": corrupted_count,
            "reason": corrupted_check.reason,
        },
    }

    if not moved:
        record["response"] = ControlResponse.NOT_EXERCISED.value
        record["why"] = (
            f"a scale of {scale} left every route's answer unchanged, so no "
            "detector was put under test. Nothing was corrupted and nothing "
            "can be concluded: this is UNDETERMINED for the control, and it is "
            "not a pass. Before round 1 this same input reported FAILED.")
    elif baseline_check.outcome is not common.Outcome.PASSED:
        record["response"] = ControlResponse.UNDEMONSTRABLE.value
        record["why"] = (
            "the reconciliation did not pass on the UNCORRUPTED input either "
            f"({baseline_check.outcome.value}: {baseline_check.reason}), so a "
            "non-passing result on the corrupted input demonstrates nothing -- "
            "the answer was already non-passing before the corruption arrived. "
            "The commonest cause is a single live route, which is also the "
            "case in which the corrupted count would flow onward unchallenged.")
    elif corrupted_check.outcome is common.Outcome.PASSED:
        record["response"] = ControlResponse.NOT_DETECTED.value
        record["why"] = (
            "the reconciliation passed the clean input and passed the "
            "corrupted one too. THE DETECTOR IS BLIND TO THIS CORRUPTION and "
            "the chip count it reports is the injected one. This is a finding "
            "about the instrument, not a failure of the control.")
    else:
        record["response"] = ControlResponse.DETECTED.value
        record["why"] = (
            "the reconciliation PASSED on the uncorrupted input and returned "
            f"{corrupted_check.outcome.value} on the corrupted one. The "
            "verdict was reached by the detector from the numbers; the "
            "injection wrote no outcome and touched no check.")
    return record


def check_basis_labels(supplied: RooflineBasis) -> common.Check:
    """Both halves must claim the per-chip basis. Labels can lie; see the ridge."""
    intermediates = {"supplied": supplied.to_dict()}
    if (supplied.flops_basis is Basis.PER_CHIP
            and supplied.bandwidth_basis is Basis.PER_CHIP):
        return common.check("m6.basis_labels", common.Outcome.PASSED,
                            "both figures are labelled per-chip", intermediates)
    return common.check(
        "m6.basis_labels", common.Outcome.FAILED,
        f"basis labels are not both per-chip: flops={supplied.flops_basis.value}, "
        f"bandwidth={supplied.bandwidth_basis.value}", intermediates)


def check_pinned_values(supplied: RooflineBasis,
                        pinned: RooflineBasis) -> common.Check:
    """The two figures must be the pinned per-chip ones, to the digit."""
    intermediates = {"supplied": supplied.to_dict(), "pinned": pinned.to_dict()}
    mismatches = []
    if supplied.dense_bf16_tflops != pinned.dense_bf16_tflops:
        mismatches.append(
            f"dense bf16 {supplied.dense_bf16_tflops} != pinned {pinned.dense_bf16_tflops} TF")
    if supplied.hbm_bandwidth_gbytes_per_s != pinned.hbm_bandwidth_gbytes_per_s:
        mismatches.append(
            f"HBM {supplied.hbm_bandwidth_gbytes_per_s} != pinned "
            f"{pinned.hbm_bandwidth_gbytes_per_s} GB/s")
    if mismatches:
        return common.check("m6.pinned_values", common.Outcome.FAILED,
                            "; ".join(mismatches), intermediates)
    return common.check("m6.pinned_values", common.Outcome.PASSED,
                        "both figures match the pinned per-chip row", intermediates)


def check_pairing(supplied: RooflineBasis, pinned: RooflineBasis,
                  relative_tolerance: float) -> common.Check:
    """THE LEG THAT MATTERS: is the pair internally consistent?

    Compares the ridge point of the supplied pair against the pinned ridge. A
    consistent re-denomination leaves this check passing -- correctly, because
    it changes no classification. An inconsistent pairing moves the ridge by a
    factor of two and fails here even if both labels claim per-chip.
    """
    ridge, pinned_ridge = supplied.ridge_flops_per_byte, pinned.ridge_flops_per_byte
    relative_error = abs(ridge - pinned_ridge) / pinned_ridge
    intermediates = {
        "supplied_ridge_flops_per_byte": ridge,
        "pinned_ridge_flops_per_byte": pinned_ridge,
        "relative_error": relative_error,
        "relative_tolerance": relative_tolerance,
        "supplied": supplied.to_dict(),
        "pinned": pinned.to_dict(),
    }
    if relative_error <= relative_tolerance:
        return common.check(
            "m6.pairing", common.Outcome.PASSED,
            "the FLOPs/bandwidth pair is internally consistent: the ridge point "
            f"is {ridge:.4f} FLOP/byte, as pinned", intermediates)
    return common.check(
        "m6.pairing", common.Outcome.FAILED,
        f"INCONSISTENTLY PAIRED BASIS: ridge {ridge:.4f} FLOP/byte against pinned "
        f"{pinned_ridge:.4f}. One side of the pair is denominated differently from "
        "the other; this is the fault that flips an arm.", intermediates)


def check_tensor_parallel(tensor_parallel_size: Optional[int],
                          ceiling: int) -> common.Check:
    intermediates = {"tensor_parallel_size": tensor_parallel_size, "ceiling": ceiling}
    if tensor_parallel_size is None:
        return common.check("m6.tensor_parallel", common.Outcome.UNDETERMINED,
                            "tensor parallel size was not supplied", intermediates)
    if tensor_parallel_size > ceiling:
        return common.check(
            "m6.tensor_parallel", common.Outcome.FAILED,
            f"tensor parallel size {tensor_parallel_size} exceeds the ceiling "
            f"{ceiling}", intermediates)
    return common.check("m6.tensor_parallel", common.Outcome.PASSED,
                        f"tensor parallel size {tensor_parallel_size} is within the "
                        f"ceiling {ceiling}", intermediates)


@dataclasses.dataclass
class DenominatorReport:
    chip_count: Optional[int]
    checks: List[common.Check]
    basis: RooflineBasis
    pinned: RooflineBasis
    injection: Injection
    # The leg 1 control record, or None when leg 1 was not active. It is
    # DELIBERATELY NOT A Check AND IS NOT IN `checks`: it answers "did the
    # corruption move the detector", which is a different question from "is
    # this run's denominator sound", and folding the two would let a control
    # result decide a run's outcome. It is published; it is not adjudicated
    # into the outcome.
    leg_1_control: Optional[Dict[str, Any]] = None

    @property
    def outcome(self) -> common.Outcome:
        return common.worst(c.outcome for c in self.checks)

    def payload(self) -> Dict[str, Any]:
        return {
            "chip_count": self.chip_count,
            "supplied_basis": self.basis.to_dict(),
            "pinned_basis": self.pinned.to_dict(),
            "injection": dataclasses.asdict(self.injection),
            "negative_control_leg_1": self.leg_1_control,
        }


def assert_denominator(*,
                       views: Sequence[DeviceView],
                       thresholds: common.Thresholds,
                       gke_allocation: Optional[int] = None,
                       gke_allocation_unit: Optional[str] = None,
                       tensor_parallel_size: Optional[int] = None,
                       supplied: Optional[RooflineBasis] = None,
                       injection: Optional[Injection] = None,
                       raise_on_failure: bool = True) -> DenominatorReport:
    """Derives the denominator, checks it, and stops the run when it is wrong.

    Args:
      views: The JAX devices, as records.
      thresholds: Loaded thresholds; the pinned table lives there.
      gke_allocation: The `google.com/tpu` allocation, if published to the pod.
      gke_allocation_unit: ``"chips"`` or ``"devices"``. Undeclared means the
        route abstains.
      tensor_parallel_size: Checked against the ceiling when supplied.
      supplied: The pair the harness intends to use. Defaults to the pinned pair.
      injection: Negative-control corruption. Never set in a real run.
      raise_on_failure: When true, anything other than PASSED raises.

    Raises:
      DenominatorAssertionError: on FAILED or UNDETERMINED.
    """
    injection = injection or Injection()
    devices_per_chip = int(thresholds.require("m6.devices_per_chip"))
    pinned = pinned_basis(thresholds)
    basis = injection.apply_to_basis(supplied or pinned)

    raw_evidence = [
        chip_count_from_coords(views, devices_per_chip),
        chip_count_from_gke(gke_allocation, gke_allocation_unit, devices_per_chip),
    ]
    # R5. Leg 1 corrupts an input and then gets out of the way. The
    # reconciliation runs twice on the same code path -- once clean, once
    # dirty -- and the control's result is the DIFFERENCE, which is measured
    # below and not asserted here. When the leg is inactive the two calls are
    # given identical input and `adjudicate_leg_1` returns None.
    injected_evidence = injection.apply_to_evidence(raw_evidence)
    baseline = reconcile_chip_count(raw_evidence)
    chip_count, count_check = reconcile_chip_count(injected_evidence)
    leg_1 = adjudicate_leg_1(scale=injection.chip_count_scale,
                             raw_evidence=raw_evidence,
                             injected_evidence=injected_evidence,
                             baseline=baseline,
                             corrupted=(chip_count, count_check))

    checks = [
        count_check,
        check_basis_labels(basis),
        check_pinned_values(basis, pinned),
        check_pairing(basis, pinned,
                      float(thresholds.require("m6.ridge_relative_tolerance"))),
        check_tensor_parallel(tensor_parallel_size,
                              int(thresholds.require("m6.max_tensor_parallel_size"))),
    ]
    report = DenominatorReport(chip_count=chip_count,
                               checks=checks,
                               basis=basis,
                               pinned=pinned,
                               injection=injection,
                               leg_1_control=leg_1)
    if raise_on_failure and report.outcome is not common.Outcome.PASSED:
        reasons = "; ".join(f"{c.name}={c.outcome.value}: {c.reason}"
                            for c in checks
                            if c.outcome is not common.Outcome.PASSED)
        raise DenominatorAssertionError(
            f"M6 stops this run ({report.outcome.value}): {reasons}")
    return report


def _int_or_none(value: Optional[str]) -> Optional[int]:
    return None if value in (None, "") else int(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="artifact path to write")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--gke-allocation", type=int, default=None)
    parser.add_argument("--gke-allocation-unit", choices=("chips", "devices"),
                        default=None,
                        help="declare what google.com/tpu counts on this pool; "
                             "undeclared means the route abstains")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--nc-chip-count-scale", type=float, default=None,
                        help="NEGATIVE CONTROL leg 1: scale the derived chip count")
    parser.add_argument("--nc-inconsistent-pairing", action="store_true",
                        help="NEGATIVE CONTROL leg 2: per-device FLOPs against "
                             "per-chip bandwidth")
    parser.add_argument("--nc-consistent-redenomination", action="store_true",
                        help="demonstration: halve both sides; the ridge does not "
                             "move and nothing is flipped")
    args = parser.parse_args(argv)

    thresholds = common.Thresholds.load(args.thresholds)
    injection = Injection(chip_count_scale=args.nc_chip_count_scale,
                          inconsistent_pairing=args.nc_inconsistent_pairing,
                          consistent_redenomination=args.nc_consistent_redenomination)
    allocation = args.gke_allocation
    if allocation is None:
        allocation = _int_or_none(os.environ.get(GKE_ALLOCATION_ENV))
    unit = args.gke_allocation_unit or os.environ.get(GKE_ALLOCATION_UNIT_ENV)

    views = device_views_from_jax()
    try:
        report = assert_denominator(views=views,
                                    thresholds=thresholds,
                                    gke_allocation=allocation,
                                    gke_allocation_unit=unit,
                                    tensor_parallel_size=args.tensor_parallel_size,
                                    injection=injection,
                                    raise_on_failure=False)
    except common.ThresholdError as exc:
        print(f"M6: {exc}", file=sys.stderr)
        return 2

    common.write_artifact(args.out,
                          kind="M6",
                          payload=report.payload(),
                          checks=report.checks,
                          thresholds=thresholds,
                          # Reached only after assert_denominator returned, and
                          # assert_denominator is where the injection is applied
                          # -- `apply_to_basis` for leg 2, `apply_to_evidence`
                          # for leg 1, both named rather than cited by a line
                          # number that the next edit invalidates. So on this
                          # path an ACTIVE injection has by construction already
                          # run, and `executed` says so. `as_negative_control`
                          # still returns None when the injection is inactive,
                          # so this cannot claim a firing that did not happen.
                          # NOTE that `executed=True` says the leg RAN. Whether
                          # it DETECTED anything is a separate finding and is
                          # in the payload under `negative_control_leg_1`.
                          negative_control=injection.as_negative_control(
                              executed=True),
                          extra_provenance={"env": common.env_snapshot(RECORDED_ENV)})
    print(f"M6: {report.outcome.value}; artifact written to {args.out}")
    for item in report.checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    if report.leg_1_control is not None:
        # The control's response goes to the headline channel, because a
        # control result that only reaches the artifact is a control result
        # nobody reads. It is printed as its own line and NOT merged into the
        # outcome line above: the run's outcome and the control's response are
        # two different measurements and the exit status carries neither.
        print(f"  NEGATIVE CONTROL leg 1: {report.leg_1_control['response']} "
              f"-- {report.leg_1_control['why']}")
    return 0 if report.outcome is common.Outcome.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
