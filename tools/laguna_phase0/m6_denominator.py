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


class Basis(str, enum.Enum):
    """Which denominator a figure is expressed against."""

    PER_CHIP = "per_chip"
    PER_DEVICE = "per_device"


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
      chip_count_scale: Leg 1. Multiplies the derived chip count. Expressed as
        a scale rather than as a literal so that no artifact in this repository
        ever contains a device count written as a chip count.
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

    def as_negative_control(self) -> Optional[common.NegativeControl]:
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
            executed=False)

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

    @property
    def outcome(self) -> common.Outcome:
        return common.worst(c.outcome for c in self.checks)

    def payload(self) -> Dict[str, Any]:
        return {
            "chip_count": self.chip_count,
            "supplied_basis": self.basis.to_dict(),
            "pinned_basis": self.pinned.to_dict(),
            "injection": dataclasses.asdict(self.injection),
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

    evidence = [
        chip_count_from_coords(views, devices_per_chip),
        chip_count_from_gke(gke_allocation, gke_allocation_unit, devices_per_chip),
    ]
    chip_count, count_check = reconcile_chip_count(evidence)
    if chip_count is not None and injection.chip_count_scale is not None:
        scaled = int(round(chip_count * injection.chip_count_scale))
        count_check = common.check(
            "m6.chip_count", common.Outcome.FAILED,
            "negative control leg 1: the derived chip count was overridden by a "
            f"scale factor of {injection.chip_count_scale}",
            {
                **count_check.intermediates, "derived_chip_count": chip_count,
                "injected_chip_count": scaled
            })
        chip_count = scaled

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
                               injection=injection)
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
                          negative_control=injection.as_negative_control(),
                          extra_provenance={"env": common.env_snapshot(RECORDED_ENV)})
    print(f"M6: {report.outcome.value}; artifact written to {args.out}")
    for item in report.checks:
        print(f"  {item.name}: {item.outcome.value} -- {item.reason}")
    return 0 if report.outcome is common.Outcome.PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
