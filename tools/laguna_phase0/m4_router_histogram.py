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
the engineering manager and are gated on M4 actually running.

-------------------------------------------------------------------------------
THE DISPERSAL LEG WAS REPLACED WHOLESALE. READ THIS BEFORE CHANGING IT BACK.
-------------------------------------------------------------------------------

An earlier version of this module classified ``E(1)`` against two anchors: a
stated band of 10-20 and a reference value of 119. BOTH ARE RETIRED, ruled by
the architect at 2026-08-28 17:07Z as DECIDED AND NOT AS AN ABSTENTION, and the
replacement rule below has no tolerance in it anywhere.

WHY 119 WENT, AND WHY A BETTER 119 DOES NOT BRING IT BACK. It is not a
measurement: it is a uniform-routing model evaluated at 16 rows, so it CARRIES
THE EXACT ASSUMPTION M4 EXISTS TO TEST. It is also a CEILING, NOT A CENTRE --
any correlation between rows puts the true value below it, with no lower limit
short of top-k -- so a symmetric band around it is wrong in form and not merely
in width.

  A LATER RULING (OPT-1850) ESTABLISHED THAT 119.14 CAME FROM THE WRONG UNIFORM
  MODEL AND THAT THE CORRECT ONE GIVES 120.68. THAT DOES NOT UN-RETIRE THE
  ANCHOR, AND THE CORRECTED NUMBER IS THE MORE DANGEROUS OF THE TWO. 119.14 was
  easy to retire because it was about to be shown wrong; 120.68 arrives
  independently re-derived and combinatorially correct, and therefore reads as
  safe to put back. THE RETIREMENT WAS NEVER ABOUT THE VALUE BEING WRONG. IT WAS
  ABOUT THE QUANTITY BEING MODELLED AT ALL, AND A BETTER MODEL IS NOT AN ANSWER
  TO THAT. Model A is a ceiling too. See ``UniformModel`` below: the correct
  model is implemented here for LABELLED ILLUSTRATION ONLY and is deliberately
  unreachable from any decision path.

THE REPLACEMENT RULE IS STRUCTURAL. At c1 there is exactly one real token, and
top-k selects exactly top-k DISTINCT experts, so real routing can touch at most
``cut`` distinct experts per MoE layer per step -- by construction, not
approximately:

  cut = top_k_runtime + n_shared_counted

  E(1) == cut on every MoE layer, every step  -> NON-DISPERSAL CONFIRMED. PASSED.
  E(1)  > cut on any  MoE layer, any   step  -> DISPERSAL CONFIRMED. PASSED.
                                                Only padding rows activating
                                                experts the real token did not
                                                can produce an excess.
  E(1)  < cut anywhere                       -> INSTRUMENT FAULT or expert-
                                                capacity dropping. NOT a
                                                hypothesis result. Do not
                                                classify. Report and stop.

Both hypothesis arms now reach PASSED from a config fact rather than from a
model, and the one confirmed most sharply is DISPERSAL -- the damaging one. The
asymmetry the previous version had to declare as a known bias is gone, because
it was created by the anchors and died with them.

THREE THINGS THE COUNTER MUST SAY OR ITS NUMBER MEANS NOTHING.

1. WHICH INSTRUMENTATION POINT IT COUNTED. This architecture has one shared
   expert per MoE layer, always on, outside the router. So ``n_shared_counted``
   is 0 or 1 AND IT IS DETERMINED BY WHERE YOU INSTRUMENT, not by the model:
   count routed GMM/Megablox rows only and it is 0; count experts whose weights
   were read and it is 1. A COUNTER REPORTING 11 IS EITHER A CLEAN SYSTEM
   COUNTING THE SHARED EXPERT OR A DISPERSING SYSTEM THAT IS NOT -- IDENTICAL
   OUTPUT, OPPOSITE VERDICTS. It is not inferable from the number. Never emit a
   bare integer.
2. WHAT TOP-K ACTUALLY WAS IN THE RUNNING MODEL, read from the effective config
   and not from a file on disk. Three values are reachable by reading a real
   file in a real directory: 10, authoritative, from the checkpoint; 8, the
   serving stack's OWN config-class default; and 16, the framework default. A
   wrong constant here does not produce an implausible number, it produces a
   confident wrong verdict -- and the two wrong arms fail in opposite
   directions, one filing a clean result as INSTRUMENT FAULT and the other
   reporting DISPERSAL CONFIRMED on a clean system.
3. ITS WINDOW LENGTH. Distinct counts must be PER STEP, NEVER POOLED. A count
   unioned over a window rises with the step count and will cross the cut with
   certainty, reporting DISPERSAL CONFIRMED on a non-dispersing system. Worse
   than a bias: the downstream statistic is NON-MONOTONIC in the window length,
   so two honest runs at different window lengths return different answers with
   no disagreement between them. A bare number cannot distinguish a per-step
   count from a pooled one, so the window length is emitted with every count.

PROVENANCE OF ITEMS 1 TO 3 AND OF THE 17:07Z RULING. Inherited from
``laguna-tpu-opt-arch`` and received by this module's author as
``bench-campaign/evidence/opt/em-succession.md`` at commit ``64a3e599``,
sections 4 and 6, plus coordinator broadcasts OPT-1850 and OPT-1860 as received.
Per the citation rule, that document is cited and the sources it names were NOT
opened or re-derived here.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import enum
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tools.laguna_phase0 import basis as basis_mod
from tools.laguna_phase0 import common

ROUTER_COUNTER = common.CounterProvenance(
    name="distinct_experts",
    units="distinct expert ids per MoE layer per decode step",
    source=("counted by this module from the router's own top-k index tensor; "
            "the unit is established by construction rather than read out of a "
            "profiler. THE UNIT ALONE DOES NOT MAKE THE NUMBER READABLE: see "
            "the counting basis and window length emitted beside it"),
    units_confirmed=True)


# -----------------------------------------------------------------------------
# 1. WHERE THE COUNTER WAS PLACED, WHICH DECIDES WHETHER THE SHARED EXPERT IS IN
# -----------------------------------------------------------------------------


class ExpertCountBasis(str, enum.Enum):
    """Which instrumentation point produced the distinct count.

    This architecture has ONE SHARED EXPERT PER MoE LAYER, ALWAYS ON, sitting
    outside the router as a separate dense matmul. Whether it lands in the count
    is a fact about the probe, not about the model, and the two probes differ by
    exactly one.
    """

    #: Counts routed GMM / Megablox rows only. The shared expert is a separate
    #: dense matmul and never appears among them: n_shared_counted = 0.
    ROUTED_ROWS = "ROUTED_GMM_OR_MEGABLOX_ROWS"

    #: Counts experts whose weights were read. The shared expert is read on
    #: every token, so it is always in: n_shared_counted = 1.
    WEIGHTS_READ = "EXPERTS_WHOSE_WEIGHTS_WERE_READ"

    #: The probe did not say. The cut cannot be formed and nothing is classified.
    NOT_DECLARED = "NOT_DECLARED"


#: n_shared_counted per basis. ``None`` means undeterminable, which is not zero.
SHARED_COUNTED_BY_BASIS: Mapping[ExpertCountBasis, Optional[int]] = {
    ExpertCountBasis.ROUTED_ROWS: 0,
    ExpertCountBasis.WEIGHTS_READ: 1,
    ExpertCountBasis.NOT_DECLARED: None,
}


# -----------------------------------------------------------------------------
# 2. WHAT TOP-K ACTUALLY WAS
# -----------------------------------------------------------------------------


class TopKSource(str, enum.Enum):
    """Where the top-k value came from, ranked by what it can establish."""

    #: Read out of the model object the server is actually serving with. The
    #: only source that can establish what the running model used.
    RUNNING_MODEL_EFFECTIVE_CONFIG = "RUNNING_MODEL_EFFECTIVE_CONFIG"

    #: Read from a config file on disk. Cannot establish the effective value:
    #: the serving stack may override it, and two wrong values are reachable
    #: this way from real files in real directories.
    CONFIG_FILE = "CONFIG_FILE_ON_DISK"

    #: Baked into code. A defect, reported as one.
    HARDCODED = "HARDCODED_CONSTANT"

    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class TopKReading:
    """One reading of top-k, with where it came from attached to it."""

    value: Optional[int]
    source: TopKSource = TopKSource.UNKNOWN
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        out["source"] = self.source.value
        return out


def check_top_k(reading: TopKReading, thresholds: common.Thresholds) -> common.Check:
    """Asserts top-k, and names what a wrong value would have done.

    Two axes and they do not collapse: WHERE the value came from decides whether
    anything can be established at all, and WHAT it is decides whether the
    assertion holds.
    """
    expected = int(thresholds.require("m4.top_k_expected"))
    known_wrong = dict(thresholds.require("m4.top_k_known_wrong_values"))
    intermediates: Dict[str, Any] = {
        "reading": reading.to_dict(),
        "expected": expected,
        "expected_source": thresholds.entry("m4.top_k_expected").to_dict(),
        "known_wrong_values": known_wrong,
        "why_the_source_matters":
            "both wrong values are reachable by reading a real file in a real "
            "directory, so a reading that parses cleanly is not thereby a "
            "reading of what the running model used",
    }

    if reading.value is None or reading.source is TopKSource.UNKNOWN:
        return common.check(
            "m4.top_k_provenance", common.Outcome.UNDETERMINED,
            "top-k was not established; the cut cannot be formed and no "
            "dispersal verdict is available. An unestablished top-k is not a "
            "reason to fall back on a constant -- falling back is how the "
            "serving stack's own default gets read as the checkpoint's value",
            intermediates)

    if reading.source is TopKSource.HARDCODED:
        return common.check(
            "m4.top_k_provenance", common.Outcome.FAILED,
            f"top-k was taken from a hardcoded constant ({reading.value}). That "
            "is a defect in the instrument regardless of whether the constant "
            "happens to be right today: it cannot notice the day it stops being "
            "right", intermediates)

    if reading.source is TopKSource.CONFIG_FILE:
        return common.check(
            "m4.top_k_provenance", common.Outcome.UNDETERMINED,
            f"top-k = {reading.value} was read from a config file on disk, "
            "which cannot establish what the RUNNING model used. Read it from "
            "the running model's effective config", intermediates)

    if reading.value != expected:
        note = known_wrong.get(str(reading.value))
        consequence = (
            "a clean, non-dispersing system measures below the cut and is filed "
            "as INSTRUMENT FAULT -- a correct result routed into the "
            "do-not-classify bucket" if reading.value < expected else
            "a clean, non-dispersing system measures above the cut and reports "
            "DISPERSAL CONFIRMED")
        return common.check(
            "m4.top_k_provenance", common.Outcome.FAILED,
            f"top-k = {reading.value} from the running model, but {expected} "
            f"was asserted"
            + (f"; {reading.value} is {note}" if note else "")
            + f". Consequence if this is the effective value: {consequence}",
            intermediates)

    return common.check(
        "m4.top_k_provenance", common.Outcome.PASSED,
        f"top-k = {reading.value} read from the running model's effective "
        f"config, matching the asserted {expected}", intermediates)


# -----------------------------------------------------------------------------
# 3. UNIFORM MODELS -- ILLUSTRATION ONLY, REACHABLE FROM NO DECISION PATH
# -----------------------------------------------------------------------------


class UniformModel(str, enum.Enum):
    """The two uniform-routing models that were in circulation.

    Kept as a pair on purpose. Model B is wrong and is retained ONLY so that the
    discriminator below can be executed rather than asserted, because
    REPRODUCING A FORMULA'S ARITHMETIC CANNOT DETECT THE WRONG FORMULA -- it
    tests transcription and nothing else.
    """

    #: 256 * (1 - ((E - k)/E) ** rows). Correct: top-k picks k DISTINCT experts,
    #: so the draw is without replacement.
    A_WITHOUT_REPLACEMENT = "MODEL_A_WITHOUT_REPLACEMENT"

    #: 256 * (1 - ((E - 1)/E) ** (k * rows)). Wrong: treats k*rows independent
    #: draws with replacement.
    B_WITH_REPLACEMENT = "MODEL_B_WITH_REPLACEMENT"


def uniform_distinct_expected(rows: int, top_k: int, num_experts: int,
                              model: UniformModel = UniformModel.A_WITHOUT_REPLACEMENT
                              ) -> Dict[str, Any]:
    """A uniform-routing expectation, LABELLED AND MARKED AS A PLACEHOLDER.

    THIS FUNCTION IS NOT CALLED FROM ANY DECISION PATH AND MUST NOT BE. A
    uniform expectation carries the exact assumption M4 exists to test, and it
    is a CEILING rather than a centre: any correlation between rows puts the
    true value below it, with no lower limit short of top-k. It is here so that
    an illustrative figure, if anyone wants one, is at least the right formula
    wearing its own name.
    """
    if model is UniformModel.A_WITHOUT_REPLACEMENT:
        value = num_experts * (1.0 - ((num_experts - top_k) / num_experts) ** rows)
    else:
        value = num_experts * (1.0 - ((num_experts - 1) / num_experts) ** (top_k * rows))
    return {
        "value": value,
        "model": model.value,
        "rows": rows,
        "top_k": top_k,
        "num_experts": num_experts,
        "status": "PLACEHOLDER -- ILLUSTRATIVE ONLY, NOT A CRITERION",
        "warning":
            "a uniform expectation is a CEILING, NOT A CENTRE, and it carries "
            "the assumption M4 exists to test. Do not build a band around it "
            "and do not compare a measurement to it to reach a verdict",
    }


def discriminate_uniform_models(top_k: int, num_experts: int) -> Dict[str, Any]:
    """Executes the discriminator between the two models instead of asserting it.

    At one row the answer is EXACTLY top-k by construction -- one token, top-k
    distinct experts, no modelling involved. A uniform model that cannot
    reproduce the one value in the table that is CERTAIN rather than MODELLED is
    not the uniform model. Model B returns 9.826 where the answer is 10.
    """
    a = uniform_distinct_expected(1, top_k, num_experts,
                                  UniformModel.A_WITHOUT_REPLACEMENT)["value"]
    b = uniform_distinct_expected(1, top_k, num_experts,
                                  UniformModel.B_WITH_REPLACEMENT)["value"]
    return {
        "certain_value_at_one_row": top_k,
        "why_it_is_certain":
            "one real token, top-k selects top-k DISTINCT experts; this value "
            "is a construction, not a model output",
        "model_a_at_one_row": a,
        "model_b_at_one_row": b,
        "model_a_reproduces_the_certain_value": a == float(top_k),
        "model_b_reproduces_the_certain_value": b == float(top_k),
        "conclusion":
            "MODEL A IS THE UNIFORM MODEL. Model B is retained in code only to "
            "keep this discriminator executable; reproducing a formula's "
            "arithmetic cannot detect the wrong formula",
    }


# -----------------------------------------------------------------------------
# 4. OBSERVATIONS
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RouterObservation:
    """Distinct experts activated in one layer over one counting window.

    Attributes:
      step: Decode step index within the capture.
      layer: Layer index.
      distinct_experts: Count of distinct expert ids the counter saw.
      rows: Number of routed rows in the block, padding included or not per
        ``padding_rows_included`` on the capture.
      window_steps: How many decode steps this count was unioned over. MUST be
        1. It is carried on every observation rather than once on the capture
        because a bare number cannot distinguish a per-step count from a pooled
        one, and the pooled one crosses the cut with certainty.
    """

    step: int
    layer: int
    distinct_experts: int
    rows: Optional[int] = None
    window_steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def observations_from_router_indices(step: int,
                                     indices_by_layer: Mapping[int, Sequence[Sequence[int]]],
                                     window_steps: int = 1
                                     ) -> List[RouterObservation]:
    """Counts distinct experts per layer from top-k index rows, for ONE step.

    Args:
      step: The decode step these indices came from.
      indices_by_layer: Layer index -> rows of selected expert ids, one row per
        routed token (padding rows included or excluded by the caller, and
        declared on the capture).
      window_steps: Recorded, not used. Anything but 1 is a pooled count and the
        checks below refuse it; the parameter exists so that a pooled count can
        be REPRESENTED and rejected rather than silently passing as per-step.
    """
    out: List[RouterObservation] = []
    for layer, rows in sorted(indices_by_layer.items()):
        materialised = [list(row) for row in rows]
        flat = [int(e) for row in materialised for e in row]
        out.append(
            RouterObservation(step=step,
                              layer=layer,
                              distinct_experts=len(set(flat)),
                              rows=len(materialised),
                              window_steps=window_steps))
    return out


@dataclasses.dataclass(frozen=True)
class CountDeclaration:
    """Everything that has to travel with a distinct-expert count.

    Emitted next to every count this module publishes. Without it the number is
    unreadable: 11 is a clean system counting the shared expert or a dispersing
    system that is not, identical output, opposite verdicts.
    """

    basis: ExpertCountBasis
    top_k: TopKReading
    window_steps: Optional[int]
    shared_experts_per_layer: int

    @property
    def n_shared_counted(self) -> Optional[int]:
        return SHARED_COUNTED_BY_BASIS[self.basis]

    @property
    def cut(self) -> Optional[int]:
        """``top_k_runtime + n_shared_counted``, or None if undeterminable."""
        shared = self.n_shared_counted
        if self.top_k.value is None or shared is None:
            return None
        return int(self.top_k.value) + shared

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counting_basis": self.basis.value,
            "n_shared_counted": self.n_shared_counted,
            "shared_experts_per_moe_layer": self.shared_experts_per_layer,
            "top_k": self.top_k.to_dict(),
            "cut": self.cut,
            "cut_formula": "cut = top_k_runtime + n_shared_counted",
            "window_steps": self.window_steps,
            "why_this_travels_with_the_number":
                "a count of 11 is a clean system counting the shared expert or "
                "a dispersing system that is not -- identical output, opposite "
                "verdicts -- and a count without its window length cannot be "
                "told apart from a pooled one",
        }


@dataclasses.dataclass
class Capture:
    """One concurrency's worth of router observations, and what it means.

    Attributes:
      concurrency: The ladder point.
      shape: Benchmark shape, recorded on every emitted point.
      observations: Per layer, per step.
      count_basis: WHICH INSTRUMENTATION POINT produced the counts. Defaults to
        NOT_DECLARED, which makes the cut unformable -- deliberately, so that an
        undeclared probe cannot reach a verdict by omission.
      top_k: The runtime top-k reading.
      padding_rows_included: Whether padding rows were routed and counted.
        ``None`` means it was not recorded, which makes the padding question
        unanswerable from this capture -- and that question is what M4 is for.
    """

    concurrency: int
    shape: str
    observations: List[RouterObservation]
    count_basis: ExpertCountBasis = ExpertCountBasis.NOT_DECLARED
    top_k: TopKReading = dataclasses.field(
        default_factory=lambda: TopKReading(value=None))
    padding_rows_included: Optional[bool] = None

    def window_steps(self) -> Optional[int]:
        """The single window length in use, or ``None`` if it is not single."""
        seen = {o.window_steps for o in self.observations}
        return seen.pop() if len(seen) == 1 else None

    def declaration(self, shared_per_layer: int = 1) -> CountDeclaration:
        return CountDeclaration(basis=self.count_basis,
                                top_k=self.top_k,
                                window_steps=self.window_steps(),
                                shared_experts_per_layer=shared_per_layer)

    def per_layer_histogram(self) -> Dict[int, Dict[int, int]]:
        """Layer -> {distinct-expert count -> how many steps showed it}."""
        hist: Dict[int, Dict[int, int]] = collections.defaultdict(
            lambda: collections.defaultdict(int))
        for obs in self.observations:
            hist[obs.layer][obs.distinct_experts] += 1
        return {layer: dict(sorted(counts.items())) for layer, counts in sorted(hist.items())}

    def per_layer_e(self) -> Dict[int, float]:
        """Layer -> mean distinct experts per step. ``E`` per layer.

        A MEAN, AND NO VERDICT IS TAKEN FROM IT. Averaging over steps is a
        summary for the ladder; the dispersal rule runs over every individual
        (layer, step) observation, because a mean sitting exactly on the cut is
        consistent with half the steps above it.
        """
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
                "count_declaration": self.declaration().to_dict(),
                "padding_rows_included": self.padding_rows_included,
                "layers": len(self.per_layer_e()),
                "steps": len({o.step for o in self.observations}),
                "summary_is_a_mean_over_steps_and_layers_not_a_union":
                    "no verdict is taken from this value; see the per-layer, "
                    "per-step observations",
            })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "shape": self.shape,
            "count_declaration": self.declaration().to_dict(),
            "padding_rows_included": self.padding_rows_included,
            "per_layer_histogram": {str(k): {str(kk): vv for kk, vv in v.items()}
                                    for k, v in self.per_layer_histogram().items()},
            "per_layer_e": {str(k): v for k, v in self.per_layer_e().items()},
            "summary_e": common.counted(self.summary_e(), ROUTER_COUNTER),
            "observations": [o.to_dict() for o in self.observations],
        }


# -----------------------------------------------------------------------------
# 5. THE CHECKS THAT HAVE TO PASS BEFORE A COUNT MEANS ANYTHING
# -----------------------------------------------------------------------------


def check_window_length(capture: Capture,
                        thresholds: common.Thresholds) -> common.Check:
    """Per-step, never pooled -- and the window length must be on the count."""
    required = int(thresholds.require("m4.window_steps_required"))
    seen = sorted({o.window_steps for o in capture.observations})
    intermediates = {
        "required_window_steps": required,
        "window_steps_seen": seen,
        "concurrency": capture.concurrency,
        "observations": len(capture.observations),
        "why":
            "a count unioned over a window rises with the step count and will "
            "cross the cut with certainty, reporting DISPERSAL CONFIRMED on a "
            "non-dispersing system. Worse than a bias: the downstream statistic "
            "is NON-MONOTONIC in the window length, so two honest runs at "
            "different window lengths return different answers with no "
            "disagreement between them",
    }
    if not capture.observations:
        return common.check(
            "m4.window_length", common.Outcome.UNDETERMINED,
            "the capture is empty, so no window length was recorded; an empty "
            "input is undetermined and never a pass", intermediates)
    if seen != [required]:
        return common.check(
            "m4.window_length", common.Outcome.FAILED,
            f"counts were taken over window lengths {seen}, and only "
            f"{required} is per-step. A pooled count is not a noisy per-step "
            "count, it is a different quantity", intermediates)
    return common.check(
        "m4.window_length", common.Outcome.PASSED,
        f"every count is over a window of {required} step and says so",
        intermediates)


def check_count_basis(capture: Capture,
                      thresholds: common.Thresholds) -> common.Check:
    """The counter must declare which instrumentation point it counted."""
    shared = int(thresholds.require("m4.shared_experts_per_moe_layer"))
    declaration = capture.declaration(shared)
    intermediates = {
        "count_declaration": declaration.to_dict(),
        "concurrency": capture.concurrency,
        "shared_expert_provenance":
            thresholds.entry("m4.shared_experts_per_moe_layer").to_dict(),
        "the_ambiguity_this_resolves":
            "with top-k = 10 and one always-on shared expert, a reported 11 is "
            "a clean system counting the shared expert OR a dispersing system "
            "that is not. Identical output, opposite verdicts",
    }
    if capture.count_basis is ExpertCountBasis.NOT_DECLARED:
        return common.check(
            "m4.count_basis", common.Outcome.UNDETERMINED,
            "the counter did not declare which instrumentation point it "
            "counted, so n_shared_counted is unknown and the cut cannot be "
            "formed. This is NOT resolved by assuming 0: assuming the shared "
            "expert is out when it is in reports DISPERSAL CONFIRMED on every "
            "clean run", intermediates)
    return common.check(
        "m4.count_basis", common.Outcome.PASSED,
        f"the counter declares basis {capture.count_basis.value}, so "
        f"n_shared_counted = {declaration.n_shared_counted} and the cut is "
        f"{declaration.cut}", intermediates)


def check_moe_layer_coverage(capture: Capture,
                             thresholds: common.Thresholds) -> common.Check:
    """Did the counter see every MoE layer?

    The expected layer count is INHERITED and is not audited here. A mismatch is
    therefore reported as undetermined, not failed: it means either the probe
    missed layers or the inherited count is wrong, and this module cannot tell
    which without opening somebody else's tree.
    """
    expected = int(thresholds.require("m4.moe_layer_count"))
    layers = sorted({o.layer for o in capture.observations})
    intermediates = {
        "expected_moe_layers": expected,
        "expected_source": thresholds.entry("m4.moe_layer_count").to_dict(),
        "layers_observed": len(layers),
        "layer_indices": layers,
        "concurrency": capture.concurrency,
    }
    if len(layers) == expected:
        return common.check(
            "m4.moe_layer_coverage", common.Outcome.PASSED,
            f"all {expected} MoE layers appear in the capture", intermediates)
    return common.check(
        "m4.moe_layer_coverage", common.Outcome.UNDETERMINED,
        f"{len(layers)} layers observed against an expected {expected}; either "
        "the probe missed layers or the inherited layer count is wrong, and "
        "this module cannot tell which. The per-layer rule below still runs "
        "over the layers that are present", intermediates)


def check_byte_model(capture: Capture, thresholds: common.Thresholds) -> common.Check:
    """E(32) within 10% of 183 => the byte model stands.

    UNCHANGED BY THE 17:07Z RULING, deliberately: that ruling retired the c1
    decision anchor and this is a different leg with a different owner. The
    reference and its tolerance are the design's, and the tolerance is attached
    to THIS quantity at THIS concurrency and travels nowhere else.

    ONE THING IS PUBLISHED BESIDE IT AND NOTHING IS ACTED ON. The reference 183
    reproduces the WRONG uniform model (model B) at 32 rows to three digits,
    where the correct model gives 184.47. That is emitted as a labelled
    intermediate and escalated; it is NOT swapped in, because the constant is
    the architect's to move.
    """
    reference = float(thresholds.require("m4.e32_reference"))
    tolerance = float(thresholds.require("m4.e32_tolerance_fraction"))
    num_experts = int(thresholds.require("m4.num_experts"))
    observed = capture.summary_e()
    top_k = capture.top_k.value

    intermediates: Dict[str, Any] = {
        "observed_e": common.counted(observed, ROUTER_COUNTER),
        "count_declaration": capture.declaration().to_dict(),
        "reference": reference,
        "reference_source": thresholds.entry("m4.e32_reference").to_dict(),
        "tolerance_fraction": tolerance,
        "tolerance_scope":
            "attached to this quantity at this concurrency; importing it onto a "
            "one-sided quantity at a different concurrency is what the M4 "
            "ruling exists to prevent",
        "per_layer_e": {str(k): v for k, v in capture.per_layer_e().items()},
        "concurrency": capture.concurrency,
    }
    if top_k is not None:
        intermediates["uniform_expectation_illustrative_only"] = (
            uniform_distinct_expected(rows=capture.concurrency, top_k=int(top_k),
                                      num_experts=num_experts,
                                      model=UniformModel.A_WITHOUT_REPLACEMENT))
        intermediates["provenance_flag_on_the_reference"] = (
            "183 reproduces MODEL B (with replacement, the wrong model) at 32 "
            "rows to three digits (182.83); MODEL A gives 184.47. IF the "
            "reference were re-pointed at model A, THEN the pass band would "
            "move by about 0.9% of itself -- that is a SENSITIVITY, NOT A "
            "MEASUREMENT, and no re-pointing is done here. Escalated to the "
            "engineering manager")

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


# -----------------------------------------------------------------------------
# 6. THE DISPERSAL RULE -- STRUCTURAL, NO TOLERANCE ANYWHERE IN IT
# -----------------------------------------------------------------------------


class DispersalVerdict(str, enum.Enum):
    """What the structural rule concluded. Not the same axis as the Outcome."""

    NON_DISPERSAL_CONFIRMED = "NON_DISPERSAL_CONFIRMED"
    DISPERSAL_CONFIRMED = "DISPERSAL_CONFIRMED"

    #: E below the cut somewhere. Instrument fault or expert-capacity dropping.
    #: NOT a hypothesis result: report and stop.
    NOT_A_HYPOTHESIS_RESULT = "INSTRUMENT_FAULT_OR_CAPACITY_DROPPING"

    #: The cut could not be formed, or the capture could not be trusted.
    NOT_CLASSIFIED = "NOT_CLASSIFIED"


def classify_dispersal(observations: Sequence[RouterObservation],
                       cut: Optional[int]) -> Tuple[DispersalVerdict, Dict[str, Any]]:
    """The structural rule, applied per MoE layer PER STEP.

    Not to the mean. A mean sitting exactly on the cut is consistent with half
    the steps above it, and the excess is the entire phenomenon.

    Order matters: the below-cut arm is tested over the WHOLE capture first,
    because a fault anywhere poisons the classification everywhere.
    """
    evidence: Dict[str, Any] = {
        "cut": cut,
        "observations_examined": len(observations),
    }
    if cut is None:
        evidence["reason"] = "the cut could not be formed"
        return DispersalVerdict.NOT_CLASSIFIED, evidence
    if not observations:
        evidence["reason"] = "no observations"
        return DispersalVerdict.NOT_CLASSIFIED, evidence

    below = [o for o in observations if o.distinct_experts < cut]
    above = [o for o in observations if o.distinct_experts > cut]
    evidence["below_cut_count"] = len(below)
    evidence["above_cut_count"] = len(above)
    evidence["at_cut_count"] = len(observations) - len(below) - len(above)
    evidence["excess_over_cut_by_layer"] = {
        str(layer): sorted({o.distinct_experts - cut
                            for o in observations if o.layer == layer})
        for layer in sorted({o.layer for o in observations})
    }
    evidence["first_below_cut"] = below[0].to_dict() if below else None
    evidence["first_above_cut"] = above[0].to_dict() if above else None

    if below:
        return DispersalVerdict.NOT_A_HYPOTHESIS_RESULT, evidence
    if above:
        return DispersalVerdict.DISPERSAL_CONFIRMED, evidence
    return DispersalVerdict.NON_DISPERSAL_CONFIRMED, evidence


def check_padding_dispersal(capture: Capture, thresholds: common.Thresholds,
                            cut_established: bool) -> common.Check:
    """Do padding rows disperse across experts at c1?

    Args:
      cut_established: whether the top-k and basis checks both passed. A
        comparison against an unestablished cut is not a weak verdict, it is not
        a verdict, so this is passed in rather than re-derived: the two axes
        stay separate and the reader can see which one refused.
    """
    shared = int(thresholds.require("m4.shared_experts_per_moe_layer"))
    declaration = capture.declaration(shared)
    verdict, evidence = classify_dispersal(capture.observations, declaration.cut)

    intermediates: Dict[str, Any] = {
        "count_declaration": declaration.to_dict(),
        "observed_summary_e": common.counted(capture.summary_e(), ROUTER_COUNTER),
        "per_layer_e": {str(k): v for k, v in capture.per_layer_e().items()},
        "per_layer_histogram": {str(k): {str(kk): vv for kk, vv in v.items()}
                                for k, v in capture.per_layer_histogram().items()},
        "padding_rows_included": capture.padding_rows_included,
        "concurrency": capture.concurrency,
        "rule": ("structural, no tolerance: E == cut on every layer and step is "
                 "NON-DISPERSAL; E > cut anywhere is DISPERSAL; E < cut anywhere "
                 "is not a hypothesis result"),
        "retired_anchors":
            "the 10-20 band and the 119 reference are RETIRED FROM THE DECISION "
            "PATH (architect, 2026-08-28 17:07Z). No distance to either is "
            "published, because publishing it would imply it still means "
            "something. Correcting the uniform model does not un-retire them: "
            "the objection was that the quantity was MODELLED AT ALL, and it is "
            "a ceiling rather than a centre, neither of which a better model "
            "fixes",
        "evidence": evidence,
        "verdict": verdict.value,
    }

    if not cut_established:
        intermediates["verdict"] = DispersalVerdict.NOT_CLASSIFIED.value
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            "the cut is not established -- see the top-k and counting-basis "
            "checks -- so no comparison against it is a verdict. The counts are "
            "published above unclassified", intermediates)

    if capture.padding_rows_included is not True:
        intermediates["verdict"] = DispersalVerdict.NOT_CLASSIFIED.value
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            "the capture does not record that padding rows were routed and "
            "counted, and whether padding rows disperse IS the question",
            intermediates)

    if verdict is DispersalVerdict.NOT_CLASSIFIED:
        return common.check(
            "m4.padding_dispersal", common.Outcome.UNDETERMINED,
            f"nothing to classify: {evidence.get('reason', 'unspecified')}",
            intermediates)

    if verdict is DispersalVerdict.NOT_A_HYPOTHESIS_RESULT:
        return common.check(
            "m4.padding_dispersal", common.Outcome.FAILED,
            f"{evidence['below_cut_count']} observation(s) fall BELOW the cut "
            f"of {declaration.cut}, which real routing cannot do: top-k selects "
            "top-k distinct experts by construction. INSTRUMENT FAULT or "
            "expert-capacity dropping. This is NOT a hypothesis result and is "
            "deliberately not filed as one, and it is not filed as "
            "'could not be checked' either -- it was checked, and what was "
            "found is a defect", intermediates)

    if verdict is DispersalVerdict.DISPERSAL_CONFIRMED:
        return common.check(
            "m4.padding_dispersal", common.Outcome.PASSED,
            f"{evidence['above_cut_count']} observation(s) exceed the cut of "
            f"{declaration.cut}. Only padding rows activating experts the real "
            "token did not can produce an excess: DISPERSAL CONFIRMED",
            intermediates)

    return common.check(
        "m4.padding_dispersal", common.Outcome.PASSED,
        f"every observation equals the cut of {declaration.cut} exactly: "
        "padding rows do NOT disperse. NON-DISPERSAL CONFIRMED", intermediates)


# -----------------------------------------------------------------------------
# 7. THE NEGATIVE CONTROL, AT BOTH c1 AND c16, AND IT IS MADE TO FAIL
# -----------------------------------------------------------------------------
#
# WHY BOTH POINTS AND NOT ONE. The downstream statistic's sensitivities to E(1)
# and E(16) have OPPOSITE SIGNS and largely cancel: a counter that miscounts
# UNIFORMLY moves it about four times less than one that miscounts DIFFERENTLY
# at c1 and c16. And since dispersal is a c1 phenomenon by construction -- the
# block is 16 rows, so c1 has 1 real token and 15 padding rows while c16 has 16
# real tokens and NO padding rows at all -- the error class the statistic is
# most exposed to is exactly the shape of the phenomenon under test. A
# single-point control tests the wrong thing.
#
# WHAT THE FAILURE MODE GRANTS. A counter stuck at the cut grants NON-DISPERSAL,
# the comfortable answer, and would pass the non-dispersal leg perfectly on
# every layer forever. So the control feeds a case whose distinct count is KNOWN
# and is NOT the cut, and confirms the counter does not report the cut.


@dataclasses.dataclass(frozen=True)
class CounterControlCase:
    """A synthetic routing block whose distinct-expert count is known."""

    concurrency: int
    real_rows: int
    padding_rows: int
    known_distinct: int
    top_k: int

    @property
    def rows(self) -> int:
        return self.real_rows + self.padding_rows

    def indices(self) -> List[List[int]]:
        """Rows of expert ids whose union is exactly ``known_distinct`` ids.

        Deterministic and boring on purpose: ids ``0 .. known_distinct-1`` dealt
        round-robin, so every id appears at least once and no id outside the set
        ever appears.
        """
        if self.known_distinct > self.rows * self.top_k:
            raise ValueError(
                f"cannot place {self.known_distinct} distinct ids in "
                f"{self.rows} rows of {self.top_k}")
        return [[(r * self.top_k + i) % self.known_distinct
                 for i in range(self.top_k)] for r in range(self.rows)]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


#: Distinct counts for the two control legs. ARBITRARY BY CONSTRUCTION, and
#: that is the point: they are neither measurements nor model outputs, they are
#: numbers chosen only because they are not the cut and not each other. Nothing
#: downstream may read meaning into them.
CONTROL_KNOWN_DISTINCT = {1: 37, 16: 71}


def default_control_cases(top_k: int, block_rows: int = 16
                          ) -> List[CounterControlCase]:
    """One control at c1 and one at c16.

    THE BLOCK IS ``block_rows`` ROWS AT BOTH POINTS, and that single fact is
    what makes dispersal a c1 phenomenon: at c1 there is 1 real token and
    ``block_rows - 1`` padding rows, at c16 there are 16 real tokens and NO
    padding rows at all. ``block_rows`` is a CONFIGURATION FACT, inherited, not
    a measurement and not a model output; it is a parameter here rather than a
    literal so that a different block size cannot silently keep the old
    padding-row count.
    """
    return [
        CounterControlCase(concurrency=1, real_rows=1,
                           padding_rows=block_rows - 1,
                           known_distinct=CONTROL_KNOWN_DISTINCT[1],
                           top_k=top_k),
        CounterControlCase(concurrency=16, real_rows=block_rows, padding_rows=0,
                           known_distinct=CONTROL_KNOWN_DISTINCT[16],
                           top_k=top_k),
    ]


def stuck_at_cut_counter(case: CounterControlCase, cut: int
                         ) -> List[RouterObservation]:
    """A DELIBERATELY BROKEN counter that always reports the cut.

    Exists so the control can be made to fail. A control that has never rejected
    anything is indistinguishable from a control that cannot.
    """
    return [RouterObservation(step=0, layer=0, distinct_experts=cut,
                              rows=case.rows, window_steps=1)]


def run_counter_control(case: CounterControlCase, cut: int,
                        counter=None) -> Dict[str, Any]:
    """Runs one control leg. Synthetic input only; nothing touches hardware."""
    if counter is None:
        observations = observations_from_router_indices(
            step=0, indices_by_layer={0: case.indices()}, window_steps=1)
    else:
        observations = counter(case, cut)
    reported = observations[0].distinct_experts if observations else None
    detected = (reported == case.known_distinct and reported != cut)
    return {
        "case": case.to_dict(),
        "cut": cut,
        "known_distinct": case.known_distinct,
        "reported_distinct": reported,
        "reported_the_cut_instead": reported == cut,
        "passed": detected,
        "what_the_failure_mode_would_grant":
            "a counter stuck at the cut grants NON-DISPERSAL, the comfortable "
            "answer, on every layer forever",
    }


def check_counter_controls(top_k: int, cut: int,
                           block_rows: int = 16) -> common.Check:
    """Runs both control legs and the stuck-at-cut mutant against both."""
    cases = default_control_cases(top_k, block_rows)
    honest = [run_counter_control(c, cut) for c in cases]
    mutant = [run_counter_control(c, cut, counter=stuck_at_cut_counter)
              for c in cases]
    intermediates = {
        "honest_counter_legs": honest,
        "stuck_at_cut_mutant_legs": mutant,
        "concurrencies_controlled": [c.concurrency for c in cases],
        "padded_block_rows": block_rows,
        "known_distinct_counts_are_arbitrary":
            "chosen only because they are not the cut; neither measurements "
            "nor model outputs, and nothing downstream may read meaning into "
            "them",
        "why_both_points":
            "the downstream statistic's sensitivities to E(1) and E(16) have "
            "opposite signs and largely cancel, so it is about four times more "
            "exposed to a counter that miscounts DIFFERENTLY at the two points "
            "than to one that miscounts uniformly -- and dispersal is a c1 "
            "phenomenon by construction, so the differential error class is "
            "exactly the shape of the phenomenon under test",
    }
    if len(honest) < 2:
        return common.check("m4.counter_negative_control",
                            common.Outcome.UNDETERMINED,
                            "fewer than two control points were run",
                            intermediates)
    if not all(leg["passed"] for leg in honest):
        failed = [leg["case"]["concurrency"] for leg in honest if not leg["passed"]]
        return common.check(
            "m4.counter_negative_control", common.Outcome.FAILED,
            f"the counter did not recover the known distinct count at c{failed}",
            intermediates)
    if any(leg["passed"] for leg in mutant):
        return common.check(
            "m4.counter_negative_control", common.Outcome.FAILED,
            "the stuck-at-cut counter was NOT detected by the control, so the "
            "control cannot reject the one failure mode it exists to reject",
            intermediates)
    return common.check(
        "m4.counter_negative_control", common.Outcome.PASSED,
        "the counter recovered the known distinct count at both c1 and c16, and "
        "the stuck-at-cut counter was rejected at both -- the control has been "
        "made to fail and did", intermediates)


# -----------------------------------------------------------------------------
# 8. ASSEMBLY
# -----------------------------------------------------------------------------


def _tagged(item: common.Check, tag: str) -> common.Check:
    """Renames a check so its ladder point travels with its disposition."""
    return dataclasses.replace(item, name=item.name + tag)


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

    shared = int(thresholds.require("m4.shared_experts_per_moe_layer"))
    controlled: Optional[Tuple[int, int]] = None
    for capture in captures:
        # EVERY CHECK IS NAMED WITH ITS LADDER POINT. Three captures produce
        # three copies of each check with different intermediates, and an
        # unqualified name makes them indistinguishable in a log -- which is
        # how a c16 pass gets read as covering c1.
        tag = f"@c{capture.concurrency}"
        top_k_check = _tagged(check_top_k(capture.top_k, thresholds), tag)
        basis_check = _tagged(check_count_basis(capture, thresholds), tag)
        window_check = _tagged(check_window_length(capture, thresholds), tag)
        checks.extend([top_k_check, basis_check, window_check,
                       _tagged(check_moe_layer_coverage(capture, thresholds), tag)])
        cut_established = all(
            c.outcome is common.Outcome.PASSED
            for c in (top_k_check, basis_check, window_check))
        if capture.concurrency == 32:
            checks.append(_tagged(check_byte_model(capture, thresholds), tag))
        if capture.concurrency == 1:
            checks.append(_tagged(
                check_padding_dispersal(capture, thresholds, cut_established),
                tag))
        cut = capture.declaration(shared).cut
        if cut is not None and capture.top_k.value is not None:
            controlled = (int(capture.top_k.value), cut)

    # The control exercises the counter itself on synthetic input, so it is run
    # ONCE rather than re-run identically per capture: repeating an unchanged
    # control does not make it stronger, it makes a reader think three things
    # were checked.
    if controlled is None:
        checks.append(common.check(
            "m4.counter_negative_control", common.Outcome.UNDETERMINED,
            "no capture established a cut, so the counter control has nothing "
            "to compare against and was not run",
            {"captures": len(captures)}))
    else:
        checks.append(check_counter_controls(
            controlled[0], controlled[1],
            int(thresholds.require("m4.padded_block_rows"))))
    return checks


def capture_from_dict(data: Mapping[str, Any]) -> Capture:
    top_k = data.get("top_k") or {}
    return Capture(
        concurrency=int(data["concurrency"]),
        shape=data.get("shape", "unspecified"),
        count_basis=ExpertCountBasis(
            data.get("count_basis", ExpertCountBasis.NOT_DECLARED.value)),
        top_k=TopKReading(
            value=(None if top_k.get("value") is None else int(top_k["value"])),
            source=TopKSource(top_k.get("source", TopKSource.UNKNOWN.value)),
            detail=top_k.get("detail", "")),
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
            "uniform_model_discriminator": discriminate_uniform_models(
                top_k=int(thresholds.require("m4.top_k_expected")),
                num_experts=int(thresholds.require("m4.num_experts"))),
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
    return common.exit_code(common.worst(c.outcome for c in checks))


if __name__ == "__main__":
    raise SystemExit(main())
