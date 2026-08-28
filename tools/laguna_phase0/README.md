# Phase 0 measurement plumbing (laguna-tpu)

Instruments, not results. Phase 0 replaces *modelled* magnitudes with *measured*
ones for the v7x decode optimisation effort, and everything in this directory is
part of that plumbing: M0 (warm-cache protocol), M1 (device profile and its two
deciders), M2 (replication), M3 (decode-only microbenchmark), M4 (router-output
histogram) and M6 (denominator assertion).

## STATUS: WRITTEN AND REVIEWED. NOT ADOPTED.

**Nothing in this directory has ever been executed.** No server, no benchmark, no
measurement run, no unit test. That is deliberate: the work was produced without
run authorisation, and a check that has never been made to fail is not an adopted
check.

Two items in particular are only adopted once they have been *made to fail* on
hardware, and making them fail is a run:

| control | what it corrupts | how invoked | executed? |
|---|---|---|---|
| M0 negative control | leaves a padding bucket unwarmed, so the timed window recompiles | `m0_warm_cache.py --nc-leave-bucket-unwarmed <bucket>` | **no** |
| M6 leg 1 | injects a wrong chip count | `m6_denominator.py --nc-chip-count-scale <k>` | **no** |
| M6 leg 2 | injects an inconsistently paired basis (per-device FLOPs against per-chip bandwidth) | `m6_denominator.py --nc-inconsistent-pairing` | **no** |

The presence of a test file in `tests/` is not evidence that the test passes. The
tests have not been collected or run either.

## Standing rules this package enforces in code

1. **Three buckets, never two.** `PASSED`, `FAILED`,
   `COULD_NOT_BE_CHECKED_MECHANICALLY`. An empty input is UNDETERMINED, never a
   pass. See `common.Outcome` and `common.worst`.
2. **Publish intermediates, not verdicts.** `common.check` and
   `common.write_artifact` refuse to emit a disposition that does not carry the
   quantities it was computed from.
3. **Thresholds are inputs with provenance.** Every number lives in
   `thresholds.json` with a `value`, a `kind` and a `source`. A `value` of
   `null` means there is no default: the code raises `ThresholdError` at the
   point of use and the instrument reports UNDETERMINED. No threshold in this
   package was chosen by its author.
4. **Credential values are never written.** `common.env_snapshot` records
   presence and length for anything whose *name* looks like a credential, never
   the value.

## `thresholds.json`

Read it before reading any code: it is the complete list of every number that can
change a verdict here, each with the party it came from. Entries of kind
`deliberately-absent` are thresholds a named party ruled must **not** exist (the
M1 closure residual is emitted, not judged); entries of kind `rejected` are
values that appear in an upstream document and have been overruled, held at
`null` so that no code path can read them.

## `push_guard.sh` -- run this before every push

Four checkouts of this project sit side by side on the campaign workstation and
**the directory names are anti-correlated with the trust relationship**: three
directories named `tpu-inference` are the upstream we are barred from pushing to,
and the one named `nmc` is our fork. A push token with account-wide scope is
present in the environment by default, so an accidental upstream push would
authenticate and land. The guard is the last line of defence, not a nicety.

```sh
tools/laguna_phase0/push_guard.sh /path/to/tree [remote] && git -C /path/to/tree push ...
```

It resolves the push URL and string-compares it against one literal. Exit `0`
ours, `3` mismatch, `4` unresolvable (UNDETERMINED, not a pass), `5` mismatch and
the URL is upstream.

**It never contacts a server.** Do not add a reachability check: `git ls-remote`,
`git fetch` and `git push --dry-run` all authenticate against the remote, and
against an upstream URL that contact is the thing the guard exists to prevent.

Demonstrated aborts (all five legs fired; recorded in the campaign notes):
exit 5 on each of the three upstream trees, exit 3 on a fixture whose URL differs
only by a `.git` suffix, exit 4 on a directory that is not a repository, exit 0
on our fork.

## M6 -- `m6_denominator.py`

Derives the chip count at run time, checks the per-chip roofline pair against
the pinned table, and **fails the run** on mismatch (including on
UNDETERMINED: a run whose denominator was not asserted is not reportable).

Two routes to the chip count, and neither is preferred:

* **JAX device coordinates.** Two chiplets of one chip share coordinates and
  differ in `core_on_chip`, so distinct coordinates count chips. If distinct
  coordinates *equal* the device count, the route abstains rather than return
  the device count under a chip's name.
* **The GKE `google.com/tpu` allocation**, published to the pod through the
  downward API as `LAGUNA_GKE_TPU_ALLOCATION`. **Its unit must be declared**
  (`--gke-allocation-unit chips|devices`). Whether that resource counts chips
  or chiplets on v7x was not established by the author of this module, and an
  undeclared unit makes the route abstain. Guessing it is the factor-of-two
  hazard in its purest form.

Routes that disagree are a FAILURE, not a tie-break. A single route is
UNDETERMINED, not a pass. There is no code path that reads a chip count from a
startup banner.

### The two negative-control legs, and why the second is the one that matters

```sh
python -m tools.laguna_phase0.m6_denominator --out m6.json \
    --nc-chip-count-scale 2          # leg 1: wrong chip count
python -m tools.laguna_phase0.m6_denominator --out m6.json \
    --nc-inconsistent-pairing        # leg 2: per-device FLOPs, per-chip bandwidth
python -m tools.laguna_phase0.m6_denominator --out m6.json \
    --nc-consistent-redenomination   # demonstration: halve both; nothing flips
```

Leg 1 is the design's own control and it exercises the *least* consequential
input. A consistent re-denomination halves FLOPs and bandwidth together; the
ridge point is their quotient and does not move; the classification is unchanged
under all four pairings. What flips an arm is an **inconsistent pairing**, so
`check_pairing` compares the ridge point of the supplied pair against the pinned
ridge, independently of the basis labels -- a label can lie, a quotient cannot.
The third invocation exists to demonstrate that distinction rather than to
control anything: it must leave the pairing check *passing*.

Leg 1 is expressed as a scale factor rather than a literal count so that no
artifact in this repository ever contains a device count written as a chip
count.

**Neither leg has been executed.** They need real JAX coordinates on a real
pod, so they are exactly the class that cannot be made to fail without a run.

## M0 -- `m0_warm_cache.py`

Warm-up before every timed window over the **same padding buckets the window
will use**, `VLLM_XLA_CHECK_RECOMPILATION=1`, and zero recompilation events
inside the window.

**A window that recompiles is VOID, not adjusted.** There is no adjusted path in
the module: nothing subtracts the compile time, annotates it, or carries the
window forward with a caveat.

**Warmth is per pod and per configuration.** The compile cache is on an
emptyDir, so it does not survive a pod restart; and a configuration change
alters the compiled graph set, so a pod warmed under one configuration is not
warm under the next. `warmth_identity()` fingerprints both, and the identity
travels in the evidence.

Warm-up evidence is emitted per window, not asserted: which buckets were warmed
and when, the warmth identity, the environment the protocol depends on, and the
recompilation counter's reading at window start and at window end.

Harness usage:

```python
ledger = WarmupLedger(plan, RecompilationProbe(counter=engine.recompilation_count),
                      configuration=launch_args)
ledger.warm_all(lambda bucket: engine.warm(bucket))
with ledger:            # the timed window
    ...
checks = evaluate_window(ledger.evidence, thresholds)
verdict = window_verdict(checks)     # VALID / VOID / COULD_NOT_BE_CHECKED_MECHANICALLY
```

Offline: `m0_warm_cache.py plan --bucket decode_tokens=32 --out plan.json` and
`m0_warm_cache.py evaluate --evidence ev.json --out m0.json`.

### The counter binding is unconfirmed, and says so

The probe prefers an in-process counter supplied by the harness. Its fallback is
a log scan whose patterns live in `thresholds.json` as
`m0.recompilation_event_log_patterns`, **value null**: no log from a warmed v7x
pod has been read by the author. With neither source the probe returns no
reading and the window is UNDETERMINED -- not zero events. "I saw no evidence"
and "I have evidence of none" are different statements and only one of them
supports a claim.

### Negative control (review finding B7-ii) -- NOT EXECUTED

```sh
python -m tools.laguna_phase0.m0_warm_cache plan \
    --bucket decode_tokens=32 --bucket decode_tokens=64 \
    --nc-leave-bucket-unwarmed decode_tokens=64 --out plan.json
```

The window then uses a bucket that was never warmed, compiles in flight, and is
marked VOID on two independent grounds (the unwarmed bucket and the advancing
counter). M0 gates strictly more than M6 does -- it is a precondition of every
criterion in the design -- and until this control has been run on hardware **M0
is written, not adopted.**

## M2 -- `m2_replication.py`, and `basis.py`

Replicates the Shape B (isl512/osl256) cell at **c32 and at c1**, at least four
replicates each. c1 is review finding B8: without a spread estimate there, two
later comparisons have one side with no precision basis at all -- including the
criterion that separates "the mechanism acted" from "something moved".

* **The raw values are always published.** There is no mode that emits a
  coefficient of variation without the replicates it was computed from.
* **A difference smaller than the measured spread is UNDETERMINED**, reported as
  "not resolvable at this replication". Not "no effect", not "a small effect".
  `compare_with_spread()` emits the spread next to every difference it computes.
* **Too few replicates is UNDETERMINED**, and the cell stays an unreplicated
  point estimate.

`basis.py` is the other half of this and it exists because of review finding
B3: a residual curve was assembled from wall-clock and TPOT-derived points, and
the mix understated a load-bearing term by about 57%. A measured point in this
package is never a bare float -- it is a `LadderPoint` carrying its concurrency,
units, basis and instrument, and every combining operation calls
`require_single_basis()` first. An `UNKNOWN` basis is not a wildcard: it is a
point that cannot be combined with anything.

`plan` renders the replicate invocations without running them; `summarise`
ingests the results, and requires `--basis` on the command line because an
ingest that guesses the basis reintroduces the defect.

## M3 -- `m3_decode_microbench.py`

Fixed batch, pre-filled KV, no HTTP and no prefill inside the timed window, so
that engine step time can be separated from serving-stack overhead.

**The 1 ms threshold in the design is not implemented and cannot be read.**
Review finding B8 rules that a 1 ms threshold on a TPOT-derived quantity, with
no spread estimate anywhere, violates M2's own rule. So:

* the difference between engine step time and served TPOT is emitted as an
  intermediate, always;
* the verdict needs `m3.serving_stack_cost_threshold_ms`, which is **null** in
  `thresholds.json` and must be derived from M2's measured spread at the same
  cell;
* with no threshold the outcome is `COULD_NOT_BE_CHECKED_MECHANICALLY`, and
  that is the expected outcome today;
* a supplied threshold *finer than the measured spread* is refused, for the same
  reason the 1 ms one was.

The design's 1 ms sits in `thresholds.json` as kind `rejected`, value null, so
no code path can read it.

Two enforced properties:

* **A device sync is required.** JAX dispatch is asynchronous, so an unsynced
  step loop measures dispatch rather than execution -- and reports it as
  impossibly fast, which is the flattering direction. `run_step_loop` raises
  without one.
* **The cross-basis comparison is declared.** Engine-step wall clock against
  served TPOT is the measurement, not a mistake, so it goes through
  `basis.declare_cross_basis()` and the crossing appears in the artifact in
  words.

Uninitialised KV is allowed and recorded, but it makes the result UNDETERMINED:
MoE routing depends on the hidden states, so garbage KV can route differently
from a real workload and change the grouped-matmul time. **Direction of that
bias: unknown.** Prefer `kv_prefilled=True`, i.e. one real prefill before the
window opens.

### The engine binding is a named seam

This module owns the protocol, timing, statistics and verdict. The caller
supplies `step()` and `sync()`. The intended in-tree binding is
`TPUModelRunner.execute_model` driven from a decode-only scheduler output whose
requests already have `num_computed_tokens` at the context length, with
`jax.block_until_ready` as the sync -- named here rather than written into the
module, because that call sequence has never been exercised by the author and a
guessed binding that runs is worse than an explicit seam that does not.

## M4 -- `m4_router_histogram.py`

Distinct experts per layer per step, at c1, c16 and c32. It costs no timed run
and, per review finding B4, it gates the whole ranking rather than one
candidate: its result selects the Phase 1 candidate.

* **The per-layer histogram is the output**, alongside per-layer `E` and the
  summary `E`. The summary is a verdict; the histogram is the intermediate.
* **Every ladder point carries its basis on the point** (`router_count`) and the
  counter's units and source next to the value.
* **The c1 and c16 points are emitted in a form that can feed the Phase 1
  naming statistic. This module does not compute that statistic and does not
  name a candidate.**

Acceptance:

| leg | rule | source |
|---|---|---|
| byte model | E(32) within 10% of 183 | design section 7, M4 |
| padding rows do NOT disperse | E(1) in the band 10-20 | design section 7, M4 |
| padding rows DO disperse | E(1) "near 119" -- **no tolerance exists** | -- |

The dispersal leg reports `COULD_NOT_BE_CHECKED_MECHANICALLY` and publishes the
distance from 119. No named party has supplied a tolerance for "near", and
borrowing the 10% that was designed for E(32) would be inventing one;
`m4.e1_disperse_tolerance_fraction` is present in `thresholds.json` with value
null and the escalation recorded in its source field.

A capture that does not record whether padding rows were routed and counted
cannot answer the padding question at all, and says so rather than answering it.

## M1 -- `m1_profile.py`

The device profile emitter and the two-rule decider. It ingests an
already-captured trace and decides; it starts no profiler and runs no model,
which is what makes the decider a pure function of emitted quantities and
therefore checkable against synthetic fixtures with no TPU in the room.

### Two axes, deliberately not collapsed

Each leg reports a `common.Outcome` **and**, where a rule was evaluated, a rule
verdict inside its intermediates.

| axis | question | values |
|---|---|---|
| outcome | could this be decided mechanically from what was emitted? | PASSED / FAILED / COULD_NOT_BE_CHECKED_MECHANICALLY |
| verdict | *which* defined disposition | HOLDS / MIXED / REFUTED, BYTE_BOUND / PARTIAL / EFFICIENCY_DEFICIT / VOID |

A PASSED check can carry a REFUTED verdict: the rule ran and the answer was no.
Folding the axes would make "the rule refuted the hypothesis" and "the rule
could not be run" the same string, and those two must be told apart.

### Rule 1

`f_host = I / S` on the **strict** idle definition, cuts inclusive: `>= 0.45`
HOLDS, `<= 0.12` REFUTED, strictly between MIXED. **MIXED is a defined branch
of the rule, not a failure of it** -- 0.30 is answered, not deferred.

Steps of one run landing in *different* branches is a different fact and is not
allowed to borrow the name MIXED: that returns
COULD_NOT_BE_CHECKED_MECHANICALLY with `branches_present` published.
`f_host` outside `[0, 1]` is an instrument fault and returns FAILED.

### The non-overlap term

`I_loose - I_strict`: DMA in flight with no compute running. **Its own channel.
Never added into `I`, never substituted into Rule 1.** The design asserts it
sits nowhere near the cuts; "near" has no tolerance from any named party and
none is invented here. What is mechanical, and stronger, is the substitution
test: recompute the branch with the loose definition and see whether it moves.
If it moves, the leg FAILS -- the verdict would be a property of a definitional
choice rather than of the machine.

### Rule 2

`e_dec` against `e_ref`, both from the **same profile run** -- the run ids
travel with the values and a mismatch is UNDETERMINED, not a ratio. `e_ref`
below 0.5 makes the cut VOID before the ratio is read at all. Otherwise
`>= 0.85 x e_ref` BYTE_BOUND, `<= 0.60 x e_ref` EFFICIENCY DEFICIT, between
them PARTIAL.

### E6a: closure residual and bucket overlap

```
unattributed = S - (I_strict + every named term + unmapped trace events)
```

emitted per step as a fraction of `S`, first-class, never folded into another
term. Unmapped events are kept rather than dropped: they are exactly the
material of the residual, and discarding them would make it look smaller.

**There is no threshold for the residual and none is invented here.** Both
`m1.closure_residual_threshold` and `m1.bucket_overlap_threshold` are null with
kind `deliberately-absent`. The leg therefore returns only:

* FAILED when the residual is negative -- the named terms sum to more wall time
  than the step contains, which is a double count. Zero is not a tuning
  constant; it is the edge of arithmetic possibility.
* COULD_NOT_BE_CHECKED_MECHANICALLY otherwise, publishing the residual. It
  never returns PASSED, because a pass would assert a standard nobody set.

The **bucket overlap** -- per-term durations summed, minus their union -- is
emitted next to the residual because it is what makes a residual of zero
uninformative: overlapping terms can sum exactly to `S` while double-counting
one region and omitting another. When the overlap is non-zero the reason string
says the residual is not a closure measure. When the overlap was never
measured, the leg says the residual cannot be read as one at all.

### Counter provenance

Every emitted counter carries its units and their source next to the value.
Quantities this module computes from interval arithmetic are confirmed by
construction. `hbm_bytes` is declared **unconfirmed** and emits its unit as
`UNKNOWN (claimed: bytes)` until a caller confirms it against the profiler's
own documentation: a bytes/KiB/elements confusion does not produce an
implausible number, it produces a different verdict. `m1.counter_units_confirmed`
lists every counter still unconfirmed.

`m1.reference_step_ms` exists in `thresholds.json` for orientation only. No leg
reads it, and a test asserts that assessing a run never touches it.
