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
