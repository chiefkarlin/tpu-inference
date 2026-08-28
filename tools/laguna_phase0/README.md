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
