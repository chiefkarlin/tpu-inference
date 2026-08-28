#!/bin/bash
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
#
# push_guard.sh -- refuse to push a tree that is not our fork.
#
# WHY THIS EXISTS. Four checkouts of this project sit side by side on the
# campaign workstation and the directory names are anti-correlated with the
# trust relationship: three directories whose name contains "tpu-inference" are
# the upstream we are barred from pushing to, and the one named "nmc" is our
# fork. Navigating by name-matching lands you in the barred repository, where
# `git push origin` is an upstream push.
#
# WHAT IT IS. A machine comparison that aborts, not a printout that a human
# reads. A remote *name* is not a repository *name*, so the check resolves the
# push URL and string-compares it against one literal.
#
# WHAT IT MUST NEVER BECOME. Do not "improve" this by checking that the remote
# is reachable. No `git ls-remote`, no `git fetch`, and specifically no
# `git push --dry-run`: a dry run authenticates, which is a live contact with
# the remote, and against an upstream URL that contact is the exact thing this
# guard exists to prevent. THIS SCRIPT COMPARES STRINGS AND NEVER SPEAKS TO A
# SERVER.
#
# Usage:
#   push_guard.sh <path-to-tree> [remote-name]
#
# Exit status:
#   0  the push URL is our fork
#   2  usage error
#   3  MISMATCH -- the push URL is not our fork. Stop and escalate.
#   4  the push URL could not be resolved at all (UNDETERMINED, not a pass)
#   5  MISMATCH, and the resolved URL is an upstream repository. Stop.

set -euo pipefail

EXPECTED_PUSH_URL="https://github.com/chiefkarlin/tpu-inference.git"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: push_guard.sh <path-to-tree> [remote-name]" >&2
  exit 2
fi

tree="$1"
remote="${2:-origin}"

if ! resolved="$(git -C "$tree" remote get-url --push "$remote" 2>/dev/null)"; then
  echo "PUSH GUARD: UNDETERMINED -- could not resolve the push URL of remote" \
       "'${remote}' in '${tree}'. This is not a pass." >&2
  exit 4
fi

# Belt, before braces. The equality check below already rejects everything that
# is not our fork, so this clause can never be the only thing that fires -- it
# exists so that the barred case is named out loud in the output rather than
# reported as a generic mismatch. Stricter, never looser: anything added here
# in future tightens the guard or leaves it identical.
case "$resolved" in
  *vllm-project*)
    echo "PUSH GUARD: ABORT -- resolved push URL is an UPSTREAM repository." >&2
    echo "  tree     : ${tree}" >&2
    echo "  remote   : ${remote}" >&2
    echo "  resolved : ${resolved}" >&2
    echo "A push from here is an upstream push. We prepare; a human submits." >&2
    exit 5
    ;;
esac

if [ "$resolved" != "$EXPECTED_PUSH_URL" ]; then
  echo "PUSH GUARD: ABORT -- refusing to push." >&2
  echo "  tree     : ${tree}" >&2
  echo "  remote   : ${remote}" >&2
  echo "  resolved : ${resolved}" >&2
  echo "  expected : ${EXPECTED_PUSH_URL}" >&2
  echo "A push from here would leave our fork. Stop and escalate; do not" \
       "re-point, edit or delete the remote to make this pass." >&2
  exit 3
fi

echo "PUSH GUARD: ok -- ${tree} (${remote}) resolves to ${resolved}"
