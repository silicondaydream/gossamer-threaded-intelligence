#!/usr/bin/env bash
# Fail a PR that changes the shipped package without bumping its version.
#
# Artifact Registry refuses to overwrite a published version (HTTP 400), so a
# package change that keeps the old version cannot be released at all -- and a
# release attempted anyway fails at the very last step of the build. That
# happened with 0.13.0. This check moves the failure to review time.
#
# Usage: check_version_bump.sh <base-ref>     (e.g. origin/main)
set -euo pipefail
base="${1:?usage: $0 <base-ref>}"

version_at() { git show "$1:setup.py" | sed -n 's/^[[:space:]]*version="\([^"]*\)".*/\1/p' | head -1; }

changed=$(git diff --name-only "$base"...HEAD -- gossamer/ | grep -v '/tests/' || true)
if [ -z "$changed" ]; then
  echo "no package files changed; no version bump needed"
  exit 0
fi
old=$(version_at "$base")
new=$(version_at HEAD)
echo "package files changed:"; echo "$changed" | sed 's/^/  /'
echo "setup.py version: $old -> $new"
if [ "$old" = "$new" ]; then
  echo "::error file=setup.py::gossamer/ changed but setup.py still says $old. Artifact Registry will refuse to publish it (a re-upload is HTTP 400). Bump the version (maneuver-map/scripts/bump.sh gossamer X.Y.Z)."
  exit 1
fi
# A version must move forward, or the "new" wheel sorts below the old one.
if [ "$(printf '%s\n%s\n' "$old" "$new" | sort -V | tail -1)" != "$new" ]; then
  echo "::error file=setup.py::version went backwards: $old -> $new"
  exit 1
fi
