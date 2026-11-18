#!/usr/bin/env python3
"""Do two wheels carry the same package code?

    python .github/scripts/same_package_code.py NEW.whl PUBLISHED.whl

Used by cloudbuild.publish.yaml. The publish trigger runs on every push to main,
and Artifact Registry refuses to re-upload an existing version (HTTP 400), so a
merge that touches only CI, docs or packaging metadata used to fail the build.
This decides whether that refusal matters:

* exit 0: every file under the `gossamer/` package is byte-identical in both
  wheels. The published wheel already ships this code; nothing needs uploading.
* exit 1: the package code differs under the same version number. That is the
  stale-wheel trap (a consumer pinned to this version would silently get the
  old code), so the build must fail and the version must be bumped.

Only the package's own files are compared: `*.dist-info/` (METADATA, RECORD,
WHEEL) legitimately differs between two builds of the same code.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile

PACKAGE = "gossamer/"


def package_files(path: str) -> dict:
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest()
                for n in z.namelist() if n.startswith(PACKAGE) and not n.endswith("/")}


def main(argv: list) -> int:
    if len(argv) != 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    new, old = package_files(argv[1]), package_files(argv[2])
    if not new:
        print(f"no {PACKAGE} files in {argv[1]}; refusing to call that identical", file=sys.stderr)
        return 2
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(f for f in set(new) & set(old) if new[f] != old[f])
    if not (added or removed or changed):
        print(f"identical: {len(new)} files under {PACKAGE}")
        return 0
    for label, files in (("added", added), ("removed", removed), ("changed", changed)):
        for f in files[:20]:
            print(f"  {label}: {f}")
    print(f"package code differs ({len(added)} added, {len(removed)} removed, "
          f"{len(changed)} changed)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
