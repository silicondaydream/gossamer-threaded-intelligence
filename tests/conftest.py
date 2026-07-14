"""ARBORIA_REQUIRE_TESTS — a skip is a failure when the run is supposed to enforce.

Gossamer's optional dependencies are reached through `pytest.importorskip` (torch for the
MAPPO/learning baselines, ortools for the MILP comparator). That is the right default on a
laptop and the wrong one in CI: a missing dependency does not fail the run, it DELETES the
test, and the suite goes green having never executed it — this stack's signature failure
mode, a no-op that reads as coverage. It is the same hole `LEVIATHAN_REQUIRE_TESTS=ON`
closes on the C++ side.

Gossamer's suite is, as of this writing, the one that was already honest: the workflow
installs both extras, so all 292 tests actually run and nothing skips. This guard is here
to keep it that way. The failure it prevents is not today's — it is the `importorskip`
someone adds next year alongside a dependency the workflow never learns about, which will
subtract a test from CI without subtracting anything from the green tick.

Corollary: do not add an `importorskip` to buy a green tick. In CI there is no such thing
as an optional dependency — either the test runs or the build is red. If a comparator is
worth shipping in `extras_require`, it is worth installing in the job that checks it.
"""
import os

import pytest

_REQUIRE_TESTS = os.environ.get("ARBORIA_REQUIRE_TESTS") == "1"

_SKIP_IS_FAILURE = (
    "ARBORIA_REQUIRE_TESTS=1, so a skipped test is a failed one: this run is the one "
    "that ENFORCES the suite. Something this test needs is absent from the "
    "environment. Install it (or fix the workflow) — do not let the suite go green "
    "without having run this.\n\nThe original skip was: {reason}"
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    # `wasxfail` is a declared expectation, not an absent dependency. Leave it alone.
    if _REQUIRE_TESTS and report.skipped and not hasattr(report, "wasxfail"):
        report.outcome = "failed"
        report.longrepr = _SKIP_IS_FAILURE.format(reason=report.longrepr)


@pytest.hookimpl(hookwrapper=True)
def pytest_collectreport(report):
    # A MODULE-level importorskip skips during COLLECTION and never reaches makereport
    # above — which is how both of this suite's gates (torch, ortools) are written.
    if _REQUIRE_TESTS and report.skipped:
        report.outcome = "failed"
        report.longrepr = _SKIP_IS_FAILURE.format(reason=report.longrepr)
    yield
