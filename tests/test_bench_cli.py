"""`gossamer-bench` must run on the reference engine without the compiled one."""
import builtins

import pytest

from gossamer.benchmarks.cli import main


def test_reference_engine_cli_runs_without_leviathan(monkeypatch, tmp_path, capsys):
    real_import = builtins.__import__

    def no_leviathan(name, *a, **k):
        if name == "leviathan" or name.startswith("leviathan."):
            raise ImportError("leviathan deliberately unavailable in this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_leviathan)
    out = tmp_path / "lb.md"
    assert main(["--scenarios", "rendezvous", "--baselines", "random",
                 "--agents", "30", "--steps", "20", "--engine", "reference",
                 "--out", str(out)]) == 0
    assert "engine=reference" in capsys.readouterr().err
    assert out.read_text().strip()


def test_unknown_engine_is_refused():
    with pytest.raises(SystemExit):
        main(["--engine", "warp-drive"])
