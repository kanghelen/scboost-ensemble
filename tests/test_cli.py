"""CLI smoke tests via subprocess."""

from __future__ import annotations
import subprocess
import sys


def test_cli_help():
    """`scboost --help` should succeed and list subcommands."""
    r = subprocess.run([sys.executable, "-m", "scboost_ensemble.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for sub in ("fit", "build-latents", "build-concat", "build-from-probs"):
        assert sub in r.stdout, f"subcommand {sub} not in --help"


def test_cli_version():
    r = subprocess.run([sys.executable, "-m", "scboost_ensemble.cli", "--version"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "scboost-ensemble" in r.stdout


def test_cli_fit_help():
    r = subprocess.run([sys.executable, "-m", "scboost_ensemble.cli", "fit", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    for flag in ("--input", "--methods", "--out_dir", "--cv_scheme", "--base_cache_dir"):
        assert flag in r.stdout
