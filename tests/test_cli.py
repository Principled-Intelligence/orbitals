"""Smoke tests for the `orbitals` Typer CLI.

The CLI is documented in the README as the entry point for running the
self-hosted server: `orbitals scope-guard serve ...`. We just check that the
command tree loads and `--help` runs cleanly for every subcommand.
"""

from __future__ import annotations

from typer.testing import CliRunner


def test_root_help():
    from orbitals.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "scope-guard" in result.stdout


def test_scope_guard_help():
    from orbitals.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["scope-guard", "--help"])
    assert result.exit_code == 0


def test_scope_guard_serve_help():
    from orbitals.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["scope-guard", "serve", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.stdout


def test_scope_guard_convert_default_model_name_resolves_alias():
    from orbitals.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["scope-guard", "convert-default-model-name", "scope-guard-q"],
    )
    assert result.exit_code == 0
    assert "principled-intelligence/scope-guard-4B-q-2601" in result.stdout


def test_scope_guard_convert_default_model_name_passes_through_unknown():
    from orbitals.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["scope-guard", "convert-default-model-name", "my-org/custom-model"],
    )
    assert result.exit_code == 0
    assert "my-org/custom-model" in result.stdout
