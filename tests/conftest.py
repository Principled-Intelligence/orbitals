import pytest


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch):
    """Keep the suite off the network and off the developer's HF Hub cache.

    The ScopeGuard V2 prompt-drift check resolves `system_prompt.txt` through
    `huggingface_hub`, and several tests construct guards with placeholder model
    names such as `"m"`. Without this the suite would issue real requests and its
    result would depend on whoever owns that repo. Tests that exercise the lookup
    re-patch these two names themselves; a later `monkeypatch.setattr` wins.
    """
    try:
        import huggingface_hub
    except ImportError:  # api-only install; nothing to block
        return

    def _offline_download(**kwargs):
        # An OSError is what a real offline run raises, so the check goes quiet
        # exactly as it would on a machine with no network.
        raise OSError("HF Hub access is disabled in tests")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda **kw: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _offline_download)


@pytest.fixture(autouse=True)
def _plain_cli_help(monkeypatch):
    """Render Typer's help without ANSI styling so substring assertions hold.

    Typer derives `FORCE_TERMINAL` from `FORCE_COLOR`, which plenty of
    terminals and CI runners export, and rich then styles the help panel even
    though `CliRunner` captures to a pipe. Typer's highlighter matches an
    option name twice over: `--output-fields` as an `option`, and `-output` as
    a `switch` once the leading dash is consumed as `\\W`. Rich splits those
    overlapping spans into separate escape-wrapped runs, so the captured text
    holds `-`, `-output` and `-fields` and never the literal
    `--output-fields`. Turning the forced terminal off keeps the help plain,
    and a test that greps it for an option name means what it says.
    """
    try:
        import typer.rich_utils
    except ImportError:  # pragma: no cover - typer ships with the CLI extra
        return

    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", False)
