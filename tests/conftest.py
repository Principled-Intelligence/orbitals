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
