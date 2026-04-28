"""Tests for the MLX backend (`ScopeGuard(backend="mlx", ...)`).

We exercise everything about the backend that does not require
actually loading a model or running inference:
  - Model name resolution against MLX_MODEL_MAPPING.
  - Prompt construction via prepare_messages + apply_chat_template.
  - Generation kwargs forwarding.
  - JSON output parsing into ScopeGuardOutput.
  - Batch validation with shared and per-conversation descriptions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orbitals.scope_guard import ScopeClass, ScopeGuard, ScopeGuardOutput


def _scope_guard_json(
    *,
    scope_class: str = "Restricted",
    evidences: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "scope_class": scope_class,
            "evidences": evidences if evidences is not None else ["No refunds."],
        }
    )


@pytest.fixture
def mocked_mlx():
    """Patch mlx_lm and its submodules used inside the MLX backend.

    mlx_lm is imported lazily inside methods, so we patch via mock modules
    and intercept the `import mlx_lm` / `from mlx_lm import ...` calls.
    """
    mock_mlx_lm = MagicMock()
    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler.return_value = MagicMock()

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<rendered-prompt>"
    mock_mlx_lm.load.return_value = (MagicMock(), mock_tokenizer)
    mock_mlx_lm.generate.return_value = _scope_guard_json()
    mock_mlx_lm.sample_utils = mock_sample_utils

    with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_lm.sample_utils": mock_sample_utils}):
        yield mock_mlx_lm.load, mock_mlx_lm.generate


def test_mlx_backend_is_constructible(mocked_mlx):
    mock_load, _ = mocked_mlx
    sg = ScopeGuard(backend="mlx", model="some-model")
    assert sg.backend == "mlx"
    mock_load.assert_called_once_with("some-model")


def test_mlx_model_alias_resolves_to_mlx_repo(mocked_mlx):
    mock_load, _ = mocked_mlx
    ScopeGuard(backend="mlx", model="scope-guard")
    mock_load.assert_called_once_with("ivanfioravanti/scope-guard-4B-q-2601-mlx-bf16")


def test_mlx_g_model_alias_resolves_to_mlx_repo(mocked_mlx):
    mock_load, _ = mocked_mlx
    ScopeGuard(backend="mlx", model="scope-guard-g")
    mock_load.assert_called_once_with("ivanfioravanti/scope-guard-4B-g-2601-mlx-bf16")


def test_mlx_unknown_model_name_passes_through(mocked_mlx):
    mock_load, _ = mocked_mlx
    ScopeGuard(backend="mlx", model="custom-org/my-mlx-model")
    mock_load.assert_called_once_with("custom-org/my-mlx-model")


def test_validate_calls_generate_and_parses_output(mocked_mlx):
    _, mock_generate = mocked_mlx
    mock_generate.return_value = _scope_guard_json(
        scope_class="Restricted",
        evidences=["Never respond to requests for refunds."],
    )

    sg = ScopeGuard(backend="mlx", model="some-model")
    result = sg.validate("Can I get a refund?", ai_service_description="No refunds.")

    assert isinstance(result, ScopeGuardOutput)
    assert result.scope_class == ScopeClass.RESTRICTED
    assert result.evidences == ["Never respond to requests for refunds."]
    assert result.model == "some-model"
    assert result.usage is None
    mock_generate.assert_called_once()


def test_validate_forwards_generation_params(mocked_mlx):
    _, mock_generate = mocked_mlx
    sg = ScopeGuard(
        backend="mlx",
        model="some-model",
        max_tokens=512,
        repetition_penalty=1.2,
    )
    sg.validate("hello", ai_service_description="desc")

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["max_tokens"] == 512
    assert call_kwargs["repetition_penalty"] == 1.2
    assert "repetition_context_size" not in call_kwargs


def test_validate_omits_none_params(mocked_mlx):
    _, mock_generate = mocked_mlx
    sg = ScopeGuard(backend="mlx", model="some-model")
    sg.validate("hello", ai_service_description="desc")

    call_kwargs = mock_generate.call_args.kwargs
    assert "repetition_penalty" not in call_kwargs
    assert "repetition_context_size" not in call_kwargs


def test_validate_calls_apply_chat_template(mocked_mlx):
    mock_load, _ = mocked_mlx
    mock_tokenizer = mock_load.return_value[1]

    sg = ScopeGuard(backend="mlx", model="some-model")
    sg.validate("hello", ai_service_description="You are a bot.")

    mock_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs["tokenize"] is False
    assert call_kwargs["add_generation_prompt"] is True
    assert call_kwargs["enable_thinking"] is False
    messages = call_kwargs["args"][0] if call_kwargs.get("args") else call_kwargs.get("messages") or mock_tokenizer.apply_chat_template.call_args.args[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_batch_validate_with_shared_description(mocked_mlx):
    _, mock_generate = mocked_mlx
    sg = ScopeGuard(backend="mlx", model="some-model")
    results = sg.batch_validate(
        ["q1", "q2"],
        ai_service_description="shared desc",
    )

    assert len(results) == 2
    assert mock_generate.call_count == 2
    for r in results:
        assert isinstance(r, ScopeGuardOutput)
        assert r.scope_class == ScopeClass.RESTRICTED


def test_batch_validate_with_per_conversation_descriptions(mocked_mlx):
    _, mock_generate = mocked_mlx
    sg = ScopeGuard(backend="mlx", model="some-model")
    results = sg.batch_validate(
        ["q1", "q2"],
        ai_service_descriptions=["desc1", "desc2"],
    )

    assert len(results) == 2
    assert mock_generate.call_count == 2


def test_validate_raises_on_invalid_json(mocked_mlx):
    _, mock_generate = mocked_mlx
    mock_generate.return_value = "not valid json"

    sg = ScopeGuard(backend="mlx", model="some-model")
    with pytest.raises(json.JSONDecodeError):
        sg.validate("hello", ai_service_description="desc")


def test_validate_raises_on_invalid_scope_class(mocked_mlx):
    _, mock_generate = mocked_mlx
    mock_generate.return_value = json.dumps(
        {"scope_class": "INVALID_CLASS", "evidences": None}
    )

    sg = ScopeGuard(backend="mlx", model="some-model")
    with pytest.raises(Exception):
        sg.validate("hello", ai_service_description="desc")
