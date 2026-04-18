"""Tests for the AIServiceDescription structured object.

The README documents that `ai_service_description` can be provided either as a
plain string or as a structured `orbitals.types.AIServiceDescription` object.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orbitals.types import AIServiceDescription, Principle


def test_minimal_service_description_requires_identity_role_and_context():
    svc = AIServiceDescription(
        identity_role="Parcel delivery assistant",
        context="Online logistics company.",
    )
    assert svc.identity_role == "Parcel delivery assistant"
    assert svc.context == "Online logistics company."
    assert svc.knowledge_scope is None
    assert svc.functionalities is None
    assert svc.principles is None
    assert svc.website_url is None


def test_identity_role_is_required():
    with pytest.raises(ValidationError):
        AIServiceDescription(context="Some context")  # type: ignore[call-arg]


def test_context_is_required():
    with pytest.raises(ValidationError):
        AIServiceDescription(identity_role="Role")  # type: ignore[call-arg]


def test_functionalities_can_be_string_or_list():
    svc_str = AIServiceDescription(
        identity_role="Role",
        context="Ctx",
        functionalities="Track packages",
    )
    assert svc_str.functionalities == "Track packages"

    svc_list = AIServiceDescription(
        identity_role="Role",
        context="Ctx",
        functionalities=["Track packages", "Estimate delivery"],
    )
    assert svc_list.functionalities == ["Track packages", "Estimate delivery"]


def test_principles_can_be_strings_or_structured_principles():
    svc = AIServiceDescription(
        identity_role="Role",
        context="Ctx",
        principles=[
            "Never give refunds.",
            Principle(
                title="No financial advice",
                description="Do not recommend investments.",
                supporting_materials=None,
            ),
        ],
    )
    assert svc.principles is not None
    assert len(svc.principles) == 2
    assert svc.principles[0] == "Never give refunds."
    assert isinstance(svc.principles[1], Principle)


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        AIServiceDescription(
            identity_role="Role",
            context="Ctx",
            unknown_field="nope",  # type: ignore[call-arg]
        )


def test_model_dump_roundtrip_is_lossless_for_minimal_object():
    svc = AIServiceDescription(identity_role="Role", context="Ctx")
    round_tripped = AIServiceDescription(**svc.model_dump())
    assert round_tripped == svc
