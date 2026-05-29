from pydantic import ValidationError

from ..types import AIServiceDescriptionV2
from ..scope_guard.safety_principles import ADDITIONAL_SAFETY_RULES


def _add_safety_constraints(desc: AIServiceDescriptionV2) -> AIServiceDescriptionV2:
    if desc.constraints is None:
        merged: list[str] = [ADDITIONAL_SAFETY_RULES]
    elif isinstance(desc.constraints, str):
        merged = [desc.constraints, ADDITIONAL_SAFETY_RULES]
    else:
        merged = [*desc.constraints, ADDITIONAL_SAFETY_RULES]
    return desc.model_copy(update={"constraints": merged})


def augment_with_default_safety_principles_v2(
    ai_service_description: str | AIServiceDescriptionV2,
) -> str | AIServiceDescriptionV2:
    if isinstance(ai_service_description, AIServiceDescriptionV2):
        return _add_safety_constraints(ai_service_description)

    try:
        parsed = AIServiceDescriptionV2.model_validate_json(ai_service_description)
    except (ValueError, ValidationError):
        return f"{ai_service_description}\n\n{ADDITIONAL_SAFETY_RULES}"

    return _add_safety_constraints(parsed).model_dump_json()
