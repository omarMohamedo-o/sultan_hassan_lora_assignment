"""Standard test prompt definitions for FLUX LoRA evaluation."""

from pydantic import BaseModel


class PromptDefinition(BaseModel):
    """Specification of a test prompt."""

    group_id: int
    name: str
    prompt: str
    has_trigger_word: bool
    description: str


REQUIRED_TEST_PROMPTS: list[PromptDefinition] = [
    PromptDefinition(
        group_id=1,
        name="elevation",
        prompt="sltnhsn, elevation",
        has_trigger_word=True,
        description="Assesses full monumental exterior vertical facade, stone masonry, and window bays.",
    ),
    PromptDefinition(
        group_id=2,
        name="entrance_gates",
        prompt="sltnhsn, entrance gates",
        has_trigger_word=True,
        description="Assesses the monumental entrance portal, muqarnas stalactite corbelling, and carved lintels.",
    ),
    PromptDefinition(
        group_id=3,
        name="courtyard_sunset",
        prompt="sltnhsn, a courtyard with a central fountain at sunset",
        has_trigger_word=True,
        description="Assesses 4-iwan open courtyard, central fountain ablution pavilion, and sunset lighting fidelity.",
    ),
    PromptDefinition(
        group_id=4,
        name="mosque_night",
        prompt="sltnhsn, a mosque at night",
        has_trigger_word=True,
        description="Assesses nighttime atmosphere, lighting, minaret silhouettes, and architectural preservation.",
    ),
    PromptDefinition(
        group_id=5,
        name="control_office_tower",
        prompt="a modern glass office tower",
        has_trigger_word=False,
        description="Negative control to verify zero style-bleeding; must NOT produce Mamluk stones, arches, or minarets.",
    ),
]
