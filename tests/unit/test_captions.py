"""Unit tests for caption generation and validation."""

from sultan_hassan.dataset.captions import generate_caption_for_category, validate_caption
from sultan_hassan.domain.enums import CategoryEnum


def test_generate_caption_contains_trigger() -> None:
    cap = generate_caption_for_category(CategoryEnum.ENTRANCE, trigger_word="sltnhsn")
    assert "sltnhsn" in cap
    assert "entrance portal" in cap or "muqarnas" in cap


def test_validate_caption_success() -> None:
    text = "sltnhsn, monumental high stone facade with tall vertical window bays"
    is_valid, err, rec = validate_caption(text, trigger_word="sltnhsn")
    assert is_valid
    assert err == ""
    assert rec.trigger_word_present
    assert rec.word_count >= 4


def test_validate_caption_missing_trigger() -> None:
    text = "monumental high stone facade without trigger"
    is_valid, err, _ = validate_caption(text, trigger_word="sltnhsn")
    assert not is_valid
    assert "missing required trigger word" in err
