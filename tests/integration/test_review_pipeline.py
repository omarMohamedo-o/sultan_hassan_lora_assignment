"""Integration test for human review persistence."""

from sultan_hassan.config.models    import AppConfig
from sultan_hassan.domain.enums     import CategoryEnum, ReviewDecisionEnum
from sultan_hassan.review.decisions import DecisionStore


def test_review_decision_persistence(test_config: AppConfig) -> None:
    store = DecisionStore(test_config)

    # Save decision
    store.save_decision(
        frame_id="000100_5.00s",
        decision=ReviewDecisionEnum.KEEP,
        category=CategoryEnum.COURTYARD,
        reviewer="omar",
        notes="High clarity central courtyard fountain",
    )

    # Reload fresh store instance from disk to verify resumability
    reloaded_store = DecisionStore(test_config)
    assert reloaded_store.is_reviewed("000100_5.00s")
    dec = reloaded_store.get_decision("000100_5.00s")
    assert dec is not None
    assert dec.decision == ReviewDecisionEnum.KEEP
    assert dec.category == CategoryEnum.COURTYARD
