"""Metadata heuristic filter analyzing filenames, tags, and origin context."""

import re

from sultan_hassan.config.models import AppConfig


class MetadataFilter:
    """Analyzes text in filenames, tags, and video metadata for architectural cues."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sultan_keywords = [k.lower() for k in config.semantic.sultan_hassan_keywords]
        self.rifai_keywords = [k.lower() for k in config.semantic.rifai_keywords]

    def _normalize(self, text: str) -> str:
        """Normalize spaces, underscores, and hashtags."""
        lower = text.lower()
        # Replace underscores, hyphens, hashtags with space
        spaced = re.sub(r"[_\-#]+", " ", lower)
        return f"{lower} {spaced}"

    def score_text(self, text: str) -> tuple[float, float, str]:
        """Score text string for Sultan Hassan vs. Al-Rifa'i indications.

        Returns:
            Tuple of (sultan_hassan_score, al_rifai_score, rationale).
        """
        clean_text = self._normalize(text)
        has_sultan = any(
            re.search(re.escape(k), clean_text)
            or re.search(re.escape(k.replace(" ", "")), clean_text)
            for k in self.sultan_keywords
        )
        has_rifai = any(
            re.search(re.escape(k), clean_text)
            or re.search(re.escape(k.replace(" ", "")), clean_text)
            for k in self.rifai_keywords
        )

        if has_sultan and not has_rifai:
            return 0.90, 0.10, "Metadata explicitly references Sultan Hassan Mosque"
        if has_rifai and not has_sultan:
            return 0.10, 0.90, "Metadata explicitly references Al-Rifa'i Mosque"
        if has_sultan and has_rifai:
            return 0.50, 0.50, "Metadata references both Sultan Hassan and Al-Rifa'i"
        return 0.50, 0.50, "Neutral metadata with no explicit mosque keywords"
