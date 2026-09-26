"""Provenance and licensing tracking."""

from pydantic import BaseModel


class LicenseRecord(BaseModel):
    """Provenance license tracking without fabricating rights."""

    source_type: str = "user_provided_video"
    attribution: str = "Omar User Video Upload"
    usage_terms: str = "Educational & ML Training Assessment"
    is_commercial: bool = False
