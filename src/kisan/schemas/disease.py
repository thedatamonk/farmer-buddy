"""Disease detection related Pydantic schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class Severity(StrEnum):
    """Disease severity levels."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class Treatment(BaseModel):
    """Recommended treatment for a disease."""

    method: str = Field(..., description="Treatment method")
    description: str = Field(..., description="Detailed description of the treatment")
    is_organic: bool = Field(default=False, description="Whether treatment is organic/natural")


class DiseaseDetectionResult(BaseModel):
    """Result of disease detection from an image."""

    crop_identified: str | None = Field(default=None, description="Identified crop type")
    disease_detected: bool = Field(..., description="Whether a disease was detected")
    disease_name: str | None = Field(default=None, description="Name of the detected disease")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of detection")
    severity: Severity = Field(default=Severity.NONE, description="Severity of the disease")
    symptoms: list[str] = Field(default_factory=list, description="Observed symptoms")
    causes: list[str] = Field(default_factory=list, description="Possible causes")
    treatments: list[Treatment] = Field(default_factory=list, description="Recommended treatments")
    prevention_tips: list[str] = Field(default_factory=list, description="Prevention tips")
    additional_notes: str | None = Field(default=None, description="Any additional observations")


class DiseaseDetectionRequest(BaseModel):
    """Request for disease detection."""

    image_base64: str = Field(..., description="Base64 encoded image data")
    crop_hint: str | None = Field(default=None, description="Optional hint about the crop type")
