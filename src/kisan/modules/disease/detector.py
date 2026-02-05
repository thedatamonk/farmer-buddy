"""Disease detection using GPT-4o vision."""

import re

from kisan.core.config import Settings
from kisan.core.exceptions import VisionError
from kisan.core.logging import logger
from kisan.modules.disease.prompts import (
    DISEASE_ANALYSIS_PROMPT,
    DISEASE_ANALYSIS_SYSTEM_PROMPT,
)
from kisan.schemas.disease import DiseaseDetectionResult, Severity, Treatment
from kisan.services.llm import LLMService


class DiseaseDetector:
    """Detects crop diseases from images using GPT-4o vision."""

    def __init__(self, llm_service: LLMService, settings: Settings):
        self.llm = llm_service
        self.settings = settings

    async def detect(
        self,
        image_base64: str,
        crop_hint: str | None = None,
    ) -> DiseaseDetectionResult:
        """Detect diseases in a crop image."""
        try:
            prompt = DISEASE_ANALYSIS_PROMPT
            if crop_hint:
                prompt = f"Hint: This may be a {crop_hint} plant.\n\n" + prompt

            response = await self.llm.chat_with_vision(
                text_prompt=prompt,
                image_base64=image_base64,
                system_prompt=DISEASE_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.3,
            )

            return self._parse_response(response)

        except VisionError:
            raise
        except Exception as e:
            logger.error(f"Disease detection failed: {e}")
            raise VisionError(f"Failed to analyze crop image: {e}") from e

    def _parse_response(self, response: str) -> DiseaseDetectionResult:
        """Parse LLM response into structured result."""
        # Extract sections from the response
        crop = self._extract_section(response, "Crop Identification", "Crop Identified")
        health_assessment = self._extract_section(response, "Health Assessment")
        disease_section = self._extract_section(
            response, "Disease/Problem Identification", "Disease Identification"
        )
        symptoms_text = self._extract_section(response, "Symptoms Observed", "Symptoms")
        causes_text = self._extract_section(response, "Causes", "Possible Causes")
        treatment_text = self._extract_section(
            response, "Treatment Recommendations", "Treatment"
        )
        prevention_text = self._extract_section(response, "Prevention", "Prevention Tips")
        notes = self._extract_section(response, "Additional Notes", "Notes")

        # Determine if disease was detected
        disease_detected = self._check_disease_detected(health_assessment, disease_section)
        disease_name = self._extract_disease_name(disease_section) if disease_detected else None
        confidence = self._extract_confidence(disease_section)
        severity = self._extract_severity(disease_section)

        # Parse lists
        symptoms = self._parse_list(symptoms_text)
        causes = self._parse_list(causes_text)
        treatments = self._parse_treatments(treatment_text)
        prevention_tips = self._parse_list(prevention_text)

        return DiseaseDetectionResult(
            crop_identified=crop.strip() if crop else None,
            disease_detected=disease_detected,
            disease_name=disease_name,
            confidence=confidence,
            severity=severity,
            symptoms=symptoms,
            causes=causes,
            treatments=treatments,
            prevention_tips=prevention_tips,
            additional_notes=notes.strip() if notes else None,
        )

    def _extract_section(self, text: str, *section_names: str) -> str:
        """Extract content of a section from the response."""
        for name in section_names:
            # Try to find section with various markdown formats
            patterns = [
                rf"\*\*{name}\*\*[:\s]*(.+?)(?=\n\*\*|\n\d+\.\s*\*\*|$)",
                rf"#{1,3}\s*{name}[:\s]*(.+?)(?=\n#|$)",
                rf"{name}[:\s]*(.+?)(?=\n\n|\n\d+\.|$)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return ""

    def _check_disease_detected(self, health: str, disease: str) -> bool:
        """Check if a disease was detected."""
        combined = (health + " " + disease).lower()
        healthy_indicators = ["healthy", "no disease", "no issues", "good health"]
        disease_indicators = [
            "disease",
            "infection",
            "affected",
            "symptoms",
            "deficiency",
            "pest",
        ]

        has_healthy = any(ind in combined for ind in healthy_indicators)
        has_disease = any(ind in combined for ind in disease_indicators)

        if has_disease and not ("no disease" in combined or "healthy" in combined):
            return True
        return not has_healthy

    def _extract_disease_name(self, text: str) -> str | None:
        """Extract disease name from text."""
        # Look for patterns like "Disease name: X" or just the first significant line
        patterns = [
            r"disease\s*name[:\s]+([^\n,]+)",
            r"identified\s*as[:\s]+([^\n,]+)",
            r"diagnosis[:\s]+([^\n,]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Clean up markdown
                name = re.sub(r"\*+", "", name)
                return name

        # Return first non-empty line if no pattern matched
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            return re.sub(r"^[-*•]\s*", "", lines[0])[:100]
        return None

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level from text."""
        text_lower = text.lower()

        # Look for explicit confidence values
        match = re.search(r"confidence[:\s]*([\d.]+)", text_lower)
        if match:
            try:
                return min(float(match.group(1)), 1.0)
            except ValueError:
                pass

        # Map text-based confidence to numbers
        if "high" in text_lower:
            return 0.85
        elif "medium" in text_lower or "moderate" in text_lower:
            return 0.65
        elif "low" in text_lower:
            return 0.45
        return 0.5

    def _extract_severity(self, text: str) -> Severity:
        """Extract severity level from text."""
        text_lower = text.lower()

        if "severe" in text_lower or "critical" in text_lower:
            return Severity.SEVERE
        elif "high" in text_lower:
            return Severity.HIGH
        elif "moderate" in text_lower or "medium" in text_lower:
            return Severity.MODERATE
        elif "low" in text_lower or "mild" in text_lower:
            return Severity.LOW
        return Severity.NONE

    def _parse_list(self, text: str) -> list[str]:
        """Parse a text section into a list of items."""
        if not text:
            return []

        items = []
        # Split by common list markers
        lines = re.split(r"\n[-*•]\s*|\n\d+\.\s*", text)
        for line in lines:
            line = line.strip()
            line = re.sub(r"^[-*•]\s*", "", line)
            if line and len(line) > 3:
                items.append(line)
        return items[:10]  # Limit to 10 items

    def _parse_treatments(self, text: str) -> list[Treatment]:
        """Parse treatment text into structured treatments."""
        if not text:
            return []

        treatments = []
        items = self._parse_list(text)

        for item in items[:5]:  # Limit to 5 treatments
            is_organic = any(
                word in item.lower()
                for word in ["organic", "natural", "neem", "bio", "compost"]
            )
            treatments.append(
                Treatment(
                    method=item[:100],
                    description=item,
                    is_organic=is_organic,
                )
            )

        return treatments
