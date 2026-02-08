"""Phase 1: Disease detection response parsing tests.

Tests DiseaseDetector._parse_response with mock GPT-4o outputs.
Verifies that structured DiseaseDetectionResult is correctly produced
for healthy plants, diseased plants, ambiguous responses, and edge cases.
No LLM calls — pure parsing logic.
"""

import pytest

from kisan.core.config import Settings
from kisan.modules.disease.detector import DiseaseDetector
from kisan.schemas.disease import DiseaseDetectionResult, Severity, Treatment


@pytest.fixture
def detector():
    """Create a DiseaseDetector with a dummy LLM service."""
    from unittest.mock import MagicMock

    settings = Settings(openai_api_key="test-key", qdrant_url="http://localhost:6333")
    return DiseaseDetector(llm_service=MagicMock(), settings=settings)


# --- Mock GPT-4o responses ---

HEALTHY_RESPONSE = """
**Crop Identification**
Wheat (Triticum aestivum), appears to be in the tillering stage.

**Health Assessment**
This plant is healthy and showing no signs of disease or stress. Health score: 9/10

**Disease/Problem Identification**
No disease detected. The plant appears to be in good health with no visible symptoms.

**Symptoms Observed**
No abnormal symptoms observed. Leaves are green and upright.

**Causes**
N/A - Plant is healthy.

**Treatment Recommendations**
No treatment needed. Continue current farming practices.

**Prevention**
- Maintain proper irrigation schedule
- Apply balanced fertilizers as per soil test recommendations

**Additional Notes**
The crop appears well-maintained with good vigor.
"""

DISEASED_RESPONSE = """
**Crop Identification**
Wheat (Triticum aestivum), heading stage

**Health Assessment**
The plant is showing clear signs of disease. Health score: 3/10

**Disease/Problem Identification**
Disease name: Yellow Rust (Puccinia striiformis) / पीला रतुआ
Pathogen type: Fungal
Confidence level: 0.85
Severity: high

**Symptoms Observed**
- Yellow-orange pustules arranged in stripes on leaf surfaces
- Chlorosis (yellowing) between pustule lines
- Premature leaf senescence on lower leaves

**Causes**
- Fungal infection by Puccinia striiformis
- Cool and humid weather conditions favorable for rust development
- Susceptible wheat variety

**Treatment Recommendations**
- Apply Propiconazole 25% EC (Tilt) at 1ml/litre of water as foliar spray
- Alternatively, use Tebuconazole 25.9% EC at 1ml/litre
- Organic alternative: Neem oil (5ml/litre) as preventive measure
- Repeat spray after 15 days if symptoms persist

**Prevention**
- Use rust-resistant varieties like HD-3086, PBW-725
- Timely sowing to avoid peak rust season
- Remove and destroy crop residues after harvest

**Additional Notes**
Consult local KVK (Krishi Vigyan Kendra) for variety-specific recommendations.
Early detection improves treatment effectiveness.
"""

BLURRY_RESPONSE = """
**Crop Identification**
Unable to clearly identify the crop due to image quality.

**Health Assessment**
Cannot assess health status from this image. The image is blurry and out of focus.

**Disease/Problem Identification**
No diagnosis possible. Image quality insufficient.

**Symptoms Observed**
Cannot observe specific symptoms due to poor image quality.

**Causes**
N/A

**Treatment Recommendations**
Please upload a clearer, well-lit photo of the affected plant part for accurate diagnosis.

**Prevention**
N/A

**Additional Notes**
For best results, take a close-up photo of the affected leaves in natural daylight.
"""

MILD_DEFICIENCY_RESPONSE = """
**Crop Identification**
Tomato (Solanum lycopersicum), fruiting stage

**Health Assessment**
The plant shows mild symptoms of nutritional deficiency.

**Disease/Problem Identification**
Identified as: Nitrogen Deficiency
Type: Nutritional deficiency
Confidence: 0.7
Severity: low

**Symptoms Observed**
- Pale green to yellow lower leaves
- Stunted growth compared to healthy plants
- Older leaves affected first (mobile nutrient)

**Causes**
- Insufficient nitrogen in soil
- Poor soil organic matter content
- Leaching due to excessive irrigation

**Treatment Recommendations**
- Apply urea (46-0-0) at 25 kg/acre as side dressing
- Organic: Apply well-decomposed FYM or compost at 5 tonnes/acre
- Foliar spray of 2% urea solution for quick correction

**Prevention**
- Regular soil testing before each crop season
- Balanced NPK application based on soil test values
- Green manuring with dhaincha or moong

**Additional Notes**
The deficiency is mild and easily correctable with timely intervention.
"""

MULTI_ISSUE_RESPONSE = """
**Crop Identification**
Rice (Oryza sativa), vegetative stage

**Health Assessment**
The plant shows signs of both disease and pest damage. Health score: 4/10

**Disease/Problem Identification**
Disease name: Brown Spot (Bipolaris oryzae) with suspected Stem Borer (Scirpophaga incertulas) damage
Confidence: 0.6
Severity: moderate

**Symptoms Observed**
- Oval brown lesions on leaves with gray centers
- Dead hearts visible in some tillers
- Yellowing of affected leaves

**Causes**
- Fungal infection favored by nutrient deficiency
- Stem borer larvae boring into tillers
- High humidity and poor field drainage

**Treatment Recommendations**
- For brown spot: Spray Mancozeb 75% WP at 2g/litre
- For stem borer: Apply Cartap Hydrochloride 4G granules at 25 kg/ha
- Bio-control: Install Trichogramma egg parasitoid cards at 1 lakh eggs/ha

**Prevention**
- Ensure balanced nutrition especially potassium
- Use light traps to monitor stem borer adult moths
- Maintain proper plant spacing for air circulation

**Additional Notes**
Consider an integrated pest management approach. Contact the nearest agricultural officer.
"""


class TestHealthyPlantParsing:
    """Test parsing of responses for healthy plants."""

    def test_disease_not_detected(self, detector):
        result = detector._parse_response(HEALTHY_RESPONSE)
        assert result.disease_detected is False

    def test_crop_identified(self, detector):
        result = detector._parse_response(HEALTHY_RESPONSE)
        assert result.crop_identified is not None
        assert "wheat" in result.crop_identified.lower() or "Triticum" in result.crop_identified

    def test_severity_is_none(self, detector):
        result = detector._parse_response(HEALTHY_RESPONSE)
        assert result.severity == Severity.NONE

    def test_disease_name_is_none(self, detector):
        result = detector._parse_response(HEALTHY_RESPONSE)
        assert result.disease_name is None


class TestDiseasedPlantParsing:
    """Test parsing of responses for diseased plants."""

    def test_disease_detected(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert result.disease_detected is True

    def test_disease_name_extracted(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert result.disease_name is not None
        assert len(result.disease_name) > 0

    def test_confidence_extracted(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert 0.0 <= result.confidence <= 1.0
        # Should be high confidence
        assert result.confidence >= 0.7

    def test_severity_extracted(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert result.severity in (Severity.HIGH, Severity.SEVERE)

    def test_symptoms_parsed(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert len(result.symptoms) >= 1

    def test_causes_parsed(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert len(result.causes) >= 1

    def test_treatments_parsed(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert len(result.treatments) >= 1
        assert all(isinstance(t, Treatment) for t in result.treatments)

    def test_prevention_tips_parsed(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert len(result.prevention_tips) >= 1

    def test_crop_identified(self, detector):
        result = detector._parse_response(DISEASED_RESPONSE)
        assert result.crop_identified is not None


# TODO: We need to handle blurry images in a better way!
class TestBlurryImageParsing:
    """
    Test parsing of responses for unclear images.
    """

    def test_blurry_still_produces_valid_result(self, detector):
        result = detector._parse_response(BLURRY_RESPONSE)
        assert isinstance(result, DiseaseDetectionResult)

    def test_blurry_severity_is_none(self, detector):
        result = detector._parse_response(BLURRY_RESPONSE)
        # Even though disease_detected may be True due to parser limitation,
        # severity should still be NONE since no actual severity is mentioned
        assert result.severity == Severity.NONE

    def test_additional_notes_present(self, detector):
        result = detector._parse_response(BLURRY_RESPONSE)
        assert result.additional_notes is not None


class TestMildDeficiencyParsing:
    """Test parsing of mild/nutritional deficiency responses."""

    def test_deficiency_detected_as_disease(self, detector):
        result = detector._parse_response(MILD_DEFICIENCY_RESPONSE)
        assert result.disease_detected is True

    def test_low_severity(self, detector):
        result = detector._parse_response(MILD_DEFICIENCY_RESPONSE)
        assert result.severity == Severity.LOW

    def test_confidence_moderate(self, detector):
        result = detector._parse_response(MILD_DEFICIENCY_RESPONSE)
        assert 0.5 <= result.confidence <= 0.9


class TestMultiIssueParsing:
    """Test parsing of responses with multiple issues."""

    def test_severity_moderate(self, detector):
        result = detector._parse_response(MULTI_ISSUE_RESPONSE)
        assert result.severity == Severity.MODERATE

    def test_multiple_symptoms(self, detector):
        result = detector._parse_response(MULTI_ISSUE_RESPONSE)
        assert len(result.symptoms) >= 2

    def test_multiple_treatments(self, detector):
        result = detector._parse_response(MULTI_ISSUE_RESPONSE)
        assert len(result.treatments) >= 2


class TestParsingHelpers:
    """Test individual parsing helper methods."""

    def test_extract_confidence_from_number(self, detector):
        assert detector._extract_confidence("Confidence: 0.85") == 0.85

    def test_extract_confidence_high(self, detector):
        assert detector._extract_confidence("High confidence in this diagnosis") == 0.85

    def test_extract_confidence_medium(self, detector):
        assert detector._extract_confidence("Medium confidence") == 0.65

    def test_extract_confidence_low(self, detector):
        assert detector._extract_confidence("Low confidence") == 0.45

    def test_extract_confidence_default(self, detector):
        assert detector._extract_confidence("") == 0.5

    def test_extract_severity_severe(self, detector):
        assert detector._extract_severity("Severe damage observed") == Severity.SEVERE

    def test_extract_severity_moderate(self, detector):
        assert detector._extract_severity("Moderate infection level") == Severity.MODERATE

    def test_extract_severity_low(self, detector):
        assert detector._extract_severity("Low/mild symptoms") == Severity.LOW

    def test_extract_severity_none(self, detector):
        assert detector._extract_severity("Plant is healthy") == Severity.NONE

    def test_check_disease_detected_healthy(self, detector):
        assert detector._check_disease_detected("healthy", "no disease") is False

    def test_check_disease_detected_diseased(self, detector):
        assert detector._check_disease_detected("", "fungal infection detected") is True

    def test_check_disease_detected_deficiency(self, detector):
        assert detector._check_disease_detected("showing deficiency symptoms", "") is True

    def test_parse_list_empty(self, detector):
        assert detector._parse_list("") == []

    def test_parse_list_bullet_points(self, detector):
        text = "- Item one\n- Item two\n- Item three"
        result = detector._parse_list(text)
        assert len(result) >= 2

    def test_parse_treatments_organic_detection(self, detector):
        text = "- Apply neem oil spray at 5ml/litre\n- Use chemical fungicide Mancozeb 2g/l"
        treatments = detector._parse_treatments(text)
        organic_count = sum(1 for t in treatments if t.is_organic)
        assert organic_count >= 1
