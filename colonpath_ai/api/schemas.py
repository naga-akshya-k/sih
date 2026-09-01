"""
Pydantic v2 Request and Response Schemas for the FastAPI REST API.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    models_ready: bool


class ImageQualitySchema(BaseModel):
    passed: bool
    resolution: str
    blur_laplacian_variance: float
    blur_status: str
    mean_brightness: float
    brightness_status: str
    contrast_std: float
    contrast_status: str
    mean_saturation: float


class PredictionSchema(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    calibrated_confidence: float
    tumor_probability: float
    binary_class: str
    multiclass_probabilities: Dict[str, float] = Field(default_factory=dict)


class UncertaintySchema(BaseModel):
    score: float
    level: str  # "LOW", "MEDIUM", "HIGH"
    entropy: float
    normalized_entropy: float
    ood_score: float = 0.0
    ood_status: str = "IN_DISTRIBUTION"
    is_ood: bool = False
    review_required: bool
    message: str


class ModelAgreementSchema(BaseModel):
    level: str  # "HIGH", "MEDIUM", "LOW"
    score: float
    concordant_sources: List[str] = Field(default_factory=list)
    discordant_sources: List[str] = Field(default_factory=list)
    summary: str


class RegionDetailSchema(BaseModel):
    region_id: str
    index: int
    x: int
    y: int
    width: int
    height: int
    prediction: str
    confidence: float
    tumor_probability: float
    uncertainty_score: float
    uncertainty_level: str
    priority_score: float
    priority_level: str
    priority_label: str
    nuclei_count: int
    glands_count: int
    agreement_level: str
    rationale: str


class ReferenceComparisonSchema(BaseModel):
    label: str
    top_category: str
    top_similarity_percent: float
    top_reference_id: str
    insight: str
    comparisons: List[Dict[str, Any]] = Field(default_factory=list)


class CaseResultResponse(BaseModel):
    case_id: str
    timestamp: str
    status: str
    image_quality: Dict[str, Any]
    digepath: Dict[str, Any]
    prediction: Dict[str, Any]
    uncertainty: Dict[str, Any]
    model_agreement: Dict[str, Any]
    nuclear_evidence: Dict[str, Any]
    gland_evidence: Dict[str, Any]
    reference_comparison: Dict[str, Any]
    priority_regions: List[RegionDetailSchema]
    visualizations: Dict[str, str] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    explanation: Optional[Dict[str, Any]] = None


class NextRegionResponse(BaseModel):
    case_id: str
    region: RegionDetailSchema
    navigation: Dict[str, Any]


class ReviewRequest(BaseModel):
    action: str = Field(description="MARK_REVIEWED, FLAG_REGION, ADD_NOTE, REQUEST_REANALYSIS")
    notes: Optional[str] = ""
    pathologist_id: Optional[str] = "Dr. Pathologist"


class NoteRequest(BaseModel):
    note_text: str
    author: Optional[str] = "Pathologist"


class CaseSummaryItem(BaseModel):
    case_id: str
    timestamp: str
    prediction_class: Optional[str] = None
    confidence: Optional[float] = None
    uncertainty_level: Optional[str] = None
    review_status: str
