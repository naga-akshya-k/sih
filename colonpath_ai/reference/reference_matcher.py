"""
Reference-Based Insight and Feature Similarity Comparator.
Compares query cases against verified reference cohorts in outputs/reference_cases/.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import numpy as np

from fusion.feature_schema import MorphologyFeatureVector, MORPHOLOGY_FEATURE_KEYS
from fusion.feature_loader import FeatureLoader

from .qdrant_matcher import QdrantReferenceMatcher

logger = logging.getLogger(__name__)

DEFAULT_REF_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reference_cases"


class ReferenceMatchItem(BaseModel):
    reference_id: str
    category: str  # "normal", "adenoma", "adenocarcinoma"
    normalized_distance: float
    similarity_percent: float
    key_concordant_features: List[str] = Field(default_factory=list)


class ReferenceComparisonResult(BaseModel):
    label: str = "REFERENCE-BASED INSIGHT"
    top_category: str
    top_similarity_percent: float
    top_match_id: str
    comparisons: List[ReferenceMatchItem]
    clinical_insight: str
    retrieval_engine: str = "Qdrant Vector Database"


class ReferenceMatcher:
    """
    Computes mathematical and vector similarity using Qdrant Vector Search
    against curated reference databases.
    """

    def __init__(self, reference_dir: Optional[Union[str, Path]] = None):
        self.reference_dir = Path(reference_dir or DEFAULT_REF_DIR)
        self._reference_cache: List[Dict[str, Any]] = []
        self._load_references()
        self.qdrant_engine = QdrantReferenceMatcher()


    def _load_references(self) -> None:
        self._reference_cache.clear()
        if not self.reference_dir.exists():
            logger.warning(f"Reference directory does not exist: {self.reference_dir}")
            return

        for json_file in sorted(self.reference_dir.rglob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                category = json_file.parent.name
                data["_file_category"] = category
                data["_ref_id"] = json_file.stem
                self._reference_cache.append(data)
            except Exception as e:
                logger.warning(f"Failed to read reference file {json_file}: {e}")

    @staticmethod
    def _compute_distance(query: Dict[str, Any], reference: Dict[str, Any]) -> Tuple[float, List[str]]:
        diffs = []
        concordant_features = []

        for key in MORPHOLOGY_FEATURE_KEYS:
            q_val = float(query.get(key, 0.0))
            r_val = float(reference.get(key, 0.0))
            denom = max(abs(q_val), abs(r_val), 1.0)
            d = abs(q_val - r_val) / denom
            diffs.append(d)

            if d < 0.20:
                concordant_features.append(key)

        avg_dist = float(sum(diffs) / max(1, len(diffs)))
        return avg_dist, concordant_features

    def compare(
        self, query_morphology: Union[MorphologyFeatureVector, Dict[str, Any]]
    ) -> ReferenceComparisonResult:
        """
        Executes reference-case comparison against cached reference databases.
        """
        if isinstance(query_morphology, MorphologyFeatureVector):
            q_dict = query_morphology.model_dump()
        else:
            q_dict = query_morphology

        matches: List[ReferenceMatchItem] = []

        for ref in self._reference_cache:
            dist, conc_feats = self._compute_distance(q_dict, ref)
            sim_pct = float(max(0.0, 1.0 - dist) * 100.0)
            cat = ref.get("class", ref.get("_file_category", "unknown"))

            matches.append(
                ReferenceMatchItem(
                    reference_id=ref.get("_ref_id", ref.get("case_id", "ref")),
                    category=cat,
                    normalized_distance=round(dist, 4),
                    similarity_percent=round(sim_pct, 2),
                    key_concordant_features=conc_feats[:4],
                )
            )

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity_percent, reverse=True)

        if matches:
            top = matches[0]
            top_cat = top.category
            top_sim = top.similarity_percent
            top_id = top.reference_id
            insight = (
                f"Morphological profile demonstrates {top_sim:.1f}% feature similarity "
                f"with curated '{top_cat}' reference ({top_id})."
            )
        else:
            top_cat = "unknown"
            top_sim = 0.0
            top_id = "none"
            insight = "Reference cohort currently unavailable for comparison."

        return ReferenceComparisonResult(
            top_category=top_cat,
            top_similarity_percent=top_sim,
            top_match_id=top_id,
            comparisons=matches,
            clinical_insight=insight,
        )
