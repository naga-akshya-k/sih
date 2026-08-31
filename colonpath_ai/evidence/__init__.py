"""
Evidence Package for COLONPATH-AI.
Constructs evidence.json and case_result.json strictly from verifiable computational outputs.
"""

from .evidence_builder import EvidenceBuilder
from .explainer import EvidenceGroundedExplainer

__all__ = ["EvidenceBuilder", "EvidenceGroundedExplainer"]
