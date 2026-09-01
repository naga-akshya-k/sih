"""
Unit tests for Qdrant Dual-Vector Reference Retrieval.
"""

import numpy as np
import pytest
from reference.qdrant_matcher import QdrantReferenceMatcher, QDRANT_AVAILABLE


def test_qdrant_initialization_and_search():
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client not installed")

    matcher = QdrantReferenceMatcher()
    
    # Test morphological query
    dummy_morph = np.array([0.4, 0.5, 0.7, 0.2, 0.35, 0.4] + [0.0] * 10, dtype=np.float32)
    results = matcher.search_similar_cases(query_morph_vec=dummy_morph, top_k=3)
    
    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert "reference_id" in top
    assert "similarity_percent" in top
    assert top["similarity_percent"] >= 0.0
