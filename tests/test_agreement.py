"""
Unit tests for model and multi-source evidence agreement engine.
"""

from agreement.agreement_engine import AgreementEngine
from fusion.feature_schema import MorphologyFeatureVector


def test_agreement_high():
    # Concordant tumor morphology
    morph = MorphologyFeatureVector(
        case_id="c1",
        nuclei_total=120,
        nuclei_type_1=80,
        nuclei_mean_area_px2=150.0,
        glands_total=3,
        glands_mean_circularity=0.32,
    )
    res = AgreementEngine.evaluate(
        fusion_prediction="TUM",
        tumor_probability=0.92,
        morphology=morph,
        reference_top_class="adenocarcinoma",
        digepath_prediction="TUM",
    )
    assert res.level == "HIGH"
    assert len(res.discordant_sources) == 0
    assert len(res.concordant_sources) >= 3


def test_agreement_discordant():
    # Conflicting: Visual predicts TUM, but morphology is completely normal
    morph = MorphologyFeatureVector(
        case_id="c2",
        nuclei_total=20,
        nuclei_type_1=5,
        nuclei_mean_area_px2=70.0,
        glands_total=2,
        glands_mean_circularity=0.85,
    )
    res = AgreementEngine.evaluate(
        fusion_prediction="TUM",
        tumor_probability=0.85,
        morphology=morph,
        reference_top_class="normal",
        digepath_prediction="TUM",
    )
    assert res.level in ["MEDIUM", "LOW"]
    assert len(res.discordant_sources) > 0
    assert res.review_recommended is True
