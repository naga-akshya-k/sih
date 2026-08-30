"""
End-to-End Multimodal Analysis Orchestrator for COLONPATH-AI.
Executes the unified AI-assisted decision-support pipeline sequentially.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image

from foundation.digepath.inference import DigepathFeatureExtractor
from fusion.feature_loader import FeatureLoader
from fusion.feature_schema import MorphologyFeatureVector
from classifiers.tissue_classifier import TissueClassifier
from uncertainty.uncertainty_estimator import UncertaintyEstimator
from agreement.agreement_engine import AgreementEngine
from regions.region_analyzer import RegionAnalyzer
from reference.reference_matcher import ReferenceMatcher
from visualization.visualizer import CaseVisualizer
from evidence.evidence_builder import EvidenceBuilder
from evidence.explainer import EvidenceGroundedExplainer
from agent.evidence_validator import EvidenceValidator
from storage.case_repository import CaseRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def evaluate_image_quality(image_path: Path) -> Dict[str, Any]:
    """
    Evaluates image quality metrics (blur, brightness, contrast, saturation).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"passed": False, "error": "Could not read image file"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[:, :, 1]))

    blur_ok = lap_var >= 40.0
    bright_ok = 40.0 <= brightness <= 230.0
    contrast_ok = contrast >= 15.0

    passed = blur_ok and bright_ok and contrast_ok

    return {
        "passed": passed,
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "blur_laplacian_variance": round(lap_var, 2),
        "blur_status": "ACCEPTABLE" if blur_ok else "HIGH_BLUR",
        "mean_brightness": round(brightness, 2),
        "brightness_status": "ACCEPTABLE" if bright_ok else ("TOO_DARK" if brightness < 40 else "VERY_BRIGHT"),
        "contrast_std": round(contrast, 2),
        "contrast_status": "ACCEPTABLE" if contrast_ok else "LOW_CONTRAST",
        "mean_saturation": round(saturation, 2),
    }


class CaseOrchestrator:
    """
    Master pipeline orchestrator running the entire decision-support sequence.
    """

    def __init__(
        self,
        extractor: Optional[DigepathFeatureExtractor] = None,
        classifier: Optional[TissueClassifier] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        region_analyzer: Optional[RegionAnalyzer] = None,
        reference_matcher: Optional[ReferenceMatcher] = None,
        visualizer: Optional[CaseVisualizer] = None,
        repository: Optional[CaseRepository] = None,
    ):
        self.extractor = extractor or DigepathFeatureExtractor()
        self.classifier = classifier or TissueClassifier()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.region_analyzer = region_analyzer or RegionAnalyzer(
            extractor=self.extractor,
            classifier=self.classifier,
            uncertainty_estimator=self.uncertainty_estimator,
        )
        self.reference_matcher = reference_matcher or ReferenceMatcher()
        self.visualizer = visualizer or CaseVisualizer()
        self.repository = repository or CaseRepository()

    def run(
        self,
        image_path: Union[str, Path],
        case_id: Optional[str] = None,
        nuclei_csv: Optional[Union[str, Path]] = None,
        glands_csv: Optional[Union[str, Path]] = None,
        gland_mask_path: Optional[Union[str, Path]] = None,
        nuclei_overlay_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at {img_path}")

        cid = case_id or img_path.stem
        logger.info(f"--- Running COLONPATH-AI Pipeline on Case: {cid} ---")

        # 1. Quality Check
        logger.info("Step 1/9: Image Quality Check")
        img_quality = evaluate_image_quality(img_path)

        # 2. Digepath Visual Embedding
        logger.info("Step 2/9: Digepath GI Visual Embedding Extraction")
        v_emb = self.extractor.extract(img_path, cache_key=f"digepath_{cid}")

        # 3. Load / Derive Morphological Features
        logger.info("Step 3/9: Morphological Integration")
        if nuclei_csv or glands_csv:
            morphology = FeatureLoader.from_measurements(cid, nuclei_csv=nuclei_csv, glands_csv=glands_csv)
        else:
            # Check default output paths or load standard sample
            default_fv = OUTPUT_DIR / "morphology" / "feature_vector.json"
            if default_fv.exists():
                morphology = FeatureLoader.load_feature_vector(default_fv)
            else:
                morphology = MorphologyFeatureVector(case_id=cid)

        # 4. Multimodal Fusion & Classification
        logger.info("Step 4/9: Multimodal Feature Fusion & Classification")
        pred_res = self.classifier.predict(v_emb, morphology)
        logits = pred_res["logits"]

        # 5. Uncertainty Estimation & Calibration
        logger.info("Step 5/9: Uncertainty Estimation & Calibration")
        unc_res = self.uncertainty_estimator.estimate(
            logits=logits,
            probabilities=np.array(list(pred_res["multiclass_probabilities"].values())),
            image_quality_passed=img_quality["passed"],
        )

        # 6. Reference Comparison
        logger.info("Step 6/9: Reference Case Similarity Matching")
        ref_res = self.reference_matcher.compare(morphology)

        # 7. Model Agreement Analysis
        logger.info("Step 7/9: Model & Multi-Source Agreement Analysis")
        agr_res = AgreementEngine.evaluate(
            fusion_prediction=pred_res["prediction"],
            tumor_probability=pred_res["tumor_probability"],
            morphology=morphology,
            reference_top_class=ref_res.top_category,
            digepath_prediction=pred_res["prediction"],
        )

        # 8. Region-Level Analysis & Ranking
        logger.info("Step 8/9: AI-Prioritized Region Analysis")
        regions = self.region_analyzer.analyze_image(
            img_path,
            nuclei_csv=nuclei_csv,
            glands_csv=glands_csv,
        )

        # 9. Visualizations, Evidence & Validation
        logger.info("Step 9/9: Generating Visualizations & Synthesizing Evidence")
        vis_paths = self.visualizer.render_all(
            case_id=cid,
            image_path=img_path,
            regions=regions,
            gland_mask_path=gland_mask_path,
            nuclei_overlay_path=nuclei_overlay_path,
            nuclei_csv=nuclei_csv,
        )

        # Build full case_result
        case_result = EvidenceBuilder.build_case_result(
            case_id=cid,
            image_quality=img_quality,
            digepath_meta=self.extractor.metadata,
            prediction_result=pred_res,
            uncertainty=unc_res,
            model_agreement=agr_res,
            morphology=morphology,
            reference_result=ref_res,
            priority_regions=regions,
            visualizations=vis_paths,
        )

        # Build & validate explanation
        explanation = EvidenceGroundedExplainer.generate_explanation(case_result)
        val_res = EvidenceValidator.validate(explanation, case_result)
        case_result["explanation"] = {
            "text": explanation,
            "validated": val_res.is_valid,
            "validation_errors": val_res.errors,
        }

        # Save to disk: case_result.json and evidence.json
        case_out_dir = OUTPUT_DIR / "cases" / cid
        case_out_dir.mkdir(parents=True, exist_ok=True)

        res_path = case_out_dir / "case_result.json"
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(case_result, f, indent=2)

        ev_path = case_out_dir / "evidence.json"
        evidence_json = EvidenceBuilder.build_evidence_json(case_result)
        with open(ev_path, "w", encoding="utf-8") as f:
            json.dump(evidence_json, f, indent=2)

        # Persist to SQLite
        self.repository.save_case(
            case_id=cid,
            case_result=case_result,
            result_json_path=res_path,
            evidence_json_path=ev_path,
            image_path=img_path,
        )

        logger.info(f"✓ Pipeline execution completed for case {cid}.")
        return case_result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="COLONPATH-AI Master Case Pipeline")
    parser.add_argument("--image", default="outputs/hovernet_test/input/00000.png", help="H&E Image path")
    parser.add_argument("--case_id", default="CASE_DEMO_001", help="Case ID")
    parser.add_argument("--nuclei_csv", default="outputs/morphology/nuclei_measurements.csv")
    parser.add_argument("--glands_csv", default="outputs/morphology/gland_measurements.csv")
    parser.add_argument("--gland_mask", default="outputs/unet/testA_1_prediction.png")
    parser.add_argument("--nuclei_overlay", default="outputs/hovernet_test/result/overlay/00000.png")
    args = parser.parse_args()

    orchestrator = CaseOrchestrator()
    result = orchestrator.run(
        image_path=args.image,
        case_id=args.case_id,
        nuclei_csv=args.nuclei_csv,
        glands_csv=args.glands_csv,
        gland_mask_path=args.gland_mask,
        nuclei_overlay_path=args.nuclei_overlay,
    )

    print("\n" + "=" * 60)
    print("COLONPATH-AI ANALYSIS RESULT")
    print("=" * 60)
    print(f"Case ID:        {result['case_id']}")
    print(f"Prediction:     {result['prediction']['class']} (Confidence: {result['prediction']['confidence']:.2f})")
    print(f"Uncertainty:    {result['uncertainty']['level']} (Score: {result['uncertainty']['score']:.2f})")
    print(f"Agreement:      {result['model_agreement']['level']}")
    print(f"Top Reference:  {result['reference_comparison']['top_category']} ({result['reference_comparison']['top_similarity_percent']:.1f}%)")
    print(f"Prioritized Regions: {len(result['priority_regions'])}")
    print(f"\nExplanation:\n{result['explanation']['text']}")


if __name__ == "__main__":
    main()
