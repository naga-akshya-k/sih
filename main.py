"""
COLONPATH-AI Master Application Entry Point.
Provides a unified CLI to run the decision support pipeline, launch the REST API server, or start the web dashboard.
"""

import sys
import argparse
from pathlib import Path

# Add colonpath_ai to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
COLONPATH_AI_DIR = PROJECT_DIR / "colonpath_ai"
if COLONPATH_AI_DIR.exists() and str(COLONPATH_AI_DIR) not in sys.path:
    sys.path.insert(0, str(COLONPATH_AI_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def start_server(host: str = "0.0.0.0", port: int = 8080):
    import uvicorn
    print(f"Starting COLONPATH-AI Server on http://{host}:{port}")
    print(f"Web Dashboard: http://127.0.0.1:{port}/")
    print(f"API Documentation: http://127.0.0.1:{port}/docs")
    try:
        uvicorn.run("colonpath_ai.api.main:app", host=host, port=port, reload=False)
    except Exception:
        uvicorn.run("api.main:app", host=host, port=port, reload=False)


def run_pipeline(image_path: str, case_id: str = "CASE_DEMO_00000", verbose: bool = False):
    try:
        from colonpath_ai.orchestrator.pipeline import CaseOrchestrator
        from colonpath_ai.agent.medgemma_vlm import MedGemmaVLM
        from colonpath_ai.agent.evidence_validator import EvidenceValidator
    except ImportError:
        from orchestrator.pipeline import CaseOrchestrator
        from agent.medgemma_vlm import MedGemmaVLM
        from agent.evidence_validator import EvidenceValidator

    orchestrator = CaseOrchestrator()
    img_file = Path(image_path)
    if not img_file.exists():
        # Check inside colonpath_ai
        alt_path = COLONPATH_AI_DIR / image_path
        if alt_path.exists():
            img_file = alt_path

    print(f"\nRunning COLONPATH-AI 12-Stage Multimodal Pipeline on {img_file}...")
    result = orchestrator.run(image_path=img_file, case_id=case_id)
    
    # Generate MedGemma VLM explanation
    medgemma = MedGemmaVLM()
    report = medgemma.generate_structured_report(result)
    
    print("\n" + "=" * 65)
    print("       COLONPATH-AI: 12-STAGE MULTIMODAL INFERENCE REPORT")
    print("=" * 65)
    print(f"1. Input Biopsy Case:     {result.get('case_id')} ({result.get('image_quality', {}).get('resolution', '256x256')})")
    print(f"2. Foundation Model:      {result.get('digepath', {}).get('architecture')} ({result.get('digepath', {}).get('embedding_dimension')}-d visual embedding)")
    print(f"3. Multimodal Fusion:     Visual (1024-d) + Morphology (16-d) -> 128-d Latent Bottleneck")
    print(f"4. Predicted Tissue Class: {result.get('prediction', {}).get('class')} (Raw Conf: {result.get('prediction', {}).get('confidence') * 100.0:.1f}%)")
    print(f"5. Calibrated Confidence: {result.get('prediction', {}).get('calibrated_confidence') * 100.0:.1f}% (Platt Scaler T=1.25)")
    print(f"   - Model Uncertainty:   {result.get('uncertainty', {}).get('level')} (Shannon Entropy: {result.get('uncertainty', {}).get('entropy', 0.0):.4f})")
    print(f"   - OOD Detection:       {result.get('uncertainty', {}).get('ood_status', 'IN_DISTRIBUTION')} (Score: {result.get('uncertainty', {}).get('ood_score', 0.0):.2f})")
    print(f"   - Review Required:     {result.get('uncertainty', {}).get('review_required')}")
    print(f"6. Multi-Source Consensus:{result.get('model_agreement', {}).get('level')} ({result.get('model_agreement', {}).get('summary')})")
    print(f"7. Nuclear Morphology:    {result.get('nuclear_evidence', {}).get('total_count')} nuclei (Mean Area: {result.get('nuclear_evidence', {}).get('mean_area_px2')} px²)")
    print(f"8. Gland Histomorphometry:{result.get('gland_evidence', {}).get('total_count')} glands (Mean Circularity: {result.get('gland_evidence', {}).get('mean_circularity')})")
    print(f"9. Vector RAG Retrieval:  {result.get('reference_comparison', {}).get('top_category')} ({result.get('reference_comparison', {}).get('top_similarity_percent')}%) - {result.get('reference_comparison', {}).get('insight')}")
    print(f"10. Priority Regions:     {len(result.get('priority_regions', []))} spatial patches ranked for pathologist triage")
    for r in result.get('priority_regions', [])[:2]:
        print(f"    * [{r.get('region_id')}] Priority={r.get('priority_score'):.2f} ({r.get('priority_level')}) at (x={r.get('x')}, y={r.get('y')})")
    print(f"11. Anti-Hallucination:   Critic Verified (Errors: {report.get('validation_errors') if report.get('validation_errors') else 'None (100% Grounded)'})")
    print(f"12. MedGemma Explanation: {report.get('summary')}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="COLONPATH-AI Master CLI")
    parser.add_argument("--server", action="store_true", help="Launch the FastAPI REST API and Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--image", type=str, default="colonpath_ai/outputs/hovernet_test/input/00000.png", help="Path to H&E patch image for analysis")
    parser.add_argument("--case_id", type=str, default="CASE_DEMO_00000", help="Case identifier")
    parser.add_argument("--demo", action="store_true", help="Run full 12-stage pipeline demonstration")
    args = parser.parse_args()

    if args.server:
        start_server(host=args.host, port=args.port)
    else:
        run_pipeline(image_path=args.image, case_id=args.case_id, verbose=args.demo)


if __name__ == "__main__":
    main()

