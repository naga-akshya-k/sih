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


def run_pipeline(image_path: str, case_id: str = "CASE_DEMO_00000"):
    try:
        from colonpath_ai.orchestrator.pipeline import CaseOrchestrator
    except ImportError:
        from orchestrator.pipeline import CaseOrchestrator

    orchestrator = CaseOrchestrator()
    img_file = Path(image_path)
    if not img_file.exists():
        # Check inside colonpath_ai
        alt_path = COLONPATH_AI_DIR / image_path
        if alt_path.exists():
            img_file = alt_path

    print(f"Running COLONPATH-AI Analysis Pipeline on {img_file}...")
    result = orchestrator.run(image_path=img_file, case_id=case_id)
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Case ID:        {result.get('case_id')}")
    print(f"Prediction:     {result.get('prediction', {}).get('class')} (Confidence: {result.get('prediction', {}).get('confidence')})")
    print(f"Uncertainty:    {result.get('uncertainty', {}).get('level')} (Score: {result.get('uncertainty', {}).get('score')})")
    print(f"Model Agreement: {result.get('model_agreement', {}).get('level')}")
    print(f"Top Reference:  {result.get('reference_comparison', {}).get('top_category')} ({result.get('reference_comparison', {}).get('top_similarity_percent')}%)")
    print("\nClinical Explanation:")
    print(result.get("explanation", {}).get("text"))


def main():
    parser = argparse.ArgumentParser(description="COLONPATH-AI Master CLI")
    parser.add_argument("--server", action="store_true", help="Launch the FastAPI REST API and Web Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--image", type=str, default="colonpath_ai/outputs/hovernet_test/input/00000.png", help="Path to H&E patch image for analysis")
    parser.add_argument("--case_id", type=str, default="CASE_DEMO_00000", help="Case identifier")
    args = parser.parse_args()

    if args.server:
        start_server(host=args.host, port=args.port)
    else:
        # If no arguments provided, or image provided, run pipeline
        run_pipeline(image_path=args.image, case_id=args.case_id)


if __name__ == "__main__":
    main()
