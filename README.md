# COLONPATH-AI: Multimodal Decision-Support System for Colorectal Histopathology

COLONPATH-AI is a comprehensive multimodal AI decision-support system for colorectal cancer (CRC) histopathology. It integrates:
- **Vision Foundation Model:** Digepath (ViT-L/16 DINO-v2) 1024-dimensional feature representations.
- **Morphological Phenotyping:** U-Net gland segmentation & HoVer-Net nuclear instance segmentation and classification.
- **Multimodal Late-Fusion Network:** Fuses visual embeddings and 16-d structured morphology.
- **Uncertainty & Abstention Engine:** Post-hoc Temperature Scaling and Shannon entropy confidence scoring.
- **Model Agreement Consensus:** Multi-source cross-checking across vision, morphology, and reference cohorts.
- **AI-Prioritized Region Ranking & Navigation:** Spatial patch prioritization with "Next Region" viewport panning.
- **Anti-Hallucination Gatekeeper:** Factually validated decision support explanations.
- **FastAPI REST API:** Full backend server powering the Android mobile application.

---

## Repository Structure

```
colonPath/
├── api/                   # FastAPI REST API endpoints & routes
├── foundation/            # Digepath ViT-L/16 feature extraction & embedding cache
├── fusion/                # Multimodal Late-Fusion network & feature loader
├── classifiers/           # 9-class tissue & binary tumor classifiers
├── uncertainty/           # Temperature scaling & uncertainty estimation
├── agreement/             # Multi-source model agreement engine
├── regions/               # AI-prioritized region analysis & navigation
├── reference/             # Reference case similarity matcher
├── visualization/         # Genuine overlays & pseudo-3D topography
├── evidence/              # Deterministic evidence.json & case_result.json builder
├── agent/                 # Anti-hallucination evidence validator
├── storage/               # SQLite database & case repository
├── orchestrator/          # End-to-end master pipeline orchestrator
├── web/                   # Interactive decision-support web dashboard
├── tests/                 # 18 automated unit and integration tests
├── outputs/               # Models, benchmarks, results, and visualizations
│   └── android_handover/  # Handover package for Android app developer
└── docs/                  # System audit, Android API specs, and reproducibility
```

---

## Quickstart

### 1. Start the FastAPI Backend Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```
- **Web Dashboard:** `http://127.0.0.1:8080/`
- **Interactive API Docs:** `http://127.0.0.1:8080/docs`

### 2. Run the Full Test Suite
```powershell
python -m pytest tests/ -v
```

### 3. Execute End-to-End Pipeline on a Slide
```powershell
python -m orchestrator.pipeline --image "outputs/hovernet_test/input/00000.png" --case_id "CASE_DEMO_00000"
```
