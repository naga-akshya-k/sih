# COLONPATH-AI: Remaining Work Plan & Execution Roadmap

**Date:** August 30, 2026  
**System:** COLONPATH-AI Research Decision Support System  

---

## 1. Project Component Breakdown

### Completed Components (Preserved & Reused)
1. **H&E Preprocessing & Quality Check**:
   - `preprocessing/quality_check.py`, `preprocessing/load_image.py`, `preprocessing/image_info.py`
   - Laplacian blur variance, mean brightness, contrast (std), HSV saturation.
2. **Gland Segmentation (U-Net)**:
   - `models/unet/unet_model.py`, `models/unet/predict_unet.py`
   - Trained checkpoint: `outputs/unet/best_model.pth`
3. **Nuclear Segmentation & Classification (HoVer-Net)**:
   - `hovernet_reference/`, pretrained checkpoint `hovernet_original_consep_type_tf2pytorch`
   - Predicts instances, contours, centroids, and nuclear types (1: Epithelial, 2: Inflammatory, 3: Spindle/Stromal, 4: Miscellaneous).
4. **Morphology Pipelines**:
   - `morphology/analyze_hovernet.py`, `morphology/morphology_analysis.py`
   - Nuclear & gland parameter extraction (`nuclei_measurements.csv`, `gland_measurements.csv`).
5. **Morphological Case Summary & Feature Vector**:
   - `morphology/case_summary.py` -> `case_summary.json`
   - `morphology/feature_vector.py` -> `feature_vector.json` (16-d structured features).
6. **Reference Cases & Base Comparison**:
   - `morphology/compare_case.py`
   - Reference cases database: `outputs/reference_cases/` (`normal`, `adenoma`, `adenocarcinoma`).
7. **Datasets**:
   - 4,981 CoNIC processed samples + maps, GlaS cohort, reference test image `00000.png`.

---

## 2. Remaining Components to Build

```
                           H&E IMAGE (256x256 / WSI Tile)
                                         │
                                         ▼
                                IMAGE QUALITY CHECK
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                         ▼                               ▼
                 EXISTING PIPELINE                    DIGEPATH
                         │                    (ViT-L/16 GI Foundation)
                   ┌─────┴─────┐                         │
                   ▼           ▼                         ▼
                 U-Net      HoVer-Net           GI VISUAL EMBEDDING
                   │           │                     (1024-d)
                   ▼           ▼                         │
                Glands       Nuclei                      │
                   │           │                         │
                   ▼           ▼                         │
                Gland       Nuclear                      │
              Morphology   Morphology                    │
                   │           │                         │
                   └─────┬─────┴─────────────────────────┘
                         ▼
                  FEATURE FUSION
            (Late Fusion: Visual + Morphology)
                         │
                         ▼
                  TISSUE CLASSIFIER
           (9-Class Tissue + Binary TUM/Non-TUM)
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Prediction  Confidence  Uncertainty
      (Softmax/Cal) (Platt/Temp)  (Entropy/ECE)
                         │
                         ▼
                  MODEL AGREEMENT
            (Cross-Check Visual vs Morph vs Ref)
                         │
                         ▼
                  REGION ANALYSIS
          (Patch Extraction & Prioritization)
                         │
                         ▼
               AI-PRIORITIZED REGIONS
          (Ranked Scores & Next-Region Engine)
                         │
                         ▼
                REFERENCE COMPARISON
             (Cosine / Metric Similarity)
                         │
                         ▼
                    EVIDENCE JSON
           (Deterministic Structured Facts)
                         │
                         ▼
                OPTIONAL VLM EXPLAINER
          (Evidence-Grounded Prompting Only)
                         │
                         ▼
                 EVIDENCE VALIDATOR
           (Anti-Hallucination Gatekeeper)
                         │
                         ▼
                  FASTAPI BACKEND
            (REST API + SQLite Store + Pydantic)
                         │
                         ▼
                    ANDROID APP
         (Pathologist-in-the-Loop Decision Support)
```

---

## 3. Module Dependencies & Architecture

```
colon_model/
├── foundation/
│   └── digepath/
│       ├── __init__.py
│       ├── model_loader.py       # ViT-L/16 DINO-v2 GI foundation loader (CUDA/CPU)
│       ├── preprocess.py         # 224x224 / ImageNet-norm preprocessing
│       ├── inference.py          # Embedding extraction
│       └── embedding_cache.py    # Local disk/memory caching for embeddings
├── fusion/
│   ├── __init__.py
│   ├── feature_schema.py         # Pydantic & dataclass schemas for features
│   ├── feature_loader.py         # Robust loader for case_summary.json / feature_vector.json
│   ├── normalization.py          # Fitted StandardScaler / MinMax parameter preservation
│   └── fusion_model.py           # Late-fusion neural network (Visual 1024d + Morph 16d)
├── classifiers/
│   ├── __init__.py
│   ├── tissue_classifier.py      # 9-class tissue & binary classifier heads
│   ├── train_classifier.py       # Training pipeline with stratified train/val/test splits
│   └── evaluate_classifier.py    # Generates classification report, confusion matrix, ROC/PR curves
├── uncertainty/
│   ├── __init__.py
│   ├── calibration.py            # Temperature scaling / Platt scaling & ECE computation
│   └── uncertainty_estimator.py  # Entropy, confidence score, abstention flags (review_required)
├── agreement/
│   ├── __init__.py
│   └── agreement_engine.py       # Multi-source cross-checking (Digepath vs Morph vs Fusion vs Ref)
├── regions/
│   ├── __init__.py
│   ├── region_analyzer.py        # Grid/tiled patch decomposition & region feature extraction
│   ├── priority_ranking.py       # Configurable scoring formula (TUM prob, abnormal morph, uncertainty)
│   └── region_navigator.py       # Next-Region stateful/indexed navigation helper
├── reference/
│   ├── __init__.py
│   └── reference_matcher.py      # Standardized metric & cosine similarity against reference database
├── visualization/
│   ├── __init__.py
│   └── visualizer.py             # Overlays (Glands, Nuclei, Prioritized regions, Uncertainty, Pseudo-3D)
├── evidence/
│   ├── __init__.py
│   ├── evidence_builder.py       # Builds evidence.json and case_result.json from computational facts
│   └── explainer.py              # Evidence-grounded explanation builder
├── agent/
│   ├── __init__.py
│   └── evidence_validator.py     # Strict anti-hallucination verification against evidence.json
├── storage/
│   ├── __init__.py
│   ├── database.py               # SQLite engine & table definitions
│   └── case_repository.py        # Case CRUD, review status, pathologist notes persistence
├── orchestrator/
│   ├── __init__.py
│   └── pipeline.py               # Master end-to-end case analysis pipeline
├── api/
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── schemas.py                # Pydantic request/response schemas
│   ├── routes/
│   │   ├── health.py
│   │   ├── analysis.py
│   │   ├── cases.py
│   │   ├── regions.py
│   │   └── review.py
│   └── services/
│       ├── analysis_service.py
│       ├── case_service.py
│       └── region_service.py
├── results/                      # Evaluation metrics, reports, and calibration plots
├── docs/
│   ├── existing_system_audit.md
│   ├── remaining_work_plan.md
│   ├── android_api.md
│   └── reproducibility.md
└── tests/
    ├── test_digepath.py
    ├── test_fusion.py
    ├── test_classifier.py
    ├── test_uncertainty.py
    ├── test_agreement.py
    ├── test_regions.py
    ├── test_evidence_validator.py
    ├── test_api.py
    └── test_end_to_end.py
```

---

## 4. Implementation Order (Phase by Phase)

* **Phase 1**: Audit Existing Project (`docs/existing_system_audit.md`, `docs/remaining_work_plan.md`) ✅
* **Phase 2 & 3**: Verify existing pipeline outputs & feature schemas (`00000.png` sample)
* **Phase 4, 5, 6**: Integrate Digepath foundation model, test single-image inference, implement embedding cache
* **Phase 7, 8**: Integrate morphology features & construct Late-Fusion model
* **Phase 9, 10**: Train downstream classifier on stratified dataset splits; evaluate (ECE, Brier, F1, AUC, Confusion Matrix)
* **Phase 11, 12**: Implement calibration, entropy-based uncertainty estimation & abstention logic
* **Phase 13**: Build multi-source model agreement engine (`HIGH`, `MEDIUM`, `LOW`)
* **Phase 14, 15, 16**: Implement region-level inference, AI-prioritized ranking formula, Next-Region navigation, and reference comparison
* **Phase 17, 18, 19**: Build `evidence.json`, grounded explanation generator, and strict `agent/evidence_validator.py`
* **Phase 20, 21, 22**: Build pipeline orchestrator, SQLite case storage, and full FastAPI REST backend
* **Phase 23**: Produce Android API documentation (`docs/android_api.md`)
* **Phase 24, 25, 26**: Comprehensive automated testing, demo execution on `00000.png`, and reproducibility documentation (`docs/reproducibility.md`)

---

## 5. Testing & Verification Plan

1. **Unit Tests (`tests/`)**:
   - `test_digepath.py`: Verifies model loading, tensor transformations, embedding dimensionality (1024-d).
   - `test_fusion.py`: Validates missing feature handling, NaN/Inf rejection, late fusion concatenation.
   - `test_classifier.py`: Validates 9-class and binary probability distributions sum to 1.0.
   - `test_uncertainty.py`: Validates entropy scaling, temperature calibration, and review required flag.
   - `test_agreement.py`: Tests concordant vs discordant evidence scenarios.
   - `test_regions.py`: Tests region tile splitting, coordinate indexing, and priority score sort order.
   - `test_evidence_validator.py`: Tests rejection of hallucinated cell counts, incorrect classes, or fake coordinates.
   - `test_api.py`: Validates all FastAPI endpoints (`/health`, `/analyze`, `/cases/{id}`, `/regions`, `/review`, `/notes`).
2. **End-to-End Test (`test_end_to_end.py`)**:
   - Executes full pipeline on `00000.png`, verifying complete `case_result.json`, `evidence.json`, visualizations, and database storage.

---

## 6. Final Deliverables

1. Full modular Python package for COLONPATH-AI.
2. Trained and calibrated multimodal classifier with verified evaluation reports (`results/metrics.json`, `results/classification_report.json`, `results/confusion_matrix.png`, `results/calibration.png`).
3. Anti-hallucination evidence validation system.
4. FastAPI backend serving analysis, region navigation, visualizations, and pathologist review.
5. Android API contract documentation (`docs/android_api.md`).
6. Reproducibility documentation (`docs/reproducibility.md`).
7. Complete end-to-end verified demo on test case `00000.png`.
