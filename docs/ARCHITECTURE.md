# COLONPATH-AI: System Architecture Specification

**Document Date:** 2026-08-31  
**Auditor:** Senior AI/ML & Medical Systems Architect  
**Project:** COLONPATH-AI (Multimodal Colorectal Decision-Support Platform)

---

## 1. Architectural Paradigm

COLONPATH-AI implements an **Agentic Multimodal Decision-Support Architecture** designed for clinical digital pathology workflows. The architecture bridges the gap between deep vision foundation models and fine-grained cellular morphology while enforcing deterministic mathematical safety guardrails.

```
                           H&E SLIDE IMAGE (256x256 / WSI Tile)
                                         │
                                         ▼
                                IMAGE QUALITY GATE
                          (Pass / Warn / Fail Protocol)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 DIGEPATH FOUNDATION             PATHOLOGY PIPELINE
                (ViT-L/16 GI Backbone)           (U-Net + HoVer-Net)
                         │                               │
                         ▼                               ▼
                 1024-d Visual Vector            16-d Morphology Vector
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                              MULTIMODAL FUSION NET
                         (Batch Normalization & Dropout)
                                         │
                                         ▼
                            TISSUE CLASSIFICATION HEAD
                           (9 NCT Classes + Binary Tumor)
                                         │
                         ┌───────────────┼───────────────┐
                         ▼               ▼               ▼
                    Prediction     Calibration     Uncertainty
                     (Softmax)       (T=1.25)     (Entropy/Margin)
                         │               │               │
                         └───────────────┼───────────────┘
                                         ▼
                              MODEL AGREEMENT ENGINE
                          (Multi-Source Consensus Matrix)
                                         │
                                         ▼
                             AI-PRIORITIZED REGIONS
                          (Spatial Patch Decomposition)
                                         │
                                         ▼
                              REFERENCE COMPARISON
                           (Cosine Similarity Matcher)
                                         │
                                         ▼
                                DETERMINISTIC EVIDENCE
                                    (evidence.json)
                                         │
                                         ▼
                             ANTI-HALLUCINATION CRITIC
                           (EvidenceValidator Guardrail)
                                         │
                                         ▼
                                PRODUCTION REST API
                            (FastAPI + SQLite Persistence)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                WEB DECISION DASHBOARD           ANDROID MOBILE APP
                 (Interactive Viewer)           (Pathologist-in-Loop)
```

---

## 2. Module Specifications

### Module 1: Image Quality Gate (`preprocessing/quality_check.py`)
- Evaluates mean brightness ($[30, 230]$), contrast ($> 15.0$), and tissue area ($> 10\%$).
- Blocks uninterpretable or severely corrupted images before neural network execution.

### Module 2: Vision Foundation Embedding (`foundation/digepath/`)
- Extracts 1024-d high-level visual representations using ViT-L/16 patch-based attention.
- Managed by a two-tier LRU memory and disk cache (`foundation/digepath/embedding_cache.py`).

### Module 3: Quantitative Morphology Adapter (`morphology/`)
- Aggregates nuclear instance phenotyping (HoVer-Net) and glandular boundary segmentation (U-Net).
- Produces a standardized 16-dimensional feature vector $\mathbf{m} \in \mathbb{R}^{16}$.

### Module 4: Multimodal Late-Fusion Network (`fusion/fusion_model.py`)
- Projects 1024-d visual vectors to 256-d and 16-d morphology vectors to 64-d.
- Fuses projected representations into a 128-d bottleneck layer.

### Module 5: Decision Calibration & Uncertainty (`uncertainty/`)
- Scales logits with temperature parameter $T = 1.25$.
- Calculates normalized Shannon entropy $\tilde{H}(p)$ and margin $M(p)$. Automatically triggers `review_required = true` on high uncertainty.

### Module 6: Consensus & Agreement Engine (`agreement/agreement_engine.py`)
- Cross-validates visual class predictions against nuclear pleomorphism, glandular loss-of-circularity, and reference cohort matches.
- Returns explicit consensus tiers: **`HIGH`**, **`MEDIUM`**, or **`LOW`**.

### Module 7: Spatial Region Prioritization (`regions/priority_ranking.py`)
- Divides images into coordinate-indexed spatial patches ($R_{01}, R_{02}, \dots$).
- Computes priority score: $0.45 P(\text{Tumor}) + 0.30 S(\text{Nuclear Atypia}) + 0.15 S(\text{Uncertainty}) + 0.10 S(\text{Gland Distortion})$.

### Module 8: Evidence Gatekeeper & Anti-Hallucination (`agent/evidence_validator.py`)
- Intercepts all natural language summaries.
- Validates that every numerical claim in the explanation matches ground-truth values in `evidence.json`.

### Module 9: Storage & REST API Backend (`storage/`, `api/`)
- Persists all case analyses, review actions (`MARK REVIEWED`, `FLAG REGION`), and clinical notes in SQLite (`storage/case_repository.py`).
- Serves endpoints on port 8080: `/health`, `/analyze`, `/cases/{id}/result`, `/cases/{id}/regions`, `/cases/{id}/visualization/{type}`, `/cases/{id}/review`, `/cases/{id}/notes`.
