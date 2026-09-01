# Multimodal RAG & Qdrant Vector Retrieval Architecture

This document describes the Retrieval-Augmented Generation (RAG) and Qdrant Vector Database architecture implemented in the COLONPATH-AI intelligence platform.

---

## 1. High-Level RAG Architecture

```
Current Biopsy Patch (H&E)
          │
    ┌─────┴────────────────┐
    ▼                      ▼
Digepath ViT-L/16      HoVer-Net & U-Net
Visual Feature         Quantitative Morphology
  (1024-d)               (16-d)
    │                      │
    └──────────┬───────────┘
               ▼
   Qdrant Vector Database
   (Dual Space: visual + morphology)
               │
               ▼
     Metadata Filtering & Top-K Cosine Search
               │
               ▼
     Multi-Factor Reranking Layer
               │
               ▼
   Retrieved Reference Evidence
   + Verified Case Metrics (evidence.json)
               │
               ▼
   Google MedGemma 1.5 4B IT
   (Evidence-Grounded Explainer)
               │
               ▼
   EvidenceValidator Gatekeeper (Anti-Hallucination Critic)
               │
               ▼
   Approved Clinical Report & Copilot Answers
```

---

## 2. Qdrant Dual-Vector Schema

- **Collection Name:** `colonpath_reference_cohorts`
- **Vectors Configuration:**
  - `visual`: 1024-dimensional vector with `Distance.COSINE`
  - `morphology`: 16-dimensional vector with `Distance.COSINE`
- **Metadata Payload:**
  - `case_id`: Unique reference case identifier
  - `category`: Reference diagnostic group (`normal`, `adenoma`, `adenocarcinoma`)
  - `tissue_class`: Standardized tissue label
  - `nuclear_summary`: Mean area, count, circularity
  - `gland_summary`: Gland count, circularity, aspect ratio
  - `source_dataset`: Curated provenance

---

## 3. Multi-Factor Reranking & Retrieval Weighting

Reranking combines visual and morphological similarities:

$$\text{Rerank Score} = w_{\text{visual}} \cdot S_{\text{visual}} + w_{\text{morph}} \cdot S_{\text{morph}} - \text{Penalty}_{\text{discordance}}$$

Where $w_{\text{visual}} = 0.5$ and $w_{\text{morph}} = 0.5$, ensuring retrieval considers both visual architectural patterns and cytological measurements.
