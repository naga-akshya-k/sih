# RAG_AUDIT.md — Retrieval-Augmented Generation (RAG) Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Multimodal AI & RAG Architecture Team  

---

## 1. RAG Triad Architecture

The RAG implementation in COLONPATH-AI combines two distinct evidence streams into a unified prompt context for Google MedGemma:

```
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│     STREAM A: COMPUTATIONAL EVIDENCE    │   │      STREAM B: RETRIEVED EVIDENCE       │
│  - Digepath 1024-d Prediction & Calib   │   │  - Qdrant Dual-Vector Cosine Matches    │
│  - U-Net Gland Area & Circularity       │   │  - Top-K Reference Case IDs             │
│  - HoVer-Net 117 Nuclei & Phenotypes    │   │  - Reference Histological Characteristics│
│  - Entropy Uncertainty & Energy OOD     │   │  - Multi-Factor Reranked Similarity %   │
└────────────────────┬────────────────────┘   └────────────────────┬────────────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                                ┌──────────────────────┐
                                │ Unified RAG Context  │
                                └──────────┬───────────┘
                                           ▼
                                ┌──────────────────────┐
                                │ Google MedGemma 1.5  │
                                └──────────┬───────────┘
                                           ▼
                                ┌──────────────────────┐
                                │  EvidenceValidator   │
                                └──────────────────────┘
```

---

## 2. Reranking Strategy
1. **Initial Vector Search:** Fetches candidate cohort matches from Qdrant based on visual and morphological similarity.
2. **Clinical Evidence Reranker:** Reranks candidates based on prediction class alignment and nuclear density compatibility.
3. **Context Construction:** Injects the top-1 and top-2 reference cases into the system prompt with exact similarity percentages and provenance.
