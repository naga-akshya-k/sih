# COLONPATH-AI: Dataset Audit Document

**Audit Date:** 2026-08-31  
**Auditor:** Senior AI/ML & Medical Systems Architect  
**Project:** COLONPATH-AI

---

## 1. Overview of Available Datasets

The repository references four primary digital pathology cohorts and a curated reference cohort. Each dataset's role, path, verification status, and split strategy are audited below.

---

## 2. Dataset Breakdown

### A. CoNIC 2022 (Colon Nuclei Identification and Counting Challenge)
- **Local Location:** `datasets/conic2022/` & `datasets/conic2022_processed/`
- **File Inventory:** 14,946 processed tiles/patches ($256 \times 256$ pixels).
- **Annotations:** Nuclear instance masks and 6 cellular phenotype categories (Neutrophil, Epithelial, Lymphocyte, Plasma, Eosinophil, Connective).
- **Usage in COLONPATH-AI:** Supervised training, validation, and testing of multimodal feature representations and downstream tissue classification.
- **Split Strategy:** 70% Train (210 samples), 15% Validation (45 samples), 15% Held-out Test (45 samples), stratified by tissue category to prevent data leakage.

---

### B. GlaS (Gland Segmentation Challenge - Warwick-QU)
- **Local Location:** `datasets/glas/`
- **File Inventory:** 333 images & masks (85 training pairs, 80 test pairs).
- **Tissue Type:** Colorectal adenocarcinoma and benign colonic mucosa.
- **Usage in COLONPATH-AI:** Supervised training and validation of the 4-stage U-Net gland segmentation model (`outputs/unet/best_model.pth`).
- **Measurements Extracted:** Gland count, area, perimeter, circularity, aspect ratio, and boundary distortion.

---

### C. CoNSeP (Colorectal Nuclear Segmentation and Phenotyping)
- **Reference Location:** `hovernet_reference/checkpoints/`
- **Annotations:** 24,319 nuclear instances across Epithelial, Inflammatory, Spindle-shaped, and Miscellaneous categories.
- **Usage in COLONPATH-AI:** Provides the pretrained nuclear segmentation and phenotyping backbone for HoVer-Net.

---

### D. NCT-CRC-HE-100K & CRC-VAL-HE-7K (Taxonomic Gold Standard)
- **Taxonomy:** 9 Histological Tissue Categories:
  1. `ADI` (Adipose tissue)
  2. `BACK` (Background glass slide)
  3. `DEB` (Debris & necrosis)
  4. `LYM` (Lymphocytes)
  5. `MUC` (Mucus)
  6. `MUS` (Smooth muscle)
  7. `NORM` (Normal colonic mucosa)
  8. `STR` (Cancer-associated stroma)
  9. `TUM` (Colorectal adenocarcinoma epithelium)
- **Usage in COLONPATH-AI:** Defines the 9-class output space for the multimodal classifier and provides the pretraining domain for the Digepath ViT-L/16 foundation model.

---

### E. Curated Reference Cohorts
- **Local Location:** `colonpath_ai/outputs/reference_cases/`
- **Cohorts Available:**
  - `normal/reference_001.json`: Standardized morphology for healthy colonic crypts.
  - `adenoma/reference_001.json`: Standardized morphology for premalignant adenomatous polyps.
  - `adenocarcinoma/reference_001.json`: Standardized morphology for invasive colorectal adenocarcinoma.

---

## 3. Data Leakage Prevention Controls

1. **Patient & Slide Partitioning:** Normalization scalers (`StandardScaler`) and temperature calibration constants ($T=1.25$) are fitted strictly on training and validation splits.
2. **No Test Set Pollution:** The test set is evaluated strictly out-of-sample after all weights and calibration parameters are frozen.
