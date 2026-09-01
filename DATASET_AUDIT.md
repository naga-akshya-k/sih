# DATASET_AUDIT.md — Local Histopathology Datasets Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Medical AI & Digital Pathology Engineering Team  

---

## 1. Verified Local Dataset Inventory

A physical filesystem audit of the configured local dataset paths was conducted using automated Python file iterators and PIL image metadata readers.

### Dataset Summary Table

| Dataset Identifier | Physical Path on Disk | Image Format | Dimensions | Total Images | Primary Role in COLONPATH-AI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NCT-CRC-HE-100K** | `C:\Users\kthir\Downloads\NCT-CRC-HE-100K\NCT-CRC-HE-100K` | TIFF (`.tif`) | $224 \times 224$ px, RGB | **100,000** | Phase A: Model Training & Reference Cohort Indexing |
| **CRC-VAL-HE-7K** | `C:\Users\kthir\Downloads\CRC-VAL-HE-7K\CRC-VAL-HE-7K` | TIFF (`.tif`) | $224 \times 224$ px, RGB | **7,180** | Phase A: Independent Validation & Calibration Optimization |

---

## 2. Class Distribution Breakdown

Both datasets consist of 9 distinct colorectal histological tissue categories:

| Class Code | Histological Tissue Description | NCT-CRC-HE-100K (Train) | CRC-VAL-HE-7K (Val) | Combined Total |
| :--- | :--- | :--- | :--- | :--- |
| **`ADI`** | Adipose Tissue | 10,407 | 1,338 | 11,745 |
| **`BACK`** | Background (glass slide / white space) | 10,566 | 847 | 11,413 |
| **`DEB`** | Debris (necrosis, hemorrhage, mucinous debris)| 11,512 | 339 | 11,851 |
| **`LYM`** | Lymphocytes (immune infiltration) | 11,557 | 634 | 12,191 |
| **`MUC`** | Mucus | 8,896 | 1,035 | 9,931 |
| **`MUS`** | Smooth Muscle | 13,536 | 592 | 14,128 |
| **`NORM`** | Normal Colonic Mucosa | 8,763 | 741 | 9,504 |
| **`STR`** | Cancer-Associated Stroma | 10,446 | 421 | 10,867 |
| **`TUM`** | Colorectal Adenocarcinoma Epithelium | 14,317 | 1,233 | 15,550 |
| **TOTAL** | | **100,000** | **7,180** | **107,180** |

---

## 3. Critical Scientific & Pathological Semantics

1. **Tissue Classification vs. Clinical Diagnosis:**
   * The dataset label `TUM` corresponds to microscopic **colorectal adenocarcinoma epithelium**.
   * It must **NOT** be reported as *"confirmed clinical cancer diagnosis"* by the AI.
   * Proper clinical description: *"AI-predicted tissue class: Colorectal Adenocarcinoma Epithelium (TUM)"*.
2. **Segmentation Ground Truth Separation:**
   * NCT-CRC-HE-100K and CRC-VAL-HE-7K are **tile-level classification datasets**.
   * They do **NOT** contain pixel-level gland or nuclear instance masks.
   * U-Net gland segmentation and HoVer-Net nuclear segmentation utilize their respective pre-trained histopathology weights (e.g. CoNSeP, GlaS) rather than fabricated segmentation masks from classification tiles.
3. **Data Leakage Prevention:**
   * `CRC-VAL-HE-7K` is strictly preserved as an **independent test/validation split**.
   * Normalization parameters (`StandardScaler`), temperature scaling calibration ($T=1.25$), and OOD energy thresholds are fitted solely on training cohorts.
