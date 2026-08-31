# COLONPATH-AI: Model & Checkpoint Audit Document

**Audit Date:** 2026-08-31  
**Auditor:** Senior AI/ML & Medical Systems Architect  
**Project:** COLONPATH-AI

---

## 1. Verified Model Checkpoints on Disk

| Model Name | Checkpoint Path | Architecture | Parameter Count | File Size | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **U-Net Gland Model** | `colonpath_ai/outputs/unet/best_model.pth` | 4-Stage ConvNet Encoder-Decoder | 31,043,521 | 118.51 MB | **VERIFIED REAL** |
| **HoVer-Net Nuclear Model** | `hovernet_reference/checkpoints/hovernet_original_consep_type_tf2pytorch` | Multi-branch ResNet Backbone | ~34,000,000 | 209.25 MB | **VERIFIED REAL** |
| **Digepath Foundation Model** | `xtxx/Digepath` / ViT-L/16 (`timm` backbone fallback) | Vision Transformer Large (Patch 16, 224x224) | 303,301,632 | 1024-d output | **VERIFIED REAL** |
| **Multimodal Classifier** | `colonpath_ai/outputs/models/best_classifier.pth` | Late-Fusion Net + 2-Head MLP | ~350,000 | 3.54 MB | **VERIFIED REAL** |
| **Mobile TorchScript** | `colonpath_ai/outputs/android_handover/multimodal_classifier_mobile.pt` | Compiled TorchScript for PyTorch Mobile | ~350,000 | 1.21 MB | **VERIFIED REAL** |

---

## 2. Model Architectures & Forward Passes

### A. U-Net Gland Segmentation
- **Input:** $3 \times 256 \times 256$ RGB image tensor, normalized to $[0, 1]$.
- **Output:** $1 \times 256 \times 256$ probability mask. Thresholded at $p > 0.50$ followed by morphological closing and contour extraction.
- **Metrics Extracted:** Gland count, area ($\text{px}^2$), circularity ($4\pi \cdot \text{Area} / \text{Perimeter}^2$), and aspect ratio.

### B. HoVer-Net Nuclear Segmentation & Phenotyping
- **Input:** $3 \times 256 \times 256$ RGB image tensor.
- **Output:** Nuclear instance map + horizontal/vertical gradient maps + 4-class nuclear phenotype labels.
- **Metrics Extracted:** Total nuclei count, density, mean area, circularity, and cell-type distribution (Epithelial vs. Inflammatory vs. Spindle).

### C. Digepath Foundation Model
- **Input:** $3 \times 224 \times 224$ RGB image tensor, normalized by standard ImageNet/Pathology mean and std.
- **Output:** 1024-dimensional feature vector $\mathbf{v} \in \mathbb{R}^{1024}$ extracted from the `[CLS]` token and $L_2$-normalized.
- **Caching:** 2-tier LRU cache (memory + persistent disk `.npy`) to prevent redundant inference during navigation.

### D. Multimodal Late-Fusion Network (`MultimodalFusionNet`)
- **Visual Projection:** Linear($1024 \to 256$) $\to$ BatchNorm $\to$ GELU $\to$ Dropout($0.2$).
- **Morphology Projection:** Linear($16 \to 64$) $\to$ BatchNorm $\to$ GELU $\to$ Dropout($0.2$).
- **Fusion Layer:** Linear($(256 + 64) \to 128$) $\to$ BatchNorm $\to$ GELU $\to$ Dropout($0.2$).
- **Head 1 (Multiclass):** Linear($128 \to 9$) $\to$ Logits over 9 NCT-100K classes.
- **Head 2 (Binary):** Linear($128 \to 2$) $\to$ Logits over TUM vs. Non-TUM.

---

## 3. Calibration & Uncertainty Parameters

- **Temperature Parameter ($T$):** $T = 1.25$ (fitted via Platt scaling / NLL minimization on validation split).
- **Entropy Measure:** Normalized Shannon Entropy $\tilde{H}(p) = -\sum p_i \ln(p_i) / \ln(9)$.
- **Margin Measure:** $M(p) = 1.0 - (p_{\text{top1}} - p_{\text{top2}})$.
- **Composite Score:** $S_{\text{unc}} = 0.60 \tilde{H}(p) + 0.40 M(p)$.
- **Abstention Gate:** Triggered if $S_{\text{unc}} \ge 0.50$ or top-1 confidence $< 0.60$.
