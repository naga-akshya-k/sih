# Model Sources, Checkpoints, and Verification Status

This document catalogs every neural network model, foundation model, vision-language model, and checkpoint used within the COLONPATH-AI intelligence platform.

---

## 1. Inventory Table

| Model Name | Backbone / Arch | Source / Checkpoint Path | Purpose | Dimension | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Digepath** | ViT-L/16 (`vit_large_patch16_224`) | `xtxx/Digepath` on HuggingFace Hub | Pathology visual foundation feature extraction | 1024-d visual embedding | **VERIFIED & LOADED (CUDA)** |
| **HoVer-Net** | HoVer-Net (ResNet-50 / Dense) | `hovernet_reference/checkpoints/` (209.25 MB) | Nuclear instance segmentation & 4-type classification | Pixel-level nuclear maps | **VERIFIED & LOADED** |
| **U-Net** | 2D U-Net (Encoder-Decoder) | `colonpath_ai/outputs/unet/best_model.pth` (118.51 MB) | Glandular boundary segmentation | Binary gland mask | **VERIFIED & LOADED** |
| **MultimodalFusionNet** | Multimodal Late-Fusion Bottleneck | `colonpath_ai/outputs/models/best_classifier.pth` (3.54 MB) | Multimodal fusion & 9-class tissue prediction | 128-d latent -> 9 logits | **VERIFIED & LOADED** |
| **Platt Calibrator** | Temperature Scaler | `colonpath_ai/uncertainty/calibration.py` ($T=1.25$) | Probability calibration (ECE = 0.1570) | Scalar temperature | **VERIFIED & CALIBRATED** |
| **Qdrant Vector Engine** | Qdrant Dual-Vector Space | `colonpath_ai/reference/qdrant_matcher.py` | Reference cohort vector similarity retrieval | Dual: 1024-d & 16-d | **VERIFIED & INITIALIZED** |
| **Google MedGemma** | MedGemma 1.5 4B IT | `google/medgemma-1.5-4b-it` (HuggingFace) | Evidence-grounded multimodal clinical explainer & Copilot | 4 Billion Parameters | **VERIFIED & CONNECTED** |
| **EvidenceValidator** | Rule-based critic agent | `colonpath_ai/agent/evidence_validator.py` | Anti-hallucination verification gatekeeper | Regex & metric matching | **VERIFIED (100% Pass)** |
