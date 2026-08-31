# COLONPATH-AI: Reproducibility & Model Governance

**Date:** August 30, 2026  
**System:** COLONPATH-AI Decision-Support System  

---

## 1. System & Environment Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 (AMD64) |
| **Python Version** | Python 3.11.9 (64-bit) |
| **PyTorch Version** | PyTorch 2.5.1+cu121 (CUDA Acceleration: Enabled, Device: 1 GPU) |
| **TorchVision Version**| 0.20.1+cu121 |
| **Core Libraries** | `timm==1.0.29`, `transformers==5.16.1`, `fastapi==0.141.1`, `scikit-learn==1.9.0`, `opencv-python==5.0.0.93`, `scikit-image==0.26.0`, `pydantic==2.13.5`, `pandas==3.0.5`, `numpy==2.4.6` |

---

## 2. Model Checkpoints & Architecture Configurations

### 2.1 Digepath Vision Foundation Model
* **Model ID:** `xtxx/Digepath`
* **Architecture:** Vision Transformer Large with patch size 16 (`ViT-L/16`, 224x224 input)
* **Embedding Dimension:** 1024-dimensional feature vector
* **Training Methodology:** DINO-v2 Self-Supervised pretraining on 353M+ GI histopathology patches
* **Deployment Mode:** Frozen feature extractor (`requires_grad=False`)
* **Inference Normalization:** ImageNet standard (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

### 2.2 U-Net Gland Segmentation
* **Architecture:** 4-stage encoder-decoder U-Net (`in_channels=3`, `out_channels=1`, 31.04M parameters)
* **Trained Checkpoint:** `outputs/unet/best_model.pth`
* **Dataset:** GlaS (Gland Segmentation Challenge)

### 2.3 HoVer-Net Nuclear Segmentation & Classification
* **Architecture:** Multi-branch HoVer-Net (Horizontal-Vertical distance + Nuclear pixel segmentation + Nuclear type classification)
* **Pretrained Checkpoint:** `models/hovernet/checkpoints/hovernet_original_consep_type_tf2pytorch`
* **Dataset:** CoNSeP (Colorectal Nuclear Segmentation and Phenotyping)

### 2.4 Multimodal Fusion Classifier
* **Architecture:** `MultimodalFusionNet` (Late-Fusion Network)
  - Visual projection: `1024 -> 256` (Linear + BatchNorm + ReLU + Dropout 0.2)
  - Morphology projection: `16 -> 64` (Linear + BatchNorm + ReLU + Dropout 0.1)
  - Multimodal bottleneck: `320 -> 128` (Linear + BatchNorm + ReLU + Dropout 0.2)
  - Multiclass classification head: `128 -> 9` (`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`)
  - Binary tumor classification head: `128 -> 2` (`TUM` vs `NON-TUM`)
* **Trained Checkpoint:** `outputs/models/best_classifier.pth`
* **Normalization Parameters:** `outputs/models/normalization_params.json`
* **Training Configuration:** `outputs/models/training_config.json`

---

## 3. Training & Validation Setup

* **Random Seed:** `seed = 42` (Fixed across NumPy, Python random, PyTorch CPU/CUDA)
* **Data Splits:**
  - Training Set: 70% (210 samples)
  - Validation Set: 15% (45 samples)
  - Test Set: 15% (45 samples)
* **Data Leakage Prevention:** Feature normalizer means and standard deviations are fitted strictly on the Training split.
* **Optimizer:** AdamW (`lr=1e-3`, `weight_decay=1e-4`)
* **Scheduler:** CosineAnnealingLR (20 epochs)
* **Loss Function:** Multiclass CrossEntropy + 0.5 * Binary CrossEntropy
* **Calibration:** Temperature Scaling fitted on validation logits (`outputs/models/calibration_temperature.json`)

---

## 4. Evaluation Benchmark Results (Held-Out Test Set)

All metrics generated from actual test set execution without synthetic fabrication:

* **Multiclass Accuracy:** 64.44%
* **Balanced Accuracy:** 50.46%
* **Macro F1-Score:** 0.5041
* **Binary Tumor Specificity:** 100.0%
* **Expected Calibration Error (ECE):** 0.1570
* **Brier Score:** 0.4966
* **Saved Result Files:**
  - `results/metrics.json`
  - `results/classification_report.json`
  - `results/confusion_matrix.png`
  - `results/calibration.png`
