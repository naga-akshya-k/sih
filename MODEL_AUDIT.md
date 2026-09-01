# MODEL_AUDIT.md — Deep Learning & Foundation Model Architecture Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Medical AI & Deep Learning Engineering Team  

---

## 1. Deep Learning Model Inventory

| Model Name | Backbone Architecture | Parameters | Input Dimension | Output Dimension | Framework / Checkpoint | Execution Device |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Digepath** | `vit_large_patch16_224` | 303 Million | $224 \times 224 \times 3$ RGB | 1024-d Visual Embedding | PyTorch / `xtxx/Digepath` ($2.04\text{ GB}$) | CUDA (`cuda:0`) |
| **Google MedGemma 1.5 4B IT** | `Gemma3ForConditionalGeneration` | 4.0 Billion | Text + Image Tokens | Autoregressive Clinical Text | Transformers / Safetensors ($8.05\text{ GB}$) | CUDA + CPU Offload |
| **U-Net Gland Segmenter** | 4-Stage Encoder-Decoder | ~31 Million | $256 \times 256 \times 3$ RGB | $256 \times 256 \times 1$ Binary Mask | PyTorch / `best_model.pth` ($118.51\text{ MB}$) | CUDA (`cuda:0`) |
| **HoVer-Net Nuclear Segmenter** | Multi-Branch ResNet-50 | ~42 Million | $256 \times 256 \times 3$ RGB | HoVer Distance Maps + Type Logits | PyTorch / `hovernet_consep` ($209.25\text{ MB}$) | CUDA (`cuda:0`) |
| **MultimodalFusionNet** | 2-Layer MLP Bottleneck | ~140,000 | 1040-d ($1024\text{ visual} + 16\text{ morph}$) | 9-Class NCT Logits | PyTorch / `best_classifier.pth` ($3.54\text{ MB}$) | CUDA (`cuda:0`) |
| **Platt Temperature Scaler** | Single Learnable Parameter | 1 ($T=1.25$) | 9-Class Raw Logits | 9-Class Calibrated Posterior $p_i$ | PyTorch Statistical Calibration | CPU / CUDA |

---

## 2. Quantitative Model Input/Output Specs

### Digepath Foundation Model (`colonpath_ai/models/digepath.py`)
* Preprocessing: Resize to $224 \times 224$, Normalize with ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
* Forward Pass: Passes through 24 transformer blocks with multi-head self-attention.
* Output: 1024-d penultimate feature representation with L2 normalization.

### Multimodal Late-Fusion Network (`colonpath_ai/classifiers/multimodal_fusion.py`)
* Input Concatenation: $\mathbf{x}_{\text{input}} = [\mathbf{v}_{\text{Digepath}}^{1024} \parallel \mathbf{m}_{\text{morphology}}^{16}]$.
* Layer 1: $\text{Linear}(1040, 128) \rightarrow \text{BatchNorm1d}(128) \rightarrow \text{ReLU}() \rightarrow \text{Dropout}(p=0.3)$.
* Layer 2: $\text{Linear}(128, 9) \rightarrow \text{Raw Logits } \mathbf{z} \in \mathbb{R}^9$.

### MedGemma Vision-Language Model (`colonpath_ai/agent/medgemma_vlm.py`)
* Mode A: Neural Causal Generation through `Gemma3Processor` and safetensors weights.
* Mode B: Deterministic Evidence Synthesis Engine generating structured reports in $0.05\text{s}$ with zero VRAM overhead.
* Safety Layer: All output checked by `EvidenceValidator`.
