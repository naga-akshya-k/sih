"""
Multimodal Late-Fusion Architecture for Colorectal Histopathology.
Fuses Digepath visual foundation representations (1024-d) with structured morphology (16-d).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TISSUE_CLASSES = [
    "ADI",   # Adipose tissue
    "BACK",  # Background
    "DEB",   # Debris
    "LYM",   # Lymphocytes
    "MUC",   # Mucus
    "MUS",   # Smooth muscle
    "NORM",  # Normal mucosa
    "STR",   # Stroma
    "TUM",   # Colorectal adenocarcinoma / Tumor
]

NUM_CLASSES = len(TISSUE_CLASSES)  # 9
VISUAL_DIM = 1024
MORPH_DIM = 16


class MultimodalFusionNet(nn.Module):
    """
    Modular Late-Fusion Neural Network combining visual and morphological representations.
    """

    def __init__(
        self,
        visual_dim: int = VISUAL_DIM,
        morph_dim: int = MORPH_DIM,
        visual_hidden: int = 256,
        morph_hidden: int = 64,
        fused_hidden: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.morph_dim = morph_dim
        self.num_classes = num_classes

        # Visual branch projection
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_hidden),
            nn.BatchNorm1d(visual_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Morphology branch projection
        self.morph_proj = nn.Sequential(
            nn.Linear(morph_dim, morph_hidden),
            nn.BatchNorm1d(morph_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )

        # Multimodal fusion bottleneck
        combined_dim = visual_hidden + morph_hidden
        self.fusion_block = nn.Sequential(
            nn.Linear(combined_dim, fused_hidden),
            nn.BatchNorm1d(fused_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Multi-class tissue classification head (9 classes)
        self.multiclass_head = nn.Linear(fused_hidden, num_classes)

        # Binary tumor classification head (0: Non-Tumor, 1: Tumor)
        self.binary_head = nn.Linear(fused_hidden, 2)

    def forward_features(self, visual_emb: torch.Tensor, morph_vec: torch.Tensor) -> torch.Tensor:
        """
        Extracts the 128-dimensional fused latent representation.
        """
        # Ensure 2D shape [B, D]
        if visual_emb.ndim == 1:
            visual_emb = visual_emb.unsqueeze(0)
        if morph_vec.ndim == 1:
            morph_vec = morph_vec.unsqueeze(0)

        # Handle batch size 1 for BatchNorm in eval mode
        v_feat = self.visual_proj(visual_emb)
        m_feat = self.morph_proj(morph_vec)

        # Concatenate visual and morphology representations
        fused = torch.cat([v_feat, m_feat], dim=-1)
        latent = self.fusion_block(fused)
        return latent

    def forward(
        self, visual_emb: torch.Tensor, morph_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            multiclass_logits: [B, 9]
            binary_logits: [B, 2]
            latent_features: [B, 128]
        """
        latent = self.forward_features(visual_emb, morph_vec)
        multiclass_logits = self.multiclass_head(latent)
        binary_logits = self.binary_head(latent)
        return multiclass_logits, binary_logits, latent

    def predict_probabilities(
        self, visual_emb: torch.Tensor, morph_vec: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Inference helper returning softmax probabilities and class predictions.
        """
        self.eval()
        with torch.no_grad():
            mc_logits, bin_logits, latent = self.forward(visual_emb, morph_vec)
            mc_probs = F.softmax(mc_logits, dim=-1).squeeze(0).cpu().numpy()
            bin_probs = F.softmax(bin_logits, dim=-1).squeeze(0).cpu().numpy()
            latent_vec = latent.squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(mc_probs))
        pred_class = TISSUE_CLASSES[pred_idx]
        confidence = float(mc_probs[pred_idx])

        return {
            "prediction": pred_class,
            "prediction_index": pred_idx,
            "confidence": confidence,
            "multiclass_probabilities": {
                cls_name: float(p) for cls_name, p in zip(TISSUE_CLASSES, mc_probs)
            },
            "tumor_probability": float(bin_probs[1]),
            "non_tumor_probability": float(bin_probs[0]),
            "binary_prediction": "TUM" if bin_probs[1] >= 0.5 else "NON-TUM",
            "latent_vector": latent_vec,
        }
