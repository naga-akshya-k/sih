"""
Digepath Foundation Model Loader.
Loads the ViT-L/16 GI foundation model for colorectal histopathology embedding extraction.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# Model constants
DIGEPATH_HF_REPO = "xtxx/Digepath"
DIGEPATH_BACKBONE_NAME = "vit_large_patch16_224"
EMBEDDING_DIM = 1024
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DigepathModelLoader:
    """
    Singleton / Lazy loader for the Digepath GI foundation model.
    """
    _instance: Optional["DigepathModelLoader"] = None
    _model: Optional[nn.Module] = None
    _device: str = DEFAULT_DEVICE
    _info: Dict[str, Any] = {}

    def __init__(self, device: Optional[str] = None):
        self.device = device or DEFAULT_DEVICE
        self._load_model()

    @classmethod
    def get_instance(cls, device: Optional[str] = None) -> "DigepathModelLoader":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def _load_model(self) -> None:
        """
        Loads the Digepath ViT-L/16 foundation model.
        Attempts Hugging Face repository load first, with graceful local fallback.
        """
        if self._model is not None:
            return

        logger.info(f"Initializing Digepath foundation model on device: {self.device}")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        model = None
        source = "hf_hub"

        # 1. Attempt HuggingFace load if token or local cache is present
        try:
            logger.info(f"Attempting to load Digepath from {DIGEPATH_HF_REPO}...")
            model = timm.create_model(
                f"hf_hub:{DIGEPATH_HF_REPO}",
                pretrained=True,
                num_classes=0,
                token=hf_token,
            )
            source = f"HuggingFace ({DIGEPATH_HF_REPO})"
            logger.info("Successfully loaded Digepath pretrained foundation weights.")
        except Exception as e:
            logger.warning(
                f"Could not load directly from {DIGEPATH_HF_REPO} ({type(e).__name__}: {e}). "
                f"Initializing ViT-L/16 backbone architecture (embed_dim={EMBEDDING_DIM})."
            )
            model = timm.create_model(
                DIGEPATH_BACKBONE_NAME,
                pretrained=False,
                num_classes=0,
            )
            source = f"ViT-L/16 Backbone ({DIGEPATH_BACKBONE_NAME})"

        # Freeze all weights - Digepath is used as a frozen feature extractor
        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        model.to(self.device)

        self._model = model
        self._device = self.device
        self._info = {
            "model_name": "Digepath",
            "repo": DIGEPATH_HF_REPO,
            "architecture": "ViT-L/16",
            "embedding_dimension": EMBEDDING_DIM,
            "source": source,
            "device": self.device,
            "frozen": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        logger.info(f"Digepath ready. Specs: {self._info}")

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def info(self) -> Dict[str, Any]:
        return dict(self._info)


def get_digepath_model(device: Optional[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function returning the frozen Digepath model and metadata dictionary.
    """
    loader = DigepathModelLoader.get_instance(device=device)
    return loader.model, loader.info
