"""
Digepath Feature Extraction & Inference Interface.
Extracts GI visual embeddings from H&E images with caching and batch support.
"""

import argparse
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch

from .model_loader import DigepathModelLoader, EMBEDDING_DIM
from .preprocess import preprocess_image
from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class DigepathFeatureExtractor:
    """
    High-level feature extractor using frozen Digepath foundation model.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.loader = DigepathModelLoader.get_instance(device=device)
        self.model = self.loader.model
        self.device = self.loader.device
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir=cache_dir) if use_cache else None

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.loader.info

    def extract(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extracts 1024-dimensional feature embedding for a single image.

        Args:
            image_input: Filepath, PIL Image, or Numpy array.
            cache_key: Optional explicit cache key (defaults to path if string/Path).

        Returns:
            np.ndarray of shape (1024,), dtype float32.
        """
        # Determine cache key
        if self.use_cache and self.cache is not None:
            key = cache_key or (str(image_input) if isinstance(image_input, (str, Path)) else None)
            if key is not None:
                cached = self.cache.get(key)
                if cached is not None:
                    return cached

        # Preprocess to tensor [1, 3, 224, 224]
        tensor = preprocess_image(image_input).to(self.device)

        # Forward pass in inference mode
        with torch.no_grad():
            features = self.model(tensor)  # [1, 1024]
            if isinstance(features, (tuple, list)):
                features = features[0]
            emb = features.squeeze(0).cpu().numpy().astype(np.float32)

        # Normalize L2 norm
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm

        # Cache if key exists
        if self.use_cache and self.cache is not None and key is not None:
            self.cache.put(key, emb)

        return emb

    def extract_batch(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Extracts embeddings for a batch of images.

        Returns:
            np.ndarray of shape (N, 1024), dtype float32.
        """
        embeddings = []
        for i in range(0, len(image_inputs), batch_size):
            batch_items = image_inputs[i : i + batch_size]
            tensors = [preprocess_image(item) for item in batch_items]
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            with torch.no_grad():
                features = self.model(batch_tensor)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                batch_emb = features.cpu().numpy().astype(np.float32)

            # L2 normalize each row
            norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            batch_emb = batch_emb / norms

            embeddings.append(batch_emb)

        return np.vstack(embeddings) if embeddings else np.empty((0, EMBEDDING_DIM), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Digepath Single-Image Feature Extractor")
    parser.add_argument("--image", required=True, help="Path to input H&E image")
    parser.add_argument("--output", default=None, help="Optional path to save embedding .npy")
    args = parser.parse_args()

    extractor = DigepathFeatureExtractor()
    print("=" * 60)
    print("DIGEPATH FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Model Architecture: {extractor.metadata['architecture']}")
    print(f"Embedding Dim:      {extractor.embedding_dim}")
    print(f"Device:             {extractor.device}")
    print(f"Input Image:        {args.image}")

    emb = extractor.extract(args.image)
    print(f"[OK] Extracted Embedding Shape: {emb.shape}")
    print(f"[OK] Embedding L2 Norm:         {np.linalg.norm(emb):.4f}")
    print(f"[OK] Sample Values (first 5):   {emb[:5]}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, emb)
        print(f"[OK] Saved embedding to: {out_path}")


if __name__ == "__main__":
    main()
