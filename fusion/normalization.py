"""
Morphology Feature Normalization & Parameter Persistence.
"""

import json
from pathlib import Path
from typing import Union, Optional, List
import numpy as np


class FeatureNormalizer:
    """
    StandardScaler-like normalization with parameter persistence for exact reproducibility.
    """

    def __init__(self, means: Optional[np.ndarray] = None, stds: Optional[np.ndarray] = None):
        self.means: Optional[np.ndarray] = np.asarray(means, dtype=np.float32) if means is not None else None
        self.stds: Optional[np.ndarray] = np.asarray(stds, dtype=np.float32) if stds is not None else None

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """
        Fits mean and standard deviation across a batch of feature vectors [N, 16].
        """
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        self.means = np.mean(arr, axis=0)
        self.stds = np.std(arr, axis=0)
        # Prevent division by zero
        self.stds[self.stds < 1e-6] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transforms features using stored mean and std.
        """
        if self.means is None or self.stds is None:
            # Default identity normalization if not yet fitted
            return np.asarray(features, dtype=np.float32)

        arr = np.asarray(features, dtype=np.float32)
        normalized = (arr - self.means) / self.stds
        return np.nan_to_num(normalized, nan=0.0, posinf=5.0, neginf=-5.0)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves fitted normalization parameters to a JSON file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else [],
            "feature_dim": len(self.means) if self.means is not None else 16,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureNormalizer":
        """
        Loads fitted normalization parameters from a JSON file.
        """
        path = Path(filepath)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        means = np.array(data.get("means", []), dtype=np.float32) if data.get("means") else None
        stds = np.array(data.get("stds", []), dtype=np.float32) if data.get("stds") else None
        return cls(means=means, stds=stds)
