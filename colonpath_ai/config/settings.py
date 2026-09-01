"""
COLONPATH-AI Dynamic Configuration Settings.
Provides environment-aware configuration for local datasets, model checkpoints,
calibration parameters, and camera abstraction layers.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_ROOT = Path(r"C:\Users\kthir\Downloads")
DEFAULT_TRAIN_PATH = DEFAULT_DATASET_ROOT / "NCT-CRC-HE-100K" / "NCT-CRC-HE-100K"
DEFAULT_VAL_PATH = DEFAULT_DATASET_ROOT / "CRC-VAL-HE-7K" / "CRC-VAL-HE-7K"


class Settings(BaseModel):
    # Dataset Configurations
    dataset_root: Path = Field(
        default_factory=lambda: Path(os.environ.get("DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))
    )
    train_dataset_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("TRAIN_DATASET_PATH", str(DEFAULT_TRAIN_PATH)))
    )
    val_dataset_path: Path = Field(
        default_factory=lambda: Path(os.environ.get("VAL_DATASET_PATH", str(DEFAULT_VAL_PATH)))
    )

    # Execution Hardware
    device: str = Field(
        default_factory=lambda: os.environ.get("DEVICE", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
    )

    # Model Parameters
    temperature_scaling: float = Field(default=1.25, description="Platt temperature for probability calibration")
    ood_energy_threshold: float = Field(default=-5.0, description="Energy score threshold for OOD detection")
    laplacian_blur_threshold: float = Field(default=30.0, description="Minimum Laplacian variance for focus pass")
    min_brightness: float = Field(default=40.0, description="Minimum mean brightness")
    max_brightness: float = Field(default=220.0, description="Maximum mean brightness")
    min_contrast: float = Field(default=20.0, description="Minimum contrast standard deviation")

    # Qdrant Configuration
    qdrant_collection_name: str = Field(default="colonpath_reference_cohorts")
    qdrant_in_memory: bool = Field(default=True)
    qdrant_storage_path: Optional[str] = Field(default=None)

    # MedGemma VLM
    medgemma_model_id: str = Field(default="google/medgemma-1.5-4b-it")
    medgemma_max_new_tokens: int = Field(default=512)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)


# Global Singleton Settings Instance
settings = Settings()
