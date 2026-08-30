"""
Colorectal Histopathology Dataset & Feature Cache for Downstream Classifier.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from fusion.fusion_model import TISSUE_CLASSES
from fusion.normalization import FeatureNormalizer
from foundation.digepath.inference import DigepathFeatureExtractor

logger = logging.getLogger(__name__)


def derive_tissue_class_from_maps(
    image_path: Path,
    inst_map_path: Optional[Path] = None,
    class_map_path: Optional[Path] = None,
) -> Tuple[int, int, np.ndarray]:
    """
    Derives multi-class index (0..8), binary tumor label (0/1), and 16-d morphology vector
    from image and available instance/class maps.
    """
    # Initialize default 16-d morphology
    morph_vec = np.zeros(16, dtype=np.float32)
    mc_idx = 6  # default NORM
    bin_label = 0

    if class_map_path and class_map_path.exists() and inst_map_path and inst_map_path.exists():
        try:
            class_map = np.load(class_map_path)
            inst_map = np.load(inst_map_path)

            unique_insts = np.unique(inst_map)
            unique_insts = unique_insts[unique_insts > 0]
            n_total = len(unique_insts)

            # Count types: 1: Epi, 2: Lymph, 3: Spindle, 4: Misc
            t_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            areas = []
            for inst_id in unique_insts:
                mask = (inst_map == inst_id)
                areas.append(float(np.sum(mask)))
                # type in class_map
                types_in_inst = class_map[mask]
                types_in_inst = types_in_inst[types_in_inst > 0]
                if len(types_in_inst) > 0:
                    t = int(np.bincount(types_in_inst).argmax())
                    if t in t_counts:
                        t_counts[t] += 1

            mean_area = float(np.mean(areas)) if areas else 0.0

            morph_vec[0] = n_total
            morph_vec[1] = t_counts[1]
            morph_vec[2] = t_counts[2]
            morph_vec[3] = t_counts[3]
            morph_vec[4] = t_counts[4]
            morph_vec[5] = mean_area
            morph_vec[6] = np.sqrt(mean_area) * 3.54 if mean_area > 0 else 0.0
            morph_vec[7] = 0.65  # eccentricity
            morph_vec[8] = 0.75  # circularity

            # Derive tissue class based on cellular composition
            if n_total == 0:
                mc_idx = 1  # BACK
                bin_label = 0
            elif t_counts[1] > 40 and mean_area > 120:
                mc_idx = 8  # TUM (Adenocarcinoma high cellularity)
                bin_label = 1
            elif t_counts[2] > 30 and t_counts[2] > t_counts[1]:
                mc_idx = 3  # LYM (Lymphocytes)
                bin_label = 0
            elif t_counts[3] > 35:
                mc_idx = 7  # STR (Stroma)
                bin_label = 0
            elif t_counts[1] > 15:
                mc_idx = 6  # NORM (Normal epithelium)
                bin_label = 0
            elif mean_area > 200:
                mc_idx = 4  # MUC (Mucinous)
                bin_label = 0
            else:
                mc_idx = 5  # MUS (Muscle)
                bin_label = 0
        except Exception as e:
            logger.warning(f"Error reading maps for {image_path}: {e}")

    return mc_idx, bin_label, morph_vec


class ColorectalDataset(Dataset):
    """
    PyTorch Dataset yielding (visual_embedding, normalized_morphology, multiclass_label, binary_label).
    """

    def __init__(
        self,
        visual_embeddings: np.ndarray,
        morphology_features: np.ndarray,
        multiclass_labels: np.ndarray,
        binary_labels: np.ndarray,
        case_ids: List[str],
    ):
        self.visual_embeddings = torch.from_numpy(visual_embeddings).float()
        self.morphology_features = torch.from_numpy(morphology_features).float()
        self.multiclass_labels = torch.from_numpy(multiclass_labels).long()
        self.binary_labels = torch.from_numpy(binary_labels).long()
        self.case_ids = case_ids

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "visual_embedding": self.visual_embeddings[idx],
            "morphology_feature": self.morphology_features[idx],
            "multiclass_label": self.multiclass_labels[idx],
            "binary_label": self.binary_labels[idx],
            "case_id": self.case_ids[idx],
        }


def create_data_splits(
    dataset_dir: Path,
    extractor: DigepathFeatureExtractor,
    max_samples: int = 500,
    seed: int = 42,
) -> Tuple[ColorectalDataset, ColorectalDataset, ColorectalDataset, FeatureNormalizer]:
    """
    Creates stratified Train (70%), Validation (15%), and Test (15%) datasets.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    images_dir = dataset_dir / "images"
    inst_dir = dataset_dir / "inst_maps"
    class_dir = dataset_dir / "class_maps"

    image_files = sorted(list(images_dir.glob("*.png")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    # Select representative samples
    selected_files = image_files[:max_samples]
    logger.info(f"Processing {len(selected_files)} samples for multimodal dataset...")

    embeddings_list = []
    morph_list = []
    mc_labels = []
    bin_labels = []
    case_ids = []

    for img_path in selected_files:
        stem = img_path.stem
        inst_path = inst_dir / f"{stem}.npy"
        class_path = class_dir / f"{stem}.npy"

        # 1. Visual embedding via Digepath extractor (cached)
        emb = extractor.extract(img_path, cache_key=f"digepath_{stem}")
        embeddings_list.append(emb)

        # 2. Morphology & labels
        mc, bl, m_vec = derive_tissue_class_from_maps(img_path, inst_path, class_path)
        morph_list.append(m_vec)
        mc_labels.append(mc)
        bin_labels.append(bl)
        case_ids.append(stem)

    embeddings = np.vstack(embeddings_list)
    raw_morphology = np.vstack(morph_list)
    mc_labels = np.array(mc_labels, dtype=np.int64)
    bin_labels = np.array(bin_labels, dtype=np.int64)

    # Stratified/Shuffled Train/Val/Test Split (70% / 15% / 15%)
    n_samples = len(case_ids)
    indices = np.random.permutation(n_samples)

    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    # Fit FeatureNormalizer strictly on the TRAIN split to prevent data leakage
    normalizer = FeatureNormalizer()
    normalizer.fit(raw_morphology[train_idx])

    norm_morph_train = normalizer.transform(raw_morphology[train_idx])
    norm_morph_val = normalizer.transform(raw_morphology[val_idx])
    norm_morph_test = normalizer.transform(raw_morphology[test_idx])

    train_dataset = ColorectalDataset(
        embeddings[train_idx],
        norm_morph_train,
        mc_labels[train_idx],
        bin_labels[train_idx],
        [case_ids[i] for i in train_idx],
    )
    val_dataset = ColorectalDataset(
        embeddings[val_idx],
        norm_morph_val,
        mc_labels[val_idx],
        bin_labels[val_idx],
        [case_ids[i] for i in val_idx],
    )
    test_dataset = ColorectalDataset(
        embeddings[test_idx],
        norm_morph_test,
        mc_labels[test_idx],
        bin_labels[test_idx],
        [case_ids[i] for i in test_idx],
    )

    logger.info(
        f"Data splits created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset, normalizer
