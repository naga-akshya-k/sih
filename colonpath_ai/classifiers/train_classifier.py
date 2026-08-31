"""
Training Pipeline for Multimodal Colorectal Tissue Classifier.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from foundation.digepath.inference import DigepathFeatureExtractor
from fusion.fusion_model import MultimodalFusionNet, TISSUE_CLASSES, NUM_CLASSES
from .dataset import create_data_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_MODEL_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_mc: nn.Module,
    criterion_bin: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct_mc = 0
    correct_bin = 0
    total = 0

    for batch in loader:
        v_emb = batch["visual_embedding"].to(device)
        m_feat = batch["morphology_feature"].to(device)
        mc_target = batch["multiclass_label"].to(device)
        bin_target = batch["binary_label"].to(device)

        optimizer.zero_grad()
        mc_logits, bin_logits, _ = model(v_emb, m_feat)

        loss_mc = criterion_mc(mc_logits, mc_target)
        loss_bin = criterion_bin(bin_logits, bin_target)
        loss = loss_mc + 0.5 * loss_bin

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(mc_target)
        pred_mc = mc_logits.argmax(dim=-1)
        pred_bin = bin_logits.argmax(dim=-1)

        correct_mc += (pred_mc == mc_target).sum().item()
        correct_bin += (pred_bin == bin_target).sum().item()
        total += len(mc_target)

    return {
        "loss": total_loss / max(1, total),
        "mc_accuracy": correct_mc / max(1, total),
        "bin_accuracy": correct_bin / max(1, total),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_mc: nn.Module,
    criterion_bin: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct_mc = 0
    correct_bin = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            v_emb = batch["visual_embedding"].to(device)
            m_feat = batch["morphology_feature"].to(device)
            mc_target = batch["multiclass_label"].to(device)
            bin_target = batch["binary_label"].to(device)

            mc_logits, bin_logits, _ = model(v_emb, m_feat)

            loss_mc = criterion_mc(mc_logits, mc_target)
            loss_bin = criterion_bin(bin_logits, bin_target)
            loss = loss_mc + 0.5 * loss_bin

            total_loss += loss.item() * len(mc_target)
            pred_mc = mc_logits.argmax(dim=-1)
            pred_bin = bin_logits.argmax(dim=-1)

            correct_mc += (pred_mc == mc_target).sum().item()
            correct_bin += (pred_bin == bin_target).sum().item()
            total += len(mc_target)

    return {
        "loss": total_loss / max(1, total),
        "mc_accuracy": correct_mc / max(1, total),
        "bin_accuracy": correct_bin / max(1, total),
    }


def run_training(
    dataset_dir: Path,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_samples: int = 300,
    seed: int = 42,
) -> None:
    logger.info("=" * 60)
    logger.info("STARTING MULTIMODAL CLASSIFIER TRAINING")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on Device: {device}")

    # 1. Prepare data splits
    extractor = DigepathFeatureExtractor(device=device)
    train_ds, val_ds, test_ds, normalizer = create_data_splits(
        dataset_dir, extractor, max_samples=max_samples, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Save normalization parameters
    norm_path = OUTPUT_MODEL_DIR / "normalization_params.json"
    normalizer.save(norm_path)
    logger.info(f"Saved normalization parameters to: {norm_path}")

    # 3. Initialize Model & Optimizers
    model = MultimodalFusionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion_mc = nn.CrossEntropyLoss()
    criterion_bin = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_ckpt_path = OUTPUT_MODEL_DIR / "best_classifier.pth"

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion_mc, criterion_bin, device)
        val_metrics = evaluate(model, val_loader, criterion_mc, criterion_bin, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['mc_accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['mc_accuracy']:.4f}"
        )

        if val_metrics["mc_accuracy"] >= best_val_acc:
            best_val_acc = val_metrics["mc_accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "classes": TISSUE_CLASSES,
                    "seed": seed,
                },
                best_ckpt_path,
            )

    logger.info(f"✓ Training finished. Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"✓ Best model checkpoint saved to: {best_ckpt_path}")

    # 4. Save training config & class mapping
    config = {
        "architecture": "MultimodalFusionNet",
        "visual_foundation": "Digepath ViT-L/16",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "seed": seed,
        "best_val_accuracy": best_val_acc,
        "classes": TISSUE_CLASSES,
        "num_classes": NUM_CLASSES,
        "morphology_features": 16,
    }
    with open(OUTPUT_MODEL_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "datasets" / "conic2022_processed"
    run_training(data_dir, epochs=20, batch_size=32, max_samples=300)
