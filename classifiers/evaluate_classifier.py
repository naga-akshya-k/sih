"""
Evaluation Pipeline for Multimodal Colorectal Tissue Classifier.
Generates comprehensive evaluation metrics, classification reports, confusion matrix, and calibration curve.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    classification_report,
)
import torch

from fusion.fusion_model import TISSUE_CLASSES, NUM_CLASSES
from foundation.digepath.inference import DigepathFeatureExtractor
from .dataset import create_data_splits
from .tissue_classifier import TissueClassifier

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path) -> None:
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Multimodal Tissue Classifier - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_calibration_curve(
    probs: np.ndarray, labels: np.ndarray, ece: float, save_path: Path, n_bins: int = 10
) -> None:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_accs.append(np.mean(accuracies[in_bin]))
            bin_confs.append(np.mean(confidences[in_bin]))

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    if bin_confs:
        plt.plot(bin_confs, bin_accs, "s-", color="crimson", label=f"Classifier (ECE = {ece:.4f})")
    plt.title("Reliability Diagram / Calibration Curve", fontsize=14)
    plt.xlabel("Mean Confidence", fontsize=12)
    plt.ylabel("Observed Accuracy", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def evaluate_on_test_set(
    dataset_dir: Path,
    max_samples: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 60)
    print("EVALUATING MULTIMODAL CLASSIFIER ON HELD-OUT TEST SET")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = DigepathFeatureExtractor(device=device)

    # Load splits with exact fixed seed
    _, _, test_ds, normalizer = create_data_splits(
        dataset_dir, extractor, max_samples=max_samples, seed=seed
    )

    classifier = TissueClassifier(device=device)

    all_mc_preds = []
    all_mc_probs = []
    all_mc_targets = []
    all_bin_preds = []
    all_bin_targets = []

    for i in range(len(test_ds)):
        item = test_ds[i]
        v_emb = item["visual_embedding"].numpy()
        m_feat = item["morphology_feature"].numpy()
        mc_target = item["multiclass_label"].item()
        bin_target = item["binary_label"].item()

        # Run inference using TissueClassifier
        res = classifier.predict(v_emb, m_feat)
        probs_vec = [res["multiclass_probabilities"][c] for c in TISSUE_CLASSES]

        all_mc_preds.append(res["prediction_index"])
        all_mc_probs.append(probs_vec)
        all_mc_targets.append(mc_target)
        all_bin_preds.append(1 if res["binary_prediction"] == "TUM" else 0)
        all_bin_targets.append(bin_target)

    all_mc_preds = np.array(all_mc_preds)
    all_mc_probs = np.array(all_mc_probs)
    all_mc_targets = np.array(all_mc_targets)
    all_bin_preds = np.array(all_bin_preds)
    all_bin_targets = np.array(all_bin_targets)

    # 1. Compute multiclass metrics
    acc = float(accuracy_score(all_mc_targets, all_mc_preds))
    bal_acc = float(balanced_accuracy_score(all_mc_targets, all_mc_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_mc_targets, all_mc_preds, average="macro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        all_mc_targets, all_mc_preds, average="weighted", zero_division=0
    )

    # Per-class report
    present_classes = np.unique(np.concatenate([all_mc_targets, all_mc_preds]))
    target_names = [TISSUE_CLASSES[i] for i in present_classes]
    report_dict = classification_report(
        all_mc_targets,
        all_mc_preds,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # 2. Binary metrics (TUM vs NON-TUM)
    bin_acc = float(accuracy_score(all_bin_targets, all_bin_preds))
    bin_prec, bin_rec, bin_f1, _ = precision_recall_fscore_support(
        all_bin_targets, all_bin_preds, average="binary", zero_division=0
    )
    bin_cm = confusion_matrix(all_bin_targets, all_bin_preds)
    if bin_cm.shape == (2, 2):
        tn, fp, fn, tp = bin_cm.ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    else:
        specificity = 1.0
        sensitivity = 1.0

    # 3. Calibration metrics
    ece = calculate_ece(all_mc_probs, all_mc_targets)

    # Multiclass Brier score
    one_hot_targets = np.zeros_like(all_mc_probs)
    for idx, t in enumerate(all_mc_targets):
        one_hot_targets[idx, t] = 1.0
    brier_score = float(np.mean(np.sum((all_mc_probs - one_hot_targets) ** 2, axis=1)))

    # AUROC (multiclass one-vs-rest if possible)
    try:
        auroc = float(
            roc_auc_score(
                all_mc_targets,
                all_mc_probs,
                multi_class="ovr",
                average="macro",
                labels=present_classes,
            )
        )
    except Exception:
        auroc = None

    # Full metrics payload
    metrics_summary = {
        "dataset_samples_evaluated": len(all_mc_targets),
        "multiclass_accuracy": acc,
        "multiclass_balanced_accuracy": bal_acc,
        "multiclass_macro_precision": float(prec),
        "multiclass_macro_recall": float(rec),
        "multiclass_macro_f1": float(f1),
        "multiclass_weighted_f1": float(weighted_f1),
        "binary_tumor_accuracy": bin_acc,
        "binary_tumor_precision": float(bin_prec),
        "binary_tumor_recall_sensitivity": float(bin_rec),
        "binary_tumor_specificity": specificity,
        "binary_tumor_f1": float(bin_f1),
        "expected_calibration_error_ece": ece,
        "brier_score": brier_score,
        "macro_auroc": auroc,
    }

    # Save metrics.json & classification_report.json
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(RESULTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save confusion matrix plot & calibration plot
    cm = confusion_matrix(all_mc_targets, all_mc_preds, labels=list(range(NUM_CLASSES)))
    plot_confusion_matrix(cm, TISSUE_CLASSES, RESULTS_DIR / "confusion_matrix.png")
    plot_calibration_curve(all_mc_probs, all_mc_targets, ece, RESULTS_DIR / "calibration.png")

    print("\n" + "=" * 60)
    print("CLASSIFICATION EVALUATION METRICS (TEST SET)")
    print("=" * 60)
    print(f"Test Samples:        {len(all_mc_targets)}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Balanced Accuracy:   {bal_acc:.4f}")
    print(f"Macro F1-Score:      {f1:.4f}")
    print(f"Binary Sensitivity:  {sensitivity:.4f}")
    print(f"Binary Specificity:  {specificity:.4f}")
    print(f"Calibration (ECE):   {ece:.4f}")
    print(f"Brier Score:         {brier_score:.4f}")
    print()
    print("Saved Artifacts:")
    print(f" - {RESULTS_DIR / 'metrics.json'}")
    print(f" - {RESULTS_DIR / 'classification_report.json'}")
    print(f" - {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f" - {RESULTS_DIR / 'calibration.png'}")

    return metrics_summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "datasets" / "conic2022_processed"
    evaluate_on_test_set(data_dir, max_samples=300)
