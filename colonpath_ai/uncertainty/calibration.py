"""
Temperature Scaling and Probability Calibration Module.
"""

import json
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling to calibrate confidence probabilities on unseen validation logits.
    """

    def __init__(self, temperature: float = 1.25):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scales logits by temperature T.
        """
        t = torch.clamp(self.temperature, min=0.1, max=10.0)
        return logits / t

    def calibrate_probabilities(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Transforms unscaled logits to calibrated probabilities.
        """
        if isinstance(logits, np.ndarray):
            t_logits = torch.from_numpy(logits).float()
        else:
            t_logits = logits.float()

        if t_logits.ndim == 1:
            t_logits = t_logits.unsqueeze(0)

        with torch.no_grad():
            scaled = self.forward(t_logits)
            probs = F.softmax(scaled, dim=-1).squeeze(0).cpu().numpy()
        return probs

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 50) -> "TemperatureScaler":
        """
        Learns temperature parameter T by minimizing NLL on validation logits.
        """
        logits_tensor = torch.from_numpy(val_logits).float()
        labels_tensor = torch.from_numpy(val_labels).long()

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_tensor), labels_tensor)
            loss.backward()
            return loss

        try:
            optimizer.step(eval_step)
        except Exception:
            pass

        return self

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"temperature": float(self.temperature.item())}, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TemperatureScaler":
        path = Path(filepath)
        if not path.exists():
            return cls(temperature=1.25)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(temperature=data.get("temperature", 1.25))
