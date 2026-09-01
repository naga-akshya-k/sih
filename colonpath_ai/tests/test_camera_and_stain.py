"""
Tests for Camera Abstraction Layer, Dynamic Settings, and Stain Normalization.
"""

import os
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app
from config.settings import settings
from camera.camera_source import ImageReplaySource, AndroidUVCSource, USBCameraSource
from quality.stain_normalizer import ReinhardStainNormalizer, DomainShiftDetector

client = TestClient(app)


def test_dynamic_settings():
    assert settings.temperature_scaling == 1.25
    assert settings.ood_energy_threshold == -5.0
    assert settings.laplacian_blur_threshold == 30.0
    assert settings.train_dataset_path.exists() or "Downloads" in str(settings.train_dataset_path)
    assert settings.val_dataset_path.exists() or "Downloads" in str(settings.val_dataset_path)


def test_stain_normalizer_and_domain_shift():
    normalizer = ReinhardStainNormalizer()
    detector = DomainShiftDetector(shift_threshold=40.0)

    # Test dummy H&E image
    dummy_img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)

    # Normalize
    normed = normalizer.normalize(dummy_img)
    assert normed.shape == (224, 224, 3)
    assert normed.dtype == np.uint8

    # Detect shift
    shift = detector.evaluate_shift(dummy_img)
    assert "domain_shift_detected" in shift
    assert "shift_score" in shift
    assert "recommendation" in shift


def test_camera_replay_and_android_sources():
    # Test Android source
    android_cam = AndroidUVCSource()
    assert android_cam.start() is True
    assert android_cam.is_active() is True

    frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    android_cam.push_frame(frame, {"scale_um_per_px": 0.5})

    ret, read_f, meta = android_cam.read_frame()
    assert ret is True
    assert read_f.shape == (100, 100, 3)
    assert meta.physical_scale_um_per_px == 0.5

    android_cam.stop()
    assert android_cam.is_active() is False


def test_camera_api_routes():
    # 1. Start Android camera
    res = client.post("/camera/start?source_type=android")
    assert res.status_code == 200
    assert res.json()["status"] == "active"

    # 2. Get status
    res = client.get("/camera/status")
    assert res.status_code == 200
    assert res.json()["is_active"] is True

    # 3. Stop camera
    res = client.post("/camera/stop")
    assert res.status_code == 200
    assert res.json()["status"] == "stopped"

    # 4. Status after stop
    res = client.get("/camera/status")
    assert res.status_code == 200
    assert res.json()["is_active"] is False
