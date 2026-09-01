"""
COLONPATH-AI Camera API Routes.
Provides REST endpoints for camera source lifecycle management, live frame ingestion,
and streaming acquisition for Android and microscope eyepiece cameras.
"""

import time
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
import numpy as np
import cv2

from camera.camera_source import (
    CameraSource,
    USBCameraSource,
    AndroidUVCSource,
    ImageReplaySource,
    CameraMetadata,
)
from quality.image_quality import ImageQualityChecker
from quality.stain_normalizer import DomainShiftDetector, ReinhardStainNormalizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/camera", tags=["Microscope & Camera Acquisition"])

# Global Active Camera Source Singleton
_active_camera: Optional[CameraSource] = None
_quality_checker = ImageQualityChecker()
_domain_detector = DomainShiftDetector()
_stain_normalizer = ReinhardStainNormalizer()


@router.post("/start", summary="Start camera acquisition stream")
async def start_camera(source_type: str = "android", device_index: int = 0, replay_path: Optional[str] = None):
    """
    Starts camera acquisition for USB eyepiece, Android OTG, or replay stream.
    """
    global _active_camera

    if _active_camera and _active_camera.is_active():
        _active_camera.stop()

    if source_type.lower() == "usb":
        _active_camera = USBCameraSource(device_index=device_index)
    elif source_type.lower() == "replay":
        target_dir = replay_path or r"C:\Users\kthir\Downloads\CRC-VAL-HE-7K\CRC-VAL-HE-7K\NORM"
        _active_camera = ImageReplaySource(image_directory=target_dir, fps=10.0)
    else:
        _active_camera = AndroidUVCSource()

    success = _active_camera.start()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start camera source of type '{source_type}'",
        )

    return {
        "status": "active",
        "source_type": source_type,
        "device_index": device_index if source_type == "usb" else None,
        "message": f"Camera source '{source_type}' started successfully.",
    }


@router.post("/stop", summary="Stop active camera acquisition stream")
async def stop_camera():
    """
    Stops and releases the currently active camera stream.
    """
    global _active_camera

    if _active_camera and _active_camera.is_active():
        _active_camera.stop()
        _active_camera = None
        return {"status": "stopped", "message": "Camera stream stopped successfully."}

    return {"status": "idle", "message": "No active camera stream to stop."}


@router.get("/status", summary="Get active camera status and acquisition metrics")
async def get_camera_status():
    """
    Returns real-time status of microscope camera acquisition.
    """
    global _active_camera

    if _active_camera and _active_camera.is_active():
        return {
            "is_active": True,
            "source_type": type(_active_camera).__name__,
            "frame_count": _active_camera._frame_count,
            "physical_scale": "0.5 µm/px" if isinstance(_active_camera, ImageReplaySource) else "PHYSICAL SCALE NOT CALIBRATED",
        }

    return {
        "is_active": False,
        "source_type": None,
        "frame_count": 0,
        "physical_scale": "PHYSICAL SCALE NOT CALIBRATED",
    }


@router.post("/frame", summary="Ingest live frame from Android smartphone")
async def ingest_frame(
    file: UploadFile = File(...),
    scale_um_per_px: Optional[float] = Form(None),
    apply_stain_norm: bool = Form(False),
):
    """
    Receives a single frame pushed from Android smartphone over USB-C OTG.
    Executes real-time optical quality gate and domain-shift evaluation.
    """
    global _active_camera

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image payload; could not decode frame.",
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Quality Check
    quality = _quality_checker.evaluate(img_rgb)

    # 2. Domain Shift Check
    domain_shift = _domain_detector.evaluate_shift(img_rgb)

    # 3. Optional Stain Normalization
    if apply_stain_norm and domain_shift.get("domain_shift_detected"):
        img_rgb = _stain_normalizer.normalize(img_rgb)

    # Update active Android source if running
    if isinstance(_active_camera, AndroidUVCSource) and _active_camera.is_active():
        _active_camera.push_frame(img_rgb, {"scale_um_per_px": scale_um_per_px})

    return {
        "status": "received",
        "frame_shape": list(img_rgb.shape),
        "quality_gate": quality,
        "domain_shift": domain_shift,
        "physical_scale": f"{scale_um_per_px} µm/px" if scale_um_per_px else "PHYSICAL SCALE NOT CALIBRATED",
        "timestamp_utc": time.time(),
    }
