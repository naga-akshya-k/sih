"""
COLONPATH-AI Camera Abstraction Layer.
Decouples microscope and camera acquisition hardware from the core AI inference pipeline.
Supports USB eyepiece cameras, Android UVC streams, and offline replay testing.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraMetadata:
    frame_id: int = 0
    timestamp_utc: float = field(default_factory=time.time)
    width: int = 0
    height: int = 0
    source_type: str = "GENERIC"
    fps: float = 0.0
    exposure_ms: Optional[float] = None
    physical_scale_um_per_px: Optional[float] = None  # None indicates "PHYSICAL SCALE NOT CALIBRATED"


class CameraSource(ABC):
    """
    Abstract Base Class for all pathology image acquisition sources.
    """

    def __init__(self):
        self._is_active = False
        self._frame_count = 0

    @abstractmethod
    def start(self) -> bool:
        """Initializes the camera connection or stream."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], CameraMetadata]:
        """
        Captures a single optical frame.
        Returns (success, frame_bgr_or_rgb, metadata).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Releases the camera connection."""
        pass

    def is_active(self) -> bool:
        return self._is_active


class USBCameraSource(CameraSource):
    """
    DirectShow / UVC Driver for USB Microscope Eyepiece Cameras (e.g. AmScope, SVBONY).
    """

    def __init__(self, device_index: int = 0, target_width: int = 1920, target_height: int = 1080):
        super().__init__()
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self._cap = None

    def start(self) -> bool:
        try:
            # DirectShow backend on Windows
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend
                self._cap = cv2.VideoCapture(self.device_index)

            if not self._cap.isOpened():
                logger.warning(f"Unable to open USB Camera device index {self.device_index}")
                self._is_active = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._is_active = True
            logger.info(f"USBCameraSource initialized on device index {self.device_index}")
            return True
        except Exception as e:
            logger.error(f"Error starting USBCameraSource: {e}")
            self._is_active = False
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], CameraMetadata]:
        if not self._is_active or self._cap is None:
            return False, None, CameraMetadata()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None, CameraMetadata()

        self._frame_count += 1
        h, w = frame.shape[:2]
        meta = CameraMetadata(
            frame_id=self._frame_count,
            width=w,
            height=h,
            source_type="USB_MICROSCOPE_CAMERA",
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            physical_scale_um_per_px=None  # Physical scale uncalibrated by default
        )
        return True, frame, meta

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_active = False
        logger.info("USBCameraSource stopped.")


class ImageReplaySource(CameraSource):
    """
    Virtual camera that streams pre-recorded microscope slides or dataset images.
    Used for reproducible development, testing, and continuous replay.
    """

    def __init__(self, image_directory: str, fps: float = 10.0, loop: bool = True):
        super().__init__()
        self.image_directory = Path(image_directory)
        self.fps = fps
        self.loop = loop
        self._image_files: List[Path] = []
        self._current_index = 0
        self._last_frame_time = 0.0

    def start(self) -> bool:
        if not self.image_directory.exists():
            logger.warning(f"Replay directory does not exist: {self.image_directory}")
            self._is_active = False
            return False

        exts = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
        self._image_files = []
        for ext in exts:
            self._image_files.extend(list(self.image_directory.glob(ext)))
        self._image_files.sort()

        if not self._image_files:
            logger.warning(f"No images found in replay directory: {self.image_directory}")
            self._is_active = False
            return False

        self._is_active = True
        self._current_index = 0
        logger.info(f"ImageReplaySource started with {len(self._image_files)} frames from {self.image_directory}")
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], CameraMetadata]:
        if not self._is_active or not self._image_files:
            return False, None, CameraMetadata()

        # Enforce target FPS timing
        now = time.time()
        elapsed = now - self._last_frame_time
        target_interval = 1.0 / self.fps
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)

        img_path = self._image_files[self._current_index]
        frame = cv2.imread(str(img_path))
        if frame is None:
            return False, None, CameraMetadata()

        self._frame_count += 1
        self._last_frame_time = time.time()
        h, w = frame.shape[:2]

        meta = CameraMetadata(
            frame_id=self._frame_count,
            width=w,
            height=h,
            source_type="IMAGE_REPLAY_STREAM",
            fps=self.fps,
            physical_scale_um_per_px=0.5 if "NCT" in str(img_path) or "CRC" in str(img_path) else None
        )

        # Advance pointer
        self._current_index += 1
        if self._current_index >= len(self._image_files):
            if self.loop:
                self._current_index = 0
            else:
                self._is_active = False

        return True, frame, meta

    def stop(self) -> None:
        self._is_active = False
        logger.info("ImageReplaySource stopped.")


class AndroidUVCSource(CameraSource):
    """
    Ingests frames pushed asynchronously from an Android smartphone connected via USB OTG.
    """

    def __init__(self):
        super().__init__()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_meta: Optional[CameraMetadata] = None

    def start(self) -> bool:
        self._is_active = True
        logger.info("AndroidUVCSource active and ready to receive frames.")
        return True

    def push_frame(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Called by FastAPI route when Android uploads a frame."""
        self._frame_count += 1
        h, w = frame.shape[:2]
        self._latest_frame = frame
        self._latest_meta = CameraMetadata(
            frame_id=self._frame_count,
            width=w,
            height=h,
            source_type="ANDROID_SMARTPHONE_OTG",
            fps=metadata.get("fps", 15.0) if metadata else 15.0,
            physical_scale_um_per_px=metadata.get("scale_um_per_px") if metadata else None
        )

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], CameraMetadata]:
        if not self._is_active or self._latest_frame is None:
            return False, None, CameraMetadata()
        return True, self._latest_frame.copy(), self._latest_meta or CameraMetadata()

    def stop(self) -> None:
        self._is_active = False
        self._latest_frame = None
        logger.info("AndroidUVCSource stopped.")
