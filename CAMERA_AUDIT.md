# CAMERA_AUDIT.md — Camera Acquisition & Abstraction Layer Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Computer Vision & Hardware Integration Team  

---

## 1. Camera Abstraction Architecture

To support diverse digital pathology acquisition hardware without vendor lock-in, the system architecture decouples the AI pipeline from the camera hardware via a generic `CameraSource` interface:

```
┌───────────────────────────────────────────────────────────┐
│                    CameraSource (Abstract)                │
│  + start() -> bool                                        │
│  + read_frame() -> Tuple[bool, np.ndarray]                │
│  + get_metadata() -> CameraMetadata                       │
│  + stop() -> None                                         │
└─────────────────────────────┬─────────────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  USBCameraSource │ │  AndroidUVCSource│ │ ImageReplaySource│
│  (DirectShow/UVC)│ │  (Smartphone OTG)│ │ (Pre-recorded)   │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 2. Hardware Compatibility Specifications

| Camera Category | Interface / Protocol | Driver / Library | Verified Device Models | Resolution Supported |
| :--- | :--- | :--- | :--- | :--- |
| **USB Electronic Eyepiece** | UVC (USB Video Class) | `libuvc` / `UVCCamera` | AmScope MD800, SVBONY SV205 | $1920\times1080$, $3840\times2160$ |
| **Smartphone Primary Sensor** | Camera2 / CameraX | Android CameraX API | Sony IMX, Samsung ISOCELL | $12\text{ MP}$–$50\text{ MP}$ Raw / Binned |
| **Testing Replay Stream** | Virtual Loopback / File | OpenCV VideoCapture | Synthetic & Real Slide Replays | Arbitrary |

---

## 3. Optical Artifacts & Mitigation

* **Vignetting & Uneven Illumination:** Corrected via Flat-Field Calibration / adaptive brightness thresholds.
* **Fine-Focus Blur:** Quantified in real time via Laplacian variance; prompts user to turn the fine-focus dial if variance $< 30.0$.
