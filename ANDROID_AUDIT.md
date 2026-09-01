# ANDROID_AUDIT.md — Mobile Application & Integration Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Android & Mobile Digital Pathology Team  

---

## 1. Android Architecture & UI Screen Map

The complete mobile workstation architecture is structured around 15 dedicated screens:

```
Screen 01: Splash & Server Status
      ↓
Screen 02: Case List & New Case Capture
      ↓
Screen 03: Live Microscope Viewfinder (CameraX / UVC Eyepiece)
      ↓
Screen 04: Quality Gate Feedback (Blur variance, Brightness)
      ↓
Screen 05: Primary Diagnosis Dashboard (Class, Calibrated %, Entropy)
      ↓
Screen 06: Interactive 7-Layer Visual Overlay Switcher (Coil / Glide)
      ↓
Screen 07: Spatial Region Prioritization (Boxes R_01 - R_04)
      ↓
Screen 08: Region Detail & Morphometric Zoom
      ↓
Screen 09: Quantitative Histomorphometry Breakdown
      ↓
Screen 10: Multi-Source Model Agreement & Consensus
      ↓
Screen 11: Qdrant Reference Case Matches & Similarity
      ↓
Screen 12: MedGemma Structured Diagnostic Report
      ↓
Screen 13: Pathologist Copilot Interactive Chat (POST /copilot/ask)
      ↓
Screen 14: Pathologist Review Sign-off Modal
      ↓
Screen 15: PDF Diagnostic Export & Audit Log
```

---

## 2. Kotlin Data Models & Retrofit Client
* **Data Classes:** Fully defined in [`docs/ANDROID_DATA_MODELS.md`](file:///c:/Users/kthir/OneDrive/Desktop/colon_model/docs/ANDROID_DATA_MODELS.md).
* **Retrofit Interface:** `ColonPathApiService` with non-blocking Kotlin Coroutines (`suspend fun`).
* **Image Layer Streaming:** Utilizes OkHttp caching and Coil image loaders for instant switching across the 7 PNG layers.
