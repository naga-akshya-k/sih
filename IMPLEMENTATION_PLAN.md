# IMPLEMENTATION_PLAN.md — COLONPATH-AI Enhancement & Deployment Roadmap

**Date:** September 1, 2026  
**Auditor / Architect:** Senior Medical AI & Digital Pathology Engineering Team  

---

## 1. Objectives & Scope

To elevate the COLONPATH-AI platform to complete production and microscope deployment readiness by implementing:
1. **Dynamic Configurable Settings** for dataset paths and model parameters.
2. **Camera Hardware Abstraction Layer** with virtual replay and USB/UVC drivers.
3. **Authentic Qdrant Reference Indexing** using real Digepath visual and histomorphometric embeddings.
4. **Camera Streaming Endpoints** (`POST /camera/start`, `POST /camera/stop`, `GET /camera/status`, `POST /camera/frame`).
5. **Color & Stain Domain Adaptation Monitoring** for real microscope optical variance.
6. **Dataset Training & Evaluation Scripts** operating cleanly on local `NCT-CRC-HE-100K` and `CRC-VAL-HE-7K`.

---

## 2. Phase-by-Phase Implementation Steps

### Phase 1: Dynamic Configuration System
* **File:** `colonpath_ai/config/settings.py`
* **Features:** Pydantic-based settings supporting `DATASET_ROOT`, `TRAIN_DATASET_PATH` (`C:\Users\kthir\Downloads\NCT-CRC-HE-100K`), `VAL_DATASET_PATH` (`C:\Users\kthir\Downloads\CRC-VAL-HE-7K`), `QDRANT_HOST`, `DEVICE`, etc.

### Phase 2: Camera Abstraction Layer
* **File:** `colonpath_ai/camera/camera_source.py`
* **Features:** Base `CameraSource` class, `USBCameraSource`, `AndroidUVCSource`, and `ImageReplaySource` for offline test streams.

### Phase 3: Authentic Feature Seeding for Qdrant RAG
* **File:** `colonpath_ai/reference/qdrant_matcher.py`
* **Features:** Replaces simulated seed vectors with real 1024-d Digepath embeddings extracted from verified reference cases.

### Phase 4: Camera Streaming & Ingestion Routes
* **File:** `colonpath_ai/api/routes/camera.py`
* **Features:** Register `/camera/start`, `/camera/stop`, `/camera/status`, and `/camera/frame` in FastAPI.

### Phase 5: Domain Shift & Stain Normalization
* **File:** `colonpath_ai/quality/stain_normalizer.py`
* **Features:** Macenko / Reinhard stain normalization and optical distribution distance metrics.

### Phase 6: Dataset Evaluation & Benchmarking
* **File:** `colonpath_ai/scripts/evaluate_dataset.py`
* **Features:** Evaluates the multimodal model on `CRC-VAL-HE-7K` independent test set.

### Phase 7: Verification & Test Suite Expansion
* **Files:** `colonpath_ai/tests/`
* **Features:** Add test cases for camera abstraction, dynamic settings, and stain normalization.
