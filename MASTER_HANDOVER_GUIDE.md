# COLONPATH-AI: Master Project & Android Developer Guide

Welcome! This is the complete **COLONPATH-AI** decision-support system codebase, trained AI models, REST backend, and mobile integration assets.

---

## 🧭 Directory Map — Where to Find Everything

```
colon_model/
├── 📁 outputs/android_handover/  <--- 🌟 START HERE FOR ANDROID APP DEVELOPMENT
│   ├── README_FOR_DEVELOPER.md       # Ready-to-copy Kotlin data classes & Retrofit interface
│   ├── ANDROID_API_SPECIFICATION.md  # Complete REST API documentation & UI/UX flow
│   ├── sample_case_result.json       # Exact sample JSON response for data modeling
│   ├── best_classifier.pth           # Trained PyTorch multimodal classifier weights
│   ├── multimodal_classifier_mobile.pt # TorchScript model compiled for Android PyTorch Mobile
│   ├── unet_gland_model.pth          # U-Net gland segmentation checkpoint
│   └── normalization_params.json     # Feature scaling parameters
│
├── 📁 api/                           # FastAPI REST API Backend (endpoints, routes, schemas)
│   ├── main.py                       # FastAPI application entry point
│   ├── schemas.py                    # Pydantic v2 schemas for all requests/responses
│   ├── routes/                       # /health, /analyze, /cases, /regions, /review
│   └── services/                     # Business logic and service layers
│
├── 📁 web/                           # Live Interactive Web Dashboard
│   └── index.html                    # Single-page HTML5/JS histopathology layer viewer
│
├── 📁 foundation/digepath/           # Digepath ViT-L/16 GI Foundation Model extractor & cache
├── 📁 fusion/                        # Multimodal Late-Fusion Network (Visual + Morphology)
├── 📁 classifiers/                   # 9-Class tissue classifier & training/eval pipelines
├── 📁 uncertainty/                   # Temperature scaling & entropy uncertainty engine
├── 📁 agreement/                     # Multi-source model agreement engine
├── 📁 regions/                       # AI-prioritized spatial patch analyzer & navigator
├── 📁 reference/                     # Reference case similarity comparator
├── 📁 visualization/                 # 7 authentic visual layer overlay renderers
├── 📁 evidence/                      # Deterministic evidence.json & case_result.json builder
├── 📁 agent/                         # Anti-hallucination evidence validator
├── 📁 storage/                       # SQLite database manager & case repository
├── 📁 orchestrator/                  # End-to-end master analysis pipeline
│
├── 📁 outputs/
│   ├── visualizations/CASE_DEMO_00000/ # Sample rendered PNG images (Original, Glands, Nuclei, Regions, Heatmap, 3D)
│   ├── cases/CASE_DEMO_00000/          # Sample case_result.json and evidence.json
│   ├── models/                         # Trained classifier weights & calibration parameters
│   ├── unet/                           # U-Net checkpoint & predictions
│   └── hovernet_test/                  # HoVer-Net overlays & sample inputs (00000.png)
│
├── 📁 docs/                          # Comprehensive technical documentation
│   ├── android_api.md                # Full Android API contract & 10-screen UI/UX specification
│   ├── reproducibility.md            # Hardware, seeds, hyperparameters & benchmark metrics
│   └── existing_system_audit.md      # Preserved vs built components audit
│
└── 📁 tests/                         # 18 automated unit and integration tests
```

---

## 🚀 How to Run the Backend Server on Your Computer

If you want to run the live AI server locally while developing the Android app:

### 1. Install Requirements
```powershell
pip install fastapi uvicorn torch torchvision timm scikit-learn pydantic pillow opencv-python matplotlib
```

### 2. Start the Server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 3. Open in Your Browser
- **Live Interactive Dashboard:** `http://127.0.0.1:8080/`
- **Swagger API Documentation:** `http://127.0.0.1:8080/docs`

---

## 📱 Android App Connection URLs

- **Android Studio Emulator:** `http://10.0.2.2:8080`
- **Physical Android Phone (Same Wi-Fi):** `http://<HOST_COMPUTER_IP>:8080`

---

## 🧪 Run Automated Tests
```powershell
pytest tests/ -v
```
*(All 18 tests pass across API, fusion, uncertainty, regions, and end-to-end pipeline).*
