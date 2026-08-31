# SIH Project: COLONPATH-AI

This repository contains the complete **COLONPATH-AI** decision-support system for colorectal cancer (CRC) histopathology, organized into a dedicated, self-contained folder:

📂 **Project Directory:** [`colonpath_ai/`](./colonpath_ai/)

---

## 🧭 Repository Structure

```
sih/
├── 📁 colonpath_ai/          # The entire COLONPATH-AI decision-support system
│   ├── 📁 api/               # FastAPI REST Backend endpoints & schemas
│   ├── 📁 foundation/        # Digepath ViT-L/16 GI feature extraction & embedding cache
│   ├── 📁 fusion/            # Multimodal Late-Fusion Network (Visual + Morphology)
│   ├── 📁 classifiers/       # 9-class tissue & binary tumor classifiers
│   ├── 📁 uncertainty/       # Temperature scaling & entropy uncertainty engine
│   ├── 📁 agreement/         # Multi-source model agreement engine
│   ├── 📁 regions/           # AI-prioritized spatial patch analyzer & navigator
│   ├── 📁 reference/         # Reference case similarity comparator
│   ├── 📁 visualization/     # 7 authentic visual layer overlay renderers
│   ├── 📁 evidence/          # Deterministic evidence.json & case_result.json builder
│   ├── 📁 agent/             # Anti-hallucination evidence validator
│   ├── 📁 storage/           # SQLite database manager & case repository
│   ├── 📁 orchestrator/      # End-to-end master analysis pipeline
│   ├── 📁 web/               # Live interactive web dashboard (index.html)
│   ├── 📁 docs/              # Android API specs, UI flow & reproducibility
│   ├── 📁 tests/             # 18 automated unit and integration tests
│   └── 📁 outputs/
│       └── 📁 android_handover/ # Android Mobile Developer Kit (Kotlin models, Retrofit, weights)
│
├── README.md                 # Root project documentation
└── MASTER_HANDOVER_GUIDE.md  # Quickstart and integration guide for developers
```

---

## 🚀 How to Run the Backend Server

```powershell
cd colonpath_ai
pip install fastapi uvicorn torch torchvision timm scikit-learn pydantic pillow opencv-python matplotlib
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080
```

- **Interactive Web Dashboard:** `http://127.0.0.1:8080/`
- **Swagger API Explorer:** `http://127.0.0.1:8080/docs`
- **Android Studio Emulator Connection:** `http://10.0.2.2:8080`

---

## 📱 For Android Mobile App Developers
Go directly to: [`colonpath_ai/outputs/android_handover/README_FOR_DEVELOPER.md`](./colonpath_ai/outputs/android_handover/README_FOR_DEVELOPER.md) to copy the ready-to-use Kotlin data classes (`ColonPathModels.kt`) and Retrofit interface (`ColonPathApiService.kt`).
