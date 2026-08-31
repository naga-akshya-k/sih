# COLONPATH-AI: Android Application API Contract & UX Specification

**Version:** 1.0.0  
**Target Platform:** Android (Kotlin, Jetpack Compose, Material 3)  
**Base URL:** `http://<SERVER_HOST>:8000` (Development Default: `http://10.0.2.2:8000` for Android Emulator)  

---

## 1. Architectural Overview

The COLONPATH-AI Android Application connects pathologists and clinicians with the multimodal AI decision-support backend. The mobile client is responsible for image capture/upload, interactive layer visualization, AI-prioritized region navigation, evidence inspection, and pathologist-in-the-loop annotations.

### Recommended Android Tech Stack
- **Language:** Kotlin (1.9+)
- **UI Framework:** Jetpack Compose with Material 3 Design
- **Networking:** Retrofit 2 + OkHttp 4 + Kotlinx Serialization / Gson
- **Image Loading:** Coil (`io.coil-kt:coil-compose`)
- **Camera Integration:** CameraX for direct microscope eyepiece / slide capture
- **Local Cache:** Room Database / Jetpack DataStore

---

## 2. API Endpoints Contract

### 2.1 Health Check
* **Endpoint:** `GET /health`
* **Description:** Verifies server availability and GPU hardware state.
* **Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "COLONPATH-AI Multimodal Backend",
  "version": "1.0.0",
  "device": "cuda",
  "models_ready": true
}
```

---

### 2.2 Upload & Analyze Image
* **Endpoint:** `POST /analyze`
* **Content-Type:** `multipart/form-data`
* **Form Parameters:**
  - `image`: Binary image file (PNG, JPEG, BMP, TIFF)
  - `case_id` (optional): Unique string identifier for the case.
* **Response (200 OK):** Primary unified `case_result.json`
```json
{
  "case_id": "CASE_2026_001",
  "timestamp": "2026-08-30T13:35:00.000Z",
  "status": "completed",
  "image_quality": {
    "passed": true,
    "resolution": "256x256",
    "blur_laplacian_variance": 128.45,
    "blur_status": "ACCEPTABLE",
    "mean_brightness": 142.10,
    "brightness_status": "ACCEPTABLE",
    "contrast_std": 48.20,
    "contrast_status": "ACCEPTABLE",
    "mean_saturation": 88.30
  },
  "digepath": {
    "model_name": "Digepath",
    "architecture": "ViT-L/16",
    "embedding_dimension": 1024,
    "device": "cuda",
    "status": "active"
  },
  "prediction": {
    "class": "TUM",
    "confidence": 0.9340,
    "calibrated_confidence": 0.9120,
    "tumor_probability": 0.9410,
    "binary_class": "TUM",
    "multiclass_probabilities": {
      "ADI": 0.002,
      "BACK": 0.001,
      "DEB": 0.015,
      "LYM": 0.020,
      "MUC": 0.005,
      "MUS": 0.004,
      "NORM": 0.012,
      "STR": 0.030,
      "TUM": 0.911
    }
  },
  "uncertainty": {
    "score": 0.1240,
    "level": "LOW",
    "entropy": 0.3412,
    "normalized_entropy": 0.1553,
    "review_required": false,
    "message": "AI-assisted classification ready for review."
  },
  "model_agreement": {
    "level": "HIGH",
    "score": 1.0,
    "concordant_sources": [
      "Nuclear Morphology (Pleomorphism aligns with tumor likelihood)",
      "Gland Morphology (Architectural distortion aligns with tumor prediction)",
      "Digepath Visual Classifier",
      "Reference Comparison"
    ],
    "discordant_sources": [],
    "summary": "High agreement across all evaluated sources: TUM supported by visual, nuclear, and glandular evidence."
  },
  "nuclear_evidence": {
    "total_count": 117,
    "type_counts": {
      "epithelial": 110,
      "inflammatory": 3,
      "spindle_shaped": 4,
      "miscellaneous": 0
    },
    "mean_area_px2": 138.53,
    "mean_perimeter_px": 49.67,
    "mean_eccentricity": 0.741,
    "mean_circularity": 0.688,
    "interpretation": "Nuclear pleomorphism detected with prominent epithelial density."
  },
  "gland_evidence": {
    "total_count": 2,
    "mean_area_pixels": 24432.0,
    "mean_perimeter_pixels": 1234.40,
    "mean_width_pixels": 143.0,
    "mean_height_pixels": 137.0,
    "mean_aspect_ratio": 1.333,
    "mean_circularity": 0.367,
    "interpretation": "Glandular architectural distortion observed."
  },
  "reference_comparison": {
    "label": "REFERENCE-BASED INSIGHT",
    "top_category": "adenocarcinoma",
    "top_similarity_percent": 98.4,
    "top_reference_id": "reference_001",
    "insight": "Morphological profile demonstrates 98.4% feature similarity with curated 'adenocarcinoma' reference.",
    "comparisons": [
      {
        "reference_id": "reference_001",
        "category": "adenocarcinoma",
        "normalized_distance": 0.016,
        "similarity_percent": 98.4,
        "key_concordant_features": ["nuclei_total", "nuclei_mean_area_px2"]
      }
    ]
  },
  "priority_regions": [
    {
      "region_id": "R_01",
      "index": 1,
      "x": 0,
      "y": 0,
      "width": 128,
      "height": 128,
      "prediction": "TUM",
      "confidence": 0.94,
      "tumor_probability": 0.95,
      "uncertainty_score": 0.12,
      "uncertainty_level": "LOW",
      "priority_score": 0.88,
      "priority_level": "HIGH",
      "priority_label": "AI-prioritized region",
      "nuclei_count": 52,
      "glands_count": 1,
      "agreement_level": "HIGH",
      "rationale": "High AI priority due to prominent tumor probability and nuclear atypia."
    }
  ],
  "visualizations": {
    "original": "/cases/CASE_2026_001/visualization/original",
    "glands": "/cases/CASE_2026_001/visualization/glands",
    "nuclei": "/cases/CASE_2026_001/visualization/nuclei",
    "regions": "/cases/CASE_2026_001/visualization/regions",
    "uncertainty": "/cases/CASE_2026_001/visualization/uncertainty",
    "top_regions": "/cases/CASE_2026_001/visualization/top_regions",
    "pseudo_3d": "/cases/CASE_2026_001/visualization/pseudo_3d"
  },
  "limitations": [
    "Research prototype for decision support; not an autonomous diagnostic device.",
    "Pathologist review recommended for all clinical correlations and staging."
  ],
  "explanation": {
    "text": "AI-assisted classification suggests TUM with 91.2% calibrated confidence.\nNuclear Analysis: 117 nuclei detected.\nGland Analysis: 2 glandular structures segmented.\nModel Agreement: HIGH.\nReference Match: 98.4% similarity to adenocarcinoma cohort.",
    "validated": true,
    "validation_errors": []
  }
}
```

---

### 2.3 Region Navigation & "Next Region"
* **Endpoint:** `GET /cases/{case_id}/regions/next?current_region_id={optional_current_id}`
* **Description:** Provides seamless viewport zoom and navigation to the next AI-prioritized region.
* **Response (200 OK):**
```json
{
  "case_id": "CASE_2026_001",
  "region": {
    "region_id": "R_01",
    "index": 1,
    "x": 0,
    "y": 0,
    "width": 128,
    "height": 128,
    "prediction": "TUM",
    "confidence": 0.94,
    "priority_score": 0.88,
    "priority_level": "HIGH"
  },
  "navigation": {
    "current_index": 1,
    "total_regions": 4,
    "has_more": true,
    "coordinates": {
      "x": 0,
      "y": 0,
      "width": 128,
      "height": 128,
      "center_x": 64,
      "center_y": 64
    }
  }
}
```

---

### 2.4 Visualization Serving
* **Endpoint:** `GET /cases/{case_id}/visualization/{type}`
* **Supported Types:**
  - `original`: Raw H&E image
  - `glands`: U-Net gland mask contour overlay
  - `nuclei`: HoVer-Net nuclear instance/type overlay
  - `regions`: AI-prioritized region bounding boxes & scores
  - `uncertainty`: Regional uncertainty heatmap
  - `top_regions`: Side-by-side prioritized crops
  - `pseudo_3d`: 3D-style morphological scatter topography
* **Response:** Binary image stream (`image/png`)

---

### 2.5 Pathologist Review & Clinical Notes
* **Submit Review:** `POST /cases/{case_id}/review`
  ```json
  {
    "action": "MARK_REVIEWED",
    "notes": "Adenocarcinoma features correlated with nuclear pleomorphism.",
    "pathologist_id": "Dr. Smith, MD"
  }
  ```
* **Add Note:** `POST /cases/{case_id}/notes`
  ```json
  {
    "note_text": "Request immunohistochemistry panel for CDX2.",
    "author": "Dr. Smith, MD"
  }
  ```
* **List Notes:** `GET /cases/{case_id}/notes`

---

## 3. Required Android Screens & UX Workflow

```
[1. Home Dashboard] ──> [2. New Analysis] ──> [3. Camera / Microscope Capture]
                                                             │
[6. Full Image Viewer] <── [5. Results Dashboard] <── [4. Processing Animation]
    ├── Pan / Pinch-Zoom
    ├── Layer Toggle (Original, Glands, Nuclei, Regions, Uncertainty, Pseudo-3D)
    └── [7. "Next Region →" Floating Action Button]
            │
            ▼
[8. Evidence Drawer & Uncertainty Indicator]
            │
            ▼
[9. Pathologist Review & Sign-Off Modal] (MARK REVIEWED / FLAG REGION / ADD NOTE)
            │
            ▼
[10. Case History & PDF Report Export]
```

### Critical Medical Safety Rules for Android UI
1. **Never use the label "Confirm Cancer"**. The primary sign-off button must be labeled **"MARK REVIEWED"** or **"COMPLETE REVIEW"**.
2. **Prioritized Regions** must be labeled as **"AI-Prioritized Region"** (never "Confirmed Malignancy").
3. **Abstention Banner**: If `uncertainty.review_required == true`, display an amber/red top banner:
   *"High Model Uncertainty. Pathologist review recommended."*
