# COLONPATH-AI — Android Developer API Contract

This document provides the definitive, production-verified REST API specification for the Android mobile application connecting to the **COLONPATH-AI** backend server.

---

## 🌐 Base URL Configuration
- **Local Emulator:** `http://10.0.2.2:8080` (Android Studio Emulator mapping to host `127.0.0.1:8080`)
- **Physical Device:** `http://<YOUR_LOCAL_IP>:8080` (e.g. `http://192.168.1.100:8080`)
- **Swagger Documentation:** `http://127.0.0.1:8080/docs`
- **OpenAPI 3.1 Spec:** [`docs/openapi.json`](file:///c:/Users/kthir/OneDrive/Desktop/colon_model/docs/openapi.json)

---

## 📡 Endpoint Inventory

### 1. System Health Check
- **METHOD:** `GET`
- **URL:** `/health`
- **Purpose:** Verifies backend connectivity, model loading status, and CUDA acceleration.
- **Request:** None
- **Success Status:** `200 OK`
- **Example Response:**
```json
{
  "status": "healthy",
  "service": "COLONPATH-AI Backend",
  "version": "1.0.0",
  "device": "cuda:0",
  "models_ready": true
}
```
- **Android Usage:** Call on app startup or splash screen to verify backend health.

---

### 2. Full H&E Multimodal Analysis
- **METHOD:** `POST`
- **URL:** `/analyze`
- **Purpose:** Uploads an H&E slide patch, executes the 12-stage multimodal pipeline, and returns complete structured results.
- **Request Format:** `multipart/form-data`
  - `image`: Image binary (`image/png`, `image/jpeg`, `image/bmp`, `image/tiff`)
  - `case_id`: Optional string (e.g., `"CASE_PATIENT_101"`). If omitted, generated from filename.
- **Success Status:** `200 OK`
- **Response Schema:** `CaseResultResponse` (See [Case Result Structure](#case-result-structure))
- **Error Statuses:** `400 Bad Request` (no file), `500 Internal Server Error`
- **Android Usage:** Triggered when user selects a biopsy image from gallery or captures via camera.

---

### 3. Retrieve Case Result
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/result` (or alias `/case/{case_id}`)
- **Purpose:** Retrieves the cached deterministic case analysis result without re-running inference.
- **Path Parameter:** `case_id` (String)
- **Success Status:** `200 OK`
- **Error Status:** `404 Not Found`
- **Android Usage:** Populates the Case Dashboard screen when opening a saved case.

---

### 4. Retrieve Case Factual Evidence Object
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/evidence` (or alias `/case/{case_id}/evidence`)
- **Purpose:** Retrieves isolated, factual computational metrics (`evidence.json`) for raw data inspection.
- **Success Status:** `200 OK`
- **Response Example:**
```json
{
  "case_id": "CASE_DEMO_00000",
  "timestamp": "2026-09-01T08:30:22.000Z",
  "prediction_class": "LYM",
  "prediction_confidence": 1.0,
  "calibrated_confidence": 1.0,
  "tumor_probability": 0.0,
  "uncertainty_score": 0.0,
  "uncertainty_level": "LOW",
  "ood_score": 0.0,
  "ood_status": "IN_DISTRIBUTION",
  "agreement_level": "LOW",
  "nuclear_total_count": 117,
  "nuclear_mean_area_px2": 138.53,
  "gland_total_count": 2,
  "gland_mean_circularity": 0.367,
  "reference_top_category": "adenocarcinoma",
  "reference_top_similarity_percent": 100.0,
  "priority_regions_count": 4
}
```

---

### 5. Retrieve MedGemma Clinical Report
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/report` (or alias `/case/{case_id}/report`)
- **Purpose:** Retrieves structured medical VLM narrative explanation and clinical limitations.
- **Success Status:** `200 OK`
- **Response Schema:**
```json
{
  "case_id": "CASE_DEMO_00000",
  "explanation": {
    "summary": "AI-assisted multimodal analysis suggests tissue class LYM with 100.0% calibrated confidence. 117 nuclei and 2 glands segmented.",
    "visual_evidence": ["Digepath ViT-L/16 1024-d visual feature representation"],
    "nuclear_evidence": ["117 total nuclei segmented", "Mean nuclear area: 138.5 px²", "Sub-types: 3 Epithelial, 0 Inflammatory, 110 Spindle, 4 Misc"],
    "gland_evidence": ["2 glandular structures segmented", "Mean gland circularity: 0.37"],
    "prediction_evidence": ["Multimodal late-fusion predicted class LYM (100.0%)"],
    "uncertainty_explanation": "Model uncertainty is LOW (Entropy: 0.00).",
    "model_agreement": "LOW (Evidence conflict between visual embedding and morphology)",
    "reference_evidence": ["100.0% similarity to adenocarcinoma cohort"],
    "limitations": ["Research prototype for decision support; not an autonomous diagnostic device."],
    "review_recommendation": "Pathologist review recommended for all clinical correlations and staging."
  },
  "limitations": ["Research prototype; not autonomous diagnosis."],
  "status": "completed"
}
```

---

### 6. Retrieve Prioritized Regions
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/regions` (or alias `/case/{case_id}/regions`)
- **Purpose:** Returns list of all coordinate-indexed spatial patches ($R_{01}-R_{04}$) with priority scores and cell counts.
- **Success Status:** `200 OK`
- **Response Item:**
```json
[
  {
    "region_id": "R_01",
    "index": 0,
    "x": 0,
    "y": 0,
    "width": 128,
    "height": 128,
    "prediction": "LYM",
    "confidence": 1.0,
    "tumor_probability": 0.0,
    "uncertainty_score": 0.0,
    "uncertainty_level": "LOW",
    "priority_score": 0.0,
    "priority_level": "LOW",
    "priority_label": "LOW PRIORITY",
    "nuclei_count": 31,
    "glands_count": 0,
    "agreement_level": "HIGH",
    "rationale": "High confidence LYM prediction with low entropy."
  }
]
```

---

### 7. Next Region Navigation
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/regions/next` (or alias `/case/{case_id}/next-region`)
- **Query Parameter:** `current_region_id` (Optional string, e.g. `"R_01"`)
- **Purpose:** Guides the pathologist to the highest-priority unreviewed region.
- **Response Example:**
```json
{
  "case_id": "CASE_DEMO_00000",
  "has_next": true,
  "next_region": {
    "region_id": "R_03",
    "index": 2,
    "x": 0,
    "y": 128,
    "width": 128,
    "height": 128,
    "priority_score": 0.12,
    "priority_level": "MEDIUM",
    "nuclei_count": 37,
    "glands_count": 0
  },
  "remaining_unreviewed_count": 3
}
```

---

### 8. Image & Visual Overlay Delivery
- **METHOD:** `GET`
- **URL:** `/cases/{case_id}/visualization/{vis_type}` (or alias `/case/{case_id}/visualization/{vis_type}`)
- **Purpose:** Directly streams authentic PNG images for high-speed rendering in Android image views (`Coil` / `Glide`).
- **Path Parameter `vis_type` options:**
  - `original`: Raw H&E slide image
  - `glands`: U-Net gland segmentation overlay (green contours)
  - `nuclei`: HoVer-Net nuclear instance overlay (multi-colored cell types)
  - `regions`: AI-prioritized bounding boxes ($R_{01}-R_{04}$)
  - `uncertainty`: Normalized Shannon entropy heatmap
  - `top_regions`: High-priority cropped patches collage
  - `pseudo_3d`: 3D optical topography surface
- **Response:** `image/png` binary stream.

---

### 9. Interactive Pathologist Copilot (Google MedGemma)
- **METHOD:** `POST`
- **URL:** `/copilot/ask`
- **Purpose:** Interactive conversational Q&A answering clinical inquiries about the slide.
- **Request Body (`application/json`):**
```json
{
  "case_id": "CASE_DEMO_00000",
  "question": "What nuclear abnormalities and cell types were detected?",
  "region_id": "R_01"
}
```
- **Response Schema:**
```json
{
  "case_id": "CASE_DEMO_00000",
  "question": "What nuclear abnormalities and cell types were detected?",
  "selected_region_id": "R_01",
  "answer": "Nuclear Cytopathology: 117 total nuclei detected with mean area 138.5 px², circularity 0.69, and eccentricity 0.741. Distribution: 3 Epithelial, 0 Inflammatory, 110 Spindle-shaped, 4 Misc.",
  "model": "google/medgemma-1.5-4b-it",
  "validated": true,
  "validation_errors": []
}
```

---

### 10. Pathologist Sign-Off & Review
- **METHOD:** `POST`
- **URL:** `/cases/{case_id}/review` (or alias `/case/{case_id}/review`)
- **Request Body:**
```json
{
  "action": "MARK_REVIEWED",
  "notes": "Slide reviewed, concordant with clinical assessment.",
  "pathologist_id": "Dr. Smith"
}
```
- **Allowed Actions:** `MARK_REVIEWED`, `FLAG_REGION`, `ADD_NOTE`, `REQUEST_REANALYSIS`
- **Response:** `{"status": "success", "case_id": "CASE_DEMO_00000", "action": "MARK_REVIEWED"}`

---

### 11. Pathologist Feedback
- **METHOD:** `POST`
- **URL:** `/cases/{case_id}/feedback` (or alias `/case/{case_id}/feedback`)
- **Request Body:**
```json
{
  "feedback": "CORRECT",
  "notes": "Verified lymphoid infiltration.",
  "pathologist_id": "Dr. Smith"
}
```
- **Allowed Feedback:** `CORRECT`, `INCORRECT`, `UNCERTAIN`, `REVIEW_REQUIRED`
- **Response:** `{"status": "success", "case_id": "CASE_DEMO_00000", "feedback": "CORRECT", "recorded": true}`
