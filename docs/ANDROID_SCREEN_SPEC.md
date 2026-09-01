# COLONPATH-AI — Android Mobile Screen Specifications

This document defines the 15 user interface screens for the Android application, detailing APIs used, UI components, user interactions, and error states.

---

## Screen Inventory & Workflow

```
[1. Splash / Health Screen] ──► [2. Home / Case List]
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [3. Upload & Capture]         [4. Case Dashboard]
                         │                             │
                         ▼                             ├──► [5. H&E Layer Viewer]
                 [Progress Dialog]                     ├──► [6. Prioritized Regions Grid]
                         │                             ├──► [7. Region Detail View]
                         └────────────────────────────►├──► [8. Nuclear Cytopathology]
                                                       ├──► [9. Gland Histomorphometry]
                                                       ├──► [10. Qdrant Reference Cases]
                                                       ├──► [11. Pathologist Copilot Chat]
                                                       ├──► [12. Next-Region Navigator]
                                                       ├──► [13. Review & Sign-Off Modal]
                                                       ├──► [14. Final Clinical Report]
                                                       └──► [15. Error / Offline States]
```

---

### Screen 1: Splash & Backend Health Check
- **API:** `GET /health`
- **UI Components:** App Logo, animated loading spinner, server connection status indicator (`"Connected to CUDA backend"`, `"Models Ready"`).
- **Navigation:** If backend is online, navigate to **Screen 2 (Home)**. If unreachable, display retry button and IP address configuration.

---

### Screen 2: Home & Recent Cases
- **API:** `GET /cases?limit=20`
- **UI Components:** Top AppBar with Search, Floating Action Button (`+ New Analysis`), RecyclerView list of case cards with prediction chip (`LYM`, `TUM`), uncertainty badge (`LOW`, `HIGH`), and review status (`REVIEWED`, `PENDING`).
- **User Action:** Tapping a case card opens **Screen 4 (Case Dashboard)**.

---

### Screen 3: Upload & Image Capture
- **API:** `POST /analyze` (Multipart Form)
- **UI Components:** Camera capture button, Gallery image picker, thumbnail preview, optional Case ID text field, and `"Start Multimodal Analysis"` primary action button.
- **Loading State:** Displays modal progress dialog with deterministic pipeline step tracker.

---

### Screen 4: Case Master Dashboard
- **API:** `GET /cases/{case_id}/result`
- **UI Components:**
  - Case metadata card (`Case ID`, `Resolution`, `Timestamp`).
  - Prediction banner: **AI-Predicted Tissue Class** (e.g. `LYM`) with calibrated confidence bar (`100.0%`).
  - Reliability strip: Uncertainty level (`LOW` in green / `HIGH` in red), OOD status (`IN_DISTRIBUTION`), Model Agreement (`LOW/HIGH`).
  - Quick action chips: *H&E Viewer*, *Regions*, *Nuclei*, *Glands*, *References*, *Copilot Chat*, *Sign Off*.

---

### Screen 5: 7-Layer Interactive Histopathology Viewer
- **API:** `GET /cases/{case_id}/visualization/{type}`
- **UI Components:** Pinch-to-zoom interactive canvas powered by `Coil` / `Glide`, Horizontal tab selector switching between:
  1. `1. Original H&E`
  2. `2. Gland Mask (U-Net)`
  3. `3. Nuclei (HoVer-Net)`
  4. `4. AI Regions`
  5. `5. Uncertainty Heatmap`
  6. `6. Top Crops`
  7. `7. Pseudo-3D Topography`
- **User Action:** Tap tab to instantly swap PNG overlays with smooth alpha cross-fade.

---

### Screen 6: Prioritized Spatial Regions Grid
- **API:** `GET /cases/{case_id}/regions`
- **UI Components:** 2-column Grid of spatial patches ($R_{01}-R_{04}$) displaying bounding box thumbnail, priority score badge, cell count, and triage rationale.
- **User Action:** Tap patch to open **Screen 7 (Region Detail)**.

---

### Screen 7: Region Detail & Local Morphology
- **API:** `GET /cases/{case_id}/regions/{region_id}`
- **UI Components:** Zoomed view of patch $(x, y, w, h)$, local cell density, local gland count, agreement status, and `"Flag Region"` button.

---

### Screen 8: Nuclear Cytopathology Breakdown
- **API:** `GET /cases/{case_id}/result` (`nuclear_evidence`)
- **UI Components:** Total nuclei count (117), Mean nuclear area (138.5 px²), circularity gauge (0.69), Pie chart of cell sub-types (3 Epithelial, 0 Inflammatory, 110 Spindle, 4 Misc).

---

### Screen 9: Gland Histomorphometry Breakdown
- **API:** `GET /cases/{case_id}/result` (`gland_evidence`)
- **UI Components:** Total glands count (2), Mean gland area (24,432 px²), Mean circularity (0.37), Aspect ratio (1.33), Architectural distortion indicator.

---

### Screen 10: Qdrant Reference Cohorts & RAG Matches
- **API:** `GET /cases/{case_id}/result` (`reference_comparison`)
- **UI Components:** Top matched reference cohort (`adenocarcinoma`), similarity percent (`100.0%`), concordant feature chips, clinical insight text card.

---

### Screen 11: Pathologist Copilot Chat (Google MedGemma)
- **API:** `POST /copilot/ask`
- **UI Components:** Conversational chat interface with quick suggestion chips (*"Why was R_03 prioritized?"*, *"What nuclear features were detected?"*, *"What is the model uncertainty?"*), message bubbles with anti-hallucination verified checkmark.

---

### Screen 12: Next-Region Navigator
- **API:** `GET /cases/{case_id}/regions/next`
- **UI Components:** Bottom floating triage bar: `"Next Priority: R_03 (Priority: 0.12)"` with `"Review Next"` button automatically panning viewport to $(x=0, y=128)$.

---

### Screen 13: Pathologist Review & Sign-Off
- **API:** `POST /cases/{case_id}/review` & `POST /cases/{case_id}/feedback`
- **UI Components:** Sign-off dialog with action choices (*"Mark Reviewed"*, *"Add Note"*, *"Flag Region"*), Ground truth feedback radio buttons (*CORRECT*, *INCORRECT*, *UNCERTAIN*), notes field, and submit button.

---

### Screen 14: Final Structured Clinical Report
- **API:** `GET /cases/{case_id}/report`
- **UI Components:** Executive summary card, bulleted evidence breakdown, mandatory clinical limitations disclaimer, and PDF export action.

---

### Screen 15: Error & Warning Handling
- **UI Components:** Full-screen error views for `BLURRED_IMAGE`, `CORRUPTED_FILE`, `OOD_DETECTED`, and `NETWORK_UNREACHABLE` with actionable recovery instructions.
