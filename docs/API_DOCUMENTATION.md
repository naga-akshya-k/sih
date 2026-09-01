# COLONPATH-AI — Comprehensive Backend API Documentation

The COLONPATH-AI REST API exposes all 12 stages of the multimodal colorectal histopathology analysis platform, serving both the web diagnostic dashboard and the Android mobile application.

---

## 🚀 Server Execution
```powershell
python main.py --server --port 8080 --host 0.0.0.0
```
- **Base URL:** `http://127.0.0.1:8080`
- **Interactive Swagger UI:** `http://127.0.0.1:8080/docs`
- **Interactive ReDoc UI:** `http://127.0.0.1:8080/redoc`
- **OpenAPI 3.1 Spec:** [`docs/openapi.json`](file:///c:/Users/kthir/OneDrive/Desktop/colon_model/docs/openapi.json)

---

## 📑 Complete API Route Reference

| Method | Route | Description | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server & CUDA model health | `HealthResponse` |
| `POST` | `/analyze` | Run full 12-stage multimodal analysis | `CaseResultResponse` |
| `GET` | `/cases` | List all historical analyzed cases | `List[CaseSummaryItem]` |
| `GET` | `/cases/{case_id}/result` | Master case analysis result | `CaseResultResponse` |
| `GET` | `/cases/{case_id}/evidence` | Isolated factual computational evidence | `Dict[str, Any]` |
| `GET` | `/cases/{case_id}/report` | Structured MedGemma clinical narrative | `Dict[str, Any]` |
| `GET` | `/cases/{case_id}/regions` | List prioritized spatial patches ($R_{01}-R_{04}$) | `List[RegionDetailSchema]` |
| `GET` | `/cases/{case_id}/regions/next` | Next highest-priority unreviewed patch | `NextRegionResponse` |
| `GET` | `/cases/{case_id}/visualization/{type}` | Stream authentic PNG visual layers | Binary image (`image/png`) |
| `POST` | `/cases/{case_id}/review` | Pathologist review action | `Dict[str, Any]` |
| `POST` | `/cases/{case_id}/feedback` | Ground-truth clinical feedback | `Dict[str, Any]` |
| `POST` | `/cases/{case_id}/notes` | Add clinical notes | `Dict[str, Any]` |
| `GET` | `/cases/{case_id}/notes` | List clinical notes | `List[Dict[str, Any]]` |
| `POST` | `/copilot/ask` | Pathologist Copilot Q&A (Google MedGemma) | `CopilotAnswerResponse` |
| `GET` | `/case/{case_id}/next-region` | *Android Alias:* Next-region triage | `NextRegionResponse` |
| `GET` | `/case/{case_id}/regions` | *Android Alias:* Regions list | `List[RegionDetailSchema]` |
| `GET` | `/` & `/viewer` | Interactive Web Layer Viewer & Dashboard | HTML / Web Application |

---

## 🎨 7 Visual Overlay Types
Streaming endpoint: `GET /cases/{case_id}/visualization/{vis_type}`:
1. `original`: Unprocessed H&E slide tile.
2. `glands`: U-Net segmented glandular contours (green boundaries).
3. `nuclei`: HoVer-Net segmented nuclei color-coded by cell sub-type.
4. `regions`: Prioritized spatial bounding boxes with priority labels ($R_{01}-R_{04}$).
5. `uncertainty`: Normalized Shannon entropy heatmap.
6. `top_regions`: Zoomed montage of high-priority suspicious crops.
7. `pseudo_3d`: 3D optical topography surface landscape.
