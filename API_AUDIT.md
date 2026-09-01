# API_AUDIT.md — FastAPI REST Backend & Route Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Backend & API Integration Team  

---

## 1. REST Endpoint Inventory (17 Active Routes on Port 8080)

| HTTP Method | Route URL | Purpose / Operation | Verified Return Type |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Server & CUDA hardware health | `{"status": "healthy", "device": "cuda"}` |
| `POST` | `/analyze` | Multipart image upload & 12-stage analysis | `CaseResultResponse` JSON |
| `GET` | `/cases` | Lists all historical cases | `List[CaseSummary]` JSON |
| `GET` | `/cases/{id}/result` | Retrieves master diagnostic results | `CaseResultResponse` JSON |
| `GET` | `/cases/{id}/evidence` | Retrieves isolated factual measurements | `EvidenceData` JSON |
| `GET` | `/cases/{id}/report` | Retrieves MedGemma clinical narrative | `ReportData` JSON |
| `GET` | `/cases/{id}/regions` | Retrieves prioritized bounding boxes | `List[PriorityRegion]` JSON |
| `GET` | `/cases/{id}/regions/next` | Returns highest-priority unreviewed patch | `PriorityRegion` JSON |
| `GET` | `/cases/{id}/visualization/original` | Raw H&E microscope image | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/glands` | U-Net segmented green gland boundaries | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/nuclei` | HoVer-Net color-coded nuclear phenotypes | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/regions` | Spatial prioritized bounding boxes | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/uncertainty` | Normalized Shannon entropy heatmap | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/top_regions` | Zoomed collage of high-priority patches | `image/png` (Binary stream) |
| `GET` | `/cases/{id}/visualization/pseudo_3d` | Optical density surface topography | `image/png` (Binary stream) |
| `POST` | `/copilot/ask` | Interactive Q&A with MedGemma Copilot | `CopilotAnswerResponse` JSON |
| `POST` | `/cases/{id}/feedback` | Records pathologist review & ground truth | `{"status": "success"}` JSON |

---

## 2. API Contract & Security
* **No Leaked Internal Paths:** All responses return sanitized relative routes or IDs.
* **CORS:** Configured for mobile app integration (`allow_origins=["*"]`).
* **OpenAPI 3.1 Schema:** Exported in [`docs/openapi.json`](file:///c:/Users/kthir/OneDrive/Desktop/colon_model/docs/openapi.json).
