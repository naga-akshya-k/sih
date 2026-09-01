# COLONPATH-AI — Android API Request & Response Examples

This document provides realistic JSON payloads for all API interactions. All examples represent verified backend outputs.

---

## 1. Health Check

### Request:
```http
GET /health HTTP/1.1
Host: 10.0.2.2:8080
```

### Response (`200 OK`):
```json
{
  "status": "healthy",
  "service": "COLONPATH-AI Backend",
  "version": "1.0.0",
  "device": "cuda:0",
  "models_ready": true
}
```

---

## 2. Image Upload & Analysis

### Request:
```http
POST /analyze HTTP/1.1
Host: 10.0.2.2:8080
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="image"; filename="biopsy_tile_01.png"
Content-Type: image/png

<binary data>
------WebKitFormBoundary7MA4YWxkTrZu0gW
Content-Disposition: form-data; name="case_id"

CASE_DEMO_00000
------WebKitFormBoundary7MA4YWxkTrZu0gW--
```

### Response (`200 OK`):
```json
{
  "case_id": "CASE_DEMO_00000",
  "timestamp": "2026-09-01T08:30:24.448Z",
  "status": "completed",
  "image_quality": {
    "passed": true,
    "resolution": "256x256",
    "blur_laplacian_variance": 76.57,
    "blur_status": "ACCEPTABLE",
    "mean_brightness": 174.56,
    "contrast_std": 49.33
  },
  "digepath": {
    "model_name": "Digepath",
    "architecture": "ViT-L/16",
    "embedding_dimension": 1024,
    "device": "cuda",
    "status": "active"
  },
  "prediction": {
    "class": "LYM",
    "confidence": 1.0,
    "calibrated_confidence": 1.0,
    "tumor_probability": 0.0,
    "binary_class": "NON-TUM",
    "multiclass_probabilities": {
      "ADI": 0.0,
      "BACK": 0.0,
      "DEB": 0.0,
      "LYM": 1.0,
      "MUC": 0.0,
      "MUS": 0.0,
      "NORM": 0.0,
      "STR": 0.0,
      "TUM": 0.0
    }
  },
  "uncertainty": {
    "score": 0.0,
    "level": "LOW",
    "entropy": 0.0,
    "normalized_entropy": 0.0,
    "ood_score": 0.0,
    "ood_status": "IN_DISTRIBUTION",
    "is_ood": false,
    "review_required": false,
    "message": "AI-assisted classification ready for review."
  },
  "model_agreement": {
    "level": "LOW",
    "score": 0.25,
    "concordant_sources": ["visual_and_binary"],
    "discordant_sources": ["nuclear_morphology", "gland_architecture"],
    "summary": "Low agreement / Evidence Conflict: Discrepancy detected between visual classification (LYM) and morphological measurements."
  },
  "nuclear_evidence": {
    "total_count": 117,
    "type_counts": {
      "epithelial": 3,
      "inflammatory": 0,
      "spindle_shaped": 110,
      "miscellaneous": 4
    },
    "mean_area_px2": 138.53,
    "mean_perimeter_px": 44.59,
    "mean_eccentricity": 0.741,
    "mean_circularity": 0.69,
    "interpretation": "Elevated nuclear density with spindle-shaped morphology."
  },
  "gland_evidence": {
    "total_count": 2,
    "mean_area_pixels": 24432.0,
    "mean_perimeter_pixels": 820.5,
    "mean_aspect_ratio": 1.33,
    "mean_circularity": 0.367,
    "interpretation": "Glandular structures exhibit irregular contours (circularity < 0.50)."
  },
  "reference_comparison": {
    "label": "REFERENCE-BASED INSIGHT",
    "top_category": "adenocarcinoma",
    "top_similarity_percent": 100.0,
    "top_reference_id": "reference_001",
    "insight": "Morphological profile demonstrates 100.0% feature similarity with curated 'adenocarcinoma' reference (reference_001).",
    "comparisons": [
      {
        "reference_id": "reference_001",
        "category": "adenocarcinoma",
        "normalized_distance": 0.0,
        "similarity_percent": 100.0,
        "key_concordant_features": ["nuclei_total", "glands_total", "nuclei_mean_area_px2"]
      }
    ]
  },
  "priority_regions": [
    {
      "region_id": "R_03",
      "index": 2,
      "x": 0,
      "y": 128,
      "width": 128,
      "height": 128,
      "prediction": "LYM",
      "confidence": 1.0,
      "tumor_probability": 0.0,
      "uncertainty_score": 0.0,
      "uncertainty_level": "LOW",
      "priority_score": 0.12,
      "priority_level": "MEDIUM",
      "priority_label": "MEDIUM PRIORITY",
      "nuclei_count": 37,
      "glands_count": 0,
      "agreement_level": "HIGH",
      "rationale": "Moderate cell density requires review."
    }
  ],
  "visualizations": {
    "original": "outputs/visualizations/CASE_DEMO_00000/original.png",
    "glands": "outputs/visualizations/CASE_DEMO_00000/glands.png",
    "nuclei": "outputs/visualizations/CASE_DEMO_00000/nuclei.png",
    "regions": "outputs/visualizations/CASE_DEMO_00000/regions.png",
    "uncertainty": "outputs/visualizations/CASE_DEMO_00000/uncertainty.png",
    "top_regions": "outputs/visualizations/CASE_DEMO_00000/top_regions.png",
    "pseudo_3d": "outputs/visualizations/CASE_DEMO_00000/pseudo_3d.png"
  },
  "limitations": [
    "Research prototype for decision support; not an autonomous diagnostic device.",
    "Pathologist review recommended for all clinical correlations and staging.",
    "Visual and morphological features are AI-derived computational estimates."
  ]
}
```

---

## 3. Pathologist Copilot Q&A

### Request:
```http
POST /copilot/ask HTTP/1.1
Host: 10.0.2.2:8080
Content-Type: application/json

{
  "case_id": "CASE_DEMO_00000",
  "question": "What gland features were segmented by U-Net?"
}
```

### Response (`200 OK`):
```json
{
  "case_id": "CASE_DEMO_00000",
  "question": "What gland features were segmented by U-Net?",
  "selected_region_id": null,
  "answer": "Glandular Histomorphometry: Segmented 2 glandular structures by U-Net with mean area 24,432 px², circularity 0.37, and aspect ratio 1.33, indicating architectural distortion.",
  "model": "google/medgemma-1.5-4b-it",
  "validated": true,
  "validation_errors": []
}
```

---

## 4. Next Region Triage

### Request:
```http
GET /cases/CASE_DEMO_00000/regions/next HTTP/1.1
Host: 10.0.2.2:8080
```

### Response (`200 OK`):
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
    "prediction": "LYM",
    "confidence": 1.0,
    "tumor_probability": 0.0,
    "uncertainty_score": 0.0,
    "uncertainty_level": "LOW",
    "priority_score": 0.12,
    "priority_level": "MEDIUM",
    "priority_label": "MEDIUM PRIORITY",
    "nuclei_count": 37,
    "glands_count": 0,
    "agreement_level": "HIGH",
    "rationale": "Moderate cell density requires review."
  },
  "remaining_unreviewed_count": 3
}
```

---

## 5. Pathologist Review & Feedback

### Request:
```http
POST /cases/CASE_DEMO_00000/feedback HTTP/1.1
Host: 10.0.2.2:8080
Content-Type: application/json

{
  "feedback": "CORRECT",
  "notes": "Verified lymphoid infiltration with focal glandular distortion.",
  "pathologist_id": "Dr. Sarah Jenkins, MD"
}
```

### Response (`200 OK`):
```json
{
  "status": "success",
  "case_id": "CASE_DEMO_00000",
  "feedback": "CORRECT",
  "recorded": true
}
```
