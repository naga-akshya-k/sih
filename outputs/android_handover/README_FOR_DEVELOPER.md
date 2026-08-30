# COLONPATH-AI — Android Developer Quickstart & Integration Guide

Welcome! This package contains everything you need to build the **COLONPATH-AI** Android mobile application.

---

## 1. Quick Connection Setup

The backend AI server runs on **FastAPI**.

### Base URLs:
* **Android Studio Emulator:** `http://10.0.2.2:8080`
* **Physical Android Device (Same Wi-Fi):** `http://<HOST_COMPUTER_IP>:8080`
* **Swagger Interactive Docs:** `http://127.0.0.1:8080/docs`

---

## 2. Recommended Dependencies (`build.gradle.kts`)

```kotlin
dependencies {
    // Retrofit & Networking
    implementation("com.squareup.retrofit2:retrofit:2.9.0")
    implementation("com.squareup.retrofit2:converter-gson:2.9.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.12.0")

    // Coil for Histopathology Image Loading
    implementation("io.coil-kt:coil-compose:2.6.0")

    // Material 3 & Jetpack Compose
    implementation("androidx.compose.material3:material3:1.2.1")
    implementation("androidx.compose.material:material-icons-extended:1.6.5")

    // CameraX (Optional, for microscope eyepiece capture)
    implementation("androidx.camera:camera-camera2:1.3.2")
    implementation("androidx.camera:camera-lifecycle:1.3.2")
    implementation("androidx.camera:camera-view:1.3.2")
}
```

---

## 3. Ready-to-Paste Kotlin Data Classes (`ColonPathModels.kt`)

```kotlin
package com.colonpath.data.model

import com.google.gson.annotations.SerializedName

data class CaseResultResponse(
    @SerializedName("case_id") val caseId: String,
    @SerializedName("timestamp") val timestamp: String,
    @SerializedName("status") val status: String,
    @SerializedName("prediction") val prediction: PredictionData,
    @SerializedName("uncertainty") val uncertainty: UncertaintyData,
    @SerializedName("model_agreement") val modelAgreement: ModelAgreementData,
    @SerializedName("nuclear_evidence") val nuclearEvidence: NuclearEvidenceData,
    @SerializedName("gland_evidence") val glandEvidence: GlandEvidenceData,
    @SerializedName("reference_comparison") val referenceComparison: ReferenceData,
    @SerializedName("priority_regions") val priorityRegions: List<RegionItemData>,
    @SerializedName("visualizations") val visualizations: Map<String, String>,
    @SerializedName("explanation") val explanation: ExplanationData?
)

data class PredictionData(
    @SerializedName("class") val tissueClass: String, // e.g. "TUM", "LYM", "NORM"
    @SerializedName("confidence") val confidence: Float,
    @SerializedName("calibrated_confidence") val calibratedConfidence: Float,
    @SerializedName("tumor_probability") val tumorProbability: Float
)

data class UncertaintyData(
    @SerializedName("score") val score: Float,
    @SerializedName("level") val level: String, // "LOW", "MEDIUM", "HIGH"
    @SerializedName("review_required") val reviewRequired: Boolean,
    @SerializedName("message") val message: String
)

data class ModelAgreementData(
    @SerializedName("level") val level: String, // "HIGH", "MEDIUM", "LOW"
    @SerializedName("summary") val summary: String
)

data class NuclearEvidenceData(
    @SerializedName("total_count") val totalCount: Int,
    @SerializedName("mean_area_px2") val meanAreaPx2: Float
)

data class GlandEvidenceData(
    @SerializedName("total_count") val totalCount: Int,
    @SerializedName("mean_circularity") val meanCircularity: Float
)

data class ReferenceData(
    @SerializedName("top_category") val topCategory: String,
    @SerializedName("top_similarity_percent") val similarityPercent: Float,
    @SerializedName("insight") val insight: String
)

data class RegionItemData(
    @SerializedName("region_id") val regionId: String,
    @SerializedName("x") val x: Int,
    @SerializedName("y") val y: Int,
    @SerializedName("width") val width: Int,
    @SerializedName("height") val height: Int,
    @SerializedName("prediction") val prediction: String,
    @SerializedName("priority_score") val priorityScore: Float,
    @SerializedName("priority_level") val priorityLevel: String // "HIGH", "MEDIUM", "LOW"
)

data class ExplanationData(
    @SerializedName("text") val text: String,
    @SerializedName("validated") val validated: Boolean
)

data class NextRegionResponse(
    @SerializedName("case_id") val caseId: String,
    @SerializedName("region") val region: RegionItemData,
    @SerializedName("navigation") val navigation: Map<String, Any>
)

data class ReviewRequest(
    @SerializedName("action") val action: String, // "MARK_REVIEWED", "FLAG_REGION", "ADD_NOTE"
    @SerializedName("notes") val notes: String = "",
    @SerializedName("pathologist_id") val pathologistId: String = "Dr. Pathologist"
)

data class NoteRequest(
    @SerializedName("note_text") val noteText: String,
    @SerializedName("author") val author: String = "Pathologist"
)
```

---

## 4. Retrofit API Service (`ColonPathApiService.kt`)

```kotlin
package com.colonpath.data.api

import com.colonpath.data.model.*
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.*

interface ColonPathApiService {

    @GET("/health")
    suspend fun healthCheck(): Map<String, Any>

    @Multipart
    @POST("/analyze")
    suspend fun analyzeImage(
        @Part image: MultipartBody.Part,
        @Part("case_id") caseId: RequestBody? = null
    ): CaseResultResponse

    @GET("/cases/{caseId}/result")
    suspend fun getCaseResult(@Path("caseId") caseId: String): CaseResultResponse

    @GET("/cases/{caseId}/regions/next")
    suspend fun getNextRegion(
        @Path("caseId") caseId: String,
        @Query("current_region_id") currentRegionId: String? = null
    ): NextRegionResponse

    @POST("/cases/{caseId}/review")
    suspend fun submitReview(
        @Path("caseId") caseId: String,
        @Body review: ReviewRequest
    ): Map<String, Any>

    @POST("/cases/{caseId}/notes")
    suspend fun addNote(
        @Path("caseId") caseId: String,
        @Body note: NoteRequest
    ): Map<String, Any>
}
```

---

## 5. UI/UX Flow & Screen Recommendations

1. **Upload / Capture Screen**: File picker or CameraX capture for microscope slides.
2. **Analysis Progress Screen**: Shows loading spinner while backend runs foundation models.
3. **Results & Layer Viewer**:
   - High-resolution viewer with pinch-to-zoom and pan.
   - **Layer Selector Tabs**:
     - *Original H&E*: `/cases/{id}/visualization/original`
     - *Gland Mask (U-Net)*: `/cases/{id}/visualization/glands`
     - *Nuclei (HoVer-Net)*: `/cases/{id}/visualization/nuclei`
     - *AI Prioritized Regions*: `/cases/{id}/visualization/regions`
     - *Uncertainty Heatmap*: `/cases/{id}/visualization/uncertainty`
     - *Pseudo-3D Topography*: `/cases/{id}/visualization/pseudo_3d`
4. **"Next Region ➔" Floating Action Button**: Calls `/cases/{id}/regions/next` and auto-pans/zooms to the highest priority bounding box.
5. **Pathologist Action Bar**:
   - Button 1: **"MARK REVIEWED"** (`action = "MARK_REVIEWED"`)
   - Button 2: **"FLAG REGION"** (`action = "FLAG_REGION"`)
   - Button 3: **"ADD NOTE"** (Opens text modal)

---

## 6. Critical Medical Safety Rules

* ⚠️ **Do NOT label buttons as "Confirm Cancer"**. Must use **"MARK REVIEWED"** or **"COMPLETE REVIEW"**.
* ⚠️ **Do NOT label boxes as "Malignant Area"**. Must use **"AI-Prioritized Region"**.
* ⚠️ If `uncertainty.review_required == true`, display an amber/red banner: *"High model uncertainty. Pathologist review recommended."*
