# COLONPATH-AI — Android Kotlin Data Models & Retrofit Interface

This document contains copy-paste ready Kotlin data classes (using `kotlinx.serialization` or `Gson`) and the complete `Retrofit2` API service interface for the Android application.

---

## 📱 Kotlin Data Classes

```kotlin
package com.colonpath.ai.data.model

import com.google.gson.annotations.SerializedName

// 1. Health Response
data class HealthResponse(
    val status: String,
    val service: String,
    val version: String,
    val device: String,
    @SerializedName("models_ready") val modelsReady: Boolean
)

// 2. Case Summary Item (for Case List Screen)
data class CaseSummaryItem(
    @SerializedName("case_id") val caseId: String,
    val timestamp: String,
    val status: String,
    val prediction: String,
    val confidence: Float,
    @SerializedName("uncertainty_level") val uncertaintyLevel: String,
    @SerializedName("reviewed_by") val reviewedBy: String?
)

// 3. Complete Master Case Result Response
data class CaseResultResponse(
    @SerializedName("case_id") val caseId: String,
    val timestamp: String,
    val status: String,
    @SerializedName("image_quality") val imageQuality: ImageQualityData,
    val digepath: DigepathMetadata,
    val prediction: PredictionData,
    val uncertainty: UncertaintyData,
    @SerializedName("model_agreement") val modelAgreement: ModelAgreementData,
    @SerializedName("nuclear_evidence") val nuclearEvidence: NuclearEvidenceData,
    @SerializedName("gland_evidence") val glandEvidence: GlandEvidenceData,
    @SerializedName("reference_comparison") val referenceComparison: ReferenceComparisonData,
    @SerializedName("priority_regions") val priorityRegions: List<RegionDetailItem>,
    val visualizations: Map<String, String>,
    val limitations: List<String>,
    val explanation: MedGemmaExplanation?
)

// 4. Sub-Models
data class ImageQualityData(
    val passed: Boolean,
    val resolution: String,
    @SerializedName("blur_laplacian_variance") val blurVariance: Float,
    @SerializedName("blur_status") val blurStatus: String,
    @SerializedName("mean_brightness") val meanBrightness: Float,
    @SerializedName("contrast_std") val contrastStd: Float
)

data class DigepathMetadata(
    @SerializedName("model_name") val modelName: String,
    val architecture: String,
    @SerializedName("embedding_dimension") val embeddingDimension: Int,
    val device: String,
    val status: String
)

data class PredictionData(
    @SerializedName("class") val predictedClass: String,
    val confidence: Float,
    @SerializedName("calibrated_confidence") val calibratedConfidence: Float,
    @SerializedName("tumor_probability") val tumorProbability: Float,
    @SerializedName("binary_class") val binaryClass: String,
    @SerializedName("multiclass_probabilities") val probabilities: Map<String, Float>
)

data class UncertaintyData(
    val score: Float,
    val level: String, // "LOW", "MEDIUM", "HIGH"
    val entropy: Float,
    @SerializedName("normalized_entropy") val normalizedEntropy: Float,
    @SerializedName("ood_score") val oodScore: Float,
    @SerializedName("ood_status") val oodStatus: String, // "IN_DISTRIBUTION", "OOD_DETECTED"
    @SerializedName("is_ood") val isOod: Boolean,
    @SerializedName("review_required") val reviewRequired: Boolean,
    val message: String
)

data class ModelAgreementData(
    val level: String, // "HIGH", "MEDIUM", "LOW"
    val score: Float,
    @SerializedName("concordant_sources") val concordantSources: List<String>,
    @SerializedName("discordant_sources") val discordantSources: List<String>,
    val summary: String
)

data class NuclearEvidenceData(
    @SerializedName("total_count") val totalCount: Int,
    @SerializedName("type_counts") val typeCounts: Map<String, Int>,
    @SerializedName("mean_area_px2") val meanAreaPx2: Float,
    @SerializedName("mean_perimeter_px") val meanPerimeterPx: Float,
    @SerializedName("mean_eccentricity") val meanEccentricity: Float,
    @SerializedName("mean_circularity") val meanCircularity: Float,
    val interpretation: String
)

data class GlandEvidenceData(
    @SerializedName("total_count") val totalCount: Int,
    @SerializedName("mean_area_pixels") val meanAreaPixels: Float,
    @SerializedName("mean_aspect_ratio") val meanAspectRatio: Float,
    @SerializedName("mean_circularity") val meanCircularity: Float,
    val interpretation: String
)

data class ReferenceComparisonData(
    val label: String,
    @SerializedName("top_category") val topCategory: String,
    @SerializedName("top_similarity_percent") val topSimilarityPercent: Float,
    @SerializedName("top_reference_id") val topReferenceId: String,
    val insight: String,
    val comparisons: List<Map<String, Any>>
)

data class RegionDetailItem(
    @SerializedName("region_id") val regionId: String,
    val index: Int,
    val x: Int,
    val y: Int,
    val width: Int,
    val height: Int,
    val prediction: String,
    val confidence: Float,
    @SerializedName("tumor_probability") val tumorProbability: Float,
    @SerializedName("uncertainty_score") val uncertaintyScore: Float,
    @SerializedName("uncertainty_level") val uncertaintyLevel: String,
    @SerializedName("priority_score") val priorityScore: Float,
    @SerializedName("priority_level") val priorityLevel: String,
    @SerializedName("nuclei_count") val nucleiCount: Int,
    @SerializedName("glands_count") val glandsCount: Int,
    val rationale: String
)

data class NextRegionResponse(
    @SerializedName("case_id") val caseId: String,
    @SerializedName("has_next") val hasNext: Boolean,
    @SerializedName("next_region") val nextRegion: RegionDetailItem?,
    @SerializedName("remaining_unreviewed_count") val remainingUnreviewedCount: Int
)

data class MedGemmaExplanation(
    val summary: String,
    @SerializedName("visual_evidence") val visualEvidence: List<String>,
    @SerializedName("nuclear_evidence") val nuclearEvidence: List<String>,
    @SerializedName("gland_evidence") val glandEvidence: List<String>,
    @SerializedName("prediction_evidence") val predictionEvidence: List<String>,
    @SerializedName("uncertainty_explanation") val uncertaintyExplanation: String,
    @SerializedName("model_agreement") val modelAgreement: String,
    @SerializedName("reference_evidence") val referenceEvidence: List<String>,
    val limitations: List<String>,
    @SerializedName("review_recommendation") val reviewRecommendation: String
)

data class CopilotQuestionRequest(
    @SerializedName("case_id") val caseId: String,
    val question: String,
    @SerializedName("region_id") val regionId: String? = null
)

data class CopilotAnswerResponse(
    @SerializedName("case_id") val caseId: String,
    val question: String,
    @SerializedName("selected_region_id") val selectedRegionId: String?,
    val answer: String,
    val model: String,
    val validated: Boolean,
    @SerializedName("validation_errors") val validationErrors: List<String>
)

data class ReviewRequest(
    val action: String, // "MARK_REVIEWED", "FLAG_REGION", "ADD_NOTE"
    val notes: String? = null,
    @SerializedName("pathologist_id") val pathologistId: String? = null
)

data class FeedbackRequest(
    val feedback: String, // "CORRECT", "INCORRECT", "UNCERTAIN", "REVIEW_REQUIRED"
    val notes: String? = null,
    @SerializedName("pathologist_id") val pathologistId: String? = null
)
```

---

## 🌐 Retrofit 2 API Service Interface

```kotlin
package com.colonpath.ai.data.remote

import com.colonpath.ai.data.model.*
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.*

interface ColonPathApiService {

    @GET("health")
    suspend fun checkHealth(): Response<HealthResponse>

    @Multipart
    @POST("analyze")
    suspend fun analyzeImage(
        @Part image: MultipartBody.Part,
        @Part("case_id") caseId: RequestBody? = null
    ): Response<CaseResultResponse>

    @GET("cases")
    suspend fun listCases(
        @Query("limit") limit: Int = 50
    ): Response<List<CaseSummaryItem>>

    @GET("cases/{case_id}/result")
    suspend fun getCaseResult(
        @Path("case_id") caseId: String
    ): Response<CaseResultResponse>

    @GET("cases/{case_id}/evidence")
    suspend fun getCaseEvidence(
        @Path("case_id") caseId: String
    ): Response<Map<String, Any>>

    @GET("cases/{case_id}/report")
    suspend fun getCaseReport(
        @Path("case_id") caseId: String
    ): Response<Map<String, Any>>

    @GET("cases/{case_id}/regions")
    suspend fun getRegions(
        @Path("case_id") caseId: String
    ): Response<List<RegionDetailItem>>

    @GET("cases/{case_id}/regions/next")
    suspend fun getNextRegion(
        @Path("case_id") caseId: String,
        @Query("current_region_id") currentRegionId: String? = null
    ): Response<NextRegionResponse>

    @Streaming
    @GET("cases/{case_id}/visualization/{vis_type}")
    suspend fun getVisualization(
        @Path("case_id") caseId: String,
        @Path("vis_type") visType: String
    ): Response<ResponseBody>

    @POST("copilot/ask")
    suspend fun askCopilot(
        @Body request: CopilotQuestionRequest
    ): Response<CopilotAnswerResponse>

    @POST("cases/{case_id}/review")
    suspend fun submitReview(
        @Path("case_id") caseId: String,
        @Body request: ReviewRequest
    ): Response<Map<String, Any>>

    @POST("cases/{case_id}/feedback")
    suspend fun submitFeedback(
        @Path("case_id") caseId: String,
        @Body request: FeedbackRequest
    ): Response<Map<String, Any>>
}
```
