# COLONPATH-AI — Android Developer Integration Guide

Welcome to the **COLONPATH-AI** Android Client Integration Guide! This document gives you everything you need to connect your Android app (Jetpack Compose or XML Views) to the backend.

---

## ⚡ 5-Minute Quick Start

### 1. Network Configuration
Add the following to your `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<!-- Required if testing over local HTTP without SSL certificate -->
<application
    android:usesCleartextTraffic="true"
    ... >
```

### 2. Retrofit Client Setup
```kotlin
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object NetworkModule {
    // Use 10.0.2.2 for Android Studio Emulator, or machine IP for physical phone
    private const val BASE_URL = "http://10.0.2.2:8080/"

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .addInterceptor(HttpLoggingInterceptor().apply {
            level = HttpLoggingInterceptor.Level.BODY
        })
        .build()

    val apiService: ColonPathApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ColonPathApiService::class.java)
    }
}
```

---

## 🖼️ Rendering Authentic 7-Layer Visual Overlays in Coil

Use `AsyncImage` in Jetpack Compose to stream visual overlays directly from `/cases/{case_id}/visualization/{type}`:

```kotlin
@Composable
fun HistopathologyLayerViewer(caseId: String, selectedLayer: String) {
    // selectedLayer can be: "original", "glands", "nuclei", "regions", "uncertainty", "top_regions", "pseudo_3d"
    val imageUrl = "http://10.0.2.2:8080/cases/$caseId/visualization/$selectedLayer"

    AsyncImage(
        model = ImageRequest.Builder(LocalContext.current)
            .data(imageUrl)
            .crossfade(true)
            .build(),
        contentDescription = "Histopathology Layer: $selectedLayer",
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f),
        contentScale = ContentScale.Fit
    )
}
```

---

## 📤 Uploading an H&E Image for Analysis

```kotlin
suspend fun uploadBiopsy(imageFile: File, caseId: String?): Result<CaseResultResponse> {
    return try {
        val requestFile = imageFile.asRequestBody("image/png".toMediaTypeOrNull())
        val body = MultipartBody.Part.createFormData("image", imageFile.name, requestFile)
        val caseIdPart = caseId?.toRequestBody("text/plain".toMediaTypeOrNull())

        val response = NetworkModule.apiService.analyzeImage(body, caseIdPart)
        if (response.isSuccessful && response.body() != null) {
            Result.success(response.body()!!)
        } else {
            Result.failure(Exception("API Error: ${response.code()} ${response.message()}"))
        }
    } catch (e: Exception) {
        Result.failure(e)
    }
}
```

---

## 🤖 Asking the Pathologist Copilot

```kotlin
suspend fun askCopilot(caseId: String, question: String): String {
    val req = CopilotQuestionRequest(caseId = caseId, question = question)
    val res = NetworkModule.apiService.askCopilot(req)
    return if (res.isSuccessful) {
        res.body()?.answer ?: "No answer received"
    } else {
        "Error contacting copilot: ${res.code()}"
    }
}
```

---

## 🛡️ Medical AI UI Safety Rules for Android

1. **Never Display "Confirmed Cancer" for Model Output:**  
   Always display: `"AI-Predicted Tissue Class: [CLASS]"`.
2. **Never Treat High Priority as Definite Malignancy:**  
   Display: `"AI-Prioritized Region (Priority: 0.85) — Pathologist Review Recommended"`.
3. **Always Display Uncertainty / OOD Warnings:**  
   If `review_required == true` or `is_ood == true`, show a high-visibility amber/red banner:  
   *`"High model uncertainty detected. Autonomous prediction abstained. Mandatory pathologist review required."`*
