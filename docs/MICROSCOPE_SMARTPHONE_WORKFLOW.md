# Microscope-to-Smartphone Live Clinical Workflow

This document specifies the exact real-time clinical workflow for the Android developer when connecting a smartphone camera or digital microscope adapter to the **COLONPATH-AI** intelligence platform.

---

## 🔬 Hardware & Network Architecture

```
┌─────────────────────────┐
│ Optical Microscope      │
│ (4x / 10x / 20x / 40x)  │
└───────────┬─────────────┘
            │ Optical Eyepiece / C-Mount
            ▼
┌─────────────────────────┐
│ Smartphone Camera       │
│ (Android App Captures   │
│  Live H&E Slide View)   │
└───────────┬─────────────┘
            │ Wi-Fi / Local Network HTTP Request
            │ POST /analyze (multipart/form-data)
            ▼
┌────────────────────────────────────────────────────────┐
│ COLONPATH-AI Workstation / Server (GPU Acceleration)   │
│ ├── Image Quality Gate (Blur / Brightness / Contrast)  │
│ ├── Digepath ViT-L/16 Foundation Model (1024-d)        │
│ ├── U-Net Gland Segmentation (best_model.pth)          │
│ ├── HoVer-Net Nuclear Segmentation (209 MB Checkpoint) │
│ ├── Multimodal Late-Fusion (best_classifier.pth)       │
│ ├── Platt Temperature Calibration (T=1.25)             │
│ ├── Qdrant Multimodal Vector RAG (Reference Cohorts)   │
│ ├── MedGemma 1.5 4B IT Evidence Explainer              │
│ └── 7 Authentic Dynamic Visual Overlays Generator      │
└───────────┬────────────────────────────────────────────┘
            │ Instant Response (0.05s - 1.2s)
            ▼
┌────────────────────────────────────────────────────────┐
│ Android App Display (At the Microscope Bench)          │
│ 1. AI-Predicted Tissue Class + Calibrated Confidence   │
│ 2. Interactive 7-Layer Overlay Switcher (Coil/Glide)   │
│ 3. Spatial Prioritized Regions Grid (R_01 - R_04)      │
│ 4. MedGemma Clinical Explanation Report                │
│ 5. Interactive Pathologist Copilot Q&A (/copilot/ask)  │
│ 6. Pathologist Review Sign-Off (MARK_REVIEWED)         │
└────────────────────────────────────────────────────────┘
```

---

## 📱 Step-by-Step Android Implementation

### Step 1: Capturing the Microscope Frame
In Android, use `CameraX` to capture a high-resolution still frame of the biopsy tissue through the microscope eyepiece:

```kotlin
val imageCapture = ImageCapture.Builder()
    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
    .build()

// Save to temporary file
val photoFile = File(context.cacheDir, "microscope_capture_${System.currentTimeMillis()}.png")
val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

imageCapture.takePicture(outputOptions, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
        // Send photoFile to backend for dynamic multimodal analysis
        analyzeMicroscopeSample(photoFile)
    }
    override fun onError(exc: ImageCaptureException) {
        // Handle capture error
    }
})
```

---

### Step 2: Uploading for Dynamic Multimodal Inference
Call `POST /analyze` with the captured microscope image. The backend runs all dynamic weights on GPU and returns the full diagnosis in ~1 second:

```kotlin
suspend fun analyzeMicroscopeSample(photoFile: File) {
    val requestFile = photoFile.asRequestBody("image/png".toMediaTypeOrNull())
    val body = MultipartBody.Part.createFormData("image", photoFile.name, requestFile)
    
    val response = NetworkModule.apiService.analyzeImage(body)
    if (response.isSuccessful && response.body() != null) {
        val result = response.body()!!
        
        // 1. Display Tissue Class & Confidence
        displayDiagnosis(
            predictedClass = result.prediction.predictedClass, // e.g. "LYM", "TUM", "NORM"
            confidence = result.prediction.calibratedConfidence, // e.g. 98.5%
            tumorProb = result.prediction.tumorProbability // e.g. 1.2%
        )
        
        // 2. Check Quality & Uncertainty Warnings
        if (result.uncertainty.reviewRequired || result.uncertainty.isOod) {
            showUncertaintyWarning(result.uncertainty.message)
        }
        
        // 3. Load 7 Visual Layers for this specific sample
        loadVisualLayers(result.caseId)
        
        // 4. Render MedGemma Clinical Narrative
        displayReport(result.explanation?.summary)
    }
}
```

---

### Step 3: Displaying the 7 Dynamic Overlays in the App
The Android app lets the pathologist toggle between visual layers with one tap, streaming the authentic PNGs generated specifically for that microscope frame:

```kotlin
@Composable
fun MicroscopeOverlayViewer(caseId: String) {
    var selectedLayer by remember { mutableStateOf("original") }
    
    Column {
        // 1. Layer Tabs
        ScrollableTabRow(selectedTabIndex = getIndex(selectedLayer)) {
            Tab(selected = selectedLayer == "original", onClick = { selectedLayer = "original" }, text = { Text("1. Original") })
            Tab(selected = selectedLayer == "glands", onClick = { selectedLayer = "glands" }, text = { Text("2. Glands (U-Net)") })
            Tab(selected = selectedLayer == "nuclei", onClick = { selectedLayer = "nuclei" }, text = { Text("3. Nuclei (HoVer-Net)") })
            Tab(selected = selectedLayer == "regions", onClick = { selectedLayer = "regions" }, text = { Text("4. AI Regions") })
            Tab(selected = selectedLayer == "uncertainty", onClick = { selectedLayer = "uncertainty" }, text = { Text("5. Heatmap") })
            Tab(selected = selectedLayer == "top_regions", onClick = { selectedLayer = "top_regions" }, text = { Text("6. Top Crops") })
            Tab(selected = selectedLayer == "pseudo_3d", onClick = { selectedLayer = "pseudo_3d" }, text = { Text("7. 3D View") })
        }
        
        // 2. Live Layer Image
        val layerUrl = "http://10.0.2.2:8080/cases/$caseId/visualization/$selectedLayer"
        AsyncImage(
            model = ImageRequest.Builder(LocalContext.current)
                .data(layerUrl)
                .crossfade(true)
                .build(),
            contentDescription = selectedLayer,
            modifier = Modifier.fillMaxWidth().aspectRatio(1f)
        )
    }
}
```

---

### Step 4: Pathologist Copilot at the Microscope Bench
The pathologist can speak or type questions into the app about the sample on the glass slide:

```kotlin
// Pathologist taps suggestion chip: "What nuclear abnormalities were detected?"
val copilotAnswer = apiService.askCopilot(
    CopilotQuestionRequest(
        caseId = currentCaseId,
        question = "What nuclear abnormalities were detected?"
    )
)

// App displays verified response:
// "Detected 117 total nuclei with mean area 138.5 px², circularity 0.69. 
//  Phenotype distribution: 3 Epithelial, 110 Spindle-shaped..."
```

---

## 🎯 Clinical Safety Rules for the Microscope UI
1. **Labeling:** Always show `"AI-Predicted Tissue Class: [CLASS]"` — never state `"Confirmed Cancer"`.
2. **Quality Warning:** If the slide is out of focus (Laplacian blur < 30), show:  
   *`"Blurry capture detected. Please adjust microscope fine-focus knob and re-capture."`*
3. **Decision Support:** Provide the *"Mark Reviewed"* button so the pathologist's final sign-off is permanently recorded in the audit trail.
