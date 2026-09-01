# Medical Safety, Limitations, and Pathology Interpretation Guidelines

This document outlines the clinical reasoning chain, terminology constraints, non-independent feature rules, and safety limitations governing the COLONPATH-AI decision-support system, derived from the official Colorectal Histopathology Feature Interpretation Reference.

---

## 1. Mandatory Reasoning Chain

Every downstream AI explanation, report, and Copilot answer must strictly follow the 3-step reasoning chain:

$$\text{Quantitative / Morphometric Feature} \longrightarrow \text{Biological Meaning (Tissue Biology)} \longrightarrow \text{Qualified Interpretation (Bounded Statement)}$$

> [!CAUTION]
> **No Direct Diagnostic Leaps:** The system must NEVER skip from a raw numeric metric directly to a diagnostic claim.

---

## 2. Controlled Pathology Vocabulary (Preferred Usage)

| Avoid / Imprecise Term | Preferred Pathology Term |
| :--- | :--- |
| "Cancer cells look weird" | **Nuclear pleomorphism / nuclear atypia** |
| "Glands are messed up" | **Architectural distortion / glandular disorganization** |
| "Dark nuclei" | **Hyperchromasia** |
| "Cells stacked up" | **Nuclear stratification / pseudostratification** |
| "Tumor has spread through the wall" | **Invasion through the muscularis mucosae into submucosa** |
| "Precancerous growth" | **Adenoma / dysplasia** (specify low-grade or high-grade) |
| "Cancer" (used loosely for adenoma) | Reserve **"carcinoma / adenocarcinoma"** strictly for invasive disease |
| "Dead tissue in gland" | **Luminal (dirty) necrosis** |
| "Messy cell borders" | **Loss of nuclear/cellular polarity** |
| "Very active dividing cells" | **Increased mitotic activity / atypical mitoses** |

---

## 3. Features That Must NOT Be Interpreted Independently

1. **Nuclear enlargement alone:** Can occur in reactive/inflammatory atypia, regeneration, or artifact; not specific to malignancy.
2. **Hyperchromasia alone:** Heavily affected by staining variation, section thickness, and fixation.
3. **Mitotic count alone:** Must be contextualized to field area, crypt zone location (basal vs. luminal surface), and section orientation.
4. **Gland crowding/density alone:** Can result from tangential sectioning, sampling angle, or normal crypt branching at mucosal flexures.
5. **N:C ratio alone:** Varies with plane of section and cell type; must be evaluated alongside chromatin texture and architecture.
6. **Single computational score:** No individual computational score meets diagnostic thresholds.

---

## 4. Fundamental Clinical & Pathological Limitations

- **Decision Support Only:** This system is a research prototype for decision support and does NOT replace the gestalt judgment of a qualified pathologist.
- **Sectioning Artifacts:** Tangential or oblique sectioning can mimic crowding, stratification, or pseudo-invasion.
- **Sampling Limitations:** A biopsy patch represents only a partial field; lesion heterogeneity may not be captured in a single tile.
- **Spectrum vs. Discrete Cut-Points:** The adenoma-to-carcinoma sequence is a continuous spectrum; computational numbers shift gradually without universal hard cut-points.
- **Immunohistochemistry:** Definitive subclassification (MSI/MMR status, KRAS/BRAF mutations) requires molecular testing not capturable by H&E morphology alone.
