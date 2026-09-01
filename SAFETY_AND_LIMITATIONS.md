# SAFETY_AND_LIMITATIONS.md — Clinical Safety Constraints & Limitations

**Date:** September 1, 2026  
**Auditor:** Senior Medical AI & Digital Pathology Safety Committee  

---

## 1. Non-Diagnostic Decision Support Notice

1. **Supportive Evidence vs. Standalone Diagnosis:**  
   COLONPATH-AI is an evidence-grounded computational decision support workstation. **It does NOT produce a standalone or autonomous clinical diagnosis.** All predictions, morphometric parameters, and AI-generated text are designed for review and sign-off by a licensed, board-certified pathologist.
2. **Label Semantics:**  
   The model label `TUM` corresponds to microscopic **colorectal adenocarcinoma epithelium** in training datasets. It is reported as *"AI-Predicted Tissue Class"* and must never be converted into an unsupported definitive patient diagnosis without histological correlation of the muscularis mucosae and submucosal invasion depth.

---

## 2. Mandatory Pathology Safety Principles

* **No Single-Metric Malignancy Diagnosis:**  
  * Nuclear enlargement alone $\neq$ malignancy (occurs in reactive atypia).
  * Hyperchromasia alone $\neq$ malignancy (influenced by section thickness and hematoxylin staining time).
  * Gland crowding alone $\neq$ adenocarcinoma (can occur in tangential / oblique knife cuts).
  * High mitotic count alone $\neq$ malignancy (must be contextualized to crypt proliferative zones).
* **Automated Abstention Protocols:**  
  * If optical blur variance is $< 30.0$, the system flags `BLUR_DETECTED` and prompts fine-focus adjustment.
  * If predictive entropy is high ($> 1.50$), the system outputs `REVIEW_REQUIRED`.
  * If free energy $E(\mathbf{x}; T) > \tau_{\text{OOD}}$, the system flags `OOD_DETECTED` (foreign artifact / surgical ink / air bubble) and refuses automated classification.
