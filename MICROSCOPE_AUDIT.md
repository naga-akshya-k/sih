# MICROSCOPE_AUDIT.md — Microscope Optical & Physical Scale Audit

**Date of Audit:** September 1, 2026  
**Auditor:** Senior Digital Pathology & Optical Engineering Team  

---

## 1. Microscope Optical Configuration

| Optical Component | Specification | Clinical Relevance |
| :--- | :--- | :--- |
| **Microscope Type** | Standard Laboratory Optical Light Microscope (Upright) | Standard hospital pathology bench equipment. |
| **Illumination** | Kohler Illumination / LED Substage Condenser | Uniform background illumination ($[40, 220]$ intensity range). |
| **Objective Lenses** | $4\times$ (Scanning), $10\times$ (Low Power), $20\times$ (Intermediate), $40\times$ (High Power Dry) | Tissue architecture evaluated at $10\times/20\times$; nuclear pleomorphism at $40\times$. |
| **Ocular Tube Diameter**| $23.2\text{ mm}$ (Standard DIN) / $30.0\text{ mm}$ (Trinocular) | Mounts USB eyepiece camera directly into drawtube. |

---

## 2. Magnification, Resolution & Physical Scale Policy

### ⚠️ Physical Scale Calibration Notice:
Histopathological tiles from NCT-CRC-HE-100K and CRC-VAL-HE-7K are digitized at **$0.5\text{ µm/pixel}$ ($20\times$ magnification)**.

**Strict Scientific Policy:**
* When an optical calibration micrometer has not been calibrated on a specific microscope:
  * The system reports measurements in **Image-Space Pixels (`px` / `px²`)**.
  * The system displays: `"PHYSICAL SCALE NOT CALIBRATED"`.
* When a $10\text{ µm}$ stage micrometer calibration profile is supplied:
  * Measurements are converted to **Micrometers (`µm` / `µm²`)**.
