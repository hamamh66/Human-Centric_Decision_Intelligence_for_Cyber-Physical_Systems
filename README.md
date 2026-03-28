# Human-Centric Decision Intelligence for Cyber-Physical Systems

## Short Description
A reproducible, explainable, and calibrated decision intelligence framework for trust-aware monitoring and risk prioritization in cyber-physical systems.

---

## Overview
This repository provides an end-to-end implementation of a human-centric decision intelligence framework designed for cyber-physical systems (CPS). The framework moves beyond traditional detection-centric approaches by integrating predictive modeling, calibration, explainability, and structured decision support into a unified pipeline.

The objective is to transform raw predictive outputs into interpretable and actionable decision categories that support human operators in monitoring, prioritization, and intervention.

---

## Key Features
- Trust-aware feature engineering
- Calibrated predictive modeling
- Human-centric decision layer (NORMAL → CRITICAL)
- Explainable AI analytics (feature importance, score distributions, etc.)
- Reproducible pipeline (Colab-ready)

---

## Methodology
The framework consists of the following components:

1. Data acquisition (KDDCup99 benchmark)
2. Binary problem formulation (normal vs attack)
3. Feature engineering (trust, latency, anomaly, integrity)
4. Predictive modeling (regularized logistic regression)
5. Score calibration (shrinkage + noise)
6. Decision mapping (multi-level risk categories)
7. Explainability and visualization

---

## Decision Categories
- NORMAL
- MONITOR
- HIGH_PRIORITY_ALERT
- CRITICAL_INTERVENTION

---

## Repository Structure
```
├── notebooks/
├── data/
├── figures/
├── tables/
├── src/
└── README.md
```

---

## Reproducibility
This repository is designed for full reproducibility:
- Deterministic pipeline
- Standard datasets
- Colab-compatible implementation
- Exported figures and tables

---

## Citation
If you use this work, please cite:

Human-Centric Decision Intelligence for Cyber-Physical Systems: A Calibrated and Explainable Framework for Trust-Aware Monitoring and Risk Prioritization

---

## License
This project is intended for academic and research purposes.
