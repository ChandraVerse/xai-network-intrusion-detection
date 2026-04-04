# Research Paper

**Title:** *Building an Explainable AI-Based Network Intrusion Detection System Using Machine Learning and SHAP*

**Authors:** Chandra Sekhar Chakraborty  
**Affiliation:** Maulana Abul Kalam Azad University of Technology (MAKAUT), West Bengal, India  
**Status:** Draft in progress — expected completion April 2026  
**Target Venue:** IEEE ICMLA 2026 / Arxiv preprint  

---

## Abstract (Draft)

This paper presents an explainable artificial intelligence (XAI) approach to network intrusion detection using the CICIDS-2017 benchmark dataset. Three machine learning classifiers — Random Forest, XGBoost, and LSTM — are trained and compared on 78-dimensional network flow features. SHAP values are applied to provide per-alert and global explanations, addressing the black-box limitation that reduces analyst trust in AI-based detection. The proposed system achieves a macro F1-score of 0.997 with a false positive rate below 0.3%, outperforming several baseline approaches from recent literature. A production-grade Streamlit dashboard surfaces SHAP waterfall explanations per alert, bridging the gap between ML accuracy and SOC analyst transparency.

---

## Paper Structure (IEEE Format)

1. **Introduction** — Motivation, problem statement, contributions
2. **Related Work** — Comparison with prior NIDS and XAI-NIDS approaches
3. **Methodology** — Dataset, preprocessing, SMOTE, model architectures
4. **XAI Integration** — SHAP TreeExplainer, DeepExplainer, waterfall interpretation
5. **Experimental Results** — Accuracy, macro F1, FAR, latency across three models
6. **Deployment** — Streamlit dashboard architecture, Docker containerisation
7. **Conclusion & Future Work** — Limitations, extensions (live PCAP, MITRE mapping)
8. **References**

---

## File

The final paper PDF will be placed in this directory as `xai_ids_paper.pdf` upon completion.

> **Note:** This file is a placeholder. The full IEEE-format paper is in active writing.
